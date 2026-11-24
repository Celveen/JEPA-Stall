"""Validate all recorded pushes and produce paired continuous videos and a GitHub index."""
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent
CASES = [('rope_ablation', '原组一 · 同节奏 ECP 消融', 'Frozen DINO-WM 3/3', 'Ours 3/3 + ECP'),
         ('rope_methods', '原组二 · 完整方法对比', 'Frozen DINO-WM 5/5', 'Ours 3/3 + ECP'),
         ('rope_additional', '新增组 · 同节奏 ECP 对比', 'Frozen DINO-WM 3/3', 'Ours 3/3 + ECP')]


def duration(path):
    return float(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','default=nw=1:nk=1',str(path)],text=True))


def paired(base, ours, output, left, right):
    length = max(duration(base), duration(ours))
    graph = (f"[0:v]scale=448:448:flags=neighbor,pad=448:490:0:42:white,drawtext=text='{left}':x=10:y=12:fontsize=18:fontcolor=black,tpad=stop_mode=clone:stop_duration={length}[a];"
             f"[1:v]scale=448:448:flags=neighbor,pad=448:490:0:42:white,drawtext=text='{right}':x=10:y=12:fontsize=18:fontcolor=black,tpad=stop_mode=clone:stop_duration={length}[b];"
             "[2:v]scale=224:224,pad=224:490:0:130:white,drawtext=text='Goal':x=80:y=100:fontsize=20:fontcolor=black[g];[a][b][g]hstack=inputs=3[v]")
    subprocess.run(['ffmpeg','-v','error','-y','-filter_complex_threads','1','-i',str(base),'-i',str(ours),
                    '-loop','1','-framerate','30','-i',str(base.parent/'goal.png'),'-filter_complex',graph,'-map','[v]','-t',str(length),
                    '-c:v','libx264','-threads','2','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(output)],check=True)


def main():
    summaries=[]
    lines=['# Rope 每次推动的连续真实仿真录像', '',
           '三组配对素材覆盖原来的两组和新增的一组。每个方案包含全部已执行推动的独立 MP4、整段录像和逐推 PNG 帧包。', '',
           '画面由原 PyFleX 仿真器执行保存动作重新录制，覆盖下压、接触推动、抬起及稳定过程。每4个物理步采样一次，仿真步长为1/60秒，MP4以30fps播放，对应约2倍物理速度。PNG帧的准确物理步编号和校验值在同名 JSON 中。', '',
           '原评测在每个规划块开始恢复记录的粒子位置，本次重放保留该流程。每一推内部连续执行，规划块之间存在原流程的状态恢复与机械臂重置。各方案在原记录结束步停止。', '',
           '并排视频按播放时间同步，较短一侧保持终帧。逐推对比只覆盖双方均执行的推动编号，所有剩余推动保存在各自目录。', '']
    total=0
    for key,title,left,right in CASES:
        folder=ROOT/key
        data=[json.loads((folder/arm/'complete.json').read_text()) for arm in ['baseline','ours']]
        source_arrays=[np.load(folder/arm/'source_execution.npz') for arm in ['baseline','ours']]
        np.testing.assert_array_equal(source_arrays[0]['states'][0], source_arrays[1]['states'][0])
        np.testing.assert_array_equal(source_arrays[0]['goal'], source_arrays[1]['goal'])
        receipts=[json.loads((folder/arm/'source_manifest.json').read_text()) for arm in ['baseline','ours']]
        for name in ['checkpoint_sha256', 'encoder_sha256', 'action_mean', 'action_std']:
            assert receipts[0][name] == receipts[1][name], (key,name)
        for arm,record in zip(['baseline','ours'],data):
            for push in record['pushes']:
                n=push['push'];prefix=folder/arm/f'push_{n:02d}'
                metadata=json.loads(prefix.with_suffix('.json').read_text())
                with tarfile.open(str(prefix)+'_frames.tar.gz') as tar:
                    assert len(tar.getmembers())==push['frame_count']
                    for frame in metadata['frames']:
                        content=tar.extractfile(f"frame_{frame['frame']:05d}.png").read()
                        assert hashlib.sha256(content).hexdigest()==frame['sha256']
                steps=[f['physics_step'] for f in metadata['frames']]
                assert steps[0]==0 and steps[-1]==push['physics_steps']
                assert all(0<b-a<=4 for a,b in zip(steps,steps[1:]))
                count=int(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries',
                                                   'stream=nb_frames','-of','default=nw=1:nk=1',str(prefix.with_suffix('.mp4'))],text=True))
                assert count == len(steps)
                final=np.load(folder/arm/f'after_push_{n:02d}.npy')[:, :3]
                goal=source_arrays[0]['goal']
                cd=cKDTree(goal).query(final)[0].mean()+cKDTree(final).query(goal)[0].mean()
                assert abs(cd-push['replay_cd']) < 1e-9
                total+=len(steps)
        b,o=data
        assert o['final_replay_cd']<b['final_replay_cd']
        paired(folder/'baseline/full_execution.mp4',folder/'ours/full_execution.mp4',folder/'comparison.mp4',left,right)
        comparisons=folder/'per_push_comparison';comparisons.mkdir(exist_ok=True)
        for n in range(1,min(len(b['pushes']),len(o['pushes']))+1):
            paired(folder/f'baseline/push_{n:02d}.mp4',folder/f'ours/push_{n:02d}.mp4',comparisons/f'push_{n:02d}.mp4',f'{left} - Push {n}',f'{right} - Push {n}')
        subprocess.run(['ffmpeg','-v','error','-y','-i',str(folder/'comparison.mp4'),'-vf','fps=3,scale=672:-1:flags=lanczos','-loop','0',str(folder/'preview.gif')],check=True)
        improvement=100*(1-o['final_replay_cd']/b['final_replay_cd'])
        summary={'case':key,'baseline_pushes':len(b['pushes']),'ours_pushes':len(o['pushes']),
                 'baseline_replay_cd':b['final_replay_cd'],'ours_replay_cd':o['final_replay_cd'],'cd_reduction_percent':improvement,
                 'baseline_original_cd':b['original_final_cd'],'ours_original_cd':o['original_final_cd'],
                 'maximum_particle_coordinate_difference':max(p['max_particle_coordinate_difference_from_source'] for r in data for p in r['pushes'])}
        summaries.append(summary)
        lines += [f'## {title}', '', f"重放最终 CD 为 **{b['final_replay_cd']:.4f} → {o['final_replay_cd']:.4f}**，降低 **{improvement:.1f}%**。基线共 {len(b['pushes'])} 推，Ours 共 {len(o['pushes'])} 推。", '',
                  f'[并排完整录像]({key}/comparison.mp4) · [GIF 预览]({key}/preview.gif) · [逐推并排录像]({key}/per_push_comparison)', '',
                  f'[Frozen 全部推动与原始帧]({key}/baseline) · [Ours 全部推动与原始帧]({key}/ours)', '',
                  f'![{title}]({key}/preview.gif)', '']
    lines += ['## 方法和来源', '',
              '原组一使用 seed101 target15，双方3/3节奏，Ours原记录从第7推应用ECP。新增组使用 seed101 target24，双方3/3节奏，Ours原记录从第7推应用ECP。原组二使用 target13，Frozen使用5/5节奏，Ours使用3/3节奏，Ours该例没有触发ECP。', '',
              '每次推动提供 `push_XX.mp4`、`push_XX_frames.tar.gz` 和 `push_XX.json`。每个方案的 `complete.json` 保留重放CD、原始CD、逐推最大粒子坐标差异、原始记录路径和SHA-256。`source_execution.npz` 保留原动作、粒子状态与目标，`after_push_XX.npy` 保存本次重放的实际粒子状态。', '',
              'PNG来自直接调用PyFleX渲染接口，没有插值。录制采样不增加物理步，保持原环境的接触/非接触数据采样调用。重放指标按这次真实录制的粒子状态计算。', '',
              '新增案例从已有20目标、seed100/101的配对记录中筛选，选择Ours成功、实际应用ECP且CD改善明显的另一目标。该选择用于定性视频展示。', '',
              '`record.py` 负责连续录制，`package.py` 校验每张PNG哈希和采样完整性并输出视频。`validation.json` 汇总三组实际结果，`SHA256SUMS.json` 提供交付文件校验。', '']
    (ROOT/'README.md').write_text('\n'.join(lines))
    (ROOT/'validation.json').write_text(json.dumps({'continuous_frames':total,'cases':summaries,'png_sha256_verified':True,'all_pushes_captured_start_to_end':True},indent=2))
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.rglob('*') if p.is_file() and p.name not in ('SHA256SUMS.json','recording.lock')}
    (ROOT/'SHA256SUMS.json').write_text(json.dumps(hashes,indent=2))
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
