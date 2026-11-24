"""Package two audited, fully visible same-LeWM color-shift rescues."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import numpy as np
from PIL import Image
from screen import ROOT, RUNS, OUT, stop, success


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def video(folder, filename):
    subprocess.run(['ffmpeg','-v','error','-y','-framerate','10','-i',str(folder/'frame_%04d.png'),
                    '-c:v','libx264','-threads','2','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(filename)],check=True)


def main():
    candidates=json.loads((OUT/'candidates.json').read_text())
    assert len(candidates)>=2
    # Rank position and angle improvements in units of their success thresholds.
    candidates.sort(key=lambda r:(r['native_joint_xy']-r['ours_joint_xy'])/20 +
                    (r['native_angle_deg']-r['ours_angle_deg'])/20,reverse=True)
    chosen=[candidates[0]]
    chosen.append(next((r for r in candidates[1:] if r['condition']!=chosen[0]['condition']),candidates[1]))
    spec=importlib.util.spec_from_file_location('export',ROOT/'media/community_simulations/export_frames.py')
    export=importlib.util.module_from_spec(spec);spec.loader.exec_module(export);export.OUT=OUT
    receipts=[]
    for number,row in enumerate(chosen,1):
        key=f"case_{number:02d}_{row['condition']}_target{row['target']:02d}"
        folder=OUT/key;condition=row['condition'];i=row['target']
        records=[dict(np.load(RUNS/f'{condition}_{arm}'/'recorded_execution.npz')) for arm in ['native','ours']]
        arms=[]
        for arm,label,data in zip(['native','ours'],['LeWM | intervention off','Ours | same LeWM + intervention'],records):
            n=stop(data,i)
            arms.append(dict(name=label,rgb=data['rgb'][i,:n+1],states=data['states'][i,:n+1],
                             success=bool(data['successes'][i]),source=str((RUNS/f'{condition}_{arm}'/'recorded_execution.npz').relative_to(ROOT)),
                             metrics={k:v for k,v in row.items() if k.startswith('native_' if arm=='native' else 'ours_')},
                             capture='new evaluation, recorded simulator RGB at every physical control'))
        notes='Same checkpoint and frozen model tensor hash, targets, seed, GD100 planning budget and color shift. Intervention includes latent memory and contact/proprio objectives using observed object geometry. Agent stays fully inside original viewport in both executions.'
        export.package(key,'PushT',condition.replace('_',' ')+' | Same backbone',arms,records[0]['goal_rgb'][i,0],notes,overlay=False)
        for arm,display,data in zip(['native','ours'],['baseline','ours'],records):
            directory=folder/display;n=stop(data,i);src=RUNS/f'{condition}_{arm}'
            video(directory,directory/'execution.mp4')
            np.savez_compressed(directory/'recorded_execution.npz',rgb=data['rgb'][i,:n+1],states=data['states'][i,:n+1],controls=data['controls'][i,:n],state_g=data['state_g'][i])
            evidence=directory/'evidence';evidence.mkdir(exist_ok=True)
            for name in ['config.yaml','complete.json','frozen_model_check.json','source_sha256.json','lewm_ecp_episodes.jsonl','lewm_contact_objective.jsonl','lewm_proprio_weight.jsonl']:
                if (src/name).exists():shutil.copyfile(src/name,evidence/name)
            for step in range(n+1):
                np.testing.assert_array_equal(np.asarray(Image.open(directory/f'frame_{step:04d}.png')),data['rgb'][i,step])
            assert success(data['states'][i,n],data['state_g'][i]) == (arm=='ours')
        video(folder/'comparison_frames',folder/'comparison.mp4')
        receipt=dict(row,key=key,notes=notes,source_recording_sha256={arm:sha(RUNS/f'{condition}_{arm}'/'recorded_execution.npz') for arm in ['native','ours']})
        receipts.append(receipt)
        (folder/'audit.json').write_text(json.dumps(receipt,indent=2)+'\n')
    evidence=OUT/'screening_evidence';evidence.mkdir(exist_ok=True)
    for src in RUNS.iterdir():
        if not (src/'complete.json').exists():continue
        dest=evidence/src.name;dest.mkdir(exist_ok=True)
        z=np.load(src/'recorded_execution.npz')
        np.savez_compressed(dest/'all_50_trajectories.npz',**{k:z[k] for k in z.files if k not in ('rgb','goal_rgb')})
        for name in ['complete.json','config.yaml','frozen_model_check.json','lewm_ecp_episodes.jsonl','lewm_ecp_replans.jsonl']:
            if (src/name).exists():shutil.copyfile(src/name,dest/name)
    lines=['# PushT 颜色变化 · 同 backbone 新增两组', '',
           '两组均比较同一份冻结 LeWM 模型的干预开关。Ours 完成目标，对比方法在150个控制步预算内未完成，双方 agent 全程完整留在原始画面内。', '',
           '每组保存新评测的全部真实控制步RGB、原始视频、同步并排MP4、GIF预览及目标图。视频10fps，原始224×224像素保持完整。', '']
    for row in receipts:
        key=row['key']
        lines += [f"## {row['condition']} · target {row['target']}", '',
                  f"Ours 在第 **{row['ours_controls']}** 控制步完成，基线执行150步后未完成。最终联合位置误差 **{row['native_joint_xy']:.2f} → {row['ours_joint_xy']:.2f} px**，角度误差 **{row['native_angle_deg']:.2f}° → {row['ours_angle_deg']:.2f}°**。", '',
                  f"双方agent中心到画面边缘的全程最小距离为 **{row['agent_edge_margin_px']:.2f} px**，agent半径15px。", '',
                  f'[并排录像]({key}/comparison.mp4) · [基线原始视频]({key}/baseline/execution.mp4) · [Ours原始视频]({key}/ours/execution.mp4) · [逐帧图片与核验]({key})', '',
                  f'![{key}]({key}/preview.gif)', '']
    lines += ['## 比较条件与验证', '',
              '使用同一 LeWM checkpoint，模型张量哈希完全一致，评测期间权重冻结。相同目标、初始状态、seed100、GD100、25个物理控制步规划范围、每次执行5步、150控制步总预算。干预开启前的动作已逐步核验一致。', '',
              'Ours沿用已存档的LeWM干预实现，包含latent stagnation memory及条件触发的proprio/contact objective，contact项使用观察到的物体位姿和T形几何。具体配置随每个方案附在evidence目录。', '',
              '成功标准为agent与object联合位置误差<20px且物体角度误差<20°。单独的Goal图是任务目标，场景中的固定彩色T形anchor属于环境背景。原始RGB从本次最终评估执行中保存，按各自首次成功步截断，比较视频会明确标注保持终帧。', '',
              '在三个颜色条件各50个配对目标中，筛选Ours成功、基线失败且agent中心全程位于[16,496]²的案例，按联合位置与角度误差改善量除以对应成功阈值后的和排序，并优先覆盖不同颜色条件。screening.csv保留全部筛选结果，screening_evidence保留全部数值轨迹和配置。这两组用于定性展示。', '',
              '`run.py`补跑完整配对评测，`screen.py`验证模型/目标配对、触发前动作、实际终态成功判定与agent可见范围，`package.py`逐像素核验PNG并导出视频。', '']
    (OUT/'README.md').write_text('\n'.join(lines))
    (OUT/'validation.json').write_text(json.dumps(dict(cases=receipts,same_frozen_model_verified=True,paired_initial_states_and_goals_verified=True,pretrigger_actions_verified=True,all_raw_png_pixels_verified=True,terminal_success_recomputed=True,agent_full_viewport_verified=True),indent=2)+'\n')
    hashes={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='SHA256SUMS.json' and '__pycache__' not in str(p)}
    (OUT/'SHA256SUMS.json').write_text(json.dumps(hashes,indent=2)+'\n')
    print(json.dumps(receipts,indent=2))


if __name__=='__main__':main()
