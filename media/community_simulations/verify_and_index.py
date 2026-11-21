"""Verify PNG sequences against recorded RGB, collect execution evidence, and write the index."""
import hashlib
import io
import json
from pathlib import Path
import tarfile
import pickle
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
INDICES = {'pushobj_ablation': 4, 'pushobj_methods': 9, 'pusht_ablation': 6,
           'pusht_methods': 6, 'rope_ablation': 0, 'rope_methods': 0,
           'maze_ablation': 17, 'maze_methods': 13}


def main():
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    verified = []
    rows = []
    cases = []
    raw_total = 0
    display_total = 0
    for key, sample in INDICES.items():
        folder = ROOT / key
        case = json.loads((folder / 'manifest.json').read_text())
        for frame in case['frames']:
            path = ROOT / frame['path']
            assert hashlib.sha256(path.read_bytes()).hexdigest() == frame['sha256']
        for index, arm in enumerate(case['arms']):
            arm_dir = folder / ('baseline' if index == 0 else 'ours')
            files = sorted(arm_dir.glob('frame_*.png'))
            assert len(files) == arm['last_step'] + 1
            assert [p.name for p in files] == [f'frame_{i:04d}.png' for i in range(len(files))]
            path, separator, member = arm['source'].partition('::')
            evidence = arm_dir / 'evidence'
            evidence.mkdir(exist_ok=True)
            if separator:
                with tarfile.open(REPO / path) as tar:
                    for filename in ['config.yaml', '.hydra/config.yaml', 'stag_meta.json', 'probe_meta.json',
                                     'stag_episodes.jsonl', 'escape_episodes.jsonl', 'per_iter_logs.json']:
                        name = member + '/' + filename
                        if name in tar.getnames():
                            (evidence / filename.replace('/', '_')).write_bytes(tar.extractfile(name).read())
                    z = np.load(io.BytesIO(tar.extractfile(member + '/final_eval_results.npz').read()))
                    np.savez_compressed(evidence / 'saved_planner_actions.npz', **{k: z[k][sample:sample+1] for k in z.files})
            else:
                z = np.load(REPO / path)
                original = z['obs_visual'][0] if key.startswith('rope') else z['rgb'][sample, :len(files)]
                assert len(original) == len(files)
                for file, frame in zip(files, original):
                    np.testing.assert_array_equal(np.asarray(Image.open(file)), frame)
                source_dir = (REPO / path).parent
                for name in ['config.yaml', 'run_manifest.json', 'complete.json', 'encoder_audit.json',
                             'ecp_records.json', 'stag_meta.json', 'stag_episodes.jsonl', 'lewm_ecp_episodes.jsonl']:
                    if (source_dir / name).exists():
                        (evidence / name).write_bytes((source_dir / name).read_bytes())
                payload = {k: z[k][sample:sample+1] for k in z.files if k not in ('rgb', 'obs_visual', 'goal_rgb', 'target_rgb')}
                np.savez_compressed(evidence / 'recorded_controls_and_states.npz', **payload)
                final_path = source_dir / 'final_eval_results.npz'
                if final_path.exists():
                    final = np.load(final_path)
                    np.savez_compressed(evidence / 'saved_planner_actions.npz', **{k: final[k][sample:sample+1] for k in final.files})
            raw_total += len(files)
            verified.append({'case': key, 'arm': index, 'frames': len(files), 'hashes_verified': True,
                             'pixel_exact_to_recording': not bool(separator), 'capture': arm['capture']})
        np.testing.assert_array_equal(np.asarray(Image.open(folder/'baseline/frame_0000.png')),
                                      np.asarray(Image.open(folder/'ours/frame_0000.png')))
        if key.startswith('rope'):
            target_path = (REPO / case['arms'][0]['source']).parent / 'target.pkl'
            target = pickle.load(target_path.open('rb'))
            goal = np.asarray(target['state_g']).reshape(-1, 4)[:, :3]
            for index, arm in enumerate(case['arms']):
                arm_dir = folder / ('baseline' if index == 0 else 'ours')
                final = np.load(arm_dir / 'execution.npz')['states'][-1, :, :3]
                cd = cKDTree(goal).query(final)[0].mean() + cKDTree(final).query(goal)[0].mean()
                assert abs(cd - float(arm['metrics']['final_cd'])) < 1e-6
                assert bool(cd < .3) == arm['success']
                arm['independently_verified_final_cd'] = float(cd)
        else:
            for index, arm in enumerate(case['arms']):
                arm_dir = folder / ('baseline' if index == 0 else 'ours')
                final = np.load(arm_dir / 'execution.npz')['states'][-1]
                saved = np.load(arm_dir / 'evidence/saved_planner_actions.npz')
                goal = saved['state_g'][0]
                if key.startswith('maze'):
                    success = np.linalg.norm(final[:2] - goal[:2]) < .5
                else:
                    angle_error = abs((final[4] - goal[4] + np.pi) % (2 * np.pi) - np.pi)
                    success = np.linalg.norm(final[:4] - goal[:4]) < 20 and angle_error < np.pi / 9
                assert bool(success) == arm['success'], (key, arm['name'])
        b, o = case['arms']
        metric = lambda a: ('CD ' + format(float(a['metrics']['final_cd']), '.4f')) if 'final_cd' in a['metrics'] else ('成功' if a['success'] else '预算内未完成')
        labels = (b['name'] + ' → ' + o['name']).replace('|', '·')
        rows.append(f"| [{key}]({key}/preview.gif) | {labels} | {b['last_step']} / {o['last_step']} | {metric(b)} → {metric(o)} |")
        display_total += len(list((folder / 'comparison_frames').glob('frame_*.png')))
        cases.append({k:v for k,v in case.items() if k != 'frames'})
    manifest['cases'] = cases
    for name in ['export_frames.py', 'verify_and_index.py']:
        manifest['sources'][str((ROOT / name).relative_to(REPO))] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    (ROOT / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    audit = {'case_count': len(cases), 'raw_png_count': raw_total, 'comparison_png_count': display_total,
             'checks': verified, 'all_paired_initial_rgb_equal': True,
             'all_terminal_outcomes_independently_verified': True}
    (ROOT / 'validation.json').write_text(json.dumps(audit, indent=2)+'\n')
    readme = '''# 社区视频真实仿真逐帧素材

PushObj、PushT、Rope 和 Maze 各两组，共八组。每个任务提供一组机制开关对比、一组完整方法对比，展示同起点和目标下的执行差异。案例按可见差异筛选，用于视频展示。

| 素材与动态预览 | 对比 | 基线 / Ours 结束步 | 结果 |
|---|---|---|---|
''' + '\n'.join(rows) + '''

## 剪辑入口

每组 `baseline/frame_0000.png` 与 `ours/frame_0000.png` 起按编号连续保存原始画面。PushObj、PushT、Maze 每帧对应一个物理控制步；Rope 每帧对应一次完整推动后的真实观测。画面保持仿真器 224×224 原分辨率。

`comparison_frames/frame_*.png` 是 1280×670 的逐步同步展示帧，带方法名称、当前步、独立目标图和实际状态轨迹。`preview.gif` 用于快速选片，`storyboard.jpg` 展示四个过程节点。两方案结束时间不同，展示帧会明确写出 terminal frame held。原始序列在各自真实结束步停止。

轨迹线连接逐控制观测的位置。PushObj 和 PushT 在画面中显示真实 pusher 路径；右侧另附完整路径。Maze 路径使用仿真世界坐标单独展示，Rope 路径是粒子质心的 x/z 轨迹。GIF 控制步按 150ms、推动步按 450ms 播放，剪辑时可自行调速。

## 八组实验各自说明什么

- **PushObj 消融**展示同一 Frozen 模型与规划配置下，ECP 如何让 Z 形物体任务走出停滞并达到目标。350 个配对任务中按最终物体距离改善筛选，chunk16 sample4，阈值 0.45。
- **PushObj 方法对比**展示 Frozen+ECP 与 AdaJEPA 在同一 small-T 目标上的执行差异，chunk24 sample9。
- **PushT 消融**展示 LeWM 干预开关对联合 pusher/object 目标完成的作用，red-goal target6。干预包含 latent memory、proprio 与使用物体位姿和几何的 contact objective。物体在基线末尾已经接近目标，主要差异来自 pusher 的最终对齐。
- **PushT 方法对比**在同一目标上比较 LeWM+干预与 augmented AdaJEPA。实际目标以单独的 Goal 图为准，仿真背景中固定的红色 T 是环境 anchor。
- **Rope 消融**使用 seed101 target15，双方均为 3/3 推动节奏，ECP 从第7次推动开始应用。关闭干预的末端 CD 为 1.7939，开启后为 0.1681。
- **Rope 方法对比**使用独立录制的 target13，比较原生 DINO-WM 5/5 与 Ours 3/3 完整规划流程，CD 从 0.7160 降至 0.0836。该组节奏不同，且 Ours 本次执行没有触发 ECP；ECP 的机制作用由前一组同节奏消融呈现。
- **Maze 消融**使用 medium seed100 target17，比较 AdaJEPA 与 AdaJEPA+gated ECP，干预后在第110控制步完成。
- **Maze 方法对比**使用同一任务集 target13，比较 Frozen MPC 与 AdaJEPA+gated ECP，Ours 在第130控制步完成。

PushObj 与 PushT 的判定同时检查 pusher/object 联合位置与物体角度，Rope 按 CD<0.3 停止，Maze 按位置距离<0.5 停止。各方法名称保留实际模型与规划配置。

## 真实执行与来源

PushT、Rope、Maze 原始 PNG 直接导出已保存的真实仿真 RGB，并逐像素核验。PushObj 以及 PushT 方法对比的 AdaJEPA 行用原始动作在原 Pymunk 仿真器重放，逐五控制步核对原实验日志端点，最大允许误差 0.00051。PushObj 的目标图由原渲染器在保存的目标位姿生成。

每组 `manifest.json` 记录步骤、来源、结果和每张原始 PNG 的 SHA-256。各方案的 `execution.npz` 保存逐步状态，`evidence/` 保存动作、配置与干预记录。顶层 `manifest.json` 汇总源文件校验值，`validation.json` 汇总帧数和原始像素核验。源实验配置中的绝对路径对应原执行机器。

导出使用 `export_frames.py`，核验与索引使用 `verify_and_index.py`。运行需要本仓库已有的记录文件和 `eval_outputs/rope_local/venv` 中的 NumPy、Pillow、PyTorch、Pymunk、Gym 等依赖。PushObj 原始动作与日志在 `reports/frozen_ecp_cost_20261117/raw_results.tar.gz`、`reports/publication_v14/archives/main_pushobj_*.tar.gz`，PushT AdaJEPA 原始记录在 `reports/publication_v14/archives/pusht_visual_complete.tar.gz`。

展示参考 [DINO-WM 项目页](https://dino-wm.github.io/) 的并排方法与独立目标图布局。
'''
    (ROOT / 'README.md').write_text(readme)
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
