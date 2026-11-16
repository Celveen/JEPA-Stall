"""Export recorded simulator RGB frames, three-push timelines and paired figures.

Never replay physics or render world-model predictions. Figures use cached observations
from the evaluated runs. PNG conversion must preserve every stored RGB value exactly.
"""
import argparse
import csv
import hashlib
import json
import os
import pickle
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/jepa-stall-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ARMS = ('native5', 'fixed3', 'dynamic3', 'fixed3_nowarm')
LABELS = {'native5': 'DINO-WM · Native 5/5', 'fixed3': 'Ours · fixed S 3/3 · warmup 5',
          'dynamic3': 'Dynamic S 3/3 · warmup 5', 'fixed3_nowarm': 'Fixed S 3/3 · no extra warmup'}
COLORS = {'native5': '#496479', 'fixed3': '#14806a', 'dynamic3': '#7161a8', 'fixed3_nowarm': '#c97a26'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rgb(value):
    value = np.asarray(value)
    assert value.ndim == 3 and value.shape[-1] == 3
    assert np.isfinite(value).all() and value.min() >= 0 and value.max() <= 255
    result = value.astype(np.uint8)
    assert np.array_equal(value, result), 'RGB conversion would change stored pixel values'
    return result


def load(root, arm, i):
    p = root / arm / f'sample_{i:02d}'
    assert (p / 'complete.json').exists(), p
    with np.load(p / 'recorded_execution.npz') as cache:
        frames = np.stack([rgb(v) for v in cache['obs_visual'][0]])
        physical = cache['physical_actions'][0].copy()
        states = cache['states'][0].copy()
    with np.load(p / 'rope_metrics.npz') as metrics:
        cd = np.r_[metrics['initial_cd'][0], metrics['cd_by_replan'][0]]
        pushes = int(metrics['actual_pushes'][0])
        assert len(frames) == len(cd) == pushes + 1
        assert len(physical) == pushes and len(states) == pushes + 1
        np.testing.assert_array_equal(states[-1], metrics['final_states'][0])
        assert cd[-1] == metrics['final_cd'][0]
        assert pushes == 20 or cd[-1] < .3
        hits = np.flatnonzero(cd < .3)
        assert not len(hits) or hits[0] == pushes
    manifest = json.loads((p / 'run_manifest.json').read_text())
    with np.load(p / 'final_eval_results.npz') as final:
        actual = final['actions'][0] * np.asarray(manifest['action_std'], dtype=np.float32) + np.asarray(manifest['action_mean'], dtype=np.float32)
        np.testing.assert_array_equal(actual, physical)
    with (p / 'target.pkl').open('rb') as stream:
        target = pickle.load(stream)
    goal = rgb(target['obs_g']['visual'][0, 0])
    records = json.loads((p / 'ecp_records.json').read_text())
    triggers = 0 if arm == 'native5' else sum(r.get('severity_weight', 0) > 0 for r in records)
    return dict(path=p, frames=frames, goal=goal, cd=cd, pushes=pushes, physical=physical, triggers=triggers)


def save_figure(fig, path):
    fig.savefig(path.with_suffix('.png'), dpi=180, facecolor='white')
    fig.savefig(path.with_suffix('.pdf'), metadata={'CreationDate': None}, facecolor='white')
    plt.close(fig)


def show(ax, frame, subtitle='', color='#414b56'):
    ax.imshow(frame, interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(.7)
        spine.set_edgecolor(color)
    ax.set_xlabel(subtitle, fontsize=9, color=color, labelpad=5)


def timeline(data, i, out):
    arms = ('native5', 'fixed3')
    # Fixed schedule; a dedicated final column preserves stops between scheduled snapshots.
    slots = list(range(0, 21, 3)) + ['final', 'goal']
    fig, axes = plt.subplots(2, len(slots), figsize=(19, 5.7))
    fig.subplots_adjust(left=.025, right=.99, bottom=.12, top=.81, wspace=.09, hspace=.43)
    for row, arm in enumerate(arms):
        d = data[arm, i]
        for col, slot in enumerate(slots):
            ax = axes[row, col]
            if slot == 'goal':
                show(ax, d['goal'], 'Shared goal')
            else:
                step = d['pushes'] if slot == 'final' else slot
                if step > d['pushes']:
                    ax.axis('off')
                    ax.text(.5, .5, f'Stopped after\npush {d["pushes"]}\nCD < 0.3',
                            ha='center', va='center', transform=ax.transAxes, fontsize=10, color='#77828d')
                else:
                    label = f'CD {d["cd"][step]:.3f}'
                    if slot == 'final':
                        label = f'Push {step} · ' + label + ('\nSuccess' if d['cd'][-1] < .3 else '\nBudget exhausted')
                    show(ax, d['frames'][step], label, COLORS[arm])
            if row == 0:
                ax.set_title('Actual final' if slot == 'final' else 'Goal' if slot == 'goal'
                             else 'Start' if slot == 0 else f'Push {slot}', fontsize=11, pad=7)
        pos = axes[row, 0].get_position()
        fig.text(.025, pos.y1 + (.065 if row == 0 else .027), LABELS[arm] + (f' · S triggers: {d["triggers"]}' if arm != 'native5' else ''),
                 color=COLORS[arm], fontsize=12, weight='bold')
    fig.suptitle(f'Real Rope execution · target {i + 10} · independent visualization run', x=.025, ha='left', y=.98,
                 fontsize=17, weight='bold')
    fig.text(.025, .035, 'Recorded simulator RGB; every 3 pushes plus actual final. No replay, predicted frames, or invented post-success frames.\n'
             'Separate visualization execution, excluded from formal-20 statistics. Native 5/5 vs fixed S 3/3 changes both chunk size and S.', fontsize=10, color='#52606d')
    save_figure(fig, out / 'trajectory_every3')


def comparison(data, i, out):
    n, o = data['native5', i], data['fixed3', i]
    frames = [n['frames'][0], n['frames'][-1], o['frames'][-1], n['goal']]
    # One shared, axis-aligned crop across all four frames. Preserve original aspect ratio.
    masks = [(f[..., 0].astype(int) - f[..., 1].astype(int)) > 40 for f in frames]
    yy, xx = np.where(np.any(masks, axis=0))
    assert len(xx), 'Rope foreground not found'
    h, w = frames[0].shape[:2]
    side = min(max(int(max(xx.max()-xx.min(), yy.max()-yy.min())) + 32, 80), h, w)
    x0 = min(max(int((xx.min()+xx.max()-side)//2), 0), w-side)
    y0 = min(max(int((yy.min()+yy.max()-side)//2), 0), h-side)
    titles = ['Start', 'DINO-WM · Native 5/5', 'Ours · fixed S 3/3', 'Shared goal']
    captions = [f'Push 0 · CD {n["cd"][0]:.3f}', f'Push {n["pushes"]} · CD {n["cd"][-1]:.3f}',
                f'Push {o["pushes"]} · CD {o["cd"][-1]:.3f}', 'Target observation']
    fig, axes = plt.subplots(2, 4, figsize=(12, 7.9))
    fig.subplots_adjust(left=.045, right=.985, bottom=.12, top=.86, wspace=.08, hspace=.25)
    for col, frame in enumerate(frames):
        show(axes[0, col], frame, captions[col])
        axes[0, col].set_title(titles[col], fontsize=12, pad=8)
        show(axes[1, col], frame[y0:y0+side, x0:x0+side], 'Same crop · nearest-neighbor enlargement')
    fig.suptitle(f'Real simulator comparison · target {i + 10} · separate run', x=.045, ha='left', y=.965, fontsize=17, weight='bold')
    fig.text(.045, .92, f'Ours: fixed S, 3/3, warmup 5; S triggers in this run: {o["triggers"]}. CD uses simulator particle states.', fontsize=10)
    fig.text(.045, .035, 'Top: unaltered full frames. Bottom: the same crop in every frame; no goal overlays or geometry edits.\n'
             'Independent visualization run, excluded from formal-20 statistics; not an isolated estimate of the S effect.', fontsize=10, color='#52606d')
    save_figure(fig, out / 'final_comparison')
    return dict(x=x0, y=y0, width=side, height=side)


def main(root, output, formal):
    root, output = Path(root), Path(output)
    queue = json.loads((root/'queue.json').read_text())
    assert queue['status'] == 'complete'
    samples = queue['samples']
    data = {(arm, i): load(root, arm, i) for arm in ('native5', 'fixed3') for i in samples}
    output.mkdir(parents=True, exist_ok=True)
    sources, rows, selections, replicate_audit = [], [], [], []
    for (arm, i), d in data.items():
        np.testing.assert_array_equal(d['goal'], data['native5', i]['goal'])
        original = Path(formal) / arm / f'sample_{i:02d}'
        current = json.loads((d['path']/'complete.json').read_text())
        prior = json.loads((original/'complete.json').read_text())
        for key in ('checkpoint_sha256','encoder_sha256','implementation_sha256','full_target_sha256','action_mean','action_std'):
            assert current[key] == prior[key], key
        assert current['config']['planner'] == prior['config']['planner']
        with np.load(original/'final_eval_results.npz') as a, np.load(d['path']/'final_eval_results.npz') as b:
            for key in ('state_0', 'state_g'):
                np.testing.assert_array_equal(a[key], b[key])
        with np.load(original/'rope_metrics.npz') as m:
            replicate_audit.append(dict(arm=arm, sample=i, formal_final_cd=float(m['final_cd'][0]),
                visualization_final_cd=float(d['cd'][-1]), same_goal_model_planner_source_and_seeds=True))
        dest = output / 'raw_frames' / f'sample_{i:02d}' / arm
        dest.mkdir(parents=True, exist_ok=True)
        Image.fromarray(d['goal']).save(dest.parent / 'goal.png')
        files = []
        for step, frame in enumerate(d['frames']):
            path = dest / f'push_{step:02d}.png'
            Image.fromarray(frame).save(path)
            np.testing.assert_array_equal(np.asarray(Image.open(path)), frame)
            files.append(dict(push=step, cd=float(d['cd'][step]), path=str(path.relative_to(output)), sha256=digest(path)))
            rows.append(dict(arm=arm, sample=i, global_sample=i+10, push=step, cd=float(d['cd'][step]), actual_final=step==d['pushes']))
        sources.append(dict(arm=arm, sample=i, original_cache=str(d['path']/'recorded_execution.npz'),
                            cache_sha256=digest(d['path']/'recorded_execution.npz'),
                            metrics_sha256=digest(d['path']/'rope_metrics.npz'),
                            target_sha256=digest(d['path']/'target.pkl'),
                            goal_png_sha256=digest(dest.parent/'goal.png'), frames=files))
    for i in samples:
        out = output / f'sample_{i:02d}'
        out.mkdir(exist_ok=True)
        timeline(data, i, out)
        crop = comparison(data, i, out)
        n, o = data['native5', i], data['fixed3', i]
        selections.append(dict(sample=i, global_sample=i+10,
                               native_cd=float(n['cd'][-1]), ours_cd=float(o['cd'][-1]),
                               native_pushes=n['pushes'], ours_pushes=o['pushes'], ours_S_triggers=o['triggers'],
                               native_minus_ours_cd=float(n['cd'][-1]-o['cd'][-1]), crop=crop))
    with (output/'cd_by_push.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    manifest = dict(source='Recorded real simulator observations from separate visualization executions; no physics replay',
        excluded_from_formal20=True,
        rgb_conversion='Stored float32 RGB values in [0,255] verified exactly equal to exported uint8 pixels',
        ours='fixed3: 3/3, fixed strength, warmup5', cadence='push 0, every 3 pushes, actual final, goal',
        selection_rule=queue['selection'], selection=selections, sources=sources)
    (output/'formal_vs_visualization.json').write_text(json.dumps(replicate_audit, indent=2)+'\n')
    (output/'provenance.json').write_text(json.dumps(manifest, indent=2)+'\n')
    lines = ['# Rope 真实仿真截图', '',
        '**本目录是独立可视化复跑，不计入正式 20 样本结果。** 原正式运行仅 Native 将逐推画面存盘，S 组没有存盘；因此没有将复跑画面冒充原正式轨迹。', '',
        '画面直接取自本次实际执行时保存的仿真 RGB 观测，不重放动作、不使用世界模型预测画面。浮点 RGB 转 PNG 前逐像素检查为无损转换。图中 CD 为复跑自身的粒子状态指标。', '',
        'Ours 指定为 `fixed3`：3/3、固定 S、预热 5 推。对照为 DINO-WM Native 5/5。目标、模型、种子和控制配置与正式运行一致；物理仿真存在波动，复跑指标可能不同。', '',
        '## 预先选定的两对案例', '',
        '开始画面记录前，选定正式试验中改善的 sample 03 和退步的 sample 07；两组各独立执行一次，不重复试跑到有利结果为止。保留两对的全部结果。', '',
        '| 原目标 | Native CD / 推数 | Ours CD / 推数 | Ours S 触发 | Native CD − Ours CD |',
        '|---|---:|---:|---:|---:|']
    for s in selections:
        lines.append(f"| {s['global_sample']} | {s['native_cd']:.6f} / {s['native_pushes']} | {s['ours_cd']:.6f} / {s['ours_pushes']} | {s['ours_S_triggers']} | {s['native_minus_ours_cd']:.6f} |")
    lines += ['', '## 图与原始材料', '']
    for s in selections:
        folder = f"sample_{s['sample']:02d}"
        lines += [f"- 目标 {s['global_sample']}：[最终状态对比]({folder}/final_comparison.png) · [每三推完整序列]({folder}/trajectory_every3.png) · [每一步原始截图](raw_frames/{folder}/)"]
    lines += ['', '成功后停止的时间段明确留空；实际最后一推单列，即使它不是 3 的倍数。最终对比图上排为完整原图，下排采用四幅图相同的裁剪范围放大，未叠加目标、未修改几何形状。', '',
        '`raw_frames/` 保存这四次执行的每一步原图。`cd_by_push.csv` 保存逐推 CD；`provenance.json` 保存原缓存、指标、目标、PNG 哈希与选例规则。`formal_vs_visualization.json` 核验两次运行目标、模型、规划参数及种子相同，并并列保存各自成绩。PNG/PDF 图均由仓库中的 visualize.py 生成。', '',
        '布局参考内部 SE-JEPA 仓库的每三推图示。这里只保留内部来源说明，不向当前论文添加未发表论文的引用。', '',
        '案例用于说明可观测的行为差异，不代表普遍收益；Native-5 与 S-3 同时改变动作块和 S，不是 S 的单因素消融。完整统计请见 [正式结果](../RESULTS.md)。', '']
    (output/'README.md').write_text('\n'.join(lines))
    print(json.dumps({k:v for k,v in manifest.items() if k!='sources'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('eval_outputs/rope_qualitative20'))
    parser.add_argument('--output', type=Path, default=Path('reports/rope_formal20/visuals'))
    parser.add_argument('--formal-root', type=Path, default=Path('eval_outputs/rope_formal20'))
    args = parser.parse_args()
    main(args.root, args.output, args.formal_root)
