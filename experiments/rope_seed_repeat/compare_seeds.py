"""Compare archived seeds with one common physical-control difference threshold."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
ARMS = ('native3', 'fixed3', 'dynamic3', 'fixed3_nowarm')
LABELS = ('Native 3/3', '固定 S 3/3', '动态 S 3/3', '固定 S 3/3 无预热')
THRESHOLD = 1e-6


def write_csv(path, rows):
    with path.open('w') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def compare(root, output):
    root, output = Path(root), Path(output)
    summary = json.loads((output / 'summary.json').read_text())
    seed = summary['base_seed']
    assert seed == 100, 'This follow-up interpretation is scoped to the requested seed-100 repeat.'
    elapsed_minutes = json.loads((root / 'queue.json').read_text())['elapsed_s'] / 60
    old = json.loads((REPO / 'reports/rope_formal20_native3/summary.json').read_text())
    assert summary['status'] == old['status'] == 'complete'
    rows, aggregates, targets = [], {}, {}
    for current_seed in (99, seed):
        def folder(arm, i):
            base = root if current_seed == seed else REPO / ('eval_outputs/rope_formal20_native3' if arm == 'native3' else 'eval_outputs/rope_formal20')
            return base / arm / f'sample_{i:02d}'
        for arm in ARMS:
            for i in range(20):
                p = folder(arm, i)
                receipt = json.loads((p / 'complete.json').read_text())
                assert receipt['config']['seed'] == current_seed
                with np.load(p / 'final_eval_results.npz') as f:
                    final = dict(f)
                with np.load(p / 'rope_metrics.npz') as f:
                    metrics = dict(f)
                key = i
                for name in ('state_0', 'state_g'):
                    if (key, name) in targets:
                        np.testing.assert_array_equal(final[name], targets[key, name])
                    targets[key, name] = final[name]
                if arm == 'native3':
                    continue
                with np.load(folder('native3', i) / 'final_eval_results.npz') as f:
                    native = f['actions'][0]
                std = np.asarray(receipt['action_std'], dtype=np.float32)
                mean = np.asarray(receipt['action_mean'], dtype=np.float32)
                a, b = native * std + mean, final['actions'][0] * std + mean
                common = min(len(a), len(b))
                distance = np.linalg.norm(b[:common] - a[:common], axis=1)
                changed = np.flatnonzero(distance > THRESHOLD)
                first_s = int(metrics['first_intervention'][0])
                rows.append(dict(seed=current_seed, arm=arm, sample=i, global_sample=i + 10,
                    common_pushes=common, changed_executed_pushes=int(len(changed)),
                    first_control_difference_push=int(changed[0] + 1) if len(changed) else None,
                    first_S_application_push=first_s + 1 if first_s >= 0 else None,
                    difference_before_or_without_S=bool(len(changed) and (first_s < 0 or changed[0] < first_s)),
                    changed_pushes_at_or_after_S=int(sum(changed >= first_s)) if first_s >= 0 else 0,
                    maximum_physical_control_L2=float(distance.max()) if common else 0.,
                    physical_L2_threshold=THRESHOLD))
        for arm in ARMS[1:]:
            selected = [r for r in rows if r['seed'] == current_seed and r['arm'] == arm]
            aggregates[current_seed, arm] = dict(changed_samples=sum(r['changed_executed_pushes'] > 0 for r in selected),
                changed_pushes=sum(r['changed_executed_pushes'] for r in selected),
                common_pushes=sum(r['common_pushes'] for r in selected),
                samples_with_difference_before_or_without_S=sum(r['difference_before_or_without_S'] for r in selected),
                applied_samples=sum(r['first_S_application_push'] is not None for r in selected),
                changed_pushes_at_or_after_S=sum(r['changed_pushes_at_or_after_S'] for r in selected))
    comparison = []
    for arm in ARMS:
        previous, current = old['conditions'][arm], summary['conditions'][arm]
        row = dict(arm=arm, old_seed=99, new_seed=seed,
                   old_cd=previous['mean_cd'], new_cd=current['mean_cd'],
                   cd_delta=current['mean_cd'] - previous['mean_cd'],
                   old_successes=previous['successes'], new_successes=current['successes'],
                   old_S_trigger_events=previous['S_trigger_events'], new_S_trigger_events=current['S_trigger_events'],
                   old_applied_samples=previous['applied_samples'], new_applied_samples=current['applied_samples'])
        for prefix, current_seed in [('old', 99), ('new', seed)]:
            for key in ('changed_samples', 'changed_pushes', 'common_pushes', 'samples_with_difference_before_or_without_S'):
                row[prefix + '_' + key] = aggregates.get((current_seed, arm), {}).get(key, 0)
        comparison.append(row)
        if arm != 'native3':
            assert row['old_applied_samples'] == aggregates[99, arm]['applied_samples']
            assert row['new_applied_samples'] == aggregates[seed, arm]['applied_samples']
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / 'action_comparisons_both_seeds.csv', rows)
    write_csv(output / 'seed_comparison.csv', comparison)
    artifact = dict(old_seed=99, new_seed=seed, samples_per_arm=20,
                    physical_control_L2_threshold=THRESHOLD, paired_targets_verified=True,
                    action_comparison_definition='Within each seed, compare S versus Native3 over their common executed prefix; four-dimensional denormalized Rope controls.',
                    interpretation='Differences before S or without S exist under independent physics. Counts after S are temporal associations, not a counterfactual attribution to S.',
                    comparisons=comparison, action_summaries=[dict(seed=s, arm=a, **v) for (s, a), v in aggregates.items()])
    (output / 'seed_comparison.json').write_text(json.dumps(artifact, indent=2) + '\n')
    lines = [f'# Rope：种子 99 → {seed} 完整对照', '',
             f'新种子四组全部完成，共 80 条 episode，运行耗时约 {elapsed_minutes:.1f} 分钟。两轮使用相同的 20 个目标、模型、3/3 节奏和 CEM 参数；最多 20 推，每推观测，CD <0.3 即停。', '',
             f'下表箭头均为 **种子 99 → {seed}**。', '',
             '| 条件 | 平均 CD↓ | 成功数 /20 | S 触发次数 | 实际施加 S 的样本 /20 |',
             '|---|---:|---:|---:|---:|']
    for r, label in zip(comparison, LABELS):
        lines.append(f"| {label} | {r['old_cd']:.4f} → {r['new_cd']:.4f} | {r['old_successes']} → {r['new_successes']} | {r['old_S_trigger_events']} → {r['new_S_trigger_events']} | {r['old_applied_samples']} → {r['new_applied_samples']} |")
    lines += ['', '## 实际执行动作变化', '',
              '每个种子内，与同目标的 Native 3/3 比较。将四维动作反归一化后，以 L2 差 >1e-6 判定变化；只统计两边都实际执行过的推，不把成功后的空缺补成动作。两轮均从原始动作重新计算，采用相同阈值。', '',
              '| 条件 | 有动作差异的样本 /20（99→新） | 改变推数 /共同执行推数（99→新） | 首次差异早于 S 或始终未用 S 的样本（99→新） |',
              '|---|---:|---:|---:|']
    for r, label in zip(comparison[1:], LABELS[1:]):
        lines.append(f"| {label} | {r['old_changed_samples']} → {r['new_changed_samples']} | {r['old_changed_pushes']}/{r['old_common_pushes']} → {r['new_changed_pushes']}/{r['new_common_pushes']} | {r['old_samples_with_difference_before_or_without_S']} → {r['new_samples_with_difference_before_or_without_S']} |")
    lines += ['', '动作差异不能全部归因于 S：各组独立执行物理仿真，干预前也会分叉。S 触发与实际使用也不同：在一个动作块内产生的记忆，只有下一块规划时才会用到，任务可能在此之前已成功。', '',
              '## 本轮结论', '']
    for arm, label in zip(ARMS[1:], LABELS[1:]):
        s = summary['paired_vs_native3'][arm]
        lines.append(f"- {label} 相对同种子 Native：平均 CD 差 {s['mean_cd_delta']:+.4f}，新增成功 {s['gained_successes']} 个、丢失成功 {s['lost_successes']} 个。")
    lines += ['', '固定 S 的两种配置在种子 100 均达到 16/20，无预热的平均 CD 最低；动态 S 在种子 99 的优势没有在本轮延续。两种子结果支持固定 S 作为后续重点，但不足以证明其对所有种子都优于动态 S。', '',
              '两轮动态 S 各只有 2 条实际使用的记忆强度处于 0 与 2 之间，固定／动态强度的实质差异仍较少；不能把全部成绩差异都解释为强度策略本身。', '',
              '[汇总 CSV](seed_comparison.csv) · [两轮全部动作对照](action_comparisons_both_seeds.csv) · [机器可读统计与定义](seed_comparison.json) · [完整实验报告和证据](README.md)', '']
    (output / 'SEED_COMPARISON.md').write_text('\n'.join(lines))
    return artifact


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=REPO / 'eval_outputs/rope_seed100_20')
    parser.add_argument('--output', type=Path, default=REPO / 'reports/rope_seed100_20')
    args = parser.parse_args()
    result = compare(args.root, args.output)
    print(json.dumps(result['comparisons'], indent=2))
