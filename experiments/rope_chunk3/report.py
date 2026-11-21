"""Audit and summarize all ten episodes in each chunk/strength condition."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ARMS = ('native3', 'fixed3', 'dynamic3', 'native5')
LABELS = {'native3': 'Native-3', 'fixed3': '固定强度 S',
          'dynamic3': '动态强度 S', 'native5': 'Native-5'}


def summarize(root, output, arms=ARMS, maximum_pushes=20):
    root, output = Path(root), Path(output)
    if tuple(arms) not in (ARMS, ARMS[:3]):
        raise ValueError('Expected native3, fixed3, dynamic3, and optionally native5, in that order.')
    missing = [f'{arm}/sample_{sample:02d}' for arm in arms for sample in range(10)
               if not (root / arm / f'sample_{sample:02d}/complete.json').exists()]
    if missing:
        raise RuntimeError(f'All {10 * len(arms)} predetermined episodes must complete; missing {missing}')
    rows, data = [], {}
    for arm in arms:
        for sample in range(10):
            path = root / arm / f'sample_{sample:02d}'
            receipt = json.loads((path / 'complete.json').read_text())
            assert receipt['config']['planner']['max_iter'] == maximum_pushes
            audit = json.loads((path / 'frozen_model_audit.json').read_text())
            encoder = json.loads((path / 'encoder_audit.json').read_text())
            chunk = json.loads((path / 'chunk_audit.json').read_text())
            records = json.loads((path / 'ecp_records.json').read_text())
            costs = json.loads((path / 'ecp_costs.json').read_text())
            metrics = dict(np.load(path / 'rope_metrics.npz'))
            results = dict(np.load(path / 'final_eval_results.npz'))
            assert audit['unchanged'] and encoder['matches_official_pretrained_weights']
            cd, pushes = float(metrics['final_cd'][0]), int(metrics['actual_pushes'][0])
            expected_queries = list(range(0, pushes, chunk['chunk_size']))
            assert chunk['optimizer_push_indices_zero_based'] == expected_queries
            assert chunk['optimizer_calls'] == len(expected_queries)
            assert bool(metrics['successes'][0]) == (cd < .3)
            assert pushes <= maximum_pushes and (cd < .3 or pushes == maximum_pushes)
            timeline = np.r_[metrics['initial_cd'][0], metrics['cd_by_replan'][0]]
            hits = np.flatnonzero(timeline < .3)
            assert not len(hits) or int(hits[0]) == pushes
            weights = [r['new_memory_strength'] for r in records if r.get('trigger_weight', [0.])[0] > 0]
            row = dict(arm=arm, sample=sample, initial_cd=float(metrics['initial_cd'][0]),
                       final_cd=cd, success=cd < .3, actual_pushes=pushes,
                       optimizer_calls=chunk['optimizer_calls'],
                       candidate_evaluations=chunk['optimizer_calls'] * 10 * 300,
                       detector_events=len(weights), min_trigger_strength=min(weights) if weights else 0.,
                       fractional_severity_events=sum(0 < r.get('severity_weight', 0.) < 1 for r in records),
                       max_trigger_strength=max(weights) if weights else 0.,
                       first_S_application_push=(int(metrics['first_intervention'][0]) + 1
                                                 if metrics['first_intervention'][0] >= 0 else None),
                       first_changed_push=(int(metrics['first_action_change'][0]) + 1
                                           if metrics['first_action_change'][0] >= 0 else None),
                       branch_chunk_push=(chunk['first_branched_chunk_push_zero_based'] + 1
                                          if chunk['first_branched_chunk_push_zero_based'] is not None else None),
                       positive_penalty_candidate_evaluations=sum(c['positive_candidates'] for c in costs.values()),
                       maximum_added_cost=max([c['maximum_added_cost'] for c in costs.values()] or [0.]),
                       elapsed_s=audit['elapsed_s'],
                       planning_s=sum(record['planning_s'] for record in records),
                       final_cd_minus_native3=0.,
                       compared_executed_pushes=0, changed_executed_pushes=0,
                       maximum_physical_action_component_difference=0.)
            data[arm, sample] = dict(receipt=receipt, audit=audit, chunk=chunk, metrics=metrics,
                                     results=results, row=row)
            rows.append(row)
    for arm in arms[1:]:
        for sample in range(10):
            baseline, actual = data['native3', sample], data[arm, sample]
            for key in ('sample', 'full_target_sha256', 'checkpoint_sha256', 'encoder_sha256', 'implementation_sha256'):
                assert baseline['receipt'][key] == actual['receipt'][key], key
            assert baseline['audit']['parameters_and_buffers_before'] == actual['audit']['parameters_and_buffers_before']
            for key in ('state_0', 'state_g'):
                np.testing.assert_array_equal(baseline['results'][key], actual['results'][key])
            row, reference = actual['row'], baseline['row']
            row['final_cd_minus_native3'] = row['final_cd'] - reference['final_cd']
            if arm == 'native5':
                continue
            common = min(row['actual_pushes'], reference['actual_pushes'])
            a = baseline['results']['actions'][0, :common]
            b = actual['results']['actions'][0, :common]
            branch = actual['chunk']['first_branched_chunk_push_zero_based']
            cutoff = common if branch is None else min(branch, common)
            np.testing.assert_array_equal(a[:cutoff], b[:cutoff])
            np.testing.assert_array_equal(baseline['metrics']['cd_by_replan'][0, :cutoff],
                                          actual['metrics']['cd_by_replan'][0, :cutoff])
            if branch is None:
                assert row['actual_pushes'] == reference['actual_pushes']
                assert row['final_cd'] == reference['final_cd']
            std = np.asarray(actual['receipt']['action_std'], dtype=np.float32)
            mean = np.asarray(actual['receipt']['action_mean'], dtype=np.float32)
            delta = (b * std + mean) - (a * std + mean)
            row.update(compared_executed_pushes=common,
                       changed_executed_pushes=int(np.any(delta != 0, axis=1).sum()),
                       maximum_physical_action_component_difference=float(np.abs(delta).max()) if common else 0.)
    strength_pairs = []
    for sample in range(10):
        fixed, dynamic = data['fixed3', sample], data['dynamic3', sample]
        a, b = fixed['results']['actions'][0], dynamic['results']['actions'][0]
        common = min(len(a), len(b))
        equal = a.shape == b.shape and np.array_equal(a, b)
        control_diff = np.flatnonzero(np.any(a[:common] != b[:common], axis=1))
        cd_diff = np.flatnonzero(fixed['metrics']['cd_by_replan'][0, :common] !=
                                 dynamic['metrics']['cd_by_replan'][0, :common])
        strength_pairs.append(dict(sample=sample, all_executed_controls_identical=equal,
                                   controls_identical_on_common_steps=np.array_equal(a[:common], b[:common]),
                                   fixed_pushes=len(a), dynamic_pushes=len(b),
                                   first_control_difference_push=int(control_diff[0]) + 1 if len(control_diff) else None,
                                   first_cd_difference_push=int(cd_diff[0]) + 1 if len(cd_diff) else None,
                                   dynamic_minus_fixed_cd=dynamic['row']['final_cd'] - fixed['row']['final_cd']))
    summary = dict(status='complete', samples_per_condition=10, maximum_pushes=maximum_pushes,
                   success_rule='CD < 0.3', warmup_pushes=5, conditions={},
                   fixed_dynamic_control_comparison=strength_pairs)
    for arm in arms:
        selected = [row for row in rows if row['arm'] == arm]
        delta = np.asarray([row['final_cd_minus_native3'] for row in selected])
        summary['conditions'][arm] = dict(
            mean_cd=float(np.mean([row['final_cd'] for row in selected])),
            successes=sum(row['success'] for row in selected),
            total_pushes=sum(row['actual_pushes'] for row in selected),
            optimizer_calls=sum(row['optimizer_calls'] for row in selected),
            detector_events=sum(row['detector_events'] for row in selected),
            fractional_severity_events=sum(row['fractional_severity_events'] for row in selected),
            triggered_samples=sum(row['detector_events'] > 0 for row in selected),
            action_changed_samples=sum(row['first_changed_push'] is not None for row in selected),
            changed_executed_pushes=sum(row['changed_executed_pushes'] for row in selected),
            positive_penalty_candidate_evaluations=sum(row['positive_penalty_candidate_evaluations'] for row in selected),
            maximum_added_cost=max(row['maximum_added_cost'] for row in selected),
            summed_episode_elapsed_s=sum(row['elapsed_s'] for row in selected),
            summed_planning_s=sum(row['planning_s'] for row in selected),
            mean_cd_minus_native3=float(delta.mean()), improved_vs_native3=int((delta < -2e-5).sum()),
            worsened_vs_native3=int((delta > 2e-5).sum()),
            branches_without_executed_action_change=sum(row['branch_chunk_push'] is not None and
                                                       row['first_changed_push'] is None for row in selected))
    summary['strength_contrast_exercised'] = any(
        summary['conditions'][arm]['fractional_severity_events'] > 0
        for arm in ('fixed3', 'dynamic3'))
    output.mkdir(parents=True, exist_ok=True)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    with (output / 'per_sample.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# Rope：动作块与固定/动态惩罚强度', '',
             f'每组同一批 10 个目标，最多 {maximum_pushes} 推，逐推观测，CD < 0.3 即停；官方预训练编码器逐张量核验，模型保持冻结。', '',
             '| 条件 | 平均最终 CD | 成功数 | 总推动数 | CEM 调用数 | 检测事件 | 动作改变样本 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for arm, result in summary['conditions'].items():
        changed = result['action_changed_samples'] if arm in ('fixed3', 'dynamic3') else '—'
        lines.append(f"| {LABELS[arm]} | {result['mean_cd']:.6f} | {result['successes']}/10 | {result['total_pushes']} | {result['optimizer_calls']} | {result['detector_events']} | {changed} |")
    lines += ['', 'Native 的检测事件只作诊断，不参与动作优化。固定 S 的每条触发记忆系数为 2，动态 S 为 2×w；触发规则相同。5 推预热，新的记忆在下一块规划生效，3 推块最早影响第 7 推。', '',
              f"动态版本共有 {summary['conditions']['dynamic3']['detector_events']} 次触发，其中 {summary['conditions']['dynamic3']['fractional_severity_events']} 次使用了 0<w<1 的非饱和强度。", '',
              '各组使用独立 episode 环境、相同目标和种子安排；块内不恢复场景。此前批量逐推实验的样本隔离和随机流组织不同，其数值只能作为补充参考。S 的首次分支块之前，动作和 CD 已逐项核验一致；首次实际动作改变另列在逐样本 CSV 中。', '',
              f"固定/动态强度之间，有 {sum(pair['all_executed_controls_identical'] for pair in strength_pairs)}/10 个样本的全部执行动作完全相同；这些样本若 CD 不同，应按独立物理仿真的波动解释，不能归因于强度策略。w=1 的触发下，两种系数都等于 2。", '',
              '耗时字段为各 episode 评测阶段的累加时间，不是队列墙钟时间。`summed_planning_s` 记录规划时间，其余包括物理执行、观测、检测与记录。S 的相同动作前缀复用 Native 轨迹，因此 S 的运行耗时不能当作独立部署速度。', '',
              '分支沿用原环境在块边界恢复粒子位置的接口。无执行动作改变的分支结果不能归因于惩罚收益。这是固定 10 样本试跑，不是多种子确认。', '']
    if not summary['strength_contrast_exercised']:
        lines += ['**本轮没有实际区分固定/动态强度。** 两个 S 版本的全部触发均为 w=1，单条记忆系数都是 2。固定版的较低 CD 不能归因于固定强度优于动态强度。', '']
        for pair in strength_pairs:
            if pair['first_control_difference_push'] is not None:
                lines.append(f"- 样本 {pair['sample']}：相同控制下的 CD 在第 {pair['first_cd_difference_push']} 推开始不同，控制到第 {pair['first_control_difference_push']} 推才分歧。闭环会放大仿真波动。")
        lines.append('')
    (output / 'RESULTS.md').write_text('\n'.join(lines))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-pushes', type=int, choices=(15, 20), default=20)
    parser.add_argument('--arms', default=','.join(ARMS))
    args = parser.parse_args()
    print(json.dumps(summarize(args.root, args.output, tuple(args.arms.split(',')), args.max_pushes), indent=2))
