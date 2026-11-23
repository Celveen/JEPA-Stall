"""Compare one completed ten-task Native/+S pair, respecting early stopping."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def compare(root, report):
    root, report = Path(root), Path(report)
    paths = [root / arm / 'chunk_00' for arm in ('native', 'S')]
    receipts = [json.loads((path / 'complete.json').read_text()) for path in paths]
    for key in ('target_sha256', 'checkpoint_sha256', 'encoder_sha256', 'implementation_sha256', 'action_mean', 'action_std'):
        assert receipts[0][key] == receipts[1][key], key
    audits = [json.loads((path / 'frozen_model_audit.json').read_text()) for path in paths]
    encoder_audits = [json.loads((path / 'encoder_audit.json').read_text()) for path in paths]
    assert all(audit['matches_official_pretrained_weights'] for audit in encoder_audits)
    assert all(audit['unchanged'] for audit in audits)
    assert audits[0]['parameters_and_buffers_before'] == audits[1]['parameters_and_buffers_before']
    metrics = [dict(np.load(path / 'rope_metrics.npz')) for path in paths]
    results = [dict(np.load(path / 'final_eval_results.npz')) for path in paths]
    for key in ('state_0', 'state_g'):
        np.testing.assert_array_equal(results[0][key], results[1][key])
    records = [json.loads((path / 'ecp_records.json').read_text()) for path in paths]
    costs = json.loads((paths[1] / 'ecp_costs.json').read_text())
    mean = np.asarray(receipts[0]['action_mean'], dtype=np.float32)
    std = np.asarray(receipts[0]['action_std'], dtype=np.float32)
    physical = [result['actions'] * std + mean for result in results]
    rows = []
    for index in range(10):
        n_push, s_push = [int(m['actual_pushes'][index]) for m in metrics]
        common = min(n_push, s_push)
        first_change = int(metrics[1]['first_action_change'][index])
        cutoff = common if first_change < 0 else min(first_change, common)
        np.testing.assert_array_equal(physical[0][index, :cutoff], physical[1][index, :cutoff])
        np.testing.assert_array_equal(metrics[0]['cd_by_replan'][index, :cutoff],
                                      metrics[1]['cd_by_replan'][index, :cutoff])
        if first_change < 0:
            assert n_push == s_push
            assert metrics[0]['final_cd'][index] == metrics[1]['final_cd'][index]
        for arm in range(2):
            np.testing.assert_array_equal(results[arm]['successes'], metrics[arm]['successes'])
            np.testing.assert_array_equal(metrics[arm]['executed_mask'].sum(1), metrics[arm]['actual_pushes'])
        delta = physical[1][index, :common] - physical[0][index, :common]
        changed = np.any(delta != 0, axis=1)
        # Count newly added memories, not cumulative memory occupancy.
        triggers = [sum(record.get('trigger_weight', [0] * 10)[index] > 0 for record in history)
                    for history in records]
        active_costs = [value for key, value in costs.items() if key.endswith(f'_sample{index}')]
        n_cd, s_cd = [float(m['final_cd'][index]) for m in metrics]
        rows.append(dict(sample=index, initial_cd=float(metrics[0]['initial_cd'][index]),
                         native_cd=n_cd, S_cd=s_cd, S_minus_native=s_cd - n_cd,
                         native_success=bool(metrics[0]['successes'][index]),
                         S_success=bool(metrics[1]['successes'][index]),
                         native_pushes=n_push, S_pushes=s_push,
                         native_detector_events=triggers[0], S_trigger_events=triggers[1],
                         first_S_application_push=(int(metrics[1]['first_intervention'][index]) + 1
                                                   if metrics[1]['first_intervention'][index] >= 0 else None),
                         first_changed_push=first_change + 1 if first_change >= 0 else None,
                         compared_executed_pushes=common, changed_executed_pushes=int(changed.sum()),
                         maximum_physical_action_component_difference=float(np.abs(delta).max()) if common else 0.,
                         mean_physical_action_L2_difference=float(np.linalg.norm(delta, axis=1).mean()) if common else 0.,
                         positive_penalty_candidate_evaluations=sum(c['positive_candidates'] for c in active_costs),
                         maximum_added_cost=max([c['maximum_added_cost'] for c in active_costs] or [0.]),
                         checked_identical_prefix_pushes=cutoff))
    difference = metrics[1]['final_cd'] - metrics[0]['final_cd']
    summary = dict(status='complete', paired_samples=10, success_rule='CD < 0.3',
                   S_warmup_pushes=receipts[1]['config']['planner']['ecp']['warmup_pushes'],
                   cadence_pushes=1, maximum_pushes=20,
                   native_mean_cd=float(metrics[0]['final_cd'].mean()),
                   S_mean_cd=float(metrics[1]['final_cd'].mean()),
                   mean_S_minus_native=float(difference.mean()),
                   relative_mean_cd_reduction_percent=float(-100 * difference.mean() / metrics[0]['final_cd'].mean()),
                   native_successes=int(metrics[0]['successes'].sum()), S_successes=int(metrics[1]['successes'].sum()),
                   native_total_pushes=int(metrics[0]['actual_pushes'].sum()),
                   S_total_pushes=int(metrics[1]['actual_pushes'].sum()),
                   native_detector_events=sum(row['native_detector_events'] for row in rows),
                   S_trigger_events=sum(row['S_trigger_events'] for row in rows),
                   S_triggered_samples=sum(row['S_trigger_events'] > 0 for row in rows),
                   action_changed_samples=sum(row['first_changed_push'] is not None for row in rows),
                   changed_executed_pushes_in_common_prefix=sum(row['changed_executed_pushes'] for row in rows),
                   positive_penalty_candidate_evaluations=sum(row['positive_penalty_candidate_evaluations'] for row in rows),
                   maximum_added_cost=max(row['maximum_added_cost'] for row in rows),
                   improved_samples=int((difference < -2e-5).sum()),
                   worsened_samples=int((difference > 2e-5).sum()),
                   unchanged_samples=int((np.abs(difference) <= 2e-5).sum()),
                   native_elapsed_s=audits[0]['elapsed_s'], S_elapsed_s=audits[1]['elapsed_s'],
                   frozen_model_verified=True, exact_pre_divergence_pairing_verified=True)
    report.mkdir(parents=True, exist_ok=True)
    (report / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    with (report / 'per_sample.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# Rope：逐推反馈，10 样本 Native / +S 对照', '',
             '同一权重、同一批 chunk_0 目标和随机种子；每推观测，最多 20 推，CD < 0.3 即停。',
             f"S 预热 {summary['S_warmup_pushes']} 推，从第 {summary['S_warmup_pushes'] + 1} 推起允许干涉。", '',
             '| 指标 | Native | +S |', '|---|---:|---:|',
             f"| 平均最终 CD | {summary['native_mean_cd']:.8f} | {summary['S_mean_cd']:.8f} |",
             f"| 成功数 | {summary['native_successes']}/10 | {summary['S_successes']}/10 |",
             f"| 实际推动总数 | {summary['native_total_pushes']} | {summary['S_total_pushes']} |",
             f"| 检测事件数 | {summary['native_detector_events']}（只记录） | {summary['S_trigger_events']} |", '',
             f"+S 触发 {summary['S_triggered_samples']}/10 个样本；实际改变动作 {summary['action_changed_samples']}/10 个样本。",
             f"共同执行区间有 {summary['changed_executed_pushes_in_common_prefix']} 推动作不同，"
             f"{summary['positive_penalty_candidate_evaluations']} 次候选评估获得正惩罚。", '',
             f"CD 改善 {summary['improved_samples']} 个、退步 {summary['worsened_samples']} 个、容差内不变 {summary['unchanged_samples']} 个。", '',
             '| 样本 | Native CD | +S CD | Native 推数 | +S 推数 | S 触发数 | 首次改变第几推 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    lines.extend(f"| {r['sample']} | {r['native_cd']:.6f} | {r['S_cd']:.6f} | {r['native_pushes']} | {r['S_pushes']} | {r['S_trigger_events']} | {r['first_changed_push'] or '无'} |" for r in rows)
    lines += ['', '成功样本立即退出 CEM 优化和物理执行，后续数组填充不计为动作。权重未改变，首次动作分歧之前的控制与 CD 均完全一致。', '',
              '这是固定 10 个目标的一次配对评测。旧报告使用不同的任务数量、观测频率及推动预算，不能据此单独归因逐推反馈的效果。仿真仍沿用每次 rollout 恢复粒子位置的接口。', '']
    (report / 'RESULTS.md').write_text('\n'.join(lines))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compare(args.root, args.report), indent=2))
