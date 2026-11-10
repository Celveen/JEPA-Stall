"""Paired CD and pre-intervention checks; no simulator or model inference."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def analyze_pair(native_dir, ecp_dir, chunk):
    native_dir, ecp_dir = Path(native_dir), Path(ecp_dir)
    a_receipt = json.loads((native_dir / 'complete.json').read_text())
    b_receipt = json.loads((ecp_dir / 'complete.json').read_text())
    assert a_receipt['checkpoint_sha256'] == b_receipt['checkpoint_sha256']
    assert a_receipt['target_sha256'] == b_receipt['target_sha256']
    a_audit = json.loads((native_dir / 'frozen_model_audit.json').read_text())
    b_audit = json.loads((ecp_dir / 'frozen_model_audit.json').read_text())
    assert a_audit['unchanged'] and b_audit['unchanged']
    assert a_audit['parameters_and_buffers_before'] == b_audit['parameters_and_buffers_before']
    a, b = np.load(native_dir / 'final_eval_results.npz'), np.load(ecp_dir / 'final_eval_results.npz')
    for key in ['state_0', 'state_g']:
        np.testing.assert_array_equal(a[key], b[key])
    am, bm = np.load(native_dir / 'rope_metrics.npz'), np.load(ecp_dir / 'rope_metrics.npz')
    rows = []
    for index, first in enumerate(bm['first_intervention']):
        cutoff = a['actions'].shape[1] if first < 0 else int(first) * 5
        np.testing.assert_allclose(a['actions'][index, :cutoff], b['actions'][index, :cutoff],
                                   atol=1e-5, rtol=1e-5)
        prefix_replans = am['cd_by_replan'].shape[1] if first < 0 else int(first)
        np.testing.assert_allclose(am['cd_by_replan'][index, :prefix_replans],
                                   bm['cd_by_replan'][index, :prefix_replans],
                                   atol=2e-5, rtol=2e-5)
        native_cd = float(am['cd_by_replan'][index, -1])
        ecp_cd = float(bm['cd_by_replan'][index, -1])
        rows.append({'chunk': chunk, 'sample': index, 'native_cd': native_cd, 'S_cd': ecp_cd,
                     'S_minus_native': ecp_cd - native_cd,
                     'first_intervention_plan_zero_based': int(first),
                     'pre_intervention_actions_checked': cutoff,
                     'maximum_prefix_action_difference': float(np.max(np.abs(
                         a['actions'][index, :cutoff] - b['actions'][index, :cutoff])))})
    return rows


def summarize(root, report):
    root, report = Path(root), Path(report)
    rows = []
    for chunk in range(5):
        native, ecp = root / 'native' / f'chunk_{chunk:02d}', root / 'S' / f'chunk_{chunk:02d}'
        if (native / 'complete.json').exists() and (ecp / 'complete.json').exists():
            rows.extend(analyze_pair(native, ecp, chunk))
    result = {'status': 'complete' if len(rows) == 50 else 'partial', 'paired_episodes': len(rows),
              'planned_paired_episodes': 50, 'primary_metric': 'final particle Chamfer distance; lower is better',
              'training_or_online_updates': False, 'independent_seed_repetitions': 1,
              'interpretation': 'fixed 50-task pilot; not three-seed confirmation',
              'prefix_action_tolerance': {'atol': 1e-5, 'rtol': 1e-5},
              'prefix_cd_tolerance': {'atol': 2e-5, 'rtol': 2e-5}}
    if rows:
        native = np.asarray([row['native_cd'] for row in rows])
        ecp = np.asarray([row['S_cd'] for row in rows])
        difference = ecp - native
        rng = np.random.RandomState(20261110)
        bootstrap = difference[rng.randint(len(rows), size=(10000, len(rows)))].mean(1)
        result.update({'native_mean_cd': float(native.mean()), 'S_mean_cd': float(ecp.mean()),
                       'mean_S_minus_native': float(difference.mean()),
                       'relative_mean_cd_reduction_percent': float(100 * (native.mean() - ecp.mean()) / native.mean()),
                       'improved_episodes': int((difference < -2e-5).sum()),
                       'worsened_episodes': int((difference > 2e-5).sum()),
                       'unchanged_within_2e5': int((np.abs(difference) <= 2e-5).sum()),
                       'triggered_episodes': sum(row['first_intervention_plan_zero_based'] >= 0 for row in rows),
                       'exploratory_paired_mean_delta_ci95': np.quantile(bootstrap, [.025, .975]).tolist(),
                       'all_pre_intervention_checks_passed': True})
    report.mkdir(parents=True, exist_ok=True)
    name = 'summary.json' if result['status'] == 'complete' else 'progress_summary.json'
    (report / name).write_text(json.dumps(result, indent=2) + '\n')
    if rows:
        with (report / 'per_episode.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    if result['status'] == 'complete':
        text = ('# Rope 冻结模型 + 停滞正则：50 目标试跑\n\n'
                '仅 S，无训练、在线更新或本体加权；同一模型、目标、种子和 CEM。'
                '主指标为 15 次推动后的粒子 CD，越低越好。\n\n'
                f"| 条件 | 平均最终 CD |\n|---|---:|\n| Native | {result['native_mean_cd']:.8f} |\n"
                f"| +S | {result['S_mean_cd']:.8f} |\n\n"
                f"S−Native 配对均差：{result['mean_S_minus_native']:.8f}；"
                f"平均 CD 相对降低 {result['relative_mean_cd_reduction_percent']:.2f}%。\n\n"
                f"50 个任务中，改善 {result['improved_episodes']}，退步 {result['worsened_episodes']}，"
                f"容差内不变 {result['unchanged_within_2e5']}；"
                f"实际触发 {result['triggered_episodes']}。全部干预前动作与 CD 核验通过。\n\n"
                '这是固定旧目标集的一次冻结迁移试跑，不是三种子确认。'
                '配对 bootstrap 区间仅作探索性任务重采样描述；未建立训练来源依赖模型。'
                '没有按本次结果修改阈值或选择种子。粒子状态仅参与评测。\n')
        (report / 'RESULTS.md').write_text(text)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.root, args.report), indent=2))
