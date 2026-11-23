"""Separate the within-trajectory 15/20 comparison from an independent 15-push rerun."""

import argparse
import copy
import csv
import json
from pathlib import Path

import numpy as np

ARMS = ('native3', 'fixed3', 'dynamic3')


def compare(reference, rerun, output):
    reference, rerun, output = map(Path, (reference, rerun, output))
    rows, summary = [], {'status': 'complete', 'samples_per_condition': 10, 'conditions': {}}
    for arm in ARMS:
        selected = []
        for sample in range(10):
            paths = [root / arm / f'sample_{sample:02d}' for root in (reference, rerun)]
            receipts = [json.loads((p / 'complete.json').read_text()) for p in paths]
            for key in ('sample', 'full_target_sha256', 'checkpoint_sha256', 'encoder_sha256',
                        'action_mean', 'action_std'):
                assert receipts[0][key] == receipts[1][key], key
            configs = [copy.deepcopy(receipt['config']) for receipt in receipts]
            assert [c['planner']['max_iter'] for c in configs] == [20, 15]
            for config in configs:
                for key in ('saved_folder', 'goal_file_path'):
                    config.pop(key)
                config['planner'].pop('max_iter')
                config['planner']['ecp'].pop('reference_dir', None)
            assert configs[0] == configs[1], 'Another protocol setting changed.'
            for key, digest in receipts[0]['implementation_sha256'].items():
                if key != 'experiments/rope_chunk3/worker.py':
                    assert receipts[1]['implementation_sha256'][key] == digest, key
            metrics = [dict(np.load(p / 'rope_metrics.npz')) for p in paths]
            results = [dict(np.load(p / 'final_eval_results.npz')) for p in paths]
            for key in ('state_0', 'state_g'):
                np.testing.assert_array_equal(results[0][key], results[1][key])
            old, new = metrics
            old_pushes, new_pushes = int(old['actual_pushes'][0]), int(new['actual_pushes'][0])
            cutoff = min(15, old_pushes)
            prefix_cd = float(old['cd_by_replan'][0, cutoff - 1]) if cutoff else float(old['initial_cd'][0])
            common = min(cutoff, new_pushes)
            a, b = [result['actions'][0, :common] for result in results]
            action_diff = np.flatnonzero(np.any(a != b, axis=1))
            cd_diff = np.flatnonzero(old['cd_by_replan'][0, :common] != new['cd_by_replan'][0, :common])
            row = dict(arm=arm, sample=sample, original_15_prefix_cd=prefix_cd,
                       original_15_prefix_success=prefix_cd < .3,
                       original_15_prefix_pushes=cutoff,
                       original_20_cd=float(old['final_cd'][0]),
                       original_20_success=bool(old['successes'][0]), original_20_pushes=old_pushes,
                       rerun_15_cd=float(new['final_cd'][0]), rerun_15_success=bool(new['successes'][0]),
                       rerun_15_pushes=new_pushes,
                       late_success_in_original=bool(old['successes'][0]) and prefix_cd >= .3,
                       first_rerun_action_difference_push=int(action_diff[0]) + 1 if len(action_diff) else None,
                       first_rerun_cd_difference_push=int(cd_diff[0]) + 1 if len(cd_diff) else None)
            assert new_pushes <= 15 and (row['rerun_15_success'] or new_pushes == 15)
            assert row['rerun_15_success'] == (row['rerun_15_cd'] < .3)
            selected.append(row)
            rows.append(row)
        summary['conditions'][arm] = dict(
            original_15_prefix_mean_cd=float(np.mean([r['original_15_prefix_cd'] for r in selected])),
            original_15_prefix_successes=sum(r['original_15_prefix_success'] for r in selected),
            original_20_mean_cd=float(np.mean([r['original_20_cd'] for r in selected])),
            original_20_successes=sum(r['original_20_success'] for r in selected),
            rerun_15_mean_cd=float(np.mean([r['rerun_15_cd'] for r in selected])),
            rerun_15_successes=sum(r['rerun_15_success'] for r in selected),
            late_success_samples=[r['sample'] for r in selected if r['late_success_in_original']])
    summary['interpretation'] = ('Original-15-prefix and original-20 share the same trajectories and isolate '
                                 'the stopping budget. The fresh 15-push rerun is an independent physics '
                                 'replicate; its difference from the original prefix is not a budget effect.')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'budget_comparison.json').write_text(json.dumps(summary, indent=2) + '\n')
    with (output / 'per_sample_budget.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# Rope 3／3：15 推与 20 推', '',
             '所有条件为同一批 10 个目标，逐推观测，5 推预热，CD < 0.3 即停。新运行只改变推动上限；目标、权重、种子及其他配置已核验。', '',
             '| 条件 | 原运行截到 15 推 CD／成功 | 原运行 20 推 CD／成功 | 独立补跑 15 推 CD／成功 |',
             '|---|---:|---:|---:|']
    for arm, result in summary['conditions'].items():
        lines.append(f"| {arm} | {result['original_15_prefix_mean_cd']:.6f}／{result['original_15_prefix_successes']}/10 | "
                     f"{result['original_20_mean_cd']:.6f}／{result['original_20_successes']}/10 | "
                     f"{result['rerun_15_mean_cd']:.6f}／{result['rerun_15_successes']}/10 |")
    lines += ['', '原轨迹截到 15 推不是新实验，表示若同一次执行在第 15 推终止会得到什么结果；它与原 20 推结果之间的差异才能隔离预算效果。独立补跑重新执行模拟器，可能因微小物理/图像差异而在后续规划中分叉，不能把这种变化归因于推动上限。', '',
              'S 的触发、强度与动作差异以本目录 RESULTS.md 和 per_sample.csv 为准。运行器增加了预算参数，原 20 推入口已保存在 ../rope_chunk3_strength_20261116/source/；新旧实现其余审计文件哈希一致。', '']
    (output / 'BUDGET_COMPARISON.md').write_text('\n'.join(lines))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--rerun', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compare(args.reference, args.rerun, args.output), indent=2))
