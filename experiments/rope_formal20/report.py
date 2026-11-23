"""Audit the four independent twenty-target formal Rope conditions."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

ARMS = ('native5', 'fixed3', 'dynamic3', 'fixed3_nowarm')


def report(root, output):
    root, output = Path(root), Path(output)
    missing = [f'{a}/{i:02d}' for a in ARMS for i in range(20)
               if not (root / a / f'sample_{i:02d}/complete.json').exists()]
    if missing:
        raise RuntimeError(f'Wait for all 80 episodes: {missing}')
    rows, data = [], {}
    for arm in ARMS:
        for i in range(20):
            p = root / arm / f'sample_{i:02d}'
            receipt = json.loads((p / 'complete.json').read_text())
            frozen = json.loads((p / 'frozen_model_audit.json').read_text())
            encoder = json.loads((p / 'encoder_audit.json').read_text())
            chunk = json.loads((p / 'chunk_audit.json').read_text())
            records = json.loads((p / 'ecp_records.json').read_text())
            costs = json.loads((p / 'ecp_costs.json').read_text())
            reuse = json.loads((p / 'prefix_reuse.json').read_text())
            assert not any(r['reused'] for r in reuse), 'Formal episodes must use independent physics'
            m, r = dict(np.load(p / 'rope_metrics.npz')), dict(np.load(p / 'final_eval_results.npz'))
            cfg = receipt['config']['planner']
            h, warm = (5 if arm == 'native5' else 3), (0 if arm == 'fixed3_nowarm' else 5)
            assert cfg['max_iter'] == 20 and cfg['chunk_size'] == h
            assert cfg['sub_planner']['horizon'] == h and cfg['ecp']['warmup_pushes'] == warm
            assert cfg['ecp']['independent_execution'] and 'reference_dir' not in cfg['ecp']
            assert receipt['global_sample'] == i + 10
            assert frozen['unchanged'] and encoder['matches_official_pretrained_weights']
            cd, pushes = float(m['final_cd'][0]), int(m['actual_pushes'][0])
            assert bool(m['successes'][0]) == (cd < .3)
            assert 0 <= pushes <= 20 and (cd < .3 or pushes == 20)
            timeline = np.r_[m['initial_cd'][0], m['cd_by_replan'][0]]
            hits = np.flatnonzero(timeline < .3)
            assert not len(hits) or int(hits[0]) == pushes
            assert chunk['optimizer_push_indices_zero_based'] == list(range(0, pushes, h))
            events = [v for v in records if v.get('severity_weight', 0.) > 0]
            queries = chunk['optimizer_push_indices_zero_based']
            used_events = [v for v in events if any(q >= v['completed_pushes'] for q in queries)]
            first = int(m['first_intervention'][0])
            if first >= 0:
                assert first >= (3 if warm == 0 else 6) and first % h == 0
            row = dict(arm=arm, sample=i, global_sample=i + 10, final_cd=cd, success=cd < .3,
                       actual_pushes=pushes, optimizer_calls=chunk['optimizer_calls'],
                       detector_events=len(events), S_trigger_events=0 if arm == 'native5' else len(events),
                       used_trigger_memories=0 if arm == 'native5' else len(used_events),
                       used_fractional_strength_memories=sum(0 < v['new_memory_strength'] < 2 for v in used_events),
                       fractional_detector_events=sum(0 < v['severity_weight'] < 1 for v in events),
                       fractional_strength_events=sum(0 < v['new_memory_strength'] < 2 for v in events),
                       first_S_application_push=first + 1 if first >= 0 else None,
                       positive_penalty_evaluations=sum(v['positive_candidates'] for v in costs.values()),
                       maximum_added_cost=max([v['maximum_added_cost'] for v in costs.values()] or [0.]),
                       elapsed_s=frozen['elapsed_s'])
            rows.append(row)
            data[arm, i] = receipt, frozen, m, r
    anchor = data['native5', 0][0]
    for (arm, i), (receipt, frozen, m, r) in data.items():
        baseline = data['native5', i]
        for key in ('checkpoint_sha256', 'encoder_sha256', 'implementation_sha256', 'action_mean', 'action_std'):
            assert receipt[key] == anchor[key], key
        assert receipt['full_target_sha256'] == baseline[0]['full_target_sha256']
        assert frozen['parameters_and_buffers_before'] == baseline[1]['parameters_and_buffers_before']
        for key in ('state_0', 'state_g'):
            np.testing.assert_array_equal(r[key], baseline[3][key])
    summary = dict(status='complete', samples_per_condition=20, maximum_pushes=20,
                   success_rule='CD < 0.3', target_chunks=[1, 2], conditions={})
    for arm in ARMS:
        selected = [r for r in rows if r['arm'] == arm]
        summary['conditions'][arm] = dict(mean_cd=float(np.mean([r['final_cd'] for r in selected])),
            std_cd_across_targets=float(np.std([r['final_cd'] for r in selected], ddof=1)),
            successes=sum(r['success'] for r in selected), total_pushes=sum(r['actual_pushes'] for r in selected),
            optimizer_calls=sum(r['optimizer_calls'] for r in selected),
            S_trigger_events=sum(r['S_trigger_events'] for r in selected),
            triggered_samples=sum(r['S_trigger_events'] > 0 for r in selected),
            used_trigger_memories=sum(r['used_trigger_memories'] for r in selected),
            used_fractional_strength_memories=sum(r['used_fractional_strength_memories'] for r in selected),
            applied_samples=sum(r['first_S_application_push'] is not None for r in selected),
            fractional_strength_events=sum(r['fractional_strength_events'] for r in selected),
            positive_penalty_evaluations=sum(r['positive_penalty_evaluations'] for r in selected),
            elapsed_s_sum=sum(r['elapsed_s'] for r in selected))
    pairs = []
    for left, right in (('native5', 'fixed3'), ('fixed3', 'dynamic3'), ('fixed3', 'fixed3_nowarm')):
        for i in range(20):
            l, r = data[left, i], data[right, i]
            a, b = l[3]['actions'][0], r[3]['actions'][0]
            n = min(len(a), len(b))
            std = np.asarray(l[0]['action_std'], dtype=np.float32)
            mean = np.asarray(l[0]['action_mean'], dtype=np.float32)
            delta = (b[:n] * std + mean) - (a[:n] * std + mean)
            changed = np.flatnonzero(np.any(delta != 0, axis=1))
            cd_changed = np.flatnonzero(l[2]['cd_by_replan'][0, :n] != r[2]['cd_by_replan'][0, :n])
            pairs.append(dict(left=left, right=right, sample=i, common_pushes=n,
                changed_executed_pushes=len(changed),
                complete_controls_identical=a.shape == b.shape and np.array_equal(a, b),
                first_control_difference_push=int(changed[0]) + 1 if len(changed) else None,
                first_cd_difference_push=int(cd_changed[0]) + 1 if len(cd_changed) else None,
                maximum_physical_control_L2=float(np.linalg.norm(delta, axis=1).max()) if n else 0.))
    summary['fixed_dynamic_strength_contrast_exercised'] = any(
        summary['conditions'][a]['used_fractional_strength_memories'] > 0 for a in ('fixed3', 'dynamic3'))
    summary['paired_vs_native5'] = {}
    for arm in ARMS[1:]:
        native_cd = np.asarray([data['native5', i][2]['final_cd'][0] for i in range(20)])
        other_cd = np.asarray([data[arm, i][2]['final_cd'][0] for i in range(20)])
        summary['paired_vs_native5'][arm] = dict(
            lower_cd=int((other_cd < native_cd).sum()), higher_cd=int((other_cd > native_cd).sum()),
            equal_cd=int((other_cd == native_cd).sum()),
            gained_successes=int(((other_cd < .3) & (native_cd >= .3)).sum()),
            lost_successes=int(((other_cd >= .3) & (native_cd < .3)).sum()))
    summary['action_comparison_limit'] = ('All arms execute independent physics. Native uses 5/5, S uses 3/3; '
        'there is no matched Native-3 control. Action differences from Native-5 cannot isolate S effects.')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    with (output / 'per_sample.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    with (output / 'action_comparisons.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(pairs[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(pairs)
    lines = ['# Rope：四组正式 20 目标实验', '',
        '每组按原顺序评估 chunk 1、2 的 20 个目标，最多 20 推，每推观测，CD < 0.3 即停。该目标集未参加此前 chunk 0 的 10 样本试跑。', '',
        '| 条件 | 平均 CD | 成功数 | 实际总推数 | S 触发次数 | 触发样本 | 非饱和强度次数 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for arm, r in summary['conditions'].items():
        lines.append(f"| {arm} | {r['mean_cd']:.6f} | {r['successes']}/20 | {r['total_pushes']} | {r['S_trigger_events']} | {r['triggered_samples']} | {r['fractional_strength_events']} |")
    lines += ['', '有预热的 S 为 5 推，最早影响第 7 推；无预热仍使用三推检测窗口，最早影响第 4 推。固定强度是每条记忆系数 2；动态强度为 2×w。', '',
        '四组独立执行物理仿真，不复用此前 Native 轨迹。Native-5 与 S-3 的对比同时改变动作块和 S，不能作为 S 的单因素消融。固定/动态及预热对照还须结合实际强度和仿真波动解读。', '']
    lines += ['| 实际动作比较 | 动作序列不同的目标数 | 共有执行推数中动作不同 |',
              '|---|---:|---:|']
    for left, right in (('native5', 'fixed3'), ('fixed3', 'dynamic3'), ('fixed3', 'fixed3_nowarm')):
        compared = [r for r in pairs if r['left'] == left and r['right'] == right]
        lines.append(f"| {left} → {right} | {sum(not r['complete_controls_identical'] for r in compared)}/20 | {sum(r['changed_executed_pushes'] for r in compared)}/{sum(r['common_pushes'] for r in compared)} |")
    lines += ['', '上述差异来自实际执行控制量；共有部分的差异按反归一化后的物理控制量比较。它们同时受规划节奏、S 和独立仿真分叉影响，不作为 S 的单独因果效果。', '']
    lines += ['[真实仿真截图与每三推序列](visuals/README.md)采用另外两对可视化复跑，未计入本表；图中标注各自复跑的 CD。原正式实验未保存 S 组逐推画面。', '',
        '全部目标散点与成功曲线：![正式结果](results.png)', '',
        '固定/动态非饱和强度对照是否实际发生：' + str(summary['fixed_dynamic_strength_contrast_exercised']) + '。若为 False，已触发记忆的动态权重全部饱和为 1，不能把两组成绩差异解释为强度规则的收益。', '',
        'S 触发计数表示检测器建立记忆的次数，不等于实际使用 S 的规划次数，也不等于成功脱困次数。实际施加 S 的样本、正惩罚候选数见 summary.json；逐样本见 per_sample.csv。', '',
        '原始指标、动作、配置、模型核验和日志见 evidence.tar.gz，逐文件哈希见 evidence_manifest.json。', '']
    (output / 'RESULTS.md').write_text('\n'.join(lines))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(report(args.root, args.output), indent=2))
