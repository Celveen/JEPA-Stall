"""Audit a complete four-arm seed repeat, compare seed 99, and bundle evidence."""
import argparse
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import tarfile

import numpy as np

REPO = Path(__file__).resolve().parents[2]
ARMS = ('native3', 'fixed3', 'dynamic3', 'fixed3_nowarm')
LABELS = ('Native 3/3', 'Fixed S 3/3', 'Dynamic S 3/3', 'Fixed S, no warmup')


def read(path):
    return json.loads(path.read_text())


def write_csv(path, rows):
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def audit_episode(path, arm, sample, seed):
    receipt = read(path / 'complete.json')
    frozen = read(path / 'frozen_model_audit.json')
    encoder = read(path / 'encoder_audit.json')
    chunk = read(path / 'chunk_audit.json')
    records = read(path / 'ecp_records.json')
    costs = read(path / 'ecp_costs.json')
    reuse = read(path / 'prefix_reuse.json')
    with np.load(path / 'rope_metrics.npz') as f:
        metrics = dict(f)
    with np.load(path / 'final_eval_results.npz') as f:
        final = dict(f)
    cfg = receipt['config']['planner']
    native = arm == 'native3'
    assert receipt['arm'] == arm and receipt['sample'] == sample
    assert receipt['global_sample'] == sample + 10 and receipt['base_seed'] == seed
    assert receipt['config']['seed'] == seed
    assert cfg['episode_seed'] == seed * (sample + 10) + 1
    assert cfg['planning_seed'] == seed + sample + 10
    assert cfg['chunk_size'] == cfg['sub_planner']['horizon'] == 3
    assert cfg['max_iter'] == 20 and cfg['success_cd'] == .3 and cfg['n_taken_actions'] == 1
    assert cfg['sub_planner']['num_samples'] == 300 and cfg['sub_planner']['topk'] == 30
    assert cfg['sub_planner']['opt_steps'] == 10 and cfg['reuse_actions'] and not cfg['resim_from_start']
    assert cfg['ecp']['kappa'] == (0 if native else 2)
    assert cfg['ecp']['warmup_pushes'] == (0 if arm == 'fixed3_nowarm' else 5)
    assert cfg['strength_mode'] == ('fixed' if arm in ('fixed3', 'fixed3_nowarm') else 'dynamic')
    assert cfg['ecp']['independent_execution'] and 'reference_dir' not in cfg['ecp']
    assert not any(r['reused'] for r in reuse)
    assert frozen['unchanged'] and frozen['all_parameters_frozen'] and frozen['all_modules_eval']
    assert encoder['matches_official_pretrained_weights'] and encoder['verified_tensors'] == 175
    pushes, cd = int(metrics['actual_pushes'][0]), float(metrics['final_cd'][0])
    assert np.isfinite(cd) and 0 <= pushes <= 20
    assert bool(metrics['successes'][0]) == (cd < .3) and (cd < .3 or pushes == 20)
    timeline = np.r_[metrics['initial_cd'][0], metrics['cd_by_replan'][0]]
    assert len(timeline) == pushes + 1 and final['actions'].shape[1] == pushes
    hits = np.flatnonzero(timeline < .3)
    assert not len(hits) or hits[0] == pushes
    assert chunk['optimizer_push_indices_zero_based'] == list(range(0, pushes, 3))
    assert chunk['environment_seed'] == cfg['episode_seed'] and chunk['planning_seed'] == cfg['planning_seed']
    with np.load(path / 'recorded_execution.npz') as recorded:
        assert recorded['obs_visual'].shape[1] == pushes + 1
        np.testing.assert_array_equal(recorded['states'][0, -1], metrics['final_states'][0])
        physical = final['actions'] * np.asarray(receipt['action_std'], dtype=np.float32) + np.asarray(receipt['action_mean'], dtype=np.float32)
        np.testing.assert_array_equal(recorded['physical_actions'], physical)
    first = int(metrics['first_intervention'][0])
    if native:
        assert first == -1 and not costs
    elif first >= 0:
        assert first % 3 == 0 and first >= (3 if arm == 'fixed3_nowarm' else 6)
    events = [] if native else [r for r in records if r.get('severity_weight', 0) > 0]
    used = [r for r in events if any(q >= r['completed_pushes'] for q in chunk['optimizer_push_indices_zero_based'])]
    row = dict(arm=arm, sample=sample, global_sample=sample + 10, base_seed=seed,
               initial_cd=float(timeline[0]), final_cd=cd, success=cd < .3, actual_pushes=pushes,
               optimizer_calls=chunk['optimizer_calls'], S_trigger_events=len(events),
               used_trigger_memories=len(used), first_S_application_push=first + 1 if first >= 0 else None,
               used_fractional_strength_memories=sum(0 < r['new_memory_strength'] < 2 for r in used),
               positive_penalty_evaluations=sum(v['positive_candidates'] for v in costs.values()),
               elapsed_s=frozen['elapsed_s'])
    return dict(receipt=receipt, frozen=frozen, final=final, metrics=metrics, timeline=timeline, row=row)


def report(root, output):
    root, output = Path(root), Path(output)
    queue = read(root / 'queue.json')
    assert queue['status'] == 'complete' and set(queue['arms']) == set(ARMS)
    seed = queue['base_seed']
    missing = [f'{arm}/{i:02d}' for arm in ARMS for i in range(20)
               if not (root / arm / f'sample_{i:02d}/complete.json').exists()]
    assert not missing, f'Wait for all 80 episodes: {missing}'
    data = {(arm, i): audit_episode(root / arm / f'sample_{i:02d}', arm, i, seed)
            for arm in ARMS for i in range(20)}
    original = read(REPO / 'reports/rope_formal20_native3/summary.json')
    original_roots = {'native3': REPO / 'eval_outputs/rope_formal20_native3'}
    original_roots.update({a: REPO / 'eval_outputs/rope_formal20' for a in ARMS[1:]})
    for (arm, i), d in data.items():
        ref = data['native3', i]
        for key in ('checkpoint_sha256', 'encoder_sha256', 'full_target_sha256', 'action_mean', 'action_std', 'implementation_sha256'):
            assert d['receipt'][key] == ref['receipt'][key], key
        assert d['frozen']['parameters_and_buffers_before'] == ref['frozen']['parameters_and_buffers_before']
        for key in ('state_0', 'state_g'):
            np.testing.assert_array_equal(d['final'][key], ref['final'][key])
        configs = [copy.deepcopy(x['receipt']['config']) for x in (d, ref)]
        for cfg in configs:
            cfg.pop('saved_folder')
            cfg.pop('goal_file_path')
            cfg['planner'].pop('strength_mode')
            for key in ('kappa', 'warmup_pushes'):
                cfg['planner']['ecp'].pop(key)
        assert configs[0] == configs[1], 'Paired settings differ beyond method switches'
        old_path = original_roots[arm] / arm / f'sample_{i:02d}'
        old = read(old_path / 'complete.json')
        assert old['config']['seed'] == 99
        for key in ('checkpoint_sha256', 'encoder_sha256', 'full_target_sha256', 'action_mean', 'action_std'):
            assert d['receipt'][key] == old[key], key
        for key, value in old['implementation_sha256'].items():
            if key != 'experiments/rope_formal20/worker.py':
                assert d['receipt']['implementation_sha256'][key] == value, key
        with np.load(old_path / 'final_eval_results.npz') as previous:
            for key in ('state_0', 'state_g'):
                np.testing.assert_array_equal(d['final'][key], previous[key])
    rows = [d['row'] for d in data.values()]
    summary = dict(status='complete', base_seed=seed, previous_seed=99, samples_per_condition=20,
                   formal_episodes=80, maximum_pushes=20, success_rule='CD < 0.3',
                   global_targets=list(range(10, 30)), conditions={}, paired_vs_native3={},
                   seed99_comparison={}, audit=dict(paired_targets_exact=True, unchanged_targets_vs_seed99=True,
                   same_frozen_model=True, official_encoder_verified=True, same_planner_core_as_seed99=True,
                   all_actual_frames_saved=True, independent_physics=True))
    for arm in ARMS:
        selected = [r for r in rows if r['arm'] == arm]
        s = dict(mean_cd=float(np.mean([r['final_cd'] for r in selected])),
                 std_cd_across_targets=float(np.std([r['final_cd'] for r in selected], ddof=1)),
                 successes=sum(r['success'] for r in selected),
                 total_pushes=sum(r['actual_pushes'] for r in selected),
                 S_trigger_events=sum(r['S_trigger_events'] for r in selected),
                 triggered_samples=sum(r['S_trigger_events'] > 0 for r in selected),
                 applied_samples=sum(r['first_S_application_push'] is not None for r in selected),
                 used_trigger_memories=sum(r['used_trigger_memories'] for r in selected),
                 used_fractional_strength_memories=sum(r['used_fractional_strength_memories'] for r in selected),
                 positive_penalty_evaluations=sum(r['positive_penalty_evaluations'] for r in selected),
                 elapsed_s_sum=sum(r['elapsed_s'] for r in selected))
        summary['conditions'][arm] = s
        old = original['conditions'][arm]
        summary['seed99_comparison'][arm] = dict(previous_mean_cd=old['mean_cd'], new_mean_cd=s['mean_cd'],
            mean_cd_delta=s['mean_cd'] - old['mean_cd'], previous_successes=old['successes'], new_successes=s['successes'])
    pairs = []
    baseline = np.array([data['native3', i]['row']['final_cd'] for i in range(20)])
    for arm in ARMS[1:]:
        cds = np.array([data[arm, i]['row']['final_cd'] for i in range(20)])
        summary['paired_vs_native3'][arm] = dict(mean_cd_delta=float(np.mean(cds - baseline)),
            lower_cd=int((cds < baseline).sum()), higher_cd=int((cds > baseline).sum()),
            gained_successes=int(((cds < .3) & (baseline >= .3)).sum()),
            lost_successes=int(((cds >= .3) & (baseline < .3)).sum()))
        for i in range(20):
            left, right = data['native3', i], data[arm, i]
            std, mean = [np.asarray(left['receipt'][key], dtype=np.float32) for key in ('action_std', 'action_mean')]
            a, b = [d['final']['actions'][0] * std + mean for d in (left, right)]
            n = min(len(a), len(b))
            norms = np.linalg.norm(b[:n] - a[:n], axis=1)
            changed = np.flatnonzero(norms > 1e-6)
            first = int(changed[0]) + 1 if len(changed) else None
            applied = right['row']['first_S_application_push']
            pairs.append(dict(left='native3', right=arm, sample=i, common_pushes=n,
                changed_executed_pushes=len(changed), first_control_difference_push=first,
                first_S_application_push=applied,
                control_difference_before_first_S=bool(first is not None and (applied is None or first < applied)),
                maximum_physical_control_L2=float(norms.max()) if n else 0.,
                cd_delta=right['row']['final_cd'] - left['row']['final_cd']))
    summary['fixed_dynamic_strength_contrast_exercised'] = summary['conditions']['dynamic3']['used_fractional_strength_memories'] > 0
    summary['action_difference_limit'] = 'Independent physics may diverge before S; action differences alone are not all attributable to the intervention.'
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / 'per_sample.csv', rows)
    write_csv(output / 'action_comparisons.csv', pairs)
    write_csv(output / 'cd_by_push.csv', [dict(arm=arm, sample=i, push=t, cd=float(cd))
              for (arm, i), d in data.items() for t, cd in enumerate(d['timeline'])])
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    lines = [f'# Rope 3/3：种子 {seed} 的 20 样本复测', '',
             '四组共用原编号 10–29 的 20 个目标；仅更换基础随机种子。最多 20 推，每推观测，CD <0.3 即停。', '',
             '| 条件 | 平均 CD↓ | 成功 | S 触发 | 实际施加 S 的样本 | 总推数 |',
             '|---|---:|---:|---:|---:|---:|']
    for arm, label in zip(ARMS, LABELS):
        s = summary['conditions'][arm]
        lines.append(f"| {label} | {s['mean_cd']:.4f} | {s['successes']}/20 | {s['S_trigger_events']} | {s['applied_samples']}/20 | {s['total_pushes']} |")
    lines += ['', '## 与上一轮种子 99 对照', '', '| 条件 | 99：平均 CD / 成功 | 新种子：平均 CD / 成功 |', '|---|---|---|']
    for arm, label in zip(ARMS, LABELS):
        s = summary['seed99_comparison'][arm]
        lines.append(f"| {label} | {s['previous_mean_cd']:.4f} / {s['previous_successes']}/20 | {s['new_mean_cd']:.4f} / {s['new_successes']}/20 |")
    lines += ['', '## 解释与证据', '',
              '- 固定 S 和动态 S 均额外预热 5 推；无预热组仍保留 3 推检测窗口。强度分别为 2 和 2×触发严重程度，观测／执行节奏一致。',
              '- 每组独立运行物理仿真，未复用 Native 缓存。相同动作也可能出现物理轨迹差异；动作对照表明确保留干预前的分叉，不能把所有动作变化归因于 S。',
              f"- 动态强度是否实际使用了介于 0 与 2 之间的值：{summary['fixed_dynamic_strength_contrast_exercised']}。若为 False，不能据本轮 CD 差异宣称固定／动态强度孰优。",
              '- 这是相同目标的新种子复测，不是新增 20 个未见目标。两轮结果分别保留，不替换旧结果。', '',
              '[逐样本结果](per_sample.csv) · [逐推 CD](cd_by_push.csv) · [实际动作对照](action_comparisons.csv) · [汇总与核验](summary.json)', '',
              '![两个种子的对照](results.png)', '',
              '[复现说明](../../experiments/rope_seed_repeat/README.md)', '']
    (output / 'README.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    x = np.arange(4)
    for ax, key, title in zip(axes, ['mean_cd', 'successes'], ['Mean final CD (lower is better)', 'Successes / 20']):
        ax.bar(x - .18, [original['conditions'][a][key] for a in ARMS], .36, label='Seed 99', color='#92a3b5')
        ax.bar(x + .18, [summary['conditions'][a][key] for a in ARMS], .36, label=f'Seed {seed}', color='#227d84')
        ax.set_xticks(x, LABELS, rotation=15, ha='right')
        ax.set_title(title)
        ax.legend()
    axes[1].set_ylim(0, 20)
    fig.tight_layout()
    fig.savefig(output / 'results.png', dpi=180)
    plt.close(fig)
    bundle(root, output)
    print(json.dumps(summary, indent=2))


def bundle(root, output):
    manifests = {}
    for arm in ARMS:
        archive = output / f'{arm}_evidence.tar.gz'
        members = []
        with tarfile.open(archive, 'w:gz', compresslevel=6) as tar:
            for sample in range(20):
                folder = root / arm / f'sample_{sample:02d}'
                for path in sorted(folder.iterdir()):
                    # The capture contains Native's cache data as well. Retain targets once.
                    if not path.is_file() or path.name == 'native_execution_cache.npz':
                        continue
                    if path.name == 'target.pkl' and arm != 'native3':
                        continue
                    if path.suffix not in ('.json', '.npz', '.pkl'):
                        continue
                    name = str(path.relative_to(root))
                    content = path.read_bytes()
                    info = tarfile.TarInfo(name)
                    info.size, info.mode = len(content), 0o644
                    tar.addfile(info, io.BytesIO(content))
                    members.append(dict(path=name, bytes=len(content), sha256=hashlib.sha256(content).hexdigest()))
        with tarfile.open(archive) as tar:
            for entry in members:
                assert hashlib.sha256(tar.extractfile(entry['path']).read()).hexdigest() == entry['sha256']
        manifests[arm] = dict(archive=archive.name, bytes=archive.stat().st_size,
                             sha256=hashlib.sha256(archive.read_bytes()).hexdigest(), members=members)
    with tarfile.open(output / 'execution_sources_and_logs.tar.gz', 'w:gz') as tar:
        for name in ('source_snapshot', 'logs', 'queue.json'):
            tar.add(root / name, arcname=name)
    (output / 'evidence_manifest.json').write_text(json.dumps(manifests, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report(args.root.resolve(), args.output.resolve())
