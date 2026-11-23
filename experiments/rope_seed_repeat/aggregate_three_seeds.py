"""Recompute descriptive Rope statistics from paired seeds 99, 100, 101."""
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / 'reports/rope_seeds99_100_101'
SEEDS = (99, 100, 101)
ARMS = ('native3', 'fixed3', 'dynamic3', 'fixed3_nowarm')
LABELS = ('Native 3/3', 'Fixed S, warmup 5', 'Dynamic S, warmup 5', 'Fixed S, no warmup')


def read(path):
    return json.loads(path.read_text())


def folder(seed, arm, sample):
    name = ('rope_formal20_native3' if arm == 'native3' else 'rope_formal20') if seed == 99 else f'rope_seed{seed}_20'
    return REPO / 'eval_outputs' / name / arm / f'sample_{sample:02d}'


def write_csv(name, rows):
    with (OUT / name).open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(exist_ok=True, parents=True)
    data, targets, source_hashes = {}, {}, {}
    summaries = {}
    model_ref = None
    for seed in SEEDS:
        sp = REPO / 'reports' / ('rope_formal20_native3' if seed == 99 else f'rope_seed{seed}_20') / 'summary.json'
        summaries[seed] = read(sp)
        source_hashes[str(sp.relative_to(REPO))] = hashlib.sha256(sp.read_bytes()).hexdigest()
        assert summaries[seed]['status'] == 'complete'
        for arm in ARMS:
            for i in range(20):
                p = folder(seed, arm, i)
                receipt = read(p / 'complete.json')
                cfg = receipt['config']['planner']
                assert receipt['config']['seed'] == seed
                assert cfg['max_iter'] == 20 and cfg['success_cd'] == .3
                assert cfg['chunk_size'] == cfg['sub_planner']['horizon'] == 3
                assert cfg['ecp']['kappa'] == (0 if arm == 'native3' else 2)
                assert cfg['ecp']['warmup_pushes'] == (0 if arm == 'fixed3_nowarm' else 5)
                signature = [receipt[k] for k in ('checkpoint_sha256', 'encoder_sha256')]
                if model_ref is None:
                    model_ref = signature
                assert signature == model_ref
                if (i, 'target_hash') in targets:
                    assert receipt['full_target_sha256'] == targets[i, 'target_hash']
                targets[i, 'target_hash'] = receipt['full_target_sha256']
                with np.load(p / 'final_eval_results.npz') as f:
                    for key in ('state_0', 'state_g'):
                        if (i, key) in targets:
                            np.testing.assert_array_equal(f[key], targets[i, key])
                        targets[i, key] = f[key]
                    actions = f['actions'][0] * np.asarray(receipt['action_std'], np.float32) + np.asarray(receipt['action_mean'], np.float32)
                with np.load(p / 'rope_metrics.npz') as f:
                    cd, pushes = float(f['final_cd'][0]), int(f['actual_pushes'][0])
                    first_s = int(f['first_intervention'][0])
                    timeline = np.r_[f['initial_cd'][0], f['cd_by_replan'][0]]
                    assert bool(f['successes'][0]) == (cd < .3)
                assert len(actions) == pushes == len(timeline) - 1
                hits = np.flatnonzero(timeline < .3)
                assert not len(hits) or hits[0] == pushes
                assert np.isfinite(cd) and (cd < .3 or pushes == 20)
                events = [] if arm == 'native3' else [r for r in read(p / 'ecp_records.json') if r.get('severity_weight', 0) > 0]
                used = [r for r in events if any(q >= r['completed_pushes'] for q in range(0, pushes, 3))]
                row = dict(seed=seed, arm=arm, sample=i, global_sample=i + 10,
                           final_cd=cd, success=cd < .3, pushes=pushes,
                           S_trigger_events=len(events), S_applied=first_s >= 0,
                           used_trigger_memories=len(used),
                           used_fractional_strength_memories=sum(0 < r['new_memory_strength'] < 2 for r in used))
                data[seed, arm, i] = dict(row=row, actions=actions, first_s=first_s)
    rows = [v['row'] for v in data.values()]
    by_seed, pooled, pairs, action_rows = [], [], [], []
    for arm in ARMS:
        for seed in SEEDS:
            selected = [r for r in rows if r['seed'] == seed and r['arm'] == arm]
            old = summaries[seed]['conditions'][arm]
            cd = np.array([r['final_cd'] for r in selected])
            row = dict(seed=seed, arm=arm, mean_cd=float(cd.mean()),
                       successes=sum(r['success'] for r in selected),
                       S_trigger_events=sum(r['S_trigger_events'] for r in selected),
                       applied_samples=sum(r['S_applied'] for r in selected),
                       total_pushes=sum(r['pushes'] for r in selected))
            np.testing.assert_allclose(row['mean_cd'], old['mean_cd'], rtol=0, atol=1e-12)
            for k in ('successes', 'S_trigger_events', 'applied_samples', 'total_pushes'):
                assert row[k] == old[k], (seed, arm, k)
            by_seed.append(row)
        selected = [r for r in rows if r['arm'] == arm]
        cd = np.array([r['final_cd'] for r in selected])
        seed_rows = [r for r in by_seed if r['arm'] == arm]
        pooled.append(dict(arm=arm, runs=60, unique_targets=20, mean_cd=float(cd.mean()),
                           seed_mean_cd_sd=float(np.std([r['mean_cd'] for r in seed_rows], ddof=1)),
                           median_cd=float(np.median(cd)), p90_cd=float(np.quantile(cd, .9)),
                           failed_run_mean_cd=float(cd[cd >= .3].mean()),
                           successes=int((cd < .3).sum()), success_rate=float((cd < .3).mean()),
                           mean_pushes=float(np.mean([r['pushes'] for r in selected])),
                           S_trigger_events=sum(r['S_trigger_events'] for r in selected),
                           triggered_runs=sum(r['S_trigger_events'] > 0 for r in selected),
                           applied_runs=sum(r['S_applied'] for r in selected),
                           used_trigger_memories=sum(r['used_trigger_memories'] for r in selected),
                           used_fractional_strength_memories=sum(r['used_fractional_strength_memories'] for r in selected)))
        if arm == 'native3':
            continue
        for seed in SEEDS:
            for i in range(20):
                base, method = data[seed, 'native3', i], data[seed, arm, i]
                a, b = base['actions'], method['actions']
                n = min(len(a), len(b))
                norms = np.linalg.norm(b[:n] - a[:n], axis=1)
                changed = np.flatnonzero(norms > 1e-6)
                first_s = method['first_s']
                action_rows.append(dict(seed=seed, arm=arm, sample=i, common_pushes=n,
                    changed_pushes=int(len(changed)),
                    first_difference_push=int(changed[0] + 1) if len(changed) else None,
                    first_S_application_push=first_s + 1 if first_s >= 0 else None,
                    difference_before_or_without_S=bool(len(changed) and (first_s < 0 or changed[0] < first_s)),
                    changed_pushes_at_or_after_S=int(sum(changed >= first_s)) if first_s >= 0 else 0))
    # All pairwise comparisons preserve the same seed and target, including fixed vs dynamic.
    for left_idx, left in enumerate(ARMS):
        for right in ARMS[left_idx + 1:]:
            a = np.array([data[s, left, i]['row']['final_cd'] for s in SEEDS for i in range(20)])
            b = np.array([data[s, right, i]['row']['final_cd'] for s in SEEDS for i in range(20)])
            pairs.append(dict(left=left, right=right, mean_cd_delta_right_minus_left=float((b - a).mean()),
                              lower_cd=int((b < a).sum()), higher_cd=int((b > a).sum()), equal_cd=int((b == a).sum()),
                              gained_successes=int(((b < .3) & (a >= .3)).sum()),
                              lost_successes=int(((b >= .3) & (a < .3)).sum())))
    action_summary = []
    for arm in ARMS[1:]:
        selected = [r for r in action_rows if r['arm'] == arm]
        action_summary.append(dict(arm=arm, changed_runs=sum(r['changed_pushes'] > 0 for r in selected),
                                   **{k: sum(r[k] for r in selected) for k in ('changed_pushes', 'common_pushes', 'difference_before_or_without_S', 'changed_pushes_at_or_after_S')}))
    baseline = pooled[0]
    for row in pooled:
        row['cd_reduction_vs_native_pct'] = 100 * (baseline['mean_cd'] - row['mean_cd']) / baseline['mean_cd']
        row['push_reduction_vs_native_pct'] = 100 * (baseline['mean_pushes'] - row['mean_pushes']) / baseline['mean_pushes']
        row['success_gain_vs_native_pp'] = 100 * (row['success_rate'] - baseline['success_rate'])
    artifact = dict(seeds=SEEDS, unique_targets=20, runs_per_arm=60, total_runs=240,
                    maximum_pushes=20, success_rule='CD < 0.3', source_summary_sha256=source_hashes,
                    paired_targets_verified=True, original_aggregates_recomputed=True,
                    seed_sd_definition='Sample SD (ddof=1) of the three seed-level mean CDs, not a confidence interval.',
                    dependence='The same 20 targets were repeated under three seeds; these are not 60 distinct targets. Descriptive statistics only.',
                    action_definition='Denormalized 4D physical controls; L2 difference >1e-6 over the common actually executed prefix.',
                    action_limit='Independent physics can diverge before S; observed action changes are not a causal attribution to S.',
                    conditions=pooled, by_seed=by_seed, paired_comparisons=pairs, action_summaries=action_summary)
    (OUT / 'summary.json').write_text(json.dumps(artifact, indent=2) + '\n')
    for name, values in [('aggregate.csv', pooled), ('by_seed.csv', by_seed), ('per_run.csv', rows),
                         ('paired_comparisons.csv', pairs), ('action_comparisons.csv', action_rows),
                         ('action_summary.csv', action_summary)]:
        write_csv(name, values)
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/rope-three-seed-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    x = np.arange(4)
    for j, seed in enumerate(SEEDS):
        values = [next(r for r in by_seed if r['seed'] == seed and r['arm'] == a) for a in ARMS]
        for ax, key, scale in [(axes[0], 'mean_cd', 1), (axes[1], 'successes', 5), (axes[2], 'total_pushes', .05)]:
            ax.bar(x + (j - 1) * .24, [r[key] * scale for r in values], .24, label=f'Seed {seed}')
    for ax, title in zip(axes, ['Mean final CD (lower is better)', 'Success rate (%)', 'Mean executed pushes (lower is better)']):
        ax.set_title(title)
        ax.set_xticks(x, LABELS, rotation=12, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    axes[1].set_ylim(0, 100)
    axes[2].set_ylim(0, 20)
    fig.suptitle('Rope: same 20 targets, 3 seeds; 20-push cap, success CD < 0.3')
    fig.tight_layout()
    fig.savefig(OUT / 'results.png', dpi=200)
    fig.savefig(OUT / 'results.pdf')
    plt.close(fig)
    print(json.dumps(artifact, indent=2))


if __name__ == '__main__':
    main()
