#!/usr/bin/env python3
"""Independent paired audit for the two completed threshold conditions."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


CONDITIONS = ('adajepa_aug_g060', 'frozen_aug_g045')


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def native_path(repo, condition, chunk):
    name = 'stag_frozen_bias16k_it40' if condition.startswith('frozen') else 'stag_adajepa_bias16k_it40'
    return repo / 'eval_outputs' / name / 'ref' / chunk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    root = repo / 'eval_outputs/ecp_threshold_v3'
    summary = json.loads((root / 'summary.json').read_text())
    manifest = json.loads((root / 'evaluation_manifest.json').read_text())
    assert summary['status'] == 'complete'
    assert manifest['condition_order'] == list(CONDITIONS)
    assert manifest['total_task_runs'] == 700
    assert manifest['excluded_gpus'] == [2]
    assert manifest['checkpoints'][CONDITIONS[0]] == manifest['checkpoints'][CONDITIONS[1]]
    assert sha(repo / 'checkpoints/pushobj_bias_16k_e3_seed0/checkpoints/model_3.pth') == manifest['checkpoints'][CONDITIONS[0]]

    all_rows = []
    outcomes = {}
    for condition in CONDITIONS:
        counts = Counter()
        by_shape = Counter()
        firsts = []
        tasks = []
        chunks = sorted((root / condition).glob('chunk_*'))
        assert len(chunks) == 35
        for chunk in chunks:
            receipt = json.loads((chunk / 'complete.json').read_text())
            expected_thresholds = [.15, .45] if condition.startswith('frozen') else [.20, .60]
            assert receipt['status'] == 'complete' and receipt['n'] == 10
            assert receipt['condition'] == condition and receipt['thresholds'] == expected_thresholds
            assert receipt['max_mpc_replans'] == 30 and receipt['executed_controls_per_plan'] == 5
            assert receipt['max_environment_controls'] == 150 and receipt['checkpoint_file_unchanged']
            assert receipt['checkpoint_sha256'] == manifest['checkpoints'][condition]
            assert receipt['source_sha256']['evaluate.py'] == sha(repo / 'reports/ecp_threshold_v3/evaluate.py')
            assert receipt['source_sha256']['stagnation.py'] == sha(repo / 'planning/stagnation.py')
            if condition.startswith('frozen'):
                assert receipt['frozen_weights_unchanged'] and not receipt['online_adaptation']
            else:
                assert receipt['online_adaptation']

            old = native_path(repo, condition, chunk.name)
            episode = {r['sample_idx']: r for r in rows(chunk / 'stag_episodes.jsonl')}
            assert set(episode) == set(range(10))
            native_logs = rows(old / 'per_iter_logs.json')
            with np.load(chunk / 'final_eval_results.npz') as result, np.load(old / 'final_eval_results.npz') as native:
                assert len(result['successes']) == 10
                assert int(result['successes'].sum()) == receipt['successes']
                for field in ('state_0', 'state_g'):
                    np.testing.assert_array_equal(result[field], native[field])
                for idx in range(10):
                    old_success = any(r['sample_idx'] == idx and r['step'] <= 30 and r['success'] for r in native_logs)
                    success = bool(result['successes'][idx])
                    trigger = episode[idx]['first_trigger']
                    prefix = min(result['actions'].shape[1], native['actions'].shape[1], 30,
                                 trigger if trigger is not None else 30)
                    np.testing.assert_array_equal(result['actions'][idx, :prefix], native['actions'][idx, :prefix])
                    counts['native'] += old_success
                    counts['ecp'] += success
                    counts['triggered'] += trigger is not None
                    if trigger is not None:
                        firsts.append(trigger)
                    shape = chunk.name.split('_', 2)[2]
                    by_shape[(shape, 'n')] += 1
                    by_shape[(shape, 'successes')] += success
                    row = dict(condition=condition, chunk=chunk.name, shape=shape, sample_idx=idx,
                               native_success=int(old_success), ecp_success=int(success), first_trigger=trigger)
                    if condition == 'adajepa_aug_g060':
                        with np.load(repo / 'eval_outputs/ecp_threshold_v2_ada045/adajepa_aug_g045' / chunk.name / 'final_eval_results.npz') as previous:
                            np.testing.assert_array_equal(result['state_0'], previous['state_0'])
                            np.testing.assert_array_equal(result['state_g'], previous['state_g'])
                            row['ada045_success'] = int(previous['successes'][idx])
                    tasks.append(row)
                    all_rows.append(row)
        assert len(tasks) == 350
        s = summary['results'][condition]
        assert counts['ecp'] == s['successes'] and counts['triggered'] == s['triggered']
        assert counts['native'] == (214 if condition.startswith('frozen') else 262)
        rescue = [f"{r['chunk']}:{r['sample_idx']}" for r in tasks if r['ecp_success'] and not r['native_success']]
        regress = [f"{r['chunk']}:{r['sample_idx']}" for r in tasks if r['native_success'] and not r['ecp_success']]
        assert rescue == s['paired']['rescues'] and regress == s['paired']['regressions']
        outcome = dict(counts=dict(counts), rescues=rescue, regressions=regress,
                       median_first_trigger=float(np.median(firsts)) if firsts else None,
                       by_shape={shape: dict(n=by_shape[(shape, 'n')], successes=by_shape[(shape, 'successes')])
                                 for shape, key in by_shape if key == 'n'})
        if condition == 'adajepa_aug_g060':
            against = 'ada045_success'
            outcome['versus_ada045'] = dict(
                successes=sum(r[against] for r in tasks),
                rescues=[f"{r['chunk']}:{r['sample_idx']}" for r in tasks if r['ecp_success'] and not r[against]],
                regressions=[f"{r['chunk']}:{r['sample_idx']}" for r in tasks if r[against] and not r['ecp_success']])
            assert outcome['versus_ada045']['successes'] == 272
        outcomes[condition] = outcome

    args.output.mkdir(parents=True, exist_ok=True)
    audit = dict(status='passed', task_runs=700, chunks=70, outcomes=outcomes,
                 scope='Exploratory threshold comparison on the same fixed EVAL50 task set')
    (args.output / 'completed_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    fields = ['condition', 'chunk', 'shape', 'sample_idx', 'native_success', 'ecp_success',
              'ada045_success', 'first_trigger']
    with (args.output / 'paired_tasks.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(all_rows)
    print(json.dumps(dict(status='passed', outcomes={c: v['counts'] for c, v in outcomes.items()})))


if __name__ == '__main__':
    main()
