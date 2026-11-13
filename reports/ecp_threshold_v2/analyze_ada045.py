#!/usr/bin/env python3
"""CPU-only paired audit for the augmented AdaJEPA threshold ablation."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recorded_source_hash(receipt, relative_path):
    hashes = receipt['source_sha256']
    if relative_path in hashes:
        return hashes[relative_path]
    return hashes[Path(relative_path).name]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.repo / 'eval_outputs/ecp_threshold_v2_ada045'
    current = root / 'adajepa_aug_g045'
    old_base = args.repo / 'eval_outputs/stag_adajepa_bias16k_it40'
    g030 = args.repo / 'eval_outputs/ecp_threshold_v1/adajepa_aug_g030'
    summary = json.loads((root / 'summary.json').read_text())
    assert summary['status'] == 'complete'
    chunks = sorted(current.glob('chunk_*'))
    assert len(chunks) == 35
    task_rows = []
    by_shape = Counter()
    counts = Counter()
    first = []
    for chunk in chunks:
        receipt = json.loads((chunk / 'complete.json').read_text())
        assert receipt['status'] == 'complete' and receipt['n'] == 10
        assert receipt['thresholds'] == [.15, .45] and receipt['max_environment_controls'] == 150
        assert receipt['checkpoint_file_unchanged']
        assert receipt['source_sha256']['ecp_v2_evaluate.py'] == sha(args.repo / 'reports/ecp_threshold_v2/ecp_v2_evaluate.py')
        assert recorded_source_hash(receipt, 'planning/stagnation.py') == sha(args.repo / 'planning/stagnation.py')
        old = old_base / 'ref' / chunk.name
        old15 = old_base / 'T0SP' / chunk.name
        old30 = g030 / chunk.name
        ecp_eps = {e['sample_idx']: e for e in rows(chunk / 'stag_episodes.jsonl')}
        assert set(ecp_eps) == set(range(10))
        native_logs = rows(old / 'per_iter_logs.json')
        old15_logs = rows(old15 / 'per_iter_logs.json')
        with np.load(chunk / 'final_eval_results.npz') as a, np.load(old / 'final_eval_results.npz') as n, np.load(old30 / 'final_eval_results.npz') as b:
            assert len(a['successes']) == 10 and int(a['successes'].sum()) == receipt['successes']
            assert a['actions'].shape[1] * 5 <= 150
            for field in ('state_0', 'state_g'):
                np.testing.assert_array_equal(a[field], n[field])
                np.testing.assert_array_equal(a[field], b[field])
            for sample in range(10):
                entry = ecp_eps[sample]
                native = any(e['success'] for e in native_logs if e['sample_idx'] == sample and e['step'] <= 30)
                old15_records = [e for e in old15_logs if e['sample_idx'] == sample and e['step'] <= 30]
                gamma15 = any(e['success'] for e in old15_records) if old15_records else native
                gamma30 = bool(b['successes'][sample])
                gamma45 = bool(a['successes'][sample])
                trigger = entry['first_trigger']
                prefix = min(a['actions'].shape[1], n['actions'].shape[1], 30,
                             trigger if trigger is not None else 30)
                np.testing.assert_array_equal(a['actions'][sample, :prefix], n['actions'][sample, :prefix])
                compared_steps = min(a['actions'].shape[1], n['actions'].shape[1], 30)
                counts['native'] += native
                counts['gamma15'] += gamma15
                counts['gamma30'] += gamma30
                counts['gamma45'] += gamma45
                counts['triggered'] += trigger is not None
                if trigger is not None:
                    first.append(trigger)
                shape = chunk.name.split('_', 2)[2]
                by_shape[(shape, 'n')] += 1
                by_shape[(shape, 'successes')] += gamma45
                task_rows.append(dict(chunk=chunk.name, shape=shape, sample_idx=sample,
                                      native_success=int(native), gamma015_success=int(gamma15),
                                      gamma030_success=int(gamma30), gamma045_success=int(gamma45),
                                      first_trigger=trigger,
                                      actions_changed_vs_native=int(not np.array_equal(a['actions'][sample, :compared_steps], n['actions'][sample, :compared_steps]))))
    assert len(task_rows) == 350
    assert counts['native'] == 262 and counts['gamma15'] == 269 and counts['gamma30'] == 271
    assert counts['gamma45'] == summary['results']['adajepa_aug_g045']['successes']
    assert counts['triggered'] == summary['results']['adajepa_aug_g045']['triggered']
    comparisons = {}
    for label, field in [('native', 'native_success'), ('gamma15', 'gamma015_success'),
                         ('gamma30', 'gamma030_success')]:
        rescues = [f"{r['chunk']}:{r['sample_idx']}" for r in task_rows if r['gamma045_success'] and not r[field]]
        regressions = [f"{r['chunk']}:{r['sample_idx']}" for r in task_rows if r[field] and not r['gamma045_success']]
        comparisons[label] = dict(rescues=rescues, regressions=regressions)
    assert comparisons['native']['rescues'] == summary['results']['adajepa_aug_g045']['paired']['rescues']
    assert comparisons['native']['regressions'] == summary['results']['adajepa_aug_g045']['paired']['regressions']
    args.output.mkdir(parents=True, exist_ok=True)
    audit = dict(status='passed', chunks=35, task_runs=350, checkpoints='augmented LeWM epoch 3',
                 counts=dict(counts), comparisons=comparisons,
                 median_first_trigger=float(np.median(first)) if first else None,
                 by_shape={shape: dict(n=by_shape[(shape, 'n')], successes=by_shape[(shape, 'successes')])
                           for shape, _ in by_shape if _ == 'n'},
                 output=str(root), scope='Exploratory threshold ablation on the same fixed task set')
    (args.output / 'completed_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    with (args.output / 'paired_tasks.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(task_rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(task_rows)
    print(json.dumps(dict(status='passed', counts=dict(counts),
                          comparisons={k: {name: len(value) for name, value in v.items()}
                                       for k, v in comparisons.items()})))


if __name__ == '__main__':
    main()
