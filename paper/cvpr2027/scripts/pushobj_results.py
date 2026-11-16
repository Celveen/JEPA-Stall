"""Read per-shape PushObj outcomes from the archived evaluation records."""

from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import tarfile

import numpy as np


SHAPES = ('I', 'L', 'square', 'Z', 'small_tee', 'plus', 'T')
TASKS_PER_SHAPE = 50
REPLANS = 30


def shape(chunk):
    return chunk.split('_', 2)[2]


def baseline_rows(repo, provenance):
    manifest = json.loads((provenance / 'pushobj_baseline_data_v11.json').read_text())
    rows = {}
    for row in manifest['rows']:
        counts = {}
        for source in row['files']:
            path = Path(source['path'])
            raw = (repo / path).read_bytes()
            assert hashlib.sha256(raw).hexdigest() == source['sha256'], path
            with np.load(io.BytesIO(raw)) as record:
                successes = record['successes']
                assert successes.shape == (TASKS_PER_SHAPE,) and np.isin(successes, [0, 1]).all(), path
                counts[shape(path.parent.name)] = int(successes.sum())
        assert set(counts) == set(SHAPES) and sum(counts.values()) == row['successes'], row['method']
        rows[row['method'], row['planner']] = {
            'task_seed': 2000, 'successes_by_shape': counts,
            'source': 'provenance/pushobj_baseline_data_v11.json',
        }
    return rows


def paired_rows(repo, provenance):
    evidence = json.loads((provenance / 'evidence.json').read_text())['pushobj_main30']
    audited = {(row['model'], row['base']): row for row in evidence}
    rows = {}
    for base in ('frozen', 'adajepa'):
        archive_path = Path(f'reports/publication_v14/archives/main_pushobj_{base}.tar.gz')
        released = f'stag_{base}_official_it40/ref'
        augmented = f'stag_{base}_bias16k_it40/ref'
        intervention = f'stag_{base}_bias16k_it40/T0SP'
        episodes = {run: {} for run in (released, augmented, intervention)}
        with tarfile.open(repo / archive_path) as archive:
            manifest = json.load(archive.extractfile('ARCHIVE_MANIFEST.json'))
            hashes = {entry['path']: entry['sha256'] for entry in manifest['files']}
            for member in archive:
                parts = member.name.split('/')
                run = '/'.join(parts[1:3])
                if run not in episodes or parts[-1] != 'per_iter_logs.json':
                    continue
                raw = archive.extractfile(member).read()
                assert hashlib.sha256(raw).hexdigest() == hashes[member.name], member.name
                for line in raw.splitlines():
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    task = parts[-2], record['sample_idx']
                    succeeded = record['success'] and record['step'] <= REPLANS
                    episodes[run][task] = episodes[run].get(task, False) or succeeded

        native, augmented_native, rerun = (episodes[run] for run in (released, augmented, intervention))
        assert native.keys() == augmented_native.keys() and rerun.keys() <= native.keys(), base
        corrected = {**augmented_native, **rerun}
        for label, outcomes, expected, runs in (
            (f'released_{base}', native, audited['official', base]['baseline_success'], [released]),
            (f'ours_{base}', corrected, audited['bias16k', base]['arms']['T0SP']['success'],
             [augmented, intervention]),
        ):
            totals = Counter(shape(chunk) for chunk, _ in outcomes)
            assert totals == dict.fromkeys(SHAPES, TASKS_PER_SHAPE), (label, totals)
            successes = Counter(shape(chunk) for (chunk, idx), ok in outcomes.items() if ok)
            assert sum(successes.values()) == expected, label
            rows[label, 'GD'] = {
                'task_seed': 100,
                'successes_by_shape': {s: successes[s] for s in SHAPES},
                'source': archive_path.as_posix(),
                'runs': runs,
            }
    return rows


def followup_rows(repo):
    """Read the completed threshold and LeWM contact-objective comparisons."""
    conditions = [
        ('ours_frozen', 'reports/ecp_threshold_v3/paired_tasks.csv',
         'frozen_aug_g045', 'ecp_success', 'native_success', 100),
        ('ours_adajepa', 'reports/ecp_threshold_v2/paired_tasks.csv',
         None, 'gamma045_success', 'native_success', 100),
        ('ours_lewm', 'reports/lewm_contact_schedule_20261113/'
         'lewm_ecp_045_contact_objective/paired_tasks.csv',
         None, 'objective', 'native', 2000),
    ]
    rows = {}
    for label, source, condition, outcome_key, native_key, seed in conditions:
        path = repo / source
        with path.open() as file:
            records = list(csv.DictReader(file))
        if condition:
            records = [r for r in records if r['condition'] == condition]
        assert len(records) == 350
        identifiers = [(r.get('chunk', r['shape']), r['sample_idx']) for r in records]
        assert len(set(identifiers)) == 350
        assert Counter(r['shape'] for r in records) == dict.fromkeys(SHAPES, 50)
        assert all(r[k] in ('0', '1') for r in records for k in (outcome_key, native_key))
        counts = {s: sum(int(r[outcome_key]) for r in records if r['shape'] == s) for s in SHAPES}
        native = sum(int(r[native_key]) for r in records)
        rescued = sum(r[native_key] == '0' and r[outcome_key] == '1' for r in records)
        regressed = sum(r[native_key] == '1' and r[outcome_key] == '0' for r in records)
        assert native + rescued - regressed == sum(counts.values())
        if label == 'ours_lewm':
            root = path.parent
            hashes = json.loads((root.parent / 'SHA256SUMS.json').read_text())
            for shape_name in SHAPES:
                raw = root / 'full' / shape_name / 'final_eval_results.npz'
                key = raw.relative_to(root.parent).as_posix()
                assert hashlib.sha256(raw.read_bytes()).hexdigest() == hashes[key]
                with np.load(raw) as result:
                    expected = [int(r[outcome_key]) for r in records if r['shape'] == shape_name]
                    np.testing.assert_array_equal(result['successes'], expected)
            summary = json.loads((root / 'results.json').read_text())['total']
            assert (native, sum(counts.values()), rescued, regressed) == (
                summary['native'], summary['objective'],
                summary['rescues_vs_native'], summary['regressions_vs_native'])
        rows[label, 'GD'] = {
            'task_seed': seed, 'successes_by_shape': counts,
            'native_successes': native, 'rescues': rescued, 'regressions': regressed,
            'source': source, 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'trigger_interval': [0.15, 0.45],
            'contact_objective_weight': 10 if label == 'ours_lewm' else None,
        }
    return rows
