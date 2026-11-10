"""Read per-shape PushObj outcomes from the archived evaluation records."""

from collections import Counter
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
