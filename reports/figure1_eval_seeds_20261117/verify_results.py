"""Recompute published counts, sample SD, parity checks and hashes from the archives.

Usage: python reports/figure1_eval_seeds_20261117/verify_results.py
Requires NumPy; no model checkpoint or GPU required. Archives are read in place.
"""
import csv
import hashlib
import io
import json
import statistics
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def main():
    manifest = json.loads((ROOT / 'checksums.json').read_text())
    for name, expected in manifest.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    counts = defaultdict(lambda: [0, 0])
    tasks = {}
    checks = 0
    batches = 0
    with tarfile.open(ROOT / 'raw_results.tar.gz') as tar:
        for member in tar.getmembers():
            if member.name.startswith('runs/') and member.name.endswith('/complete.json'):
                r = json.load(tar.extractfile(member))
                assert r['physics_replay_verified'] and not r['smoke']
                base = str(Path(member.name).parent)
                with np.load(io.BytesIO(tar.extractfile(base + '/final_eval_results.npz').read())) as d:
                    successes = d['successes'].copy()
                    assert len(successes) == r['tasks'] and int(successes.sum()) == r['successes']
                    assert np.isfinite(d['actions']).all()
                with np.load(io.BytesIO(tar.extractfile(base + '/replayed_states.npz').read())) as d:
                    np.testing.assert_array_equal(successes, d['successes'])
                key = (r['family'], r['stage'], r['seed'])
                counts[key][0] += r['successes']; counts[key][1] += r['tasks']
                for i, success in enumerate(successes):
                    tasks[key + (Path(r['key']).name, i)] = int(success)
                batches += 1
            elif member.name.endswith('/graph_validation.jsonl'):
                for line in tar.extractfile(member):
                    row = json.loads(line)
                    assert row['exact'] and row['rng_equal']
                    checks += 1
    hist = {g['label'].lower(): g for g in json.loads((ROOT / 'historical_figure1.json').read_text())['groups']}
    results = json.loads((ROOT / 'summary.json').read_text())['results']
    seed_rows = {(r['family'], r['stage'], int(r['seed'])): r for r in csv.DictReader((ROOT/'per_seed.csv').open())}
    for r in results:
        original = hist[r['family']]['success_counts'][['base', 'aug', 'ecp'].index(r['stage'])]
        assert r['successes'][0] == original
        for seed, success in zip(r['seeds'], r['successes']):
            key = (r['family'], r['stage'], seed)
            published = seed_rows[key]
            assert int(published['successes']) == success and int(published['tasks']) == 350
            assert abs(float(published['success_percent']) - success / 3.5) < 1e-10
            if seed != r['seeds'][0]:
                assert counts[key] == [success, 350]
        rates = [x / 3.5 for x in r['successes']]
        assert abs(statistics.mean(rates) - r['mean_percent']) < 1e-10
        assert abs(statistics.stdev(rates) - r['sample_std_percent']) < 1e-10
    rows = list(csv.DictReader((ROOT / 'per_task_new_seeds.csv').open()))
    assert len(rows) == len(tasks) == 3500
    for r in rows:
        key = (r['family'], r['stage'], int(r['seed']), r['chunk'], int(r['sample_index']))
        assert tasks[key] == int(r['success'])
    assert batches == 294 and checks == 125
    print(f'PASS: {batches} batches, {len(tasks)} task results, {checks} exact action/RNG comparisons, five three-seed summaries.')
    print('Physical replay success labels and run-time verification receipts agree; this check does not rerun the simulator.')


if __name__ == '__main__':
    main()
