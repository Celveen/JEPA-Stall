"""Verify packaged measurements and recompute all summaries. Requires NumPy, no GPU.

This checks recorded physical replay labels; it does not rerun the simulator.
"""
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def main():
    for name, expected in json.loads((ROOT / 'checksums.json').read_text()).items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    audit = dict(verified_pairs=0, verified_jobs=0, task_evaluations=0, reference_matched_jobs=0)
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        with tarfile.open(ROOT / 'raw_results.tar.gz') as tar:
            for member in tar.getmembers():
                assert member.isfile() and not Path(member.name).is_absolute()
                assert '..' not in Path(member.name).parts
                target = work / member.name
                target.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as src, target.open('wb') as dst:
                    shutil.copyfileobj(src, dst)
        for pair in manifest['pairs']:
            receipts = {}
            for arm in ('off', 'on'):
                path = work / 'runs' / pair['chunk'] / arm
                rec = json.loads((path / 'complete.json').read_text())
                receipts[arm] = rec
                assert rec['arm'] == arm and rec['chunk'] == pair['chunk']
                assert rec['physics_replay_verified'] and rec['model_unchanged'] and not rec['smoke']
                assert rec['checkpoint_sha256'] == manifest['checkpoint_sha256']
                eps = [json.loads(line) for line in (path / 'cost_episodes.jsonl').read_text().splitlines()]
                assert len(eps) == rec['tasks'] == pair['n'] == 10
                with np.load(path / 'final_eval_results.npz') as final, np.load(path / 'replayed_states.npz') as replay:
                    np.testing.assert_array_equal(final['successes'], replay['successes'])
                    np.testing.assert_array_equal(final['successes'], [e['success'] for e in eps])
                    assert int(final['successes'].sum()) == rec['successes']
                    assert np.isfinite(final['actions']).all()
                audit['verified_jobs'] += 1
                audit['task_evaluations'] += rec['tasks']
                if rec['reference_actions_exact'] is not None:
                    assert rec['reference_actions_exact'] is True
                    audit['reference_matched_jobs'] += 1
            assert receipts['off']['gpu'] == receipts['on']['gpu']
            assert receipts['off']['model_parameters'] == receipts['on']['model_parameters']
            audit['verified_pairs'] += 1
        assert audit == json.loads((ROOT / 'audit.json').read_text())
        assert audit['verified_jobs'] == 70 and audit['task_evaluations'] == 700
        for name in ('analyze.py', 'manifest.json'):
            shutil.copy2(ROOT / name, work / name)
        subprocess.run([sys.executable, str(work / 'analyze.py')], check=True)
        assert json.loads((work / 'summary.json').read_text()) == json.loads((ROOT / 'summary.json').read_text())
        assert (work / 'per_task_cost.csv').read_bytes() == (ROOT / 'per_task_cost.csv').read_bytes()
        with tarfile.open(ROOT / 'runtime_sources.tar.gz') as tar:
            for name, expected in manifest['source_sha256'].items():
                assert hashlib.sha256(tar.extractfile(name).read()).hexdigest() == expected, name
    print('PASS: 35 paired chunks, 70 jobs, 700 task evaluations; summaries, per-task costs, replay labels, checkpoint receipts and source hashes agree.')
    print('Recorded reference actions matched in 39 jobs. Simulator and historical reference comparisons are not rerun by this verifier.')


if __name__ == '__main__':
    main()
