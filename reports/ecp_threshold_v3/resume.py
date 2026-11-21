#!/usr/bin/env python3
"""Resume only missing chunks after the original queue host session ended."""

import hashlib
import json
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import queue_evaluation as q


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = json.loads((q.OUT / 'evaluation_manifest.json').read_text())
    assert manifest['condition_order'] == list(q.CONDITIONS)
    assert manifest['total_task_runs'] == 700
    assert manifest['workers_per_gpu'] == 2
    assert manifest['checkpoints']['adajepa_aug_g060'] == manifest['checkpoints']['frozen_aug_g045']
    assert manifest['source_sha256']['evaluate.py'] == sha(q.B / 'evaluate.py')
    q.TARGETS = sorted((q.R / 'data/targets/EVAL50').glob('chunk_*.pkl'))
    assert len(q.TARGETS) == 35
    counts = {c: 0 for c in q.CONDITIONS}
    missing = []
    for condition in q.CONDITIONS:
        for target in q.TARGETS:
            dest = q.OUT / condition / target.stem
            marker = dest / 'complete.json'
            if not marker.exists():
                assert not dest.exists(), f'Partial output requires manual review: {dest}'
                missing.append((condition, target))
                continue
            receipt = json.loads(marker.read_text())
            expected = [.15, .45] if condition.startswith('frozen') else [.20, .60]
            assert receipt['status'] == 'complete' and receipt['condition'] == condition
            assert receipt['n'] == 10 and receipt['thresholds'] == expected
            assert receipt['targets_sha256'] == sha(target)
            assert receipt['checkpoint_sha256'] == manifest['checkpoints'][condition]
            assert receipt['source_sha256']['evaluate.py'] == sha(q.B / 'evaluate.py')
            assert receipt['checkpoint_file_unchanged']
            if condition.startswith('frozen'):
                assert receipt['frozen_weights_unchanged'] and not receipt['online_adaptation']
            q.compare(condition, target.stem)
            counts[condition] += 1
    assert counts == {'adajepa_aug_g060': 35, 'frozen_aug_g045': 21}, counts
    assert len(missing) == 14
    q.STATE.update(status='evaluating', resumed_utc=q.utc(), completed_chunks=sum(counts.values()),
                   total_chunks=70, episodes_per_condition=350, total_task_runs=700,
                   by_condition={c: dict(completed_chunks=counts[c], total_chunks=35) for c in q.CONDITIONS},
                   preflight={c: 'passed' for c in q.CONDITIONS}, workers={}, failures=[], output=str(q.OUT))
    q.update()
    for item in missing:
        q.JOBS.put(item)
    gpus = manifest['gpus']
    with ThreadPoolExecutor(max_workers=2 * len(gpus)) as pool:
        futures = [pool.submit(q.worker, gpu, slot) for gpu in gpus for slot in range(2)]
        for future in futures:
            future.result()
    if q.STATE['failures']:
        raise RuntimeError(json.dumps(q.STATE['failures']))
    assert q.STATE['completed_chunks'] == 70
    q.summarize()
    q.update(status='complete', summary=str(q.OUT / 'summary.json'))


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        q.update(status='failed', error=traceback.format_exc())
        q.DONE.set()
        raise
