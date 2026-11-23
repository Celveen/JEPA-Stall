"""Build the Frozen paired-cost publication package from completed measurements."""
import hashlib
import json
import shutil
import tarfile
from pathlib import Path

BASE = Path('/home/appuser/workspace/project/multimodal-test')
SOURCE = BASE / 'JEPA-Stall-cloud-20261116/experiments/frozen_ecp_cost_20261117'
DEST = BASE / 'JEPA-Stall-publish-seeds-20261117'
OUT = DEST / 'reports/frozen_ecp_cost_20261117'
HERE = Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    assert json.loads((SOURCE / 'status.json').read_text())['state'] == 'complete'
    OUT.mkdir(exist_ok=False)
    for name in ('summary.json', 'per_task_cost.csv', 'RESULTS.md', 'manifest.json', 'status.json', 'analyze.py'):
        shutil.copy2(SOURCE / name, OUT / name)
    shutil.copy2(SOURCE / 'README.md', OUT / 'PROTOCOL.md')
    for name in ('verify_results.py', 'package_results.py', 'publication_README.md'):
        shutil.copy2(HERE / name, OUT / ('README.md' if name == 'publication_README.md' else name))
    audit = dict(verified_pairs=35, verified_jobs=70, task_evaluations=700, reference_matched_jobs=39)
    # The verifier independently checks these expected counts before publication.
    (OUT / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    with tarfile.open(OUT / 'raw_results.tar.gz', 'w:gz') as tar:
        for sub in ('runs', 'smoke', 'logs'):
            for p in sorted((SOURCE / sub).rglob('*')):
                if p.is_file() and p.suffix in ('.json', '.jsonl', '.npz', '.yaml', '.pkl', '.log'):
                    tar.add(p, arcname=str(p.relative_to(SOURCE)), recursive=False)
        tar.add(SOURCE / 'queue.log', arcname='queue.log')
    with tarfile.open(OUT / 'runtime_sources.tar.gz', 'w:gz') as tar:
        for sub in ('source', 'configs'):
            for p in sorted((SOURCE / sub).rglob('*')):
                if p.is_file() and p.suffix in ('.py', '.yaml', '.yml', '.txt', '.md', '.sh'):
                    tar.add(p, arcname=str(p.relative_to(SOURCE)), recursive=False)
        for name in ('instrument.py', 'run_job.py', 'queue_run.py', 'analyze.py', 'manifest.json', 'status.json', 'README.md'):
            tar.add(SOURCE / name, arcname=name)
        tar.add(HERE / 'setup.py', arcname='setup.py')
        for p in sorted((SOURCE.parent / 'figure1_three_eval_seeds_20261117/inputs/resnet').glob('*.pkl')):
            tar.add(p, arcname='inputs/resnet/' + p.name)
    for filename, link in [('README.md', 'reports/frozen_ecp_cost_20261117/README.md'),
                           ('reports/publication_v14/README.md', '../frozen_ecp_cost_20261117/README.md')]:
        p = DEST / filename
        original = p.read_text()
        entry = ('- [Frozen ECP paired cost analysis](' + link + '): 350 matched tasks per arm with GD100 + CUDA Graph. '
                 'ECP increases success from **61.14% to 72.29%**, adds **2.81%** per-replan latency, and reduces '
                 'mean task-loop wall time by **12.06%**; peak allocated GPU memory rises from **0.823 to 0.967 GiB**. '
                 'Includes all 700 task measurements, raw actions/replays, configurations, source snapshots and a CPU-only verifier.\n\n')
        if filename == 'README.md':
            marker = '## Latest results (2026-11-17)\n\n'
            assert marker in original
            updated = original.replace(marker, marker + entry, 1)
        else:
            first, rest = original.split('\n', 1)
            updated = first + '\n\n' + entry + rest.lstrip('\n')
        p.write_text(updated)
    checks = {str(p.relative_to(OUT)): sha(p) for p in sorted(OUT.iterdir()) if p.is_file()}
    (OUT / 'checksums.json').write_text(json.dumps(checks, indent=2) + '\n')
    print(json.dumps({'output': str(OUT), 'bytes': {p.name: p.stat().st_size for p in OUT.glob('*.tar.gz')}}))


if __name__ == '__main__':
    main()
