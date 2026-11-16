"""Archive every preselected additional visualization execution and source audit."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile

from bundle import FILES


def main(root, compatibility, output):
    queue = json.loads((root / 'queue.json').read_text())
    assert queue['status'] == 'complete'
    members = [(root / 'queue.json', 'qualitative_extra/queue.json')]
    for arm in queue['arms']:
        for sample in queue['samples']:
            episode = root / arm / f'sample_{sample:02d}'
            assert (episode / 'complete.json').exists()
            assert (episode / 'recorded_execution.npz').exists()
            for name in (*FILES, 'recorded_execution.npz'):
                path = episode / name
                if path.exists():
                    members.append((path, f'qualitative_extra/{arm}/{episode.name}/{name}'))
    for name in ('worker_before.py', 'worker_native3.py', 'worker.diff'):
        members.append((compatibility / name, f'source_snapshot/{name}'))
    output.mkdir(parents=True, exist_ok=True)
    archive = output / 'evidence.tar.gz'
    manifest = []
    with tarfile.open(archive, 'w:gz', compresslevel=6) as tar:
        for path, name in members:
            content = path.read_bytes()
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(content), 0o644
            tar.addfile(info, io.BytesIO(content))
            manifest.append(dict(path=name, bytes=len(content), sha256=hashlib.sha256(content).hexdigest()))
    with tarfile.open(archive, 'r:gz') as tar:
        assert tar.getnames() == [m['path'] for m in manifest]
        for member in manifest:
            assert hashlib.sha256(tar.extractfile(member['path']).read()).hexdigest() == member['sha256']
    receipt = dict(archive=archive.name, archive_bytes=archive.stat().st_size,
                   archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(), members=manifest,
                   qualitative_episodes=len(queue['arms']) * len(queue['samples']), excluded_from_formal20=True)
    (output / 'evidence_manifest.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({k: v for k, v in receipt.items() if k != 'members'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('eval_outputs/rope_qualitative_extra'))
    parser.add_argument('--worker-compatibility', type=Path, default=Path('eval_outputs/rope_formal20_native3/source_snapshot'))
    parser.add_argument('--output', type=Path, default=Path('reports/rope_formal20/visuals_extra'))
    args = parser.parse_args()
    main(args.root, args.worker_compatibility, args.output)
