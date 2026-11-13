#!/usr/bin/env python3
"""Build a checksummed archive of both completed threshold evaluations."""

import hashlib
import io
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path


R = Path('/home/appuser/workspace/project/multimodal-test/JEPA-Stall')
O = R / 'eval_outputs/ecp_threshold_v3'
B = R / 'reports/ecp_threshold_v3'
DEST = Path('/tmp/ecp_threshold_v3/published')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    summary = json.loads((O / 'summary.json').read_text())
    audit = json.loads((B / 'completed_audit.json').read_text())
    assert summary['status'] == 'complete' and audit['status'] == 'passed'
    assert audit['task_runs'] == 700 and audit['chunks'] == 70
    assert all(len(list((O / c).glob('chunk_*/complete.json'))) == 35 for c in
               ('adajepa_aug_g060', 'frozen_aug_g045'))
    files = [B / name for name in (
        'PROTOCOL.md', 'RESULTS.md', 'evaluate.py', 'queue_evaluation.py',
        'resume.py', 'audit.py', 'package.py', 'test_config.py', 'completed_audit.json', 'paired_tasks.csv',
        'pipeline_status.json', 'evaluation_manifest.json')]
    files += [p for p in O.rglob('*') if p.is_file() and not p.is_symlink()
              and p.suffix in ('.json', '.jsonl', '.yaml', '.npz', '.log', '.csv')]
    DEST.mkdir(parents=True, exist_ok=True)
    archive = DEST / 'threshold_v3_complete_raw.tar.gz'
    entries = []
    with tarfile.open(archive, 'w:gz', compresslevel=6) as stream:
        for path in sorted(set(files)):
            assert path.exists(), path
            data = path.read_bytes()
            name = str(path.relative_to(R))
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o644
            stream.addfile(info, io.BytesIO(data))
            entries.append(dict(path=name, bytes=len(data), sha256=sha(data)))
        data = json.dumps(dict(classification='Complete 700-task threshold evaluation',
                               files=entries), indent=2).encode()
        info = tarfile.TarInfo('ARCHIVE_MANIFEST.json')
        info.size = len(data)
        stream.addfile(info, io.BytesIO(data))
    with tarfile.open(archive) as stream:
        for entry in entries:
            assert sha(stream.extractfile(entry['path']).read()) == entry['sha256']
    assert archive.stat().st_size < 95 * 1024 ** 2
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                    archive=archive.name, bytes=archive.stat().st_size,
                    files=len(entries), sha256=sha(archive.read_bytes()),
                    scope='Complete AdaJEPA 0.60 and in-house Frozen 0.45 PushObj outputs, logs and audit; checkpoint excluded.')
    (DEST / 'archive_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest))


if __name__ == '__main__':
    main()
