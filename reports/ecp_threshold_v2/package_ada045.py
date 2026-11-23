"""Package only completed AdaJEPA 0.45 artifacts for branch publication."""

import hashlib
import io
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path

R = Path('/home/appuser/workspace/project/multimodal-test/JEPA-Stall')
O = R / 'eval_outputs/ecp_threshold_v2_ada045'
B = R / 'reports/ecp_threshold_v2'
DEST = Path('/tmp/ecp_threshold_v2/published')
DEST.mkdir(parents=True, exist_ok=True)
summary = json.loads((O / 'summary.json').read_text())
audit = json.loads((B / 'completed_audit.json').read_text())
assert summary['status'] == 'complete' and audit['status'] == 'passed'
assert audit['task_runs'] == 350 and len(list((O / 'adajepa_aug_g045').glob('chunk_*/complete.json'))) == 35
selected = [B / name for name in (
    'PROTOCOL_ADA045.md', 'RESULTS_ADA045.md', 'queue_ada045.py',
    'ecp_v2_evaluate.py', 'analyze_ada045.py', 'test_config.py',
    'test_intra_interval.py', 'completed_audit.json', 'paired_tasks.csv',
    'pipeline_status_ada045.json', 'evaluation_manifest_ada045.json',
    'ada045_pipeline.log')]
selected += [p for p in O.rglob('*') if p.is_file() and not p.is_symlink()
             and p.suffix in ('.json', '.jsonl', '.yaml', '.npz', '.log', '.csv')]
entries = []
archive = DEST / 'ada045_complete_raw.tar.gz'
with tarfile.open(archive, 'w:gz', compresslevel=6) as stream:
    for path in sorted(set(selected)):
        assert path.exists(), path
        data = path.read_bytes()
        name = str(path.relative_to(R))
        info = tarfile.TarInfo(name)
        info.size = len(data)
        info.mode = 0o644
        stream.addfile(info, io.BytesIO(data))
        entries.append(dict(path=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
    data = json.dumps(dict(classification='Complete AdaJEPA 0.45 ablation: 35 chunks, 350 tasks',
                           files=entries), indent=2).encode()
    info = tarfile.TarInfo('ARCHIVE_MANIFEST.json')
    info.size = len(data)
    stream.addfile(info, io.BytesIO(data))
with tarfile.open(archive) as stream:
    for item in entries:
        assert hashlib.sha256(stream.extractfile(item['path']).read()).hexdigest() == item['sha256']
assert archive.stat().st_size < 95 * 1024 ** 2
manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), archive=archive.name,
                bytes=archive.stat().st_size, files=len(entries),
                sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                scope='Complete AdaJEPA 0.45 outputs, per-task states/actions, logs, configurations and audit; checkpoints and rendered videos excluded.')
(DEST / 'archive_manifest_ada045.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(manifest))
