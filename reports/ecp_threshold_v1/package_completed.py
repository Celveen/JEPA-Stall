from pathlib import Path
import io,json,tarfile,hashlib,shutil
from datetime import datetime,timezone
R=Path('/home/appuser/workspace/project/multimodal-test/JEPA-Stall');O=Path('/tmp/ecp_threshold_publish');entries=[];archive=O/'completed_raw.tar.gz'
roots=[R/'eval_outputs/ecp_threshold_v1',R/'reports/ecp_threshold_v1']
with tarfile.open(archive,'w:gz',compresslevel=6)as tf:
 for root in roots:
  for p in sorted(root.rglob('*')):
   if not p.is_file()or p.is_symlink()or '__pycache__'in p.parts or 'invalid_preflight'in p.parts:continue
   if p.suffix not in ['.json','.jsonl','.yaml','.npz','.csv','.log','.md','.py']:continue
   data=p.read_bytes();name=str(p.relative_to(R));info=tarfile.TarInfo(name);info.size=len(data);info.mode=0o644;tf.addfile(info,io.BytesIO(data));entries.append(dict(path=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
 data=json.dumps(dict(classification='Complete: 140 chunks, 1400 task runs; invalid startup checks excluded',files=entries),indent=2).encode();info=tarfile.TarInfo('ARCHIVE_MANIFEST.json');info.size=len(data);tf.addfile(info,io.BytesIO(data))
with tarfile.open(archive)as tf:
 for e in entries:assert hashlib.sha256(tf.extractfile(e['path']).read()).hexdigest()==e['sha256']
assert archive.stat().st_size<95*1024**2
m=dict(snapshot_utc=datetime.now(timezone.utc).isoformat(),archive=archive.name,bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),files=len(entries),scope='Complete outcomes, states/actions, logs, configurations, evaluation source, pairing checks and audit. Checkpoints and duplicate rendered-observation caches excluded.',invalid_preflight='Excluded from outcomes; source version and reason preserved separately in invalid_preflight_record.json.')
(O/'archive_manifest.json').write_text(json.dumps(m,indent=2)+'\n')
invalid=[]
for p in(R/'reports/ecp_threshold_v1/invalid_preflight').glob('*/reason.json'):invalid.append(dict(directory=p.parent.name,**json.loads(p.read_text())))
(O/'invalid_preflight_record.json').write_text(json.dumps(invalid,indent=2)+'\n');print(json.dumps(m))
