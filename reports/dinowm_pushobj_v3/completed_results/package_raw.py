#!/usr/bin/env python3
import argparse,hashlib,io,json,tarfile
from pathlib import Path
from datetime import datetime,timezone
p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
roots=[a.repo/'eval_outputs/dinowm_pushobj_3epoch_v3_cem200x10_mpc6x25_bf16',a.repo/'reports/dinowm_pushobj_v3']
entries=[];archive=a.output/'complete_raw.tar.gz';now=datetime.now(timezone.utc).isoformat()
with tarfile.open(archive,'w:gz',compresslevel=6)as tf:
 for root in roots:
  for path in sorted(root.rglob('*')):
   if not path.is_file()or path.is_symlink()or any(part in ['__pycache__','completed_results']for part in path.relative_to(root).parts):continue
   if path.suffix not in ['.json','.jsonl','.yaml','.yml','.npz','.csv','.log','.py','.md','.sh']:continue
   data=path.read_bytes();rel=str(path.relative_to(a.repo));item=tarfile.TarInfo(rel);item.size=len(data);item.mode=0o644;tf.addfile(item,io.BytesIO(data));entries.append(dict(path=rel,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
 raw=json.dumps(dict(snapshot_utc=now,classification='Complete v3 benchmark: 140 chunks, 1400 task runs; integration and FP32 diagnostics separate',files=entries),indent=2).encode();item=tarfile.TarInfo('ARCHIVE_MANIFEST.json');item.size=len(raw);tf.addfile(item,io.BytesIO(raw))
with tarfile.open(archive,'r:gz')as tf:
 for e in entries:assert hashlib.sha256(tf.extractfile(e['path']).read()).hexdigest()==e['sha256']
assert archive.stat().st_size<95*1024**2
manifest=dict(snapshot_utc=now,file=archive.name,bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),files=len(entries),verification='Every archive member re-read and SHA-256 verified',scope='Completed numerical results, full actions/states, per-iteration and episode logs, configs, execution receipts, CEM timing, source scripts, and separate integration/precision diagnostics.',omitted='Checkpoint binaries, image corpora and redundant rendered observation pickle caches. Weights are identified by recorded SHA-256; simulation replay uses saved actions and states.')
(a.output/'archive_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest))
