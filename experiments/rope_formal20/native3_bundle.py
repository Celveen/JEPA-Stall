"""Archive all twenty Native-3 executions, including their actual observation caches."""
import argparse
import difflib
import hashlib
import io
import json
from pathlib import Path
import tarfile
from bundle import FILES


def main(root, output):
    root,output=Path(root),Path(output)
    assert json.loads((root/'queue.json').read_text())['status']=='complete'
    members=[]
    for i in range(20):
        p=root/'native3'/f'sample_{i:02d}'
        assert (p/'complete.json').exists()
        for name in (*FILES,'native_execution_cache.npz'):
            path=p/name
            if path.exists():members.append(path)
    members.extend(root/name for name in ('queue.json','protocol_audit.json'))
    members.extend(sorted((root/'source_snapshot').iterdir()))
    output.mkdir(parents=True,exist_ok=True)
    archive=output/'native3_evidence.tar.gz';manifest=[]
    with tarfile.open(archive,'w:gz',compresslevel=6) as tar:
        for path in members:
            payload=path.read_bytes();name='supplement/'+str(path.relative_to(root))
            info=tarfile.TarInfo(name);info.mode=0o644;info.size=len(payload)
            tar.addfile(info,io.BytesIO(payload))
            manifest.append(dict(path=name,bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest()))
    with tarfile.open(archive,'r:gz') as tar:
        assert tar.getnames()==[r['path'] for r in manifest]
        for row in manifest:
            assert hashlib.sha256(tar.extractfile(row['path']).read()).hexdigest()==row['sha256']
    result=dict(archive=archive.name,archive_bytes=archive.stat().st_size,
        archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),formal_episodes=20,
        role='Native 3/3 supplement to original four-arm formal evaluation',
        recorded_frames='All actual per-push RGB observations and particle states included',members=manifest)
    (output/'evidence_manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    for name in ('protocol_audit.json',): (output/name).write_bytes((root/name).read_bytes())
    source = root/'source_snapshot'
    (output/'worker_change.diff').write_text(''.join(difflib.unified_diff(
        (source/'worker_before.py').read_text().splitlines(True),
        (source/'worker_native3.py').read_text().splitlines(True),
        fromfile='worker_before.py', tofile='worker_native3.py', n=0)))
    print(json.dumps({k:v for k,v in result.items() if k!='members'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('eval_outputs/rope_formal20_native3'))
    p.add_argument('--output',type=Path,default=Path('reports/rope_formal20_native3'))
    a=p.parse_args();main(a.root,a.output)
