"""Bundle compact primary evidence from all completed formal and qualitative executions."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile

FILES = ('complete.json', 'run_manifest.json', 'encoder_audit.json', 'frozen_model_audit.json',
         'chunk_audit.json', 'rope_metrics.npz', 'ecp_records.json', 'ecp_costs.json',
         'prefix_reuse.json', 'final_eval_results.npz', 'target.pkl', 'logs.json', 'capture_provenance.json')


def main(formal, qualitative, output):
    roots = [('formal', Path(formal), ('native5','fixed3','dynamic3','fixed3_nowarm'), list(range(20))),
             ('qualitative', Path(qualitative), ('native5','fixed3'), [3,7])]
    output = Path(output)
    members = []
    for role, root, arms, samples in roots:
        assert json.loads((root/'queue.json').read_text())['status'] == 'complete'
        for arm in arms:
            for i in samples:
                episode = root/arm/f'sample_{i:02d}'
                assert (episode/'complete.json').exists()
                for name in FILES:
                    path = episode/name
                    if path.exists():
                        members.append((path, f'{role}/{arm}/sample_{i:02d}/{name}'))
                if role=='qualitative':
                    members.append((episode/'recorded_execution.npz', f'{role}/{arm}/sample_{i:02d}/recorded_execution.npz'))
        for name in ('queue.json','protocol_audit.json'):
            path=root/name
            if path.exists(): members.append((path,f'{role}/{name}'))
    manifest=[]
    archive=output/'evidence.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=6) as tar:
        for path, name in members:
            content=path.read_bytes()
            info=tarfile.TarInfo(name);info.size=len(content); info.mode=0o644
            tar.addfile(info,io.BytesIO(content))
            manifest.append(dict(path=name,bytes=len(content),sha256=hashlib.sha256(content).hexdigest()))
    # Verify every archive member against the exported manifest before reporting success.
    with tarfile.open(archive,'r:gz') as tar:
        assert tar.getnames()==[m['path'] for m in manifest]
        for m in manifest:
            assert hashlib.sha256(tar.extractfile(m['path']).read()).hexdigest()==m['sha256']
    receipt=dict(archive=archive.name,archive_bytes=archive.stat().st_size,
                 archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),members=manifest,
                 formal_episodes=80,qualitative_episodes=4,
                 note='Formal statistics and independent qualitative reruns kept separate. Formal S RGB was not persisted. Qualitative caches contain recorded RGB, states and physical actions.')
    (output/'evidence_manifest.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k!='members'},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--formal',type=Path,default=Path('eval_outputs/rope_formal20'))
    p.add_argument('--qualitative',type=Path,default=Path('eval_outputs/rope_qualitative20'))
    p.add_argument('--output',type=Path,default=Path('reports/rope_formal20'))
    a=p.parse_args();main(a.formal,a.qualitative,a.output)
