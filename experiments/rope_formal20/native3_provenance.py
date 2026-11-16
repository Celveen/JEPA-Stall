"""Record and verify the one-option worker change against the original formal run."""
import difflib
import hashlib
import json
from pathlib import Path


def snapshot(root, reference, repo):
    root,reference,repo=Path(root),Path(reference),Path(repo)
    key='experiments/rope_formal20/worker.py'
    after=(repo/key).read_bytes()
    option=b"choices=('native5', 'native3', 'fixed3',"
    assert after.count(option)==1, 'Review worker changes before reusing the formal protocol'
    before=after.replace(option,b"choices=('native5', 'fixed3',",1)
    receipt=json.loads((reference/'dynamic3/sample_00/complete.json').read_text())
    assert hashlib.sha256(before).hexdigest()==receipt['implementation_sha256'][key]
    for path,expected in receipt['implementation_sha256'].items():
        if path!=key:
            assert hashlib.sha256((repo/path).read_bytes()).hexdigest()==expected,path
    folder=root/'source_snapshot';folder.mkdir(parents=True,exist_ok=True)
    diff=''.join(difflib.unified_diff(before.decode().splitlines(True),after.decode().splitlines(True),
        fromfile='worker_before.py',tofile='worker_native3.py')).encode()
    for name,payload in (('worker_before.py',before),('worker_native3.py',after),('worker.diff',diff)):
        path=folder/name
        if path.exists():assert path.read_bytes()==payload, 'Existing source snapshot differs'
        else:path.write_bytes(payload)
    audit=root/'protocol_audit.json'
    if not audit.exists():
        audit.write_text(json.dumps(dict(samples=20,global_targets=list(range(10,30)),
            reference_result_dir=str(reference),worker_before_sha256=hashlib.sha256(before).hexdigest(),
            worker_native3_sha256=hashlib.sha256(after).hexdigest(),
            only_worker_change='Add native3 to CLI choices; existing native prefix and chunk selection logic unchanged',
            planner_core_sources_unchanged=True,physical_execution='independent; no cached Native trajectory reuse'),indent=2)+'\n')
