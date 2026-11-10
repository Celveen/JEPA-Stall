from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,tarfile,io,shutil
r=Path('/home/appuser/workspace/project/multimodal-test/JEPA-Stall');out=Path('/tmp/jepa_publication_v14');out.mkdir(exist_ok=True);archives=out/'archives';archives.mkdir(exist_ok=True)
now=datetime.now(timezone.utc).isoformat();all_entries=[]
# Snapshot completed and running numerical artifacts; derived images and duplicate frame caches
# are reconstructible from saved actions and are deliberately not part of the data archive.
def include(p,root):
    rel=p.relative_to(root)
    if any(x in ['__pycache__','paper_completed_v10','checkpoints','wandb'] for x in rel.parts):return False
    return p.is_file() and not p.is_symlink() and (p.suffix in ['.json','.jsonl','.yaml','.yml','.npz','.log','.csv','.md','.sh','.py'] or p.name.endswith('.json.gz'))
def pack(name,roots,classification):
    entries=[];archive=archives/(name+'.tar.gz')
    with tarfile.open(archive,'w:gz',compresslevel=6) as tf:
        for root in roots:
            if not root.exists():continue
            for p in sorted(root.rglob('*')):
                if not include(p,root):continue
                data=p.read_bytes();rel=str(p.relative_to(r))
                # A running logger may be between writes. Keep only complete JSONL records.
                if p.suffix=='.jsonl' and data and not data.endswith(b'\n'):data=data[:data.rfind(b'\n')+1]
                item=tarfile.TarInfo(rel);item.size=len(data);item.mode=0o644
                tf.addfile(item,io.BytesIO(data))
                entries.append({'path':rel,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
        manifest={'snapshot_utc':now,'classification':classification,'files':entries}
        data=(json.dumps(manifest,indent=2)+'\n').encode();item=tarfile.TarInfo('ARCHIVE_MANIFEST.json');item.size=len(data);item.mode=0o644;tf.addfile(item,io.BytesIO(data))
    h=hashlib.sha256(archive.read_bytes()).hexdigest()
    with tarfile.open(archive,'r:gz') as tf:
        for entry in entries:
            assert hashlib.sha256(tf.extractfile(entry['path']).read()).hexdigest()==entry['sha256']
    result={'file':str(archive.relative_to(out)),'classification':classification,'files':len(entries),'bytes':archive.stat().st_size,'sha256':h}
    assert result['bytes']<95*1024**2
    all_entries.append(result);print(json.dumps(result),flush=True)
pack('pusht_visual_complete',[r/'eval_outputs/pusht_visual_v8',r/'eval_outputs/pusht_visual_v8_smoke'],'complete: 56 primary jobs, 2800 task runs; smoke separate')
pack('maze_frozen_complete',[r/'eval_outputs/maze_frozen_v10'],'complete: both S and SP controls, 100 tasks each including verified inactive reuse')
pack('dinowm_pushobj_training_and_validation',[r/'reports/dinowm_pushobj_v1',r/'reports/dinowm_pushobj_v2',r/'reports/dinowm_pushobj_v3',r/'checkpoints/dinowm_pushobj_v1/ordinary',r/'checkpoints/dinowm_pushobj_v1/augmented'],'training complete at epoch 3; integration and precision diagnostics; logs of superseded protocols retained')
pack('dinowm_pushobj_v3_progress',[r/'eval_outputs/dinowm_pushobj_3epoch_v3_cem200x10_mpc6x25_bf16'],'in-progress snapshot, not final benchmark results')
pack('dinowm_superseded_evaluations',[r/'eval_outputs/dinowm_pushobj_3epoch_v1',r/'eval_outputs/dinowm_pushobj_3epoch_v2_cem200x10_bf16'],'superseded partial evaluations; not pooled with v3')
meta=out/'dataset_and_checkpoint_metadata';meta.mkdir(exist_ok=True)
for root in [r/'data/dinowm_pushobj_ordinary_matched16k_v1',r/'data/pushobj_bias_16k',r/'checkpoints/lewm_pushobj_bias16k_e10',r/'checkpoints/dinowm_pushobj_v1/ordinary',r/'checkpoints/dinowm_pushobj_v1/augmented']:
    for p in root.iterdir():
        if p.is_file() and p.suffix in ['.json','.jsonl','.yaml']:
            dst=meta/p.relative_to(r);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
manifest={'snapshot_utc':now,'archives':all_entries,'scope':'Numerical results, complete raw actions/states, configurations, run and training logs, validation/provenance, and dataset/checkpoint metadata. No model weights or image training corpora.','omitted_reconstructible_files':['observation/image pickle caches','videos and per-run rendered previews','automatically generated old paper copies'],'current_dinowm_status':json.loads((r/'reports/dinowm_pushobj_v3/pipeline_status.json').read_text())}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(out/'README.md').write_text('''# CVPR 2027 paper and evaluation artifacts — v14

The main manuscript is [paper_v14.pdf](../../paper/cvpr2027/paper_v14.pdf); editable source and native simulator figures are in `paper/cvpr2027/`.

## Results included

- **PushT and six visual shifts:** all 56 checkpoint/update/ECP/condition evaluations, 2,800 task runs; 1,400 pre-intervention pairs verified.
- **Frozen Maze:** native 48/100, prespecified S 50/100, S+P component control 54/100; completed paired records and inactive-task reuse manifest.
- **DINO-WM PushT CEM:** random T16k, fixed epoch 3, 300 candidates x 30 iterations, 6 x 25 environment controls. Seven scores are 48/40/2/44/40/24/48 percent (clean/blur/salt-pepper/dark/agent/object/goal); six-shift mean 33.0%. Five raw NPZ conditions and two completed-summary conditions are explicitly distinguished in `paper/cvpr2027/provenance/dinowm_pusht_cem_e3_v14.json`. The historical source folder name includes `e10`; checkpoint receipts identify epoch 3.
- **New DINO-WM PushObj experiment:** both three-epoch trainings are complete. The v3 200 x 10 CEM evaluation uses six plans executing 25 controls each and remains in progress at the snapshot time. Partial results are archived separately, not added to the paper's complete benchmark table.

## Reading the archives

`manifest.json` records the snapshot time, classifications and SHA-256 hashes. Each `.tar.gz` includes an `ARCHIVE_MANIFEST.json` listing every member and its content hash. Extract an archive from the repository root to recover original relative paths:

```bash
tar -xzf reports/publication_v14/archives/pusht_visual_complete.tar.gz
```

Archives retain raw success arrays, start/goal states, normalized executed actions, per-iteration/episode logs, configurations and completion evidence. Duplicated rendered observation caches, videos and old generated PDF copies are excluded; the saved actions and simulator provide replay. Model binaries and image training corpora remain outside Git, with source/download instructions, configuration and identity hashes retained. `dataset_and_checkpoint_metadata/` contains collection manifests and training/checkpoint metadata. Existing main PushObj and AdaJEPA Maze reports remain in their original tracked report folders.

No new experiment is launched by packaging or paper-building scripts. Historical v1/v2 archives are superseded and must not be aggregated with v3. The publication branch freezes a point-in-time snapshot; live processes continue locally.
''')
