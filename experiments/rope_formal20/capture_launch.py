"""Run predeclared qualitative pairs on one GPU per condition, without altering formal runs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import time
from launch import HERE, REPO, LOCAL


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--samples',type=int,nargs='+',default=[3,7])
    p.add_argument('--gpus',type=int,nargs=2,default=[5,6])
    p.add_argument('--selection-note',default=None)
    a=p.parse_args(); root=a.root.resolve(); root.mkdir(parents=True,exist_ok=True)
    if len(set(a.gpus))!=2 or len(set(a.samples))!=len(a.samples):
        raise ValueError('Distinct GPUs and samples required')
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'],text=True)
    dev={int(r.split(',')[0]):(r.split(',')[1].strip(),int(r.split(',')[2])) for r in rows.splitlines()}
    assert all(dev[g][1]<500 for g in a.gpus), 'GPU occupied'
    selection=a.selection_note
    if selection is None:
        if a.samples != [3,7]:
            raise ValueError('Describe the prospective target selection with --selection-note')
        selection='Targets chosen before capture: sample 03 improved and sample 07 regressed in completed formal pairs. No repeat-until-favorable selection.'
    manifest=dict(role='independent qualitative reruns, excluded from formal-20 results', samples=a.samples,
        arms=['native5','fixed3'], gpu_assignments=dict(zip(['native5','fixed3'],a.gpus)), started=time.time(),
        selection=selection)
    (root/'queue.json').write_text(json.dumps(manifest,indent=2)+'\n')
    def run(arm,gpu):
        env=os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=dev[gpu][0], EGL_GPU=str(gpu),PYFLEXROOT=str(LOCAL/'AdaptiGraph-main/PyFleX'),
            PYTHONPATH=str(LOCAL/'pyflex-build'),SDL_VIDEODRIVER='dummy',
            __EGL_VENDOR_LIBRARY_FILENAMES='/usr/share/glvnd/egl_vendor.d/10_nvidia.json',
            OMP_NUM_THREADS='4',TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',PYTHONUNBUFFERED='1')
        for i in a.samples:
            with (root/f'{arm}_sample_{i:02d}.log').open('w') as stream:
                subprocess.run([str(LOCAL/'venv/bin/python'),str(HERE/'capture_worker.py'),'--root',str(root),
                    '--sample',str(i),'--arm',arm,'--max-pushes','20'],env=env,cwd=REPO,stdout=stream,stderr=subprocess.STDOUT,check=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(run,arm,gpu) for arm,gpu in zip(manifest['arms'],a.gpus)]
        for future in futures: future.result()
    manifest.update(status='complete',elapsed_s=time.time()-manifest['started'])
    (root/'queue.json').write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__': main()
