"""Assign one complete twenty-episode experiment to each selected GPU."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
LOCAL = REPO / 'eval_outputs/rope_local'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--gpu-map', default='native5=1,fixed3=2,dynamic3=3,fixed3_nowarm=4')
    parser.add_argument('--max-pushes', type=int, choices=(15, 20), default=20)
    args = parser.parse_args()
    root = args.root.resolve()
    (root / 'logs').mkdir(parents=True, exist_ok=True)
    assignments = {item.split('=')[0]: int(item.split('=')[1]) for item in args.gpu_map.split(',')}
    if set(assignments) != {'native5','fixed3','dynamic3','fixed3_nowarm'}:
        raise ValueError('Assign the four requested formal conditions.')
    gpus = list(assignments.values())
    if len(set(gpus)) != len(gpus):
        raise ValueError('Each experiment needs its own GPU.')
    inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used',
                                         '--format=csv,noheader,nounits'], text=True)
    devices = {int(row.split(',')[0]): (row.split(',')[1].strip(), int(row.split(',')[2]))
               for row in inventory.splitlines()}
    if any(devices[gpu][1] > 500 for gpu in gpus):
        raise RuntimeError('A requested GPU is occupied; refusing to share it silently.')
    manifest = dict(started=time.time(), gpu_assignments=assignments, samples=20,
                    arms=list(assignments), maximum_pushes=args.max_pushes, completed=[], target_chunks=[1, 2], protocol='independent physics; no Native-3 reference')
    (root / 'queue.json').write_text(json.dumps(manifest, indent=2) + '\n')

    def worker(arm, gpu):
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=devices[gpu][0], EGL_GPU=str(gpu),
                   PYFLEXROOT=str(LOCAL / 'AdaptiGraph-main/PyFleX'),
                   PYTHONPATH=str(LOCAL / 'pyflex-build'), SDL_VIDEODRIVER='dummy',
                   __EGL_VENDOR_LIBRARY_FILENAMES='/usr/share/glvnd/egl_vendor.d/10_nvidia.json',
                   OMP_NUM_THREADS='4', TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1', PYTHONUNBUFFERED='1')
        done = []
        for sample in range(20):
            output = root / arm / f'sample_{sample:02d}'
            if (output / 'complete.json').exists():
                completed = json.loads((output / 'complete.json').read_text())
                if completed['config']['planner']['max_iter'] != args.max_pushes:
                    raise RuntimeError('An existing episode has a different push limit.')
                done.append([sample, arm])
                continue
            log = root / 'logs' / f'{arm}_sample_{sample:02d}.log'
            print(f'GPU {gpu}: sample {sample}, {arm}', flush=True)
            with log.open('w') as stream:
                status = subprocess.run([str(LOCAL / 'venv/bin/python'), str(HERE / 'worker.py'),
                                         '--root', str(root), '--sample', str(sample), '--arm', arm,
                                         '--max-pushes', str(args.max_pushes)],
                                        env=env, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT)
            if status.returncode:
                (root / 'failure.json').write_text(json.dumps(dict(arm=arm, sample=sample,
                                                                  returncode=status.returncode)) + '\n')
                raise RuntimeError(f'{arm} sample {sample} failed; see {log}')
            done.append([sample, arm])
        return done

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [pool.submit(worker, arm, gpu) for arm, gpu in assignments.items()]
        for future in as_completed(futures):
            manifest['completed'].extend(future.result())
            (root / 'queue.json').write_text(json.dumps(manifest, indent=2) + '\n')
    manifest.update(status='complete', elapsed_s=time.time() - manifest['started'])
    (root / 'queue.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
