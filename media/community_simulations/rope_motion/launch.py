"""Record all selected Rope pairs on an explicitly selected GPU."""
import argparse
import os
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
LOCAL = REPO / 'eval_outputs/rope_local'
CASES = {'rope_ablation': ('rope_seed101_20', 5, ['native3', 'fixed3']),
         'rope_methods': ('rope_qualitative20', 3, ['native5', 'fixed3']),
         'rope_additional': ('rope_seed101_20', 14, ['native3', 'fixed3'])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', required=True, type=int)
    parser.add_argument('--case', choices=CASES)
    args = parser.parse_args()
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
    devices = {int(line.split(',')[0]): line.split(',')[1].strip() for line in rows.splitlines()}
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=devices[args.gpu], EGL_GPU=str(args.gpu),
               PYFLEXROOT=str(LOCAL / 'AdaptiGraph-main/PyFleX'), PYTHONPATH=str(LOCAL / 'pyflex-build'),
               SDL_VIDEODRIVER='dummy', __EGL_VENDOR_LIBRARY_FILENAMES='/usr/share/glvnd/egl_vendor.d/10_nvidia.json',
               OMP_NUM_THREADS='2', TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1', PYTHONUNBUFFERED='1')
    for case, (root, sample, arms) in CASES.items():
        if args.case and case != args.case:
            continue
        for arm, label in zip(arms, ['baseline', 'ours']):
            output = HERE / case / label
            output.mkdir(parents=True, exist_ok=True)
            with (output / 'recording.log').open('a') as log:
                subprocess.run([str(LOCAL / 'venv/bin/python'), str(HERE / 'record.py'),
                                str(REPO / f'eval_outputs/{root}/{arm}/sample_{sample:02d}'), str(output)],
                               cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


if __name__ == '__main__':
    main()
