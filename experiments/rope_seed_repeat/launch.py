"""Run the four paired Rope 3/3 conditions, one experiment per GPU, then report."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import difflib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
LOCAL = REPO / 'eval_outputs/rope_local'
ARMS = ('native3', 'fixed3', 'dynamic3', 'fixed3_nowarm')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--gpu-map', default='native3=1,fixed3=2,dynamic3=3,fixed3_nowarm=4')
    args = parser.parse_args()
    if args.seed < 0 or args.seed == 99:
        raise ValueError('Choose a nonnegative seed different from the original 99.')
    root = args.root.resolve()
    if root.exists():
        raise RuntimeError('Use a fresh output root; never mix two independent runs.')
    assignments = {a: int(g) for a, g in (item.split('=') for item in args.gpu_map.split(','))}
    assert set(assignments) == set(ARMS) and len(set(assignments.values())) == 4
    inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used',
                                         '--format=csv,noheader,nounits'], text=True)
    devices = {int(row.split(',')[0]): (row.split(',')[1].strip(), int(row.split(',')[2]))
               for row in inventory.splitlines()}
    assert all(devices[g][1] < 500 for g in assignments.values()), 'A selected GPU is occupied'
    (root / 'logs').mkdir(parents=True)
    source = root / 'source_snapshot'
    source.mkdir()
    originals = [HERE / 'worker.py', HERE / 'launch.py',
                 REPO / 'experiments/rope_chunk3/chunk_planner.py',
                 REPO / 'experiments/rope_feedback/run.py',
                 REPO / 'experiments/rope_feedback/feedback_planner.py',
                 REPO / 'experiments/rope_feedback/protocol.yaml',
                 REPO / 'experiments/rope_ecp_20261110/rope_ecp.py',
                 LOCAL / 'runtime/plan.py', LOCAL / 'runtime/planning/evaluator.py',
                 LOCAL / 'runtime/models/visual_world_model.py']
    source_hashes = {}
    for path in originals:
        relative = path.relative_to(REPO)
        dest = source / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(path.read_bytes())
        source_hashes[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    before = (REPO / 'experiments/rope_formal20/worker.py').read_text()
    after = (HERE / 'worker.py').read_text()
    (source / 'worker_change.diff').write_text(''.join(difflib.unified_diff(
        before.splitlines(True), after.splitlines(True), fromfile='original_formal_worker.py', tofile='seed_repeat_worker.py')))
    manifest = dict(status='running', started=time.time(), base_seed=args.seed,
                    gpu_assignments=assignments, samples=20, arms=list(ARMS), maximum_pushes=20,
                    target_chunks=[1, 2], global_targets=list(range(10, 30)),
                    target_policy='same ordered 20 targets as seed99; only random seeds change',
                    scene_seed=args.seed, environment_seed_formula='base_seed * global_target + 1',
                    planning_seed_formula='base_seed + global_target',
                    source_sha256=source_hashes, completed=[])
    (root / 'queue.json').write_text(json.dumps(manifest, indent=2) + '\n')

    def run_arm(arm, gpu):
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=devices[gpu][0], EGL_GPU=str(gpu),
                   PYFLEXROOT=str(LOCAL / 'AdaptiGraph-main/PyFleX'),
                   PYTHONPATH=str(LOCAL / 'pyflex-build'), SDL_VIDEODRIVER='dummy',
                   __EGL_VENDOR_LIBRARY_FILENAMES='/usr/share/glvnd/egl_vendor.d/10_nvidia.json',
                   OMP_NUM_THREADS='4', TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1', PYTHONUNBUFFERED='1')
        for sample in range(20):
            # Abort if the experiment implementation changes while queued.
            for name, expected in source_hashes.items():
                assert hashlib.sha256((REPO / name).read_bytes()).hexdigest() == expected, name
            log = root / 'logs' / f'{arm}_sample_{sample:02d}.log'
            print(f'GPU {gpu}: {arm} sample {sample:02d}', flush=True)
            with log.open('w') as stream:
                command = [str(LOCAL / 'venv/bin/python'), str(HERE / 'worker.py'),
                           '--root', str(root), '--sample', str(sample), '--arm', arm,
                           '--seed', str(args.seed), '--max-pushes', '20']
                status = subprocess.run(command, cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT)
            if status.returncode:
                (root / f'failure_{arm}.json').write_text(json.dumps(dict(
                    arm=arm, sample=sample, returncode=status.returncode, log=str(log))) + '\n')
                raise RuntimeError(f'{arm} sample {sample} failed; inspect {log}')
        return arm

    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(run_arm, arm, gpu) for arm, gpu in assignments.items()]
            for future in as_completed(futures):
                manifest['completed'].append(future.result())
                (root / 'queue.json').write_text(json.dumps(manifest, indent=2) + '\n')
        manifest.update(status='complete', elapsed_s=time.time() - manifest['started'])
        (root / 'queue.json').write_text(json.dumps(manifest, indent=2) + '\n')
        env = os.environ.copy()
        env['MPLCONFIGDIR'] = '/tmp/rope-seed-repeat-matplotlib'
        with (root / 'report.log').open('w') as stream:
            subprocess.run([str(LOCAL / 'venv/bin/python'), str(HERE / 'report.py'),
                            '--root', str(root), '--output', str(args.report.resolve())],
                           cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        manifest['report_status'] = 'complete'
        (root / 'queue.json').write_text(json.dumps(manifest, indent=2) + '\n')
    except BaseException as error:
        manifest.update(failure=str(error), elapsed_s=time.time() - manifest['started'])
        if manifest['status'] != 'complete':
            manifest['status'] = 'failed'
        else:
            manifest['report_status'] = 'failed'
        (root / 'queue.json').write_text(json.dumps(manifest, indent=2) + '\n')
        raise


if __name__ == '__main__':
    main()
