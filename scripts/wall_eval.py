"""Run the public DINO-WM Wall checkpoint with a fixed, paired G0 protocol.

python scripts/wall_eval.py prepare
python scripts/wall_eval.py run --gpus 0,1
python scripts/wall_eval.py summarize

Both formal arms use 50 tasks, seed 99, CEM 300/30/10, horizon/execute 25
physical steps (5 model steps). MPC has a predeclared 10-replan/250-step cap.
This does not train a model or run the JEPA-Stall adaptation arms.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/wall_single_eval'
CKPT = ROOT / 'checkpoints/dinowm_wall_single'
DEFAULT_OUTPUT = ROOT / 'eval_outputs/wall_dinowm/20261103'
ARMS = {'open_loop': 1, 'mpc10': 10}


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def prepare():
    import torch
    archive = ROOT / 'data/_dl_dinowm/wall_single.zip'
    weights = ROOT / 'data/_dl_dinowm_ckpt/outputs.zip'
    with zipfile.ZipFile(archive) as z:
        states = torch.load(io.BytesIO(z.read('wall_single/states.pth')),
                            map_location='cpu', weights_only=True)
        assert states.shape == (1920, 50, 2), states.shape
        val = torch.randperm(len(states), generator=torch.Generator().manual_seed(42)).tolist()[1728:]
        names = [f'{n}.pth' for n in ('states', 'actions', 'door_locations', 'wall_locations')]
        names += [f'obses/episode_{i:03d}.pth' for i in val]
        for name in names:
            extract(z, 'wall_single/' + name, DATA / name)
    with zipfile.ZipFile(weights) as z:
        for name in ('hydra.yaml', 'checkpoints/model_latest.pth'):
            extract(z, 'outputs/wall_single/' + name, CKPT / name)
    for name in ('states', 'actions', 'door_locations', 'wall_locations'):
        tensor = torch.load(DATA / f'{name}.pth', map_location='cpu', weights_only=True)
        assert len(tensor) == 1920 and torch.isfinite(tensor).all(), name
    write_json(DATA / 'eval_manifest.json', {
        'purpose': 'Evaluation only: full normalization tensors, validation images only',
        'n_trajectories': 1920, 'split_seed': 42, 'train_fraction': 0.9,
        'validation_indices': val,
        'dataset_archive_sha256': sha256(archive),
        'checkpoint_archive_sha256': sha256(weights),
    })
    print(f'Prepared {len(val)} validation trajectories in {DATA}', flush=True)


def extract(z, member, dest):
    info = z.getinfo(member)
    if dest.exists():
        # Check CRC as well as length when reusing an extracted file.
        import zlib
        crc = 0
        with dest.open('rb') as f:
            for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
                crc = zlib.crc32(b, crc)
        if dest.stat().st_size != info.file_size or crc != info.CRC:
            raise ValueError(f'Existing file differs from archive: {dest}')
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + '.partial')
    with z.open(member) as src, tmp.open('wb') as out:
        shutil.copyfileobj(src, out)
    tmp.replace(dest)


def summarize(output):
    import numpy as np
    result = {'protocol': 'Wall G0, n=50, seed=99, CEM=300/30/10, frameskip=5', 'arms': {}}
    completed = {}
    for arm in ARMS:
        folder = output / arm
        rows = [json.loads(line) for line in (folder / 'logs.json').read_text().splitlines()
                if line.strip()] if (folder / 'logs.json').exists() else []
        curve = [{'replans': int(r['step']), 'env_steps': int(r['step']) * 25,
                  'success_rate': r['mpc/success_rate']} for r in rows if 'mpc/success_rate' in r]
        entry = {'complete': False, 'curve': curve}
        if (folder / 'final_eval_results.npz').exists() and any('final_eval/success_rate' in r for r in rows):
            with np.load(folder / 'final_eval_results.npz') as data:
                completed[arm] = {k: data[k].copy() for k in data.files}
            data = completed[arm]
            assert data['successes'].shape == (50,), data['successes'].shape
            assert np.isfinite(data['actions']).all()
            entry.update(complete=True, successes=int(data['successes'].sum()), n=50,
                         success_rate=float(data['successes'].mean()))
        result['arms'][arm] = entry
    if len(completed) == 2:
        a, b = completed['open_loop'], completed['mpc10']
        paired = all(np.array_equal(a[k], b[k]) for k in ('state_0', 'state_g'))
        prefix = np.array_equal(a['actions'], b['actions'][:, :5])
        result['same_tasks'] = paired
        result['same_first_plan_actions'] = prefix
        if not paired or not prefix:
            raise ValueError('Paired target/action-prefix audit failed')
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'summary.json', result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def run(output, gpu_ids):
    if len(gpu_ids) != 2 or len(set(gpu_ids)) != 2 or not all(g.isdigit() for g in gpu_ids):
        raise ValueError('--gpus requires two distinct physical GPU indices, e.g. 0,1')
    # Never fall back to CPU for a formal run or displace an existing GPU process.
    for gpu in gpu_ids:
        check_env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
        subprocess.run([sys.executable, '-c', 'import torch; assert torch.cuda.is_available(); '
                        'assert torch.cuda.device_count() == 1; print(torch.cuda.get_device_name(0))'],
                       env=check_env, check=True)
        occupied = subprocess.check_output(['nvidia-smi', '-i', gpu,
                     '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
        if occupied:
            raise RuntimeError(f'GPU {gpu} has existing compute processes: {occupied}')
    for path in (DATA / 'eval_manifest.json', CKPT / 'hydra.yaml', CKPT / 'checkpoints/model_latest.pth'):
        if not path.is_file():
            raise FileNotFoundError(f'{path}; run prepare first')
    for arm in ARMS:
        if (output / arm).exists():
            raise FileExistsError(f'{output / arm}; select a fresh --output to avoid mixing runs')
    output.mkdir(parents=True, exist_ok=True)
    manifest = {'started_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                'checkpoint_sha256': sha256(CKPT / 'checkpoints/model_latest.pth'),
                'training_config_sha256': sha256(CKPT / 'hydra.yaml'),
                'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'n_evals': 50, 'seed': 99, 'cem': {'samples': 300, 'topk': 30, 'iterations': 10},
                'frameskip': 5, 'horizon_env_steps': 25, 'execute_env_steps': 25, 'jobs': {}}
    running = []
    try:
        for (arm, cap), gpu in zip(ARMS.items(), gpu_ids):
            folder = output / arm
            folder.mkdir()
            cmd = [sys.executable, '-u', str(ROOT / 'plan.py'), '--config-name', 'plan_wall_dinowm',
                   f'ckpt_base_path={CKPT}', f'+eval_data_path={DATA}', '+wandb_logging=false',
                   'seed=99', 'n_evals=50', 'n_plot_samples=0', 'decode_for_viz=false',
                   f'planner.max_iter={cap}', 'planner.sub_planner.num_samples=300',
                   'planner.sub_planner.topk=30', 'planner.sub_planner.opt_steps=10',
                   f'hydra.run.dir={folder}']
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, WANDB_MODE='disabled',
                       HYDRA_FULL_ERROR='1', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8')
            log = (folder / 'console.log').open('w')
            process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            log.close()
            running.append((arm, process))
            manifest['jobs'][arm] = {'pid': process.pid, 'gpu': gpu, 'max_replans': cap, 'command': cmd}
            write_json(output / 'run_manifest.json', manifest)
            print(f'Started {arm} on GPU {gpu}, pid={process.pid}', flush=True)
        while running:
            for arm, process in list(running):
                code = process.poll()
                if code is not None:
                    manifest['jobs'][arm]['returncode'] = code
                    manifest['jobs'][arm]['finished_at'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
                    write_json(output / 'run_manifest.json', manifest)
                    print(f'{arm} exited {code}', flush=True)
                    running.remove((arm, process))
            if running:
                time.sleep(5)
    except BaseException:
        for _, process in running:
            process.terminate()
        for _, process in running:
            process.wait()
        raise
    summarize(output)
    if any(j.get('returncode') != 0 for j in manifest['jobs'].values()):
        raise RuntimeError(f'Wall evaluation failed; see {output}/*/console.log')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=['prepare', 'run', 'summarize'])
    ap.add_argument('--gpus', default='0,1')
    ap.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    a = ap.parse_args()
    output = a.output.resolve()
    if a.command == 'prepare':
        prepare()
    elif a.command == 'run':
        run(output, a.gpus.split(','))
    else:
        summarize(output)


if __name__ == '__main__':
    main()
