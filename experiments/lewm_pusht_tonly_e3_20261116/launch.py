"""Launch one four-GPU T-only training job with persistent status and final checks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
RUN = REPO / 'eval_outputs/lewm_pusht_tonly_e3_20261116'
PYTHON = REPO / 'eval_outputs/rope_local/venv/bin/python'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(name, value):
    target = RUN / name
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(target)


def supervise():
    status = {'status': 'preparing', 'started_unix': time.time(), 'pid': os.getpid()}
    save('status.json', status)
    try:
        import yaml
        cfg = yaml.safe_load((ROOT / 'config.yaml').read_text())
        audit = json.loads((ROOT / 'data_audit.json').read_text())
        preflight = json.loads((ROOT / 'preflight.json').read_text())
        assert audit['passed'] and preflight['passed'] and preflight['gpu_count'] == 4
        assert cfg['training']['epochs'] == 3 and cfg['training']['batch_size'] == 128
        assert cfg['env']['dataset']['data_path'] == audit['data_path']
        output = Path(cfg['saved_folder'])
        assert not output.exists(), 'Use a new output directory; do not overwrite a previous model'
        inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used',
                                             '--format=csv,noheader,nounits'], text=True)
        devices = {int(row.split(',')[0]): (row.split(',')[1].strip(), int(row.split(',')[2]))
                   for row in inventory.splitlines()}
        assert all(devices[i][1] < 500 for i in (4, 5, 6, 7)), 'A requested GPU is occupied'
        nccl = REPO / 'eval_outputs/lewm_pusht_prepare/nccl_runtime/nvidia/nccl/lib/libnccl.so.2'
        assert nccl.is_file() and preflight['nccl_runtime'] == 22707
        overrides = {'CUDA_VISIBLE_DEVICES': ','.join(devices[i][0] for i in (4, 5, 6, 7)),
                     'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
                     'NCCL_P2P_DISABLE': '1', 'NCCL_IB_DISABLE': '1', 'NCCL_DEBUG': 'WARN',
                     'NCCL_CUMEM_HOST_ENABLE': '0', 'NCCL_CUMEM_ENABLE': '0', 'LD_PRELOAD': str(nccl)}
        command = [str(PYTHON), '-u', '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4',
                   str(ROOT / 'runtime/train_lewm_distributed.py'), str(ROOT / 'config.yaml')]
        save('manifest.json', {'command': command, 'environment_overrides': overrides,
            'physical_gpus': [4, 5, 6, 7], 'epochs': 3, 'initialization': 'random seed 0; no pretrained checkpoint',
            'global_batch': 128, 'local_batch': 32, 'data_audit': audit, 'preflight': preflight,
            'config_sha256': sha(ROOT / 'config.yaml'),
            'nccl_library_sha256': sha(nccl),
            'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in ROOT.rglob('*.py')},
            'distinct_from_historical_T_only_epoch10': True})
        environment = os.environ.copy(); environment.update(overrides)
        with (RUN / 'train.log').open('w') as stream:
            child = subprocess.Popen(command, cwd=ROOT / 'runtime', env=environment,
                                     stdout=stream, stderr=subprocess.STDOUT)
            status.update(status='training', trainer_pid=child.pid, checkpoint_dir=str(output))
            save('status.json', status)
            while child.poll() is None:
                status['updated_unix'] = time.time()
                metrics = output / 'metrics.jsonl'
                if metrics.exists():
                    lines = metrics.read_text().splitlines()
                    for line in reversed(lines):
                        try:
                            status['latest_metric'] = json.loads(line)
                            break
                        except json.JSONDecodeError:
                            continue
                save('status.json', status)
                time.sleep(30)
            assert child.returncode == 0, f'Training exited {child.returncode}; inspect train.log'
        import torch
        from omegaconf import OmegaConf
        sys.path.insert(0, str(ROOT / 'runtime'))
        from models.lewm import build_lewm
        saved = OmegaConf.load(output / 'hydra.yaml')
        model = build_lewm(saved.model)
        checkpoint = torch.load(output / 'checkpoints/model_3.pth', map_location='cpu', weights_only=True)
        assert checkpoint['epoch'] == 3 and checkpoint['step'] == 3 * 12759
        assert all(torch.isfinite(tensor).all() for tensor in checkpoint['model'].values())
        model.load_state_dict(checkpoint['model'], strict=True)
        latest = torch.load(output / 'checkpoints/model_latest.pth', map_location='cpu', weights_only=True)
        assert latest['epoch'] == 3 and latest['step'] == checkpoint['step']
        assert all(torch.equal(tensor, latest['model'][key]) for key, tensor in checkpoint['model'].items())
        result = {'epoch': 3, 'step': checkpoint['step'], 'strict_load': True, 'all_tensors_finite': True,
                  'checkpoint': str(output / 'checkpoints/model_3.pth'),
                  'checkpoint_sha256': sha(output / 'checkpoints/model_3.pth'),
                  'hydra_sha256': sha(output / 'hydra.yaml'), 'latest_tensors_match': True}
        save('training_result.json', result)
        status.update(status='complete', result=result, ended_unix=time.time())
    except Exception as exc:
        status.update(status='failed', error=repr(exc), ended_unix=time.time())
        save('status.json', status)
        raise
    save('status.json', status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--supervise', action='store_true')
    args = parser.parse_args()
    if args.supervise:
        supervise()
        return
    assert not RUN.exists(), 'Existing training run; inspect its status instead of launching twice'
    RUN.mkdir(parents=True)
    with (RUN / 'supervisor.log').open('w') as stream:
        process = subprocess.Popen([str(PYTHON), str(Path(__file__).resolve()), '--supervise'],
                                   cwd=REPO, start_new_session=True, stdout=stream, stderr=subprocess.STDOUT)
    save('launch.json', {'supervisor_pid': process.pid, 'launched_unix': time.time()})
    print(json.dumps({'supervisor_pid': process.pid, 'status_file': str(RUN / 'status.json')}))


if __name__ == '__main__':
    main()
