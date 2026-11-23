"""One frozen Rope chunk, with a pinned local runtime and fixed target identity."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time

RUNTIME = Path('/data/zjc/adaJEPA/CL-JEPA-cudagraph')
CHECKPOINT = Path('/data/zjc/adaJEPA/CL-JEPA/checkpoints/rope_dino_h1_fs1_e100_seed0')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=['native', 'S'], required=True)
    parser.add_argument('--chunk', type=int, choices=range(5), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--first-only', action='store_true', help='Integration only, excluded from 50-task results.')
    parser.add_argument('--force-trigger', action='store_true', help='Integration only; exercise S on the second plan.')
    args = parser.parse_args()
    if args.force_trigger and not args.first_only:
        raise ValueError('Forced activation is forbidden in benchmark runs.')
    args.output = args.output.resolve()
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError('Refusing to overwrite a nonempty run directory.')
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(RUNTIME))
    from omegaconf import OmegaConf
    import torch
    # Cache-only hub registration: no pretrained model or data download.
    original_hub_load = torch.hub.load
    cache = Path(torch.hub.get_dir()) / 'facebookresearch_dinov2_main'
    if not cache.is_dir():
        raise RuntimeError('Required local DINO cache is absent.')
    def local_hub_load(repository, *positional, **keywords):
        if repository == 'facebookresearch/dinov2':
            repository = str(cache)
            keywords['source'] = 'local'
            keywords['pretrained'] = False  # registration only; checkpoint supplies all encoder weights
        return original_hub_load(repository, *positional, **keywords)
    torch.hub.load = local_hub_load
    import plan
    from rope_ecp import RopeStagnationMPCPlanner  # registers the importable Hydra target

    cfg = OmegaConf.to_container(OmegaConf.load(RUNTIME / 'conf/plan_rope_dset.yaml'), resolve=False)
    cfg.pop('defaults', None)
    cfg.pop('hydra', None)
    source = RUNTIME / f'eval_outputs/rope_cl_single/stream/chunk_{args.chunk}.pkl'
    target = source
    if args.first_only:
        data = pickle.loads(source.read_bytes())
        for key in ['obs_0', 'obs_g']:
            data[key] = {name: value[:1] for name, value in data[key].items()}
        for key in ['state_0', 'state_g', 'gt_actions']:
            if data[key] is not None:
                data[key] = data[key][:1]
        target = args.output / 'integration_target.pkl'
        target.write_bytes(pickle.dumps(data))
    cfg.update({'ckpt_base_path': str(CHECKPOINT), 'model_epoch': '100',
                'seed': 99 + args.chunk, 'n_evals': 1 if args.first_only else 10,
                'goal_source': 'file', 'goal_file_path': str(target),
                'eval_data_path': '/data/zjc/adaJEPA/CL-JEPA/data/deformable',
                'decode_for_viz': False, 'n_plot_samples': 0,
                'saved_folder': str(args.output), 'wandb_logging': False})
    cfg['planner'].update({'_target_': 'rope_ecp.RopeStagnationMPCPlanner',
                           'resim_from_start': False, 'max_iter': 3,
                           'n_taken_actions': 5,
                           'ecp': {'kappa': 0.0 if args.arm == 'native' else 2.0,
                                   'warmup_pushes': 5, 'window_pushes': 3,
                                   'gamma_lo': 0.05, 'gamma_hi': 0.15,
                                   'beta': 0.25, 'alpha': 1.0, 'base': 2.0,
                                   'component': 'S-only', 'objective_scale': 'terminal'}})
    cfg['planner']['sub_planner']['target'] = 'rope_ecp.ContextCEMPlanner'
    if args.arm == 'S':
        cfg['planner']['ecp']['reference_dir'] = str(args.output.parent.parent / 'native' / args.output.name)
    if args.force_trigger:
        cfg['planner']['max_iter'] = 2
        cfg['planner']['ecp'].update({'gamma_lo': 1e6, 'gamma_hi': 1e6 + 1,
                                      'forced_integration_only': True})
    checkpoint_file = CHECKPOINT / 'checkpoints/model_100.pth'
    before = sha256(checkpoint_file)
    receipt = {'arm': args.arm, 'chunk': args.chunk, 'integration_only': args.first_only,
               'target_source': str(source), 'target_sha256': sha256(source),
               'checkpoint': str(checkpoint_file), 'checkpoint_sha256': before,
               'runtime': str(RUNTIME), 'config': cfg}
    (args.output / 'run_manifest.json').write_text(json.dumps(receipt, indent=2) + '\n')
    os.chdir(args.output)
    started = time.perf_counter()
    logs = plan.planning_main(cfg)
    if sha256(checkpoint_file) != before:
        raise RuntimeError('Checkpoint file changed during evaluation.')
    receipt.update({'status': 'complete', 'elapsed_s': time.perf_counter() - started,
                    'final_logs': {key: float(value) for key, value in logs.items()},
                    'checkpoint_file_unchanged': True})
    (args.output / 'complete.json').write_text(json.dumps(receipt, indent=2) + '\n')


if __name__ == '__main__':
    main()
