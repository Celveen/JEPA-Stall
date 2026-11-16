"""Evaluate ten fixed Rope tasks with feedback after each of at most twenty pushes."""

import argparse
import copy
import json
import os
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
LOCAL = REPO / 'eval_outputs/rope_local'
RUNTIME = LOCAL / 'runtime'
CHECKPOINT = REPO / 'checkpoints/rope_dino_h1_fs1_e100_seed0'
DINO_WEIGHTS = LOCAL / 'torch/hub/checkpoints/dinov2_vits14_pretrain.pth'
DINO_SHA256 = 'b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9'
sys.path[:0] = [str(HERE), str(RUNTIME), str(REPO / 'experiments/rope_ecp_20261110')]

from run_chunk import sha256

# Reproduces all 750 archived normalized/physical action pairs exactly in float32.
ACTION_MEAN = [0.01266330387443304, -0.1220608651638031, -0.0023628182243555784, -0.021762890741229057]
ACTION_STD = [2.809385061264038, 2.169008255004883, 0.8589053750038147, 0.8438628911972046]


def build_config(arm, chunk, output, reference=None):
    from omegaconf import OmegaConf

    config = OmegaConf.to_container(OmegaConf.load(HERE / 'protocol.yaml'))
    config.update(ckpt_base_path=str(CHECKPOINT), seed=99 + chunk,
                  goal_file_path=str(LOCAL / f'targets/chunk_{chunk}.pkl'),
                  saved_folder=str(output))
    if arm == 'native':
        config['planner']['ecp']['kappa'] = 0.0
    if reference is not None:
        config['planner']['ecp']['reference_dir'] = str(reference)
    return config


def validate_reference(previous, receipt):
    if previous.get('arm') != 'native':
        raise ValueError('The reference must be a completed Native run.')
    for key in ('chunk', 'target_sha256', 'checkpoint_sha256', 'encoder_sha256', 'action_mean', 'action_std',
                'implementation_sha256'):
        if previous.get(key) != receipt[key]:
            raise ValueError(f'Native reference disagrees on {key}.')
    # Both arms must share cadence, execution budget, sampler, seeds and detector.
    configs = [copy.deepcopy(previous['config']), copy.deepcopy(receipt['config'])]
    for config in configs:
        config.pop('saved_folder', None)
        config['planner']['ecp'].pop('reference_dir', None)
        config['planner']['ecp'].pop('kappa')
    if configs[0] != configs[1]:
        raise ValueError('Native reference uses a different evaluation protocol.')


def verify_encoder(model, expected):
    """The frozen encoder is absent from the Rope checkpoint: check its real tensors."""
    import torch

    actual = model.encoder.base_model.state_dict()
    if actual.keys() != expected.keys():
        raise RuntimeError('DINO backbone keys do not match the official pretrained encoder.')
    for name, value in actual.items():
        if not torch.equal(value.detach().cpu(), expected[name]):
            raise RuntimeError(f'DINO backbone is not the official pretrained encoder: {name}')
    return {'matches_official_pretrained_weights': True, 'verified_tensors': len(actual),
            'encoder_sha256': DINO_SHA256,
            'source': 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth'}


def configure_runtime():
    import hydra
    import torch
    import plan

    if sha256(DINO_WEIGHTS) != DINO_SHA256:
        raise RuntimeError('Missing or changed official DINOv2 encoder weights.')
    encoder_state = torch.load(DINO_WEIGHTS, map_location='cpu', weights_only=True)

    original_hub_load = torch.hub.load

    def load_local_encoder(repository, *positional, **keywords):
        is_dino = repository == 'facebookresearch/dinov2'
        if is_dino:
            if positional != ('dinov2_vits14',):
                raise ValueError('This runner is pinned to the DINOv2 ViT-S/14 backbone.')
            repository = str(LOCAL / 'torch/hub/facebookresearch_dinov2_main')
            # Construct locally, then load EVERY pretrained tensor explicitly.
            # The frozen backbone was intentionally omitted from the Rope checkpoint.
            keywords.update(source='local', pretrained=False)
        encoder = original_hub_load(repository, *positional, **keywords)
        if is_dino:
            encoder.load_state_dict(encoder_state, strict=True)
        return encoder

    torch.hub.load = load_local_encoder
    original_load_model = plan.load_model

    def load_verified_model(*positional, **keywords):
        model = original_load_model(*positional, **keywords)
        audit = verify_encoder(model, encoder_state)
        Path('encoder_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
        print(f"Verified all {audit['verified_tensors']} DINOv2 pretrained tensors.", flush=True)
        return model

    plan.load_model = load_verified_model
    original_call = hydra.utils.call

    def load_fixed_target_stats(config, *positional, **keywords):
        if config.get('_target_') != 'datasets.deformable_env_dset.load_deformable_dset_slice_train_val':
            return original_call(config, *positional, **keywords)
        dataset = SimpleNamespace(
            action_dim=4, action_mean=torch.tensor(ACTION_MEAN), action_std=torch.tensor(ACTION_STD),
            state_mean=None, state_std=None, proprio_mean=torch.zeros(1),
            proprio_std=torch.full((1,), 1e-6), transform=original_call(config.transform),
        )
        return None, {'valid': dataset}

    hydra.utils.call = load_fixed_target_stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=['native', 'S'], required=True)
    parser.add_argument('--chunk', type=int, choices=range(5), default=0)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path, help='Completed Native run with this protocol.')
    parser.add_argument('--prepare', action='store_true', help='Validate inputs and save the plan without simulation.')
    args = parser.parse_args()
    if (args.arm == 'S') != (args.reference is not None):
        parser.error('--reference is required for +S and only applies to +S.')

    import numpy as np

    output = args.output.resolve()
    reference = args.reference.resolve() if args.reference else None
    config = build_config(args.arm, args.chunk, output, reference)
    target = Path(config['goal_file_path'])
    checkpoint = CHECKPOINT / 'checkpoints/model_latest.pth'
    with target.open('rb') as stream:
        tasks = pickle.load(stream)
    for key in ('obs_0', 'obs_g'):
        if any(len(value) != 10 for value in tasks[key].values()):
            parser.error('Each evaluation must contain exactly ten tasks.')
    if len(tasks['state_0']) != 10 or len(tasks['state_g']) != 10:
        parser.error('Each evaluation must contain exactly ten initial and goal states.')
    if sha256(DINO_WEIGHTS) != DINO_SHA256:
        raise RuntimeError('The pinned pretrained DINOv2 weights are required before evaluation.')
    source_files = [HERE / 'run.py', HERE / 'protocol.yaml', HERE / 'feedback_planner.py',
                    REPO / 'experiments/rope_ecp_20261110/rope_ecp.py',
                    RUNTIME / 'planning/mpc.py', RUNTIME / 'plan.py']
    receipt = {'arm': args.arm, 'chunk': args.chunk, 'target_sha256': sha256(target),
               'checkpoint_sha256': sha256(checkpoint), 'encoder_sha256': DINO_SHA256, 'config': config,
               'action_mean': ACTION_MEAN, 'action_std': ACTION_STD,
               'implementation_sha256': {str(path.relative_to(REPO)): sha256(path) for path in source_files}}
    previous = None
    if reference:
        previous = json.loads((reference / 'complete.json').read_text())
        validate_reference(previous, receipt)
        with np.load(reference / 'native_execution_cache.npz') as cache:
            shape = cache['physical_actions'].shape
            if len(shape) != 3 or shape[0] != 10 or not 0 <= shape[1] <= 20 or shape[2] != 4:
                parser.error('Native reference must contain ten tasks and at most twenty pushes.')

    prepared = output / 'prepared.json'
    if output.exists() and any(path != prepared for path in output.iterdir()):
        parser.error('Choose an empty output directory or a directory containing only prepared.json.')
    output.mkdir(parents=True, exist_ok=True)
    if args.prepare:
        prepared.write_text(json.dumps(receipt, indent=2) + '\n')
        print(f'Prepared ten tasks, at most twenty pushes, per-push feedback, stop at CD < 0.3. {prepared}')
        return

    import torch
    import plan

    if not torch.cuda.is_available():
        raise RuntimeError('Rope requires a visible NVIDIA GPU for PyFleX.')
    receipt.update(torch=torch.__version__, gpu=torch.cuda.get_device_name())
    if previous is not None:
        for key in ('torch', 'gpu'):
            if previous.get(key) != receipt[key]:
                raise ValueError(f'Native reference disagrees on {key}.')
    configure_runtime()
    (output / 'run_manifest.json').write_text(json.dumps(receipt, indent=2) + '\n')
    os.chdir(output)
    logs = plan.planning_main(config)
    if sha256(checkpoint) != receipt['checkpoint_sha256']:
        raise RuntimeError('Checkpoint changed during evaluation.')
    if sha256(DINO_WEIGHTS) != receipt['encoder_sha256']:
        raise RuntimeError('Pretrained DINOv2 weights changed during evaluation.')
    with np.load(output / 'rope_metrics.npz') as metrics:
        cd = metrics['cd_by_replan']
        if cd.shape[0] != 10 or cd.shape[1] > 20 or not np.isfinite(cd).all():
            raise RuntimeError('Invalid per-push CD measurements.')
        np.testing.assert_array_equal(metrics['successes'], metrics['final_cd'] < 0.3)
        np.testing.assert_array_equal(metrics['actual_pushes'], metrics['executed_mask'].sum(axis=1))
        if np.any((~metrics['successes']) & (metrics['actual_pushes'] != 20)):
            raise RuntimeError('An unsuccessful task stopped before exhausting its budget.')
        for index in range(10):
            timeline = np.r_[metrics['initial_cd'][index], cd[index]]
            hits = np.flatnonzero(timeline < 0.3)
            if len(hits):
                if metrics['actual_pushes'][index] != hits[0]:
                    raise RuntimeError('Task continued after its first successful state.')
                np.testing.assert_array_equal(timeline[hits[0]:], timeline[hits[0]])
    receipt['final_logs'] = {key: float(value) for key, value in logs.items()}
    (output / 'complete.json').write_text(json.dumps(receipt, indent=2) + '\n')


if __name__ == '__main__':
    main()
