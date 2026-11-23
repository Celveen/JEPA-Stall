"""One seed-repeat Rope episode, retaining the original 20 ordered targets."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import pickle
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
spec = importlib.util.spec_from_file_location('feedback_run', REPO / 'experiments/rope_feedback/run.py')
feedback = importlib.util.module_from_spec(spec)
spec.loader.exec_module(feedback)
sys.path.insert(0, str(REPO / 'experiments/rope_chunk3'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=('native5', 'native3', 'fixed3', 'dynamic3', 'fixed3_nowarm'), required=True)
    parser.add_argument('--sample', type=int, choices=range(20), required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--max-pushes', type=int, choices=(15, 20), default=20)
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()
    import numpy as np
    import torch
    import plan

    root = args.root.resolve()
    output = root / args.arm / f'sample_{args.sample:02d}'
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f'Refusing to overwrite {output}')
    output.mkdir(parents=True, exist_ok=True)
    native = args.arm.startswith('native')
    reference = None
    global_sample = 10 + args.sample
    target_chunk, target_index = divmod(global_sample, 10)
    config = feedback.build_config('native' if native else 'S', target_chunk, output, reference)
    original_target = Path(config['goal_file_path'])
    with original_target.open('rb') as stream:
        targets = pickle.load(stream)
    for key in ('obs_0', 'obs_g'):
        targets[key] = {name: value[target_index:target_index + 1] for name, value in targets[key].items()}
    for key in ('state_0', 'state_g', 'gt_actions'):
        if targets.get(key) is not None:
            targets[key] = targets[key][target_index:target_index + 1]
    target = output / 'target.pkl'
    target.write_bytes(pickle.dumps(targets))
    chunk = 5 if args.arm == 'native5' else 3
    config.update(n_evals=1, seed=args.seed, goal_file_path=str(target))
    config['planner'].update(_target_='chunk_planner.ObservedChunkPlanner', chunk_size=chunk,
                             max_iter=args.max_pushes,
                             strength_mode='fixed' if args.arm in ('fixed3', 'fixed3_nowarm') else 'dynamic',
                             episode_seed=args.seed * global_sample + 1, planning_seed=args.seed + global_sample)
    config['planner']['sub_planner']['horizon'] = chunk
    config['planner']['ecp'].update(warmup_pushes=0 if args.arm == 'fixed3_nowarm' else 5,
                                    independent_execution=True)
    sources = [HERE / 'worker.py', REPO / 'experiments/rope_chunk3/chunk_planner.py',
               REPO / 'experiments/rope_feedback/run.py', REPO / 'experiments/rope_feedback/feedback_planner.py',
               REPO / 'experiments/rope_feedback/protocol.yaml',
               REPO / 'experiments/rope_ecp_20261110/rope_ecp.py',
               feedback.RUNTIME / 'plan.py', feedback.RUNTIME / 'planning/evaluator.py',
               feedback.RUNTIME / 'models/visual_world_model.py']
    checkpoint = feedback.CHECKPOINT / 'checkpoints/model_latest.pth'
    receipt = dict(arm=args.arm, sample=args.sample, global_sample=global_sample, base_seed=args.seed,
                   target_chunk=target_chunk, target_index=target_index, config=config,
                   full_target_sha256=feedback.sha256(original_target),
                   checkpoint_sha256=feedback.sha256(checkpoint), encoder_sha256=feedback.DINO_SHA256,
                   implementation_sha256={str(path.relative_to(REPO)): feedback.sha256(path) for path in sources},
                   action_mean=feedback.ACTION_MEAN, action_std=feedback.ACTION_STD,
                   physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'],
                   torch=torch.__version__, gpu=torch.cuda.get_device_name())
    feedback.configure_runtime()
    (output / 'run_manifest.json').write_text(json.dumps(receipt, indent=2) + '\n')
    os.chdir(output)
    from chunk_planner import ObservedChunkPlanner
    original_plan = ObservedChunkPlanner.plan

    def record_execution(planner, *values, **kwargs):
        result = original_plan(planner, *values, **kwargs)
        # Save existing execution arrays only after planning; no extra simulation or RNG.
        physical = planner.preprocessor.denormalize_actions(result[0].detach().cpu()).numpy()
        np.savez_compressed('recorded_execution.npz',
                            obs_visual=planner.execution_observations['visual'],
                            obs_proprio=planner.execution_observations['proprio'],
                            states=planner.execution_states, physical_actions=physical)
        return result

    ObservedChunkPlanner.plan = record_execution
    logs = plan.planning_main(config)
    if feedback.sha256(checkpoint) != receipt['checkpoint_sha256']:
        raise RuntimeError('Checkpoint changed during execution.')
    with np.load('rope_metrics.npz') as metrics:
        pushes = int(metrics['actual_pushes'][0])
        if not np.isfinite(metrics['final_cd']).all() or pushes > args.max_pushes:
            raise RuntimeError('Invalid result.')
        success = bool(metrics['successes'][0])
        if success != (metrics['final_cd'][0] < .3) or (not success and pushes != args.max_pushes):
            raise RuntimeError('Invalid success stopping behavior.')
        timeline = np.r_[metrics['initial_cd'][0], metrics['cd_by_replan'][0]]
        hits = np.flatnonzero(timeline < .3)
        if len(hits) and hits[0] != pushes:
            raise RuntimeError('Continued executing after success.')
    receipt['final_logs'] = {key: float(value) for key, value in logs.items()}
    (output / 'complete.json').write_text(json.dumps(receipt, indent=2) + '\n')


if __name__ == '__main__':
    main()
