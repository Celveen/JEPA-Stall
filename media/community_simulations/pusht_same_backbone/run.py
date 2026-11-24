"""Record paired, frozen-LeWM color-shift evaluations using the archived GD recipe."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'reports/lewm_contact_schedule_20261113/lewm_ecp_045_contact_objective/source'
TEMPLATE = SOURCE.parent / 'full/T/config.yaml'
TARGETS = ROOT / 'reports/pusht_visual_v8/raw_baselines/research_snapshots/2026-11-08-pusht-gd/references/lewm/targets_T.pkl'
CHECKPOINT = ROOT / 'checkpoints/lewm_pusht_tonly_e3_4gpu_20261116'
sys.path.insert(0, str(SOURCE))
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/lewm-pusht-matplotlib')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_sha(model):
    result = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        result.update(name.encode())
        result.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--condition', choices=['red_goal', 'red_agent', 'red_object'], required=True)
    parser.add_argument('--arm', choices=['native', 'ours'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--n-evals', type=int, default=50)
    parser.add_argument('--smoke', action='store_true', help='One replan only; not a result')
    args = parser.parse_args()
    import numpy as np
    import torch
    import yaml
    import plan
    import env  # Register the genuine PushT simulator.
    assert torch.cuda.is_available(), 'GPU required for this evaluation'
    torch.set_num_threads(1)
    assert sha(TARGETS) == '1950b39e5d3ba13b30c14a435a400d84cd3fbe30138cdfed55e98b202d76277b'
    assert sha(CHECKPOINT / 'checkpoints/model_3.pth') == '9e2b8afee10e46f3b7ae2bd5a9b67ebc28e86af0ccc0d0d293e2fd77471d2576'
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cfg = yaml.safe_load(TEMPLATE.read_text())
    cfg.pop('audit', None)
    cfg.update(ckpt_base_path=str(CHECKPOINT), model_epoch=3, seed=100,
               n_evals=args.n_evals, eval_data_path=str(TARGETS), saved_folder=str(output),
               env_kwargs_override={dict(red_goal='goal_color', red_agent='agent_color', red_object='color')[args.condition]: 'Red'})
    cfg['planner']['continual']['tag'] = args.condition
    if args.arm == 'native':
        cfg['planner']['_target_'] = 'planning.continual.SlowFrozenMPCPlanner'
        for name in ['stag', 'bias', 'proprio', 'contact', 'burst', 'contact_objective']:
            cfg['planner'].pop(name)
    if args.smoke:
        cfg['planner']['max_iter'] = 1
    (output / 'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    source_hashes = {str(p.relative_to(SOURCE)): sha(p) for p in sorted(SOURCE.rglob('*.py'))}
    (output / 'source_sha256.json').write_text(json.dumps(source_hashes, indent=2) + '\n')

    class RecordedWorkspace(plan.PlanWorkspace):
        def __init__(self, *values, **kwargs):
            super().__init__(*values, **kwargs)
            assert self.frameskip == 5 and self.planner.n_taken_actions == 1
            assert self.planner.sub_planner.horizon == 5
            assert self.planner.sub_planner.opt_steps == 100
            assert self.planner.resim_from_start and self.planner.reuse_actions
            assert self.cfg_dict['planner']['continual']['state_in'] is None
            self.wm.requires_grad_(False).eval()
            self.frozen_before = model_sha(self.wm)
            original_eval = self.evaluator.eval_actions

            def record(actions, action_len=None, filename='output', save_video=False):
                result = original_eval(actions, action_len, filename, save_video)
                if filename == 'output_final':
                    _, successes, observations, states = result
                    raw = observations['visual']
                    assert raw.dtype == np.uint8 and raw.shape[-1] == 3
                    controls = actions.detach().cpu().reshape(len(raw), -1, 2)
                    controls = self.data_preprocessor.denormalize_actions(controls).numpy()
                    np.savez_compressed('recorded_execution.npz', rgb=raw,
                                        states=states, proprio=observations['proprio'],
                                        controls=controls, successes=successes,
                                        action_len=action_len, goal_rgb=self.obs_g['visual'],
                                        state_0=self.state_0, state_g=self.state_g)
                return result

            self.evaluator.eval_actions = record

        def perform_planning(self):
            result = super().perform_planning()
            assert self.frozen_before == model_sha(self.wm), 'Model changed'
            Path('frozen_model_check.json').write_text(json.dumps(
                {'unchanged': True, 'weights_sha256': self.frozen_before}, indent=2))
            self.env.close()
            return result

    plan.PlanWorkspace = RecordedWorkspace
    os.chdir(output)
    start = time.monotonic()
    plan.planning_main(cfg)
    with np.load('final_eval_results.npz') as result:
        successes = result['successes']
        assert len(successes) == args.n_evals and np.isfinite(result['actions']).all()
    receipt = dict(status='complete', condition=args.condition, arm=args.arm,
                   diagnostic_only=args.smoke, successes=int(successes.sum()), episodes=args.n_evals,
                   checkpoint_sha256=sha(CHECKPOINT / 'checkpoints/model_3.pth'),
                   target_sha256=sha(TARGETS), runner_sha256=sha(Path(__file__)),
                   target_indices=list(range(args.n_evals)), seed=100, planner='GD100',
                   wall_seconds=time.monotonic() - start,
                   success_rule='joint agent/object xy distance <20px AND wrapped angle error <pi/9',
                   capture='unmodified RGB/states from the final evaluated simulator rollout; resim_from_start protocol',
                   method='off' if args.arm == 'native' else 'latent stagnation memory + conditional analytic pusher objective + contact objective (observed object pose and known T geometry)')
    Path('complete.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
