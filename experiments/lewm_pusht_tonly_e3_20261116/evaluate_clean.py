"""Evaluate the new epoch-3 model with the archived clean PushT CEM protocol."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
RUNTIME = REPO / 'eval_outputs/rope_local/runtime'
TARGETS = REPO / 'reports/pusht_visual_v8/raw_baselines/research_snapshots/2026-11-08-pusht-gd/references/lewm/targets_T.pkl'
TEMPLATE = REPO / 'reports/pusht_visual_v8/raw_baselines/research_snapshots/2026-11-07/completed_runs/CL-JEPA-cudagraph/baselines/lewm_VIS50/blur/.hydra/config.yaml'
TARGET_SHA = '1950b39e5d3ba13b30c14a435a400d84cd3fbe30138cdfed55e98b202d76277b'
sys.path.insert(0, str(RUNTIME))
os.environ.setdefault('MPLCONFIGDIR', '/tmp/lewm-pusht-matplotlib')
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def weights_sha(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--diagnostic-checkpoint', type=Path, help='Random model for pipeline smoke test only; never formal results')
    args = parser.parse_args()
    import numpy as np
    import yaml
    import plan
    import env  # registers PushObj
    smoke = args.diagnostic_checkpoint is not None
    assert sha(TARGETS) == TARGET_SHA
    checkpoint = (args.diagnostic_checkpoint or REPO / 'checkpoints/lewm_pusht_tonly_e3_4gpu_20261116').resolve()
    model_file = checkpoint / 'checkpoints/model_3.pth'
    if not smoke:
        result = json.loads((REPO / 'eval_outputs/lewm_pusht_tonly_e3_20261116/training_result.json').read_text())
        assert result['checkpoint_sha256'] == sha(model_file)
        assert result['epoch'] == 3 and result['step'] == 38277
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cfg = yaml.safe_load(TEMPLATE.read_text())
    cfg.update(ckpt_base_path=str(checkpoint), model_epoch=3, ood_corruption=None,
               eval_data_path=str(TARGETS), saved_folder=str(output), n_plot_samples=0)
    cfg['planner']['continual']['tag'] = 'clean_epoch3'
    if smoke:
        cfg['n_evals'] = 2
        cfg['planner']['max_iter'] = 1
    assert cfg['seed'] == 100 and cfg['segments_order'] == 'file'
    assert cfg['goal_source'] == 'segments' and cfg['env_kwargs_override'] is None
    (output / 'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    sources = ['plan.py', 'models/lewm.py', 'planning/cem.py', 'planning/mpc.py', 'planning/continual.py',
               'planning/evaluator.py', 'planning/objectives.py', 'env/pusht/pusht_env.py',
               'env/pusht/pusht_wrapper.py', 'datasets/pusht_dset.py']
    source_hashes = {}
    for name in sources:
        path = RUNTIME / name
        source_hashes[name] = sha(path)
        dest = output / 'source_snapshot' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(path.read_bytes())

    class AuditedWorkspace(plan.PlanWorkspace):
        def __init__(self, *values, **kwargs):
            super().__init__(*values, **kwargs)
            assert self.frameskip == 5 and self.planner.n_taken_actions == 1
            assert self.planner.sub_planner.horizon == 5 and self.goal_H == 25
            assert self.planner.sub_planner.opt_steps == 10
            assert self.planner.sub_planner.num_samples == 200 and self.planner.sub_planner.topk == 30
            assert self.planner.max_iter == (1 if smoke else 30)
            assert self.n_evals == (2 if smoke else 50)
            assert self.planner.resim_from_start and self.planner.reuse_actions
            assert self.cfg_dict['planner']['continual']['state_in'] is None
            assert self.cfg_dict['objective'] == {'_target_': 'planning.objectives.create_objective_fn', 'alpha': 1, 'base': 2, 'mode': 'staged'}
            self.wm.requires_grad_(False)
            self.wm.eval()
            self.frozen_before = weights_sha(self.wm)

        def perform_planning(self):
            result = super().perform_planning()
            assert self.frozen_before == weights_sha(self.wm), 'Frozen model changed'
            Path('frozen_model_check.json').write_text(json.dumps({'unchanged': True, 'weights_sha256': self.frozen_before}, indent=2))
            self.env.close()
            return result

    plan.PlanWorkspace = AuditedWorkspace
    os.chdir(output)
    logs = plan.planning_main(cfg)
    with np.load('final_eval_results.npz') as result:
        successes = result['successes']
        assert len(successes) == (2 if smoke else 50)
        assert np.isfinite(result['actions']).all()
    receipt = {'status': 'complete', 'diagnostic_only': smoke, 'condition': 'clean', 'model_epoch': 0 if smoke else 3,
               'successes': int(successes.sum()), 'episodes': len(successes),
               'success_rate': float(successes.mean()), 'checkpoint_sha256': sha(model_file),
               'hydra_sha256': sha(checkpoint / 'hydra.yaml'), 'target_sha256': TARGET_SHA,
               'source_sha256': source_hashes, 'evaluation_runner_sha256': sha(Path(__file__)),
               'max_physical_steps': 5 if smoke else 150,
               'success_rule': 'joint agent/object xy distance < 20 px and shortest-angle error < pi/9',
               'goal_construction': 'replay stored segment actions; raw last state is not used as goal',
               'historical_epoch10_results_not_replaced': True}
    Path('complete.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    main()
