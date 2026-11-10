"""CPU checks for CEM parity, objective scale, episode isolation and physical time."""

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

sys.path.insert(0, '/data/zjc/adaJEPA/CL-JEPA-cudagraph')
import numpy as np
import torch
from planning.cem import CEMPlanner
from planning.objectives import create_objective_fn
from rope_ecp import ContextCEMPlanner, RopeStagnationMPCPlanner, mean_square_distances, stall_penalty


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def encode_obs(self, obs):
        return obs

    def rollout(self, obs_0, act):
        visual = torch.cat([obs_0['visual'], act.cumsum(1)], dim=1)
        return {'visual': visual, 'proprio': visual[..., :1] * 0}, None


def check():
    objective = create_objective_fn(alpha=1, base=2, mode='last')
    common = dict(horizon=5, topk=4, num_samples=12, var_scale=1, opt_steps=3,
                  eval_every=-1, wm=DummyModel(), action_dim=4,
                  objective_fn=objective,
                  preprocessor=SimpleNamespace(transform_obs=lambda x: x),
                  evaluator=None, wandb_run=SimpleNamespace(log=lambda *a, **kw: None),
                  log_filename=None)
    obs = {'visual': torch.zeros(2, 1, 4), 'proprio': torch.zeros(2, 1, 1)}
    goal = {'visual': torch.ones(2, 1, 4), 'proprio': torch.zeros(2, 1, 1)}
    native = CEMPlanner(**common)
    contextual = ContextCEMPlanner(**common)
    torch.manual_seed(727)
    expected, _ = native.plan(obs, goal)
    torch.manual_seed(727)
    actual, _ = contextual.plan(obs, goal)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    torch.manual_seed(117)
    predictions, memories = torch.randn(7, 5, 17), torch.randn(3, 17)
    brute = (predictions[:, :, None] - memories[None, None]).square().mean(-1)
    efficient = mean_square_distances(predictions, memories)
    torch.testing.assert_close(efficient, brute, rtol=2e-6, atol=2e-6)
    memory = {'visual': torch.tensor([[1., 2., 3.]]), 'proprio': torch.zeros(1, 1),
              'weight': torch.ones(1), 'goal_distance': torch.tensor([7.]),
              'rho': torch.tensor([1.75])}
    stationary = {'visual': memory['visual'][None].repeat(2, 6, 1),
                  'proprio': torch.zeros(2, 6, 1)}
    torch.testing.assert_close(stall_penalty(stationary, memory), torch.full((2,), 7.))
    changed_context = {k: v.clone() for k, v in stationary.items()}
    changed_context['visual'][:, 0] += 100
    torch.testing.assert_close(stall_penalty(changed_context, memory), stall_penalty(stationary, memory))
    at_goal = {'visual': torch.tensor([5., 0., 2.])[None, None].repeat(2, 6, 1),
               'proprio': torch.zeros(2, 6, 1)}
    assert torch.equal(stall_penalty(at_goal, memory), torch.zeros(2))

    planner = RopeStagnationMPCPlanner.__new__(RopeStagnationMPCPlanner)
    planner.native_objective = lambda p, g, step=None: torch.zeros(p['visual'].shape[0])
    planner.sub_planner = SimpleNamespace(active_trajectory=1)
    planner.memories = [[{k: v[0] for k, v in memory.items()}], []]
    planner.kappa, planner.alpha, planner.base, planner.iter = 2., 1., 2., 1
    planner.cost_diagnostics = {}
    assert torch.equal(planner._objective(stationary, None), torch.zeros(2))
    planner.sub_planner.active_trajectory = 0
    torch.testing.assert_close(planner._objective(stationary, None), torch.full((2,), 14.))
    planner.kappa = 0
    assert torch.equal(planner._objective(stationary, None), torch.zeros(2))

    planner.kappa, planner.iter, planner.max_iter = 2., 0, 3
    planner.window, planner.warmup = 3, 5
    planner.gamma_lo, planner.gamma_hi, planner.beta = .05, .15, .25
    planner.memories, planner.first_intervention, planner.records = [[]], [None], []
    planner.cd_by_replan, planner.t_plan_s = [np.asarray([.9])], .0
    planner.reference = None
    planner.cpu_rng_boundaries, planner.cuda_rng_boundaries = [], []
    planner.goal_latents = {'visual': torch.tensor([[[5.]]]), 'proprio': torch.zeros(1, 1, 1)}
    planner._encode = lambda x: {k: torch.tensor(v) for k, v in x.items()}
    observations = {'visual': np.ones((1, 6, 1), dtype=np.float32),
                    'proprio': np.zeros((1, 6, 1), dtype=np.float32)}
    with tempfile.TemporaryDirectory() as directory:
        previous = os.getcwd()
        try:
            os.chdir(directory)
            planner._post_env_feedback(None, observations)
        finally:
            os.chdir(previous)
    assert planner.first_intervention == [1]
    assert planner.records[0]['window_start_push'] == 2
    assert planner.records[0]['completed_pushes'] == 5
    assert planner.records[0]['trigger_weight'] == [1.]
    # A native prefix must be reused by activation state alone. A second task
    # with a memory executes its branch, while the untouched task stays exact.
    planner.iter, planner.n_taken_actions = 0, 5
    planner.prefix_reuse = []
    planner.memories = [[], [object()]]
    planner.first_action_change, planner.diverged_from_native = [None, None], [False, False]
    calls = []
    def executed(seed, initial, actions):
        calls.append(seed)
        return {'visual': np.full((6, 1), 7.), 'proprio': np.zeros((6, 1))}, np.full((6, 2, 4), 7.)
    planner.env = SimpleNamespace(envs=[SimpleNamespace(rollout=executed)] * 2)
    planner.reference = {'physical_actions': np.zeros((2, 5, 4)),
                         'obs_visual': np.ones((2, 6, 1)),
                         'obs_proprio': np.zeros((2, 6, 1)),
                         'states': np.ones((2, 6, 2, 4))}
    selected_actions = np.zeros((2, 5, 4))
    selected_actions[1] = 1.
    observations, states = planner._paired_env_rollout([1, 100], np.zeros((2, 8)), selected_actions)
    assert calls == [100]
    np.testing.assert_array_equal(observations['visual'][0], np.ones((6, 1)))
    np.testing.assert_array_equal(states[0], np.ones((6, 2, 4)))
    assert (states[1] == 7.).all() and planner.prefix_reuse[0]['reused_samples'] == [0]
    planner.diverged_from_native = [False, False]
    planner.first_action_change = [None, None]
    observations, states = planner._paired_env_rollout([1, 100], np.zeros((2, 8)), np.zeros((2, 5, 4)))
    assert calls == [100]  # No additional simulation when memory leaves actions unchanged.
    assert planner.prefix_reuse[-1]['memory_active_but_identical_control_samples'] == [1]
    assert planner.first_action_change == [None, None]
    return {'passed': True, 'native_cem_actions_bitwise_identical': True,
            'distance_matches_broadcast_reference': True, 'terminal_scale_matches_goal_cost': True,
            'goal_unpenalized': True, 'fixed_context_excluded': True,
            'episode_memories_isolated': True, 'zero_kappa_exact_native': True,
            'five_push_warmup_three_push_window': True,
            'native_prefix_reuse_and_intervention_branch_isolated': True,
            'no_action_change_cannot_create_solver_noise_gain': True,
            'new_simulator_episodes': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = check()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
