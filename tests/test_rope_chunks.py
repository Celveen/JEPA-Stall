"""Chunk commitment, uninterrupted physics, and fixed/dynamic S-strength checks."""

import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('feedback_run', ROOT / 'experiments/rope_feedback/run.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
sys.path.insert(0, str(ROOT / 'experiments/rope_chunk3'))
import numpy as np
import torch
from chunk_planner import ObservedChunkPlanner
from rope_ecp import chamfer_particles


class ChunkTests(unittest.TestCase):
    def test_chunk_physics_and_mid_chunk_success(self):
        for chunk, limit, finish in ((3, 20, None), (5, 20, None), (3, 20, 2),
                                     (3, 15, None), (3, 15, 14)):
            with self.subTest(chunk=chunk, limit=limit, finish=finish), tempfile.TemporaryDirectory() as directory:
                settings = runner.build_config('native', 0, Path(directory))['planner']
                settings['sub_planner']['horizon'] = chunk
                settings['ecp']['warmup_pushes'] = 5
                obs = {'visual': np.zeros((1, 1, 1)), 'proprio': np.zeros((1, 1, 1))}
                # Archived targets are flat; the live simulator returns N x 4 particles.
                state = np.ones((1, 8))
                goal = np.zeros((1, 2, 4))
                current = [obs, state]
                simulated, prepared, queries = [], [], []
                evaluator = SimpleNamespace(frameskip=1, state_0=state, state_g=goal,
                                             get_init_cond=lambda: current.copy())
                evaluator.assign_init_cond = lambda obs_0, state_0: current.__setitem__(slice(None), [obs_0, state_0])

                def prepare(seed, initial):
                    prepared.append(len(simulated))
                    return {key: value[0, -1] for key, value in current[0].items()}, {}

                def step_multiple(actions):
                    simulated.append(actions.copy())
                    after = {key: value[0].copy() for key, value in current[0].items()}
                    after['visual'] += .001
                    next_state = np.zeros_like(goal) if len(simulated) == finish else state.reshape(1, 2, 4).copy()
                    return after, 0, False, {'state': next_state}

                env = SimpleNamespace(rollout=None, envs=[SimpleNamespace(prepare=prepare, step_multiple=step_multiple)])
                planner = ObservedChunkPlanner(
                    chunk_size=chunk, episode_seed=298, planning_seed=102,
                    max_iter=limit, n_taken_actions=1,
                    sub_planner=copy.deepcopy(settings['sub_planner']), wm=torch.nn.Linear(1, 1),
                    env=env, action_dim=4, objective_fn=lambda *a, **k: None,
                    preprocessor=SimpleNamespace(denormalize_actions=lambda x: x),
                    evaluator=evaluator, wandb_run=SimpleNamespace(log=lambda *a, **k: None),
                    log_filename=None, resim_from_start=False, ecp=settings['ecp'])
                planner._encode = lambda data: {key: torch.tensor(value) for key, value in data.items()}

                def optimize(obs_0, obs_g, actions, step):
                    self.assertIsNone(actions)
                    np.testing.assert_allclose(obs_0['visual'], .001 * step)
                    queries.append(step)
                    return torch.full((1, chunk, 4), float(step + 1)), None

                planner._optimize = optimize

                def evaluate(actions, *args, **kwargs):
                    observed, states = env.rollout(evaluator.seed, current[1], actions.numpy())
                    cd = chamfer_particles(states[0, -1], goal[0])
                    return {'mean_chamfer_distance': cd}, np.zeros(1, bool), observed, states

                evaluator.eval_actions = evaluate
                previous = os.getcwd()
                try:
                    os.chdir(directory)
                    with patch('torch.cuda.is_available', return_value=False):
                        planner.plan(obs, obs)
                    audit = json.loads(Path('chunk_audit.json').read_text())
                finally:
                    os.chdir(previous)
                pushes = finish or limit
                self.assertEqual(len(simulated), pushes)
                self.assertEqual(queries, list(range(0, pushes, chunk)))
                self.assertEqual(prepared, queries)
                self.assertEqual(len(planner.records), pushes)
                self.assertEqual(audit['optimizer_calls'], len(queries))
                self.assertEqual(evaluator.seed, [298])

    def test_fixed_and_dynamic_share_trigger_but_use_different_strength(self):
        records = []
        for mode in ('fixed', 'dynamic'):
            planner = ObservedChunkPlanner.__new__(ObservedChunkPlanner)
            planner.warmup, planner.window, planner.max_iter = 5, 3, 20
            planner.chunk_size, planner.strength_mode = 3, mode
            planner.kappa, planner.alpha, planner.beta = 2., 1., .25
            planner.gamma_lo, planner.gamma_hi = .05, .15
            planner.memories, planner.first_intervention = [[]], [None]
            planner._applied_at = None
            planner.is_success = np.array([False])
            planner.goal_latents = {'visual': torch.ones((1, 1, 1)), 'proprio': torch.zeros((1, 1, 1))}
            planner._encode = lambda obs: {key: torch.tensor(value) for key, value in obs.items()}
            planner.cd_by_replan, planner.t_plan_s = [np.ones(1)], 0.
            planner.records, planner.cpu_rng_boundaries, planner.cuda_rng_boundaries = [], [], []
            planner.reference = None
            planner.iter = 4
            obs = {key: np.zeros((1, 6, 1)) for key in ('visual', 'proprio')}
            obs['visual'][0, 2] = -.1  # movement/remaining ratio = .1, severity = .5
            with tempfile.TemporaryDirectory() as directory, patch('torch.cuda.is_available', return_value=False):
                previous = os.getcwd()
                try:
                    os.chdir(directory)
                    planner._post_env_feedback(None, obs)
                finally:
                    os.chdir(previous)
            self.assertEqual(len(planner.memories[0]), 1)
            self.assertIsNone(planner.first_intervention[0])
            self.assertEqual(planner.records[0]['next_chunk_push_one_based'], 7)
            records.append(planner.records[0])
        self.assertAlmostEqual(records[0]['severity_weight'], records[1]['severity_weight'])
        self.assertAlmostEqual(records[0]['new_memory_strength'], 2.)
        self.assertAlmostEqual(records[1]['new_memory_strength'], 1.)


if __name__ == '__main__':
    unittest.main()
