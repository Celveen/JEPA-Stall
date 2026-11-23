"""Regression checks for single-push feedback and paired action indexing, on CPU."""

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
sys.path[:0] = [str(ROOT / 'eval_outputs/rope_local/runtime'),
                str(ROOT / 'experiments/rope_ecp_20261110')]

import numpy as np
import torch
from analyze import analyze_pair
from rope_ecp import RopeStagnationMPCPlanner, chamfer_particles

spec = importlib.util.spec_from_file_location('rope_feedback_run', ROOT / 'experiments/rope_feedback/run.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
from feedback_planner import FeedbackRopePlanner


class FeedbackTests(unittest.TestCase):
    def test_random_or_partial_encoder_is_rejected(self):
        encoder = torch.nn.Linear(2, 2)
        model = SimpleNamespace(encoder=SimpleNamespace(base_model=encoder))
        expected = {key: value.clone() for key, value in encoder.state_dict().items()}
        self.assertTrue(runner.verify_encoder(model, expected)['matches_official_pretrained_weights'])
        with torch.no_grad():
            encoder.weight[0, 0] += 1
        with self.assertRaisesRegex(RuntimeError, 'not the official pretrained'):
            runner.verify_encoder(model, expected)
        with self.assertRaisesRegex(RuntimeError, 'keys do not match'):
            runner.verify_encoder(model, {})

    def test_warmup5_reference_allows_only_the_declared_warmup_difference(self):
        import run_warmup5 as variant
        identity = {'chunk': 0, 'target_sha256': 'same', 'checkpoint_sha256': 'same', 'encoder_sha256': 'same',
                    'action_mean': runner.ACTION_MEAN, 'action_std': runner.ACTION_STD,
                    'implementation_sha256': {'runner': 'same'}}
        previous = dict(identity, arm='native', config=runner.build_config('native', 0, Path('/tmp/native')))
        receipt = dict(identity, arm='S', config=variant.build_config('S', 0, Path('/tmp/S')))
        self.assertEqual(receipt['config']['planner']['ecp']['warmup_pushes'], 5)
        variant.validate_reference(previous, receipt)
        wrong = copy.deepcopy(receipt)
        wrong['config']['planner']['success_cd'] = .4
        with self.assertRaisesRegex(ValueError, 'different evaluation protocol'):
            variant.validate_reference(previous, wrong)

    def test_warmup5_first_memory_is_available_for_sixth_push(self):
        planner = RopeStagnationMPCPlanner.__new__(RopeStagnationMPCPlanner)
        planner.warmup, planner.window, planner.max_iter = 5, 3, 20
        planner.kappa, planner.alpha, planner.beta = 2., 1., .25
        planner.gamma_lo, planner.gamma_hi = .05, .15
        planner.memories, planner.first_intervention = [[]], [None]
        planner.is_success = np.array([False])
        planner.goal_latents = {'visual': torch.ones((1, 1, 1)), 'proprio': torch.zeros((1, 1, 1))}
        planner._encode = lambda obs: {key: torch.tensor(value) for key, value in obs.items()}
        planner.cd_by_replan, planner.t_plan_s = [np.ones(1)], 0.
        planner.records, planner.cpu_rng_boundaries, planner.cuda_rng_boundaries = [], [], []
        planner.reference = None
        with tempfile.TemporaryDirectory() as directory, patch('torch.cuda.is_available', return_value=False):
            previous = os.getcwd()
            try:
                os.chdir(directory)
                for pushes in (4, 5):
                    planner.iter = pushes - 1
                    obs = {key: np.zeros((1, pushes + 1, 1)) for key in ('visual', 'proprio')}
                    planner._post_env_feedback(None, obs)
                    self.assertEqual(len(planner.memories[0]), int(pushes == 5))
            finally:
                os.chdir(previous)
        self.assertEqual(planner.first_intervention, [5])  # Zero-based plan 5 executes push 6.

    def test_success_stops_planning_and_simulation_per_sample(self):
        settings = runner.build_config('native', 0, Path('/tmp/test'))['planner']
        for all_done in (False, True):
            with self.subTest(all_done=all_done), tempfile.TemporaryDirectory() as directory:
                state = np.zeros((4, 2, 4), dtype=float)
                if not all_done:
                    state[:, :, 0] = np.array([0., .15, .5, .5])[:, None]
                goal = np.zeros_like(state)
                obs = {'visual': np.zeros((4, 1, 1)), 'proprio': np.zeros((4, 1, 1))}
                current = [obs, state]
                calls, masks = [], []
                evaluator = SimpleNamespace(frameskip=1, state_0=state, state_g=goal,
                                            get_init_cond=lambda: current.copy())
                evaluator.assign_init_cond = lambda obs_0, state_0: current.__setitem__(slice(None), [obs_0, state_0])

                def rollout(seed, initial, actions):
                    calls.append(seed)
                    final = initial.copy()
                    if seed == 1:
                        final[:, 0] = .1
                    elif seed == 2:
                        final[:, 0] = .2 if calls.count(2) == 1 else .1
                    return ({key: np.zeros((2, 1)) for key in obs}, np.stack([initial, final]))

                env = SimpleNamespace(rollout=None, envs=[SimpleNamespace(rollout=rollout)] * 4)
                planner = FeedbackRopePlanner(
                    success_cd=.3, max_iter=3, n_taken_actions=1,
                    sub_planner=copy.deepcopy(settings['sub_planner']), wm=torch.nn.Linear(1, 1),
                    env=env, action_dim=4, objective_fn=lambda *a, **k: None,
                    preprocessor=SimpleNamespace(denormalize_actions=lambda x: x),
                    evaluator=evaluator, wandb_run=SimpleNamespace(log=lambda *a, **k: None),
                    log_filename=None, resim_from_start=False, ecp=settings['ecp'])
                planner._encode = lambda data: {key: torch.tensor(value) for key, value in data.items()}

                def plan_step(*args, **kwargs):
                    masks.append(planner.sub_planner.finished_mask.copy())
                    return torch.ones((4, 5, 4)), None

                planner.sub_planner.plan = plan_step

                def evaluate(actions, *args, **kwargs):
                    observed, states = env.rollout(range(4), current[1], actions.numpy())
                    cd = [chamfer_particles(x, y) for x, y in zip(states[:, -1], goal)]
                    return {'mean_chamfer_distance': np.mean(cd)}, np.zeros(4, bool), observed, states

                evaluator.eval_actions = evaluate
                previous = os.getcwd()
                try:
                    os.chdir(directory)
                    planner.plan(obs, obs)
                    with np.load('rope_metrics.npz') as metrics:
                        np.testing.assert_array_equal(metrics['actual_pushes'], [0, 0, 0, 0] if all_done else [0, 1, 2, 3])
                        np.testing.assert_array_equal(metrics['successes'], [True] * 4 if all_done else [True, True, True, False])
                        if not all_done:
                            self.assertEqual(metrics['initial_cd'][1], .3)  # Strict inequality.
                            np.testing.assert_array_equal(metrics['cd_by_replan'][1], [.2] * 3)
                finally:
                    os.chdir(previous)
                self.assertEqual(calls, [] if all_done else [1, 2, 3, 2, 3, 3])
                if not all_done:
                    np.testing.assert_array_equal(masks, [[True, False, False, False],
                                                         [True, True, False, False],
                                                         [True, True, True, False]])

    def test_cem_skips_finished_candidates_without_shifting_other_random_streams(self):
        from rope_ecp import ContextCEMPlanner
        model = torch.nn.Linear(1, 1)
        calls = []
        model.encode_obs = lambda data: data

        def rollout(obs_0, act):
            calls.append(act.clone())
            return act, None

        model.rollout = rollout
        planner = ContextCEMPlanner(
            horizon=2, topk=2, num_samples=4, var_scale=1, opt_steps=2, eval_every=-1,
            wm=model, action_dim=4, objective_fn=lambda pred, goal, **kw: pred.square().sum((1, 2)),
            preprocessor=SimpleNamespace(transform_obs=lambda data: data), evaluator=None,
            wandb_run=SimpleNamespace(log=lambda *a, **kw: None), log_filename=None)
        obs = {'visual': torch.zeros(2, 1, 1)}
        torch.manual_seed(19)
        full, _ = planner.plan(obs, obs)
        full_rng = torch.get_rng_state().clone()
        full_calls = calls.copy()
        calls.clear()
        planner.finished_mask = np.array([True, False])
        torch.manual_seed(19)
        stopped, _ = planner.plan(obs, obs)
        self.assertEqual(len(calls), 2)
        torch.testing.assert_close(stopped[1], full[1])
        torch.testing.assert_close(torch.get_rng_state(), full_rng)
        for actual, expected in zip(calls, full_calls[1::2]):
            torch.testing.assert_close(actual, expected)

    def test_twenty_feedback_updates_for_ten_tasks(self):
        config = runner.build_config('native', 0, Path('/tmp/rope-test'))
        self.assertEqual(config['n_evals'], 10)
        settings = config['planner']
        model = torch.nn.Linear(1, 1)
        observations = {'visual': np.zeros((10, 1, 1), np.float32),
                        'proprio': np.zeros((10, 1, 1), np.float32)}
        goals = {'visual': np.ones((10, 1, 1), np.float32),
                 'proprio': np.zeros((10, 1, 1), np.float32)}
        initial_state = np.zeros((10, 2, 4), np.float32)
        goal_state = np.ones_like(initial_state)
        evaluator = SimpleNamespace(frameskip=1, state_0=initial_state, state_g=goal_state)
        current = [observations, initial_state]
        feedback_inputs, warm_starts = [], []
        planner = FeedbackRopePlanner(
            max_iter=settings['max_iter'], n_taken_actions=settings['n_taken_actions'],
            sub_planner=copy.deepcopy(settings['sub_planner']), wm=model,
            env=SimpleNamespace(rollout=None), action_dim=4,
            objective_fn=lambda *args, **kwargs: None,
            preprocessor=SimpleNamespace(denormalize_actions=lambda x: x, normalize_actions=lambda x: x),
            evaluator=evaluator, wandb_run=SimpleNamespace(log=lambda *args, **kwargs: None),
            log_filename=None, resim_from_start=False, ecp=settings['ecp'],
        )
        planner._encode = lambda obs: {key: torch.tensor(value) for key, value in obs.items()}

        def plan_step(obs_0, obs_g, actions, step):
            np.testing.assert_allclose(obs_0['visual'], step * .001, atol=1e-7)
            feedback_inputs.append(step)
            warm_starts.append(actions)
            if step > 0:
                torch.testing.assert_close(actions, torch.full((10, 4, 4), float(step)))
            return torch.full((10, 5, 4), float(step + 1)), np.full(10, np.inf)

        def assign(obs_0, state_0):
            current[:] = [obs_0, state_0]

        def evaluate(actions, *args, **kwargs):
            self.assertEqual(tuple(actions.shape), (10, 1, 4))
            new_obs = {'visual': np.concatenate([current[0]['visual'], current[0]['visual'] + .001], axis=1),
                       'proprio': np.zeros((10, 2, 1), np.float32)}
            states = np.stack([current[1], current[1]], axis=1)
            cd = chamfer_particles(states[0, -1], goal_state[0])
            return {'mean_chamfer_distance': cd}, np.zeros(10, bool), new_obs, states

        planner.sub_planner.plan = plan_step
        evaluator.get_init_cond = lambda: current.copy()
        evaluator.assign_init_cond = assign
        evaluator.eval_actions = evaluate
        with tempfile.TemporaryDirectory() as directory, patch('torch.cuda.is_available', return_value=False):
            previous = os.getcwd()
            try:
                os.chdir(directory)
                actions, _ = planner.plan(observations, goals)
                with np.load('rope_metrics.npz') as metrics:
                    self.assertEqual(metrics['cd_by_replan'].shape, (10, 20))
                    np.testing.assert_array_equal(metrics['completed_pushes_by_replan'], np.arange(1, 21))
            finally:
                os.chdir(previous)
        self.assertEqual(tuple(actions.shape), (10, 20, 4))
        self.assertEqual(feedback_inputs, list(range(20)))
        self.assertIsNone(warm_starts[0])
        self.assertFalse(planner.records[1]['detector_eligible'])
        self.assertEqual(planner.records[2]['window_start_push'], 0)
        self.assertEqual(planner.records[2]['trigger_weight'], [1.] * 10)
        self.assertEqual(planner.records[3]['memory_count'], [2] * 10)

    def test_reference_rejects_old_cadence_and_budget(self):
        identity = {'chunk': 0, 'target_sha256': 'same', 'checkpoint_sha256': 'same', 'encoder_sha256': 'same',
                    'action_mean': runner.ACTION_MEAN, 'action_std': runner.ACTION_STD,
                    'implementation_sha256': {'runner': 'same'}}
        receipt = dict(identity, arm='S', config=runner.build_config('S', 0, Path('/tmp/S')))
        previous = dict(identity, arm='native', config=runner.build_config('native', 0, Path('/tmp/native')))
        runner.validate_reference(previous, receipt)
        for key, value in [('n_taken_actions', 5), ('max_iter', 3)]:
            wrong = copy.deepcopy(previous)
            wrong['config']['planner'][key] = value
            with self.assertRaisesRegex(ValueError, 'different evaluation protocol'):
                runner.validate_reference(wrong, receipt)

    def test_single_push_prefix_reuse_and_branch(self):
        planner = RopeStagnationMPCPlanner.__new__(RopeStagnationMPCPlanner)
        planner.iter, planner.n_taken_actions = 4, 1
        planner.memories = [[] for _ in range(10)]
        planner.memories[1] = [object()]
        planner.prefix_reuse = []
        planner.diverged_from_native = [False] * 10
        planner.first_action_change = [None] * 10
        sequence = np.broadcast_to(np.arange(21)[None, :, None], (10, 21, 1)).copy()
        planner.reference = {'physical_actions': np.zeros((10, 20, 4)),
                             'obs_visual': sequence, 'obs_proprio': sequence,
                             'states': np.zeros((10, 21, 2, 4))}
        calls = []

        def execute(seed, state, actions):
            calls.append((seed, actions.shape))
            return {'visual': np.full((2, 1), 9), 'proprio': np.full((2, 1), 9)}, np.zeros((2, 2, 4))

        planner.env = SimpleNamespace(envs=[SimpleNamespace(rollout=execute)] * 10)
        actions = np.zeros((10, 1, 4))
        actions[1] = 1
        observations, _ = planner._paired_env_rollout(list(range(10)), np.zeros((10, 2, 4)), actions)
        np.testing.assert_array_equal(observations['visual'][0, :, 0], [4, 5])
        self.assertEqual(calls, [(1, (1, 4))])
        self.assertEqual(planner.first_action_change[1], 4)
        self.assertEqual(planner.prefix_reuse[0]['reused_pushes_per_sample'], 1)

    def test_S_prefix_reuse_also_skips_successful_samples(self):
        planner = FeedbackRopePlanner.__new__(FeedbackRopePlanner)
        planner.iter = 4
        planner.is_success = np.array([True, False])
        planner.memories = [[], [object()]]
        planner.prefix_reuse = []
        planner.diverged_from_native = [False, False]
        planner.first_action_change = [None, None]
        planner.reference = {'physical_actions': np.zeros((2, 20, 4)),
                             'obs_visual': np.zeros((2, 21, 1)),
                             'obs_proprio': np.zeros((2, 21, 1)),
                             'states': np.zeros((2, 21, 2, 4))}
        current_obs = {'visual': np.full((2, 1, 1), 7.), 'proprio': np.zeros((2, 1, 1))}
        current_states = np.full((2, 2, 4), 3.)
        planner.evaluator = SimpleNamespace(get_init_cond=lambda: (current_obs, current_states))
        calls = []

        def rollout(seed, initial, actions):
            calls.append(seed)
            return {key: np.zeros((2, 1)) for key in current_obs}, np.zeros((2, 2, 4))

        planner.env = SimpleNamespace(envs=[SimpleNamespace(rollout=rollout)] * 2)
        observed, states = planner._feedback_rollout([0, 1], current_states, np.ones((2, 1, 4)))
        self.assertEqual(calls, [1])
        np.testing.assert_array_equal(observed['visual'][0], 7.)
        np.testing.assert_array_equal(states[0], 3.)
        self.assertEqual(planner.prefix_reuse[0]['stopped_samples'], [0])
        self.assertEqual(planner.first_action_change, [None, 4])

    def test_paired_report_uses_actual_push_cadence(self):
        for cadence in (1, 5):
            with self.subTest(cadence=cadence), tempfile.TemporaryDirectory() as directory:
                native, intervention = Path(directory) / 'native', Path(directory) / 'S'
                for path in (native, intervention):
                    path.mkdir()
                    receipt = {'checkpoint_sha256': 'same', 'target_sha256': 'same',
                               'config': {'planner': {'n_taken_actions': cadence}}}
                    (path / 'complete.json').write_text(json.dumps(receipt))
                    (path / 'frozen_model_audit.json').write_text(json.dumps({
                        'unchanged': True, 'parameters_and_buffers_before': 'same'}))
                    actions = np.zeros((10, 20 * cadence, 4))
                    if path == intervention:
                        actions[:, 3 * cadence:] = 1
                    np.savez(path / 'final_eval_results.npz', actions=actions,
                             state_0=np.zeros((10, 2, 4)), state_g=np.ones((10, 2, 4)))
                    np.savez(path / 'rope_metrics.npz', cd_by_replan=np.ones((10, 20)),
                             first_intervention=np.full(10, 3), first_action_change=np.full(10, 3))
                rows = analyze_pair(native, intervention, 0)
                self.assertEqual(len(rows), 10)
                self.assertEqual(rows[0]['pre_intervention_actions_checked'], 3 * cadence)


if __name__ == '__main__':
    unittest.main()
