"""Frozen Rope MPC with observation-only S; runtime imports use the local CL-JEPA code.

The native CEM candidate loop is retained, with an explicit trajectory index for
episode-local memories. No particle states enter the detector or action objective.
"""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from einops import repeat
from scipy.spatial import cKDTree
from tqdm import tqdm

from planning.cem import CEMPlanner
from planning.mpc import MPCPlanner
from utils import move_to_device


def model_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def chamfer_particles(current, goal):
    """The environment's symmetric mean Euclidean CD, without quadratic storage."""
    current = np.asarray(current).reshape(-1, 4)[:, :3]
    goal = np.asarray(goal).reshape(-1, 4)[:, :3]
    return float(cKDTree(goal).query(current)[0].mean()
                 + cKDTree(current).query(goal)[0].mean())


def mean_square_distances(predictions, memories):
    # (candidates, horizon, features) against (memories, features).
    # Avoid materializing candidates x horizon x memories x DINO-patch-features.
    dim = predictions.shape[-1]
    distance = (predictions.square().mean(-1, keepdim=True)
                + memories.square().mean(-1)[None, None, :]
                - 2 * torch.matmul(predictions, memories.T) / dim)
    return distance.clamp_min(0)


def stall_penalty(predicted, memory, alpha=1.0, base=2.0):
    # One immutable observed context, followed by the predicted push sequence.
    visual = predicted['visual'][:, 1:].flatten(2)
    proprio = predicted['proprio'][:, 1:].flatten(2)
    distance = (mean_square_distances(visual, memory['visual'])
                + alpha * mean_square_distances(proprio, memory['proprio']))
    kernels = (1 - distance / memory['rho'].clamp_min(1e-12)).clamp_min(0).square()
    per_frame = (kernels * memory['weight'] * memory['goal_distance']).sum(-1)
    coefficients = base ** torch.arange(1, visual.shape[1] + 1,
                                         device=visual.device, dtype=visual.dtype)
    # Native Rope uses terminal MSE, without the all-frame objective's extra
    # division by H+1. A stationary prediction therefore costs kappa*D(m, goal).
    return (per_frame * coefficients).sum(-1) / coefficients.sum()


class ContextCEMPlanner(CEMPlanner):
    """Native sampling/order/elites; additionally exposes the current episode."""

    active_trajectory = None

    def plan(self, obs_0, obs_g, actions=None, step=None):
        trans_obs_0 = move_to_device(self.preprocessor.transform_obs(obs_0), self.device)
        trans_obs_g = move_to_device(self.preprocessor.transform_obs(obs_g), self.device)
        with torch.no_grad():
            z_obs_g = self.wm.encode_obs(trans_obs_g)
        mu, sigma = self.init_mu_sigma(obs_0, actions)
        mu, sigma = mu.to(self.device), sigma.to(self.device)
        for iteration in tqdm(range(self.opt_steps)):
            losses = []
            for trajectory in range(mu.shape[0]):
                # Reserve the same random stream for every task even after early
                # success, so another task's paired candidates do not change.
                if getattr(self, 'finished_mask', np.zeros(mu.shape[0], bool))[trajectory]:
                    torch.randn(self.num_samples, self.horizon, self.action_dim)
                    continue
                self.active_trajectory = trajectory
                current = {key: repeat(value[trajectory].unsqueeze(0), '1 ... -> n ...',
                                       n=self.num_samples)
                           for key, value in trans_obs_0.items()}
                goal = {key: repeat(value[trajectory].unsqueeze(0), '1 ... -> n ...',
                                    n=self.num_samples)
                        for key, value in z_obs_g.items()}
                sampled = (torch.randn(self.num_samples, self.horizon, self.action_dim)
                           .to(self.device) * sigma[trajectory] + mu[trajectory])
                sampled[0] = mu[trajectory]
                with torch.no_grad():
                    predicted, _ = self.wm.rollout(obs_0=current, act=sampled)
                    loss = self.objective_fn(predicted, goal, step=step)
                if not torch.isfinite(loss).all():
                    raise RuntimeError('Non-finite candidate objective.')
                elite_indices = torch.argsort(loss)[:self.topk]
                elite_actions = sampled[elite_indices]
                losses.append(loss[elite_indices[0]].item())
                mu[trajectory] = elite_actions.mean(dim=0)
                sigma[trajectory] = elite_actions.std(dim=0)
            self.wandb_run.log({f'{self.logging_prefix}/loss': np.mean(losses),
                                'step': iteration + 1})
        self.active_trajectory = None
        return mu, np.full(mu.shape[0], np.inf)


class RopeStagnationMPCPlanner(MPCPlanner):
    def __init__(self, ecp=None, **kwargs):
        super().__init__(**kwargs)
        self.ecp = dict(ecp or {})
        self.kappa = float(self.ecp.get('kappa', 0.0))
        self.window = int(self.ecp.get('window_pushes', 3))
        self.warmup = int(self.ecp.get('warmup_pushes', 5))
        self.gamma_lo = float(self.ecp.get('gamma_lo', 0.05))
        self.gamma_hi = float(self.ecp.get('gamma_hi', 0.15))
        self.beta = float(self.ecp.get('beta', 0.25))
        self.alpha = float(self.ecp.get('alpha', 1.0))
        self.base = float(self.ecp.get('base', 2.0))
        if self.resim_from_start or self.evaluator.frameskip != 1:
            raise ValueError('This pilot requires continuation and frameskip=1.')
        if not isinstance(self.sub_planner, ContextCEMPlanner):
            raise TypeError('Explicit CEM trajectory context is required.')
        self.native_objective = self.sub_planner.objective_fn
        self.sub_planner.objective_fn = self._objective

    @torch.no_grad()
    def _encode(self, observations):
        encoded = self.wm.encode_obs(move_to_device(
            self.preprocessor.transform_obs(observations), self.device))
        return {key: value.flatten(2).detach() for key, value in encoded.items()}

    def plan(self, obs_0, obs_g, actions=None):
        self.wm.eval()
        self.wm.requires_grad_(False)
        n = obs_0['visual'].shape[0]
        self.memories = [[] for _ in range(n)]
        self.first_intervention = [None] * n
        self.first_action_change = [None] * n
        self.diverged_from_native = [False] * n
        self.records = []
        self.cost_diagnostics = {}
        self.cd_by_replan = []
        self.cpu_rng_boundaries = []
        self.cuda_rng_boundaries = []
        self.prefix_reuse = []
        self.execution_observations = None
        self.execution_states = None
        self.final_states = None
        self.goal_latents = self._encode(obs_g)
        self.reference = None
        original_rollout = self.env.rollout
        if self.kappa > 0 and not self.ecp.get('independent_execution', False):
            reference_dir = Path(self.ecp['reference_dir'])
            with np.load(reference_dir / 'native_execution_cache.npz') as cache:
                self.reference = {key: cache[key] for key in cache.files}
            reference_targets = np.load(reference_dir / 'final_eval_results.npz')
            np.testing.assert_array_equal(reference_targets['state_0'], self.evaluator.state_0)
            np.testing.assert_array_equal(reference_targets['state_g'], self.evaluator.state_g)
        self.env.rollout = self._select_rollout(original_rollout)
        before = model_digest(self.wm)
        started = time.perf_counter()
        try:
            result = self._plan_actions(obs_0, obs_g, actions)
        finally:
            self.env.rollout = original_rollout
        after = model_digest(self.wm)
        if before != after or any(parameter.requires_grad for parameter in self.wm.parameters()):
            raise RuntimeError('Frozen model parameters or buffers changed.')
        if any(module.training for module in self.wm.modules()):
            raise RuntimeError('A model module left evaluation mode.')
        np.savez_compressed('rope_metrics.npz',
                            cd_by_replan=(np.stack(self.cd_by_replan, axis=1)
                                          if self.cd_by_replan else np.empty((n, 0))),
                            completed_pushes_by_replan=np.asarray([
                                record['completed_pushes'] for record in self.records]),
                            final_states=self.final_states,
                            first_intervention=np.asarray([
                                -1 if x is None else x for x in self.first_intervention]),
                            first_action_change=np.asarray([
                                -1 if x is None else x for x in self.first_action_change]))
        Path('ecp_records.json').write_text(json.dumps(self.records, indent=2) + '\n')
        Path('ecp_costs.json').write_text(json.dumps(self.cost_diagnostics, indent=2) + '\n')
        Path('prefix_reuse.json').write_text(json.dumps(self.prefix_reuse, indent=2) + '\n')
        if self.kappa == 0:
            physical_actions = self.preprocessor.denormalize_actions(result[0].detach().cpu()).numpy()
            np.savez_compressed('native_execution_cache.npz',
                                obs_visual=self.execution_observations['visual'],
                                obs_proprio=self.execution_observations['proprio'],
                                states=self.execution_states, physical_actions=physical_actions,
                                cpu_rng=(np.stack(self.cpu_rng_boundaries) if self.cpu_rng_boundaries
                                         else np.empty((0, 0), dtype=np.uint8)),
                                cuda_rng=(np.stack(self.cuda_rng_boundaries) if self.cuda_rng_boundaries
                                          else np.empty((0, 0), dtype=np.uint8)))
        Path('frozen_model_audit.json').write_text(json.dumps({
            'parameters_and_buffers_before': before,
            'parameters_and_buffers_after': after,
            'unchanged': before == after, 'all_parameters_frozen': True,
            'all_modules_eval': True, 'elapsed_s': time.perf_counter() - started,
            'ecp': self.ecp, 'particle_states_used_for_planning': False,
            'native_prefix_reuse': self.kappa > 0,
        }, indent=2) + '\n')
        return result

    def _select_rollout(self, original_rollout):
        return self._paired_env_rollout if self.reference is not None else original_rollout

    def _plan_actions(self, obs_0, obs_g, actions=None):
        return super().plan(obs_0, obs_g, actions)

    def _paired_env_rollout(self, seeds, initial_states, actions):
        """Reuse actual native execution until this episode's first intervention.

        The native MPC already resets/restores particle positions at every block.
        At intervention, the branch uses that same ordinary environment API. CD
        and particle states never determine whether a prefix is reused.
        """
        start = self.iter * self.n_taken_actions
        end = start + self.n_taken_actions + 1
        observations, states, reused = [], [], []
        unchanged_controls = []
        for index, memory in enumerate(self.memories):
            expected = self.reference['physical_actions'][index, start:end - 1]
            identical_actions = np.array_equal(actions[index], expected)
            if not memory and (self.diverged_from_native[index] or not identical_actions):
                raise RuntimeError('Unmodified native action prefix did not reproduce exactly.')
            if not self.diverged_from_native[index] and identical_actions:
                obs = {key: self.reference['obs_' + key][index, start:end].copy()
                       for key in ['visual', 'proprio']}
                state = self.reference['states'][index, start:end].copy()
                reused.append(index)
                if memory:
                    unchanged_controls.append(index)
            else:
                if not self.diverged_from_native[index]:
                    self.first_action_change[index] = self.iter
                self.diverged_from_native[index] = True
                obs, state = self.env.envs[index].rollout(seeds[index], initial_states[index], actions[index])
            observations.append(obs)
            states.append(state)
        self.prefix_reuse.append({'plan_zero_based': self.iter,
                                  'reused_samples': reused, 'reused_pushes_per_sample': self.n_taken_actions,
                                  'memory_active_but_identical_control_samples': unchanged_controls,
                                  'selection': 'same action sequence and no prior divergence; never selected by CD'})
        return {key: np.stack([obs[key] for obs in observations])
                for key in ['visual', 'proprio']}, np.stack(states)

    def _objective(self, predicted, goal, step=None):
        native = self.native_objective(predicted, goal, step=step)
        if self.kappa == 0:
            return native
        trajectory = self.sub_planner.active_trajectory
        entries = self.memories[trajectory]
        if not entries:
            return native
        memory = {key: torch.stack([entry[key] for entry in entries])
                  for key in entries[0]}
        added = self.kappa * stall_penalty(predicted, memory, self.alpha, self.base)
        key = f'plan{self.iter}_sample{trajectory}'
        diagnostic = self.cost_diagnostics.setdefault(key, {
            'candidate_batches': 0, 'positive_candidates': 0, 'maximum_added_cost': 0.0})
        diagnostic['candidate_batches'] += 1
        diagnostic['positive_candidates'] += int((added > 0).sum().item())
        diagnostic['maximum_added_cost'] = max(diagnostic['maximum_added_cost'], added.max().item())
        return native + added

    def _record_eval(self, output):
        logs, _, observations, states = output
        self.execution_observations = observations
        self.execution_states = states
        current = np.asarray(states[:, -1])
        cd = np.asarray([chamfer_particles(state, goal)
                         for state, goal in zip(current, self.evaluator.state_g)])
        np.testing.assert_allclose(cd.mean(), logs['mean_chamfer_distance'], atol=2e-6, rtol=2e-6)
        if not np.isfinite(cd).all():
            raise RuntimeError('Non-finite particle CD.')
        self.cd_by_replan.append(cd)
        self.final_states = current

    @torch.no_grad()
    def _post_env_feedback(self, taken_actions, e_obses):
        completed_pushes = e_obses['visual'].shape[1] - 1
        record = {'completed_replans': self.iter + 1,
                  'completed_pushes': completed_pushes,
                  'cd': self.cd_by_replan[-1].tolist(), 'planning_s': self.t_plan_s,
                  'detector_eligible': False}
        if completed_pushes >= max(self.warmup, self.window) and self.iter + 1 < self.max_iter:
            encoded = self._encode({key: value[:, [completed_pushes - self.window, completed_pushes]]
                                    for key, value in e_obses.items()})
            visual, proprio = encoded['visual'], encoded['proprio']
            goal_v, goal_p = self.goal_latents['visual'][:, 0], self.goal_latents['proprio'][:, 0]
            movement = ((visual[:, 1] - visual[:, 0]).square().mean(-1)
                        + self.alpha * (proprio[:, 1] - proprio[:, 0]).square().mean(-1))
            remaining = ((visual[:, 1] - goal_v).square().mean(-1)
                         + self.alpha * (proprio[:, 1] - goal_p).square().mean(-1))
            rate = (movement / remaining.clamp_min(1e-12)).sqrt()
            if not torch.isfinite(rate).all():
                raise RuntimeError('Non-finite movement trigger.')
            weights = ((self.gamma_hi - rate) / (self.gamma_hi - self.gamma_lo)).clamp(0, 1)
            weights[torch.as_tensor(self.is_success, device=weights.device)] = 0
            for index in range(len(self.memories)):
                if weights[index].item() > 0:
                    self.memories[index].append({
                        'visual': visual[index, 1], 'proprio': proprio[index, 1],
                        'weight': weights[index], 'goal_distance': remaining[index],
                        'rho': self.beta * remaining[index],
                    })
                    if self.kappa > 0 and self.first_intervention[index] is None:
                        # Zero-based index of the next plan, not the detecting plan.
                        self.first_intervention[index] = self.iter + 1
            record.update({'detector_eligible': True,
                           'window_start_push': completed_pushes - self.window,
                           'movement_ratio': rate.cpu().tolist(),
                           'trigger_weight': weights.cpu().tolist(),
                           'memory_count': [len(memory) for memory in self.memories]})
        self.records.append(record)
        if self.reference is not None and self.iter < len(self.reference['cpu_rng']):
            # Use the native candidate random stream at the same block boundary,
            # including when some tasks have already activated and others have not.
            torch.set_rng_state(torch.from_numpy(self.reference['cpu_rng'][self.iter]).clone())
            if torch.cuda.is_available():
                torch.cuda.set_rng_state(torch.from_numpy(self.reference['cuda_rng'][self.iter]).clone())
        self.cpu_rng_boundaries.append(torch.get_rng_state().cpu().numpy().copy())
        self.cuda_rng_boundaries.append(torch.cuda.get_rng_state().cpu().numpy().copy()
                                        if torch.cuda.is_available() else np.empty(0, dtype=np.uint8))
        Path('ecp_progress.json').write_text(json.dumps(record, indent=2) + '\n')
        return None
