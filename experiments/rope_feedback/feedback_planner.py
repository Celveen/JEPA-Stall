"""Single-push Rope feedback with absorbing success states and paired execution."""

import json
from pathlib import Path
import time

import numpy as np
import torch

from rope_ecp import RopeStagnationMPCPlanner, chamfer_particles


class FeedbackRopePlanner(RopeStagnationMPCPlanner):
    def __init__(self, success_cd=0.3, **kwargs):
        super().__init__(**kwargs)
        self.success_cd = float(success_cd)
        if not np.isfinite(self.success_cd) or self.success_cd <= 0:
            raise ValueError('success_cd must be finite and positive.')
        if self.n_taken_actions != 1:
            raise ValueError('Success must be checked after every individual push.')

    def _select_rollout(self, original_rollout):
        return self._feedback_rollout

    def _feedback_rollout(self, seeds, initial_states, actions):
        """Never send even a zero absolute-position action to a finished task."""
        observations, states, reused, identical, stopped = [], [], [], [], []
        start, end = self.iter, self.iter + 2
        current_obs, _ = self.evaluator.get_init_cond()
        for index, memory in enumerate(self.memories):
            if self.is_success[index]:
                obs = {key: np.repeat(value[index, -1:], 2, axis=0)
                       for key, value in current_obs.items()}
                state = np.repeat(np.asarray(initial_states[index])[None], 2, axis=0)
                stopped.append(index)
            elif self.reference is None:
                obs, state = self.env.envs[index].rollout(seeds[index], initial_states[index], actions[index])
            else:
                expected = self.reference['physical_actions'][index, start:end - 1]
                same = np.array_equal(actions[index], expected)
                if not memory and (self.diverged_from_native[index] or not same):
                    raise RuntimeError('Unmodified Native action prefix did not reproduce exactly.')
                if not self.diverged_from_native[index] and same:
                    obs = {key: self.reference['obs_' + key][index, start:end].copy()
                           for key in ('visual', 'proprio')}
                    state = self.reference['states'][index, start:end].copy()
                    reused.append(index)
                    if memory:
                        identical.append(index)
                else:
                    if not self.diverged_from_native[index]:
                        self.first_action_change[index] = self.iter
                    self.diverged_from_native[index] = True
                    obs, state = self.env.envs[index].rollout(seeds[index], initial_states[index], actions[index])
            observations.append(obs)
            states.append(state)
        self.prefix_reuse.append({
            'plan_zero_based': self.iter, 'reused_samples': reused,
            'reused_pushes_per_sample': 1, 'stopped_samples': stopped,
            'memory_active_but_identical_control_samples': identical,
            'selection': 'exact controls and no prior divergence; successful samples never execute',
        })
        return ({key: np.stack([obs[key] for obs in observations]) for key in observations[0]},
                np.stack(states))

    def _plan_actions(self, obs_0, obs_g, actions=None):
        n = len(obs_0['visual'])
        initial_obs, initial_states = self.evaluator.get_init_cond()
        self.initial_cd = np.asarray([chamfer_particles(state, goal)
                                      for state, goal in zip(initial_states, self.evaluator.state_g)])
        if not np.isfinite(self.initial_cd).all():
            raise RuntimeError('Non-finite initial CD.')
        self.is_success = self.initial_cd < self.success_cd
        self.action_len = np.where(self.is_success, 0., np.inf)
        self.push_counts = np.zeros(n, dtype=int)
        self.executed_masks = []
        self.iter, self.planned_actions = 0, []
        self.execution_observations = {key: value.copy() for key, value in obs_0.items()}
        self.execution_states = np.asarray(initial_states)[:, None].copy()
        self.final_states = np.asarray(initial_states).copy()
        current_obs, memo = obs_0, None
        final_logs = {'mean_chamfer_distance': float(self.initial_cd.mean())}
        while not self.is_success.all() and self.iter < self.max_iter:
            active = ~self.is_success.copy()
            self.sub_planner.finished_mask = self.is_success.copy()
            self.sub_planner.logging_prefix = f'plan_{self.iter}'
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            started = time.perf_counter()
            proposed, _ = self.sub_planner.plan(current_obs, obs_g, actions=memo, step=self.iter)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.t_plan_s = time.perf_counter() - started
            taken = proposed.detach()[:, :1].clone()
            # Padding is only a storage convention; _feedback_rollout skips it.
            taken[torch.as_tensor(~active, device=taken.device)] = 0
            memo = proposed.detach()[:, 1:] if self.reuse_actions else None
            self.planned_actions.append(taken)
            self.executed_masks.append(active)
            print(f'MPC iter {self.iter}: executing {int(active.sum())} active samples', flush=True)
            logs, _, observed, states = self.evaluator.eval_actions(
                taken, None, filename=f'plan{self.iter}', save_video=False)
            if self.iter:
                observed = {key: np.concatenate([self.execution_observations[key], value[:, 1:]], axis=1)
                            for key, value in observed.items()}
                states = np.concatenate([self.execution_states, states[:, 1:]], axis=1)
            self._record_eval((logs, None, observed, states))
            self.push_counts += active
            new_success = active & (self.cd_by_replan[-1] < self.success_cd)
            self.is_success |= new_success
            self.action_len[new_success] = self.push_counts[new_success]
            logs['success_rate'] = float(self.is_success.mean())
            final_logs = dict(logs)
            self._post_env_feedback(taken, observed)
            self.records[-1].update(success=self.is_success.tolist(),
                                    actual_pushes=self.push_counts.tolist(),
                                    newly_successful=np.flatnonzero(new_success).tolist())
            Path('ecp_progress.json').write_text(json.dumps(self.records[-1], indent=2) + '\n')
            self.dump_logs({**logs, 'step': self.iter + 1})
            print(f'CD={self.cd_by_replan[-1].tolist()}; success={self.is_success.tolist()}', flush=True)
            current_obs = {key: value[:, -1:] for key, value in observed.items()}
            self.evaluator.assign_init_cond(obs_0=current_obs, state_0=states[:, -1])
            self.iter += 1
        self.evaluator.assign_init_cond(obs_0=initial_obs, state_0=initial_states)
        final_logs['success_rate'] = float(self.is_success.mean())
        self.final_eval = (final_logs, self.is_success.copy())
        return (torch.cat(self.planned_actions, dim=1) if self.planned_actions
                else torch.empty((n, 0, self.action_dim), device=self.device)), self.action_len

    def plan(self, obs_0, obs_g, actions=None):
        result = super().plan(obs_0, obs_g, actions)
        with np.load('rope_metrics.npz') as archive:
            metrics = dict(archive)
        metrics.update(initial_cd=self.initial_cd, success_cd=self.success_cd,
                       successes=self.is_success, actual_pushes=self.push_counts,
                       executed_mask=(np.stack(self.executed_masks, axis=1)
                                      if self.executed_masks else np.empty((len(self.is_success), 0), bool)),
                       final_cd=(self.cd_by_replan[-1] if self.cd_by_replan else self.initial_cd))
        np.savez_compressed('rope_metrics.npz', **metrics)
        return result
