"""Execute a complete action chunk while observing and checking success after each push."""

import json
from pathlib import Path

import numpy as np
import torch

from feedback_planner import FeedbackRopePlanner


class ObservedChunkPlanner(FeedbackRopePlanner):
    def __init__(self, chunk_size=3, strength_mode='dynamic', episode_seed=1,
                 planning_seed=99, **kwargs):
        super().__init__(**kwargs)
        self.chunk_size = int(chunk_size)
        self.strength_mode = strength_mode
        self.planning_seed = int(planning_seed)
        if self.chunk_size not in (3, 5) or strength_mode not in ('fixed', 'dynamic'):
            raise ValueError('Expected chunk size 3/5 and fixed/dynamic S strength.')
        self.evaluator.seed = [int(episode_seed)]
        self._optimize = self.sub_planner.plan
        self.sub_planner.plan = self._query_chunk

    def _select_rollout(self, original_rollout):
        return self._continuous_chunk_rollout

    def _plan_actions(self, obs_0, obs_g, actions=None):
        if len(obs_0['visual']) != 1:
            raise ValueError('Each process owns exactly one Rope episode and its live simulator.')
        torch.manual_seed(self.planning_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.planning_seed)
        self.chunk_queries = []
        self.branch_push = None
        self._applied_at = None
        return super()._plan_actions(obs_0, obs_g, actions)

    def _query_chunk(self, obs_0, obs_g, actions=None, step=None):
        if step % self.chunk_size:
            if actions is None or actions.shape[1] == 0:
                raise RuntimeError('The unexecuted chunk tail was lost.')
            return actions, np.full(1, np.inf)
        proposed, lengths = self._optimize(obs_0, obs_g, actions=None, step=step)
        self.chunk_queries.append(step)
        if self.kappa > 0 and self.memories[0] and self._applied_at is None:
            self._applied_at = step
        self.first_intervention[0] = self._applied_at
        if self.reference is not None and not self.diverged_from_native[0]:
            end = min(step + self.chunk_size, self.max_iter,
                      self.reference['physical_actions'].shape[1])
            expected = self.reference['physical_actions'][0, step:end]
            actual = self.preprocessor.denormalize_actions(proposed.detach().cpu()).numpy()[0, :end - step]
            if not len(expected):
                raise RuntimeError('An unchanged task continued beyond the completed Native episode.')
            if not np.array_equal(actual, expected):
                if not self.memories[0]:
                    raise RuntimeError('A chunk changed before any S memory became available.')
                self.diverged_from_native[0] = True
                self.branch_push = step
        return proposed, lengths

    def _continuous_chunk_rollout(self, seeds, initial_states, actions):
        current_obs, _ = self.evaluator.get_init_cond()
        if self.reference is not None and not self.diverged_from_native[0]:
            obs = {key: self.reference['obs_' + key][:, self.iter:self.iter + 2].copy()
                   for key in ('visual', 'proprio')}
            states = self.reference['states'][:, self.iter:self.iter + 2].copy()
            self.prefix_reuse.append({'push_zero_based': self.iter, 'reused': True})
            return obs, states
        env = self.env.envs[0]
        if self.iter % self.chunk_size == 0:
            # Restore only at the planning boundary; keep the live simulator for
            # all pushes within this chunk, including its particle velocities.
            before_obs, _ = env.prepare(seeds[0], initial_states[0])
        else:
            before_obs = {key: value[0, -1] for key, value in current_obs.items()}
        after_obs, _, _, info = env.step_multiple(actions[0])
        obs = {key: np.concatenate([before_obs[key][None], after_obs[key]], axis=0)[None]
               for key in before_obs}
        initial = np.asarray(initial_states[0]).reshape(info['state'].shape[1:])
        states = np.concatenate([initial[None], info['state']], axis=0)[None]
        if self.reference is not None:
            expected = self.reference['physical_actions'][0, self.iter:self.iter + 1]
            if not np.array_equal(actions[0], expected) and self.first_action_change[0] is None:
                self.first_action_change[0] = self.iter
        self.prefix_reuse.append({'push_zero_based': self.iter, 'reused': False,
                                  'prepared_scene': self.iter % self.chunk_size == 0})
        return obs, states

    def _post_env_feedback(self, taken_actions, e_obses):
        previous_count = len(self.memories[0])
        result = super()._post_env_feedback(taken_actions, e_obses)
        record = self.records[-1]
        raw = record.get('trigger_weight', [0.])[0]
        record['severity_weight'] = raw
        if self.strength_mode == 'fixed' and len(self.memories[0]) > previous_count:
            for memory in self.memories[0][previous_count:]:
                memory['weight'] = torch.ones_like(memory['weight'])
            record['trigger_weight'][0] = 1.
        record['new_memory_strength'] = self.kappa * record.get('trigger_weight', [0.])[0]
        record['chunk_query'] = self.iter % self.chunk_size == 0
        record['next_chunk_push_one_based'] = (self.iter // self.chunk_size + 1) * self.chunk_size + 1
        # The base detector marks the next push; here S is applied only when a
        # new chunk is actually optimized, after the fixed chunk has finished.
        self.first_intervention[0] = self._applied_at
        return result

    def plan(self, obs_0, obs_g, actions=None):
        result = super().plan(obs_0, obs_g, actions)
        Path('chunk_audit.json').write_text(json.dumps({
            'chunk_size': self.chunk_size, 'strength_mode': self.strength_mode,
            'optimizer_calls': len(self.chunk_queries),
            'optimizer_push_indices_zero_based': self.chunk_queries,
            'first_branched_chunk_push_zero_based': self.branch_push,
            'observation_after_every_push': True,
            'scene_restored_only_at_chunk_boundaries': True,
            'planning_seed': self.planning_seed, 'environment_seed': self.evaluator.seed[0],
        }, indent=2) + '\n')
        return result
