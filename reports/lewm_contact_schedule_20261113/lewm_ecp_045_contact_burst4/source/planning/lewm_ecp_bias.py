"""Apply the PushObj goal-directed offset after LeWM's regularized plan."""

import json
from pathlib import Path

import numpy as np
import torch

from .lewm_ecp import LeWMStagnationMPCPlanner


class LeWMStagnationBiasMPCPlanner(LeWMStagnationMPCPlanner):
    def __init__(self, bias, **kwargs):
        super().__init__(**kwargs)
        self.cap_px = float(bias["cap_px"])
        self.action_scale = float(bias["action_scale"])
        self.min_gap = int(bias["min_gap"])
        self.max_interventions = int(bias["max_interventions"])
        if self.cap_px <= 0 or self.action_scale <= 0 or self.min_gap < 1 or self.max_interventions < 1:
            raise ValueError("Invalid goal-directed offset settings")
        if self.action_dim != 2 * self.evaluator.frameskip:
            raise ValueError("Expected two coordinates per PushObj control")
        self._plan_actions = self.sub_planner.plan
        self.sub_planner.plan = self._plan_with_bias

    def plan(self, obs_0, obs_g, actions=None):
        count = len(obs_0["visual"])
        self._pending = np.zeros(count, dtype=bool)
        self._last_bias = np.full(count, -self.min_gap)
        self._bias_count = np.zeros(count, dtype=int)
        self._applied = []
        return super().plan(obs_0, obs_g, actions=actions)

    def _post_env_feedback(self, taken_actions, e_obses):
        previous_memory_size = len(self._memory_w)
        logs = super()._post_env_feedback(taken_actions, e_obses)
        self._pending.fill(False)
        if len(self._memory_w) > previous_memory_size:
            self._pending = self._memory_w[-1].cpu().numpy() > 0
        logs["ecp/bias_applied"] = len(self._applied)
        if self._applied:
            with Path("lewm_action_bias.jsonl").open("a") as f:
                f.write(json.dumps({"replan": self.iter, "actions": self._applied}) + "\n")
        return logs

    def _plan_with_bias(self, obs_0, obs_g, actions=None, step=None):
        self._applied = []
        planned, lengths = self._plan_actions(obs_0=obs_0, obs_g=obs_g, actions=actions, step=step)
        eligible = self._eligible_for_bias()
        indices = np.flatnonzero(eligible)
        if not len(indices):
            return planned, lengths

        direction = self._bias_direction(obs_0, obs_g, indices)
        distance = np.linalg.norm(direction, axis=1)
        moving = distance > 1e-6
        if not moving.any():
            return planned, lengths
        indices, direction, distance = indices[moving], direction[moving], distance[moving]
        controls = self.n_taken_actions * self.evaluator.frameskip
        magnitude = np.minimum(self.cap_px, distance / controls)
        offset_px = direction * (magnitude / distance)[:, None]

        batch = torch.as_tensor(indices, device=planned.device)
        offset = torch.as_tensor(offset_px / self.action_scale, device=planned.device, dtype=planned.dtype)
        std = self.preprocessor.action_std.to(device=planned.device, dtype=planned.dtype)
        # Adding a physical offset is equivalent to adding offset/std in normalized actions.
        out = planned.detach().clone()
        prefix = out[batch, :self.n_taken_actions]
        updated = prefix.reshape(len(batch), self.n_taken_actions, self.evaluator.frameskip, 2)
        updated = updated + (offset / std)[:, None, None]
        out[batch, :self.n_taken_actions] = updated.reshape_as(prefix)

        actual_px = (
            (updated - prefix.reshape_as(updated)) * std * self.action_scale
        ).cpu().numpy()
        self._last_bias[indices] = self.iter
        self._bias_count[indices] += 1
        self._applied = [
            {
                "sample_idx": int(i),
                "pusher_to_goal_px": v.tolist(),
                "offset_px_per_control": delta.tolist(),
                "measured_offset_px_per_control": applied.reshape(controls, 2).tolist(),
            }
            for i, v, delta, applied in zip(indices, direction, offset_px, actual_px)
        ]
        return out, lengths

    def _eligible_for_bias(self):
        eligible = self._pending & ~self.is_success
        eligible &= (self.iter - self._last_bias >= self.min_gap)
        eligible &= (self._bias_count < self.max_interventions)
        return eligible

    def _bias_direction(self, obs_0, obs_g, indices):
        return obs_g["proprio"][indices, -1, :2] - obs_0["proprio"][indices, -1, :2]
