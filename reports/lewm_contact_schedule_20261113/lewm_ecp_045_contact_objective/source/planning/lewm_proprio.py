"""Use observed pusher state and differentiable PushObj dynamics in MPC."""

import json
from pathlib import Path

import numpy as np
import torch

from .lewm_ecp_bias import LeWMStagnationBiasMPCPlanner


def pusher_rollout_map(controls, kp, kv, sim_hz, control_hz):
    """Return linear maps from initial velocity and relative target offsets to positions."""
    steps, remainder = divmod(sim_hz, control_hz)
    if steps < 1 or remainder:
        raise ValueError("Expected an integer number of simulator steps per control")
    dt = 1.0 / sim_hz
    retention = 1 - kv * dt
    substep = np.array([[1 - kp * dt**2, retention * dt], [-kp * dt, retention]])
    transition = np.linalg.matrix_power(substep, steps)
    # Coefficient 0 multiplies initial velocity. Coefficients 1..N multiply
    # each control's target offset, which is relative to the current position.
    pos = np.zeros(controls + 1)
    vel = np.zeros(controls + 1)
    vel[0] = 1
    rows = [pos.copy()]
    for control in range(1, controls + 1):
        pos = pos + transition[0, 1] * vel
        pos[control] += 1 - transition[0, 0]
        vel = transition[1, 1] * vel
        vel[control] -= transition[1, 0]
        rows.append(pos.copy())
    return np.stack(rows)


class LeWMProprioMPCPlanner(LeWMStagnationBiasMPCPlanner):
    def __init__(self, proprio, **kwargs):
        super().__init__(**kwargs)
        self.proprio_weight = float(proprio["weight"])
        self.followup_replans = int(proprio["followup_replans"])
        self.frameskip = self.evaluator.frameskip
        self.horizon = self.sub_planner.horizon
        if self.sub_planner.cuda_graph or self.proprio_weight <= 0 or self.followup_replans < 0:
            raise ValueError("Expected eager GD and a positive proprioceptive weight")
        matrix = pusher_rollout_map(
            self.horizon * self.frameskip,
            float(proprio["kp"]), float(proprio["kv"]),
            int(proprio["sim_hz"]), int(proprio["control_hz"]),
        )[::self.frameskip]
        self._position_map = torch.as_tensor(matrix, dtype=torch.float32, device=self.device)
        weights = self.base ** torch.arange(self.horizon + 1, device=self.device, dtype=torch.float32)
        self._frame_weights = weights / weights.sum()
        self._action_mean = self.preprocessor.action_mean.to(self.device)
        self._action_std = self.preprocessor.action_std.to(self.device)
        self._position_std = self.preprocessor.proprio_std[:2].to(self.device)
        self.sub_planner.action_objective = self._action_objective
        self._active = None

    def plan(self, obs_0, obs_g, actions=None):
        self._boost_until = np.full(len(obs_0["visual"]), -1)
        return super().plan(obs_0, obs_g, actions=actions)

    def _plan_with_bias(self, obs_0, obs_g, actions=None, step=None):
        self._boost_until[self._pending] = self.iter + self.followup_replans
        active = np.flatnonzero(self._proprio_active())
        self._active = None
        if len(active):
            current = obs_0["proprio"][active, -1]
            if current.shape[1] < 4:
                raise ValueError("Expected observed pusher position and velocity")
            self._active = torch.as_tensor(active, device=self.device)
            self._current = torch.as_tensor(current[:, :4], dtype=torch.float32, device=self.device)
            goal = obs_g["proprio"][active, -1, :2]
            self._pusher_goal = torch.as_tensor(goal, dtype=torch.float32, device=self.device)
        try:
            result = super()._plan_with_bias(obs_0, obs_g, actions=actions, step=step)
            if self._active is not None:
                with torch.no_grad():
                    costs = self.proprio_weight * self._position_cost(result[0], step)
                with Path("lewm_proprio_weight.jsonl").open("a") as file:
                    record = {"replan": self.iter, "sample_idx": active.tolist(),
                              "weighted_position_cost_of_plan": costs.cpu().tolist()}
                    file.write(json.dumps(record) + "\n")
            return result
        finally:
            self._active = None

    def _proprio_active(self):
        return (self.iter <= self._boost_until) & ~self.is_success

    def _position_cost(self, actions, step):
        relative_targets = actions[self._active].reshape(len(self._active), -1, 2)
        relative_targets = self.action_scale * (relative_targets * self._action_std + self._action_mean)
        inputs = torch.cat((self._current[:, None, 2:], relative_targets), dim=1)
        position = self._current[:, None, :2] + self._position_map @ inputs
        frame_error = ((position - self._pusher_goal[:, None]) / self._position_std).square().mean(dim=-1)
        if step is not None and step < self.horizon:
            return frame_error[:, -1]
        return (frame_error * self._frame_weights).mean(dim=1)

    def _action_objective(self, predicted, goal, actions, step):
        loss = self.sub_planner.objective_fn(predicted, goal, step=step)
        if self._active is None:
            return loss
        return loss.index_add(0, self._active, self.proprio_weight * self._position_cost(actions, step))
