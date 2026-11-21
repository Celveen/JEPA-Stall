"""Optimize contact-waypoint tracking jointly with the world-model objective."""

import json
from pathlib import Path

import numpy as np
import torch

from .lewm_contact import wrapped
from .lewm_contact_burst import LeWMContactBurstMPCPlanner
from .lewm_proprio import pusher_rollout_map


class LeWMContactObjectiveMPCPlanner(LeWMContactBurstMPCPlanner):
    def __init__(self, contact_objective, proprio, **kwargs):
        super().__init__(proprio=proprio, **kwargs)
        self.contact_weight = float(contact_objective["weight"])
        if not np.isfinite(self.contact_weight) or self.contact_weight <= 0:
            raise ValueError("Expected a finite positive contact-objective weight")
        self.executed_controls = self.n_taken_actions * self.frameskip
        matrix = pusher_rollout_map(
            self.executed_controls, float(proprio["kp"]), float(proprio["kv"]),
            int(proprio["sim_hz"]), int(proprio["control_hz"]),
        )[1:]
        self._contact_map = torch.as_tensor(matrix, dtype=torch.float32, device=self.device)
        self._contact_indices = None
        self._contact_records = []
        Path("lewm_contact_objective_meta.json").write_text(json.dumps({
            "weight": self.contact_weight, "control_window": self.executed_controls,
            "target": "observed contact waypoint, inward push target, or final pusher goal",
            "position_normalization": "dataset proprio position standard deviation",
            "post_optimization_action_offset": False,
            "legacy_bias_amplitude_limits_used": False,
            "object_state_source": "observed simulator state and known shape geometry",
        }, indent=2) + "\n")

    def _contact_positions(self, actions):
        commands = actions[self._contact_indices, :self.n_taken_actions].reshape(
            len(self._contact_indices), self.executed_controls, 2)
        commands = self.action_scale * (commands * self._action_std + self._action_mean)
        inputs = torch.cat((self._contact_current[:, None, 2:], commands), dim=1)
        return self._contact_current[:, None, :2] + self._contact_map @ inputs

    def _contact_cost(self, actions):
        positions = self._contact_positions(actions)
        error = (positions - self._contact_target[:, None]) / self._position_std
        return error.square().mean(dim=(1, 2))

    def _action_objective(self, predicted, goal, actions, step):
        loss = super()._action_objective(predicted, goal, actions, step)
        if self._contact_indices is None:
            return loss
        cost = self.contact_weight * self._contact_cost(actions)
        if self._contact_first_cost is None:
            self._contact_first_cost = cost.detach()
            self._contact_first_base = loss[self._contact_indices].detach()
            with torch.no_grad():
                self._contact_initial_positions = self._contact_positions(actions).detach()
        return loss.index_add(0, self._contact_indices, cost)

    def _plan_with_bias(self, obs_0, obs_g, actions=None, step=None):
        # Reuse the detector, burst schedule, and geometric target selection.
        # Never call the parent's post-optimization additive-bias implementation.
        self._applied = []
        self._contact_records = []
        _, self._physical_state = self.evaluator.get_init_cond()
        self._physical_goal = self.evaluator.state_g
        error = self._physical_goal - self._physical_state
        position_error = np.linalg.norm(error[:, 2:4], axis=1)
        angle_error = np.abs(wrapped(error[:, 4]))
        self._object_ready = (position_error < self.position_tol) & (angle_error < self.angle_tol)
        self._bias_phase = {}

        self._boost_until[self._pending] = self.iter + self.followup_replans
        proprio_ids = np.flatnonzero(self._proprio_active())
        self._active = None
        if len(proprio_ids):
            self._active = torch.as_tensor(proprio_ids, device=self.device)
            self._current = torch.as_tensor(obs_0["proprio"][proprio_ids, -1, :4],
                                            dtype=torch.float32, device=self.device)
            self._pusher_goal = torch.as_tensor(obs_g["proprio"][proprio_ids, -1, :2],
                                                dtype=torch.float32, device=self.device)

        ids = np.flatnonzero(self._eligible_for_bias())
        self._contact_indices = None
        self._contact_first_cost = None
        if len(ids):
            direction = self._bias_direction(obs_0, obs_g, ids)
            moving = np.linalg.norm(direction, axis=1) > 1e-6
            ids, direction = ids[moving], direction[moving]
        if len(ids):
            self._contact_indices = torch.as_tensor(ids, device=self.device)
            current = obs_0["proprio"][ids, -1, :4]
            target = current[:, :2] + direction
            self._contact_current = torch.as_tensor(current, dtype=torch.float32, device=self.device)
            self._contact_target = torch.as_tensor(target, dtype=torch.float32, device=self.device)
        try:
            planned, lengths = self._plan_actions(obs_0=obs_0, obs_g=obs_g, actions=actions, step=step)
            if len(ids):
                if self._contact_first_cost is None:
                    raise RuntimeError("Contact objective was not evaluated by the action optimizer")
                with torch.no_grad():
                    final_cost = self.contact_weight * self._contact_cost(planned)
                    final_positions = self._contact_positions(planned)
                self._last_bias[ids] = self.iter
                self._bias_count[ids] += 1
                self._contact_records = [
                    {"sample_idx": int(i), "phase": self._bias_phase[int(i)],
                     "target_px": target[j].tolist(), "direction_px": direction[j].tolist(),
                     "current_pusher_state": current[j].tolist(),
                     "weighted_contact_cost_before": float(self._contact_first_cost[j]),
                     "weighted_contact_cost_after": float(final_cost[j]),
                     "base_objective_before": float(self._contact_first_base[j]),
                     "initial_predicted_positions": self._contact_initial_positions[j].cpu().tolist(),
                     "optimized_predicted_positions": final_positions[j].cpu().tolist(),
                     "object_position_error_px": float(position_error[i]),
                     "object_angle_error_deg": float(np.degrees(angle_error[i])),
                     "burst_start_replan": int(self._burst_start[i]),
                     "burst_end_replan": int(self._burst_until[i]),
                     "detector_trigger_for_current_plan": bool(self._pending[i]),
                     "post_optimization_action_offset": False}
                    for j, i in enumerate(ids)
                ]
                with Path("lewm_contact_objective.jsonl").open("a") as file:
                    file.write(json.dumps({"replan": self.iter, "actions": self._contact_records}) + "\n")
            if self._active is not None:
                with torch.no_grad():
                    costs = self.proprio_weight * self._position_cost(planned, step)
                with Path("lewm_proprio_weight.jsonl").open("a") as file:
                    file.write(json.dumps({"replan": self.iter, "sample_idx": proprio_ids.tolist(),
                                           "weighted_position_cost_of_plan": costs.cpu().tolist()}) + "\n")
            # Return the optimized controls unchanged, including the warm-start tail.
            return planned, lengths
        finally:
            self._contact_indices = None
            self._active = None

    def _post_env_feedback(self, taken_actions, e_obses):
        logs = super()._post_env_feedback(taken_actions, e_obses)
        logs["ecp/contact_objective_applied"] = len(self._contact_records)
        return logs
