"""Batched latent stall memory for frozen LeWM PushObj evaluation.

LeWM has no proprioceptive representation, so its ECP deployment uses the
visual latent movement trigger and memory penalty. Before a trigger the native
planner objective is returned unchanged.
"""
import json
from pathlib import Path

import numpy as np
import torch

from utils import move_to_device
from .continual import SlowFrozenMPCPlanner


class LeWMStagnationMPCPlanner(SlowFrozenMPCPlanner):
    def __init__(self, stag=None, **kwargs):
        super().__init__(**kwargs)
        s = dict(stag or {})
        self.kappa = float(s.get("kappa", 2.0))
        self.start = int(s.get("start", 10))
        self.window = int(s.get("window", 3))
        self.gamma_lo = float(s.get("gamma_lo", .15))
        self.gamma_hi = float(s.get("gamma_hi", .45))
        self.beta = float(s.get("beta", .25))
        self.base = float(s.get("base", 2.0))
        if not (0 <= self.gamma_lo < self.gamma_hi and 0 < self.beta < 1):
            raise ValueError("Invalid ECP trigger or radius settings")
        if self.sub_planner.microbatch_size < self.evaluator.obs_0["visual"].shape[0]:
            raise ValueError("LeWM ECP requires one GD microbatch per shape chunk")
        self._native_objective = self.sub_planner.objective_fn
        self.sub_planner.objective_fn = self._ecp_objective
        self._memory_z = []
        self._memory_w = []
        self._history = []
        self._first = None
        self._trigger_counts = None
        Path("lewm_ecp_meta.json").write_text(json.dumps({
            "kappa": self.kappa, "start": self.start, "window": self.window,
            "gamma_lo": self.gamma_lo, "gamma_hi": self.gamma_hi,
            "beta": self.beta, "base": self.base,
            "distance_space": "visual_latent", "proprio_boost": "inapplicable",
        }, indent=2) + "\n")

    @torch.no_grad()
    def _encode(self, obs):
        z = self.wm.encode_obs(move_to_device(self.preprocessor.transform_obs(obs), self.device))
        return z["visual"][:, 0].flatten(1).detach()

    def plan(self, obs_0, obs_g, actions=None):
        count = obs_0["visual"].shape[0]
        self._memory_z, self._memory_w, self._history = [], [], []
        self._first = [None] * count
        self._trigger_counts = [0] * count
        self._goal_z = self._encode(obs_g)
        out = super().plan(obs_0, obs_g, actions=actions)
        with Path("lewm_ecp_episodes.jsonl").open("a") as f:
            for i in range(count):
                f.write(json.dumps({"sample_idx": i, "success": bool(self.is_success[i]),
                                    "first_trigger": self._first[i],
                                    "trigger_count": self._trigger_counts[i],
                                    "action_len": None if not np.isfinite(self.action_len[i])
                                    else int(self.action_len[i])}) + "\n")
        return out

    def _post_env_feedback(self, taken_actions, e_obses):
        current = {k: np.asarray(v[:, -1:]).copy() for k, v in e_obses.items()}
        self._history = (self._history + [current])[-(self.window + 1):]
        nxt = self.iter + 1
        if len(self._history) <= self.window or nxt < self.start or nxt >= self.max_iter:
            return {"ecp/active_memory": len(self._memory_z), "ecp/triggers": 0}
        z_now = self._encode(self._history[-1])
        z_then = self._encode(self._history[0])
        d_move = ((z_now - z_then) ** 2).mean(-1)
        d_goal = ((z_now - self._goal_z) ** 2).mean(-1)
        rate = torch.sqrt(d_move / d_goal.clamp_min(1e-12))
        weight = ((self.gamma_hi - rate) / (self.gamma_hi - self.gamma_lo)).clamp(0, 1)
        active = torch.as_tensor(~self.is_success, device=self.device)
        weight = weight * active
        fired = weight > 0
        if bool(fired.any()):
            self._memory_z.append(z_now)
            self._memory_w.append(weight.detach())
            for i in fired.nonzero(as_tuple=True)[0].tolist():
                self._trigger_counts[i] += 1
                if self._first[i] is None:
                    self._first[i] = nxt
        with Path("lewm_ecp_replans.jsonl").open("a") as f:
            f.write(json.dumps({"replan": nxt, "triggered": int(fired.sum()),
                                "active_memory": len(self._memory_z),
                                "movement_ratio": [round(float(x), 5) for x in rate],
                                "weight": [round(float(x), 5) for x in weight]}) + "\n")
        return {"ecp/active_memory": len(self._memory_z), "ecp/triggers": int(fired.sum())}

    def _ecp_objective(self, z_obs_pred, z_obs_tgt, step=None):
        native = self._native_objective(z_obs_pred, z_obs_tgt, step=step)
        if not self._memory_z or self.kappa == 0:
            return native
        predicted = z_obs_pred["visual"][:, 1:].flatten(2)
        mem = torch.stack(self._memory_z, dim=1)
        weights = torch.stack(self._memory_w, dim=1)
        distance = ((predicted[:, :, None] - mem[:, None]) ** 2).mean(-1)
        to_goal = ((mem - self._goal_z[:, None]) ** 2).mean(-1)
        rho = (self.beta * to_goal).clamp_min(1e-12)
        horizon = predicted.shape[1]
        proximity = (1 - distance / rho[:, None]).clamp_min(0).square()
        penalty = (proximity * (weights * to_goal / (horizon + 1))[:, None]).sum(-1)
        coeff = self.base ** torch.arange(1, horizon + 1,
                                          device=predicted.device, dtype=predicted.dtype)
        penalty = (penalty * coeff).sum(-1) / coeff.sum()
        return native + self.kappa * penalty
