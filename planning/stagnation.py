"""Stall regulariser for AdaJEPA planning (planner side).

Most of AdaJEPA's remaining failures are stalls: the agent jitters in place and the scene stops
changing (reports/paper2_planner/report.md). This planner adds a history-dependent term to the
planning objective that makes plans which stay where the episode got stuck more expensive.
Everything is measured in the world model's latent space with the planning objective's own metric,
so nothing depends on the task (pixels, pusher, object):

    D(z, z') = MSE_visual(z, z') + alpha * MSE_proprio(z, z'),   d = sqrt(D)

Trigger (from replan `start` on, observations only): the latent state moved
r = d(z_s, z_{s-W}) / d(z_s, z_goal) of its remaining distance to the goal over the last W replans
(a scale-free movement rate). The stall weight is w = clip((gamma_hi - r) / (gamma_hi - gamma_lo), 0, 1);
when w > 0 the current observation joins the episode's memory with weight w. The memory lasts for the
episode, so a plan cannot drift back into a stall once it has left, and it is re-encoded every replan
with the current (adapted) model.

Penalty (on the planned latents z_h, h = 1..H; the objective's frame weights c_h = base**h):

    J(a) = J_task(a) + kappa * P(a)
    P(a) = sum_h c_h sum_j w_j s_j max(0, 1 - D_sp(z_h, m_j) / rho_j)^2 / sum_h c_h

m_j are the memory latents, s_j = D(m_j, z_goal) / T is the task objective of staying at m_j (T frames)
and rho_j = beta * D_sp(m_j, z_goal): the repelled region around a stall reaches beta of the way to the
goal (in squared distance), so it never covers the goal. Staying on a full-weight memory point thus
costs (1 + kappa) times its task objective. `space` decides D_sp:

    proprio  MSE of the proprio latent (where the agent is)
    latent   D itself (the whole scene)

trigger = arrived (instead of the movement rate above) fires when the agent has reached its own goal but the
scene has not: the proprio share of the remaining distance, alpha * MSE_proprio / D, is below q, and the visual
part sqrt(MSE_visual(z, z_goal)) shrank by less than theta (relative) over the last W replans. It can fire
before the agent starts to hover, e.g. when a pusher parks between the object and the object's goal.

prop_boost > 1 multiplies the objective's proprio term by that factor on every replan from a trigger to
boost_replans replans after it (re-armed by each trigger), so the planner also cares about where the agent
ends up while it escapes.

Until the first trigger the objective is untouched, so a run pairs with escape.mode=none (AdaJEPA)
bit for bit up to that replan.

only_samples (within-chunk indices) plans only those samples; the others are skipped (zero actions,
no logs) but keep their position in the chunk, which fixes their episode seed (seed * i + 1).

Per replan, per_iter_logs.json gets stag_r / stag_share / stag_vprog (both triggers' quantities) / stag_w /
stag_mem_n while the trigger is armed and
stag_task / stag_pen (objective and kappa * penalty of the executed plan) while the penalty is on;
stag_episodes.jsonl gets one line per planned episode.
"""

import json
from pathlib import Path

import numpy as np
import torch

from utils import move_to_device
from .escape import EscapeMPCPlanner

SPACES = ("proprio", "latent")


class StagnationMPCPlanner(EscapeMPCPlanner):
    def __init__(self, stag=None, **kwargs):
        super().__init__(**kwargs)
        s = dict(stag or {})
        self.stag_space = s.get("space", "latent")
        if self.stag_space not in SPACES:
            raise ValueError(f"stag.space must be one of {SPACES}, got {self.stag_space!r}")
        self.stag_kappa = float(s.get("kappa", 0.0))
        self.stag_start = int(s.get("start", 10))
        self.stag_window = int(s.get("window", 3))
        self.stag_gamma_lo = float(s.get("gamma_lo", 0.05))
        self.stag_gamma_hi = float(s.get("gamma_hi", 0.15))
        self.stag_beta = float(s.get("beta", 0.25))
        self.stag_trigger = s.get("trigger", "movement")
        if self.stag_trigger not in ("movement", "arrived"):
            raise ValueError(f"stag.trigger must be movement or arrived, got {self.stag_trigger!r}")
        self.stag_q = float(s.get("q", 0.2))
        self.stag_theta = float(s.get("theta", 0.0))
        self.stag_prop_boost = float(s.get("prop_boost", 1.0))
        self.stag_boost_replans = int(s.get("boost_replans", 2))
        self.stag_alpha = float(s.get("alpha", 1.0))  # the objective's proprio weight
        self.stag_base = float(s.get("base", 2.0))  # the objective's frame-weight base
        only = s.get("only_samples")
        self.only_samples = None if only is None else {int(i) for i in only}
        self._stag_inner = self.sub_planner.objective_fn
        self.sub_planner.objective_fn = self._stag_objective
        self._reset_stag()
        Path("stag_meta.json").write_text(json.dumps({
            "space": self.stag_space, "kappa": self.stag_kappa, "start": self.stag_start, "window": self.stag_window,
            "gamma_lo": self.stag_gamma_lo, "gamma_hi": self.stag_gamma_hi, "beta": self.stag_beta,
            "alpha": self.stag_alpha, "base": self.stag_base, "trigger": self.stag_trigger, "q": self.stag_q,
            "theta": self.stag_theta, "prop_boost": self.stag_prop_boost, "boost_replans": self.stag_boost_replans,
            "only_samples": None if self.only_samples is None else sorted(self.only_samples),
        }, indent=1))

    def _reset_stag(self):
        self._hist, self._mem_obs, self._mem_w = [], [], []
        self._mem, self._obs_g = None, None
        self._stag_first, self._stag_log, self._boost_until = None, None, None

    # ------------------------------------------------------------------ episode
    def _plan_single(self, **kw):
        if self.only_samples is not None and self._sample_idx not in self.only_samples:
            return torch.zeros(1, self.n_taken_actions, self.action_dim, device=self.device), np.full(1, np.inf)
        self._reset_stag()
        self._obs_g = {k: np.array(v) for k, v in kw["obs_g_i"].items()}
        out = super()._plan_single(**kw)
        with Path("stag_episodes.jsonl").open("a") as f:
            f.write(json.dumps({
                "sample_idx": self._sample_idx, "seed": self._ep_seed, "success": bool(self.is_success[0]),
                "replans": self.iter, "first_trigger": self._stag_first, "mem_n": len(self._mem_w),
                "mem_w_sum": round(float(sum(self._mem_w)), 4), **self._episode_extra(),
            }) + "\n")
        return out

    def _episode_extra(self):
        """Extra fields of this episode's stag_episodes.jsonl line (subclasses)."""
        return {}

    def _task_progress(self):
        """PushObj logs object/pusher distances (inherited); other envs (e.g. the maze, state = position and
        velocity) log the position distance to the goal and the state."""
        s = np.asarray(self._last_eval[3][0, -1], dtype=np.float64)
        if len(s) >= 7:
            return super()._task_progress()
        g = np.asarray(self.evaluator.state_g[0], dtype=np.float64)
        return {"dist_pos": float(np.linalg.norm(g[:2] - s[:2])), "state": [round(float(x), 4) for x in s]}

    @torch.no_grad()
    def _encode(self, obs_list):
        """Encode single-frame observations with the current model: visual (N, P*D), proprio (N, Dp)."""
        obs = {k: np.concatenate([o[k] for o in obs_list], axis=0) for k in obs_list[0]}
        z = self.wm.encode_obs(move_to_device(self.preprocessor.transform_obs(obs), self.device))
        return z["visual"][:, 0].flatten(1), z["proprio"][:, 0].flatten(1)

    def _post_env_feedback(self, taken_actions, e_obses):
        out = super()._post_env_feedback(taken_actions, e_obses)
        rec = self._records[-1]
        if self._stag_log is not None:  # objective and penalty of the plan executed on this replan
            rec.update(self._stag_log)
            self._stag_log = None
        W, nxt = self.stag_window, self.iter + 1
        self._hist = (self._hist + [{k: np.array(v[:, -1:]) for k, v in e_obses.items()}])[-(W + 1):]
        if len(self._hist) > W and self.stag_start <= nxt < self.max_iter and not self.is_success[0]:
            zv, zp = self._encode([self._hist[-1], self._hist[0], self._obs_g])  # now, W replans ago, goal

            mv = lambda i, j: float(((zv[i] - zv[j]) ** 2).mean())  # noqa: E731
            mp = lambda i, j: float(((zp[i] - zp[j]) ** 2).mean())  # noqa: E731
            # both triggers' quantities are logged (a reference run then selects episodes for either)
            r = ((mv(0, 1) + self.stag_alpha * mp(0, 1)) / max(mv(0, 2) + self.stag_alpha * mp(0, 2), 1e-12)) ** 0.5
            share = self.stag_alpha * mp(0, 2) / max(mv(0, 2) + self.stag_alpha * mp(0, 2), 1e-12)
            v_now, v_then = mv(0, 2) ** 0.5, mv(1, 2) ** 0.5
            vprog = (v_then - v_now) / max(v_then, 1e-12)
            rec["stag_r"], rec["stag_share"], rec["stag_vprog"] = round(r, 5), round(share, 5), round(vprog, 5)
            if self.stag_trigger == "movement":
                w = min(max((self.stag_gamma_hi - r) / (self.stag_gamma_hi - self.stag_gamma_lo), 0.0), 1.0)
            else:
                w = 1.0 if (share < self.stag_q and vprog < self.stag_theta) else 0.0
            rec["stag_w"] = round(w, 4)
            if w > 0:
                self._mem_obs.append(self._hist[-1])
                self._mem_w.append(w)
                if self._stag_first is None:
                    self._stag_first = nxt
                if self.stag_prop_boost != 1.0:
                    self._boost_until = nxt + self.stag_boost_replans
            rec["stag_mem_n"] = len(self._mem_w)
        return out

    # ------------------------------------------------------------------ planning
    def _build_mem(self):
        """Encode the stall memory with the current model for this replan's penalty (None: penalty off)."""
        self._mem = None
        if self._mem_w and self.stag_kappa > 0:
            zv, zp = self._encode(self._mem_obs + [self._obs_g])
            mv, mp, gv, gp = zv[:-1], zp[:-1], zv[-1:], zp[-1:]
            d_goal_p = ((mp - gp) ** 2).mean(1)
            d_goal = ((mv - gv) ** 2).mean(1) + self.stag_alpha * d_goal_p
            self._mem = {"v": mv, "p": mp,
                         "w": torch.tensor(self._mem_w, device=self.device, dtype=zp.dtype),
                         "stay": d_goal,  # D(m_j, z_goal); / T in _penalty
                         "rho": self.stag_beta * (d_goal if self.stag_space == "latent" else d_goal_p)}

    def _sub_plan(self, obs_0, obs_g, actions=None, step=None):
        self._build_mem()
        acts, lens = super()._sub_plan(obs_0, obs_g, actions=actions, step=step)
        if self._mem is not None:
            with torch.no_grad():
                z_g = self.wm.encode_obs(move_to_device(self.preprocessor.transform_obs(obs_g), self.device))
                z, _ = self.wm.rollout(obs_0=move_to_device(self.preprocessor.transform_obs(obs_0), self.device),
                                       act=acts.to(self.device))
                task = self._stag_inner(z, z_g, step=step)
                pen = self.stag_kappa * self._penalty(z)
            self._stag_log = {"stag_task": round(float(task[0]), 6), "stag_pen": round(float(pen[0]), 6),
                              "stag_boost": self._boosting()}
        return acts, lens

    def _boosting(self):
        return self._boost_until is not None and self.iter <= self._boost_until

    def _stag_objective(self, z_obs_pred, z_obs_tgt, step=None):
        if self._boosting():
            s = self.stag_prop_boost ** 0.5
            loss = self._stag_inner({**z_obs_pred, "proprio": z_obs_pred["proprio"] * s},
                                    {**z_obs_tgt, "proprio": z_obs_tgt["proprio"] * s}, step=step)
        else:
            loss = self._stag_inner(z_obs_pred, z_obs_tgt, step=step)
        if self._mem is None:
            return loss
        return loss + self.stag_kappa * self._penalty(z_obs_pred)

    def _penalty(self, z):
        """P(a) of the module docstring for predicted latents z (dict, (B, T, ...)); returns (B,)."""
        m = self._mem
        prop = z["proprio"][:, 1:].flatten(2)  # (B, H, Dp): predicted frames only
        d = ((prop[:, :, None] - m["p"][None, None]) ** 2).mean(-1)  # (B, H, M)
        if self.stag_space == "latent":
            vis = z["visual"][:, 1:].flatten(2)
            d = ((vis[:, :, None] - m["v"][None, None]) ** 2).mean(-1) + self.stag_alpha * d
        H = prop.shape[1]
        k = torch.clamp(1 - d / m["rho"].clamp_min(1e-12), min=0) ** 2 * (m["w"] * m["stay"] / (H + 1))
        c = self.stag_base ** torch.arange(1, H + 1, device=d.device, dtype=d.dtype)
        return (k.sum(-1) * c).sum(-1) / c.sum()
