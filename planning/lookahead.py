"""Two-stage lookahead on top of the stall regulariser (planner side).

The planning objective only sees `horizon` model steps ahead, so a manoeuvre whose pay-off lies beyond the
horizon (going around an obstacle, or around an object before pushing it) never wins. At chosen replans this
planner compares the usual plan with K macro moves followed by a plan:

    candidate 0      the usual warm start plus zero-action steps, all m + F steps free
    candidate k      m model steps of a constant agent action in direction 2 pi k / K (normalized action of
                     length `scale` per env step, i.e. in data standard deviations: env-agnostic), kept fixed,
                     followed by F free steps (the usual warm start, padded with zero actions)

F (`follow`, model steps) defaults to the planner's horizon; a longer F lets a detour pay off inside the
comparison. All candidates are optimised together by gradient descent through the world model on the planner's
own objective (stall penalty and proprio boost included when they are on) over m + F steps, and the lowest
final objective wins. When a macro wins, it is executed over the next ceil(m / n_taken) replans without
replanning (the commitment is what lets the agent leave the basin), and its follow-up plan becomes the warm
start after that. When candidate 0 wins, exec decides:

    macro   the replan is planned exactly as without the lookahead (the lookahead runs under a forked RNG,
            so such replans stay bit-identical to the base planner)
    best    candidate 0's long plan is executed (one replan's actions; its continuation is the next warm start):
            with follow > horizon this is a one-replan horizon extension to m + F steps

When to look ahead (`lookahead.when`): at the first replan (start), on replans where the stall trigger fires
(trigger; at most `max_n` per episode, at least `gap` replans apart), or both. Per replan, per_iter_logs.json
gets `la` (candidate objectives, the winner, its direction) on lookahead replans and `la_commit` on committed ones.
"""

import math

import numpy as np
import torch
from einops import rearrange

from utils import move_to_device
from .stagnation import StagnationMPCPlanner

WHEN = ("start", "trigger", "both")


class LookaheadMPCPlanner(StagnationMPCPlanner):
    def __init__(self, lookahead=None, **kwargs):
        super().__init__(**kwargs)
        la = dict(lookahead or {})
        self.la_when = la.get("when", "trigger")
        if self.la_when not in WHEN:
            raise ValueError(f"lookahead.when must be one of {WHEN}, got {self.la_when!r}")
        self.la_dirs = int(la.get("dirs", 8))
        self.la_steps = int(la.get("steps", 2))  # m, model steps
        self.la_scale = float(la.get("scale", 1.0))
        self.la_gap = int(la.get("gap", 4))
        self.la_max = int(la.get("max_n", 3))
        self.la_follow = la.get("follow")  # None: the sub-planner's horizon
        self.la_exec = la.get("exec", "macro")
        if self.la_exec not in ("macro", "best"):
            raise ValueError(f"lookahead.exec must be macro or best, got {self.la_exec!r}")

    def _reset_stag(self):
        super()._reset_stag()
        self._commit, self._la_last, self._la_n, self._la_rec, self._triggered = [], -(10 ** 9), 0, None, False
        self._la_first_macro = None

    def _episode_extra(self):
        # first replan (0-based) that executed a lookahead plan: the run equals the base planner before it
        return {"la_n": self._la_n, "la_first_macro": self._la_first_macro}

    def _post_env_feedback(self, taken_actions, e_obses):
        n_mem = len(self._mem_w)
        out = super()._post_env_feedback(taken_actions, e_obses)
        if self._la_rec is not None:
            self._records[-1].update(self._la_rec)
            self._la_rec = None
        self._triggered = len(self._mem_w) > n_mem  # the stall trigger fired for the next replan
        return out

    def _want_lookahead(self):
        if self.iter == 0:
            return self.la_when in ("start", "both")
        return (self.la_when in ("trigger", "both") and self._triggered and self._la_n < self.la_max
                and self.iter - self._la_last >= self.la_gap)

    def _sub_plan(self, obs_0, obs_g, actions=None, step=None):
        if self._commit:  # the rest of a chosen macro (then its follow-up plan is the warm start)
            self._la_rec = {"la_commit": True}
            return self._commit.pop(0), np.full(1, np.inf)
        if not self._want_lookahead():
            return super()._sub_plan(obs_0, obs_g, actions=actions, step=step)
        self._la_last, self._la_n = self.iter, self._la_n + 1
        self._build_mem()  # the penalty of this replan, as the usual plan would see it
        dev = torch.device(self.device)
        with torch.random.fork_rng(devices=[dev.index or 0] if dev.type == "cuda" else []):
            best, costs, plan = self._lookahead(obs_0, obs_g, actions, step)
        self._la_rec = {"la": {"winner": best, "costs": [round(c, 6) for c in costs],
                               "dir": None if best == 0 else round(360.0 * (best - 1) / self.la_dirs, 1)}}
        if best == 0 and self.la_exec == "macro":
            return super()._sub_plan(obs_0, obs_g, actions=actions, step=step)
        if self._la_first_macro is None:
            self._la_first_macro = self.iter
        n, H = self.n_taken_actions, self.sub_planner.horizon
        plan = self._finish(plan[:, :(self.la_steps if best else n) + H], obs_0)  # the warm start after it: H steps
        if best:  # a macro: the next replans execute the rest of it
            self._commit = [plan[:, k * n:] for k in range(1, math.ceil(self.la_steps / n))]
        return plan, np.full(1, np.inf)

    def _directions(self, d):
        if d == 2:
            ang = 2 * np.pi * np.arange(self.la_dirs) / self.la_dirs
            return np.stack([np.cos(ang), np.sin(ang)], 1)
        return np.concatenate([np.eye(d), -np.eye(d)])  # other action sizes: +- each axis

    def _lookahead(self, obs_0, obs_g, actions, step):
        sp, fs, m = self.sub_planner, self.evaluator.frameskip, self.la_steps
        F = sp.horizon if self.la_follow is None else int(self.la_follow)
        base = sp.init_actions(obs_0, actions).to(self.device)  # (1, H, A): the usual warm start
        A = base.shape[-1]
        d = A // fs
        zero = self.preprocessor.normalize_actions(torch.zeros(1, max(m, F), fs, d))  # zero env action, normalized
        zero = rearrange(zero, "b t f d -> b t (f d)").to(self.device, base.dtype)
        base = torch.cat([base, zero], 1)[:, :F]  # F follow-up steps
        u = torch.as_tensor(self._directions(d), device=self.device, dtype=base.dtype)  # (K, d)
        K = u.shape[0]
        macro = (self.la_scale * u)[:, None, None, :].expand(K, m, fs, d).reshape(K, m, A)
        acts = torch.cat([torch.cat([base, zero[:, :m]], 1), torch.cat([macro, base.expand(K, -1, -1)], 1)], 0)
        acts = acts.detach().clone()  # (K + 1, m + F, A)
        fixed = acts[1:, :m].clone()
        acts.requires_grad = True
        rep = lambda o: {k: v.repeat_interleave(K + 1, 0) for k, v in o.items()}  # noqa: E731
        obs0 = rep(move_to_device(self.preprocessor.transform_obs(obs_0), self.device))
        with torch.no_grad():
            z_g = rep(self.wm.encode_obs(move_to_device(self.preprocessor.transform_obs(obs_g), self.device)))
        opt = sp.get_action_optimizer(acts)
        sched = sp.get_scheduler(opt)
        for _ in range(sp.opt_steps):
            opt.zero_grad()
            z, _ = self.wm.rollout(obs_0=obs0, act=acts)
            (sp.objective_fn(z, z_g, step=step).mean() * (K + 1)).backward()
            opt.step()
            if sched is not None:
                sched.step()
            with torch.no_grad():
                if sp.action_noise:
                    acts += torch.randn_like(acts) * sp.action_noise
                acts[1:, :m] = fixed  # the macro stays fixed
        with torch.no_grad():
            z, _ = self.wm.rollout(obs_0=obs0, act=acts)
            final = sp.objective_fn(z, z_g, step=step).double().cpu().numpy()
        best = int(np.argmin(final))
        return best, final.tolist(), acts[best:best + 1].detach()
