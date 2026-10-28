"""Model-or-ruler attribution logs for latent world-model MPC (D0: logging only, no change to any action).

A stall is either the world model's fault (it promised progress that the executed actions did not deliver) or the
ruler's (the planning cost sees no better state within the horizon, and indeed nothing changes). Both are read off
the planner's own per-frame cost, with the weights the plan was made with:

    C(z) = MSE_visual(z, z_goal) + alpha * MSE_proprio(z, z_goal)     (the objective's per-frame metric)

    c_s        C(z_s), z_s the current observation's latent (frame 0 of the plan's rollout)
    C(zh_h)    h = 1..H, the executed plan's predicted latents; zh_1 follows the executed step
    C(z_s+1)   the next observation, encoded before the test-time update of this replan

so the promise is P = c_s - C(zh_1), the delivery D = c_s - C(z_s+1) and the headroom H = c_s - min_h C(zh_h). The
verdicts are computed offline (scripts/attr_d0.py); this planner only logs. Per replan, per_iter_logs.json gets

    attr_cv / attr_cp   visual / proprio parts of C over the rollout frames 0..H (unweighted MSEs; C = cv + alpha cp)
    attr_nv / attr_np   the same for the next observation
    attr_J              the executed plan's objective as the sub-planner optimised it
    attr_zp1            the predicted proprio embedding of zh_1 (positions are read off it offline, maze)
    attr_c0_init        [visual, proprio] parts of c_s under the pre-adaptation weights (encoder drift check)
    attr_rs             restart check (see below), on the replans where it runs

Restart check: on replans planned while the stall trigger is armed (the movement trigger of
planning/stagnation.py fired after the previous replan), every `restart_every`-th one starting with the first, the
sub-planner also plans from `restarts` random initialisations in one batch. attr_rs logs every restart's objective
and the per-frame cost parts of the best one; if a restart beats the executed plan clearly, the stall is a search
failure rather than the ruler's.

With save_enc (adaptive bases only), the adapted encoder tensors are saved as fp16 differences from their initial
values at the first trigger and at the episode's end (attr_enc/<sample>_{trig,end}.pt), so the ruler label can be
recomputed with the encoder the planner had.

Everything runs under no_grad with the CPU and CUDA RNG states forked: GD draws its action noise from the global RNG
even at noise 0, and the AdaJEPA step's dropout draws from it too. A run of this planner therefore pairs bit for bit
with StagnationMPCPlanner at stag.kappa = 0 (the reference runs).
"""

import json
from pathlib import Path

import numpy as np
import torch

from utils import move_to_device
from .stagnation import StagnationMPCPlanner


def _r(x):
    return float(f"{float(x):.6g}")


def _rl(t):
    return [_r(x) for x in t.detach().cpu().flatten().tolist()]


class AttributionMPCPlanner(StagnationMPCPlanner):
    def __init__(self, attr=None, **kwargs):
        super().__init__(**kwargs)
        a = dict(attr or {})
        self.attr_restarts = int(a.get("restarts", 8))
        self.attr_restart_every = int(a.get("restart_every", 3))
        self.attr_seed = int(a.get("seed", 0))
        self.attr_save_enc = bool(a.get("save_enc", True)) and self.probe_mode != "none"
        self._reset_attr()
        Path("attr_meta.json").write_text(json.dumps({
            "restarts": self.attr_restarts, "restart_every": self.attr_restart_every, "seed": self.attr_seed,
            "save_enc": self.attr_save_enc, "alpha": self.stag_alpha, "probe_mode": self.probe_mode,
        }, indent=1))

    def _reset_attr(self):
        self._attr_pending, self._attr_armed, self._attr_n_armed, self._attr_trig_saved = None, False, 0, False

    def _fork_rng(self):
        devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
        return torch.random.fork_rng(devices=devices)

    # ------------------------------------------------------------------ episode
    def _plan_single(self, **kw):
        self._reset_attr()
        out = super()._plan_single(**kw)
        planned = self.only_samples is None or self._sample_idx in self.only_samples
        if planned and self.attr_save_enc and self._stag_first is not None:
            self._save_enc("end")
        return out

    # ------------------------------------------------------------------ planning
    def _sub_plan(self, obs_0, obs_g, actions=None, step=None):
        armed = self._attr_armed
        acts, lens = super()._sub_plan(obs_0, obs_g, actions=actions, step=step)
        with self._fork_rng(), torch.no_grad():
            pend = self._plan_costs(obs_0, obs_g, acts.detach(), step)
            if armed:
                self._attr_n_armed += 1
                if self.attr_restarts > 0 and (self._attr_n_armed - 1) % self.attr_restart_every == 0:
                    pend["log"]["attr_rs"] = self._restart_check(obs_0, obs_g, step, pend["log"]["attr_J"])
        self._attr_pending = pend
        return acts, lens

    def _frame_costs(self, z, z_g):
        """Visual and proprio MSE to the goal per frame: two (B, T) tensors."""
        v = ((z["visual"] - z_g["visual"]) ** 2).mean(dim=tuple(range(2, z["visual"].ndim)))
        p = ((z["proprio"] - z_g["proprio"]) ** 2).mean(dim=tuple(range(2, z["proprio"].ndim)))
        return v, p

    def _plan_costs(self, obs_0, obs_g, acts, step):
        o0 = move_to_device(self.preprocessor.transform_obs(obs_0), self.device)
        og = move_to_device(self.preprocessor.transform_obs(obs_g), self.device)
        z_g = self.wm.encode_obs(og)
        z, _ = self.wm.rollout(obs_0=o0, act=acts.to(self.device))
        cv, cp = self._frame_costs(z, z_g)
        log = {
            "attr_cv": _rl(cv[0]), "attr_cp": _rl(cp[0]),
            "attr_J": _r(self.sub_planner.objective_fn(z, z_g, step=step)[0]),
            "attr_zp1": _rl(z["proprio"][0, 1]),
        }
        if self.probe_mode != "none":
            with self._initial_weights():
                v0, p0 = self._frame_costs(self.wm.encode_obs(o0), self.wm.encode_obs(og))
            log["attr_c0_init"] = [_r(v0[0, 0]), _r(p0[0, 0])]
        return {"log": log, "z_g": z_g}

    def _restart_check(self, obs_0, obs_g, step, j_exec):
        R = self.attr_restarts
        base = self.sub_planner.init_actions(obs_0, None)
        gen = torch.Generator().manual_seed(self.attr_seed * 1_000_003 + self._ep_seed * 1009 + self.iter)
        init = torch.randn((R,) + tuple(base.shape[1:]), generator=gen)
        rep = lambda o: {k: (v.repeat_interleave(R, 0) if torch.is_tensor(v) else np.repeat(v, R, axis=0))  # noqa: E731
                         for k, v in o.items()}
        obs_0r, obs_gr = rep(obs_0), rep(obs_g)
        with torch.enable_grad():
            acts, _ = self._inner_plan(obs_0=obs_0r, obs_g=obs_gr, actions=init, step=step)
        acts = acts.detach()
        z_g = self.wm.encode_obs(move_to_device(self.preprocessor.transform_obs(obs_gr), self.device))
        z, _ = self.wm.rollout(obs_0=move_to_device(self.preprocessor.transform_obs(obs_0r), self.device), act=acts)
        J = self.sub_planner.objective_fn(z, z_g, step=step)
        best = int(torch.argmin(J))
        cv, cp = self._frame_costs(z, z_g)
        return {"J": _rl(J), "best": best, "J_exec": j_exec, "cv": _rl(cv[best]), "cp": _rl(cp[best])}

    # ------------------------------------------------------------------ MPC hook
    def _post_env_feedback(self, taken_actions, e_obses):
        nxt = None
        if self._attr_pending is not None:  # before the test-time update: the weights the plan was made with
            with self._fork_rng(), torch.no_grad():
                obs1 = {k: np.array(v[:, -1:]) for k, v in e_obses.items()}
                z1 = self.wm.encode_obs(move_to_device(self.preprocessor.transform_obs(obs1), self.device))
                nv, np_ = self._frame_costs(z1, self._attr_pending["z_g"])
                nxt = {"attr_nv": _r(nv[0, 0]), "attr_np": _r(np_[0, 0])}
        out = super()._post_env_feedback(taken_actions, e_obses)
        rec = self._records[-1]
        if nxt is not None:
            rec.update(self._attr_pending["log"])
            rec.update(nxt)
        self._attr_pending = None
        self._attr_armed = rec.get("stag_w", 0) > 0
        if self._attr_armed and self.attr_save_enc and not self._attr_trig_saved:
            self._save_enc("trig")
            self._attr_trig_saved = True
        return out

    def _save_enc(self, tag):
        split = self._groups["pred"]
        delta = [(p.detach() - q).half().cpu() for p, q in zip(self._params[split:], self._init[split:])]
        out = Path("attr_enc")
        out.mkdir(exist_ok=True)
        torch.save({"iter": self.iter, "delta": delta}, out / f"{self._sample_idx:03d}_{tag}.pt")
