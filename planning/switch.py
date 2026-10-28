"""Staged fixing: AdaJEPA fixes the model from the first replan, the ruler fix starts at a switch (E3 / E2).

The ruler fix is the stall regulariser of planning/stagnation.py (the stall penalty kappa and the proprio boost). Here
both act only once the episode has switched; until then the stall trigger is still computed and its memory still
records the stalled observations, so a switch activates the same memory the trigger-time arm would use. switch.mode
decides when the switch happens, per replan s (record step s, after the replan's env feedback):

    trigger    at the first replan whose stall trigger fires (identical to StagnationMPCPlanner)
    loss_new   the trigger fires and AdaJEPA's error on the newest, not yet trained, transition (loss_new_cur, mean over
               replans s-2..s) fell by less than theta over the last W replans, for K consecutive replans
    delay      the first replan with the trigger firing at least D replans after the first trigger

switch.freeze stops the world-model update from the switch on (probe mode none for the rest of the episode); with
kappa 0 and no boost that is the E2 intervention. The ruler takes effect from the replan after the switch, so every
replan up to the switch is bit-identical to a stag.kappa = 0 reference run. stag_episodes.jsonl records the switch.
"""

import numpy as np

from .stagnation import StagnationMPCPlanner

MODES = ("trigger", "loss_new", "delay")


class SwitchMPCPlanner(StagnationMPCPlanner):
    def __init__(self, switch=None, **kwargs):
        super().__init__(**kwargs)
        s = dict(switch or {})
        self.sw_mode = s.get("mode", "trigger")
        if self.sw_mode not in MODES:
            raise ValueError(f"switch.mode must be one of {MODES}, got {self.sw_mode!r}")
        self.sw_W = int(s.get("W", 3))
        self.sw_theta = float(s.get("theta", 0.05))
        self.sw_K = int(s.get("K", 1))
        self.sw_D = int(s.get("D", 10))
        self.sw_freeze = bool(s.get("freeze", False))
        self._probe_mode0 = self.probe_mode
        self._reset_switch()

    def _reset_switch(self):
        self._switched, self._sw_step, self._sw_run, self._loss_new = False, None, 0, {}
        self.probe_mode = self._probe_mode0

    def _plan_single(self, **kw):
        self._reset_switch()
        return super()._plan_single(**kw)

    def _episode_extra(self):
        return {"switch_step": self._sw_step, "switch_mode": self.sw_mode, "freeze": self.sw_freeze}

    # ------------------------------------------------------------------ switch
    def _switch_due(self, rec):
        s = rec["step"]
        if rec.get("loss_new_cur") is not None:
            self._loss_new[s] = rec["loss_new_cur"]
        armed = (rec.get("stag_w") or 0) > 0
        if self.sw_mode == "trigger":
            return armed
        if self.sw_mode == "delay":
            return armed and self._stag_first is not None and s >= self._stag_first + self.sw_D
        L = self._loss_new
        l3 = lambda t: float(np.mean([L[u] for u in (t - 2, t - 1, t)])) if all(u in L for u in (t - 2, t - 1, t)) else None  # noqa: E731
        now, then = l3(s), l3(s - self.sw_W)
        flat = now is not None and then is not None and (then - now) / max(then, 1e-12) < self.sw_theta
        self._sw_run = self._sw_run + 1 if (armed and flat) else 0
        return self._sw_run >= self.sw_K

    def _post_env_feedback(self, taken_actions, e_obses):
        out = super()._post_env_feedback(taken_actions, e_obses)
        rec = self._records[-1]
        if not self._switched and self._switch_due(rec):
            self._switched, self._sw_step = True, rec["step"]
            rec["switch"] = True
            if self.sw_freeze:
                self.probe_mode = "none"
        if not self._switched:
            self._boost_until = None  # the trigger may have armed the boost; the ruler waits for the switch
        return out

    def _build_mem(self):
        if not self._switched:
            self._mem = None
            return
        super()._build_mem()
