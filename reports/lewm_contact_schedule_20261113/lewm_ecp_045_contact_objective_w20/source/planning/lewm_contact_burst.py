"""Keep contact guidance active for four consecutive replans after a stall trigger."""

import numpy as np

from .lewm_contact import LeWMContactMPCPlanner


class LeWMContactBurstMPCPlanner(LeWMContactMPCPlanner):
    def __init__(self, burst, **kwargs):
        super().__init__(**kwargs)
        self.burst_replans = int(burst["replans"])
        if self.burst_replans < 1:
            raise ValueError("A guidance burst must include at least one replan")

    def plan(self, obs_0, obs_g, actions=None):
        count = len(obs_0["visual"])
        self._burst_start = np.full(count, -1)
        self._burst_until = np.full(count, -1)
        return super().plan(obs_0, obs_g, actions=actions)

    def _eligible_for_bias(self):
        ongoing = self.iter <= self._burst_until
        start = self._pending & ~ongoing & ~self.is_success
        start &= self.iter - self._last_bias >= self.min_gap
        start &= self._bias_count < self.max_interventions
        self._burst_start[start] = self.iter
        self._burst_until[start] = self.iter + self.burst_replans - 1
        eligible = (self.iter <= self._burst_until) & ~self.is_success
        return eligible & (self._bias_count < self.max_interventions)

    def _plan_with_bias(self, obs_0, obs_g, actions=None, step=None):
        # Preserve the detector's original pending mask, which also gates the
        # separate proprioceptive objective. Only action-bias eligibility changes.
        result = super()._plan_with_bias(obs_0, obs_g, actions=actions, step=step)
        for entry in self._applied:
            sample = entry["sample_idx"]
            entry["burst_start_replan"] = int(self._burst_start[sample])
            entry["burst_end_replan"] = int(self._burst_until[sample])
            entry["detector_trigger_for_current_plan"] = bool(self._pending[sample])
        return result
