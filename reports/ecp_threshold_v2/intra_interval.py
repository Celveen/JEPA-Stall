"""Observe latent stagnation every five controls while retaining six CEM replans.

The environment already returns all frames in the executed action prefix. This
planner only samples those frames after a 25-control execution block and queues
one memory point for the next existing MPC call; it never adds a CEM call.
"""

import json
from pathlib import Path

import numpy as np

from physical_stagnation import PhysicalTimeStagnationMPCPlanner
from planning.escape import EscapeMPCPlanner


class IntraIntervalStagnationMPCPlanner(PhysicalTimeStagnationMPCPlanner):
    def _reset_stag(self):
        super()._reset_stag()
        self._first_detection_control = None
        self._first_intervention_control = None

    def _episode_extra(self):
        return {
            "first_detection_control": self._first_detection_control,
            "first_intervention_control": self._first_intervention_control,
            "detector_stride_controls": self.history_stride_controls,
        }

    def _post_env_feedback(self, taken_actions, e_obses):
        # Skip the endpoint-only detector, but retain the normal episode and
        # escape bookkeeping (escape.mode is none for this evaluation).
        out = EscapeMPCPlanner._post_env_feedback(self, taken_actions, e_obses)
        rec = self._records[-1]
        if self._stag_log is not None:
            rec.update(self._stag_log)
            self._stag_log = None

        stride = self.history_stride_controls
        window_controls = self.stag_window * stride
        endpoint = (self.iter + 1) * self.executed_controls
        length = next(iter(e_obses.values())).shape[1]
        assert all(v.shape[1] == length for v in e_obses.values())
        assert length == endpoint + 1, (length, endpoint)
        assert self.max_iter * self.executed_controls == 150
        assert self.executed_controls % stride == 0
        assert self.stag_gamma_hi > self.stag_gamma_lo

        def frame(control):
            return {k: np.array(v[:, control : control + 1]) for k, v in e_obses.items()}

        hits = []
        checks = []
        for control in range(endpoint - self.executed_controls + stride, endpoint + 1, stride):
            eligible = (control >= self.warmup_controls and endpoint < self.max_iter * self.executed_controls
                        and not bool(self.is_success[0]))
            row = {"sample_idx": self._sample_idx, "replan_completed": self.iter + 1,
                   "control": control, "eligible": eligible, "candidate": False,
                   "movement_ratio": None, "weight": 0.0}
            if control >= window_controls:
                now = frame(control)
                previous = frame(control - window_controls)
                zv, zp = self._encode([now, previous, self._obs_g])
                def distance(i, j):
                    return float(((zv[i] - zv[j]) ** 2).mean()
                                 + self.stag_alpha * ((zp[i] - zp[j]) ** 2).mean())
                ratio = (distance(0, 1) / max(distance(0, 2), 1e-12)) ** 0.5
                weight = min(max((self.stag_gamma_hi - ratio) /
                                 (self.stag_gamma_hi - self.stag_gamma_lo), 0.0), 1.0)
                row.update(movement_ratio=round(ratio, 6), weight=round(weight, 6),
                           candidate=bool(eligible and weight > 0))
                if row["candidate"]:
                    hits.append((weight, control, now))
            checks.append(row)

        if hits:
            # One memory per execution block keeps the regularizer scale
            # comparable to the endpoint-only detector; ties go to the first.
            weight, control, obs = max(hits, key=lambda hit: (hit[0], -hit[1]))
            self._mem_obs.append(obs)
            self._mem_w.append(weight)
            nxt = self.iter + 1
            if self._stag_first is None:
                self._stag_first = nxt
                self._first_detection_control = control
                self._first_intervention_control = endpoint
            if self.stag_prop_boost != 1.0:
                self._boost_until = nxt + self.stag_boost_replans
            rec.update(stag_r=next(row["movement_ratio"] for row in checks if row["control"] == control),
                       stag_w=round(weight, 4))
        rec.update(stag_mem_n=len(self._mem_w), intra_hits=len(hits),
                   elapsed_environment_controls=endpoint,
                   stag_window_environment_controls=window_controls,
                   executed_environment_controls=self.executed_controls)
        with Path("intra_detection.jsonl").open("a") as output:
            for row in checks:
                output.write(json.dumps(row) + "\n")
        return out
