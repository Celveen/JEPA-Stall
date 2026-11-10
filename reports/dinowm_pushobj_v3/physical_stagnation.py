"""Preserve detector observation times while changing the MPC execution cadence."""
import math
import numpy as np
from planning.stagnation import StagnationMPCPlanner

def history_before_endpoint(observations, window, stride):
    """Return W observed frames ending one stride before the current endpoint."""
    length=next(iter(observations.values())).shape[1]
    assert all(x.shape[1]==length for x in observations.values())
    assert length > window*stride, (length,window,stride)
    indices=[length-1-i*stride for i in range(window,0,-1)]
    return [{k:np.array(v[:,j:j+1]) for k,v in observations.items()} for j in indices]

class PhysicalTimeStagnationMPCPlanner(StagnationMPCPlanner):
    def __init__(self,control_timing,**kwargs):
        timing=dict(control_timing)
        self.executed_controls=int(timing['executed_controls'])
        self.history_stride_controls=int(timing['history_stride_controls'])
        self.warmup_controls=int(timing['warmup_controls'])
        self.boost_controls=int(timing['boost_controls'])
        super().__init__(**kwargs)
        assert self.n_taken_actions*self.evaluator.frameskip==self.executed_controls
        assert self.stag_start==math.ceil(self.warmup_controls/self.executed_controls)
        assert self.stag_boost_replans==max(0,math.ceil(self.boost_controls/self.executed_controls)-1)
        assert self.stag_window*self.history_stride_controls<=self.executed_controls

    def _post_env_feedback(self,taken_actions,e_obses):
        # The evaluator already returns every simulated control frame. Sample those frames
        # for detection, without requesting any extra observation or intermediate action plan.
        self._hist=history_before_endpoint(e_obses,self.stag_window,self.history_stride_controls)
        out=super()._post_env_feedback(taken_actions,e_obses)
        self._records[-1].update(elapsed_environment_controls=(self.iter+1)*self.executed_controls,
                               stag_window_environment_controls=self.stag_window*self.history_stride_controls,
                               executed_environment_controls=self.executed_controls)
        return out
