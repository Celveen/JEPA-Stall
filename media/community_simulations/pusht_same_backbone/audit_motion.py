"""Verify visibility also between recorded control frames, at each Pymunk physics step."""
import importlib.util
import json
import os
from pathlib import Path
import numpy as np
from screen import ROOT, RUNS, OUT, stop

os.environ.setdefault('SDL_VIDEODRIVER','dummy')
os.environ.setdefault('SDL_AUDIODRIVER','dummy')
spec=importlib.util.spec_from_file_location('native_pusht', ROOT/'reports/lewm_contact_schedule_20261113/lewm_ecp_045_contact_objective/source/env/pusht/pusht_env.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
checks=[]
for condition,index in [('red_object',29),('red_goal',17)]:
    for arm in ['native','ours']:
        z=np.load(RUNS/f'{condition}_{arm}'/'recorded_execution.npz')
        n=stop(z,index)
        controls=z['controls'][index,:n]
        reference=z['states'][index,:n+1]
        env=module.PushTEnv(with_velocity=True,with_target=True,shape='T',reset_to_state=z['state_0'][index],
                           **{dict(red_object='color',red_goal='goal_color')[condition]:'Red'})
        env.seed(100*index+1)
        _,initial=env.reset()
        positions=[np.asarray(env.agent.position).copy()]
        states=[initial]
        original=env.space.step
        def traced(dt):
            original(dt)
            positions.append(np.asarray(env.agent.position).copy())
        env.space.step=traced
        for action in controls:
            _,_,_,info=env.step(action)
            states.append(info['state'])
        maximum=float(np.max(np.abs(np.asarray(states)-reference)))
        xy=np.asarray(positions)
        margin=float(min(xy.min(),512-xy.max()))
        assert maximum < .001, (condition,arm,maximum)
        assert margin >= 16, (condition,arm,margin)
        checks.append(dict(condition=condition,target=index,arm=arm,physics_samples=len(positions),
                           minimum_agent_center_edge_margin_px=margin,maximum_replay_state_error=maximum))
        env.close()
(OUT/'physics_visibility.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(checks,indent=2))
