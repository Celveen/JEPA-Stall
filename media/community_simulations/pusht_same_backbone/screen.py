"""Audit paired outcomes and select fully visible same-checkpoint rescue candidates."""
import csv
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / 'eval_outputs/pusht_same_backbone_20261124'
OUT = Path(__file__).resolve().parent


def stop(data, index):
    length = data['action_len'][index]
    return 150 if not np.isfinite(length) else int(length) * 5


def errors(state, goal):
    return dict(joint_xy=float(np.linalg.norm(state[:4] - goal[:4])),
                object_xy=float(np.linalg.norm(state[2:4] - goal[2:4])),
                angle_deg=float(np.rad2deg(abs((state[4] - goal[4] + np.pi) % (2*np.pi) - np.pi))))


def success(state, goal):
    e = errors(state, goal)
    return e['joint_xy'] < 20 and e['angle_deg'] < 20


def main():
    rows=[]
    for condition in ['red_goal', 'red_object', 'red_agent']:
        paths=[RUNS/f'{condition}_{arm}' for arm in ['native','ours']]
        if not all((p/'complete.json').exists() for p in paths):
            continue
        b,o=[dict(np.load(p/'recorded_execution.npz')) for p in paths]
        checks=[json.loads((p/'frozen_model_check.json').read_text()) for p in paths]
        assert checks[0] == checks[1] and checks[0]['unchanged']
        episodes={r['sample_idx']:r for r in [json.loads(line) for line in (paths[1]/'lewm_ecp_episodes.jsonl').read_text().splitlines()]}
        for key in ['state_0', 'state_g', 'goal_rgb']:
            np.testing.assert_array_equal(b[key],o[key])
        for i in range(len(b['states'])):
            nb,no=stop(b,i),stop(o,i)
            bs,os=b['states'][i,:nb+1],o['states'][i,:no+1]
            np.testing.assert_array_equal(b['rgb'][i,0],o['rgb'][i,0])
            assert success(bs[-1],b['state_g'][i]) == b['successes'][i]
            assert success(os[-1],o['state_g'][i]) == o['successes'][i]
            xy=np.concatenate([bs[:,:2],os[:,:2]])
            margin=float(min(xy.min(),512-xy.max()))
            be,oe=errors(bs[-1],b['state_g'][i]),errors(os[-1],o['state_g'][i])
            first_trigger=episodes[i]['first_trigger']
            prefix=min(nb,no,5*first_trigger) if first_trigger is not None else min(nb,no)
            np.testing.assert_allclose(b['controls'][i,:prefix],o['controls'][i,:prefix],atol=1e-6,rtol=0)
            row=dict(condition=condition,target=i,native_success=bool(b['successes'][i]),ours_success=bool(o['successes'][i]),
                     native_controls=nb,ours_controls=no,agent_edge_margin_px=margin,
                     full_agent_visible=margin>=16,eligible=not b['successes'][i] and bool(o['successes'][i]) and margin>=16)
            row.update(first_trigger_replan=first_trigger,pretrigger_controls_verified=prefix)
            row.update({f'native_{k}':v for k,v in be.items()});row.update({f'ours_{k}':v for k,v in oe.items()})
            rows.append(row)
    if not rows:return
    with (OUT/'screening.csv').open('w') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    selected=sorted([r for r in rows if r['eligible']],key=lambda r:r['native_object_xy']-r['ours_object_xy'],reverse=True)
    print(json.dumps(selected,indent=2))
    (OUT/'candidates.json').write_text(json.dumps(selected,indent=2))


if __name__=='__main__':main()
