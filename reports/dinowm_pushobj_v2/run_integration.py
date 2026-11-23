"""Paired short simulator checks using the full new CEM budget."""
import hashlib
import json
import os
import pickle
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

R=Path('/home/appuser/workspace/project/multimodal-test/JEPA-Stall');B=Path(__file__).resolve().parent
V=B/'validation';V.mkdir(exist_ok=True)
PY='/home/appuser/anaconda3/envs/ts/bin/python'
source=[R/'data/targets/EVAL50/chunk_00_I.pkl',R/'data/targets/EVAL50/chunk_30_T.pkl']
loaded=[pickle.loads(p.read_bytes()) for p in source]
assert loaded[0]['traj_len']==loaded[1]['traj_len']
merged={**loaded[0],'segments':[x['segments'][0] for x in loaded]}
target=V/'paired_I_T_targets.pkl';target.write_bytes(pickle.dumps(merged))
(V/'target_provenance.json').write_text(json.dumps({'kind':'integration_only','samples':[{'source':str(p),'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'index':0} for p in source]},indent=2)+'\n')
jobs={0:[('ordinary','fp32',False)],1:[('ordinary','fp32',True)],3:[('augmented','fp32',False)],5:[('augmented','fp32',True)],6:[('ordinary','bf16',False),('ordinary','bf16',True),('augmented','bf16',False),('augmented','bf16',True)]}
def worker(gpu):
    for arm,precision,ecp in jobs[gpu]:
        out=V/f'{arm}_{precision}_{"ecp" if ecp else "native"}';out.mkdir(exist_ok=True)
        cmd=[PY,'-u',str(B/'evaluate.py'),'--repo',str(R),'--checkpoint',str(R/'checkpoints/dinowm_pushobj_v1'/arm),'--epoch','3','--n-evals','2','--targets',str(target),'--output',str(out),'--precision',precision,'--candidate-batch','64','--validation-replans','4']
        if ecp:cmd.extend(['--ecp','--forced-trigger'])
        env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'2','WANDB_MODE':'disabled','SDL_VIDEODRIVER':'dummy','SDL_AUDIODRIVER':'dummy'}
        with (out/'run.log').open('w') as f:
            proc=subprocess.Popen(cmd,cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
            (out/'launcher.json').write_text(json.dumps({'pid':proc.pid,'gpu':gpu,'command':cmd,'started_utc':datetime.now(timezone.utc).isoformat()},indent=2)+'\n')
            code=proc.wait()
        assert code==0,f'{out}/run.log: exit {code}'
        print(json.dumps({'complete':str(out),'gpu':gpu}),flush=True)
if '--check-only' not in sys.argv:
    with ThreadPoolExecutor(max_workers=5) as pool:list(pool.map(worker,jobs))
checks=[];comparisons=[]
for arm in ['ordinary','augmented']:
    for precision in ['fp32','bf16']:
        native=V/f'{arm}_{precision}_native';ecp=V/f'{arm}_{precision}_ecp'
        aa=np.load(native/'final_eval_results.npz');bb=np.load(ecp/'final_eval_results.npz')
        ca=json.loads((native/'complete.json').read_text());cb=json.loads((ecp/'complete.json').read_text())
        for k in ['state_0','state_g']:np.testing.assert_array_equal(aa[k],bb[k])
        np.testing.assert_array_equal(aa['actions'][:,:2],bb['actions'][:,:2])
        rows=[json.loads(x) for x in (ecp/'per_iter_logs.json').read_text().splitlines()]
        episodes=[json.loads(x) for x in (ecp/'stag_episodes.jsonl').read_text().splitlines()]
        assert all(x['first_trigger']==2 for x in episodes)
        # An optimized plan may already lie outside the repulsion region and have zero penalty.
        assert any('stag_pen' in x for x in rows)
        assert all(np.isfinite(x['stag_pen']) and x['stag_pen']>=0 for x in rows if 'stag_pen' in x)
        assert any(x.get('stag_boost',False) for x in rows)
        assert ca['frozen_weights_unchanged'] and cb['frozen_weights_unchanged'] and ca['model_sha256']==cb['model_sha256']
        assert np.isfinite(aa['actions']).all() and np.isfinite(bb['actions']).all()
        for c in [ca,cb]:assert (c['cem_candidates'],c['cem_iterations'],c['cem_elites'],c['is_validation'])==(200,10,30,True)
        checks.append({'arm':arm,'precision':precision,'paired_targets_equal':True,'pretrigger_actions_identical':True,'forced_trigger':2,'penalty_evaluated':True,'proprio_boost_exercised':True,'selected_plan_positive_penalty_records':sum(x.get('stag_pen',0)>0 for x in rows),'weights_unchanged':True,'native_seconds':ca['wall_seconds'],'ecp_seconds':cb['wall_seconds']})
    fp=np.load(V/f'{arm}_fp32_native/final_eval_results.npz');bf=np.load(V/f'{arm}_bf16_native/final_eval_results.npz')
    comparisons.append({'arm':arm,'actions_identical':bool(np.array_equal(fp['actions'],bf['actions'])),'action_rms_difference':float(np.sqrt(np.mean((fp['actions']-bf['actions'])**2))),'fp32_short_check_successes':fp['successes'].tolist(),'bf16_short_check_successes':bf['successes'].tolist(),'not_a_full_task_success_comparison':True})
static=[json.loads((V/f'{arm}_precision.json').read_text()) for arm in ['ordinary','augmented']]
assert all(x['frozen_weights_unchanged'] and all(y['actions_finite'] for y in x['rows']) for x in static)
memory=json.loads((B/'memory_validation.json').read_text())
assert memory['passed'] and memory['stay_penalty']>0 and memory['goal_penalty']==0
summary={'integration_passed':True,'bf16_implementation_approved':True,'cross_precision_actions_equivalent':False,'closed_loop_accuracy_equivalence_established':False,'scope':'Implementation validation: real I/T starts, two checkpoints, full 200x10 CEM, 4 replans; forced ECP activation. This does not establish full-task FP32/BF16 accuracy equivalence. BF16 full runs form a separate homogeneous protocol.','checks':checks,'cross_precision_short_rollouts':comparisons,'full_cem_saved_state_checks':static,'memory_and_penalty_validation':memory,'updated_utc':datetime.now(timezone.utc).isoformat()}
(B/'validation_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2),flush=True)
