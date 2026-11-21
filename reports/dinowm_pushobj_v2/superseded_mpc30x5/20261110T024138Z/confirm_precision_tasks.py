"""Complete-task FP32 spot checks paired to BF16 sample 0 for every shape/model."""
import hashlib
import json
import os
import pickle
import subprocess
import time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np

R=Path('/home/appuser/workspace/project/multimodal-test/JEPA-Stall');B=Path(__file__).resolve().parent
OUT=B/'precision_confirmation';OUT.mkdir(exist_ok=True)
BF=R/'eval_outputs/dinowm_pushobj_3epoch_v2_cem200x10_bf16'
PY='/home/appuser/anaconda3/envs/ts/bin/python';GPU=4
targets=sorted(t for t in (R/'data/targets/EVAL50').glob('chunk_*.pkl') if int(t.stem.split('_')[1])%5==0)
assert len(targets)==7
state={'status':'running_fp32_checks','total_tasks':14,'completed_tasks':0,'gpu':GPU,'tasks':[]}
def update(**kw):
    state.update(kw,updated_utc=datetime.now(timezone.utc).isoformat());tmp=OUT/'status.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(OUT/'status.json')
def compare_ready():
    comparisons=[]
    for task in state['tasks']:
        if not task.get('complete'):continue
        reference=BF/f"{task['arm']}_native"/task['source_chunk']
        if not (reference/'complete.json').exists():continue
        fp=np.load(Path(task['output'])/'final_eval_results.npz');bf=np.load(reference/'final_eval_results.npz')
        for k in ['state_0','state_g']:np.testing.assert_array_equal(fp[k][0],bf[k][0])
        fsuccess=bool(fp['successes'][0]);bsuccess=bool(bf['successes'][0])
        comparisons.append({'arm':task['arm'],'chunk':task['source_chunk'],'sample_idx':0,'fp32_success':fsuccess,'bf16_success':bsuccess,'bf16_rescue':bsuccess and not fsuccess,'bf16_regression':fsuccess and not bsuccess})
    return comparisons
try:
    update()
    for target in targets:
        for arm in ['ordinary','augmented']:
            out=OUT/arm/target.stem;out.mkdir(parents=True,exist_ok=True)
            subset=out/'targets.pkl'
            data=pickle.loads(target.read_bytes());data={**data,'segments':data['segments'][:1]};subset.write_bytes(pickle.dumps(data))
            task={'arm':arm,'source_chunk':target.stem,'source_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'sample_idx':0,'output':str(out),'complete':False};state['tasks'].append(task)
            assert not (out/'launcher.json').exists(),'Existing partial confirmation requires inspection'
            while True:
                rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()
                free={int(x.split(',')[0]):int(x.split(',')[1]) for x in rows}
                if free[GPU]>=16000:break
                update(status='waiting_for_memory');time.sleep(30)
            cmd=[PY,'-u',str(B/'evaluate.py'),'--repo',str(R),'--checkpoint',str(R/'checkpoints/dinowm_pushobj_v1'/arm),'--epoch','3','--n-evals','1','--targets',str(subset),'--output',str(out),'--precision','fp32','--candidate-batch','64','--validation-replans','30','--memory-limit-gib','8']
            env={**os.environ,'CUDA_VISIBLE_DEVICES':str(GPU),'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'2','SDL_VIDEODRIVER':'dummy','SDL_AUDIODRIVER':'dummy','WANDB_MODE':'disabled'}
            with (out/'run.log').open('w') as log:
                proc=subprocess.Popen(cmd,cwd=R,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                (out/'launcher.json').write_text(json.dumps({'pid':proc.pid,'gpu':GPU,'command':cmd,'started_utc':datetime.now(timezone.utc).isoformat()},indent=2)+'\n')
                while proc.poll() is None:
                    update(status='running_fp32_checks',current_job=str(out),worker_pid=proc.pid,comparisons=compare_ready());time.sleep(30)
            assert proc.returncode==0 and (out/'complete.json').exists(),f'{out}/run.log'
            task['complete']=True;state['completed_tasks']+=1;update(comparisons=compare_ready())
    while True:
        comparisons=compare_ready();update(status='waiting_for_bf16_pairs',comparisons=comparisons)
        if len(comparisons)==14:break
        main=json.loads((B/'pipeline_status.json').read_text())
        if main['status']=='failed':raise RuntimeError('Main BF16 queue failed before all precision pairs were available')
        time.sleep(30)
    summary={'status':'complete','n':14,'fp32_successes':sum(x['fp32_success'] for x in comparisons),'bf16_successes':sum(x['bf16_success'] for x in comparisons),'bf16_rescues':sum(x['bf16_rescue'] for x in comparisons),'bf16_regressions':sum(x['bf16_regression'] for x in comparisons),'paired':comparisons,'scope':'Native frozen planning; first task of each of seven shapes for both checkpoints. Full 30-replan cap; diagnostic subset, not proof of population accuracy equivalence. Excluded from main benchmark totals.'}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');update(status='complete',summary=str(OUT/'summary.json'))
except BaseException as e:
    update(status='failed',error=repr(e));raise
