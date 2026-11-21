#!/usr/bin/env python3
"""Five GPU workers; keep each precision/budget in an independent result tree."""
import argparse
import hashlib
import json
import os
import queue
import subprocess
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from omegaconf import OmegaConf

R=Path('/home/appuser/workspace/project/multimodal-test/JEPA-Stall')
B=Path(__file__).resolve().parent
PY='/home/appuser/anaconda3/envs/ts/bin/python'
ARMS=[('ordinary',False),('ordinary',True),('augmented',False),('augmented',True)]
LOCK=threading.RLock();STOP=threading.Event();STATE={}
def utc():return datetime.now(timezone.utc).isoformat()
def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
def update(**kw):
    with LOCK:
        STATE.update(kw,updated_utc=utc());write(B/'pipeline_status.json',STATE)
def heartbeat():
    while not STOP.wait(30):update()

def summarize(out,targets,precision):
    results={};paired={}
    for arm,ecp in ARMS:
        label=arm+('_ecp' if ecp else '_native');shapes={}
        for target in targets:
            shape=target.stem.split('_',2)[2]
            with np.load(out/label/target.stem/'final_eval_results.npz') as data:
                shapes.setdefault(shape,[]).extend(data['successes'].astype(bool).tolist())
        assert len(shapes)==7 and all(len(v)==50 for v in shapes.values())
        row={'by_shape':{k:{'n':len(v),'successes':sum(v),'success_rate':float(np.mean(v))} for k,v in shapes.items()}}
        for group,names in [('seen',['T','L','Z','plus']),('unseen',['I','small_tee','square']),('all',list(shapes))]:
            values=[v for name in names for v in shapes[name]]
            row[group]={'n':len(values),'successes':sum(values),'success_rate':float(np.mean(values))}
        results[label]=row
    for arm in ['ordinary','augmented']:
        pair={'rescues':[],'regressions':[],'pre_trigger_mismatches':[],'targets_equal':True}
        for target in targets:
            base=out/f'{arm}_native'/target.stem;ecpdir=out/f'{arm}_ecp'/target.stem
            with np.load(base/'final_eval_results.npz') as a,np.load(ecpdir/'final_eval_results.npz') as b:
                for key in ['state_0','state_g']:np.testing.assert_array_equal(a[key],b[key])
                episodes={x['sample_idx']:x for x in map(json.loads,(ecpdir/'stag_episodes.jsonl').read_text().splitlines())}
                for i,(sa,sb) in enumerate(zip(a['successes'],b['successes'])):
                    key=f'{target.stem}:{i}'
                    if sb and not sa:pair['rescues'].append(key)
                    if sa and not sb:pair['regressions'].append(key)
                    t=episodes[i]['first_trigger'];n=min(a['actions'].shape[1],b['actions'].shape[1])
                    if t is not None:n=min(n,t*5)
                    if not np.array_equal(a['actions'][i,:n],b['actions'][i,:n]):pair['pre_trigger_mismatches'].append(key)
        paired[arm]=pair
    valid=all(not x['pre_trigger_mismatches'] for x in paired.values())
    write(out/'summary.json',{'status':'complete' if valid else 'pairing_check_failed','training_epochs':3,'precision':precision,'results':results,'paired':paired,'all_pre_trigger_pairs_identical':valid,'protocol':str(B/'PROTOCOL.md')})
    assert valid,'Paired pre-trigger action mismatch; inspect summary before use'

def main():
    p=argparse.ArgumentParser();p.add_argument('--precision',choices=['fp32','bf16'],required=True)
    p.add_argument('--gpus',default='0,1,3,5,6');args=p.parse_args()
    gpus=[int(x) for x in args.gpus.split(',')];assert len(gpus)==len(set(gpus))
    precision=args.precision;validation=json.loads((B/'validation_summary.json').read_text())
    assert validation['integration_passed']
    if precision=='bf16':assert validation['bf16_implementation_approved']
    targets=sorted((R/'data/targets/EVAL50').glob('chunk_*.pkl'));assert len(targets)==35
    out=R/f'eval_outputs/dinowm_pushobj_3epoch_v3_cem200x10_mpc6x25_{precision}';out.mkdir(exist_ok=True)
    weights={}
    for arm,_ in ARMS:
        if arm in weights:continue
        checkpoint=R/'checkpoints/dinowm_pushobj_v1'/arm/'checkpoints/model_3.pth'
        assert json.loads((checkpoint.parent.parent/'complete.json').read_text())['epoch']==3
        weights[arm]=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    source_hashes={name:hashlib.sha256((B/name).read_bytes()).hexdigest() for name in ['evaluate.py','cem_memory.py','physical_stagnation.py']}
    assert source_hashes==validation['source_sha256'],'Evaluator changed after integration validation'
    shared_path=R/'conf/dinowm_pushobj_official_cadence.yaml'
    assert hashlib.sha256(shared_path.read_bytes()).hexdigest()==validation['protocol_sha256']
    shared=OmegaConf.load(shared_path)
    assert (shared.planner.max_iter,shared.planner.n_taken_actions,shared.planner.sub_planner.horizon)==(6,25,25)
    assert (shared.planner.sub_planner.num_samples,shared.planner.sub_planner.opt_steps,shared.planner.sub_planner.topk)==(200,10,30)
    protocol={'epochs':3,'precision':precision,'memory_limit_gib':8,'candidate_batch':64,'cem_candidates':200,'cem_iterations':10,'cem_elites':30,'max_mpc_replans':6,'executed_physical_steps_per_replan':25,'executed_model_steps_per_replan':5,'max_environment_controls':150,'protocol_id':'dinowm_pushobj_cem200x10_mpc6x25_v1','protocol_sha256':hashlib.sha256((R/'conf/dinowm_pushobj_official_cadence.yaml').read_bytes()).hexdigest(),'horizon_physical_steps':25,'objective_mode':'last','gpus':gpus,'conditions':[{'data':arm,'ecp':ecp} for arm,ecp in ARMS],'targets':[{'path':str(t),'sha256':hashlib.sha256(t.read_bytes()).hexdigest()} for t in targets],'checkpoint_sha256':weights,'source_sha256':source_hashes,'n_evals_per_chunk':10,'pytorch_memory_cap_gib':8,'output':str(out)}
    manifest=out/'evaluation_queue.json'
    if manifest.exists():assert json.loads(manifest.read_text())==protocol,'Existing output has different configuration'
    else:write(manifest,protocol)
    write(B/'evaluation_queue.json',protocol);jobs=queue.Queue();failures=[]
    STATE.update(status='evaluating',target_epochs=3,completed_chunks=0,total_chunks=140,episodes_per_condition=350,precision=precision,output=str(out),workers={},failures=failures)
    ordered=sorted(targets,key=lambda t:(int(t.stem.split('_')[1])%5,int(t.stem.split('_')[1])//5))
    for target in ordered:
        for arm,ecp in ARMS:jobs.put((arm,ecp,target))
    update();threading.Thread(target=heartbeat,daemon=True).start()
    def worker(gpu):
        while True:
            try:arm,ecp,target=jobs.get_nowait()
            except queue.Empty:return
            label=arm+('_ecp' if ecp else '_native');destination=out/label/target.stem
            try:
                marker=destination/'complete.json'
                expected={'epoch':3,'n':10,'ecp':ecp,'is_smoke':False,'is_validation':False,'precision':precision,'memory_limit_gib':8,'candidate_batch':64,'cem_candidates':200,'cem_iterations':10,'cem_elites':30,'max_mpc_replans':6,'executed_physical_steps_per_replan':25,'executed_model_steps_per_replan':5,'max_environment_controls':150,'protocol_id':'dinowm_pushobj_cem200x10_mpc6x25_v1','protocol_sha256':hashlib.sha256((R/'conf/dinowm_pushobj_official_cadence.yaml').read_bytes()).hexdigest(),'objective_mode':'last','checkpoint_sha256':weights[arm],'source_sha256':source_hashes,'targets_sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
                if not marker.exists():
                    destination.mkdir(parents=True,exist_ok=True)
                    assert not (destination/'launcher.json').exists(),f'Partial output requires inspection: {destination}'
                    while True:
                        free={int(x.split(',')[0]):int(x.split(',')[1]) for x in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()}
                        if free[gpu]>=16000:break
                        with LOCK:STATE['workers'][str(gpu)]={'status':'waiting_for_memory','job':str(destination)};update()
                        time.sleep(30)
                    cmd=[PY,'-u',str(B/'evaluate.py'),'--repo',str(R),'--checkpoint',str(R/'checkpoints/dinowm_pushobj_v1'/arm),'--epoch','3','--n-evals','10','--targets',str(target),'--output',str(destination),'--precision',precision,'--candidate-batch','64']
                    if ecp:cmd.append('--ecp')
                    env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'2','SDL_VIDEODRIVER':'dummy','SDL_AUDIODRIVER':'dummy','WANDB_MODE':'disabled'}
                    with (destination/'run.log').open('w') as log:
                        proc=subprocess.Popen(cmd,cwd=R,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                        info={'status':'running','pid':proc.pid,'gpu':gpu,'job':str(destination),'command':cmd,'started_utc':utc()}
                        write(destination/'launcher.json',info)
                        with LOCK:STATE['workers'][str(gpu)]=info;update()
                        code=proc.wait()
                    assert code==0 and marker.exists(),f'Evaluation exit {code}: {destination}/run.log'
                meta=json.loads(marker.read_text())
                for key,value in expected.items():assert meta.get(key)==value,f'{destination}: mismatch {key}'
                assert meta['frozen_weights_unchanged']
                with LOCK:
                    STATE['completed_chunks']+=1;STATE['workers'][str(gpu)]={'status':'between_jobs','last_completed':str(destination)};update()
                print(json.dumps({'completed':str(destination),'gpu':gpu,'successes':meta['successes'],'seconds':meta['wall_seconds']}),flush=True)
            except Exception:
                with LOCK:
                    failures.append({'job':str(destination),'error':traceback.format_exc()});STATE['workers'][str(gpu)]={'status':'failed','job':str(destination)};update()
                return
            finally:jobs.task_done()
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:list(pool.map(worker,gpus))
    if failures:raise RuntimeError(json.dumps(failures))
    summarize(out,targets,precision);update(status='complete',summary=str(out/'summary.json'));STOP.set()

if __name__=='__main__':
    try:main()
    except BaseException:update(status='failed',error=traceback.format_exc());STOP.set();raise
