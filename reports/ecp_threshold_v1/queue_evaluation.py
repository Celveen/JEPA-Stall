#!/usr/bin/env python3
"""Three authorized validations; a native control accompanies the cadence change."""
import argparse,hashlib,json,os,queue,subprocess,threading,time,traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from pathlib import Path
from collections import Counter
import numpy as np
from evaluate import configuration,CONDITIONS
R=Path('/home/appuser/workspace/project/multimodal-test/JEPA-Stall');B=Path(__file__).resolve().parent
PY='/home/appuser/anaconda3/envs/ts/bin/python'
OUT=R/'eval_outputs/ecp_threshold_v1'
LOCK=threading.RLock();DONE=threading.Event();STATE={};JOBS=queue.Queue();OPENED=set();PREFLIGHT={};TARGETS=[]

def utc():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(p)
def update(**kw):
    with LOCK:STATE.update(kw,updated_utc=utc());write(B/'pipeline_status.json',STATE)
def heartbeat():
    while not DONE.wait(30):update()
def rows(p):return [json.loads(l)for l in p.read_text().splitlines()if l.strip()]
def baseline(condition,chunk):
    if condition=='adajepa_aug_g030':return R/'eval_outputs/stag_adajepa_bias16k_it40/ref'/chunk
    if condition=='dinowm_aug_g030':return R/'eval_outputs/dinowm_pushobj_3epoch_v3_cem200x10_mpc6x25_bf16/augmented_native'/chunk
    return OUT/'dinowm_aug_dense_native'/chunk

def compare(condition,chunk):
    current=OUT/condition/chunk;ref=baseline(condition,chunk)
    episodes={e['sample_idx']:e for e in rows(current/'stag_episodes.jsonl')};rescues=[];regressions=[];mismatch=[]
    with np.load(current/'final_eval_results.npz') as a,np.load(ref/'final_eval_results.npz') as b:
        for key in ['state_0','state_g']:np.testing.assert_array_equal(a[key],b[key])
        if condition=='adajepa_aug_g030':
            logs=rows(ref/'per_iter_logs.json');native=[any(v['sample_idx']==i and v['step']<=30 and v['success']for v in logs)for i in range(10)]
        else:native=b['successes'].tolist()
        per_plan=5 if condition=='dinowm_aug_g030' else 1
        for i,(sa,sb) in enumerate(zip(a['successes'],native)):
            key=f'{chunk}:{i}';t=episodes[i]['first_trigger'];n=min(a['actions'].shape[1],b['actions'].shape[1],30)
            if t is not None:n=min(n,t*per_plan)
            if not np.array_equal(a['actions'][i,:n],b['actions'][i,:n]):mismatch.append(key)
            if sa and not sb:rescues.append(key)
            if sb and not sa:regressions.append(key)
    check=dict(targets_equal=True,pretrigger_actions_equal=not mismatch,pretrigger_mismatches=mismatch,rescues=rescues,regressions=regressions,reference=str(ref),native_successes=sum(native))
    write(current/'pairing_check.json',check)
    assert not mismatch,f'Pre-trigger action mismatch: {mismatch}; inspect {current}'
    return check

def release(condition):
    if condition in OPENED:return
    OPENED.add(condition)
    for target in TARGETS[1:]:JOBS.put((condition,target))
    STATE['preflight'][condition]='passed';update(status='evaluating' if len(OPENED)==len(CONDITIONS) else 'preflight')

def completed(condition,target):
    with LOCK:
        chunk=target.stem
        if target==TARGETS[0]:
            PREFLIGHT[condition]=True
            if condition in ['adajepa_aug_g030','dinowm_aug_g030']:
                compare(condition,chunk);release(condition)
            elif all(PREFLIGHT.get(c)for c in ['dinowm_aug_dense_native','dinowm_aug_dense_g030']):
                compare('dinowm_aug_dense_g030',chunk)
                for c in ['dinowm_aug_dense_native','dinowm_aug_dense_g030']:release(c)
        elif condition in ['adajepa_aug_g030','dinowm_aug_g030']:compare(condition,chunk)
        elif condition.startswith('dinowm_aug_dense'):
            if all((OUT/c/chunk/'complete.json').exists()for c in ['dinowm_aug_dense_native','dinowm_aug_dense_g030']):compare('dinowm_aug_dense_g030',chunk)
        STATE['completed_chunks']+=1;STATE['by_condition'][condition]['completed_chunks']+=1
        if STATE['completed_chunks']==STATE['total_chunks']:DONE.set()
        update()

def summarize():
    result={}
    for c in CONDITIONS:
        byshape={};events=Counter();successes=0;n=0;triggered=0;positive=0;boost=0;seconds=0.;paired={'rescues':[],'regressions':[],'pretrigger_mismatches':[]};firsts=Counter()
        for t in TARGETS:
            p=OUT/c/t.stem;m=json.loads((p/'complete.json').read_text());shape=t.stem.split('_',2)[2]
            e=rows(p/'stag_episodes.jsonl');n+=m['n'];successes+=m['successes'];triggered+=m['triggered'];positive+=m['positive_penalty_records'];boost+=m['boosted_plan_records'];seconds+=m['wall_seconds']
            ss=byshape.setdefault(shape,{'n':0,'successes':0});ss['n']+=m['n'];ss['successes']+=m['successes']
            for v in e:
                if v['first_trigger'] is not None:firsts[str(v['first_trigger'])]+=1
            if c!='dinowm_aug_dense_native':
                pair=compare(c,t.stem)
                for k in paired:paired[k].extend(pair[k])
        assert n==350 and len(byshape)==7
        for ss in byshape.values():ss['success_rate']=ss['successes']/ss['n'];assert ss['n']==50
        result[c]=dict(n=n,successes=successes,success_rate=successes/n,triggered=triggered,trigger_rate=triggered/n,first_trigger_histogram=dict(firsts),by_shape=byshape,positive_penalty_records=positive,boosted_plan_records=boost,worker_seconds=seconds,paired=paired if c!='dinowm_aug_dense_native' else None)
    write(OUT/'summary.json',dict(status='complete',results=result,protocol=str(B/'PROTOCOL.md'),note='Exploratory threshold and cadence follow-up; same existing targets, not an independent calibration/test split.'))

def worker(gpu):
    while not DONE.is_set():
        try:condition,target=JOBS.get(timeout=2)
        except queue.Empty:continue
        dest=OUT/condition/target.stem
        try:
            marker=dest/'complete.json'
            if not marker.exists():
                assert not (dest/'launcher.json').exists(),f'Partial output requires inspection: {dest}'
                while not DONE.is_set():
                    free={int(x.split(',')[0]):int(x.split(',')[1])for x in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()}
                    if free[gpu]>=14000:break
                    with LOCK:STATE['workers'][str(gpu)]=dict(status='waiting_for_memory',condition=condition,chunk=target.stem);update()
                    time.sleep(15)
                if DONE.is_set():return
                dest.mkdir(parents=True,exist_ok=True)
                cmd=[PY,'-u',str(B/'evaluate.py'),'--condition',condition,'--targets',str(target),'--output',str(dest)]
                env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'2','SDL_VIDEODRIVER':'dummy','SDL_AUDIODRIVER':'dummy','WANDB_MODE':'disabled'}
                with (dest/'run.log').open('w')as f:
                    proc=subprocess.Popen(cmd,cwd=R,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                    info=dict(status='running',pid=proc.pid,gpu=gpu,condition=condition,chunk=target.stem,command=cmd,started_utc=utc());write(dest/'launcher.json',info)
                    with LOCK:STATE['workers'][str(gpu)]=info;update()
                    code=proc.wait()
                assert code==0 and marker.exists(),f'Exit {code}: {dest}/run.log'
            m=json.loads(marker.read_text());assert m['condition']==condition and m['n']==10 and m['thresholds']==[.1,.3] and m['targets_sha256']==sha(target)
            assert m['checkpoint_file_unchanged']
            if condition.startswith('dinowm'):assert m['frozen_weights_unchanged']
            assert m['source_sha256']['reports/ecp_threshold_v1/evaluate.py']==sha(B/'evaluate.py')
            completed(condition,target)
            with LOCK:STATE['workers'][str(gpu)]=dict(status='between_jobs',last_completed=str(dest));update()
            print(json.dumps(dict(completed=str(dest),successes=m['successes'],triggered=m['triggered'],seconds=m['wall_seconds'])),flush=True)
        except Exception:
            with LOCK:STATE['failures'].append(dict(job=str(dest),error=traceback.format_exc()));STATE['workers'][str(gpu)]=dict(status='failed',job=str(dest));update(status='failed');DONE.set()
            return
        finally:JOBS.task_done()

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpus',default='0,1,3,4,5,6,7');args=p.parse_args();gpus=[int(x)for x in args.gpus.split(',')];assert 2 not in gpus and len(gpus)==len(set(gpus))
    global TARGETS
    TARGETS=sorted((R/'data/targets/EVAL50').glob('chunk_*.pkl'));assert len(TARGETS)==35
    OUT.mkdir(parents=True,exist_ok=True)
    sources={str(p.relative_to(R)):sha(p)for p in [B/'evaluate.py',B/'queue_evaluation.py',R/'planning/stagnation.py',R/'reports/dinowm_pushobj_v3/cem_memory.py',R/'reports/dinowm_pushobj_v3/physical_stagnation.py']}
    configs={c:json.loads(json.dumps(__import__('omegaconf').OmegaConf.to_container(configuration(c,TARGETS[0],OUT/c/TARGETS[0].stem),resolve=True)))for c in CONDITIONS}
    receipt=dict(authorized_on='2026-11-11',condition_order=list(CONDITIONS),gpus=gpus,excluded_gpus=[2],episodes_per_condition=350,total_task_runs=1400,targets=[dict(path=str(t),sha256=sha(t))for t in TARGETS],checkpoints={c:sha(Path(configs[c]['ckpt_base_path'])/'checkpoints/model_3.pth')for c in CONDITIONS},resolved_first_chunk_configs=configs,source_sha256=sources,third_validation_confirmed='User selected augmented frozen DINO-WM, 30 plans x 5 controls, threshold 0.3; same-cadence native control included.',protocol=str(B/'PROTOCOL.md'))
    path=OUT/'evaluation_manifest.json'
    if path.exists():assert json.loads(path.read_text())==receipt,'Existing run has different configuration'
    else:write(path,receipt)
    write(B/'evaluation_manifest.json',receipt)
    STATE.update(status='preflight',started_utc=utc(),completed_chunks=0,total_chunks=140,episodes_per_condition=350,total_task_runs=1400,by_condition={c:dict(completed_chunks=0,total_chunks=35)for c in CONDITIONS},preflight={c:'pending'for c in CONDITIONS},workers={},failures=[],output=str(OUT))
    for c in CONDITIONS:JOBS.put((c,TARGETS[0]))
    update();threading.Thread(target=heartbeat,daemon=True).start()
    with ThreadPoolExecutor(max_workers=len(gpus))as pool:list(pool.map(worker,gpus))
    if STATE['failures']:raise RuntimeError(json.dumps(STATE['failures']))
    assert STATE['completed_chunks']==140
    summarize();update(status='complete',summary=str(OUT/'summary.json'));DONE.set()

if __name__=='__main__':
    try:main()
    except BaseException:update(status='failed',error=traceback.format_exc());DONE.set();raise
