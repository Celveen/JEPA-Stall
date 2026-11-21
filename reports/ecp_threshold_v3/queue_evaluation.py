#!/usr/bin/env python3
"""Queue matched AdaJEPA 0.60 and in-house frozen 0.45 evaluations."""
import argparse,hashlib,json,os,queue,shutil,subprocess,threading,time,traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from pathlib import Path
from collections import Counter
import numpy as np
from evaluate import configuration
CONDITIONS=('adajepa_aug_g060','frozen_aug_g045')
R=Path('/home/appuser/workspace/project/multimodal-test/JEPA-Stall');B=Path(__file__).resolve().parent
PY='/home/appuser/anaconda3/envs/ts/bin/python'
OUT=R/'eval_outputs/ecp_threshold_v3'
LOCK=threading.RLock();DONE=threading.Event();STATE={};JOBS=queue.Queue();TARGETS=[]

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
    name='stag_frozen_bias16k_it40' if condition.startswith('frozen') else 'stag_adajepa_bias16k_it40'
    return R/'eval_outputs'/name/'ref'/chunk

def compare(condition,chunk):
    current=OUT/condition/chunk;ref=baseline(condition,chunk)
    episodes={e['sample_idx']:e for e in rows(current/'stag_episodes.jsonl')};rescues=[];regressions=[];mismatch=[]
    with np.load(current/'final_eval_results.npz') as a,np.load(ref/'final_eval_results.npz') as b:
        for key in ['state_0','state_g']:np.testing.assert_array_equal(a[key],b[key])
        logs=rows(ref/'per_iter_logs.json');native=[any(v['sample_idx']==i and v['step']<=30 and v['success']for v in logs)for i in range(10)]
        per_plan=1
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

def completed(condition,target):
    with LOCK:
        chunk=target.stem
        compare(condition,chunk)
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
            pair=compare(c,t.stem)
            for k in paired:paired[k].extend(pair[k])
        assert n==350 and len(byshape)==7
        for ss in byshape.values():ss['success_rate']=ss['successes']/ss['n'];assert ss['n']==50
        result[c]=dict(n=n,successes=successes,success_rate=successes/n,triggered=triggered,trigger_rate=triggered/n,first_trigger_histogram=dict(firsts),by_shape=byshape,positive_penalty_records=positive,boosted_plan_records=boost,worker_seconds=seconds,paired=paired)
    write(OUT/'summary.json',dict(status='complete',results=result,protocol=str(B/'PROTOCOL.md'),note='Exploratory threshold comparison on the same fixed task set.'))

def worker(gpu,slot):
    worker_id=f'{gpu}:{slot}'
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
                    with LOCK:STATE['workers'][worker_id]=dict(status='waiting_for_memory',gpu=gpu,condition=condition,chunk=target.stem);update()
                    time.sleep(15)
                if DONE.is_set():return
                dest.mkdir(parents=True,exist_ok=True)
                cmd=[PY,'-u',str(B/'evaluate.py'),'--condition',condition,'--targets',str(target),'--output',str(dest)]
                env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'2','SDL_VIDEODRIVER':'dummy','SDL_AUDIODRIVER':'dummy','WANDB_MODE':'disabled'}
                with (dest/'run.log').open('w')as f:
                    proc=subprocess.Popen(cmd,cwd=R,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                    info=dict(status='running',pid=proc.pid,gpu=gpu,condition=condition,chunk=target.stem,command=cmd,started_utc=utc());write(dest/'launcher.json',info)
                    with LOCK:STATE['workers'][worker_id]=info;update()
                    code=proc.wait()
                assert code==0 and marker.exists(),f'Exit {code}: {dest}/run.log'
            m=json.loads(marker.read_text());thresholds=[.15,.45] if condition.startswith('frozen') else [.20,.60]
            assert m['condition']==condition and m['n']==10 and m['thresholds']==thresholds and m['targets_sha256']==sha(target)
            assert m['checkpoint_file_unchanged']
            assert m['source_sha256']['evaluate.py']==sha(B/'evaluate.py')
            if condition.startswith('frozen'):assert m['frozen_weights_unchanged'] and not m['online_adaptation']
            completed(condition,target)
            with LOCK:STATE['workers'][worker_id]=dict(status='between_jobs',gpu=gpu,last_completed=str(dest));update()
            print(json.dumps(dict(completed=str(dest),successes=m['successes'],triggered=m['triggered'],seconds=m['wall_seconds'])),flush=True)
        except Exception:
            with LOCK:STATE['failures'].append(dict(job=str(dest),error=traceback.format_exc()));STATE['workers'][worker_id]=dict(status='failed',gpu=gpu,job=str(dest));update(status='failed');DONE.set()
            return
        finally:JOBS.task_done()

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpus',default='0,1,3,4,5,6,7');args=p.parse_args();gpus=[int(x)for x in args.gpus.split(',')];assert 2 not in gpus and len(gpus)==len(set(gpus))
    global TARGETS
    TARGETS=sorted((R/'data/targets/EVAL50').glob('chunk_*.pkl'));assert len(TARGETS)==35
    OUT.mkdir(parents=True,exist_ok=True)
    sources={p.name:sha(p)for p in [B/'evaluate.py',B/'queue_evaluation.py',R/'planning/stagnation.py']}
    configs={c:json.loads(json.dumps(__import__('omegaconf').OmegaConf.to_container(configuration(c,TARGETS[0],OUT/c/TARGETS[0].stem),resolve=True)))for c in CONDITIONS}
    receipt=dict(authorized_on='2026-11-13',scope='AdaJEPA threshold 0.60 and in-house augmented frozen threshold 0.45',started_on='2026-11-13',condition_order=list(CONDITIONS),gpus=gpus,workers_per_gpu=2,excluded_gpus=[2],episodes_per_condition=350,total_task_runs=700,targets=[dict(path=str(t),sha256=sha(t))for t in TARGETS],checkpoints={c:sha(Path(configs[c]['ckpt_base_path'])/'checkpoints/model_3.pth')for c in CONDITIONS},resolved_first_chunk_configs=configs,source_sha256=sources,protocol=str(B/'PROTOCOL.md'))
    path=OUT/'evaluation_manifest.json'
    if path.exists():assert json.loads(path.read_text())==receipt,'Existing run has different configuration'
    else:write(path,receipt)
    write(B/'evaluation_manifest.json',receipt)
    STATE.update(status='preflight',started_utc=utc(),completed_chunks=0,total_chunks=70,episodes_per_condition=350,total_task_runs=700,by_condition={c:dict(completed_chunks=0,total_chunks=35)for c in CONDITIONS},preflight={c:'pending'for c in CONDITIONS},workers={},failures=[],output=str(OUT))
    update();threading.Thread(target=heartbeat,daemon=True).start()
    with ThreadPoolExecutor(max_workers=2*len(gpus)) as pool:
        futures=[pool.submit(worker,gpu,slot) for gpu in gpus for slot in range(2)]
        for c in CONDITIONS:
            src=Path('/tmp/ecp_threshold_v3/preflight')/c/TARGETS[0].stem;dest=OUT/c/TARGETS[0].stem
            deadline=time.monotonic()+1200
            while not (src/'complete.json').exists():
                if time.monotonic()>deadline:raise TimeoutError(f'Preflight did not complete: {src}')
                time.sleep(10)
            m=json.loads((src/'complete.json').read_text())
            assert m['status']=='complete' and m['condition']==c and m['n']==10 and m['targets_sha256']==sha(TARGETS[0])
            assert m['source_sha256']['evaluate.py']==sha(B/'evaluate.py')
            if c.startswith('frozen'):assert m['frozen_weights_unchanged'] and not m['online_adaptation']
        for c in CONDITIONS:
            src=Path('/tmp/ecp_threshold_v3/preflight')/c/TARGETS[0].stem;dest=OUT/c/TARGETS[0].stem
            assert not dest.exists(),f'Preflight destination already exists: {dest}'
            shutil.copytree(src,dest)
            compare(c,TARGETS[0].stem)
            with LOCK:
                STATE['completed_chunks']+=1;STATE['by_condition'][c]['completed_chunks']+=1;STATE['preflight'][c]='passed'
                update()
        update(status='evaluating')
        for c in CONDITIONS:
            for t in TARGETS[1:]:JOBS.put((c,t))
        for future in futures:future.result()
    if STATE['failures']:raise RuntimeError(json.dumps(STATE['failures']))
    assert STATE['completed_chunks']==70
    summarize();update(status='complete',summary=str(OUT/'summary.json'));DONE.set()

if __name__=='__main__':
    try:main()
    except BaseException:update(status='failed',error=traceback.format_exc());DONE.set();raise
