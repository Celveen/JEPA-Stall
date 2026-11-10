#!/usr/bin/env python3
"""Wait for both 3-epoch jobs, then evaluate four paired frozen-model conditions."""
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib,json,os,subprocess,sys,threading,time,traceback
import numpy as np
R=Path('/home/appuser/workspace/project/multimodal-test/JEPA-Stall');B=R/'reports/dinowm_pushobj_v1'
OUT=R/'eval_outputs/dinowm_pushobj_3epoch_v1';PY='/home/appuser/anaconda3/envs/ts/bin/python'
ARMS=[('ordinary',False,0),('ordinary',True,1),('augmented',False,3),('augmented',True,5)]
LOCK=threading.Lock();STATE={'status':'waiting_for_training','target_epochs':3,'updated_utc':None,'completed_chunks':0,'total_chunks':140,'episodes_per_condition':350}

def write(path,value):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)

def update(**kw):
 with LOCK:
  STATE.update(kw,updated_utc=datetime.now(timezone.utc).isoformat());write(B/'pipeline_status.json',STATE)

def alive(pid):
 try:return Path(f'/proc/{pid}/stat').read_text().split()[2]!='Z'
 except FileNotFoundError:return False

def wait_training():
 while True:
  complete=[];progress={}
  for arm in ['ordinary','augmented']:
   folder=R/'checkpoints/dinowm_pushobj_v1'/arm
   if (folder/'complete.json').exists():
    meta=json.loads((folder/'complete.json').read_text());assert meta['epoch']==3 and Path(meta['checkpoint']).is_file();complete.append(arm)
   else:
    launcher=json.loads((B/f'{arm}_train_launcher.json').read_text())
    assert launcher['epochs']==3 and alive(launcher['pid']),f'{arm} training exited before completion; inspect training log'
   status=folder/'status.json';progress[arm]=json.loads(status.read_text()) if status.exists() else {'status':'starting'}
  update(training=progress)
  if len(complete)==2:return
  time.sleep(30)

def run_arm(arm,ecp,gpu,targets):
 label=arm+('_ecp' if ecp else '_native');checkpoint=R/'checkpoints/dinowm_pushobj_v1'/arm
 for target in targets:
  destination=OUT/label/target.stem;marker=destination/'complete.json'
  if marker.exists():
   completed=json.loads(marker.read_text());assert completed['epoch']==3 and completed['n']==10 and completed['ecp']==ecp and completed['targets_sha256']==hashlib.sha256(target.read_bytes()).hexdigest()
  else:
   destination.mkdir(parents=True,exist_ok=True)
   assert not (destination/'per_iter_logs.json').exists(),f'Partial run requires inspection: {destination}'
   while True:
    free={int(x.split(',')[0]):int(x.split(',')[1]) for x in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()}
    if free[gpu]>=16000:break
    time.sleep(30)
   cmd=[PY,str(B/'evaluate.py'),'--repo',str(R),'--checkpoint',str(checkpoint),'--epoch','3','--n-evals','10','--targets',str(target),'--output',str(destination)]
   if ecp:cmd.append('--ecp')
   env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'2','SDL_VIDEODRIVER':'dummy','SDL_AUDIODRIVER':'dummy','WANDB_MODE':'disabled'}
   with (destination/'run.log').open('w') as log:
    proc=subprocess.Popen(cmd,cwd=R,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(destination/'launcher.json',{'pid':proc.pid,'gpu':gpu,'command':cmd,'started_utc':datetime.now(timezone.utc).isoformat()})
    code=proc.wait()
   assert code==0 and marker.exists(),f'Evaluation failed (exit {code}): {destination}/run.log'
  with LOCK:
   STATE['completed_chunks']+=1;STATE['updated_utc']=datetime.now(timezone.utc).isoformat();write(B/'pipeline_status.json',STATE)
  print(json.dumps({'completed':str(destination),'gpu':gpu}),flush=True)
 return label

def summarize(targets):
 results={};paired={}
 for arm,ecp,gpu in ARMS:
  label=arm+('_ecp' if ecp else '_native');shapes={}
  for target in targets:
   shape=target.stem.split('_',2)[2];data=np.load(OUT/label/target.stem/'final_eval_results.npz')
   shapes.setdefault(shape,[]).extend(data['successes'].astype(bool).tolist())
  assert len(shapes)==7 and all(len(v)==50 for v in shapes.values())
  row={'by_shape':{k:{'n':len(v),'successes':sum(v),'success_rate':float(np.mean(v))} for k,v in shapes.items()}}
  for group,names in [('seen',['T','L','Z','plus']),('unseen',['I','small_tee','square']),('all',list(shapes))]:
   values=[v for name in names for v in shapes[name]];row[group]={'n':len(values),'successes':sum(values),'success_rate':float(np.mean(values))}
  results[label]=row
 for arm in ['ordinary','augmented']:
  pair={'rescues':[],'regressions':[],'pre_trigger_mismatches':[],'targets_equal':True}
  for target in targets:
   base=OUT/f'{arm}_native'/target.stem;ecpdir=OUT/f'{arm}_ecp'/target.stem
   a=np.load(base/'final_eval_results.npz');b=np.load(ecpdir/'final_eval_results.npz')
   for key in ['state_0','state_g']:np.testing.assert_array_equal(a[key],b[key])
   episodes=[json.loads(line) for line in (ecpdir/'stag_episodes.jsonl').read_text().splitlines()]
   for i,(sa,sb) in enumerate(zip(a['successes'],b['successes'])):
    key=f'{target.stem}:{i}'
    if sb and not sa:pair['rescues'].append(key)
    if sa and not sb:pair['regressions'].append(key)
    episode=next(x for x in episodes if x['sample_idx']==i);t=episode['first_trigger']
    n=min(a['actions'].shape[1],b['actions'].shape[1]) if t is None else t
    if not np.array_equal(a['actions'][i,:n],b['actions'][i,:n]):pair['pre_trigger_mismatches'].append(key)
  paired[arm]=pair
 write(OUT/'summary.json',{'status':'complete','training_epochs':3,'results':results,'paired':paired,'all_pre_trigger_pairs_identical':all(not x['pre_trigger_mismatches'] for x in paired.values()),'protocol':str(B/'PROTOCOL.md')})

def main():
 assert json.loads((B/'planning_validation.json').read_text())['passed']
 assert json.loads((B/'evaluation_wrapper_validation.json').read_text())['passed']
 assert json.loads((R/'checkpoints/dinowm_pushobj_v1/ordinary/initialization.json').read_text())==json.loads((R/'checkpoints/dinowm_pushobj_v1/augmented/initialization.json').read_text())
 targets=sorted((R/'data/targets/EVAL50').glob('chunk_*.pkl'));assert len(targets)==35
 OUT.mkdir(parents=True,exist_ok=True)
 write(B/'evaluation_queue.json',{'epochs':3,'conditions':[{'data':arm,'ecp':ecp,'gpu':gpu} for arm,ecp,gpu in ARMS],'targets':[{'path':str(t),'sha256':hashlib.sha256(t.read_bytes()).hexdigest()} for t in targets],'n_evals_per_chunk':10,'max_mpc_replans':30,'executed_physical_steps_per_replan':5,'cem_candidates':300,'cem_iterations':30,'pytorch_memory_cap_gib':4})
 update();wait_training();update(status='evaluating')
 failures=[]
 with ThreadPoolExecutor(max_workers=4) as pool:
  futures=[pool.submit(run_arm,*arm,targets) for arm in ARMS]
  for future in as_completed(futures):
   try:future.result()
   except Exception:failures.append(traceback.format_exc())
 if failures:raise RuntimeError('\n'.join(failures))
 summarize(targets);update(status='complete',summary=str(OUT/'summary.json'))
if __name__=='__main__':
 try:main()
 except BaseException:
  update(status='failed',error=traceback.format_exc());raise
