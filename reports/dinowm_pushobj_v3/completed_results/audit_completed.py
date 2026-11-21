#!/usr/bin/env python3
"""Recompute completed DINO-WM PushObj outcomes and trigger diagnostics from raw logs."""
import argparse,csv,hashlib,json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):return [json.loads(l)for l in p.read_text().splitlines() if l.strip()]
def main():
 p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 root=a.repo/'eval_outputs/dinowm_pushobj_3epoch_v3_cem200x10_mpc6x25_bf16';summary=json.loads((root/'summary.json').read_text());assert summary['status']=='complete'
 arms=['ordinary_native','ordinary_ecp','augmented_native','augmented_ecp'];results={};pairing={};sources=[];csvrows=[]
 for arm in arms:
  paths=sorted((root/arm).glob('chunk_*'));assert len(paths)==35
  counts={};n=0;successes=0;triggered=0;triggers=[];timings=[];failed_min=[];successful_min=[];nonprogress_fail=0
  for d in paths:
   marker=json.loads((d/'complete.json').read_text());meta=json.loads((d/'stag_meta.json').read_text());episodes=rows(d/'stag_episodes.jsonl');logs=rows(d/'per_iter_logs.json');tt=rows(d/'cem_timings.jsonl');assert len(episodes)==10
   assert marker['status']=='complete' and marker['epoch']==3 and marker['frozen_weights_unchanged'];assert marker['max_mpc_replans']==6 and marker['max_environment_controls']==150 and marker['executed_physical_steps_per_replan']==25
   assert marker['precision']=='bf16' and marker['cem_candidates']==200 and marker['cem_iterations']==10
   assert (meta['gamma_lo'],meta['gamma_hi'],meta['start'],meta['window'])==(.05,.15,2,3)
   assert meta['kappa']==(2 if arm.endswith('_ecp')else 0) and meta['prop_boost']==(10 if arm.endswith('_ecp')else 1)
   assert all(t['precision']=='bf16' and t['candidates']==200 and t['iterations']==10 for t in tt)
   with np.load(d/'final_eval_results.npz')as raw:
    assert len(raw['successes'])==10 and int(raw['successes'].sum())==marker['successes'];assert raw['actions'].shape[1]*5<=150
    for e in episodes:assert bool(raw['successes'][e['sample_idx']])==e['success']
   n+=10;successes+=marker['successes'];shape=d.name.split('_',2)[2];v=counts.setdefault(shape,dict(n=0,successes=0));v['n']+=10;v['successes']+=marker['successes'];timings.extend(t['seconds']for t in tt)
   for e in episodes:
    windows=[x for x in logs if x['sample_idx']==e['sample_idx'] and 'stag_r'in x];assert e['replans']<=6
    if e['first_trigger']is not None:triggered+=1;triggers.append(dict(chunk=d.name,**e))
    if windows:
     v=min(x['stag_r']for x in windows);(successful_min if e['success']else failed_min).append(v)
     if not e['success']:nonprogress_fail+=any(x['stag_vprog']<=0 for x in windows)
   for fn in ['complete.json','final_eval_results.npz','stag_episodes.jsonl','per_iter_logs.json','stag_meta.json','cem_timings.jsonl']:
    f=d/fn;sources.append(dict(path=str(f.relative_to(a.repo)),sha256=digest(f)))
  assert n==350 and successes==summary['results'][arm]['all']['successes'] and all(v['n']==50 for v in counts.values())
  for shape,v in counts.items():
   assert v['successes']==summary['results'][arm]['by_shape'][shape]['successes'];v['success_rate']=v['successes']/v['n'];csvrows.append(dict(condition=arm,shape=shape,**v))
  results[arm]=dict(n=n,successes=successes,success_rate=successes/n,triggered=triggered,trigger_rate=triggered/n,triggers=triggers,by_shape=counts,mean_planning_seconds=float(np.mean(timings)),median_failed_episode_minimum_movement_ratio=float(np.median(failed_min)),failed_tasks_with_at_least_one_nonprogress_window=nonprogress_fail,offline_threshold_counts=[dict(gamma_hi=t,failed_tasks_firing=int(np.sum(np.array(failed_min)<t)),successful_tasks_firing=int(np.sum(np.array(successful_min)<t)))for t in [.15,.2,.3,.4,.5]],offline_analysis_note='Counts describe saved native trajectories, not success after changing the controller; not an independent calibration set.')
 for model in ['ordinary','augmented']:
  same=0;rescues=[];regressions=[];pre=[]
  for d in sorted((root/(model+'_native')).glob('chunk_*')):
   alt=root/(model+'_ecp')/d.name;eps={e['sample_idx']:e for e in rows(alt/'stag_episodes.jsonl')}
   with np.load(d/'final_eval_results.npz')as x,np.load(alt/'final_eval_results.npz')as y:
    for k in ['state_0','state_g']:np.testing.assert_array_equal(x[k],y[k])
    for i,(sa,sb)in enumerate(zip(x['successes'],y['successes'])):
     key=f'{d.name}:{i}';t=eps[i]['first_trigger'];prefix=min(x['actions'].shape[1],y['actions'].shape[1],t*5 if t is not None else 30)
     if not np.array_equal(x['actions'][i,:prefix],y['actions'][i,:prefix]):pre.append(key)
     if np.array_equal(x['actions'][i],y['actions'][i]):same+=1
     if sb and not sa:rescues.append(key)
     if sa and not sb:regressions.append(key)
  assert not pre
  pairing[model]=dict(n=350,targets_identical=True,pretrigger_mismatches=pre,identical_full_action_sequences=same,rescues=rescues,regressions=regressions)
 output=dict(status='passed',audited_utc=datetime.now(timezone.utc).isoformat(),chunks=140,task_runs=1400,results=results,pairing=pairing,source_hashes=sources)
 (a.output/'audit.json').write_text(json.dumps(output,indent=2)+'\n');(a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 with(a.output/'by_shape.csv').open('w',newline='')as f:
  w=csv.DictWriter(f,fieldnames=['condition','shape','n','successes','success_rate'],lineterminator='\n');w.writeheader();w.writerows(csvrows)
 print(json.dumps({'status':'passed','chunks':140,'task_runs':1400,'results':{k:{f:v[f]for f in ['n','successes','triggered']}for k,v in results.items()},'pairing':pairing}))
if __name__=='__main__':main()
