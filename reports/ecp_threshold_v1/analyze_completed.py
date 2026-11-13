#!/usr/bin/env python3
"""Audit the completed threshold/frequency follow-up without running the simulator."""
import argparse,csv,hashlib,json
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import numpy as np

def read_rows(p):return [json.loads(l)for l in p.read_text().splitlines()if l.strip()]if p.exists()else[]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 root=a.repo/'eval_outputs/ecp_threshold_v1';summary=json.loads((root/'summary.json').read_text());assert summary['status']=='complete';results={};evidence=[];task_rows=[];shapes=[]
 for condition,aggregate in summary['results'].items():
  paths=sorted((root/condition).glob('chunk_*'));assert len(paths)==35
  n=successes=triggered=action_changed=0;rescues=[];regressions=[];first=[];counts={};pen=[];boosts=0
  native=condition=='dinowm_aug_dense_native';per_plan=5 if condition=='dinowm_aug_g030'else 1
  for d in paths:
   meta=json.loads((d/'complete.json').read_text());assert meta['status']=='complete'and meta['n']==10 and meta['thresholds']==[.1,.3]and meta['max_environment_controls']==150 and meta['checkpoint_file_unchanged']
   if condition.startswith('dinowm'):assert meta['frozen_weights_unchanged']
   eps={e['sample_idx']:e for e in read_rows(d/'stag_episodes.jsonl')};assert set(eps)==set(range(10));logs=read_rows(d/'per_iter_logs.json')
   with np.load(d/'final_eval_results.npz')as raw:
    assert len(raw['successes'])==10 and int(raw['successes'].sum())==meta['successes'];assert raw['actions'].shape[1]*5<=150
    n+=10;successes+=int(raw['successes'].sum());shape=d.name.split('_',2)[2];v=counts.setdefault(shape,dict(n=0,successes=0));v['n']+=10;v['successes']+=int(raw['successes'].sum())
    for i,e in eps.items():
     assert bool(raw['successes'][i])==e['success'];assert e['replans']<=meta['max_mpc_replans']
     if e['first_trigger']is not None:triggered+=1;first.append(e['first_trigger'])
    if not native:
     if condition.startswith('adajepa'):ref=a.repo/'eval_outputs/stag_adajepa_bias16k_it40/ref'/d.name
     elif condition=='dinowm_aug_g030':ref=a.repo/'eval_outputs/dinowm_pushobj_3epoch_v3_cem200x10_mpc6x25_bf16/augmented_native'/d.name
     else:ref=root/'dinowm_aug_dense_native'/d.name
     rl=read_rows(ref/'per_iter_logs.json')
     with np.load(ref/'final_eval_results.npz')as old:
      for k in ['state_0','state_g']:np.testing.assert_array_equal(raw[k],old[k])
      for i,e in eps.items():
       before=any(x['sample_idx']==i and x['step']<=30 and x['success']for x in rl)if condition.startswith('adajepa')else bool(old['successes'][i]);after=bool(raw['successes'][i]);t=e['first_trigger'];prefix=min(raw['actions'].shape[1],old['actions'].shape[1],30,t*per_plan if t is not None else 30)
       np.testing.assert_array_equal(raw['actions'][i,:prefix],old['actions'][i,:prefix]);key=f'{d.name}:{i}'
       length=min(raw['actions'].shape[1],old['actions'].shape[1],30);different=not np.array_equal(raw['actions'][i,:length],old['actions'][i,:length]);action_changed+=different
       if after and not before:rescues.append(key)
       if before and not after:regressions.append(key)
       task_rows.append(dict(condition=condition,chunk=d.name,sample_idx=i,native_success=int(before),ecp_success=int(after),first_trigger=t,actions_changed=int(different)))
   for x in logs:
    if 'stag_pen'in x and x.get('stag_task',0)>0:pen.append(x['stag_pen']/x['stag_task'])
    boosts+=bool(x.get('stag_boost',False))
   for fn in ['complete.json','final_eval_results.npz','stag_episodes.jsonl','per_iter_logs.json']:
    f=d/fn;evidence.append(dict(path=str(f.relative_to(a.repo)),sha256=sha(f)))
  assert n==350 and successes==aggregate['successes']and triggered==aggregate['triggered']
  if not native:assert rescues==aggregate['paired']['rescues']and regressions==aggregate['paired']['regressions']
  for shape,v in counts.items():assert v['n']==50;shapes.append(dict(condition=condition,shape=shape,**v,success_rate=v['successes']/50))
  results[condition]=dict(n=n,successes=successes,success_rate=successes/n,triggered=triggered,trigger_rate=triggered/n,first_trigger_median=float(np.median(first)),first_trigger_control_median=float(np.median(first))*per_plan*5,by_shape=counts,actions_changed_vs_native=action_changed if not native else None,paired_rescues=rescues if not native else None,paired_regressions=regressions if not native else None,boosted_plans=boosts,median_selected_plan_penalty_to_task_cost=float(np.median(pen))if pen else None)
 base=a.repo/'eval_outputs/stag_adajepa_bias16k_it40';old_summary=Counter();old_first=[];new_first=[];early=late=same=0;old_to_new_rescues=[];old_to_new_regressions=[]
 for d in sorted((base/'ref').glob('chunk_*')):
  ref=read_rows(d/'per_iter_logs.json');old=read_rows(base/'T0SP'/d.name/'per_iter_logs.json');new=read_rows(root/'adajepa_aug_g030'/d.name/'per_iter_logs.json')
  for i in range(10):
   rr=[x for x in ref if x['sample_idx']==i and x['step']<=30];oo=[x for x in old if x['sample_idx']==i and x['step']<=30];nn=[x for x in new if x['sample_idx']==i and x['step']<=30];assert rr and nn
   sr=any(x['success']for x in rr);so=any(x['success']for x in(oo or rr));sn=any(x['success']for x in nn)
   old_summary['native_successes']+=sr;old_summary['gamma_015_successes']+=so;old_summary['gamma_030_successes']+=sn
   key=f'{d.name}:{i}'
   if sn and not so:old_to_new_rescues.append(key)
   if so and not sn:old_to_new_regressions.append(key)
   ot=next((x['step']for x in rr if 10<=x['step']<30 and x.get('stag_r',999)<.15),None);nt=next((x['step']for x in nn if 10<=x['step']<30 and x.get('stag_w',0)>0),None)
   if ot is not None:old_first.append(ot)
   if nt is not None:new_first.append(nt)
   if ot is not None and nt is not None:early+=nt<ot;late+=nt>ot;same+=nt==ot
 comparison=dict(old_summary,gamma_015_triggered=len(old_first),gamma_030_triggered=len(new_first),gamma_015_first_trigger_median=float(np.median(old_first)),gamma_030_first_trigger_median=float(np.median(new_first)),earlier=early,later=late,same=same,rescues_vs_old_ecp=old_to_new_rescues,regressions_vs_old_ecp=old_to_new_regressions)
 assert comparison['native_successes']==262 and comparison['gamma_015_successes']==269 and comparison['gamma_030_successes']==271
 output=dict(status='passed',audited_utc=datetime.now(timezone.utc).isoformat(),chunks=140,task_runs=1400,raw_target_and_pretrigger_pairs_verified=1050,results=results,adajepa_threshold_comparison=comparison,source_hashes=evidence,scope='Exploratory follow-up on the existing fixed task set. No independent calibration split or repeated training seeds. Dense cadence changes total search computation as well as detection opportunities; its native control is evaluated separately.')
 (a.output/'completed_audit.json').write_text(json.dumps(output,indent=2)+'\n')
 for filename,data in [('paired_tasks.csv',task_rows),('by_shape.csv',shapes)]:
  with(a.output/filename).open('w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(data[0]),lineterminator='\n');w.writeheader();w.writerows(data)
 print(json.dumps(dict(status='passed',chunks=140,task_runs=1400,paired_checks=1050,results={k:{f:v[f]for f in ['successes','triggered','actions_changed_vs_native']}for k,v in results.items()},adajepa_threshold_comparison=comparison)))
if __name__=='__main__':main()
