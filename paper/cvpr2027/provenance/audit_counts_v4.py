from pathlib import Path
from collections import defaultdict
import json,hashlib,math
R=Path('/home/appuser/workspace/project/multimodal-test/JEPA-Stall')
O=Path('/tmp/jepa_revision_v4/evidence')
def load(path):
 d=defaultdict(dict)
 for f in (R/path).glob('*/per_iter_logs.json'):
  for line in f.read_text().splitlines():
   if line.strip():
    x=json.loads(line);k=(f.parent.name,x['sample_idx']);assert x['step'] not in d[k];d[k][x['step']]=x
 return d
def ok(d):return {k:any(v['success'] for s,v in rows.items() if s<=30) for k,rows in d.items()}
def paired(ref,arm):
 assert set(arm)<=set(ref)
 a=ok(ref);b=ok({**ref,**arm});w=sum(b[k] and not a[k] for k in a);l=sum(a[k] and not b[k] for k in a)
 return dict(n=len(a),base=sum(a.values()),success=sum(b.values()),gain_pp=100*(sum(b.values())-sum(a.values()))/len(a),rescued=w,regressed=l,rerun=len(arm))
d={'main':[],'ablation':{},'grid':{},'maze':{},'stage_comparisons':{}}
for m in ['official','bias16k']:
 for a in ['frozen','adajepa']:
  p=f'eval_outputs/stag_{a}_{m}_it40';ref=load(p+'/ref');assert len(ref)==350
  row={'checkpoint':m,'adapter':a,'ref_success':sum(ok(ref).values()),'S':paired(ref,load(p+'/SZ_mid')),'P':paired(ref,load(p+'/T0P')),'SP':paired(ref,load(p+'/T0SP'))}
  d['main'].append(row)
p='eval_outputs/stag_frozen_bias16k_it40';ref=load(p+'/ref')
for name in ['SZ_early','SZ_mid','T0P','T0SP','D2S','D2SP','D2SPL','T0SPL','ALPHA10','LA_start']:
 d['ablation'][name]=paired(ref,load(p+'/'+name))
ref=load('eval_outputs/escape_EVAL50/none');assert len(ref)==350
for name in ['SP_lo','SP_mid','SP_hi','SZ_lo','SZ_mid','SZ_hi']:
 d['grid'][name]=paired(ref,load('eval_outputs/stag_EVAL50/'+name))
# Check the historical and stage-table AdaJEPA baselines coincide through30.
ref2=load('eval_outputs/stag_adajepa_bias16k_it40/ref')
d['grid_reference_identical_outcomes']=ok(ref)==ok(ref2)
bad=[]
for k,rs in ref.items():
 for step,r in rs.items():
  if step<=30 and (step not in ref2[k] or r['act']!=ref2[k][step]['act']):bad.append((k,step))
d['grid_reference_action_mismatches_through30']=len(bad)
for a in ['frozen','adajepa']:
 official=load(f'eval_outputs/stag_{a}_official_it40/ref');own=load(f'eval_outputs/stag_{a}_bias16k_it40/ref')
 sp=load(f'eval_outputs/stag_{a}_bias16k_it40/T0SP')
 d['stage_comparisons'][a]={'data_vs_official':paired(official,own),'all_vs_official':paired(official,{**own,**sp}),'planner_vs_data':paired(own,sp)}
(O/'counts.json').write_text(json.dumps(d,indent=2))
print(json.dumps(d,indent=2))
