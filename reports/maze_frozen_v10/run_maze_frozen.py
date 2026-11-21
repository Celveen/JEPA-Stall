#!/usr/bin/env python3
"""Matched Frozen Maze: fixed S primary and S+P component control, 30 replans."""
import argparse,hashlib,json,os,subprocess,sys,concurrent.futures
from pathlib import Path
import numpy as np
from omegaconf import OmegaConf
R=Path('/home/appuser/workspace/project/multimodal-test/JEPA-Stall');BASE=R/'eval_outputs/stag_maze100_ar4/medium/frozen/ref'
O=R/'eval_outputs/maze_frozen_v10'

def rows(folder):
 d={}
 for p in sorted(folder.rglob('per_iter_logs.json')):
  for line in p.read_text().splitlines():
   x=json.loads(line)
   if x['step']<=30:
    assert x['step'] not in d.setdefault(x['sample_idx'],{})
    d[x['sample_idx']][x['step']]=x
 return d

def cmd(arm,ids,out):
 old=list(OmegaConf.load(BASE/'shard00/.hydra/overrides.yaml'))
 drop=('ckpt_base_path=','eval_data_path=','planner.max_iter=','++planner.stag.kappa=','++planner.stag.start=','++planner.stag.only_samples=')
 opts=[v for v in old if not v.startswith(drop)]
 opts += [f'ckpt_base_path={R}/checkpoints/diversemaze_layout_shift',f'eval_data_path={R}/data/diverse_maze','planner.max_iter=30',f'++planner.stag.kappa={0 if arm=="smoke" else 2}',f'++planner.stag.start={4 if arm=="smoke" else 7}',f'++planner.stag.prop_boost={10 if arm=="SP" else 1}','++planner.stag.boost_replans=2','++planner.stag.only_samples=['+','.join(map(str,ids))+']',f'hydra.run.dir={out}']
 return [sys.executable,str(R/'plan.py'),'--config-name','stag_plan_gd_diversemaze',*opts]

def run(arm,ids,out,gpu):
 out.mkdir(parents=True,exist_ok=True)
 command=cmd(arm,ids,out)
 record={'arm':arm,'ids':ids,'gpu':gpu,'command':command,'status':'running'}
 (out/'run_metadata.json').write_text(json.dumps(record,indent=2))
 env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1','WANDB_MODE':'offline','SDL_VIDEODRIVER':'dummy','PYTHONUNBUFFERED':'1','LD_LIBRARY_PATH':os.environ.get('LD_LIBRARY_PATH','')+':/home/appuser/.mujoco/mujoco210/bin:/usr/lib/nvidia'}
 print('START',arm,ids,'GPU',gpu,flush=True)
 with (out/'console.log').open('w') as log:p=subprocess.run(command,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT)
 record['status']='complete' if p.returncode==0 else 'failed';record['exit_code']=p.returncode
 (out/'run_metadata.json').write_text(json.dumps(record,indent=2));assert p.returncode==0,str(out)
 assert (out/'final_eval_results.npz').exists()
 print('DONE',arm,ids,flush=True)

def verify(ref,new,folder):
 meta={}
 for p in folder.rglob('stag_episodes.jsonl'):
  for line in p.read_text().splitlines():
   x=json.loads(line);meta[x['sample_idx']]=x
 for i,rs in new.items():
  t=meta[i]['first_trigger']
  for s,x in rs.items():
   if t is None or s<=t:
    np.testing.assert_allclose(x['act'],ref[i][s]['act'],atol=1e-4,rtol=1e-5)
    np.testing.assert_allclose(x['state'],ref[i][s]['state'],atol=1e-4,rtol=1e-5)
 return len(new)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['smoke','arms'],required=True);a=ap.parse_args();O.mkdir(exist_ok=True)
 ref=rows(BASE);assert len(ref)==100
 if a.phase=='smoke':
  run('smoke',[0,1],O/'smoke',6);n=verify(ref,rows(O/'smoke'),O/'smoke')
  (O/'smoke_verified.json').write_text(json.dumps({'status':'passed','episodes':n}));return
 assert json.loads((O/'smoke_verified.json').read_text())['status']=='passed'
 ids=sorted(i for i,rs in ref.items() if any(7<=t<30 and x.get('stag_r') is not None and x['stag_r']<.15 for t,x in rs.items()))
 manifest={'n':100,'budget_replans':30,'action_repeat':4,'native_successes':sum(any(x['success'] for x in v.values()) for v in ref.values()),'primary':'S','component_control':'SP','triggered_rerun_indices':ids,'untriggered_reuse_indices':sorted(set(ref)-set(ids)),'baseline':str(BASE),'data':'same100 medium tasks; checkpoint released; frozen; thresholds and memory parameters unchanged from existing Maze settings','source_hashes':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [R/'planning/stagnation.py',R/'plan.py',R/'checkpoints/diversemaze_layout_shift/checkpoints/model_latest.pth']}}
 (O/'manifest.json').write_text(json.dumps(manifest,indent=2))
 jobs=[]
 for arm in ['S','SP']:
  for k in range(0,len(ids),5):jobs.append((arm,ids[k:k+5],O/arm/f'shard{k//5:02d}'))
 gpus=[0,1,3,5,6]
 def worker(index):
  for j in range(index,len(jobs),len(gpus)):
   arm,chunk,out=jobs[j]
   if (out/'run_metadata.json').exists() and json.loads((out/'run_metadata.json').read_text())['status']=='complete':continue
   run(arm,chunk,out,gpus[index])
 with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:list(ex.map(worker,range(5)))
 baseok={i:any(x['success'] for x in rs.values()) for i,rs in ref.items()}
 summary={'baseline_successes':sum(baseok.values()),'n':100,'arms':{}}
 for arm in ['S','SP']:
  new=rows(O/arm);assert sorted(new)==ids;verify(ref,new,O/arm)
  cur={i:any(x['success'] for x in new.get(i,ref[i]).values()) for i in ref}
  summary['arms'][arm]={'successes':sum(cur.values()),'rescues':[i for i in ref if cur[i] and not baseok[i]],'regressions':[i for i in ref if baseok[i] and not cur[i]],'pretrigger_pairs_verified':len(new),'untriggered_reused':100-len(new)}
 (O/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
