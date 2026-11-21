#!/usr/bin/env python3
"""Frozen DINO-WM deployment with the shared PushObj ECP wrapper and native CEM."""
import argparse, hashlib, json, os, sys, time
from pathlib import Path
from omegaconf import OmegaConf
import numpy as np
import torch

p=argparse.ArgumentParser()
for key in ['repo','checkpoint','targets','output']:p.add_argument('--'+key,type=Path,required=True)
p.add_argument('--epoch',type=int,default=3)
p.add_argument('--n-evals',type=int,default=10)
p.add_argument('--ecp',action='store_true')
p.add_argument('--smoke',action='store_true')
p.add_argument('--forced-trigger',action='store_true')
p.add_argument('--precision',choices=['fp32','bf16'],default='fp32')
p.add_argument('--candidate-batch',type=int,default=64)
p.add_argument('--validation-replans',type=int,default=0)
p.add_argument('--memory-limit-gib',type=float,default=8)
a=p.parse_args()
assert not a.forced_trigger or a.smoke or a.validation_replans,'Forced activation is only for integration testing'
a.output.mkdir(parents=True,exist_ok=True)
assert not (a.output/'per_iter_logs.json').exists(),'Use a fresh output directory to avoid mixing records'
sys.path.insert(0,str(a.repo));sys.path.insert(0,str(a.repo/'reports/dinowm_pushobj_v1'));sys.path.insert(0,str(Path(__file__).parent))
c=OmegaConf.load(a.repo/'conf/escape_plan_gd_pushobj.yaml')
c.pop('hydra',None);c.pop('defaults',None)
c.ckpt_base_path=str(a.checkpoint);c.n_evals=2 if a.smoke else a.n_evals
c.eval_data_path=str(a.targets);c.wandb_logging=False;c.saved_folder=str(a.output)
c.model_epoch='latest' if a.smoke else str(a.epoch)
c.n_plot_samples=min(10,c.n_evals)
c.objective.mode='last'
c.planner._target_='planning.stagnation.StagnationMPCPlanner'
c.planner.probe.mode='none';c.planner.adapt.finetune_encoder=False;c.planner.adapt.finetune_every=1
c.planner.sub_planner=OmegaConf.create({'target':'cem_memory.ChunkedCEMPlanner','horizon':25,'topk':3 if a.smoke else 30,'num_samples':8 if a.smoke else 200,'var_scale':1,'opt_steps':2 if a.smoke else 10,'eval_every':-1,'candidate_batch':4 if a.smoke else a.candidate_batch,'precision':a.precision,'timing_path':'cem_timings.jsonl'})
c.planner.max_iter=4 if a.smoke else (a.validation_replans or 30);c.planner.n_taken_actions=5
c.planner.name='dinowm_cem_ecp' if a.ecp else 'dinowm_cem'
c.planner.stag=OmegaConf.create({'space':'latent','kappa':2.0 if a.ecp else 0.,'start':2 if a.forced_trigger else 10,'window':1 if a.forced_trigger else 3,'gamma_lo':1000. if a.forced_trigger else .05,'gamma_hi':2000. if a.forced_trigger else .15,'beta':.25,'prop_boost':10. if a.ecp else 1.,'boost_replans':2,'alpha':1.,'base':2.})
torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
if torch.cuda.is_available():torch.cuda.set_per_process_memory_fraction(a.memory_limit_gib*2**30/torch.cuda.get_device_properties(0).total_memory,0)
os.chdir(a.output);OmegaConf.save(c,'config.yaml')
import plan

def model_hash(model):
 h=hashlib.sha256()
 for key,value in model.state_dict().items():
  h.update(key.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()

original_load=plan.load_model;loaded={}
def checked_load(*args,**kwargs):
 model=original_load(*args,**kwargs)
 loaded['model']=model;loaded['before']=model_hash(model)
 return model
plan.load_model=checked_load
start=time.monotonic()
plan.planning_main(OmegaConf.to_container(c,resolve=True))
after=model_hash(loaded['model']);assert after==loaded['before'],'Frozen evaluation changed model parameters'
results=np.load('final_eval_results.npz');assert len(results['successes'])==c.n_evals
rows=[json.loads(line) for line in Path('stag_episodes.jsonl').read_text().splitlines()]
assert len(rows)==c.n_evals and {x['sample_idx'] for x in rows}==set(range(c.n_evals))
completion={'status':'complete','epoch':'smoke' if a.smoke else a.epoch,'n':c.n_evals,'successes':int(results['successes'].sum()),'ecp':a.ecp,'wall_seconds':time.monotonic()-start,'frozen_weights_unchanged':True,'model_sha256':after,'checkpoint_sha256':hashlib.sha256((a.checkpoint/'checkpoints'/('model_latest.pth' if a.smoke else f'model_{a.epoch}.pth')).read_bytes()).hexdigest(),'targets_sha256':hashlib.sha256(a.targets.read_bytes()).hexdigest(),'is_smoke':a.smoke,'is_validation':bool(a.validation_replans or a.smoke),'precision':a.precision,'memory_limit_gib':a.memory_limit_gib,'candidate_batch':c.planner.sub_planner.candidate_batch,'cem_candidates':c.planner.sub_planner.num_samples,'cem_iterations':c.planner.sub_planner.opt_steps,'cem_elites':c.planner.sub_planner.topk,'max_mpc_replans':c.planner.max_iter,'executed_physical_steps_per_replan':5,'objective_mode':c.objective.mode,'source_sha256':{name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in ['evaluate.py','cem_memory.py']}}
Path('complete.tmp').write_text(json.dumps(completion,indent=2)+'\n');Path('complete.tmp').replace('complete.json')
print(json.dumps(completion),flush=True)
