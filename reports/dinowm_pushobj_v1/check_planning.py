from pathlib import Path
import json,sys,hashlib
import numpy as np
import torch
from omegaconf import OmegaConf
r=Path('/home/appuser/workspace/project/multimodal-test/JEPA-Stall');s=Path(__file__).parent
sys.path.insert(0,str(r));sys.path.insert(0,str(s))
torch.set_num_threads(2);torch.manual_seed(123)
torch.cuda.set_per_process_memory_fraction(4*2**30/torch.cuda.get_device_properties(0).total_memory,0)
from plan import load_model
from cem_memory import rollout_cached,ChunkedCEMPlanner
from planning.cem import CEMPlanner
from planning.stagnation import StagnationMPCPlanner
from planning.objectives import create_objective_fn
c=OmegaConf.load(s/'smoke_train/hydra.yaml');wm=load_model(s/'smoke_train/checkpoints/model_latest.pth',c,1,'cuda');wm.eval()
errors={}
with torch.no_grad():
 for history in [1,3]:
  obs={'visual':torch.rand(4,history,3,224,224,device='cuda'),'proprio':torch.rand(4,history,4,device='cuda')};act=torch.randn(4,5,10,device='cuda')
  ref,_=wm.rollout(obs,act);out,_=rollout_cached(wm,wm.encode_obs(obs),act)
  for key in ref:
   torch.testing.assert_close(ref[key],out[key],rtol=1e-5,atol=2e-6);errors[f'history{history}/{key}']=float((ref[key]-out[key]).abs().max())
class Identity:
 def transform_obs(self,obs):return obs
class Log:
 def log(self,*a,**k):pass
obs={'visual':torch.rand(1,1,3,224,224,device='cuda'),'proprio':torch.rand(1,1,4,device='cuda')};goal={k:torch.rand_like(v) for k,v in obs.items()}
objective=create_objective_fn(alpha=1,base=2,mode='last')
kw=dict(horizon=5,topk=3,num_samples=8,var_scale=1,opt_steps=2,eval_every=-1,wm=wm,action_dim=10,objective_fn=objective,preprocessor=Identity(),evaluator=None,wandb_run=Log())
torch.manual_seed(222);ref,_=CEMPlanner(**kw).plan(obs,goal)
torch.manual_seed(222);out,_=ChunkedCEMPlanner(candidate_batch=4,**kw).plan(obs,goal)
torch.testing.assert_close(ref,out,rtol=2e-5,atol=2e-6);errors['cem_actions']=float((ref-out).abs().max())
# Exercise the actual regularizer at a stalled state and at the goal.
with torch.no_grad():
 z=wm.encode_obs(obs);zg=wm.encode_obs(goal)
 m_v=z['visual'][:,0].flatten(1);m_p=z['proprio'][:,0].flatten(1)
 d=((m_v-zg['visual'][:,0].flatten(1))**2).mean(1)+((m_p-zg['proprio'][:,0].flatten(1))**2).mean(1)
 p=object.__new__(StagnationMPCPlanner);p.stag_space='latent';p.stag_alpha=1.;p.stag_base=2.;p._mem={'v':m_v,'p':m_p,'stay':d,'rho':.25*d,'w':torch.ones_like(d)}
 stay={k:v.expand(1,6,*v.shape[2:]) for k,v in z.items()};target={k:v.expand(1,6,*v.shape[2:]) for k,v in zg.items()}
 stay_pen=float(p._penalty(stay)[0]);goal_pen=float(p._penalty(target)[0]);assert stay_pen>0 and goal_pen==0
root=s/'smoke_eval';aa=np.load(root/'native_v4/final_eval_results.npz');bb=np.load(root/'ecp_v4/final_eval_results.npz')
for key in ['state_0','state_g']:np.testing.assert_array_equal(aa[key],bb[key])
np.testing.assert_array_equal(aa['actions'][:,:2],bb['actions'][:,:2])
lines=lambda path:[json.loads(x) for x in path.read_text().splitlines()]
episodes=lines(root/'ecp_v4/stag_episodes.jsonl');assert all(x['first_trigger']==2 for x in episodes)
records=lines(root/'ecp_v4/per_iter_logs.json');assert any(x.get('stag_boost') for x in records)
assert all(x['disp_pred']==0 and x['disp_enc']==0 and x['func_change']==0 for x in records)
report={'passed':True,'rollout_and_cem_max_abs_errors':errors,'positive_stay_penalty':stay_pen,'goal_penalty':goal_pen,'paired_targets_equal':True,'pre_trigger_actions_bitwise_equal':True,'forced_smoke_first_trigger':2,'proprio_boost_exercised':True,'no_online_weight_updates_in_smoke':True,'smoke_scores_are_not_experiment_results':True}
(s/'planning_validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
