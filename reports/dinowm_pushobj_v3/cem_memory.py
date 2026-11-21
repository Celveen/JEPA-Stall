"""Official CEM sampling/update; batch candidates and cache the unchanged observation encoding."""
import contextlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from planning.cem import CEMPlanner
from utils import move_to_device

def rollout_cached(wm,zobs,act):
 assert wm.concat_dim==1
 n=next(iter(zobs.values())).shape[1];a=wm.encode_act(act[:,:n]);v=zobs['visual'];p=zobs['proprio']
 z=torch.cat([v,p[:,:,None,:].repeat(1,1,v.shape[2],wm.num_proprio_repeat),a[:,:,None,:].repeat(1,1,v.shape[2],wm.num_action_repeat)],dim=-1)
 for t in range(n,act.shape[1]):
  zn=wm.predict(z[:,-wm.num_hist:])[:,-1:];zn=wm.replace_actions_from_z(zn,act[:,t:t+1]);z=torch.cat([z,zn],dim=1)
 z=torch.cat([z,wm.predict(z[:,-wm.num_hist:])[:,-1:]],dim=1)
 return wm.separate_emb(z)[0],z

class ChunkedCEMPlanner(CEMPlanner):
 def __init__(self,candidate_batch=64,precision='fp32',timing_path=None,**kw):
  super().__init__(**kw);self.candidate_batch=candidate_batch;self.precision=precision;self.timing_path=timing_path
  if precision not in ('fp32','bf16'):raise ValueError(precision)
  assert 1 < self.topk <= self.num_samples and candidate_batch > 0
 @torch.no_grad()
 def plan(self,obs_0,obs_g,actions=None,step=None):
  if self.device.type=='cuda':torch.cuda.synchronize(self.device)
  begin=time.perf_counter()
  trans=move_to_device(self.preprocessor.transform_obs(obs_0),self.device);goal=move_to_device(self.preprocessor.transform_obs(obs_g),self.device)
  z0=self.wm.encode_obs(trans);zg=self.wm.encode_obs(goal);mu,sigma=self.init_mu_sigma(obs_0,actions);mu=mu.to(self.device);sigma=sigma.to(self.device)
  for i in range(self.opt_steps):
   best=[]
   for traj in range(len(mu)):
    # Draw every candidate together on CPU, exactly as the official implementation.
    a=torch.randn(self.num_samples,self.horizon,self.action_dim).to(self.device)*sigma[traj]+mu[traj];a[0]=mu[traj]
    costs=[]
    for j in range(0,self.num_samples,self.candidate_batch):
     aa=a[j:j+self.candidate_batch];n=len(aa)
     cur={k:v[traj:traj+1].expand(n,*v.shape[1:]) for k,v in z0.items()};g={k:v[traj:traj+1].expand(n,*v.shape[1:]) for k,v in zg.items()}
     context=torch.autocast('cuda',dtype=torch.bfloat16) if self.precision=='bf16' else contextlib.nullcontext()
     with context:pred,_=rollout_cached(self.wm,cur,aa)
     # Keep objective reductions, memory penalties, elite selection and CEM updates in FP32.
     costs.append(self.objective_fn({k:v.float() for k,v in pred.items()},g,step=step))
    loss=torch.cat(costs);idx=torch.argsort(loss)[:self.topk];elite=a[idx];mu[traj]=elite.mean(0);sigma[traj]=elite.std(0);best.append(float(loss[idx[0]]))
   self.wandb_run.log({f'{self.logging_prefix}/loss':np.mean(best),'step':i+1})
   if self.evaluator is not None and self.eval_every!=-1 and i%self.eval_every==0:
    logs,successes,_,_=self.evaluator.eval_actions(mu,filename=f'{self.logging_prefix}_output_{i+1}')
    if np.all(successes):break
  if self.device.type=='cuda':torch.cuda.synchronize(self.device)
  if self.timing_path:
   with Path(self.timing_path).open('a') as f:f.write(json.dumps({'prefix':self.logging_prefix,'step':step,'precision':self.precision,'candidates':self.num_samples,'iterations':self.opt_steps,'elites':self.topk,'candidate_batch':self.candidate_batch,'seconds':time.perf_counter()-begin})+'\n')
  return mu,np.full(len(mu),np.inf)
