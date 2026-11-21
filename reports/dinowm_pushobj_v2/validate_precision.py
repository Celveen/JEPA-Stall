"""Compare complete CEM searches on saved observations, without running new tasks."""
import argparse
import contextlib
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

p = argparse.ArgumentParser()
p.add_argument('--arm', choices=['ordinary', 'augmented'], required=True)
a = p.parse_args()
R = Path('/home/appuser/workspace/project/multimodal-test/JEPA-Stall')
B = Path(__file__).resolve().parent
sys.path[:0] = [str(B), str(R/'reports/dinowm_pushobj_v1'), str(R)]
from plan import load_model
from preprocessor import Preprocessor
from datasets.pusht_dset import ACTION_MEAN, ACTION_STD, STATE_MEAN, STATE_STD, PROPRIO_MEAN, PROPRIO_STD
from datasets.img_transforms import default_transform
from cem_memory import ChunkedCEMPlanner, rollout_cached
from planning.objectives import create_objective_fn

torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.cuda.set_per_process_memory_fraction(4*2**30 / torch.cuda.get_device_properties(0).total_memory)
folder = R/'checkpoints/dinowm_pushobj_v1'/a.arm
wm = load_model(folder/'checkpoints/model_3.pth', OmegaConf.load(folder/'hydra.yaml'), 1, 'cuda')
wm.eval()
for param in wm.parameters():param.requires_grad_(False)
def fingerprint():
    h = hashlib.sha256()
    for k,v in wm.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
before = fingerprint()
data = pickle.loads((R/'eval_outputs/dinowm_pushobj_3epoch_v1'/f'{a.arm}_native'/'chunk_00_I/plan_targets.pkl').read_bytes())
prep = Preprocessor(ACTION_MEAN,ACTION_STD,STATE_MEAN[:5],STATE_STD[:5],PROPRIO_MEAN[:4],PROPRIO_STD[:4],default_transform(224))
objective = create_objective_fn(alpha=1,base=2,mode='last')
class Log:
    def log(self,*args,**kwargs):pass
rows=[]
with torch.no_grad():
    for idx in [0,5]:
        obs={k:v[idx:idx+1] for k,v in data['obs_0'].items()}
        goal={k:v[idx:idx+1] for k,v in data['obs_g'].items()}
        z0=wm.encode_obs({k:v.cuda() for k,v in prep.transform_obs(obs).items()})
        zg=wm.encode_obs({k:v.cuda() for k,v in prep.transform_obs(goal).items()})
        outputs={};seconds={};costs={}
        for precision in ['fp32','bf16']:
            planner=ChunkedCEMPlanner(candidate_batch=64,precision=precision,horizon=5,topk=30,num_samples=200,var_scale=1,opt_steps=10,eval_every=-1,wm=wm,action_dim=10,objective_fn=objective,preprocessor=prep,evaluator=None,wandb_run=Log())
            # Same candidate random numbers at every iteration, independent of precision.
            torch.manual_seed(100*idx+1)
            torch.cuda.synchronize();t=time.perf_counter()
            outputs[precision],_=planner.plan(obs,goal)
            torch.cuda.synchronize();seconds[precision]=time.perf_counter()-t
            pred,_=rollout_cached(wm,z0,outputs[precision])
            costs[precision]=float(objective(pred,zg)[0])
        delta=outputs['bf16']-outputs['fp32']
        row={'sample_idx':idx,'seconds':seconds,'speedup':seconds['fp32']/seconds['bf16'],'actions_finite':all(bool(torch.isfinite(x).all()) for x in outputs.values()),'action_rms_difference':float(delta.square().mean().sqrt()),'executed_action_rms_difference':float(delta[:,:1].square().mean().sqrt()),'max_action_difference':float(delta.abs().max()),'fp32_rescored_costs':costs,'relative_cost_increase':(costs['bf16']-costs['fp32'])/max(abs(costs['fp32']),1e-9),'actions_identical':bool(torch.equal(outputs['fp32'],outputs['bf16']))}
        rows.append(row);print(json.dumps(row),flush=True)
        d=B/'validation';d.mkdir(exist_ok=True)
        np.savez_compressed(d/f'{a.arm}_sample{idx}_actions.npz',**{k:v.cpu().numpy() for k,v in outputs.items()})
        (d/f'{a.arm}_precision.json').write_text(json.dumps({'arm':a.arm,'rows':rows,'frozen_weights_unchanged':fingerprint()==before,'interpretation':'Numerical diagnostic, not closed-loop success equivalence.'},indent=2)+'\n')
assert fingerprint()==before
