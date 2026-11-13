#!/usr/bin/env python3
"""Isolated threshold validation; reuse the established planning implementations."""
import argparse, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf

R=Path('/home/appuser/workspace/project/multimodal-test/JEPA-Stall')
B=Path(__file__).resolve().parent
CONDITIONS=('adajepa_aug_g030','dinowm_aug_g030','dinowm_aug_dense_g030','dinowm_aug_dense_native')

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def model_hash(model):
    h=hashlib.sha256()
    for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def configuration(condition,target,output):
    assert condition in CONDITIONS
    dino=condition.startswith('dinowm');dense='dense' in condition;native=condition.endswith('native')
    c=OmegaConf.load(R/'conf/escape_plan_gd_pushobj.yaml');c.pop('hydra',None);c.pop('defaults',None)
    checkpoint=R/('checkpoints/dinowm_pushobj_v1/augmented' if dino else 'checkpoints/pushobj_bias_16k_e3_seed0')
    c.ckpt_base_path=str(checkpoint);c.model_epoch='3';c.n_evals=10;c.eval_data_path=str(target);c.wandb_logging=False;c.saved_folder=str(output)
    c.planner._target_='planning.stagnation.StagnationMPCPlanner';c.planner.max_iter=30;c.planner.n_taken_actions=5
    c.planner.stag=OmegaConf.create(dict(space='latent',kappa=0. if native else 2.,start=10,window=3,gamma_lo=.10,gamma_hi=.30,beta=.25,prop_boost=1. if native else 10.,boost_replans=2,alpha=1.,base=2.))
    if dino:
        if not dense:
            c=OmegaConf.merge(c,OmegaConf.load(R/'conf/dinowm_pushobj_official_cadence.yaml'))
            c.planner.stag.gamma_lo=.10;c.planner.stag.gamma_hi=.30
        else:
            c.planner.sub_planner=OmegaConf.create(dict(target='cem_memory.ChunkedCEMPlanner',horizon=25,num_samples=200,opt_steps=10,topk=30,var_scale=1,eval_every=-1,candidate_batch=64,precision='bf16',timing_path='cem_timings.jsonl'))
        c.objective.mode='last';c.planner.probe.mode='none';c.planner.adapt.finetune_encoder=False;c.planner.adapt.finetune_every=1
    c.planner.name=condition
    assert c.planner.max_iter*c.planner.n_taken_actions==150
    assert c.planner.stag.start*c.planner.n_taken_actions==50
    assert c.planner.stag.gamma_lo==.10 and c.planner.stag.gamma_hi==.30
    assert c.planner.escape.mode=='none'
    return c

def main():
    p=argparse.ArgumentParser();p.add_argument('--condition',choices=CONDITIONS,required=True);p.add_argument('--targets',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--config-only',action='store_true');a=p.parse_args()
    c=configuration(a.condition,a.targets,a.output)
    if a.config_only:print(OmegaConf.to_yaml(c));return
    a.output.mkdir(parents=True,exist_ok=True);assert not (a.output/'per_iter_logs.json').exists(),'Refuse to mix partial outputs'
    sys.path[:0]=[str(R/'reports/dinowm_pushobj_v3'),str(R/'reports/dinowm_pushobj_v1'),str(R)]
    import cem_memory
    assert Path(cem_memory.__file__).resolve()==R/'reports/dinowm_pushobj_v3/cem_memory.py', 'Wrong CEM implementation resolved'
    import inspect
    assert 'precision' in inspect.signature(cem_memory.ChunkedCEMPlanner.__init__).parameters
    torch.set_num_threads(2)
    if a.condition.startswith('dinowm'):
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if torch.cuda.is_available():torch.cuda.set_per_process_memory_fraction(10*2**30/torch.cuda.get_device_properties(0).total_memory,0)
    os.chdir(a.output);OmegaConf.save(c,'config.yaml')
    Path('runtime_modules.json').write_text(json.dumps({'cem_memory':str(Path(cem_memory.__file__).resolve()),'cem_memory_sha256':sha(cem_memory.__file__)},indent=2)+'\n')
    import plan
    original=plan.load_model;loaded={}
    def checked(*args,**kwargs):
        m=original(*args,**kwargs);loaded['model']=m;loaded['before']=model_hash(m);return m
    plan.load_model=checked
    begin=time.monotonic();checkpoint=Path(c.ckpt_base_path)/'checkpoints/model_3.pth';checkpoint_before=sha(checkpoint)
    plan.planning_main(OmegaConf.to_container(c,resolve=True))
    after=model_hash(loaded['model']);frozen=a.condition.startswith('dinowm')
    if frozen:
        timings=[json.loads(l)for l in Path('cem_timings.jsonl').read_text().splitlines()]
        assert timings and all(t['precision']=='bf16' and t['candidates']==200 and t['iterations']==10 for t in timings)
    if frozen:assert after==loaded['before'],'Frozen evaluation altered model parameters'
    assert sha(checkpoint)==checkpoint_before
    with np.load('final_eval_results.npz') as d:
        assert len(d['successes'])==10;assert d['actions'].shape[1]*5<=150
        successes=int(d['successes'].sum())
    episodes=[json.loads(l)for l in Path('stag_episodes.jsonl').read_text().splitlines()]
    assert len(episodes)==10 and {e['sample_idx']for e in episodes}==set(range(10))
    assert all(e['replans']<=c.planner.max_iter for e in episodes)
    logs=[json.loads(l)for l in Path('per_iter_logs.json').read_text().splitlines()]
    meta=json.loads(Path('stag_meta.json').read_text());assert meta['gamma_lo']==.10 and meta['gamma_hi']==.30
    sources=[B/'evaluate.py',R/'planning/stagnation.py',R/'planning/escape.py',R/'planning/adajepa_probe.py']
    if frozen:sources.extend([R/'reports/dinowm_pushobj_v3/cem_memory.py',R/'reports/dinowm_pushobj_v3/physical_stagnation.py'])
    completion=dict(status='complete',condition=a.condition,n=10,successes=successes,triggered=sum(e['first_trigger'] is not None for e in episodes),first_trigger_histogram={str(t):sum(e['first_trigger']==t for e in episodes)for t in sorted({e['first_trigger']for e in episodes if e['first_trigger'] is not None})},positive_penalty_records=sum(x.get('stag_pen',0)>0 for x in logs),boosted_plan_records=sum(bool(x.get('stag_boost',False))for x in logs),wall_seconds=time.monotonic()-begin,training_epochs=3,online_adaptation=not frozen,frozen_weights_unchanged=(after==loaded['before']) if frozen else None,checkpoint_file_unchanged=True,checkpoint_sha256=checkpoint_before,targets_sha256=sha(a.targets),config_sha256=sha('config.yaml'),source_sha256={str(p.relative_to(R)):sha(p)for p in sources},max_mpc_replans=c.planner.max_iter,executed_controls_per_plan=c.planner.n_taken_actions,max_environment_controls=150,thresholds=[.10,.30],seed=c.seed)
    Path('complete.tmp').write_text(json.dumps(completion,indent=2)+'\n');Path('complete.tmp').replace('complete.json');print(json.dumps(completion),flush=True)

if __name__=='__main__':main()
