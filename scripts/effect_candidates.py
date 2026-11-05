"""Frozen common candidates: two WM proposals + smooth random + recorded action perturbations.

Simulator outcomes are diagnostics ONLY, never used by B training. Candidate
arrays and IDs are sealed once and reused for every subsequent model.
"""
import os
for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[k]="1"
os.environ.setdefault("SDL_VIDEODRIVER","dummy")
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"scripts"))
from stag_common import load_world_model, make_env, goal_state, ref_episodes, stream_targets
from planning.objectives import create_objective_fn
from utils import move_to_device
torch.set_num_threads(1)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(4*1024*1024),b""):h.update(b)
    return h.hexdigest()


def select_starts():
    ref="eval_outputs/stag_frozen_bias16k_it40/ref"
    episodes=ref_episodes(ref,max_iter=30)
    rng=np.random.default_rng(20261103)
    keys=sorted(episodes)
    stalled=[]
    for key in keys:
        first=next((r for r in episodes[key] if r["step"]>=10 and not r["success"] and r.get("stalled",False)),None)
        if first is not None:stalled.append((key,first))
    if len(stalled)<32:raise RuntimeError("Fewer than 32 logged stalled decisions")
    stalled=[stalled[i] for i in rng.choice(len(stalled),32,replace=False)]
    excluded={key for key,_ in stalled}
    ordinary=[key for key in keys if key not in excluded]
    ordinary=[ordinary[i] for i in rng.choice(len(ordinary),32,replace=False)]
    selected=[(k,None) for k in ordinary]+stalled
    out=[]
    for j,(key,r) in enumerate(selected):
        chunk,i=key
        seg=stream_targets("EVAL50",chunk)[i]
        out.append({"id":f"decision_{j:03d}","chunk":chunk,"sample_idx":i,"shape":seg["shape"],
            "kind":"ordinary" if r is None else "stalled","step":0 if r is None else r["step"],
            "requested_state":seg["init"].tolist() if r is None else r["state"],
            "reference":ref,"reference_sha256":sha(ROOT/ref/chunk/"per_iter_logs.json"),
            "target_sha256":sha(ROOT/"data/targets/EVAL50"/(chunk+".pkl"))})
    return out


def normalize(pre, actions):
    x=torch.as_tensor(actions/100,dtype=torch.float32)
    return pre.normalize_actions(x).reshape(len(actions),5,10).cuda()


def propose(wm,pre,obs,goal,initial,step):
    ob={k:np.repeat(v[None,None],4,axis=0) for k,v in obs.items()}
    go={k:np.repeat(v[None,None],4,axis=0) for k,v in goal.items()}
    ob=move_to_device(pre.transform_obs(ob),"cuda:0")
    go=move_to_device(pre.transform_obs(go),"cuda:0")
    with torch.no_grad():zg=wm.encode_obs(go)
    actions=normalize(pre,initial).detach().requires_grad_(True)
    opt=torch.optim.Adam([actions],lr=.1)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=100,eta_min=0)
    objective=create_objective_fn(alpha=1,base=2,mode="staged")
    for _ in range(100):
        opt.zero_grad(set_to_none=True)
        zo,_=wm.rollout(ob,actions)
        objective(zo,zg,step=step).sum().backward()
        opt.step();scheduler.step()
    normalized=actions.detach().cpu().reshape(4,25,2)
    return (pre.denormalize_actions(normalized)*100).numpy()


def create(out):
    if (out/"candidates.npz").exists():raise RuntimeError("Candidate set already sealed")
    starts=select_starts()
    (out/"starts.json").write_text(json.dumps(starts,indent=2)+"\n")
    models=[load_world_model("checkpoints/pushobj_bias_16k_e3_seed0","3")[:2],
            load_world_model("checkpoints/pushobj_shape_shift","latest")[:2]]
    all_actions=[];all_frames=[];all_props=[];all_states=[];goals=[];goal_props=[];goal_states=[];ids=[]
    rng=np.random.default_rng(20261103)
    t0=time.monotonic()
    for j,st in enumerate(starts):
        seg=stream_targets("EVAL50",st["chunk"])[st["sample_idx"]]
        gs,go=goal_state(seg)
        env=make_env(st["shape"])
        requested=np.asarray(st["requested_state"],dtype=np.float64)
        ob,settled=env.prepare(0,requested)
        # Piecewise smooth noise in pixel action units, fixed before any scoring.
        anchors=rng.normal(0,20,(7,5,2))
        smooth=np.stack([np.stack([np.interp(np.arange(25),[0,6,12,18,24],x[:,d]) for d in range(2)],-1) for x in anchors])
        init=np.concatenate([np.zeros((1,25,2)),smooth[:3]],0)
        proposed=[propose(wm,pre,ob,go,init,st["step"]) for wm,pre in models]
        recorded=np.asarray(seg["actions"],dtype=np.float64)
        if len(recorded)<25:recorded=np.pad(recorded,((0,25-len(recorded)),(0,0)))
        perturbed=recorded[:25][None]+rng.normal(0,5,(4,25,2))
        candidates=np.concatenate([*proposed,smooth[3:7],perturbed]).astype(np.float32)
        frames=[];props=[];states=[]
        for a in candidates:
            observations,ss=env.rollout(0,requested,a/100)
            frames.append(observations["visual"][::5]);props.append(observations["proprio"][::5]);states.append(ss[::5])
        states=np.stack(states)
        if not np.allclose(states[:,0],states[0,0],atol=1e-6):raise RuntimeError("Candidate initial states differ")
        st["settled_state"]=states[0,0].tolist()
        all_actions.append(candidates);all_frames.append(np.stack(frames));all_props.append(np.stack(props));all_states.append(states)
        goals.append(go["visual"]);goal_props.append(go["proprio"]);goal_states.append(gs)
        ids.append([f"{st['id']}:{group}:{k}" for group in ("ours_gd","official_gd","smooth","recorded_perturb") for k in range(4)])
        print(f"candidate decisions {j+1}/64; elapsed={time.monotonic()-t0:.1f}s",flush=True)
    np.savez_compressed(out/"candidates.npz",actions=np.stack(all_actions),frames=np.stack(all_frames),
        proprio=np.stack(all_props),states=np.stack(all_states),goal_frames=np.stack(goals),
        goal_proprio=np.stack(goal_props),goal_states=np.stack(goal_states))
    (out/"starts.json").write_text(json.dumps(starts,indent=2)+"\n")
    meta={"starts":64,"candidates_per_start":16,"ids":ids,"seed":20261103,
        "horizon_env_steps":25,"boundary_interval":5,"candidate_sha256":sha(out/"candidates.npz"),
        "starts_sha256":sha(out/"starts.json"),"gpu_wall_seconds_generation":time.monotonic()-t0,
        "notes":["Historical frozen reference states are rounded; these are simulator restart diagnostics, not bit-exact continuation.",
                 "Stalled means the historical pusher movement trigger, not the unavailable training global-static class.",
                 "25-step physical closeness is not a reachability oracle. No simulator outcomes enter B training."]}
    (out/"manifest.json").write_text(json.dumps(meta,indent=2)+"\n")
    del models
    torch.cuda.empty_cache()


def rank_corr(a,b):
    if np.std(a)<1e-12 or np.std(b)<1e-12:return None
    return float(spearmanr(a,b).statistic)


def score(out,ckpt,epoch,name):
    with np.load(out/"candidates.npz") as archive:
        data={k:archive[k] for k in archive.files}
    starts=json.loads((out/"starts.json").read_text())
    wm,pre,_=load_world_model(ckpt,epoch)
    objective=create_objective_fn(alpha=1,base=2,mode="staged")
    visual_objective=create_objective_fn(alpha=0,base=2,mode="staged")
    records=[];representations=[];proprio_representations=[]
    with torch.no_grad():
        for i,st in enumerate(starts):
            frames=data["frames"][i];prop=data["proprio"][i].astype(np.float32)
            obs=move_to_device(pre.transform_obs({"visual":frames,"proprio":prop}),"cuda:0")
            goal=move_to_device(pre.transform_obs({"visual":data["goal_frames"][i][None,None],
                    "proprio":data["goal_proprio"][i][None,None].astype(np.float32)}),"cuda:0")
            encoded=wm.encode_obs(obs);zg=wm.encode_obs(goal)
            predicted,_=wm.rollout({k:v[:,:1] for k,v in obs.items()},normalize(pre,data["actions"][i]))
            pc=objective(predicted,zg,step=st["step"]).float().cpu().numpy()
            ec=objective(encoded,zg,step=st["step"]).float().cpu().numpy()
            pv=visual_objective(predicted,zg,step=st["step"]).float().cpu().numpy()
            ev=visual_objective(encoded,zg,step=st["step"]).float().cpu().numpy()
            normalized_errors={}
            for branch in ("visual","proprio"):
                pred=predicted[branch][:,1:].float().flatten(2)
                actual=encoded[branch][:,1:].float().flatten(2)
                pn=torch.nn.functional.layer_norm(pred,(pred.shape[-1],))
                an=torch.nn.functional.layer_norm(actual,(actual.shape[-1],))
                normalized_errors[branch]=float((pn-an).square().mean())
            states=data["states"][i];g=data["goal_states"][i]
            position=np.linalg.norm(states[...,:4]-g[:4],axis=-1)
            angle=np.abs((states[...,4]-g[4]+np.pi)%(2*np.pi)-np.pi)
            success=(position<20)&(angle<np.pi/9)
            phys=position+angle*(20/(np.pi/9))
            weights=2.**np.arange(6);weights/=weights.sum()
            physical=phys[:,-1] if st["step"]<5 else (phys*weights).mean(-1)
            pi,ei,oi=int(np.argmin(pc)),int(np.argmin(ec)),int(np.argmin(physical))
            record={"id":st["id"],"kind":st["kind"],"shape":st["shape"],"predicted_cost":pc.tolist(),
                "encoded_cost":ec.tolist(),"physical_cost":physical.tolist(),
                "predicted_cost_visual":pv.tolist(),"predicted_cost_proprio":(pc-pv).tolist(),
                "encoded_cost_visual":ev.tolist(),"encoded_cost_proprio":(ec-ev).tolist(),
                "per_sample_standardized_prediction_MSE":normalized_errors,
                "rank_pred_encoded":rank_corr(pc,ec),"rank_pred_physical":rank_corr(pc,physical),
                "rank_encoded_physical":rank_corr(ec,physical),"top4_pred":np.argsort(pc)[:4].tolist(),
                "top4_encoded":np.argsort(ec)[:4].tolist(),"top4_physical":np.argsort(physical)[:4].tolist(),
                "chosen_pred":pi,"chosen_encoded":ei,"chosen_physical":oi,
                "selected_pred_success_end":bool(success[pi,-1]),"selected_encoded_success_end":bool(success[ei,-1]),
                "selected_pred_success_first5":bool(success[pi,1]),
                "any_success_candidate_end":bool(success[:,-1].any()),
                "any_success_candidate_boundary":bool(success[:,1:].any()),
                "selected_pred_physical_cost":float(physical[pi]),
                "selected_encoded_physical_cost":float(physical[ei]),
                "selected_pred_object_displacement":float(np.linalg.norm(states[pi,-1,2:4]-states[pi,0,2:4]))}
            records.append(record)
            representations.append(encoded["visual"].float().flatten(2).reshape(-1,encoded["visual"].shape[-1]).cpu().numpy())
            proprio_representations.append(encoded["proprio"].float().reshape(-1,encoded["proprio"].shape[-1]).cpu().numpy())
    z=np.concatenate(representations);singular=np.linalg.svd(z-z.mean(0),compute_uv=False)
    p=singular/max(singular.sum(),1e-12);effective_rank=float(np.exp(-(p*np.log(np.maximum(p,1e-12))).sum()))
    summary={"model":ckpt,"epoch":epoch,"candidate_sha256":sha(out/"candidates.npz"),
        "latent_visual_mean_norm":float(np.linalg.norm(z,axis=-1).mean()),
        "latent_visual_mean_variance":float(z.var(0).mean()),"latent_visual_effective_rank":effective_rank,
        "decisions":len(records),"records":records,
        "latent_proprio_mean_variance":float(np.concatenate(proprio_representations).var(0).mean()),
        "scoring_code_sha256":sha(Path(__file__))}
    reference_path=out/"scores_B0.json"
    reference=json.loads(reference_path.read_text()) if reference_path.exists() and name!="B0" else summary
    scales={branch:reference[f"latent_{branch}_mean_variance"]/max(summary[f"latent_{branch}_mean_variance"],1e-12)
            for branch in ("visual","proprio")}
    summary["scale_matching_reference"]="B0" if reference is not summary else name
    summary["scale_matching_factors"]=scales
    for r in records:
        scaled=scales["visual"]*np.asarray(r["predicted_cost_visual"])+scales["proprio"]*np.asarray(r["predicted_cost_proprio"])
        selected=int(np.argmin(scaled))
        r["scale_matched_chosen_pred"]=selected
        r["scale_matched_rank_pred_physical"]=rank_corr(scaled,np.asarray(r["physical_cost"]))
        r["scale_matched_selected_physical_cost"]=r["physical_cost"][selected]
    summary["mean_standardized_prediction_MSE"]={b:float(np.mean([r["per_sample_standardized_prediction_MSE"][b] for r in records])) for b in ("visual","proprio")}
    for key in ("rank_pred_encoded","rank_pred_physical","rank_encoded_physical"):
        values=[r[key] for r in records if r[key] is not None]
        summary["mean_"+key]=float(np.mean(values)) if values else None
    for key in ("selected_pred_success_end","selected_encoded_success_end","any_success_candidate_end"):
        summary[key+"_count"]=sum(r[key] for r in records)
    (out/f"scores_{name}.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps({k:v for k,v in summary.items() if k!="records"},indent=2),flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",default="eval_outputs/effect_b_20261103/candidates")
    ap.add_argument("--create",action="store_true")
    ap.add_argument("--ckpt");ap.add_argument("--epoch",default="3");ap.add_argument("--name")
    a=ap.parse_args();out=ROOT/a.out;out.mkdir(parents=True,exist_ok=True)
    if a.create:
        create(out)
        score(out,"checkpoints/pushobj_bias_16k_e3_seed0","3","existing_ours")
        score(out,"checkpoints/pushobj_shape_shift","latest","existing_official")
    elif a.ckpt and a.name:score(out,a.ckpt,a.epoch,a.name)
    else:ap.error("Provide --create or --ckpt and --name")


if __name__=="__main__":main()
