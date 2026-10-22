"""Encode a PushObj dataset split with the base world model, for the cost-to-go head.

Every frame of every trajectory is encoded (base encoder, eval mode), and for a few start frames per
trajectory the world model also predicts the next 5 model steps from that frame with the data's own
actions, exactly as the planner rolls out a plan. Output in <out>/:

    vis.npy (N, Dv) fp16, prop.npy (N, Dp) fp16   latents of all N frames, trajectories back to back
    states.npy (N, 7) fp32                       agent xy, block xy, block angle, agent velocity
    ep_start.npy, ep_len.npy, shapes.json        trajectory boundaries and shapes
    pred_vis.npy, pred_prop.npy (M, D) fp16      predicted latents; pred_idx.npy (M,) is the frame each
                                                 prediction stands for, pred_k.npy (M,) its depth (1..5)

    python scripts/ctg_encode.py --data data/pushobj_bias_16k/train --out eval_outputs/ctg/bias16k_train --gpus 4,5,6,7
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import pickle  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
H, FS = 5, 5  # predicted model steps, frameskip


def work(job):
    import sys
    import torch
    sys.path.insert(0, str(ROOT / "scripts"))
    from decord import VideoReader
    from stag_common import load_world_model
    from utils import move_to_device

    import decord

    torch.set_num_threads(2)
    a, eps, wid = job["args"], job["eps"], job["wid"]
    dev = f"cuda:{wid % len(a['gpus'].split(','))}"
    wm, pre, _ = load_world_model(device=dev)
    decord.bridge.set_bridge("native")  # the dataset module switches decord to torch tensors on import
    data = Path(a["data"])
    states = torch.load(data / "states.pth").float().numpy()
    vel = torch.load(data / "velocities.pth").float().numpy()
    acts = (torch.load(data / "rel_actions.pth").float() / 100.0)
    acts = ((acts - pre.action_mean) / pre.action_std).numpy()
    lens = pickle.load(open(data / "seq_lengths.pkl", "rb"))
    rng = np.random.default_rng([a["seed"], wid])
    out = {k: [] for k in ("vis", "prop", "states", "pvis", "pprop", "pidx", "pk")}
    for e in eps:
        T = int(lens[e])
        fr = VideoReader(str(data / "obses" / f"episode_{e:03d}.mp4"), num_threads=1).get_batch(list(range(T))).asnumpy()
        prop = np.concatenate([states[e, :T, :2], vel[e, :T]], -1).astype(np.float32)
        with torch.no_grad():
            vs, ps = [], []
            for i in range(0, T, 256):
                z = wm.encode_obs(move_to_device(pre.transform_obs({"visual": fr[i:i + 256][:, None], "proprio": prop[i:i + 256][:, None]}), dev))
                vs.append(z["visual"][:, 0].flatten(1).half().cpu())
                ps.append(z["proprio"][:, 0].flatten(1).half().cpu())
            out["vis"].append(torch.cat(vs).numpy())
            out["prop"].append(torch.cat(ps).numpy())
            out["states"].append(np.concatenate([states[e, :T, :5], vel[e, :T]], -1).astype(np.float32))
            starts = np.arange(0, T - H * FS)
            if len(starts):
                starts = rng.choice(starts, size=min(a["starts"], len(starts)), replace=False)
                obs0 = {"visual": fr[starts][:, None], "proprio": prop[starts][:, None]}
                act = np.stack([acts[e, s : s + H * FS].reshape(H, FS * acts.shape[-1]) for s in starts])
                zo, _ = wm.rollout(obs_0=move_to_device(pre.transform_obs(obs0), dev),
                                   act=torch.as_tensor(act, dtype=torch.float32, device=dev))
                for k in range(1, H + 1):
                    out["pvis"].append(zo["visual"][:, k].flatten(1).half().cpu().numpy())
                    out["pprop"].append(zo["proprio"][:, k].flatten(1).half().cpu().numpy())
                    out["pidx"].append(np.stack([np.full(len(starts), e), starts + k * FS], 1))
                    out["pk"].append(np.full(len(starts), k))
    shard = Path(a["out"]) / f"_shard{wid:02d}.npz"
    np.savez(shard, eps=np.asarray(eps), **{k: (np.concatenate(v) if v else np.zeros((0,))) for k, v in out.items()})
    return str(shard)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpus", default="4,5,6,7")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--starts", type=int, default=4, help="prediction start frames per trajectory")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpus
    a.data, a.out = str(ROOT / a.data), str(ROOT / a.out)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    lens = pickle.load(open(Path(a.data) / "seq_lengths.pkl", "rb"))
    n = len(lens)
    jobs = [{"args": vars(a), "eps": list(range(w, n, a.workers)), "wid": w} for w in range(a.workers)]
    with mp.get_context("spawn").Pool(a.workers) as pool:
        shards = pool.map(work, jobs)
    parts = [dict(np.load(s)) for s in shards]
    order = {}  # episode -> (shard, position within shard)
    for si, p in enumerate(parts):
        off = 0
        L = [int(lens[e]) for e in p["eps"]]
        for e, l in zip(p["eps"], L):
            order[int(e)] = (si, off, l)
            off += l
    vis, prop, st, start = [], [], [], []
    pos = 0
    for e in range(n):
        si, off, l = order[e]
        vis.append(parts[si]["vis"][off : off + l])
        prop.append(parts[si]["prop"][off : off + l])
        st.append(parts[si]["states"][off : off + l])
        start.append(pos)
        pos += l
    start = np.asarray(start)
    out = Path(a.out)
    np.save(out / "vis.npy", np.concatenate(vis))
    np.save(out / "prop.npy", np.concatenate(prop))
    np.save(out / "states.npy", np.concatenate(st))
    np.save(out / "ep_start.npy", start)
    np.save(out / "ep_len.npy", np.asarray([int(x) for x in lens[:n]]))
    pv = np.concatenate([p["pvis"] for p in parts if len(p["pvis"])])
    pp = np.concatenate([p["pprop"] for p in parts if len(p["pprop"])])
    pi = np.concatenate([p["pidx"] for p in parts if len(p["pidx"])])
    pk = np.concatenate([p["pk"] for p in parts if len(p["pk"])])
    np.save(out / "pred_vis.npy", pv)
    np.save(out / "pred_prop.npy", pp)
    np.save(out / "pred_idx.npy", start[pi[:, 0]] + pi[:, 1])  # global frame index
    np.save(out / "pred_k.npy", pk)
    shapes_file = Path(a.data) / "shapes.pkl"
    shapes = pickle.load(open(shapes_file, "rb"))[:n] if shapes_file.exists() else ["T"] * n
    (out / "shapes.json").write_text(json.dumps(list(shapes)))
    for s in shards:
        os.remove(s)
    print(f"{n} trajectories, {len(start) and pos} frames, {len(pv)} predicted latents -> {out}")


if __name__ == "__main__":
    main()
