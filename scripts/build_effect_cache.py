"""Materialize positive-only labels; no object state enters the observer.

Reuse each decoded trajectory and each unique 5/15-step pair. The OR condition
short-circuits optical flow once pusher motion + RGB change proves visual effect.
No class thresholds are changed relative to Phase A.
"""
import os
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[k] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing as mp
import pickle
import resource
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from decord import VideoReader
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.effect_observer import motion_features, body_label, exclusion_mask, largest_component, tracked_motion
cv2.setNumThreads(1)
torch.set_num_threads(1)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for x in iter(lambda: f.read(4*1024*1024), b""):
            h.update(x)
    return h.hexdigest()


def positive_pair(rgb, gray, prop, actions, s, e, cfg):
    p = body_label(motion_features(prop[s:e+1, :2], prop[s:e+1, 2:], actions[s:e]), cfg) == 1
    changed = np.max(np.abs(rgb[e].astype(np.int16) - rgb[s].astype(np.int16)), -1) > cfg["rgb_diff_floor"]
    v = p and largest_component(changed) >= cfg["changed_component_pixels"]
    if not v:
        mask = exclusion_mask(prop[s:e+1, :2], rgb.shape[1:3], cfg)
        if largest_component(changed & (mask > 0)) >= cfg["changed_component_pixels"]:
            cv2.setRNGSeed(0)  # deterministic per unique pair, independent of scheduling
            f = tracked_motion(gray[s], gray[e], mask, cfg)
            v = f["coherent"] >= cfg["min_moving_corners"]
    return v, p


def worker(job):
    ep, path, prop, acts, cfg = job
    t0 = time.process_time()
    rgb = VideoReader(str(path), num_threads=1).get_batch(list(range(len(prop)))).asnumpy()
    gray = np.stack([cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) for im in rgb])
    labels = np.zeros((len(prop), 2, 2), dtype=bool)
    for hi, h in enumerate((5, 15)):
        for s in range(len(prop)-h):
            labels[s, hi] = positive_pair(rgb, gray, prop, acts, s, s+h, cfg)
    return ep, labels, {"episode": ep, "video_sha256": digest(path), "frames": len(prop),
                        "worker_cpu_seconds": time.process_time()-t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/pushobj_bias_16k")
    ap.add_argument("--out", default="data/pushobj_bias_16k/effect_cache_v1")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    data, out = ROOT/args.data, ROOT/args.out
    out.mkdir(parents=True, exist_ok=True)
    if (out/"complete.json").exists():
        print("Cache already complete; no regeneration", flush=True)
        return
    if (out/"train.npy").exists():
        raise RuntimeError("Partial cache exists; inspect it before restarting in a new directory")
    cfg_path = ROOT/"reports/effect_ab_20261103/protocol.json"
    cfg = json.loads(cfg_path.read_text())["observer"]
    started = time.monotonic()
    meta = {"data_path": str(data.resolve()), "observer_thresholds": cfg,
            "layout": "[episode,time,horizon(5,15),branch(visual,proprio)]",
            "label": "positive evidence only; False means no auxiliary constraint, NOT static",
            "rng": "OpenCV seed 0 per unique pair; positive OR condition short-circuited",
            "input_hashes": {}, "splits": {}}
    for path in (cfg_path, Path(__file__), ROOT/"datasets/effect_observer.py"):
        meta["input_hashes"][str(path.relative_to(ROOT))] = digest(path)
    for split in ("train", "val"):
        d = data/split
        lengths = pickle.loads((d/"seq_lengths.pkl").read_bytes())
        xy = torch.load(d/"states.pth", weights_only=True)[..., :2].numpy().copy()
        vel = torch.load(d/"velocities.pth", weights_only=True).numpy()
        acts = torch.load(d/"rel_actions.pth", weights_only=True).numpy()
        for name in ("states.pth", "velocities.pth", "rel_actions.pth", "seq_lengths.pkl", "provenance.json"):
            meta["input_hashes"][str((d/name).relative_to(ROOT))] = digest(d/name)
        labels = np.lib.format.open_memmap(out/f"{split}.npy", mode="w+", dtype=bool,
                                           shape=(len(lengths), xy.shape[1], 2, 2))
        labels[:] = False
        jobs = ((i, d/"obses"/f"episode_{i:03d}.mp4",
                 np.concatenate([xy[i, :int(n)], vel[i, :int(n)]], -1), acts[i, :int(n)].copy(), cfg)
                 for i, n in enumerate(lengths))
        with mp.get_context("spawn").Pool(args.workers) as pool, (out/f"{split}_videos.jsonl").open("w") as f:
            for done, (ep, lab, info) in enumerate(pool.imap_unordered(worker, jobs, chunksize=1), 1):
                labels[ep, :len(lab)] = lab
                f.write(json.dumps(info)+"\n")
                if done % 100 == 0:
                    f.flush()
                    print(f"{split}: {done}/{len(lengths)} episodes, wall={time.monotonic()-started:.1f}s", flush=True)
        labels.flush()
        meta["splits"][split] = {"episodes": len(lengths), "frames": int(sum(lengths)),
            "positive_counts_by_horizon_branch": labels.sum(axis=(0, 1)).tolist(),
            "sha256": digest(out/f"{split}.npy")}
        del labels, xy, vel, acts
    r = resource.getrusage(resource.RUSAGE_CHILDREN)
    meta["wall_seconds"] = time.monotonic()-started
    meta["cpu_core_hours_including_parent"] = (r.ru_utime+r.ru_stime+time.process_time())/3600
    meta["gpu_hours"] = 0
    (out/"complete.json.tmp").write_text(json.dumps(meta, indent=2)+"\n")
    (out/"complete.json.tmp").replace(out/"complete.json")
    print(json.dumps(meta["splits"], indent=2), flush=True)


if __name__ == "__main__":
    main()
