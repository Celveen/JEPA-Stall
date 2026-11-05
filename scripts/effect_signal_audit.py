"""Phase A signal audit; no training and no checkpoint modification.

The observer worker sees only video, observed pusher xy/velocity, and executed
actions. Object truth is loaded separately AFTER observational labels are saved.
Every dev window is physically audited; absent strata are reported, not filled
by relabeling moving-pusher/static-object preparation as global stagnation.
"""
import os
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing as mp
import pickle
import resource
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from decord import VideoReader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.effect_observer import PAIRS, measure_window, motion_features, body_label

torch.set_num_threads(1)
cv2.setNumThreads(1)


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
            "sha256": h.hexdigest()}


def select_windows(data, split, n, seed):
    prov = json.loads((data / split / "provenance.json").read_text())
    by_source = {}
    for row in prov:
        if row["length"] >= 20:
            by_source.setdefault(row["source_index"], []).append(row)
    rng = np.random.default_rng(seed)
    if len(by_source) < n:
        raise ValueError("Not enough distinct source episodes")
    sources = rng.choice(sorted(by_source), n, replace=False)
    samples = []
    for src in sources:
        rows = by_source[int(src)]
        row = rows[int(rng.integers(len(rows)))]
        start = int(rng.integers(row["length"] - 20 + 1))
        samples.append({"split": split, "source_index": int(src),
            "episode": row["episode"], "shape": row["shape"], "start": start,
            "end_exclusive": start + 20, "video_frames": [start, start + 15],
            "id": f"{split}:src{src}:ep{row['episode']}:t{start}"})
    return samples


def load_observable_arrays(data, split):
    # states.pth is the storage container for proprio xy. Discard object
    # dimensions immediately. Worker payloads cannot access them.
    xy = torch.load(data / split / "states.pth", weights_only=True)[..., :2].numpy().copy()
    vel = torch.load(data / split / "velocities.pth", weights_only=True).numpy()
    act = torch.load(data / split / "rel_actions.pth", weights_only=True).numpy()
    return xy, vel, act


def worker(job):
    sample, video, prop, actions, cfg, world_size = job
    started = time.process_time()
    cv2.setRNGSeed(0)
    frames = VideoReader(video, num_threads=1).get_batch(
        list(range(sample["start"], sample["start"] + 16))).asnumpy()
    out = measure_window(frames, prop, actions, cfg, world_size)
    out["sample"] = sample
    out["worker_cpu_seconds"] = time.process_time() - started
    return out


def run_observer(data, samples, protocol, out):
    split = samples[0]["split"]
    xy, vel, actions = load_observable_arrays(data, split)
    jobs = []
    for sample in samples:
        ep, s = sample["episode"], sample["start"]
        prop = np.concatenate([xy[ep, s:s+16], vel[ep, s:s+16]], -1)
        jobs.append((sample, str(data / split / "obses" / f"episode_{ep:03d}.mp4"),
                     prop, actions[ep, s:s+15].copy(), protocol["observer"], protocol["world_size"]))
    del xy, vel, actions
    records = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=protocol["A_cpu_workers"],
            mp_context=mp.get_context("spawn")) as pool, out.open("w") as f:
        for i, result in enumerate(pool.map(worker, jobs, chunksize=8)):
            f.write(json.dumps(result) + "\n")
            records.append(result)
            if (i + 1) % 100 == 0:
                f.flush()
                print(f"{split}: {i+1}/{len(samples)} measured", flush=True)
    return records


def physical_truth(states, vel, actions, protocol):
    truth = []
    c = protocol["audit_only"]
    for s, e in PAIRS:
        st = states[s:e+1]
        body = motion_features(st[:, :2], vel[s:e+1], actions[s:e])
        b = body_label(body, {**c, "min_action_rms_px": protocol["observer"]["min_action_rms_px"]})
        xy = st[:, 2:4]
        angle = np.unwrap(st[:, 4].astype(np.float64))
        d = np.linalg.norm(xy - xy[0], axis=-1)
        path = float(np.linalg.norm(np.diff(xy, axis=0), axis=-1).sum())
        rot = np.rad2deg(np.abs(angle - angle[0]))
        rot_path = float(np.rad2deg(np.abs(np.diff(angle)).sum()))
        obj_effect = ((d[-1] >= c["object_effect_translation_px"] and
                       d[-1] / max(path, 1e-8) >= c["min_net_path_ratio"]) or
                      (rot[-1] >= c["object_effect_rotation_deg"] and
                       rot[-1] / max(rot_path, 1e-8) >= c["min_net_path_ratio"]))
        obj_static = (d.max() <= c["object_static_translation_px"] and
                      rot.max() <= c["object_static_rotation_deg"] and
                      body["action_rms"] >= protocol["observer"]["min_action_rms_px"])
        obj = 1 if obj_effect else 0 if obj_static else -1
        visual = 1 if (b == 1 or obj == 1) else 0 if (b == 0 and obj == 0) else -1
        returned = ((body["excursion"] >= 5 and body["net"] < 2.5) or
                    (d.max() >= 5 and d[-1] < 2.5) or (rot.max() >= 5 and rot[-1] < 2.5))
        truth.append({"pair": [s, e], "labels": {"proprio": b, "visual": visual,
                      "object_diagnostic": obj}, "body": body,
                      "object_net_px": float(d[-1]), "object_excursion_px": float(d.max()),
                      "object_net_deg": float(rot[-1]), "object_excursion_deg": float(rot.max()),
                      "preparation_proxy": b == 1 and obj == 0, "return_proxy": bool(returned)})
    return truth


def metrics(pred, true):
    pred, true = np.asarray(pred), np.asarray(true)
    classes = [-1, 0, 1]
    confusion = [[int(((true == t) & (pred == p)).sum()) for p in classes] for t in classes]
    result = {"n": len(pred), "confusion_rows_truth_cols_prediction": confusion,
              "class_order": ["unknown", "ineffective_attempt", "effect"]}
    for k, name in ((0, "ineffective_attempt"), (1, "effect")):
        n_pred, n_true = int((pred == k).sum()), int((true == k).sum())
        tp = int(((pred == k) & (true == k)).sum())
        result[name] = {"predicted": n_pred, "true": n_true, "correct": tp,
            "precision": tp / n_pred if n_pred else None,
            "recall": tp / n_true if n_true else None,
            "coverage": n_pred / len(pred), "truth_prevalence": n_true / len(pred)}
    result["unknown_predictions"] = int((pred == -1).sum())
    return result


def audit(data, records, protocol, out):
    # Audit boundary: observer JSONL has been closed and hashed before loading
    # physical object states. No oracle labels are added to observer caches.
    states = torch.load(data / "val/states.pth", weights_only=True).numpy()
    vel = torch.load(data / "val/velocities.pth", weights_only=True).numpy()
    actions = torch.load(data / "val/rel_actions.pth", weights_only=True).numpy()
    audited = []
    for record in records:
        sample = record["sample"]
        ep, s = sample["episode"], sample["start"]
        truth = physical_truth(states[ep, s:s+16], vel[ep, s:s+16], actions[ep, s:s+15], protocol)
        audited.append({"sample": sample, "truth": truth})
    with out.open("w") as f:
        for record in audited:
            f.write(json.dumps(record) + "\n")
    result = {"windows": len(records), "physical_audited_windows": len(audited),
              "independent_source_episodes": len({r["sample"]["source_index"] for r in records})}
    branches = ("visual", "proprio", "object_diagnostic")
    for branch in branches:
        result[branch] = {}
        for h in (5, 15, "all"):
            pairs = [(r["pairs"][j], t["truth"][j]) for r, t in zip(records, audited)
                     for j, (s, e) in enumerate(PAIRS) if h == "all" or e-s == h]
            result[branch][str(h)] = metrics([p["labels"][branch] for p, _ in pairs],
                                             [t["labels"][branch] for _, t in pairs])
    ps, ts = [], []
    for branch in protocol["gate"]["branches"]:
        ps.extend(p["labels"][branch] for r in records for p in r["pairs"])
        ts.extend(t["labels"][branch] for r in audited for t in r["truth"])
    result["pooled_training_branches"] = metrics(ps, ts)
    strata = {}
    selectors = {
        "body_effect": lambda t: t["labels"]["proprio"] == 1,
        "object_effect": lambda t: t["labels"]["object_diagnostic"] == 1,
        "global_ineffective": lambda t: t["labels"]["visual"] == 0,
        "preparation_proxy": lambda t: t["preparation_proxy"],
        "return_proxy": lambda t: t["return_proxy"],
    }
    for name, check in selectors.items():
        subset = [(r, t) for r, t in zip(records, audited) if any(check(x) for x in t["truth"])]
        strata[name] = {"windows": len(subset), "source_episodes": len(subset),
            "window_ids": [r["sample"]["id"] for r, _ in subset]}
    write_json(out.with_name("audit_strata.json"), strata)
    result["stratum_windows"] = {k: v["windows"] for k, v in strata.items()}
    prep_pairs = [(p, t) for r, a in zip(records, audited) for p, t in zip(r["pairs"], a["truth"])
                  if t["preparation_proxy"]]
    result["preparation_pairs"] = len(prep_pairs)
    result["preparation_mislabeled_visual_static"] = sum(p["labels"]["visual"] == 0 for p, _ in prep_pairs)
    result["body_excursion_quantiles_world_px"] = {}
    for h in (5, 15):
        values = [t["body"]["excursion"] for a in audited for t in a["truth"] if t["pair"][1]-t["pair"][0] == h]
        result["body_excursion_quantiles_world_px"][str(h)] = dict(zip(
            ("min", "q01", "q10", "median", "q90", "max"),
            np.quantile(values, [0, .01, .1, .5, .9, 1]).tolist()))
        # Descriptive sensitivity only; does NOT alter labels/gate thresholds.
        result.setdefault("body_excursion_leq_sensitivity", {})[str(h)] = {
            str(th): float(np.mean(np.asarray(values) <= th)) for th in (1, 2, 5)}
    gate = protocol["gate"]
    checks = {}
    for name in gate["classes"]:
        m = result["pooled_training_branches"][name]
        checks[name] = {"precision_pass": m["precision"] is not None and m["precision"] >= gate["precision_each_class_min"],
                       "coverage_pass": m["coverage"] >= gate["coverage_each_class_min"]}
    result["gate"] = {"passed": all(all(x.values()) for x in checks.values()), "checks": checks,
                      "thresholds": gate, "absent_class_precision_is_undefined": True}
    result["caveats"] = [
        "Physical audit is not human semantic annotation; object rigid-body state is privileged audit-only information.",
        "Proprio audit shares the pusher sensor with the observer; its precision is a consistency check, not independent perception validation.",
        "The 4 pairs from one source window are correlated; no binomial significance claims.",
        "Preparation and return are kinematic proxies, not claims about intention or task usefulness.",
        "Full-image visual labels cannot use object-only static evidence when the pusher moves."
    ]
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="reports/effect_ab_20261103/protocol.json")
    ap.add_argument("--out", default="reports/effect_ab_20261103")
    args = ap.parse_args()
    started, cpu0 = time.monotonic(), time.process_time()
    protocol_path, out = ROOT / args.protocol, ROOT / args.out
    protocol = json.loads(protocol_path.read_text())
    data = ROOT / protocol["data"]
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists():
        raise RuntimeError("Output exists; refuse to overwrite registered samples/results")
    train_sources = {p["source_index"] for p in json.loads((data / "train/provenance.json").read_text())}
    dev_sources = {p["source_index"] for p in json.loads((data / "val/provenance.json").read_text())}
    if train_sources & dev_sources:
        raise RuntimeError("Source leakage across dataset splits")
    samples = {
        "train": select_windows(data, "train", protocol["train_windows"], protocol["seed"]),
        "val": select_windows(data, "val", protocol["dev_windows"], protocol["seed"] + 1),
    }
    write_json(out / "windows.json", samples)
    paths = [protocol_path, Path(__file__).resolve(), ROOT / "datasets/effect_observer.py", out / "windows.json"]
    for split in ("train", "val"):
        paths += [data / split / name for name in ("states.pth", "velocities.pth", "rel_actions.pth", "provenance.json", "seq_lengths.pkl", "shapes.pkl")]
        paths += [data / split / "obses" / f"episode_{r['episode']:03d}.mp4" for r in samples[split]]
    paths += sorted((ROOT / "data/targets/EVAL50").glob("*"))
    paths += [ROOT / "checkpoints/pushobj_bias_16k_e3_seed0/hydra.yaml",
              ROOT / "checkpoints/pushobj_bias_16k_e3_seed0/checkpoints/model_3.pth",
              ROOT / "checkpoints/pushobj_shape_shift/hydra.yaml",
              ROOT / "checkpoints/pushobj_shape_shift/checkpoints/model_latest.pth"]
    print("Hashing protocol, code, source metadata, sampled videos, targets and checkpoints", flush=True)
    manifest = {"git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_overlap": 0, "unique_sources": {"train": len(train_sources), "val": len(dev_sources)},
        "sample_counts_by_shape": {k: dict(Counter(s["shape"] for s in v)) for k, v in samples.items()},
        "files": [sha(p) for p in dict.fromkeys(paths) if p.is_file()],
        "required_file_count": sum(p.is_file() for p in dict.fromkeys(paths)),
        "missing": [str(p) for p in paths if not p.exists()],
        "software": {"python": sys.version, "torch": torch.__version__, "opencv": cv2.__version__}}
    if manifest["missing"]:
        raise FileNotFoundError(manifest["missing"])
    write_json(out / "manifest.json", manifest)
    train = run_observer(data, samples["train"], protocol, out / "train_observer.jsonl")
    hist = np.sum([r["background_diff_hist"] for r in train], axis=0)
    q = protocol["observer"]["background_noise_quantile"]
    quantile = int(np.searchsorted(np.cumsum(hist), hist.sum() * q))
    calibration = {"windows": len(train), "background_noise_quantile": q, "noise_value": quantile,
        "rgb_diff_floor": protocol["observer"]["rgb_diff_floor"],
        "fixed_floor_valid": quantile < protocol["observer"]["rgb_diff_floor"],
        "train_label_counts": {b: dict(Counter(p["labels"][b] for r in train for p in r["pairs"]))
                               for b in ("visual", "proprio", "object_diagnostic")},
        "note": "Only training background RGB noise validates the predeclared floor; no object states or dev outcomes used to tune thresholds."}
    write_json(out / "train_calibration.json", calibration)
    if not calibration["fixed_floor_valid"]:
        write_json(out / "status.json", {"status": "stopped_train_noise_calibration", "B_started": False})
        raise RuntimeError("Fixed RGB noise floor invalid; stop before dev audit")
    records = run_observer(data, samples["val"], protocol, out / "dev_observer.jsonl")
    observer_hash = sha(out / "dev_observer.jsonl")
    result = audit(data, records, protocol, out / "dev_physical_audit.jsonl")
    result["observer_cache_before_truth_audit"] = observer_hash
    result["train_calibration"] = calibration
    write_json(out / "signal_audit_summary.json", result)
    # getrusage includes worker initialization/teardown as well as decoding;
    # wall x (workers+parent) is an additional conservative budget upper bound.
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    elapsed = time.monotonic() - started
    cpu = time.process_time() - cpu0 + children.ru_utime + children.ru_stime
    ledger = {"stage": "A_signal_audit", "wall_seconds": elapsed,
        "cpu_seconds_including_child_processes": cpu, "cpu_core_hours": cpu / 3600,
        "cpu_reservation_upper_bound_core_hours": elapsed * (protocol["A_cpu_workers"] + 1) / 3600,
        "gpu_hours": 0.0, "gpu_devices": [], "completed_jobs": 2, "failed_jobs": 0,
        "completed_windows": len(train) + len(records), "AB_gpu_hours_remaining": 100.0}
    write_json(out / "resource_ledger.json", ledger)
    status = {"status": "A_signal_passed_pending_candidates" if result["gate"]["passed"] else "A_signal_gate_failed",
        "A_signal_completed": True, "A_candidate_diagnostic_completed": False, "B_started": False,
        "mpc_max_iter": protocol["B"]["mpc_max_iter"], "secondary_checkpoint_iter": 20,
        "next": "Run registered candidate diagnostics before B" if result["gate"]["passed"] else
                "Do not launch B under the registered stop rule; report signal support and proposed revision"}
    write_json(out / "status.json", status)
    print(json.dumps({"status": status, "gate": result["gate"], "ledger": ledger}, indent=2), flush=True)


if __name__ == "__main__":
    main()
