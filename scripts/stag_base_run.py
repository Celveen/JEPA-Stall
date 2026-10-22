"""Run the stall regulariser (planning/stagnation.py) on top of a base planner, with an online reference run.

    --phase ref   the base planner on every episode of the target set, with the regulariser's trigger logged
                  from replan LOG_START on (kappa 0, so planning is the base planner's own)
    --phase arms  each arm reruns only the episodes whose logged trigger fires inside the arm's own window
                  (start <= replan < max_iter, before success); every other episode is identical to ref

Bases: frozen (per-sample planning, no update: probe.mode=none) or adajepa (the within-episode update).
Models: bias16k (our PushObj world model) or official (the released shape-shift checkpoint).

    eval_outputs/stag_<base>_<model>/<arm>/<chunk>/

    python scripts/stag_base_run.py --base frozen --model bias16k --phase ref --gpus 4,5,6,7 --jobs 20
    python scripts/stag_base_run.py --base frozen --model bias16k --phase arms --arms SZ_mid SZ_early
"""
import argparse
import json
import queue
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from escape_run import ROOT, finished, log, worker  # noqa: E402

MODELS = {"bias16k": ("checkpoints/pushobj_bias_16k_e3_seed0", "3"), "official": ("checkpoints/pushobj_shape_shift", "latest")}
BASES = {"frozen": ["+planner.probe.mode=none"], "adajepa": []}
ARMS = {"SZ_mid": (2.0, 10), "SZ_early": (2.0, 5)}  # kappa, first replan (0-based) the penalty may start
STAG = {"space": "latent", "window": 3, "gamma_lo": 0.05, "gamma_hi": 0.15, "beta": 0.25, "alpha": 1, "base": 2}
LOG_START = 4  # the reference logs the trigger from here (it needs window + 1 observations)


def fired(root, start, max_iter, gamma_hi):
    """{chunk: [sample_idx]} whose reference trigger fires at a replan in [start, max_iter) before success."""
    out = {}
    for ch in sorted(p for p in (root / "ref").iterdir() if (p / "per_iter_logs.json").exists()):
        hit = set()
        for line in (ch / "per_iter_logs.json").open():
            if line.strip():
                r = json.loads(line)
                if start <= r["step"] < max_iter and r.get("stag_r") is not None and r["stag_r"] < gamma_hi:
                    hit.add(r["sample_idx"])
        if hit:
            out[ch.name] = sorted(hit)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", choices=list(BASES), required=True)
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--phase", choices=("ref", "arms"), required=True)
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--stream", default="EVAL50")
    ap.add_argument("--max-iter", type=int, default=30)
    ap.add_argument("--chunks", nargs="*", default=None)
    ap.add_argument("--max-samples", type=int, default=0, help="first N samples per chunk (smoke tests)")
    ap.add_argument("--root", default=None, help="default eval_outputs/stag_<base>_<model>")
    ap.add_argument("--gpus", default="4,5,6,7")
    ap.add_argument("--jobs", type=int, default=20)
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--min-free-gb", type=float, default=40.0)
    ap.add_argument("--stagger", type=float, default=5.0)
    a = ap.parse_args()

    root = ROOT / (a.root or f"eval_outputs/stag_{a.base}_{a.model}")
    ckpt, epoch = MODELS[a.model]
    blocks = json.loads((ROOT / "data/targets" / a.stream / "manifest.json").read_text())["blocks"]
    if a.chunks:
        blocks = [b for b in blocks if b["chunk"] in a.chunks]
    common = [f"+planner.stag.{k}={v}" for k, v in STAG.items()] + [f"planner.max_iter={a.max_iter}"] + BASES[a.base]
    jobs = queue.Queue()
    if a.phase == "ref":
        plan = {b["chunk"]: list(range(b["n"]))[: a.max_samples or None] for b in blocks}
        runs = [("ref", 0.0, LOG_START, plan)]
    else:
        runs = []
        for arm in a.arms:
            kappa, start = ARMS[arm]
            plan = fired(root, start, a.max_iter, STAG["gamma_hi"])
            log(f"{arm}: {sum(map(len, plan.values()))} episodes in {len(plan)} chunks")
            runs.append((arm, kappa, start, plan))
    for b in blocks:
        for arm, kappa, start, plan in runs:
            idx = plan.get(b["chunk"], [])
            out = root / arm / b["chunk"]
            if not idx or finished(out):
                continue
            cmd = [sys.executable, ROOT / "plan.py", "--config-name", "escape_plan_gd_pushobj",
                   f"ckpt_base_path={(ROOT / ckpt).resolve()}", f"model_epoch={epoch}",
                   f"eval_data_path={ROOT / 'data/targets' / a.stream / (b['chunk'] + '.pkl')}",
                   f"n_evals={b['n']}", "+wandb_logging=false", f"hydra.run.dir={out}",
                   "planner._target_=planning.stagnation.StagnationMPCPlanner",
                   f"+planner.stag.kappa={kappa}", f"+planner.stag.start={start}",
                   f"+planner.stag.only_samples=[{','.join(map(str, idx))}]"] + common + a.extra
            jobs.put((out, cmd))
    gpus = a.gpus.split(",")
    log(f"{jobs.qsize()} runs; {a.jobs} at a time on GPUs {gpus}")
    results = []
    threads = [threading.Thread(target=worker, args=(gpus[i % len(gpus)], jobs, a, results)) for i in range(a.jobs)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    log(f"finished: {sum(results)} ok, {len(results) - sum(results)} failed")
    sys.exit(0 if results and all(results) else 1)


if __name__ == "__main__":
    main()
