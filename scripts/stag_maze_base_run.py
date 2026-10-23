"""Stall regulariser on the multi-layout maze (released diversemaze_layout_shift checkpoint) on top of a base planner.

    --phase ref   the base planner on all n episodes of a level, with both triggers' quantities logged from replan
                  LOG_START on (kappa 0, so planning is the base planner's own)
    --phase arms  each arm reruns only the episodes whose reference trigger fires inside the arm's own window
                  (start <= replan < max_iter, before success); every other episode is identical to ref

Bases: frozen (per-sample planning without the test-time update: probe.mode=none) or adajepa. Levels are BFS goal
distances in maze blocks. Goals come from one RNG seeded by `seed` and episode seeds are seed * i + 1, both by
position, so the first 50 of 100 episodes are the 50-episode ones and shards (stag.only_samples) keep every
episode's goal and seed. Runs with a larger max_iter contain the smaller budgets exactly (truncate when analysing).

    eval_outputs/stag_maze100/<level>/<base>/<arm>/<shard>/

    python scripts/stag_maze_base_run.py --phase ref --levels easy medium --bases frozen adajepa
    python scripts/stag_maze_base_run.py --phase arms --levels easy medium --bases frozen adajepa --arms SZ_mid T0SP
"""
import argparse
import json
import os
import queue
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from escape_run import ROOT, finished, log, worker  # noqa: E402

LEVELS = {"easy": (3, 5), "medium": (6, 9), "hard": (10, 14)}
BASES = {"frozen": ["+planner.probe.mode=none"], "adajepa": []}
ARMS = {  # the PushObj settings unchanged (scale-free); start = a third of the paper's 20-replan budget
    "SZ_mid": {"kappa": 2.0, "start": 7},
    "T0SP": {"kappa": 2.0, "start": 7, "prop_boost": 10.0},
}
STAG = {"space": "latent", "window": 3, "gamma_lo": 0.05, "gamma_hi": 0.15, "beta": 0.25, "alpha": 1, "base": 2}
LOG_START = 4
MUJOCO_LIBS = [str(Path.home() / ".mujoco/mujoco210/bin"), "/usr/lib/nvidia"]


def fired(ref, start, max_iter, gamma_hi):
    """Episodes whose reference movement trigger fires at a replan in [start, max_iter) before success."""
    hit = set()
    for f in ref.glob("*/per_iter_logs.json"):
        for line in f.open():
            if line.strip():
                r = json.loads(line)
                if start <= r["step"] < max_iter and r.get("stag_r") is not None and r["stag_r"] < gamma_hi:
                    hit.add(r["sample_idx"])
    return sorted(hit)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("ref", "arms"), required=True)
    ap.add_argument("--levels", nargs="+", default=["easy", "medium"])
    ap.add_argument("--bases", nargs="+", default=["frozen", "adajepa"])
    ap.add_argument("--arms", nargs="+", default=["SZ_mid"])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--max-iter", type=int, default=40)
    ap.add_argument("--per-shard", type=int, default=10, help="episodes per run (ref); arms use half")
    ap.add_argument("--gpus", default="0,1,4,5,6,7")
    ap.add_argument("--jobs", type=int, default=18)
    ap.add_argument("--ckpt", default="checkpoints/diversemaze_layout_shift")
    ap.add_argument("--data", default="data/diverse_maze")
    ap.add_argument("--root", default="eval_outputs/stag_maze100")
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--min-free-gb", type=float, default=40.0)
    ap.add_argument("--stagger", type=float, default=10.0)
    a = ap.parse_args()

    os.environ["LD_LIBRARY_PATH"] = ":".join([os.environ.get("LD_LIBRARY_PATH", "")] + MUJOCO_LIBS).strip(":")
    root = ROOT / a.root
    jobs = queue.Queue()
    for level in a.levels:
        lo, hi = LEVELS[level]
        for base in a.bases:
            common = [sys.executable, ROOT / "plan.py", "--config-name", "stag_plan_gd_diversemaze",
                      f"ckpt_base_path={(ROOT / a.ckpt).resolve()}", f"eval_data_path={(ROOT / a.data).resolve()}",
                      f"n_evals={a.n}", f"seed={a.seed}", f"min_block_radius={lo}", f"max_block_radius={hi}",
                      "+wandb_logging=false", f"planner.max_iter={a.max_iter}"] + \
                [f"++planner.stag.{k}={v}" for k, v in STAG.items()] + BASES[base]
            ref = root / level / base / "ref"
            if a.phase == "ref":
                runs = [("ref", {"kappa": 0.0, "start": LOG_START}, list(range(a.n)), a.per_shard)]
            else:
                runs = []
                for arm in a.arms:
                    cfg = ARMS[arm]
                    idx = fired(ref, cfg["start"], a.max_iter, STAG["gamma_hi"])
                    log(f"{level}/{base}/{arm}: {len(idx)} episodes")
                    runs.append((arm, cfg, idx, max(1, a.per_shard // 2)))
            for arm, cfg, idx, per in runs:
                for s in range(0, len(idx), per):
                    part = idx[s:s + per]
                    out = root / level / base / arm / f"shard{s // per:02d}"
                    if finished(out):
                        continue
                    jobs.put((out, common + [f"++planner.stag.{k}={v}" for k, v in cfg.items()] +
                              [f"++planner.stag.only_samples=[{','.join(map(str, part))}]", f"hydra.run.dir={out}"] + a.extra))
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
