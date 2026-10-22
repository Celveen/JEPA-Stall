"""Run the multi-layout maze (diverse_maze, released AdaJEPA checkpoint) for the stall regulariser.

Arms:
    ref      AdaJEPA with per-replan logs and per-episode seeds (StagnationMPCPlanner with kappa 0)
    frozen   the batched frozen planner of the official config (MPCPlanner, no adaptation)
    SZ_lo / SZ_mid / SZ_hi   kappa 0.5 / 2 / 8, only on the episodes whose trigger fired in ref
Levels are BFS goal distances in maze blocks (medium 6-9, hard 10-14). Episodes are split into shards with
planner.stag.only_samples so they run in parallel; every shard still draws all n_evals goals, so each
episode keeps its goal and its seed (seed * i + 1).

    eval_outputs/stag_maze/<level>/<arm>/<shard>/

    python scripts/stag_maze_run.py --levels medium hard --arms ref frozen --gpus 4,5,6,7
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
KAPPA = {"ref": 0.0, "SZ_lo": 0.5, "SZ_mid": 2.0, "SZ_hi": 8.0}
MUJOCO_LIBS = [str(Path.home() / ".mujoco/mujoco210/bin"), "/usr/lib/nvidia"]


def touched(root, level):
    """Episodes whose trigger fired in the ref run of a level."""
    out = []
    for f in sorted((root / level / "ref").glob("*/stag_episodes.jsonl")):
        out += [d["sample_idx"] for d in map(json.loads, f.open()) if d["first_trigger"] is not None]
    return sorted(set(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", nargs="+", default=["medium", "hard"])
    ap.add_argument("--arms", nargs="+", default=["ref", "frozen"])
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--shards", type=int, default=3, help="parallel runs per level and arm")
    ap.add_argument("--max-iter", type=int, default=None, help="override planner.max_iter (smoke tests)")
    ap.add_argument("--gpus", default="4,5,6,7")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--ckpt", default="checkpoints/diversemaze_layout_shift")
    ap.add_argument("--data", default="data/diverse_maze")
    ap.add_argument("--root", default="eval_outputs/stag_maze")
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--min-free-gb", type=float, default=40.0)
    ap.add_argument("--stagger", type=float, default=20.0)
    a = ap.parse_args()

    os.environ["LD_LIBRARY_PATH"] = ":".join([os.environ.get("LD_LIBRARY_PATH", "")] + MUJOCO_LIBS).strip(":")
    root = ROOT / a.root
    jobs = queue.Queue()
    for level in a.levels:
        lo, hi = LEVELS[level]
        base = [sys.executable, ROOT / "plan.py", f"ckpt_base_path={(ROOT / a.ckpt).resolve()}",
                f"eval_data_path={(ROOT / a.data).resolve()}", f"n_evals={a.n}", f"seed={a.seed}",
                f"min_block_radius={lo}", f"max_block_radius={hi}", "+wandb_logging=false"]
        if a.max_iter:
            base.append(f"planner.max_iter={a.max_iter}")
        for arm in a.arms:
            if arm == "frozen":
                out = root / level / arm / "all"
                if not finished(out):
                    jobs.put((out, base[:2] + ["--config-name", "adajepa_plan_gd_diversemaze"] + base[2:] +
                              ["planner._target_=planning.mpc.MPCPlanner", "~planner.adapt", f"hydra.run.dir={out}"] + a.extra))
                continue
            idx = list(range(a.n)) if arm == "ref" else touched(root, level)
            if not idx:
                log(f"{level}/{arm}: no episodes to run")
                continue
            k = min(a.shards, len(idx))
            for s in range(k):
                part = idx[s::k]
                out = root / level / arm / f"shard{s}"
                if finished(out):
                    continue
                jobs.put((out, base[:2] + ["--config-name", "stag_plan_gd_diversemaze"] + base[2:] +
                          [f"planner.stag.kappa={KAPPA[arm]}", f"planner.stag.only_samples=[{','.join(map(str, part))}]",
                           f"hydra.run.dir={out}"] + a.extra))
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
