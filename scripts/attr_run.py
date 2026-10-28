"""D0 runs: the reference runs again with model-or-ruler attribution logs (planning/attribution.py).

Every run repeats an existing reference run with the same settings, so its actions must match the reference bit for
bit (scripts/attr_d0.py --check verifies that); only the logs are new.

    A1  maze easy,   original timescale,  frozen   dev:  model-error condition    stag_maze100/easy/frozen/ref
    A2  maze easy,   4 env steps/action,  AdaJEPA  dev:  accurate-model condition stag_maze100_ar4/easy/adajepa/ref
    A3  maze medium, original timescale,  frozen   test: model-error condition    stag_maze100/medium/frozen/ref
    A4  maze medium, 4 env steps/action,  AdaJEPA  test: ruler condition          stag_maze100_ar4/medium/adajepa/ref
    A5  maze medium, 4 env steps/action,  frozen   optional: mixed condition       stag_maze100_ar4/medium/frozen/ref
    A6  maze easy,   original timescale,  AdaJEPA  optional: verdict shift          stag_maze100/easy/adajepa/ref
    B1  PushObj EVAL50, AdaJEPA (bias16k), the 97 touched episodes, 30 replans    escape_EVAL50/none

    eval_outputs/attr_d0/<run>/<shard or chunk>/

    python scripts/attr_run.py --runs A1 A2
    python scripts/attr_run.py --runs A1 B1 --smoke 2 --root eval_outputs/attr_d0_smoke
"""
import argparse
import json
import os
import queue
import sys
import threading
from pathlib import Path

from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from escape_run import ROOT, finished, log, worker  # noqa: E402
from stag_maze_base_run import BASES, LEVELS, MUJOCO_LIBS, STAG as MAZE_STAG  # noqa: E402
from stag_run import STAG as PUSH_STAG  # noqa: E402

AR4 = ["++env_kwargs_override.action_repeat=4"]
MAZE = {  # level, base, extra overrides, reference run
    "A1": ("easy", "frozen", [], "eval_outputs/stag_maze100/easy/frozen/ref"),
    "A2": ("easy", "adajepa", AR4, "eval_outputs/stag_maze100_ar4/easy/adajepa/ref"),
    "A3": ("medium", "frozen", [], "eval_outputs/stag_maze100/medium/frozen/ref"),
    "A4": ("medium", "adajepa", AR4, "eval_outputs/stag_maze100_ar4/medium/adajepa/ref"),
    "A5": ("medium", "frozen", AR4, "eval_outputs/stag_maze100_ar4/medium/frozen/ref"),
    "A6": ("easy", "adajepa", [], "eval_outputs/stag_maze100/easy/adajepa/ref"),
}
PUSH = {"B1": ("EVAL50", "eval_outputs/stag_EVAL50/touched.json", "eval_outputs/escape_EVAL50/none")}
REFS = {**{k: v[3] for k, v in MAZE.items()}, **{k: v[2] for k, v in PUSH.items()}}
MAZE_N, MAZE_SEED, MAZE_ITER, MAZE_LOG_START = 100, 100, 50, 4  # as in the reference runs
ATTR = ["planner._target_=planning.attribution.AttributionMPCPlanner"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, choices=sorted(REFS))
    ap.add_argument("--restarts", type=int, default=8)
    ap.add_argument("--restart-every", type=int, default=3)
    ap.add_argument("--smoke", type=int, default=0, help="only the first k episodes (PushObj: touched ones)")
    ap.add_argument("--per-shard", type=int, default=10, help="maze episodes per run")
    ap.add_argument("--gpus", default="0,1,4,5,6,7")
    ap.add_argument("--jobs", type=int, default=18)
    ap.add_argument("--root", default="eval_outputs/attr_d0")
    ap.add_argument("--maze-ckpt", default="checkpoints/diversemaze_layout_shift")
    ap.add_argument("--maze-data", default="data/diverse_maze")
    ap.add_argument("--push-ckpt", default="checkpoints/pushobj_bias_16k_e3_seed0")
    ap.add_argument("--push-epoch", default="3")
    ap.add_argument("--min-free-gb", type=float, default=40.0)
    ap.add_argument("--stagger", type=float, default=10.0)
    a = ap.parse_args()

    os.environ["LD_LIBRARY_PATH"] = ":".join([os.environ.get("LD_LIBRARY_PATH", "")] + MUJOCO_LIBS).strip(":")
    root = ROOT / a.root
    attr = ATTR + [f"++planner.attr.restarts={a.restarts}", f"++planner.attr.restart_every={a.restart_every}"]
    jobs = queue.Queue()
    for run in a.runs:
        if run in MAZE:
            level, base, extra, _ = MAZE[run]
            lo, hi = LEVELS[level]
            common = [sys.executable, ROOT / "plan.py", "--config-name", "stag_plan_gd_diversemaze",
                      f"ckpt_base_path={(ROOT / a.maze_ckpt).resolve()}",
                      f"eval_data_path={(ROOT / a.maze_data).resolve()}",
                      f"n_evals={MAZE_N}", f"seed={MAZE_SEED}", f"min_block_radius={lo}", f"max_block_radius={hi}",
                      "+wandb_logging=false", f"planner.max_iter={MAZE_ITER}"] + \
                [f"++planner.stag.{k}={v}" for k, v in MAZE_STAG.items()] + BASES[base] + \
                ["++planner.stag.kappa=0.0", f"++planner.stag.start={MAZE_LOG_START}"] + extra + attr
            idx = list(range(a.smoke or MAZE_N))
            per = len(idx) if a.smoke else a.per_shard
            for s in range(0, len(idx), per):
                part = idx[s:s + per]
                out = root / run / f"shard{s // per:02d}"
                if not finished(out):
                    jobs.put((out, common + [f"++planner.stag.only_samples=[{','.join(map(str, part))}]",
                                             f"hydra.run.dir={out}"]))
        else:
            stream, touched_path, _ = PUSH[run]
            touched = json.loads((ROOT / touched_path).read_text())["chunks"]
            blocks = json.loads((ROOT / "data/targets" / stream / "manifest.json").read_text())["blocks"]
            obj = OmegaConf.load(ROOT / "conf/escape_plan_gd_pushobj.yaml").objective
            left = a.smoke or None
            for b in blocks:
                idx = touched.get(b["chunk"], [])
                if left is not None:
                    idx = idx[:left]
                    left -= len(idx)
                if not idx:
                    continue
                out = root / run / b["chunk"]
                if not finished(out):
                    jobs.put((out, [sys.executable, ROOT / "plan.py", "--config-name", "escape_plan_gd_pushobj",
                                    f"ckpt_base_path={(ROOT / a.push_ckpt).resolve()}", f"model_epoch={a.push_epoch}",
                                    f"eval_data_path={ROOT / 'data/targets' / stream / (b['chunk'] + '.pkl')}",
                                    f"n_evals={b['n']}", "+wandb_logging=false", f"hydra.run.dir={out}",
                                    "+planner.stag.space=latent", "+planner.stag.kappa=0.0",
                                    f"+planner.stag.alpha={obj.alpha}", f"+planner.stag.base={obj.base}",
                                    f"+planner.stag.only_samples=[{','.join(map(str, idx))}]"] +
                               [f"+planner.stag.{k}={v}" for k, v in PUSH_STAG.items()] + attr))
                if left == 0:
                    break
    gpus = a.gpus.split(",")
    log(f"{jobs.qsize()} runs ({' '.join(a.runs)}); {a.jobs} at a time on GPUs {gpus}")
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
