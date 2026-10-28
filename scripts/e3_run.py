"""E3 / E2 runs: staged fixing on the multi-layout maze (planning/switch.py), AdaJEPA base, 4 env steps per action.

Each arm reruns only the episodes whose stall trigger fired in the reference run (stag_w > 0 at any replan); the
others never activate the ruler and are identical to the reference. Episodes keep their position in the run
(stag.only_samples), hence their goal and seed, so arms pair exactly with the reference episode by episode.

    R-trig    T0SP (stall penalty kappa 2 + proprio boost x10 for 2 replans) from the first trigger
    R-switch  T0SP from the candidate switch: trigger + newest-step loss flat (W 3, theta 0.05, K 1; E1)
    R-delay   T0SP from the first trigger + 10 replans (time-only control)
    SZ-<t> / P-<t>   the stall penalty alone / the proprio boost alone, from timing t (trig, switch)
    F-trig / F-switch   no ruler; the world model stops updating at the first trigger / at the switch (E2)
    T0SP-stag  check: T0SP through StagnationMPCPlanner, which R-trig must reproduce exactly

    eval_outputs/e3/<level>_s<seed>/<arm>/shardNN/

    python scripts/e3_run.py --arms R-trig R-switch
    python scripts/e3_run.py --arms R-trig --smoke 2 --root eval_outputs/e3_smoke
"""
import argparse
import json
import os
import queue
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from attr_run import active, run_worker  # noqa: E402
from escape_run import ROOT, finished, log  # noqa: E402
from stag_maze_base_run import LEVELS, MUJOCO_LIBS, STAG  # noqa: E402

T0SP = {"kappa": 2.0, "prop_boost": 10.0, "boost_replans": 2}
SWITCH = {"mode": "loss_new", "W": 3, "theta": 0.05, "K": 1}
ARMS = {
    "R-trig": (T0SP, {"mode": "trigger"}),
    "R-switch": (T0SP, SWITCH),
    "R-delay": (T0SP, {"mode": "delay", "D": 10}),
    "SZ-trig": ({"kappa": 2.0}, {"mode": "trigger"}),
    "SZ-switch": ({"kappa": 2.0}, SWITCH),
    "P-trig": ({"kappa": 0.0, "prop_boost": 10.0, "boost_replans": 2}, {"mode": "trigger"}),
    "P-switch": ({"kappa": 0.0, "prop_boost": 10.0, "boost_replans": 2}, SWITCH),
    "F-trig": ({"kappa": 0.0}, {"mode": "trigger", "freeze": True}),
    "F-switch": ({"kappa": 0.0}, {**SWITCH, "freeze": True}),
    "T0SP-stag": (T0SP, None),  # check arm: the same T0SP through StagnationMPCPlanner (must equal R-trig)
}
START, MAX_ITER, N = 4, 50, 100  # trigger from replan 4 (as the reference runs log it), budget 50


def triggered(ref):
    """Episodes of a reference run whose stall trigger fired at some replan."""
    hit = set()
    for f in ref.glob("*/per_iter_logs.json"):
        for line in f.open():
            if line.strip():
                r = json.loads(line)
                if (r.get("stag_w") or 0) > 0:
                    hit.add(r["sample_idx"])
    return sorted(hit)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True, choices=sorted(ARMS))
    ap.add_argument("--level", default="medium", choices=sorted(LEVELS))
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--ref", default=None, help="default eval_outputs/stag_maze100_ar4/<level>/adajepa/ref (seed 100)")
    ap.add_argument("--smoke", type=int, default=0, help="only the first k triggered episodes")
    ap.add_argument("--episodes", type=int, nargs="*", default=None, help="only these episodes (smoke tests)")
    ap.add_argument("--per-shard", type=int, default=9)
    ap.add_argument("--gpus", default="0,1,5")
    ap.add_argument("--jobs", type=int, default=6, help="at most 2 maze runs per GPU (offscreen framebuffers)")
    ap.add_argument("--root", default="eval_outputs/e3")
    ap.add_argument("--ckpt", default="checkpoints/diversemaze_layout_shift")
    ap.add_argument("--data", default="data/diverse_maze")
    ap.add_argument("--min-free-gb", type=float, default=40.0)
    ap.add_argument("--stagger", type=float, default=10.0)
    a = ap.parse_args()

    os.environ["LD_LIBRARY_PATH"] = ":".join([os.environ.get("LD_LIBRARY_PATH", "")] + MUJOCO_LIBS).strip(":")
    ref = ROOT / (a.ref or f"eval_outputs/stag_maze100_ar4/{a.level}/adajepa/ref")
    idx = triggered(ref)
    if a.episodes:
        idx = [i for i in idx if i in set(a.episodes)]
    if a.smoke:
        idx = idx[: a.smoke]
    log(f"{len(idx)} triggered episodes in {ref.relative_to(ROOT)}")
    lo, hi = LEVELS[a.level]
    common = [sys.executable, ROOT / "plan.py", "--config-name", "stag_plan_gd_diversemaze",
              f"ckpt_base_path={(ROOT / a.ckpt).resolve()}", f"eval_data_path={(ROOT / a.data).resolve()}",
              f"n_evals={N}", f"seed={a.seed}", f"min_block_radius={lo}", f"max_block_radius={hi}",
              "+wandb_logging=false", f"planner.max_iter={MAX_ITER}", "++env_kwargs_override.action_repeat=4"] + \
        [f"++planner.stag.{k}={v}" for k, v in STAG.items()] + [f"++planner.stag.start={START}"]
    jobs = queue.Queue()
    root = ROOT / a.root / f"{a.level}_s{a.seed}"
    for arm in a.arms:
        stag, switch = ARMS[arm]
        per = len(idx) if a.smoke else a.per_shard
        for s in range(0, len(idx), per):
            part = idx[s:s + per]
            out = root / arm / f"shard{s // per:02d}"
            if finished(out) or active(out):
                continue
            target = "planning.stagnation.StagnationMPCPlanner" if switch is None else "planning.switch.SwitchMPCPlanner"
            jobs.put((out, common + [f"planner._target_={target}"] + [f"++planner.stag.{k}={v}" for k, v in stag.items()] +
                      [f"++planner.switch.{k}={v}" for k, v in (switch or {}).items()] +
                      [f"++planner.stag.only_samples=[{','.join(map(str, part))}]", f"hydra.run.dir={out}"]))
    gpus = a.gpus.split(",")
    log(f"{jobs.qsize()} runs ({' '.join(a.arms)}); {a.jobs} at a time on GPUs {gpus}")
    results = []
    threads = [threading.Thread(target=run_worker, args=(gpus[i % len(gpus)], jobs, a, results)) for i in range(a.jobs)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    log(f"finished: {sum(results)} ok, {len(results) - sum(results)} failed")
    sys.exit(0 if results and all(results) else 1)


if __name__ == "__main__":
    main()
