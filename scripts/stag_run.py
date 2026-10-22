"""Run stall-regulariser arms (planning/stagnation.py) on the touched episodes of a stream.

Only the episodes listed by scripts/stag_select.py are planned (planner.stag.only_samples, positions
kept so each episode keeps its seed); every other episode is identical to the AdaJEPA reference run.
All settings are scale-free (latent movement rate, goal-relative radius and strength), so the same
arms apply to other environments; gamma_hi was chosen on S7 (scripts/stag_select.py --calibrate).

    eval_outputs/stag_<stream>/<arm>/<chunk>/

    python scripts/stag_run.py --stream EVAL50 --arms SP_lo SP_mid SP_hi SZ_lo SZ_mid SZ_hi --gpus 4,5,6,7 --jobs 16
"""
import argparse
import json
import queue
import sys
import threading
from pathlib import Path

from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from escape_run import ROOT, finished, log, worker  # noqa: E402

ARMS = {  # penalty space, kappa (staying in a stalled region costs (1 + kappa) x its task objective)
    "S0": ("latent", 0.0),  # trigger and memory bookkeeping only: must equal AdaJEPA (smoke test)
    "SP_lo": ("proprio", 0.5),
    "SP_mid": ("proprio", 2.0),
    "SP_hi": ("proprio", 8.0),
    "SZ_lo": ("latent", 0.5),
    "SZ_mid": ("latent", 2.0),
    "SZ_hi": ("latent", 8.0),
}
# trigger on the latent movement rate (gamma_hi from S7: fires on 59/63 failures, 4/217 later successes);
# the repelled region around a stall reaches beta of the way to the goal (in squared latent distance)
STAG = {"start": 10, "window": 3, "gamma_lo": 0.05, "gamma_hi": 0.15, "beta": 0.25}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", default="EVAL50")
    ap.add_argument("--arms", nargs="+", default=["SP_lo", "SP_mid", "SP_hi", "SZ_lo", "SZ_mid", "SZ_hi"])
    ap.add_argument("--gpus", default="4,5,6,7")
    ap.add_argument("--jobs", type=int, default=16, help="concurrent runs (spread over --gpus)")
    ap.add_argument("--ckpt", default="checkpoints/pushobj_bias_16k_e3_seed0")
    ap.add_argument("--epoch", default="3")
    ap.add_argument("--root", default=None, help="default eval_outputs/stag_<stream>")
    ap.add_argument("--touched", default=None, help="default <root>/touched.json")
    ap.add_argument("--chunks", nargs="*", default=None, help="only these chunk names")
    ap.add_argument("--max-samples", type=int, default=0, help="cap touched samples per chunk (smoke tests)")
    ap.add_argument("--extra", nargs="*", default=[], help="extra Hydra overrides for every run")
    ap.add_argument("--min-free-gb", type=float, default=40.0)
    ap.add_argument("--stagger", type=float, default=10.0)
    a = ap.parse_args()

    root = ROOT / (a.root or f"eval_outputs/stag_{a.stream}")
    touched = json.loads(Path(a.touched or root / "touched.json").read_text())
    blocks = json.loads((ROOT / "data/targets" / a.stream / "manifest.json").read_text())["blocks"]
    trig = [f"+planner.stag.{k}={v}" for k, v in STAG.items()]
    obj = OmegaConf.load(ROOT / "conf/escape_plan_gd_pushobj.yaml").objective  # the penalty uses the objective's metric
    jobs = queue.Queue()
    for b in blocks:
        idx = touched["chunks"].get(b["chunk"], [])
        if not idx or (a.chunks and b["chunk"] not in a.chunks):
            continue
        if a.max_samples:
            idx = idx[: a.max_samples]
        for arm in a.arms:
            space, kappa = ARMS[arm]
            out = root / arm / b["chunk"]
            if finished(out):
                continue
            cmd = [sys.executable, ROOT / "plan.py", "--config-name", "escape_plan_gd_pushobj",
                   f"ckpt_base_path={(ROOT / a.ckpt).resolve()}", f"model_epoch={a.epoch}",
                   f"eval_data_path={ROOT / 'data/targets' / a.stream / (b['chunk'] + '.pkl')}",
                   f"n_evals={b['n']}", "+wandb_logging=false", f"hydra.run.dir={out}",
                   "planner._target_=planning.stagnation.StagnationMPCPlanner",
                   f"+planner.stag.space={space}", f"+planner.stag.kappa={kappa}",
                   f"+planner.stag.alpha={obj.alpha}", f"+planner.stag.base={obj.base}",
                   f"+planner.stag.only_samples=[{','.join(map(str, idx))}]"] + trig + a.extra
            jobs.put((out, cmd))
    gpus = a.gpus.split(",")
    log(f"{jobs.qsize()} runs: arms {a.arms}; {a.jobs} at a time on GPUs {gpus}")
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
