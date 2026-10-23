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
ARMS = {  # regulariser settings per arm (start = first replan, 0-based, the trigger may fire)
    "SZ_mid": {"kappa": 2.0, "start": 10},
    "SZ_early": {"kappa": 2.0, "start": 5},
    # "agent arrived, scene not" trigger (q 0.2, from replan 7; chosen on S7, scripts/stag_signal_eval.py)
    "D2S": {"kappa": 2.0, "start": 6, "trigger": "arrived", "q": 0.2, "theta": 0.0},
    # the same plus the proprio boost (x10 on the trigger replan and the 2 after it)
    "D2SP": {"kappa": 2.0, "start": 6, "trigger": "arrived", "q": 0.2, "theta": 0.0, "prop_boost": 10.0},
    # movement trigger plus the proprio boost, to separate the boost from the earlier trigger
    "T0SP": {"kappa": 2.0, "start": 10, "prop_boost": 10.0},
    # the proprio boost alone (movement trigger, no penalty): ablation of T0SP
    "T0P": {"kappa": 0.0, "start": 10, "prop_boost": 10.0},
    # two-stage lookahead (planning/lookahead.py): K macro moves + a plan vs the usual plan
    "LA_start": {"kappa": 0.0, "start": 10, "la": {"when": "start"}},  # first replan only, no regulariser
    "D2SL": {"kappa": 2.0, "start": 6, "trigger": "arrived", "q": 0.2, "theta": 0.0, "la": {"when": "trigger"}},
    "T0SL": {"kappa": 2.0, "start": 10, "la": {"when": "trigger"}},
    # the same with a 10-step follow-up (2x the horizon), so a detour can pay off inside the comparison
    "LA_startF10": {"kappa": 0.0, "start": 10, "la": {"when": "start", "follow": 10}},
    "D2SLF10": {"kappa": 2.0, "start": 6, "trigger": "arrived", "q": 0.2, "theta": 0.0,
                "la": {"when": "trigger", "follow": 10}},
    "T0SLF10": {"kappa": 2.0, "start": 10, "la": {"when": "trigger", "follow": 10}},
    # ... executing the best candidate even when it is the no-macro one (a temporary 12-step horizon)
    "D2SLF10b": {"kappa": 2.0, "start": 6, "trigger": "arrived", "q": 0.2, "theta": 0.0,
                 "la": {"when": "trigger", "follow": 10, "exec": "best"}},
    "T0SLF10b": {"kappa": 2.0, "start": 10, "la": {"when": "trigger", "follow": 10, "exec": "best"}},
    # step 2 on top of the step-1 arms: the lookahead (10-step follow-up, best candidate executed) at their triggers
    "T0SPL": {"kappa": 2.0, "start": 10, "prop_boost": 10.0, "la": {"when": "trigger", "follow": 10, "exec": "best"}},
    "D2SPL": {"kappa": 2.0, "start": 6, "trigger": "arrived", "q": 0.2, "theta": 0.0, "prop_boost": 10.0,
              "la": {"when": "trigger", "follow": 10, "exec": "best"}},
    # a static objective with a 10x proprio weight from the first replan (is T0SP more than a retuned alpha?)
    "ALPHA10": {"kappa": 0.0, "start": 10, "over": ["objective.alpha=10"], "all": True},
}
LA = {"dirs": 8, "steps": 2, "scale": 1.0, "gap": 4, "max_n": 3}  # lookahead defaults (model steps, normalized)
STAG = {"space": "latent", "window": 3, "gamma_lo": 0.05, "gamma_hi": 0.15, "beta": 0.25, "alpha": 1, "base": 2}
LOG_START = 4  # the reference logs the trigger from here (it needs window + 1 observations)


def fired_arrived(root, start, max_iter, q, theta, margin=0.05):
    """{chunk: [sample_idx]} whose arrived trigger fires on the reference's latent trace (root/trace.jsonl,
    scripts/stag_latent_trace.py, base encoder) within [start, max_iter) before success; thresholds are
    loosened by `margin` so the planner's own measurement (adapted encoder, rounding) stays inside the set."""
    out = {}
    for line in (root / "trace.jsonl").open():
        e = json.loads(line)
        d = {0: e["dg0"]}
        dp, ok = {}, {}
        for r in e["steps"]:
            d[r["s"]], dp[r["s"]], ok[r["s"]] = r["dg"], r["dg_p"], r["success"]
        v = {s: max(d[s] ** 2 - dp.get(s, 0.0) ** 2, 0.0) ** 0.5 for s in d}
        for s in sorted(dp):
            if ok[s]:
                break
            if start <= s < max_iter and s >= STAG["window"]:
                share = dp[s] ** 2 / max(d[s] ** 2, 1e-12)
                vprog = (v[s - STAG["window"]] - v[s]) / max(v[s - STAG["window"]], 1e-12)
                if share < q + margin and vprog < theta + margin:
                    out.setdefault(e["chunk"], []).append(e["sample_idx"])
                    break
    return {k: sorted(v) for k, v in out.items()}


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
        runs = [("ref", {"kappa": 0.0, "start": LOG_START}, plan)]
    else:
        runs = []
        for arm in a.arms:
            cfg = ARMS[arm]
            if cfg.get("all") or cfg.get("la", {}).get("when") in ("start", "both"):  # changes replan 0: every episode
                plan = {b["chunk"]: list(range(b["n"]))[: a.max_samples or None] for b in blocks}
            elif cfg.get("trigger", "movement") == "arrived":
                plan = fired_arrived(root, cfg["start"], a.max_iter, cfg["q"], cfg["theta"])
            else:
                plan = fired(root, cfg["start"], a.max_iter, STAG["gamma_hi"])
            log(f"{arm}: {sum(map(len, plan.values()))} episodes in {len(plan)} chunks")
            runs.append((arm, cfg, plan))
    for b in blocks:
        for arm, cfg, plan in runs:
            idx = plan.get(b["chunk"], [])
            out = root / arm / b["chunk"]
            if not idx or finished(out):
                continue
            stag = {k: v for k, v in cfg.items() if k not in ("la", "over", "all")}
            la = {**LA, **cfg["la"]} if "la" in cfg else None
            target = "planning.lookahead.LookaheadMPCPlanner" if la else "planning.stagnation.StagnationMPCPlanner"
            cmd = [sys.executable, ROOT / "plan.py", "--config-name", "escape_plan_gd_pushobj",
                   f"ckpt_base_path={(ROOT / ckpt).resolve()}", f"model_epoch={epoch}",
                   f"eval_data_path={ROOT / 'data/targets' / a.stream / (b['chunk'] + '.pkl')}",
                   f"n_evals={b['n']}", "+wandb_logging=false", f"hydra.run.dir={out}",
                   f"planner._target_={target}",
                   f"+planner.stag.only_samples=[{','.join(map(str, idx))}]"] + \
                [f"+planner.stag.{k}={v}" for k, v in stag.items()] + \
                [f"+planner.lookahead.{k}={v}" for k, v in (la or {}).items()] + cfg.get("over", []) + common + a.extra
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
