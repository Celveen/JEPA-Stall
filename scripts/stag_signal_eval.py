"""Offline comparison of stall-detection signals on latent traces (scripts/stag_latent_trace.py).

All signals read only the planner's latents (objective metric d, its proprio part d_p) after replan s:

    T0  current trigger: latent movement over W replans < gamma x distance to the goal, from replan 10
    D1  progress rate: the latent distance to the goal fell by less than theta (relative) over W replans
    D2  agent arrived, scene not: the proprio share of the remaining distance (d_p^2 / d^2) is below q and the
        visual part fell by less than theta (relative) over W replans

For each signal and threshold: how many failures it catches (recall), how many episodes that later
succeed it touches (false fires), and how much earlier than T0 it fires on the failures both catch.
Thresholds are chosen on the calibration trace and then applied unchanged to the evaluation trace.

    python scripts/stag_signal_eval.py --calib eval_outputs/stag_S7/trace.jsonl --eval eval_outputs/stag_EVAL50/trace.jsonl
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
W = 3


def series(ep):
    """Per replan s: d (to goal), d_p (proprio part), v (visual part), r (T0 ratio); index 0 = initial state."""
    S = max(r["s"] for r in ep["steps"])
    d, dp, r0 = np.full(S + 1, np.nan), np.full(S + 1, np.nan), np.full(S + 1, np.nan)
    d[0] = ep["dg0"]
    for r in ep["steps"]:
        d[r["s"]], dp[r["s"]] = r["dg"], r["dg_p"]
        if "dW" in r:
            r0[r["s"]] = r["dW"] / max(r["dg"], 1e-12)
    v = np.sqrt(np.maximum(d ** 2 - np.nan_to_num(dp) ** 2, 0))
    return d, dp, v, r0


def first_fire(ep, kind, p, start, max_iter):
    d, dp, v, r0 = series(ep)
    ok = {r["s"]: r["success"] for r in ep["steps"]}
    for s in range(max(start, W), min(max_iter, len(d))):
        if ok.get(s):
            return None  # the episode is over before this replan
        if kind == "T0":
            hit = r0[s] < p["gamma"]
        else:
            prog = (d[s - W] - d[s]) / max(d[s - W], 1e-12)
            if kind == "D1":
                hit = prog < p["theta"]
            else:
                vprog = (v[s - W] - v[s]) / max(v[s - W], 1e-12)
                hit = (dp[s] ** 2 / max(d[s] ** 2, 1e-12) < p["q"]) and vprog < p["theta"]
        if hit:
            return s
    return None


def evaluate(eps, kind, p, start, max_iter, t0):
    fails = [e for e in eps if e["success_step"] is None]
    succ = [e for e in eps if e["success_step"] is not None]
    ff = {id(e): first_fire(e, kind, p, start, max_iter) for e in eps}
    rec = sum(ff[id(e)] is not None for e in fails)
    fp = sum(ff[id(e)] is not None for e in succ)
    lead = [t0[id(e)] - ff[id(e)] for e in fails if ff[id(e)] is not None and t0[id(e)] is not None]
    first = [ff[id(e)] for e in fails if ff[id(e)] is not None]
    only = sum(ff[id(e)] is not None and t0[id(e)] is None for e in fails)
    return {"recall": rec, "n_fail": len(fails), "false": fp, "n_succ": len(succ),
            "lead_median": float(np.median(lead)) if lead else float("nan"),
            "lead_ge4": int(sum(x >= 4 for x in lead)), "first_median": float(np.median(first)) if first else float("nan"),
            "only_new": only}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default="eval_outputs/stag_S7/trace.jsonl")
    ap.add_argument("--eval", default="eval_outputs/stag_EVAL50/trace.jsonl")
    ap.add_argument("--start", type=int, default=4, help="first replan the new signals may fire")
    ap.add_argument("--max-iter", type=int, default=30)
    ap.add_argument("--false-budget", type=float, default=0.03, help="max share of later successes touched (calibration)")
    ap.add_argument("--out", default="reports/stag_signals/summary.md")
    a = ap.parse_args()

    load = lambda f: [json.loads(line) for line in (ROOT / f).open() if line.strip()]
    sets = {"calib": load(a.calib), "eval": load(a.eval)}
    t0p = {"gamma": 0.15}
    t0 = {name: {id(e): first_fire(e, "T0", t0p, 10, a.max_iter) for e in eps} for name, eps in sets.items()}
    grids = {"D1": [{"theta": th} for th in (0.0, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2)],
             "D2": [{"q": q, "theta": th} for q, th in itertools.product((0.05, 0.1, 0.2, 0.3), (0.0, 0.05, 0.1, 0.15))]}
    lines = ["# 卡住检测信号的离线比较", "",
             f"阈值在标定集（{a.calib}）上选：碰到之后成功的 episode 不超过 {a.false_budget:.0%}，在此前提下召回最高；"
             f"然后原样用到评估集（{a.eval}）。新信号从第 {a.start + 1} 次重规划起可触发，T0 从第 11 次起。", ""]
    base = {n: evaluate(sets[n], "T0", t0p, 10, a.max_iter, t0[n]) for n in sets}
    rows = [("T0（现在的触发）", t0p, base)]
    for kind, grid in grids.items():
        cand = []
        for p in grid:
            r = evaluate(sets["calib"], kind, p, a.start, a.max_iter, t0["calib"])
            if r["false"] <= a.false_budget * r["n_succ"]:
                cand.append((r["recall"], r["lead_median"] if r["lead_median"] == r["lead_median"] else -99, p))
        if not cand:
            lines.append(f"- {kind}：网格里没有满足误碰预算的阈值")
            continue
        best = max(cand, key=lambda c: (c[0], c[1]))[2]
        rows.append((kind, best, {n: evaluate(sets[n], kind, best, a.start, a.max_iter, t0[n]) for n in sets}))
    lines += ["| 信号 | 阈值 | 集合 | 碰到的失败 | 碰到的之后成功 | 首次触发中位（次） | 比 T0 提前中位（次） | 提前 ≥4 次 | T0 没碰到、它碰到 |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for name, p, res in rows:
        for n in ("calib", "eval"):
            r = res[n]
            lines.append(f"| {name} | {p} | {'标定 S7' if n == 'calib' else '评估 EVAL50'} | {r['recall']}/{r['n_fail']} | "
                         f"{r['false']}/{r['n_succ']} | {r['first_median']:.0f} | {r['lead_median']:+.0f} | {r['lead_ge4']} | {r['only_new']} |")
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
