"""Find the episodes the stall regulariser (planning/stagnation.py) can touch, from an AdaJEPA run.

The trigger reads only the latent movement rate r = d(z_s, z_{s-W}) / d(z_s, z_goal), and until it
first fires the planner is AdaJEPA bit for bit. So on the AdaJEPA (escape.mode=none) run of the same
targets and seeds, episodes whose r never drops below gamma_hi (while armed: start <= s < max_iter,
not yet successful) stay identical under every arm and need no rerun. r comes from the latent traces
of scripts/stag_latent_trace.py (base encoder); the planner measures it with the adapted encoder,
so the selection uses gamma_hi * margin to keep a superset.

With --calibrate, prints the operating points of a grid of thresholds instead (fires on failures /
on episodes that later succeed), to choose gamma_hi on a calibration stream.

    python scripts/stag_select.py --trace eval_outputs/stag_EVAL50/trace.jsonl --out eval_outputs/stag_EVAL50/touched.json
"""
import argparse
import collections
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def first_fire(ep, thr, start, max_iter):
    for r in ep["steps"]:
        if r["success"]:
            return None
        if start <= r["s"] < max_iter and "dW" in r and r["dW"] / max(r["dg"], 1e-12) < thr:
            return r["s"]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default="eval_outputs/stag_EVAL50/trace.jsonl")
    ap.add_argument("--start", type=int, default=10)
    ap.add_argument("--max-iter", type=int, default=30)
    ap.add_argument("--gamma-hi", type=float, default=0.15)
    ap.add_argument("--margin", type=float, default=1.3)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--out", default="eval_outputs/stag_EVAL50/touched.json")
    a = ap.parse_args()

    eps = [json.loads(line) for line in (ROOT / a.trace).open() if line.strip()]
    fail = [e for e in eps if e["success_step"] is None]
    if a.calibrate:
        for thr in np.round(np.arange(0.05, 0.31, 0.01), 3):
            ff = sum(first_fire(e, thr, a.start, a.max_iter) is not None for e in fail)
            fs = sum(first_fire(e, thr, a.start, a.max_iter) is not None for e in eps) - ff
            print(f"gamma_hi {thr:.2f}: fires on {ff}/{len(fail)} failures, {fs}/{len(eps) - len(fail)} later successes")
        return
    thr = a.gamma_hi * a.margin
    rows, chunks = [], collections.defaultdict(list)
    for e in eps:
        s_sel, s_trig = first_fire(e, thr, a.start, a.max_iter), first_fire(e, a.gamma_hi, a.start, a.max_iter)
        if s_sel is not None:
            chunks[e["chunk"]].append(e["sample_idx"])
            rows.append({"chunk": e["chunk"], "sample_idx": e["sample_idx"], "first_trigger_offline": s_trig,
                         "first_below_margin": s_sel, "ref_success_step": e["success_step"]})
    n_succ = sum(r["ref_success_step"] is not None for r in rows)
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "trace": a.trace, "start": a.start, "max_iter": a.max_iter, "gamma_hi": a.gamma_hi, "margin": a.margin,
        "n_episodes": len(eps), "n_ref_success": len(eps) - len(fail), "n_selected": len(rows),
        "n_selected_ref_success": n_succ,
        "n_trigger_offline": sum(r["first_trigger_offline"] is not None for r in rows),
        "chunks": {k: sorted(v) for k, v in chunks.items()}, "episodes": rows,
    }, indent=1))
    print(f"{len(eps)} episodes ({len(eps) - len(fail)} AdaJEPA successes); selected {len(rows)} "
          f"({len(rows) - n_succ} AdaJEPA failures, {n_succ} later successes) in {len(chunks)} chunks -> {out}")


if __name__ == "__main__":
    main()
