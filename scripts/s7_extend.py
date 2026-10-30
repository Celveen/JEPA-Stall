"""Extend the S7 AdaJEPA reference from 30 to 50 replans (E3 plan, section 4).

Only the episodes that had not succeeded within 30 replans are rerun, through StagnationMPCPlanner with kappa 0
(scripts/stag_run.py, arm S0: trigger bookkeeping only), so the new run also logs the stall trigger. Episodes keep
their position in their chunk, hence their seed and goal; episodes that succeeded within 30 replans are
unchanged by a longer budget and keep the reference's logs.

    --select  write <root>/touched.json: per chunk, the reference episodes with no success within its budget
    --check   every replan up to the reference's budget must equal the reference (actions, success, losses)

    python scripts/s7_extend.py --select
    python scripts/stag_run.py --stream S7 --arms S0 --root eval_outputs/stag_S7_it50 --extra planner.max_iter=50 \
        --gpus 4,7 --jobs 6
    python scripts/s7_extend.py --check
"""
import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ("act", "success", "loss_new_cur", "loss_buf_cur", "dist4")


def records(run_dir):
    """{(chunk, sample_idx): {step: record}} over the chunks of a run."""
    out = collections.defaultdict(dict)
    for f in sorted(Path(run_dir).glob("*/per_iter_logs.json")):
        for line in f.open():
            if line.strip():
                r = json.loads(line)
                out[(f.parent.name, r["sample_idx"])][r["step"]] = r
    return out


def select(a):
    ref = records(ROOT / a.ref)
    chunks = collections.defaultdict(list)
    for (ch, i), steps in sorted(ref.items()):
        if not any(r["success"] for r in steps.values()):
            chunks[ch].append(i)
    out = ROOT / a.root / "touched.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"ref": a.ref, "rule": "no success within the reference budget",
                               "n": sum(map(len, chunks.values())), "chunks": dict(chunks)}, indent=1))
    print(f"{out.relative_to(ROOT)}: {sum(map(len, chunks.values()))} of {len(ref)} episodes in {len(chunks)} chunks")


def check(a):
    ref, new = records(ROOT / a.ref), records(ROOT / a.root / a.arm)
    budget = max(max(st) for st in ref.values())
    n = bad = 0
    for k, steps in sorted(new.items()):
        for s in range(1, budget + 1):
            n += 1
            q, r = ref[k].get(s), steps.get(s)
            if q is None or r is None or any(r.get(f) != q.get(f) for f in FIELDS):
                bad += 1
    longer = sum(max(st) > budget for st in new.values())
    succ = sum(any(r["success"] for r in st.values()) for st in new.values())
    print(f"{len(new)} episodes; {n} replans up to {budget} compared with the reference; mismatches {bad}; "
          f"{longer} ran past {budget}; {succ} succeeded")
    sys.exit(0 if bad == 0 and n > 0 else 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="eval_outputs/escape_S7/none")
    ap.add_argument("--root", default="eval_outputs/stag_S7_it50")
    ap.add_argument("--arm", default="S0")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--select", action="store_true")
    g.add_argument("--check", action="store_true")
    a = ap.parse_args()
    select(a) if a.select else check(a)


if __name__ == "__main__":
    main()
