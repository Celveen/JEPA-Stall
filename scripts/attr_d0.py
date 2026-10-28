"""D0: offline model-or-ruler attribution from the logs of scripts/attr_run.py.

    --check   every (episode, replan) of a run against its reference run: executed actions, success and the stall
              trigger's ratio must be identical, and every planned replan must carry the attribution fields

    python scripts/attr_d0.py --check --runs A1 A2 B1 --root eval_outputs/attr_d0_smoke
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from attr_run import MAZE, REFS, ROOT  # noqa: E402

ATTR_FIELDS = ("attr_cv", "attr_cp", "attr_nv", "attr_np", "attr_J", "attr_zp1")


def records(run_dir, maze):
    """{(chunk, sample_idx, step): record} of all shards / chunks under run_dir."""
    out = {}
    for f in sorted(Path(run_dir).glob("*/per_iter_logs.json")):
        chunk = None if maze else f.parent.name
        for line in f.open():
            if line.strip():
                r = json.loads(line)
                out[(chunk, r["sample_idx"], r["step"])] = r
    return out


def check(run, root):
    maze = run in MAZE
    new, ref = records(ROOT / root / run, maze), records(ROOT / REFS[run], maze)
    bad, missing_attr = [], 0
    for k, r in new.items():
        q = ref.get(k)
        if q is None:
            bad.append((k, "not in reference"))
            continue
        for f in ("act", "success", "stag_r"):
            if (f in q or f != "stag_r") and r.get(f) != q.get(f):  # escape-planner references log no stag_r
                bad.append((k, f))
        missing_attr += any(f not in r for f in ATTR_FIELDS)
    eps = {(c, i) for c, i, _ in new}
    n_rs = sum("attr_rs" in r for r in new.values())
    print(f"{run}: {len(eps)} episodes, {len(new)} replans; mismatches {len(bad)}; "
          f"replans without attribution fields {missing_attr}; restart checks {n_rs}")
    for k, f in bad[:10]:
        print("   ", k, f)
    return not bad and not missing_attr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--runs", nargs="+", default=["A1", "A2"])
    ap.add_argument("--root", default="eval_outputs/attr_d0")
    a = ap.parse_args()
    if a.check:
        ok = [check(run, a.root) for run in a.runs]
        sys.exit(0 if all(ok) else 1)


if __name__ == "__main__":
    main()
