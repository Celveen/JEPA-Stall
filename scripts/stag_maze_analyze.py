"""Summarise the maze baselines and stall regulariser arms (scripts/stag_maze_run.py).

Per level: success of the frozen planner (final_eval_results.npz) and of AdaJEPA (ref, per-replan logs),
paired by episode; what AdaJEPA's failures look like (distance left to the goal, progress over the last
`tail` replans, net movement of the agent: a stall is little movement with no progress); and the latent
trigger's operating point (stag_r < gamma_hi, same threshold as on PushObj): how many failures and how
many later successes it touches. With regulariser arms present, paired success against ref.

    python scripts/stag_maze_analyze.py --levels medium hard
"""
import argparse
import collections
import json
from pathlib import Path

import numpy as np

from escape_common import jsonl, mcnemar
from stag_common import ROOT


def load_ref(d):
    """{sample_idx: records sorted by step} and episode summaries over the shards of one arm."""
    eps, meta = collections.defaultdict(list), {}
    for shard in sorted(d.glob("shard*")):
        for r in jsonl(shard / "per_iter_logs.json"):
            eps[r["sample_idx"]].append(r)
        for m in jsonl(shard / "stag_episodes.jsonl"):
            meta[m["sample_idx"]] = m
    return {k: sorted(v, key=lambda r: r["step"]) for k, v in eps.items()}, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="eval_outputs/stag_maze")
    ap.add_argument("--levels", nargs="+", default=["medium", "hard"])
    ap.add_argument("--arms", nargs="*", default=["SZ_lo", "SZ_mid", "SZ_hi"])
    ap.add_argument("--tail", type=int, default=5)
    ap.add_argument("--still", type=float, default=0.3, help="net agent movement over the tail counted as staying (maze units)")
    ap.add_argument("--progress", type=float, default=0.3, help="distance gain over the tail counted as progress")
    ap.add_argument("--out", default="reports/stag_maze/baselines.md")
    a = ap.parse_args()

    root = ROOT / a.root
    lines = ["# 多布局迷宫：基线与卡住分析", ""]
    summary = {}
    for level in a.levels:
        ref, meta = load_ref(root / level / "ref")
        if not ref:
            continue
        n = max(ref) + 1
        ok_ref = np.array([any(r["success"] for r in ref.get(i, [])) for i in range(n)])
        fz = root / level / "frozen" / "all" / "final_eval_results.npz"
        ok_fz = np.load(fz)["successes"].astype(bool)[:n] if fz.exists() else None
        lines += [f"## {level}（{n} 个 episode）", ""]
        if ok_fz is not None:
            b, c, p = mcnemar(ok_ref, ok_fz)
            lines.append(f"- 冻结规划器 {ok_fz.mean():.0%}，AdaJEPA {ok_ref.mean():.0%}（仅 AdaJEPA / 仅冻结 = {b}/{c}，p={p:.3g}）")
        else:
            lines.append(f"- AdaJEPA {ok_ref.mean():.0%}")
        fails = [i for i in range(n) if not ok_ref[i]]
        kinds = collections.Counter()
        rows = []
        for i in fails:
            rs = ref[i]
            d = np.array([r["dist_pos"] for r in rs])
            pos = np.array([r["state"][:2] for r in rs])
            t = min(a.tail, len(rs) - 1)
            prog = d[-1 - t] - d[-1]
            move = float(np.linalg.norm(pos[-1] - pos[-1 - t]))
            kind = "stall" if (move < a.still and prog < a.progress) else ("progressing" if prog >= a.progress else "moving, no progress")
            kinds[kind] += 1
            rows.append((d[-1], prog, move))
        if fails:
            r = np.array(rows)
            lines += [f"- AdaJEPA 失败 {len(fails)} 个：结束时离目标中位 {np.median(r[:, 0]):.2f}（成功阈值 0.5）；"
                      f"最后 {a.tail} 次重规划距离改善中位 {np.median(r[:, 1]):.2f}，智能体净移动中位 {np.median(r[:, 2]):.2f}",
                      f"- 失败类型（最后 {a.tail} 次重规划）：" + "，".join(f"{k} {v}" for k, v in kinds.most_common())]
        fired = {i for i, m in meta.items() if m.get("first_trigger") is not None}
        ff = len(fired & set(fails))
        fs = len(fired - set(fails))
        lines.append(f"- 潜空间触发（阈值 0.15，未在迷宫上调）：碰到 {len(fired)} 个 episode = {ff}/{len(fails)} 个失败 + {fs} 个之后成功的")
        summary[level] = {"n": n, "ada": float(ok_ref.mean()), "frozen": None if ok_fz is None else float(ok_fz.mean()),
                          "fail_kinds": dict(kinds), "trigger_fail": ff, "trigger_success": fs}
        for arm in a.arms:
            eps, _ = load_ref(root / level / arm)
            if not eps:
                continue
            ok = ok_ref.copy()
            for i, rs in eps.items():
                ok[i] = any(r["success"] for r in rs)
            b, c, p = mcnemar(ok, ok_ref)
            lines.append(f"- {arm}：{ok.mean():.0%}（仅该组 / 仅 AdaJEPA = {b}/{c}，p={p:.3g}）")
            summary[level][arm] = float(ok.mean())
        lines.append("")
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    out.with_suffix(".json").write_text(json.dumps(summary, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
