"""Success of the maze runs of scripts/stag_maze_base_run.py, per level, base and replan budget.

Episodes an arm did not rerun keep the reference outcome (identical by construction). Every rerun episode is first
checked to reproduce the reference's actions exactly up to the replan its penalty switches on (entirely when it
never switches on). Budgets are truncations of the same runs (a run with a larger max_iter contains the smaller
budgets exactly). Paired exact McNemar against the reference of the same base.

    python scripts/stag_maze_base_analyze.py --levels easy medium --bases frozen adajepa --arms SZ_mid T0SP
"""
import argparse
import collections
import json

import numpy as np

from escape_common import jsonl, mcnemar
from stag_common import ROOT


def load(d):
    eps, meta = collections.defaultdict(list), {}
    for shard in sorted(d.glob("shard*")):
        for r in jsonl(shard / "per_iter_logs.json"):
            eps[r["sample_idx"]].append(r)
        for m in jsonl(shard / "stag_episodes.jsonl"):
            meta[m["sample_idx"]] = m
    return {k: sorted(v, key=lambda r: r["step"]) for k, v in eps.items()}, meta


def ok(rs, budget):
    return any(r["success"] for r in rs if r["step"] <= budget)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="eval_outputs/stag_maze100")
    ap.add_argument("--levels", nargs="+", default=["easy", "medium"])
    ap.add_argument("--bases", nargs="+", default=["frozen", "adajepa"])
    ap.add_argument("--arms", nargs="*", default=["SZ_mid", "T0SP"])
    ap.add_argument("--budgets", nargs="+", type=int, default=[20, 30, 40, 50])
    ap.add_argument("--out", default="reports/stag_maze100/summary")
    a = ap.parse_args()

    root = ROOT / a.root
    names = {"frozen": "冻结规划器", "adajepa": "AdaJEPA"}
    res, lines = {}, ["# 多布局迷宫：停滞正则（100 个 episode，按重规划上限截取）", ""]
    for level in a.levels:
        lines += [f"## {level}", "", "| 基准 | 组 | " + " | ".join(f"{b} 次" for b in a.budgets) + " | 重跑 | 开罚前不一致 |",
                  "|---|---|" + "---:|" * len(a.budgets) + "---:|---:|"]
        for base in a.bases:
            ref, _ = load(root / level / base / "ref")
            if not ref:
                continue
            keys = sorted(ref)
            cells = [f"{np.mean([ok(ref[k], b) for k in keys]):.0%}（n={len(keys)}）" for b in a.budgets]
            lines.append(f"| {names[base]} | 参照 | " + " | ".join(cells) + " | — | — |")
            res[f"{level}/{base}/ref"] = {b: float(np.mean([ok(ref[k], b) for k in keys])) for b in a.budgets}
            for arm in a.arms:
                eps, meta = load(root / level / base / arm)
                if not eps:
                    continue
                bad = 0
                for k, rs in eps.items():
                    t = meta.get(k, {}).get("first_trigger")
                    lim = t if t is not None else 10 ** 9
                    same = all(r0["act"] == r1["act"] for r0, r1 in zip(ref[k], rs) if r1["step"] <= lim)
                    bad += (not same) or (t is None and len(ref[k]) != len(rs))
                cells, rec = [], {}
                for b in a.budgets:
                    o_ref = {k: ok(ref[k], b) for k in keys}
                    o_arm = {k: ok(eps[k], b) if k in eps else o_ref[k] for k in keys}
                    gain = sum(o_arm[k] and not o_ref[k] for k in keys)
                    loss = sum(o_ref[k] and not o_arm[k] for k in keys)
                    sr = float(np.mean([o_arm[k] for k in keys]))
                    cells.append(f"{sr:.0%}（{gain}/{loss}，p={mcnemar(gain, loss):.2g}）")
                    rec[b] = {"success": sr, "gain": gain, "loss": loss, "p": mcnemar(gain, loss)}
                res[f"{level}/{base}/{arm}"] = rec
                lines.append(f"| {names[base]} | {arm} | " + " | ".join(cells) + f" | {len(eps)} | {bad} |")
            if base == "adajepa" and "frozen" in a.bases:  # AdaJEPA vs frozen, same episodes and seeds
                fz, _ = load(root / level / "frozen" / "ref")
                common = sorted(set(fz) & set(ref))
                if common:
                    cells = []
                    for b in a.budgets:
                        gain = sum(ok(ref[k], b) and not ok(fz[k], b) for k in common)
                        loss = sum(ok(fz[k], b) and not ok(ref[k], b) for k in common)
                        cells.append(f"{gain}/{loss}，p={mcnemar(gain, loss):.2g}")
                    lines.append("| AdaJEPA 对冻结（仅 AdaJEPA / 仅冻结） | — | " + " | ".join(cells) + " | — | — |")
        lines.append("")
    lines += ["括号里是（仅该组成功 / 仅参照成功，配对精确 McNemar p）。重跑 = 该组在参照的触发窗口里会被碰到、因而重跑的 episode；"
              "其余 episode 与参照逐位相同，沿用参照结果。"]
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(res, indent=1))
    out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
