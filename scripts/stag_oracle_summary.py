"""Summarise the oracle diagnostic (scripts/stag_oracle.py): per variant, on the episodes all variants finished."""
import argparse
import json

import numpy as np

from stag_common import ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="eval_outputs/stag_EVAL50/oracle")
    ap.add_argument("--variants", nargs="+", default=["O1", "O2", "O3"])
    ap.add_argument("--out", default="reports/stag_EVAL50/oracle.md")
    a = ap.parse_args()
    res = {}
    for v in a.variants:
        f = ROOT / a.dir / f"{v}.jsonl"
        if f.exists():
            res[v] = {(r["chunk"], r["sample_idx"]): r for r in map(json.loads, f.open())}
    keys = sorted(set.intersection(*[set(d) for d in res.values()]))
    lines = [f"# 神谕诊断：真实模拟器替换世界模型（{len(keys)} 个 AdaJEPA 失败的卡住状态）", "",
             "| 设定 | 成功 | 只差推杆 / 物体没到位 | 物体改善 | 推杆绕到物体背后 | 第 1 次重规划最优计划不比原地不动好 |",
             "|---|---:|---:|---:|---:|---:|"]
    out = {"n": len(keys)}
    for v, d in res.items():
        rows = [d[k] for k in keys]
        succ = np.array([r["success"] for r in rows])
        po = np.array([r["obj_err0"] < 20 and r["ang0_deg"] < 20 for r in rows])
        better = np.mean([r["success"] or (r["obj_err0"] - r["log"][-1]["obj_err"] > 5) for r in rows])
        behind = np.mean([r["side_min"] < -0.3 for r in rows])
        stay = np.mean([r["log"][0]["best"] >= 0.99 * r["log"][0]["stay"] for r in rows])
        out[v] = {"success": int(succ.sum()), "pusher_only": [int(succ[po].sum()), int(po.sum())],
                  "object_off": [int(succ[~po].sum()), int((~po).sum())], "obj_better": better, "behind": behind, "stay_best": stay}
        lines.append(f"| {v} | {succ.sum()}/{len(rows)} ({succ.mean():.0%}) | {succ[po].sum()}/{po.sum()} / {succ[~po].sum()}/{(~po).sum()} | "
                     f"{better:.0%} | {behind:.0%} | {stay:.0%} |")
    rows0 = [res[a.variants[0]][k] for k in keys]
    lines += ["", f"起点：物体误差中位 {np.median([r['obj_err0'] for r in rows0]):.0f} px，角度 {np.median([r['ang0_deg'] for r in rows0]):.0f}°，"
              f"{np.mean([r['side0'] > 0.3 for r in rows0]):.0%} 的推杆在物体朝目标的一侧；剩余预算中位 {np.median([r['budget'] for r in rows0]):.0f} 次重规划。",
              "O1 = 模拟器 + 潜空间目标（视野 25 步）；O2 = 模拟器 + 真实状态代价（25 步）；O3 = O2 视野 50 步。AdaJEPA 在这些状态上全部失败。"]
    path = ROOT / a.out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    path.with_suffix(".json").write_text(json.dumps(out, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
