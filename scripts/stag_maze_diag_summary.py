"""Summarise scripts/stag_maze_diag.py: what the world model predicts vs what happens, on stalled and moving replans.

    python scripts/stag_maze_diag_summary.py --levels easy medium
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="eval_outputs/stag_maze_diag")
    ap.add_argument("--levels", nargs="+", default=["easy", "medium"])
    ap.add_argument("--base", default="frozen")
    ap.add_argument("--out", default="reports/stag_maze_diag/summary.md")
    a = ap.parse_args()

    med = lambda x: float(np.median(x)) if len(x) else float("nan")  # noqa: E731
    lines = ["# 迷宫卡住原因的离线诊断", "",
             "位置单位是迷宫格；成功要求离目标 < 0.5 格。预测位置由模型预测的本体嵌入反解得到。"
             "\"5 步计划\"是从该重规划的观测出发、用规划器同样的目标和优化器重新算出的计划，预测值来自模型，实际值是把同一组动作放进模拟器执行的结果。", "",
             "| 难度 | 重规划 | 个数 | 单步：实际移动 | 单步：预测移动 | 单步：预测误差 | 计划：预测向目标推进 | 计划：实际推进 | 计划：预测移动 | 计划：实际移动 | 预测路径穿墙 | 预测推进 ≥ 0.3 而实际 < 0.1 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    res = {}
    for level in a.levels:
        f = ROOT / a.dir / f"{level}_{a.base}.jsonl"
        if not f.exists():
            continue
        rows = [json.loads(line) for line in f.open() if line.strip()]
        for kind in ("stalled", "moving"):
            r = [x for x in rows if x["kind"] == kind]
            if not r:
                continue
            g = lambda k: np.array([x[k] for x in r], dtype=float)  # noqa: E731
            over = np.mean((g("plan_gain_pred") >= 0.3) & (g("plan_gain_true") < 0.1))
            wall = np.mean([x["plan_crosses_wall"] for x in r])
            rec = {"n": len(r), "step_true": med(g("step_true")), "step_pred": med(g("step_pred")),
                   "step_err": med(g("step_pred_err")), "gain_pred": med(g("plan_gain_pred")),
                   "gain_true": med(g("plan_gain_true")), "move_pred": med(g("plan_move_pred")),
                   "move_true": med(g("plan_move_true")), "wall": float(wall), "over": float(over),
                   "readout_err_true": med(g("readout_err")),
                   "lat_goal_now": med([x["lat_goal_now"][0] for x in r]),
                   "lat_goal_end_pred": med([x["lat_goal_end_pred"][0] for x in r]),
                   "lat_goal_end_true": med([x["lat_goal_end_true"][0] for x in r])}
            res[f"{level}/{kind}"] = rec
            lines.append(f"| {level} | {'卡住' if kind == 'stalled' else '正常移动'} | {len(r)} | {rec['step_true']:.2f} | "
                         f"{rec['step_pred']:.2f} | {rec['step_err']:.2f} | {rec['gain_pred']:+.2f} | {rec['gain_true']:+.2f} | "
                         f"{rec['move_pred']:.2f} | {rec['move_true']:.2f} | {wall:.0%} | {over:.0%} |")
    lines += ["", "规划器自身度量下（潜空间距离，越小越接近目标；中位数）：", "",
              "| 难度 | 重规划 | 现在离目标 | 计划终点：模型预测 | 计划终点：实际 |", "|---|---|---:|---:|---:|"]
    for k, rec in res.items():
        level, kind = k.split("/")
        lines.append(f"| {level} | {'卡住' if kind == 'stalled' else '正常移动'} | {rec['lat_goal_now']:.3f} | "
                     f"{rec['lat_goal_end_pred']:.3f} | {rec['lat_goal_end_true']:.3f} |")
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    out.with_suffix(".json").write_text(json.dumps(res, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
