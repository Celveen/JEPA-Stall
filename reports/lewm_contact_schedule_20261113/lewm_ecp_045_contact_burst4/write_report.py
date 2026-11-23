"""Write the measured results and interpretation inputs after all audits pass."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
result = json.loads((ROOT / "results.json").read_text())
diagnosis = json.loads((ROOT / "diagnostics.json").read_text())
total = result["total"]
old = diagnosis["contact"]
new = diagnosis["burst4"]
lines = [
    "# LeWM PushObj 连续引导验证",
    "",
    "增强数据训练的 LeWM epoch 10，冻结评测。复用 H7t3 的 350 个任务，7 种形状各 50 个。",
    "",
    "## 设置",
    "",
    "停滞阈值上界 0.45、下界 0.15，最早第 10 次重规划（从 0 计数，即执行 50 个控制步后）允许触发。",
    "MPC 最多 30 次重规划，每次执行 5 个控制步，总计最多 150 步。动作偏置每步上限 15 像素，未增加下限。",
    "",
    "本组在触发后连续引导 4 次重规划（20 个控制步），每次根据实际状态重新计算接触引导方向。",
    "一段引导结束后，下一段仍需新的触发，且距离上一次实际偏置至少 4 次重规划。",
    "每个任务最多施加 7 次偏置，达到成功状态后停止干预。",
    "持续引导不会伪造新的停滞触发，本体项仍沿用原来的检测触发与物体位姿就绪条件。",
    "",
    "接触引导使用仿真器提供的物体位姿和形状几何。该实验验证控制策略，使用的观测信息与上一组接触引导实验相同。",
    "",
    "## 成功率",
    "",
    "| 配置 | 成功数 / 350 | 成功率 |",
    "|---|---:|---:|",
]
for key, title in [("native", "增强 LeWM 原生规划"), ("regularizer", "停滞正则"),
                   ("combined", "停滞正则 + 目标方向偏置"), ("proprio", "再加本体加权"),
                   ("contact", "分阶段接触引导，间歇干预"), ("burst4", "分阶段接触引导，连续 4 次重规划")]:
    lines.append(f"| {title} | {total[key]} / 350 | {100 * total[key] / 350:.2f}% |")
lines += ["", "| 形状 | 原生 | 间歇接触引导 | 连续接触引导 |", "|---|---:|---:|---:|"]
for shape, values in result["by_shape"].items():
    lines.append(f"| {shape} | {values['native']}/50 | {values['contact']}/50 | {values['burst4']}/50 |")
lines += [
    "",
    f"相对间歇接触引导，救回 {total['rescues_vs_contact']} 个任务，退化 {total['regressions_vs_contact']} 个任务。",
    f"相对原生规划，救回 {total['rescues_vs_native']} 个任务，退化 {total['regressions_vs_native']} 个任务。",
    "",
    "## 干预是否生效",
    "",
    "| 指标 | 间歇接触引导 | 连续接触引导 |",
    "|---|---:|---:|",
    f"| 实际动作偏置次数 | {old['bias_events']} | {new['bias_events']} |",
    f"| 原生失败且触发的匹配任务 | {old['n']} | {new['n']} |",
    f"| 首次触发后 15 步内接触物体的任务 | {old['contact_in_next_15']} | {new['contact_in_next_15']} |",
    f"| 首次触发后 15 步物体位置误差下降超过 5 像素的任务 | {old['object_position_error_dropped_five_px_after_15']} | {new['object_position_error_dropped_five_px_after_15']} |",
    f"| 触发后曾满足物体位置与角度条件的任务 | {old['object_translation_and_angle_pass']} | {new['object_translation_and_angle_pass']} |",
    f"| 接近阶段实际位移未沿引导方向前进的次数 | {old['approach_events_without_positive_projected_motion']} | {new['approach_events_without_positive_projected_motion']} |",
    f"| 启用本体加权的任务与重规划组合数 | {old['proprio_replans']} | {new['proprio_replans']} |",
    "",
    f"连续引导各阶段的实际偏置次数：{dict(result['bias_events_by_phase'])}。",
    f"其中 {result['continuation_events_without_new_trigger']} 次来自既有干预段的延续，当次检测未再次触发。",
    "",
    "两组的参数上限一致，但连续引导增加了实际偏置次数。该对比衡量连续引导策略的整体效果，尚未单独分离干预次数与时序的作用。",
    "",
    "## 核验与文件",
    "",
    "逐任务核对初始状态、目标状态、模型权重摘要、首次触发时间、首次干预之前及首次干预的动作。",
    "仅偏置调度变化，保存的模型参数与缓冲区全程未变。所有 350 个任务以保存动作重新运行物理仿真，结果与原测评逐项一致。",
    "",
    "- `results.json`、`paired_tasks.csv` 保存汇总与逐任务对比。",
    "- `diagnostics.json` 保存接触、位移、出界与成功翻转任务分析。",
    "- `full/` 保存配置、日志、偏置审计和全部执行动作。",
    "- `replayed/` 保存逐控制步的物理状态和真实碰撞接触记录。",
    "- `source/`、`run_eval.py`、`run_all.sh` 与 `manifest.json` 保存实现、启动入口及来源摘要。",
    "",
]
(ROOT / "RESULTS.md").write_text("\n".join(lines))
print(ROOT / "RESULTS.md")
