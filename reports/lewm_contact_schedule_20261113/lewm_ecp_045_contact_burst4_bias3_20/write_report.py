"""Summarize the joint amplitude floor/cap experiment against identical burst scheduling."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
result = json.loads((ROOT / "results.json").read_text())
diagnosis = json.loads((ROOT / "diagnostics.json").read_text())
total = result["total"]
old, new = diagnosis["burst4"], diagnosis["bias3_20"]
lines = [
    "# LeWM 连续干预：动作偏置幅度 3–20 像素",
    "",
    "增强数据训练的 LeWM epoch 10，冻结评测，复用同一批 H7t3 PushObj 350 个任务，7 种形状各 50 个。",
    "",
    "## 实验设置",
    "",
    "本组仅改变动作偏置幅度，把上限从 15 提高到 20 像素，并增加 3 像素下限。",
    "对非零引导方向 d，实际附加偏置为 d / ||d|| × clip(||d|| / 5, 3, 20)，单位为每个环境控制步的相对目标偏移像素。",
    "上下限约束二维向量的长度，保持方向。零方向仍不施加偏置。偏置加在原规划动作上，实际物理位移由仿真动力学决定。",
    "",
    "沿用阈值 0.45 / 0.15，最早 50 个控制步后触发，连续引导 4 次重规划，每次执行 5 步，总预算 30 × 5 = 150 步。",
    "每个任务最多施加 7 次偏置，新引导段距离上一次实际偏置至少间隔 4 次重规划，并需新的停滞触发。",
    "停滞正则、接触点选择、本体加权条件及模型权重沿用上一组。接触引导仍使用仿真器提供的物体位姿与几何。",
    "推物阶段沿用名义长度 75 像素的引导向量，因此该阶段偏置为 15 像素；新的上下限作用于原有幅度规则，未强制所有偏置取最大值。",
    "",
    "## 成功率",
    "",
    "| 配置 | 成功数 / 350 | 成功率 |",
    "|---|---:|---:|",
]
for key, title in [("native", "原生规划"), ("contact", "间歇接触干预，上限 15"),
                   ("burst4", "连续 4 次，无下限、上限 15"),
                   ("bias3_20", "连续 4 次，下限 3、上限 20")]:
    lines.append(f"| {title} | {total[key]} / 350 | {100 * total[key] / 350:.2f}% |")
lines += [
    "",
    f"相对上一组连续干预，救回 {total['rescues_vs_burst4']} 个任务，退化 {total['regressions_vs_burst4']} 个任务。",
    f"净变化 {total['bias3_20'] - total['burst4']:+d} 个成功任务，成功率变化 {100 * (total['bias3_20'] - total['burst4']) / 350:+.2f} 个百分点。",
    "",
    "| 形状 | 连续干预，上限 15 | 连续干预，3–20 |",
    "|---|---:|---:|",
]
for shape, value in result["by_shape"].items():
    lines.append(f"| {shape} | {value['burst4']}/50 | {value['bias3_20']}/50 |")
lines += [
    "",
    "## 幅度与物理效果",
    "",
    f"新组实际施加偏置 {new['bias_events']} 次，触及 3 像素下限 {new['events_at_3px']} 次，触及 20 像素上限 {new['events_at_20px']} 次。",
    f"实际幅度分位数（最小、10%、25%、中位、最大）为 {new['bias_amplitude_quantiles_px']}。",
    "",
    "下表物理指标使用原生规划失败且有触发的 134 个匹配任务，统计首次触发后的轨迹。",
    "",
    "| 指标 | 上限 15 | 3–20 |",
    "|---|---:|---:|",
    f"| 实际动作偏置次数（全部任务） | {old['bias_events']} | {new['bias_events']} |",
    f"| 15 步内接触物体的任务 | {old['contact_in_next_15']} | {new['contact_in_next_15']} |",
    f"| 15 步后物体位置误差下降超过 5 像素的任务 | {old['object_position_error_dropped_five_px_after_15']} | {new['object_position_error_dropped_five_px_after_15']} |",
    f"| 曾满足物体位置与角度条件的任务 | {old['object_translation_and_angle_pass']} | {new['object_translation_and_angle_pass']} |",
    f"| 启用本体加权的任务与重规划组合数（全部任务） | {old['proprio_replans']} | {new['proprio_replans']} |",
    "",
    f"新组各阶段实际偏置次数：{dict(result['bias_events_by_phase'])}。",
    "",
    "## 核验与文件",
    "",
    "逐任务核对了初始状态、目标、模型摘要、首次触发时间及触发前动作，均与上一组一致。",
    "所有已施加偏置均通过物理单位换算检查，幅度符合 3–20 像素规则。全部 350 个任务已用保存动作重新运行物理仿真，成功结果逐项一致。",
    "",
    "- `results.json` 和 `paired_tasks.csv`：汇总与逐任务对比。",
    "- `diagnostics.json`：物理轨迹、接触和幅度分析。",
    "- `full/`：配置、日志、检测记录、动作偏置记录与执行动作。",
    "- `replayed/`：逐控制步状态和实际碰撞接触记录。",
    "- `source/`、启动脚本、`manifest.json`：实现和实验来源摘要。",
    "",
]
(ROOT / "RESULTS.md").write_text("\n".join(lines))
print(ROOT / "RESULTS.md")
