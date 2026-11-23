"""Write the measured contact-objective experiment and matched diagnostics."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
result = json.loads((ROOT / "results.json").read_text())
diagnostics = json.loads((ROOT / "diagnostics.json").read_text())
total = result["total"]
old, new = diagnostics["bias3_20"], diagnostics["objective"]

lines = [
    "# LeWM PushObj：把接触引导纳入动作优化",
    "",
    "增强数据 LeWM epoch 10，冻结评测，同一批 H7t3 的 350 个任务，7 种形状各 50 个。",
    "",
    "## 代码变更",
    "",
    "新增 `source/planning/lewm_contact_objective.py`，沿用已有触发器、连续干预调度、接触点选择和停滞正则。",
    "将接触准备点、向物体内推的目标点或最终推杆目标写入动作优化目标，取消优化后叠加动作偏移。",
    "",
    "新目标为 J = J_previous + 10 × mean_{k=1..5, xy} [((p_hat_k(A) - q) / sigma_p)^2]。",
    "A 为优化动作，q 为本次重规划选出的引导目标，p_hat_k 为执行第 k 个控制步后的预测推杆位置，sigma_p 为数据集推杆位置标准差。",
    "推杆位置由观测到的位置、速度及已核验的可微 PD 动力学计算。视觉目标、停滞正则和原有条件本体项仍共同参与优化。",
    "优化器求出的动作直接执行。原来 3–20 像素的附加偏移范围不再使用，也不构成新动作的硬幅度约束。",
    "",
    "触发阈值为上界 0.45 / 下界 0.15，最早 50 个环境控制步后允许触发。每段连续干预 4 次重规划，每次执行 5 步，每任务最多 7 次引导，最多 30 次重规划 / 150 控制步。GD 每次优化 100 次，规划预测范围仍为 25 控制步。",
    "接触点选择沿用观测到的仿真物体位姿与已知形状几何，与上一组使用的信息相同。",
    "",
    "## 成功率",
    "",
    "| 配置 | 成功数 / 350 | 成功率 |",
    "|---|---:|---:|",
]
for key, label in [("native", "原生规划"), ("contact", "间歇接触偏置"),
                   ("burst4", "连续接触偏置，上限 15"), ("bias3_20", "连续接触偏置，3–20"),
                   ("objective", "连续干预，接触目标进入优化")]:
    lines.append(f"| {label} | {total[key]} / 350 | {100 * total[key] / 350:.2f}% |")
lines += ["", "| 形状 | 上限 15 偏置 | 3–20 偏置 | 接触目标优化 |", "|---|---:|---:|---:|"]
for shape, counts in result["by_shape"].items():
    lines.append(f"| {shape} | {counts['burst4']}/50 | {counts['bias3_20']}/50 | {counts['objective']}/50 |")
lines += [
    "",
    f"相对 3–20 偏置组，救回 {total['rescues_vs_bias3_20']} 个任务，退化 {total['regressions_vs_bias3_20']} 个任务。",
    f"相对上限 15 的连续偏置组，救回 {total['rescues_vs_burst4']} 个任务，退化 {total['regressions_vs_burst4']} 个任务。",
    "",
    "## 干预后的实际效果",
    "",
    "| 指标 | 3–20 偏置 | 接触目标优化 |",
    "|---|---:|---:|",
    f"| 干预次数 | {old['guidance_events']} | {new['guidance_events']} |",
    f"| 接近阶段干预次数 | {old['approach_events']} | {new['approach_events']} |",
    f"| 接近阶段有正向实际位移 | {old['approach_events_with_forward_movement']} | {new['approach_events_with_forward_movement']} |",
    f"| 接近阶段随后 5 步内实际接触物体 | {old['approach_events_with_actual_contact']} | {new['approach_events_with_actual_contact']} |",
    f"| 接近阶段正向位移中位数（像素） | {old['approach_forward_displacement_median_px']:.2f} | {new['approach_forward_displacement_median_px']:.2f} |",
    "",
    "以下统计使用原生失败且触发的 134 个匹配任务。",
    "",
    "| 指标 | 3–20 偏置 | 接触目标优化 |",
    "|---|---:|---:|",
    f"| 首次触发后 15 步内接触物体 | {old['contact_in_next_15']} | {new['contact_in_next_15']} |",
    f"| 15 步后物体位置误差下降超过 5 像素 | {old['object_position_error_dropped_five_px_after_15']} | {new['object_position_error_dropped_five_px_after_15']} |",
    f"| 触发后曾满足物体位置与角度要求 | {old['object_translation_and_angle_pass']} | {new['object_translation_and_angle_pass']} |",
    "",
    f"新组引导阶段次数为 {dict(result['guidance_events_by_phase'])}。",
    f"接触目标代价在 {new['contact_cost_decreased']}/{new['guidance_events']} 次干预优化中下降，相对下降幅度的中位数为 {100 * new['median_contact_cost_reduction_fraction']:.2f}%。",
    "",
    "## 核验与复现材料",
    "",
    "新代价通过有限差分梯度检查，并验证仅对应任务的执行前缀收到接触项梯度。解析动力学与逐小步 PD 积分一致。",
    "全部任务的初始状态、目标、首次触发和触发前动作与匹配对照逐项一致，模型权重和缓冲区保持不变。",
    "350 个任务使用保存动作重新运行物理仿真，成功结果逐项一致。",
    f"记录的优化后预测推杆位置与实际执行位置的最大坐标偏差为 {new['max_predicted_vs_executed_position_error_px']:.6f} 像素，确认实际执行的是优化后的引导轨迹。",
    "",
    "`results.json`、`paired_tasks.csv`、`diagnostics.json` 保存汇总与逐任务证据。`full/` 保存配置、日志、动作、检测记录和优化前后接触代价。`replayed/` 保存真实物理状态及接触记录。",
    "`source/`、`run_eval.py`、`run_all.sh`、`check_objective.py` 和 `manifest.json` 保存实现、启动入口、验证及摘要。",
    "",
]
(ROOT / "RESULTS.md").write_text("\n".join(lines))
print(ROOT / "RESULTS.md")
