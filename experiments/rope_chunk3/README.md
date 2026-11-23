# Rope 动作块与惩罚强度对照

每种条件运行相同的 10 个固定目标，最多执行 20 推；初始状态及每推之后检查粒子 CD，严格小于 0.3 即成功并停止。全部使用已逐张量核验的官方 DINOv2 编码器和冻结的 epoch-100 Rope 预测器。

| 条件 | 预测/执行块 | 触发后的单条记忆强度系数 | GPU |
|---|---|---|---|
| Native-3 | 3 / 3 | 0 | 2 |
| 固定强度 S | 3 / 3 | κ = 2 | 3 |
| 动态强度 S | 3 / 3 | κ × w，范围 (0, 2] | 4 |
| Native-5 | 5 / 5 | 0 | 5 |

`w = clip((0.15 - r) / (0.15 - 0.05), 0, 1)`，其中 `r` 是现有三推窗口的运动比值。固定和动态版本均在 `w > 0` 时触发；固定版本将新记忆的权重设为 1，动态版本保留该连续权重。支持半径、目标距离缩放、记忆累计方式、κ、触发条件和计算预算一致。固定的是每条记忆的强度系数，多个记忆之和及候选的总惩罚仍可能变化。

预热 5 推。每推读取图像、检查成功并更新停滞记忆；已经选定的动作块继续执行，新的记忆只在下一次块规划时进入目标函数。3 推块在第 1、4、7……推前规划，因此最早在第 7 推使用 S。不是“固定版本不改块尾、动态版本重规划块尾”；两个版本仅比较强度系数。

块内调用环境的 `step_multiple` 执行单推并取回观测，保持同一物理仿真继续运行；只有块开始时使用原有 `prepare(seed, state)` 接口恢复粒子位置。20 推预算下，3 推块最多进行 7 次 CEM 优化，最后一块只执行前两推；5 推块最多进行 4 次。每次 CEM 仍为 300 个候选、30 精英、10 轮。驱动循环的 `n_taken_actions=1` 用于逐推反馈，实际规划/执行块由 `chunk_size` 决定。

这借鉴了 [ACT 实现](https://github.com/tonyzhaozh/act/blob/main/imitate_episodes.py) 中逐步接收观测、按动作块执行的组织方式。本次不加入时间集成，时间集成可作为独立后续变量。

## 调度与配对

一张 GPU 固定运行一个实验，内部按样本 0–9 顺序执行。每个 episode 有独立的模拟器生命周期，避免多个样本轮流占用同一个活动场景。与此前批量逐推评测相比，这还改变了模拟器的样本隔离方式和 CEM 随机流组织，不能把跨这两套实现的差异只归因于块长度。新增四组内部采用相同的隔离、种子与状态恢复规则；Native-3 与 Native-5 是本次块长度的参照。

每个样本的环境 seed 为 `99*i+1`，CEM seed 为 `99+i`，场景构建 seed 固定为 99。S 组等待同一样本的 Native-3 完成，核验模型、编码器、目标、代码和配置，并逐块重新计算 CEM。整块控制与 Native 相同时复用其真实执行轨迹，首次选出不同动作块后从对应块边界恢复状态并独立仿真。分别记录首次分支块和首次实际动作改变；分支后保留原环境仅恢复粒子位置的接口限制，报告不把无动作改变的仿真差异解释成惩罚收益。

## 运行

```bash
python3 experiments/rope_chunk3/launch.py \
  --root eval_outputs/rope_chunk3_strength_v2 \
  --gpu-map native3=2,fixed3=3,dynamic3=4,native5=5
```

已完整结束的样本会保留复用；未完成的输出必须先作为中断记录归档，避免覆盖原始日志。每个样本输出 `complete.json`、`encoder_audit.json`、`chunk_audit.json`、`rope_metrics.npz`、逐推触发记录和最终动作。每种条件必须凑齐预定 10 个样本后汇总，不能只看先结束的样本均值。

检查命令：

```bash
eval_outputs/rope_local/venv/bin/python -m unittest discover -s tests -p test_rope_chunks.py
```

全部 40 个 episode 结束后生成报告与图表：

```bash
eval_outputs/rope_local/venv/bin/python experiments/rope_chunk3/report.py \
  --root eval_outputs/rope_chunk3_strength_v2 \
  --output reports/rope_chunk3_strength_20261116
eval_outputs/rope_local/venv/bin/python experiments/rope_chunk3/plot.py \
  --root eval_outputs/rope_chunk3_strength_v2 \
  --report reports/rope_chunk3_strength_20261116
```

报告分别记录检测次数、正惩罚候选数、实际动作变化、首次动作分歧、成功与耗时，并检查固定/动态强度是否真的出现非饱和差异。如果两组全部 `w=1`，不能根据两次物理仿真的 CD 高低宣称某种强度策略更好。

## 15 推上限复核

Native-3、固定 S、动态 S 各独立补跑 10 个样本，仅将上限设为 15 推；其余配置保持不变。成功样本仍立即停止。

```bash
python3 experiments/rope_chunk3/launch.py \
  --root eval_outputs/rope_chunk3_cap15 \
  --gpu-map native3=2,fixed3=3,dynamic3=4 --max-pushes 15
eval_outputs/rope_local/venv/bin/python experiments/rope_chunk3/report.py \
  --root eval_outputs/rope_chunk3_cap15 --output reports/rope_chunk3_cap15_20261116 \
  --arms native3,fixed3,dynamic3 --max-pushes 15
eval_outputs/rope_local/venv/bin/python experiments/rope_chunk3/compare_budget.py \
  --reference eval_outputs/rope_chunk3_strength_v2 --rerun eval_outputs/rope_chunk3_cap15 \
  --output reports/rope_chunk3_cap15_20261116
```

预算比较同时保留原 20 推运行的前 15 推读数。只有同一条轨迹的 15／20 推差异可以隔离预算作用；独立重跑可能出现仿真波动及后续动作分歧。原 20 推运行器的源码快照保存在 `reports/rope_chunk3_strength_20261116/source/`，避免新增预算参数后丢失旧运行的实现依据。
