# 第二篇论文定位：规划距离相关的竞品对比

截至 2026-10-26。共 15 篇论文，另把 AdaJEPA 列作参照。PDF 用 `curl https://arxiv.org/pdf/<id>` 下载后转成文本阅读。阅读深度：TEMPO 和 TRM 读了方法、实验和相关附录；其余论文读了摘要、引言和方法要点，并在全文检索 test-time / online / unseen / held-out / shift / adapt 等关键词，相关段落都回原文核对过。

要回答的两个问题：
1. 有没有人在**测试时**修正规划用的距离？
2. 这些论文有没有在**分布偏移**下评测？

## 0. 结论

1. **"欧氏潜距离 ≠ 可达性/剩余步数"这个问题本身已经被做得很充分。** 2026 年至少 13 篇论文在做，做法分三类：
   - 训练时改表示：TS、TD-JEPA、RC-aux、DA-LeWM、Monotone Cost、SLIM、LRC-JEPA、Value-guided JEPA；
   - 在冻结的潜空间上后验学一个代价：TRM、TEMPO、RP1、The Objective Is the Bottleneck、Latent Projection；
   - 只做诊断：ARC-Bench、Decision-Metric Alignment、The Objective Is the Bottleneck。
2. **在 MPC 循环里、单个 episode 内、在偏移环境中修正距离，没有人做。** 最接近的是 TEMPO（2610.04988）：
   - 它的距离 F_φ 先离线训练，再进行 8 轮"在线轮次"：用规划器自己收集的 episode 更新 F_φ。
   - 正式评测时 F_φ **冻结**（附录 A："Evaluation: Fϕ frozen"）。
   - 偏移实验里 F_φ 在哪里训练，**原文前后矛盾**。正文和附录 G 的正文写"在训练环境训练，评测时固定"；附录表 6/7/8 的标题写"Fϕ trained online in the corrupted / shifted environment"。不论按哪种读法，更新都是**评测开始前在环境层面**做的，不是在评测 episode 内部做的。
   - TEMPO 在 Future work 里明确写了"continuing to update it with new experience could enable progressive improvement"。**这个空档确实存在，但已经被点名，可能很快会被人占掉。**
3. **在分布偏移下评测的只有 3 篇，而且离线学到的距离在偏移下明显变弱。** 这一点支持我们的论点：
   - **TEMPO**：分布内 TwoRoom 三段规划从 36% 提到 99%。偏移下增益小得多：PushObj 没见过的形状（LeWM，1 段）53.6 → 54.8；WallRandom 没见过的布局 67.5 → 76.9；穿墙测试 15.1 → 22.9（PLDM 6.0 → 6.5）；门移位（3 段）9.9 → 30.1。
   - **TRM**：门沿同一面墙移动时部分迁移；**墙旋转方向后，排序和成功率都崩溃**（作者自己写成边界情况）。
   - **D-JEPA**：在 PushObj 没见过的形状和外观变化上评测，但监督来自训练时在模拟器里执行候选动作得到的成功/失败标签，部署时不更新。
   - 另外，The Objective Is the Bottleneck 发现：学到的代价在作者发布的权重上反而输给线性代价（34% 对 70%），原因是"在编码帧上拟合，却在预测帧上使用"。这和我们 F1 的发现一致：没见过的形状上，换成预测潜变量后 Spearman 降到 0.43。
4. **"小智能体在潜空间里几乎看不见"已经被人写出来了，不适合单独作为卖点。**
   - TRM 原文："The XY rowspace contributes less than 1% of terminal latent MSE but carries most of the information needed for control"。
   - SLIM（2610.03137）专门研究"画面里只有很小一部分在动"：潜变量几乎对动作不敏感，加一个逆动力学辅助损失后恢复。
   - LRC-JEPA 研究可控状态被背景内容挤占。
   - 它可以放在我们的分析里，作为"距离为什么在偏移下坏"的机制解释之一。
5. **AdaJEPA 修正的是动力学，代价仍然是欧氏距离。** TEMPO 和 ARC-Bench 都把它归为"改模型"一类。目前没有论文把"测试时修正动力学"和"测试时修正距离"结合起来。

## 1. 对比表

"测试时修正距离"指：在评测 episode 内部，用部署时得到的经验更新规划所用的距离/代价。

| 论文 | 距离/代价从哪来 | 什么时候学 | 测试时修正距离？ | 分布偏移评测 | 环境 | 世界模型 |
|---|---|---|---|---|---|---|
| [Temporal Straightening](https://arxiv.org/abs/2603.12231)（ICML 2026，我们的上游） | 曲率正则把潜轨迹拉直，让欧氏距离接近测地距离 | 训练时 | 否 | 无 | Wall、UMaze、Medium maze、PushT | JEPA（DINO 投影） |
| [Value-guided JEPA](https://arxiv.org/abs/2601.00844)（workshop） | 塑造表示，使（准）距离 ≈ IQL 目标条件价值 | 训练时 | 否 | 未见 | 简单控制任务（WS/WB/Maze） | JEPA |
| [RC-aux](https://arxiv.org/abs/2605.07278) | 多步开环预测 + 按预算的可达性辅助头；可达性感知的规划器 | 训练时（续训或从头训） | 否 | 无 | TwoRoom、Reacher、PushT、Cube、Wall、LIBERO-Goal | LeWM |
| [TRM](https://arxiv.org/abs/2605.22164) | 冻结潜空间上，从日志轨迹学成对的时间代价 | 离线后验 | 否 | **部分**：门移位部分迁移；**墙旋转后崩溃** | TwoRoom、TopoNav、PushT | LeWM、PLDM |
| [Temporal-Distance-JEPA](https://arxiv.org/abs/2607.25337) | 从轨迹挖掘有向时间代价，并用它重塑表示 | 训练时 | 否 | 无（只有 held-out 示范对） | TwoRoom、PushT、OGB-Cube | LeWM |
| [RP1](https://arxiv.org/abs/2608.18669) | TD 学 quasimetric 评价器 + 学一个计划更新算子 | 离线 | 否 | 无（只有 held-out episode） | TwoRoom、Reacher、Cube | LeWM、PLDM |
| [Monotone Planning Costs](https://arxiv.org/abs/2608.09073) | 自回归 rollout 损失 + 单调代价排序损失 | 训练时 | 否 | 真实机器人零样本，在没见过的场景定性展示 | GNM 导航 | DINO 编码器 |
| [Decision-Metric Alignment](https://arxiv.org/abs/2608.18746) | 诊断指标（Plan-Real Spearman）+ 逆动力学/目标动作辅助头 | 训练时 | 否 | 无 | PushT、TwoRoom 等 | LeWM |
| [The Objective Is the Bottleneck](https://arxiv.org/abs/2608.12959) | 诊断 + 从帧间隔学一个代价头，不重训模型 | 离线后验 | 否 | 无（只有 TwoRoom） | TwoRoom | LeWM |
| [ARC-Bench](https://arxiv.org/abs/2609.05461) | 只做审计：固定候选集上测潜距离排序对不对 | — | 否 | 无 | PushT、MetaWorld reach-wall、PointMaze、Wall | 官方 JEPA-WM（DINOv2、V-JEPA） |
| [D-JEPA](https://arxiv.org/abs/2609.24749) | 用模拟器执行候选动作得到的成败标签，学候选之间的排序修正；预测器尾部微调 | 训练时（带执行标签） | 否 | **有**：PushObj 没见过的形状、外观变化 | PushT、PushObj、Reacher、TwoRoom、Cube、真实机器人、驾驶 | LeWM、TD-JEPA 等 |
| [LRC-JEPA](https://arxiv.org/abs/2609.34375) | 结构上把动力学潜变量和背景内容分开 | 训练时 | 否 | 只有无关视觉扰动和 PCA 压缩的分析 | 4 个仿真环境、Bridge-v2 | 自研 |
| [SLIM](https://arxiv.org/abs/2610.03137) | 逆动力学辅助损失，让潜变量对小物体敏感 | 训练时 | 否 | 只有语言上的未见名词，不是环境偏移 | SLIM、PushT | LeWM |
| [**TEMPO**](https://arxiv.org/abs/2610.04988) | 冻结潜空间上学时间距离映射 F_φ，和 L2 各占一半 | 离线 + 8 轮在线（用规划器自己的 episode） | **部分**：有在线轮次，但评测时冻结；偏移下在哪训练原文矛盾；不在 episode 内更新 | **有**：WallRandom 新布局、穿墙、PushObj 新形状、门移位、模糊/颜色抖动、更大的智能体/更重的手臂 | 11 个环境（TwoRoom、PointMaze、PushT、PushObj、Cube 等） | LeWM、PLDM |
| [Latent Projection](https://arxiv.org/abs/2610.26979) | 拟合线性动力学 A，把 L2 限制在不确定性最小的子空间 | 离线后验 | 否 | 未见 | 6 个环境（TwoRoom、PointMaze 等） | LeWM |
| 参照：[AdaJEPA](https://arxiv.org/abs/2606.32026) | 代价仍是欧氏距离；在 MPC 循环里修正**动力学** | 测试时（模型） | 否（修的是模型） | 有 | PushObj、PushT、迷宫等 | TS 系列 |

## 2. 对我们叙事的含义

**还能站住的差异点**（都需要实验支撑）：

1. **修正发生在 MPC 循环内、单个 episode 内，而且是在偏移环境里**，用智能体自己执行得到的转移做监督。TEMPO 的在线轮次在评测前、按环境整体进行，评测时冻结。
2. **监督信号来自执行反馈，不需要标签。** 尤其是卡住：它说明"欧氏距离上近，实际走不到"，可以当作负样本或约束。TEMPO 只用同一 episode 里的帧间隔做回归。
3. **和动力学修正（AdaJEPA）结合。** 目前没人同时修正模型和距离。我们的迷宫诊断说明，冻结模型的卡住首先来自动力学误差，所以两者需要一起修正。

**评审最可能问的问题："这不就是在测试时继续跑 TEMPO 的在线更新吗？"** 我们需要的贡献要超出"更新时间点不同"，比如：
- episode 内的样本极少，怎么让更新有效（卡住带来的负样本或约束、quasimetric 结构）；
- 距离修正和动力学修正之间怎么互相影响；
- 在偏移下，离线距离到底在哪里失效（机制分析，小智能体被忽略是其中一条）。

**必须比较的基线**：TEMPO（离线版，以及"在偏移环境里做在线轮次"的版本）、TRM、AdaJEPA，以及 AdaJEPA + TEMPO 的简单叠加。

**风险**：
- TEMPO 已经点名了这个方向；
- 我们之前学的 cost-to-go V 没通过检查，需要重新设计；
- PushObj 上卡住后剩余预算不够（有特权的求解器也只能解 7/71）。在单个 episode 内修正可能来得太晚，迷宫上的情况还不知道。

## 3. 附：TEMPO 在偏移下的数字（原文表 3、表 6、表 8）

| 设置 | LeWM | + TEMPO | PLDM | + TEMPO |
|---|---:|---:|---:|---:|
| 分布内 TwoRoom，3 段 | 36.3 | 99.2 | 70.5 | 95.7 |
| WallRandom 新布局，1 段 | 67.5 | 76.9 | 75.6 | 77.9 |
| WallRandom 穿墙 | 15.1 | 22.9 | 6.0 | 6.5 |
| PushObj 新形状，1 段 | 53.6 | 54.8 | 44.1 | 48.8 |
| PushObj 新形状，3 段 | 18.8 | 20.5 | 11.1 | 15.3 |
| TwoRoom 门移位，3 段 | 9.9 | 30.1 | 18.9 | 30.4 |
| 颜色抖动，5 个环境平均（3 段） | 33.0 | 42.8 | — | — |

"段"指目标离起点有几个 25 步的规划长度。
