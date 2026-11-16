# 随机数据 LeWM 与增强数据 LeWM 的匹配训练

2026 年 11 月 16 日，按用户要求补训本机缺少的随机数据 LeWM。参考已有增强 PushObj LeWM epoch 10 的完整训练配方，从随机初始化开始训练；模型实现、训练脚本与数据加载器逐文件匹配参考提交 `d4f6cb491ac19d0224485b056fe5b24aa8e50f94`。

| 项目 | 设置 |
|---|---|
| 数据 | `dinowm_pushobj_ordinary_matched16k_v1` |
| 训练 / 验证 | 16,000 / 4,000 条轨迹，T、L、Z、+ 四种形状均衡 |
| 配对 | 源轨迹、初始位置与速度、形状、长度与增强数据匹配 |
| 数据差异 | 源动作回放不添加增强版的 9 px 接触偏置和每轴 10 px 高斯噪声 |
| 窗口 | 训练 1,630,266；验证 403,951 |
| 模型 | 18,034,478 参数，224 图像、patch 14、embedding 192、history 3、frameskip 5 |
| 训练 | seed 0，10 epochs，batch 128，每轮 12,736 次更新，共 127,360 次 |
| 优化 | AdamW，学习率 5e-5，weight decay 1e-3，梯度裁剪 1.0 |
| 调度 | 前 1% 更新线性 warmup，之后按更新步做 cosine decay |
| 精度与正则 | bf16 autocast；SIGReg 权重 0.09，17 knots，1024 projections |
| 每轮校准与验证 | BatchNorm 重估 200 batches；验证 100 batches |
| 数据加载 | 原始 WindowBatcher，rank 0 上 24 workers、512-episode pool；全局 batch 顺序不变，分发到各卡 |
| 资源 | GPU0–7；DDP，每卡 16、全局 batch 128；SyncBatchNorm 与可微全局 SIGReg |

这里的随机数据版本使用已有普通源动作回放数据，并非新生成独立均匀白噪声动作。增强数据的轨迹选择与摆放条件保持不变。数据路径和输出路径是 YAML 配置中仅有的两项改动；分布式执行设置单独记录在 manifest，见 `reference_training_config.yaml`、`config.yaml` 和 `preflight.json`。

单卡基础预检已验证 20,000 条轨迹的长度、形状、初始状态和速度逐项配对，全部视频文件存在；模型可严格加载增强版权重。独立的随机初始化模型完成 batch 128 的三次有限损失与梯度检查，显存峰值约 11.3 GiB。正式训练另起进程并重新设置 seed，不沿用预检参数，也不从增强权重初始化。

八卡预检对比了相同输入和参数下的全局损失、梯度与 BatchNorm buffers，并验证 bf16 训练、校准、验证和原始格式 checkpoint 载入。等价性测试对齐随机 dropout；正式训练恢复原始 dropout 0.1，每卡使用独立随机流。SIGReg 使用完整全局 batch 和各卡共享的投影方向，其可微汇聚的梯度与 DDP 平均共同保持原目标的尺度。

`runtime/` 保存未经修改的参考训练、模型和数据加载器，以及独立的八卡训练驱动；`manifest.json` 保存参考 checkpoint、源提交及文件校验值。`launch.py` 启动后台监督进程，`status.json`、`train.log` 和 checkpoint 目录中的 `metrics.jsonl` 记录进度。训练完成后自动校验 epoch 10、127,360 次更新、所有参数有限、严格模型载入及 latest 权重的一致性，写入 `training_result.json`。本次不启动新的规划测评。

论文与代码已同步到云端提交 `5ab1d1b` 的独立工作目录 `JEPA-Stall-cloud-20261116`。更新包括摘要与方法公式修订、可编辑 Figure 3、Rope 停滞记忆迁移试跑、DINO-WM 完整结果、阈值消融、LeWM 接触优化及 ACT／DAgger 方法报告。论文当前方法以执行停滞记忆和临时本体加权为核心；接触目标优化与持续恢复数据聚合的研究记录分别保留在实验报告中。本次补充随机数据 LeWM 权重，用于后续与增强版本进行模型与规划对照。

用户随后授权八张 GPU 均可使用。此前单卡启动段保留在相邻 `lewm_random_matched16k_20261116` 实验目录，标记为被八卡版本替代，不作为已完成模型。正式八卡版本从 seed 0 重新开始，完成完整的 10 轮、127,360 次更新。
