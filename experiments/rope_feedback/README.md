# Rope 逐推反馈评测

每组固定 10 个任务，默认选择归档的 `chunk_0.pkl`。同一批目标分别运行 Native 和 +S，两组各执行最多 20 次推动。一个步长对应环境中的一次完整推动，推动结束后读取图像并重规划。初始状态及每推之后检查粒子 CD，严格小于 0.3 即成功，立即停止该样本的后续 CEM 优化和物理推动；其他样本继续到成功或 20 推。CD 只用于成功判定和统计，不进入 S 检测器或动作损失。

每次 CEM 预测未来 5 推，执行第 1 推，剩下 4 推用于下一次优化的初始均值。CEM 每次使用 300 个候选、10 轮优化、top 30。每个任务最多调用 20 次规划，相比旧协议的 3 次，优化计算预算约增加为 6.67 倍。

S 用最近 3 推的视觉潜变量位移检测停滞。预热设为 3 推，第 3 推之后首次检测，第 4 推起可以应用记忆惩罚。每次后续反馈继续更新检测与记忆。最后一推保存观测和 CD，随后终止。`gamma_lo=0.05`、`gamma_hi=0.15`、`kappa=2`、`beta=0.25`。详细配置在 `protocol.yaml`。

ACT 启用时间集成时，每个控制时刻都会根据当前观测查询动作块。本实现借鉴其中逐步更新动作块的节奏，执行新优化序列的首个动作，并以剩余序列热启动下一次 CEM。ACT 的重叠预测加权聚合仍属于后续扩展。[ACT 官方实现](https://github.com/tonyzhaozh/act/blob/main/imitate_episodes.py)

## 本机准备与运行

Rope checkpoint 不包含冻结的 DINO 编码器；运行器必须另行加载官方 DINOv2 ViT-S/14，并逐项核验 175 个张量。预训练文件位于 `eval_outputs/rope_local/torch/hub/checkpoints/dinov2_vits14_pretrain.pth`。缺失或校验失败时直接停止。

本机环境及 PyFleX 已安装在 `/data/zjc/JEPA-Stall/eval_outputs/rope_local/`，已下载的 epoch-100 权重位于 `checkpoints/rope_dino_h1_fs1_e100_seed0/`。本机启动脚本转到当前目录的 `run.py`。`--prepare` 会核验 10 个目标、权重文件和运行代码，写出准备记录。

在仓库根目录执行准备命令。

```bash
bash eval_outputs/rope_local/run_rope.sh --arm native --chunk 0 \
  --output eval_outputs/rope_feedback_pretrained_cd03/native/chunk_00 --prepare
```

运行同一批任务的两组对照。

```bash
bash eval_outputs/rope_local/run_rope.sh --arm native --chunk 0 \
  --output eval_outputs/rope_feedback_pretrained_cd03/native/chunk_00

bash eval_outputs/rope_local/run_rope.sh --arm S --chunk 0 \
  --reference eval_outputs/rope_feedback_pretrained_cd03/native/chunk_00 \
  --output eval_outputs/rope_feedback_pretrained_cd03/S/chunk_00
```

Native 完成后，+S 会核对任务、模型、采样器、种子和逐推协议。相同动作继续共享原生执行结果，动作首次改变后分支仿真。环境沿用归档的 `rollout` 接口，每次调用恢复粒子位置，这个调用边界现在发生在每次推动前。

主指标为停止时的粒子 Chamfer Distance，同时报告 `CD < 0.3` 成功率和实际推动次数。`rope_metrics.npz` 保存最多 `10×20` 的 CD 记录、`initial_cd`、`final_cd`、`successes`、`actual_pushes` 和 `executed_mask`。已成功样本的后续数组列重复终止状态，动作列只是填充，必须使用 `executed_mask` 排除；它们不会再进入仿真。所有样本均成功时整批立即结束。CEM 为已停止样本保留随机数位置但跳过模型预测和候选优化，保证其他样本的配对随机序列一致。

`ecp_records.json` 保存逐推的运动比值、触发权重和记忆数量。`ecp_costs.json` 保存候选中非零正则项的次数和最大值，`first_action_change` 保存首次改变动作的位置。旧报告的 50 个任务曾有 10 个触发、0 个动作改变，但 2026-11-16 核查发现其编码器加载错误，不能作为正确 DINO-WM 的结果；详见 `reports/rope_encoder_audit_20261116/DIAGNOSIS.md`。本次重新核验这三层行为。

两组完成后生成 10 样本配对报告（包含实际物理动作差异）。

```bash
eval_outputs/rope_local/venv/bin/python experiments/rope_feedback/compare.py   --root eval_outputs/rope_feedback_pretrained_cd03   --report reports/rope_feedback_20261116
```

## 5 推预热对照

`run_warmup5.py` 保留相同的三推检测窗口，将预热改为 5 推：前 5 推没有 S 干涉，第 5 推反馈后允许写入记忆，第 6 推起参与动作优化。该实验复用 `kappa=0` 的同一 Native 基线，避免再次仿真基线；校验仅允许这一处预热差异，并把变体入口文件哈希记入运行配置。其他参数、目标和权重仍严格一致。

本机用 GPU 1 运行新增的 +S；`--wait-for-reference` 使它等待 Native 的 `complete.json`，随后与 GPU 0 上原来的 +S 并行运行。

```bash
bash experiments/rope_feedback/run_warmup5.sh --arm S --chunk 0 \
  --reference eval_outputs/rope_feedback_pretrained_cd03/native/chunk_00 \
  --output eval_outputs/rope_feedback_pretrained_warmup5/S/chunk_00 \
  --wait-for-reference
```

两种 +S 的结果必须分别与同一 Native 比较。若第 3、4 推反馈时本来没有触发，两种预热可能产生完全相同的干涉轨迹。

## 代码验证

CPU 回归检查覆盖逐推反馈、热启动移位、三推窗口、配对索引、成功阈值严格边界、部分/全部样本提前成功、停止后不再仿真，以及 CEM 跳过成功样本时其他样本候选不变。

```bash
OMP_NUM_THREADS=4 eval_outputs/rope_local/venv/bin/python tests/test_rope_feedback.py
```
