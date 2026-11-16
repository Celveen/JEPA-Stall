# LeWM PushT 训练与 clean CEM 评测

按用户要求重新训练独立的 T-only LeWM，以后续补测 clean CEM。Rope 全部结果已先上传至 GitHub（de78d6e），本训练在其后准备。

数据来自用户提供的 Google Drive 目录中的 `pushobj_T16k_bias_data.zip`。本地位置为 `data/pushobj_T16k_bias`。16,000 条训练轨迹、1,000 条验证轨迹，全部为 T；动作包含原数据的接触偏置与噪声，**不是纯随机动作数据**。下载程序逐段校验 HTTP 字节范围；解压前校验全部 ZIP CRC；审计检查张量有限性、全部视频文件、形状和帧数，并抽样解码每个 split 的 32 条完整长度首尾帧。详见 data_audit.json。

沿用原 T-only LeWM 配方，训练轮数改为 3。从 seed 0 随机初始化，224 图像、history 3、frameskip 5、ViT-Tiny 192；AdamW lr 5e-5、weight decay 1e-3、梯度裁剪 1、bf16；SIGReg 权重 0.09、17 knots、1024 projections。全局 batch 128，在 GPU 4、5、6、7 上每卡 32。每轮 12,759 次更新，三轮共 38,277 次；每轮末校准 BatchNorm 200 batches，验证 100 batches，再保存模型。

runtime 中的模型、单卡训练参考、数据加载器和分布式损失实现来自已归档的 LeWM 训练源码，并保留 MIT 许可证。分布式入口只将八卡改为四卡、数据窗口断言改为 T-only、限制三轮，并增加损失有限性检查。使用全局 SyncBatchNorm、可微的全局 SIGReg 和 rank 0 的原始 WindowBatcher 保持全局训练目标。upstream_source_hashes.json 记录原始来源。

preflight_distributed.py 检查四卡损失、梯度、BatchNorm buffers 与单卡全局 batch 的一致性，然后用全局 batch 128 检查混合精度前后向、校准、验证及原生 checkpoint 严格加载。测试中关闭 dropout 以比较确定性结果，正式训练恢复原始 dropout 0.1。

运行目录：`eval_outputs/lewm_pusht_tonly_e3_20261116`。后台监督器保存启动配置、源文件哈希、状态和训练日志；训练完成后验证 epoch 3、38,277 次更新、所有参数有限、严格加载及 latest 权重一致性。`checkpoints/lewm_pusht_tonly_e3_4gpu_20261116` 保存模型和 hydra.yaml。监督器不会覆盖已有训练目录，也不会在指定 GPU 已被占用时启动。

```bash
eval_outputs/rope_local/venv/bin/python experiments/lewm_pusht_tonly_e3_20261116/launch.py
cat eval_outputs/lewm_pusht_tonly_e3_20261116/status.json
```

训练与 clean CEM 均已完成，结果为 36/50（72%），见 [结果包](../../reports/lewm_pusht_clean_20261116/README.md)。论文方法标签统一写 LeWM；训练轮数保留在配置中。Clean 使用本次训练权重，六个历史扰动条件仍来自归档权重，其来源在附录及结果清单中说明。

## 训练后自动评测 clean CEM

`evaluate_clean.py` 复用主表归档的 CEM 协议：50 个固定顺序目标、seed 100、目标由存储动作重放生成；25 步物理规划时域（5 个模型步）、每次执行 5 步、最多 30 次重规划（150 步），200 候选、30 精英、10 轮 CEM、staged 目标函数。无图像扰动、无颜色覆盖，模型冻结且在前后校验所有参数和 buffers 的哈希。

成功规则为推杆与物体联合 xy 距离小于 20 px、最短角度差小于 π/9。与 Rope 的 CD 判定不同，保留 PushT 原评测口径。

使用随机模型、两个目标、一次重规划的流程检查已完成；诊断输出在 `eval_outputs/lewm_pusht_prepare/clean_pipeline_smoke`，明确排除在正式成绩之外。后台 `queue_clean.py` 已安排在训练成功并通过 checkpoint 校验后，在 GPU 4 上运行一次正式 clean 评测；若训练失败则不继续，若 GPU 4 被占用则等待。状态在 `eval_outputs/lewm_pusht_tonly_e3_clean_queue/status.json`，正式评测输出在 `eval_outputs/lewm_pusht_tonly_e3_clean50`。

## 本机运行环境

PyTorch 2.7.1+cu128。原共享 NCCL 2.26.2 在本机通信初始化长时间未完成；本实验独立安装 NCCL 2.27.7，并通过进程级 LD_PRELOAD 指定它，四卡集合运算与训练等价性检查通过。没有修改共享环境。关闭 P2P、IB、cuMem host/device 路径的设置和实际库哈希随运行 manifest 保存。

需要重新准备依赖时：

```bash
eval_outputs/rope_local/venv/bin/python -m pip install --no-deps \
  --target eval_outputs/lewm_pusht_prepare/nccl_runtime nvidia-nccl-cu12==2.27.7
```

`preflight.json` 同时记录 PyTorch 编译时 NCCL 版本与 ncclGetVersion 读取的实际运行版本。初次启动检查误把两者视为相同，已在任何训练更新发生前纠正；该次检查日志单独保留在 eval_outputs 下。
