# 在另一台服务器上复现

截至 2026-10-30。代码和报告都在本仓库；权重、数据和运行输出不在 git 里，按下面准备。各实验的结论和数字见 `reports/`：
- E3、PushObj：[reports/e3/report.md](reports/e3/report.md)
- E2：[reports/e2/report.md](reports/e2/report.md)
- E1：[reports/e1_switch/report.md](reports/e1_switch/report.md)
- 方案和叙事：[reports/paper2_positioning/](reports/paper2_positioning/)

## 1. 环境

```bash
conda env create -f environment.yaml && conda activate ts    # Python 3.9.19
```

- **迷宫需要 MuJoCo 2.1**：`LD_LIBRARY_PATH` 要包含 `~/.mujoco/mujoco210/bin` 和 `/usr/lib/nvidia`。`scripts/e3_run.py` 和 `scripts/stag_maze_*.py` 会自己加。PushObj 和 Wall 不需要。
- **迷宫每张卡最多 2 个运行**，再多会建不起离屏渲染（"Offscreen framebuffer is not complete"）。
- **DINOv2**：torch.hub 固定在 2025-05-19 的 dinov2 版本（`models/dino.py` 的 `DINOV2_HUB`），因为最新版在 Python 3.9 上会报错。第一次加载要联网。

## 2. 不在 git 里的文件

| 放到 | 来源 | 大小 | sha256 |
|---|---|---:|---|
| `checkpoints/diversemaze_layout_shift/` | AdaJEPA 公开发布（CL-JEPA 仓库 README 里的 Google Drive，`checkpoints/diversemaze_layout_shift.zip`） | 412 MB | `950008c61d6725611b0f05ef523ae9f5cb854ca97706f13cfde4199ac779fe2b` |
| `data/diverse_maze/` | 同上，`data/diverse_maze.zip` | 0.6 MB | `e2d9fdcdf819d8d655ef623bfb48ed2988dfbfc13fe41910688eb92da885c120` |
| `checkpoints/pushobj_bias_16k_e3_seed0/` | 我们自己训练的 PushObj 模型，**没有公开**，从原服务器拷 `pushobj_bias_16k_e3.zip` | 380 MB | `295770b49f9ce142c6a7622b7b3f10ac49c85bc1b99985f56854811ef062b192` |
| `data/targets/` | 已在 git 里（PushObj 的评测目标） | — | — |
| `data/_dl_dinowm/wall_single.zip` | DINO-WM 的 Wall 数据（OSF），`python scripts/fetch_assets.py dinowm_wall` | 1.67 GB | 脚本自动核对 |
| `data/_dl_dinowm_ckpt/outputs.zip` | DINO-WM 公开的世界模型（含 Wall），`python scripts/fetch_assets.py dinowm_ckpt` | 0.95 GB | 脚本自动核对 |
| `~/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth` | DINOv2 权重，`python scripts/fetch_assets.py dinov2_vits14`（或首次加载时自动下载） | 88 MB | — |

- **解压方式**：权重的 zip 解压后是 `<name>/hydra.yaml` + `<name>/checkpoints/model_<epoch>.pth`，放进 `checkpoints/`。
- **OSF 下载很慢**：每个连接只有约 0.1–1 MB/s，而且每次连接要 70–110 秒握手。`scripts/fetch_assets.py` 会把文件切成几段，同时开多个连接下载，可以断点续传，下完核对 sha256。中断了重跑同一条命令即可接着下。
- **能直接从原服务器拷就拷**（rsync / scp），比 OSF 快得多。

## 3. 按顺序复现

所有命令都在仓库根目录、`ts` 环境下运行。GPU 用 `--gpus` 指定。

**可复现性**：各脚本的 `--check` 核对的是"同一台机器上，切换之前和参照运行逐位相同"。换了 GPU、驱动或 CUDA 版本，数字可能有细小差别，所以要先在新服务器上重跑参照，再跑各组，再拿新的参照核对。

### 3.1 迷宫 medium：参照运行和陷阱分类（E3、E2 都依赖）

```bash
# AdaJEPA 参照：修正时长（每个动作 4 个环境步），预算 50，记录卡住触发（kappa 0），种子 100
python scripts/stag_maze_base_run.py --phase ref --levels medium --bases adajepa --max-iter 50 --per-shard 25 \
    --root eval_outputs/stag_maze100_ar4 --extra ++env_kwargs_override.action_repeat=4 --gpus 0,1 --jobs 4
# 冻结规划器的同样设定（对照）
python scripts/stag_maze_base_run.py --phase ref --levels medium --bases frozen --max-iter 50 --per-shard 25 \
    --root eval_outputs/stag_maze100_ar4 --extra ++env_kwargs_override.action_repeat=4 --gpus 0,1 --jobs 4
# E0 陷阱分类（e3_analyze 读 medium_geom.jsonl）
python scripts/stag_maze_trap.py --stages geom sim latent summary --level medium --root eval_outputs/stag_maze100_ar4 \
    --action-repeat 4 --out eval_outputs/stag_maze_trap_ar4
```

### 3.2 E1：切换信号（离线，只读日志）

```bash
python scripts/e1_switch.py                                                          # 迷宫 medium + PushObj S7（30 次）
python scripts/e1_switch.py --push-ext eval_outputs/stag_S7_it50/S0 --push-budget 50 --out reports/e1_switch/s7_it50   # 需先完成 3.4
```

### 3.3 E3：迷宫 medium 上修尺子的时机（第一、二阶段）和 E2：冻结模型

```bash
python scripts/e3_run.py --arms R-trig R-switch --gpus 0,1,4,5,6,7 --jobs 12                   # 第一阶段
python scripts/e3_run.py --arms R-delay SZ-switch P-switch --gpus 0,1,4,5,6,7 --jobs 12        # 第二阶段
python scripts/e3_run.py --arms F-trig F-switch --gpus 0,1,4,5,6,7 --jobs 12                   # E2
python scripts/e3_analyze.py --check --arms R-trig R-switch R-delay SZ-switch P-switch F-trig F-switch   # 切换前逐位相同
python scripts/e3_analyze.py --arms R-trig R-switch --name stage1
python scripts/e3_analyze.py --arms R-trig R-switch R-delay SZ-switch P-switch --name stage2
python scripts/e3_analyze.py --arms R-switch F-trig F-switch --name e2
```

冒烟测试（R-trig 必须和旧的停滞规划器实现完全一致）：

```bash
python scripts/e3_run.py --arms R-trig T0SP-stag --smoke 2 --root eval_outputs/e3_smoke
python scripts/e3_analyze.py --check --arms R-trig T0SP-stag --root eval_outputs/e3_smoke --same R-trig T0SP-stag
```

### 3.4 PushObj S7（我们的权重）：30 次参照、补跑到 50 次、两组 T0SP

```bash
python scripts/escape_run.py --stream S7 --arms none --gpus 4,7 --jobs 6                     # AdaJEPA 参照，30 次
python scripts/stag_latent_trace.py --ref eval_outputs/escape_S7/none --stream S7 --out eval_outputs/stag_S7/trace.jsonl
python scripts/s7_extend.py --select                                                         # 30 次内没成功的 63 个
python scripts/stag_run.py --stream S7 --arms S0 --root eval_outputs/stag_S7_it50 --extra planner.max_iter=50 --gpus 4,7 --jobs 6
python scripts/s7_extend.py --check                                                          # 前 30 次逐位相同
python scripts/e3_run.py --env pushobj --arms R-trig R-switch --gpus 0,1,4,5,6,7 --jobs 12
python scripts/e3_analyze.py --env pushobj --check --arms R-trig R-switch
python scripts/e3_analyze.py --env pushobj --arms R-trig R-switch --name stage1
```

### 3.5 Wall G0（已完成，结果见 [Wall 报告](reports/wall_dinowm_20261103/report.md)）

```bash
python scripts/fetch_assets.py dinowm_wall dinowm_ckpt dinov2_vits14 --conn 8 --pieces 8
python scripts/wall_eval.py prepare
# 每次选择空闲 GPU 和全新的输出目录；同时运行开环与最多 10 次规划的 MPC
python scripts/wall_eval.py run --gpus 0,1 --output eval_outputs/wall_dinowm/new_run
python scripts/wall_eval.py summarize --output eval_outputs/wall_dinowm/new_run
```

原始运行的逐任务结果和配置已归档在 `reports/wall_dinowm_20261103/`。`prepare` 保留完整归一化张量，只解包固定验证集图像；该目录不能直接用于训练。GPU 作业须在能访问 NVIDIA 设备的执行环境运行。

G1、WallRandom 及正则/判别器实验尚未启动；下一轮见 [待确认方案](reports/paper2_positioning/wall_stall_proposal.md)。

### 3.6 切换参数闭环开发实验（已完成）

复现入口为 `scripts/switch_calibration.py` 的 `replay`、`prepare`、`run`、`analyze` 子命令；详细依赖、命令及配对检查见 [协议](reports/switch_calibration_20261102/protocol.md)。本地原始轨迹就绪后可重新分析，最终结果见 [报告](reports/switch_calibration_20261102/results.md)。

## 4. 各分析读哪些运行

| 分析 | 读的运行 |
|---|---|
| E3 / E2（迷宫） | `eval_outputs/stag_maze100_ar4/medium/adajepa/ref`、`eval_outputs/e3/medium_s100/<组>`、`eval_outputs/stag_maze_trap_ar4/medium_geom.jsonl` |
| E3（PushObj） | `eval_outputs/escape_S7/none` + `eval_outputs/stag_S7_it50/S0`、`eval_outputs/e3/pushobj_S7/<组>` |
| E1 | 迷宫参照、`eval_outputs/escape_S7/none`、`eval_outputs/stag_S7/trace.jsonl`，以及（50 次时）`eval_outputs/stag_S7_it50/S0` |
