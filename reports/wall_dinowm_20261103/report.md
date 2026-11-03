# Wall：DINO-WM 公开权重基线结果

2026-11-03；两组均已完成并正常退出。这里只报告冻结模型基线，未运行 AdaJEPA、判别器或停滞正则的 Wall 实验。

## 结果

| 方法 / 预算 | 环境步数上限 | 成功任务 | 成功率 |
|---|---:|---:|---:|
| 开环 CEM | 25 | 26/50 | 52% |
| MPC：2 次规划 | 50 | 44/50 | 88% |
| MPC：3 次规划 | 75 | 46/50 | 92% |
| MPC：10 次规划 | 250 | 46/50 | 92% |

MPC 从第 3 次到第 10 次规划没有增加成功任务。失败任务的零起始索引为 **21、24、42、44**。这说明在当前设置下继续重复原有规划没有解开这 4 个任务；仅凭成功率平台不能断言失败是代价陷阱，也不能断言自适应已经学不动，因为这一轮模型根本没有更新。

开环与 MPC 的动作预算不同，不能把 52% → 92% 当成新方法在同预算下的提升。已核对两组起点、目标及第一段动作完全相同。当前结果验证了本仓库的运行流程；并未据此宣布达到原论文复现门槛。开环仍低于原方案记录的参考成绩，配置、公开权重与论文实验的差异有待进一步核对。

## 固定协议

- 官方 `outputs.zip` 中的 `wall_single` 权重；checkpoint epoch **65**；冻结 DINOv2 ViT-S/14 patch 编码器及预测器，未在线更新。
- 固定 50 个任务、seed 99，任务随机种子为 `99 * sample_idx + 1`；`random_state` 在墙两侧抽样起点与目标。
- 训练元数据中的 frameskip = 5；CEM 300 个候选、topk 30、最多 10 轮优化。全部候选任务成功时沿用原实现提前结束优化。
- 规划 horizon = 25 个环境步（5 个模型步），每次执行 25 个环境步。MPC 上限在开始前固定为 10 次，最多 250 个环境步；成功阈值为位置距离 < 4.5。
- `objective.mode=last`、alpha = 1；禁用解码可视化；完整配置见两个 `*_config.yaml`。
- 全部 1920 条轨迹的状态、动作和布局张量参与归一化，仅解包固定验证划分中的 192 条图像轨迹。`wall_single_eval` 是评测目录，不是完整训练集。
- 两张空闲 RTX 3090；开环约 12 分钟，MPC 约 2 小时 4 分钟；2026-11-03 14:58:33 +08:00 全部结束。

## 归档与复现

- `summary.json`：最终成功数、完整重规划曲线和配对检查。
- `episodes.json`：每个任务的起点、目标、种子、成功状态和首次成功时间。失败的成功时间记为 null。
- `open_loop_results.npz` / `mpc10_results.npz`：原始逐任务输出，含全部动作；失败的原始 `action_len` 保持 np.inf。
- 两个 `*_metrics.jsonl` / `*_config.yaml`：原始指标和 Hydra 配置；绝对路径记录原执行位置，迁移时通过脚本替换。
- `run_manifest.json`：原始作业命令、返回码、权重哈希与执行时间；`artifact_hashes.json` 核对结果文件和运行代码。
- `dataset_manifest.json`：数据/权重归档哈希和验证划分；`target_audit.json`：相同观测和墙两侧目标的检查。

在仓库根目录、`ts` 环境内：

```bash
python scripts/wall_eval.py prepare
python scripts/wall_eval.py run --gpus 0,1 --output eval_outputs/wall_dinowm/new_run
python scripts/wall_eval.py summarize --output eval_outputs/wall_dinowm/new_run
```

运行前查询实际 GPU 占用。原始数据、权重和控制台长日志留在本地，提交的结果文件可独立核算成功数。后续方法实验方案见 [下一轮方案](../paper2_positioning/wall_stall_proposal.md)，**等待用户确认，尚未启动**。
