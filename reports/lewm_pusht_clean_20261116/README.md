# LeWM PushT clean CEM 结果

**训练与测评均已完成，clean 成功率为 72%（36/50）。** 主表及 Overleaf 版本的 LeWM / CEM / Clean 单元格已填写为 72，方法标签保持 LeWM。

| 项目 | 结果 |
|---|---:|
| 成功 / 失败 | 36 / 14 |
| 训练时间（含保存及校验） | 74.1 分钟 |
| Clean 测评时间 | 16.0 分钟 |
| 最后验证总损失 | 0.20756 |
| 最后验证预测损失 | 0.01642 |
| 参数更新次数 | 38,277 |

[逐样本成绩](per_sample.csv) · [汇总及验证损失](summary.json) · [原始证据包](evidence.tar.gz) · [证据哈希](evidence_manifest.json)

## 训练与测评设置

用户提供的 T-only 数据集经校验包含 16,000 条训练轨迹与 1,000 条验证轨迹；数据含原配方的接触偏置和动作噪声。GPU 4–7 训练 3 轮，全局 batch 128、每卡 32，seed 0 随机初始化；AdamW 学习率 5e-5，SIGReg 权重 0.09。模型、归一化、窗口及优化配方见 [训练配置](training_config.yaml) 和 [运行代码](../../experiments/lewm_pusht_tonly_e3_20261116/README.md)。

Clean 评测使用固定文件顺序的 50 个目标，seed 100，目标由存储动作重放生成。25 个物理步的预测时域、每次执行 5 步、最多 30 次重规划，总预算 150 步；CEM 200 候选、30 精英、10 轮，staged 目标函数，不使用图像扰动或颜色覆盖。成功要求推杆与物体联合 xy 误差 <20 px 且物体最短角度误差 <π/9。具体见 [评测配置](evaluation_config.yaml)。

## 核验与来源

- 从原始 NPZ 重新计算 50 个 successes，得到 36/50；各成功时点均在 30 次重规划预算内。
- 50 对初始状态和目标状态与主表原 LeWM 任务逐元素完全一致。
- 训练得到的 checkpoint 文件哈希与测评加载记录一致，所有权重有限；测评前后参数和 buffers 哈希一致。
- 四卡与单卡参考梯度相对误差约 8×10⁻⁶；混合精度前后向、BatchNorm 校准及严格加载均通过。
- 仅真实完成的 50 目标测评计入结果，随机模型的流程检查不计入。

Clean 使用本次训练的 T-only 权重；六个历史扰动条件保留原归档权重，未在此次重跑。因而该行不用于推断同一 checkpoint 从 clean 到扰动的性能降幅。轮数和两份权重的身份保留在复现记录中，不在方法标签额外加版本后缀。六扰动均值仍为 40.3%，其定义不含 clean；DDP-WM clean 仍缺测。

[训练权重收据](checkpoint_receipt.json) · [评测收据](evaluation_receipt.json) · [数据核验](data_audit.json) · [四卡预检](preflight.json)

本地训练权重在 `checkpoints/lewm_pusht_tonly_e3_4gpu_20261116/checkpoints/model_3.pth`，SHA256 为 `9e2b8afee10e46f3b7ae2bd5a9b67ebc28e86af0ccc0d0d293e2fd77471d2576`。证据包包含全部实际动作、目标观测、状态、结果、训练指标、配置、冻结检查与执行源码快照，不包含大体积训练数据和模型二进制。

解压 evidence.tar.gz 后，`evaluation/final_eval_results.npz` 中 `successes.sum()` 为 36、长度为 50。`training/metrics.jsonl` 保存全部训练记录，`implementation/` 保存实验实现；evidence_manifest.json 对所有成员逐文件记录 SHA256。
