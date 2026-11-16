# Rope 真实仿真截图

**本目录是独立可视化复跑，不计入正式 20 样本结果。** 原正式运行仅 Native 将逐推画面存盘，S 组没有存盘；因此没有将复跑画面冒充原正式轨迹。

画面直接取自本次实际执行时保存的仿真 RGB 观测，不重放动作、不使用世界模型预测画面。浮点 RGB 转 PNG 前逐像素检查为无损转换。图中 CD 为复跑自身的粒子状态指标。

Ours 指定为 `fixed3`：3/3、固定 S、预热 5 推。对照为 DINO-WM Native 5/5。目标、模型、种子和控制配置与正式运行一致；物理仿真存在波动，复跑指标可能不同。

## 预先选定的两对案例

开始画面记录前选定案例，每个目标的两组各独立执行一次，保留全部结果。选例规则：Preselected formal samples 05 and 14: fixed S succeeds where Native5 fails and S is applied. Each arm runs once per target. Display a genuinely improved paired visualization if present; retain all outcomes. These runs are excluded from formal statistics.

| 原目标 | Native CD / 推数 | Ours CD / 推数 | Ours S 触发 | Native CD − Ours CD |
|---|---:|---:|---:|---:|
| 15 | 0.235933 / 15 | 0.169454 / 18 | 3 | 0.066478 |
| 24 | 0.218838 / 11 | 0.147000 / 7 | 0 | 0.071838 |

## 图与原始材料

- 目标 15：[最终状态对比](sample_05/final_comparison.png) · [每三推完整序列](sample_05/trajectory_every3.png) · [每一步原始截图](raw_frames/sample_05/)
- 目标 24：[最终状态对比](sample_14/final_comparison.png) · [每三推完整序列](sample_14/trajectory_every3.png) · [每一步原始截图](raw_frames/sample_14/)

成功后停止的时间段明确留空；实际最后一推单列，即使它不是 3 的倍数。最终对比图上排为完整原图，下排采用四幅图相同的裁剪范围放大，未叠加目标、未修改几何形状。

`raw_frames/` 保存这四次执行的每一步原图。`cd_by_push.csv` 保存逐推 CD；`provenance.json` 保存原缓存、指标、目标、PNG 哈希与选例规则。`formal_vs_visualization.json` 核验两次运行目标、模型、规划参数及种子相同，并并列保存各自成绩。PNG/PDF 图均由仓库中的 visualize.py 生成。

布局参考内部 SE-JEPA 仓库的每三推图示。这里只保留内部来源说明，不向当前论文添加未发表论文的引用。

案例用于说明可观测的行为差异，不代表普遍收益；Native-5 与 S-3 同时改变动作块和 S，不是 S 的单因素消融。完整统计请见 [正式结果](../RESULTS.md)。

## 本次新增案例的解读

两对均保留全部运行结果，并非失败转成功案例：Native 与 Ours 都达到 CD < 0.3。

- 目标 15：Ours CD 0.1695，低于 Native 的 0.2359；S 触发 3 次，但用了 18 推，Native 为 15 推。该例展示更低的最终误差，并未减少动作数。
- 目标 24：Ours CD 0.1470，低于 Native 的 0.2188，推数为 7 对 11；本次 S 未触发，不能把收益归因于 S。

两个目标是从原正式运行中预选的有利案例，图示不代表总体收益。目标 24 在原正式运行中触发过 S，本次独立复跑没有触发；正式指标与复跑指标在 formal_vs_visualization.json 中分列。完整五组统计见 [最新正式结果](../../rope_formal20_native3/RESULTS.md)。

## 原始证据与重建

[证据包](evidence.tar.gz) 保存全部四次执行缓存、指标、动作、目标、配置及运行记录；[逐文件哈希](evidence_manifest.json) 可逐项核对。入口相对原正式评测仅新增 native3 命令行选项，未改规划逻辑；新旧源文件随包保存，导图时严格校验此唯一差异以及其他实现哈希。

从仓库根目录执行（复用同一 Python 环境）：

```bash
mkdir -p /tmp/rope_extra_evidence
tar -xzf reports/rope_formal20/evidence.tar.gz -C /tmp/rope_extra_evidence
tar -xzf reports/rope_formal20/visuals_extra/evidence.tar.gz -C /tmp/rope_extra_evidence
eval_outputs/rope_local/venv/bin/python experiments/rope_formal20/visualize.py \
  --root /tmp/rope_extra_evidence/qualitative_extra \
  --formal-root /tmp/rope_extra_evidence/formal \
  --worker-compatibility /tmp/rope_extra_evidence/source_snapshot \
  --output /tmp/rope_extra_rebuilt
```

已从发布证据包独立重建全部对比 PNG、序列 PNG 和原始截图，并检查与本目录逐字节一致。
