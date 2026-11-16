# Rope 正式评测与真实仿真图

本包包含四组各 20 个目标的正式评测，以及另外两对独立可视化运行。所有运行最多 20 推，每推观测，CD < 0.3 即停止规划和实际推动。正式目标按原池 chunk 1、2 的固定顺序选择，未参加此前 chunk 0 的 10 样本试跑。

- [正式结果与动作变化](RESULTS.md)：四组完整成绩。
- [结果解读与运行耗时](INTERPRETATION.md)：实际触发、强度差异、反例及证据边界。
- [逐样本成绩](per_sample.csv)、[实际执行动作比较](action_comparisons.csv)、[汇总 JSON](summary.json)。
- [全部目标散点与成功曲线](results.png)（[PDF](results.pdf)）。
- [真实截图与完整运行序列](visuals/README.md)：两对案例，每三推一帧、实际终点、目标，以及每一步原图；包含改善和退步的案例。
- [原始证据包](evidence.tar.gz)、[逐文件哈希](evidence_manifest.json)。
- [协议、配置和运行方式](../../experiments/rope_formal20/README.md)。
- [PushT clean 缺项核查](../pusht_clean_audit_20261116/README.md)。

## 口径与限制

Native 为 5/5；三个 S 条件为 3/3。固定和动态 S 各有 5 推预热，另有固定 S 无额外预热。无预热仍需三推检测窗口，最早第 4 推施加 S；有预热最早第 7 推施加。固定强度 κ=2，动态强度 κ=2w。

这四组不能隔离动作块变化与 S 的效果，因为没有同一目标集的 Native-3 对照。相同种子也不能消除物理仿真的数值波动。不得仅凭单例截图、触发次数或两条独立轨迹的差异宣称 S 导致了改善。

原正式评测仅 Native 将逐推画面存盘；S 组未保存。因此截图采用单独运行并同步记录画面的方式制作，使用各自运行的实际 CD，完全排除在正式 20 样本统计之外。画面来自真实环境观测，不是预测或生成图像。可视化只对预先选定的两个目标各运行一次，保留全部结果，不重复尝试到获得有利结果为止。

DINO-WM 官方 MPC 默认代码为 5/5，但其 CEM 默认 30 轮；这里沿用试跑的 10 轮、新增 20 推预算和 CD 停止规则。不能把本包的结果表述为原论文全部设置的严格复现。内部参考 SE-JEPA 的资料只用于数据审计及图示布局，本次没有在论文中新增对该未发表文章的引用。

## 从证据包重建报告

在仓库根目录解压到新建目录（避免覆盖其他评测数据）：

```bash
mkdir -p /tmp/rope_formal20_evidence
tar -xzf reports/rope_formal20/evidence.tar.gz -C /tmp/rope_formal20_evidence
eval_outputs/rope_local/venv/bin/python experiments/rope_formal20/report.py \
  --root /tmp/rope_formal20_evidence/formal --output /tmp/rope_formal20_report
eval_outputs/rope_local/venv/bin/python experiments/rope_formal20/plot.py \
  --root /tmp/rope_formal20_evidence/formal --report /tmp/rope_formal20_report
eval_outputs/rope_local/venv/bin/python experiments/rope_formal20/visualize.py \
  --root /tmp/rope_formal20_evidence/qualitative \
  --formal-root /tmp/rope_formal20_evidence/formal \
  --output /tmp/rope_formal20_report/visuals
```

证据包保存全部正式运行的指标、实际动作、目标、配置、冻结模型核验、检测/惩罚日志和队列记录，以及四次可视化运行的完整观测缓存。导图脚本检查每一张 PNG 与原始 RGB 逐像素相同，并核查记录的物理动作和终态与评测输出一致。
