# Rope：同样本 Native 3/3 补充对照

在原四组正式结果上增加 Native 3/3，形成五组各 20 个相同目标的对照。模型和目标顺序不变，仍为最多 20 推、逐推观测、CD < 0.3 即停。三个 S 条件沿用已经完成的原始运行，不重跑或挑选替换结果。

- [五组完整结果及相对 Native-3 的差异](RESULTS.md)
- [逐样本成绩](per_sample.csv) · [逐推 CD](cd_by_push.csv) · [实际动作比较](action_comparisons.csv) · [汇总校验](summary.json)
- [五组结果图](results.png)（[PDF](results.pdf)）
- [新增 20 次执行的原始证据](native3_evidence.tar.gz) · [逐文件哈希](evidence_manifest.json)
- [协议核验](protocol_audit.json) · [入口唯一变更](worker_change.diff)
- [原四组与可视化运行](../rope_formal20/README.md)

Native-3 与动态 S 的规划配置仅差 κ=0/2；与固定 S 还存在强度模式标志差异，但 κ=0 时没有 S 惩罚。三组的 horizon=3、chunk_size=3，CEM 均为 300 候选、30 精英、10 轮。环境 seed=99×原目标编号+1，规划 seed=99+原目标编号；场景构建 seed=99。目标为原编号 10–29，按固定顺序评估。

本次只在原评测入口的允许选项中加入 native3；入口此前已有按 native 前缀关闭 S、除 native5 外使用三步块的逻辑。新增报告校验旧/新入口源文件只存在这一处变更，并校验其他规划器、模型和运行时代码哈希一致。原四组的均值、成功数、实际推数与触发次数均检查未变化。

## 如何解读

Native-3 的平均 CD 为 0.7547、成功 10/20；固定 S 为 0.6112、13/20，动态 S 为 0.5152、12/20，无预热固定 S 为 0.5834、13/20。本轮三组 S 均改善同节奏 Native-3 的平均 CD 和成功数。Native-5 的成功数仍最高（14/20），动态 S 的平均 CD 最低。20 个目标的一次独立执行不足以确定固定/动态强度的稳定优势。

Native-5 与 Native-3 比较动作块长度；Native-3 与 S-3 比较同一节奏下加入 S 的表现。报告按目标给出 CD 变化、新增/丢失成功，以及实际控制量的差异。各条件仍是独立物理执行，存在数值波动；动作可能在 S 首次使用前已分叉，这些情况会单独计数。

本次 Native-3 的每一步真实观测、粒子状态和物理动作均保存在 `native_execution_cache.npz`，随补充证据包发布。原四组的可视化实例仍是独立复跑，未混入正式统计。

## 运行与重建

从仓库根目录启动，一张空闲 GPU 顺序运行整组：

```bash
python3 experiments/rope_formal20/launch.py \
  --root eval_outputs/rope_formal20_native3 --gpu-map native3=1 --max-pushes 20
python3 experiments/rope_formal20/status.py --root eval_outputs/rope_formal20_native3
```

入口通过原正式输出中的 `dynamic3/sample_00/complete.json` 自动生成兼容性快照。若原输出在其他位置，用 `--reference-root` 指定。新的独立实验应使用新的输出目录。

仅重建五组报告时，无需再次运行仿真：

```bash
mkdir -p /tmp/rope_five_arm_evidence
tar -xzf reports/rope_formal20/evidence.tar.gz -C /tmp/rope_five_arm_evidence
tar -xzf reports/rope_formal20_native3/native3_evidence.tar.gz -C /tmp/rope_five_arm_evidence
eval_outputs/rope_local/venv/bin/python experiments/rope_formal20/native3_report.py \
  --original /tmp/rope_five_arm_evidence/formal \
  --addition /tmp/rope_five_arm_evidence/supplement \
  --output /tmp/rope_five_arm_rebuilt
```

证据包的 `source_snapshot` 保存新旧入口源码及其差异；补充报告会重新核查源文件、冻结模型、配对目标、实际动作、终态及最早成功停止点。
