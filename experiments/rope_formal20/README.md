# Rope 正式 20 目标协议

四组各 20 个样本，最多 20 推，逐推观测，严格 CD < 0.3 即停。采用原目标池 chunk 1、2 的固定顺序，排除之前试跑的 chunk 0。该目标池已有历史基线记录，不能称为全新未见基准。

| 条件 | 预测／执行 | 额外预热 | 单条记忆强度 | GPU |
|---|---|---|---|---|
| native5 | 5／5 | 无 S | 0 | 1 |
| fixed3 | 3／3 | 5 推 | 2 | 2 |
| dynamic3 | 3／3 | 5 推 | 2×w | 3 |
| fixed3_nowarm | 3／3 | 0 推 | 2 | 4 |

三推检测窗口保持不变。因此无预热最早在完成第 3 推后建立记忆，第 4 推使用；预热 5 推的两组最早在第 7 推使用。块内每推持续观测并检查成功，动作块不被重规划打断。CEM 保持 300 候选、30 精英、10 轮；模型、编码器逐张量核验后冻结。

每张卡固定运行一组，按样本顺序执行，各 episode 独立进程。四组直接模拟，不依赖旧 Native-3 的轨迹缓存。全局样本编号 10–29，环境 seed=99×编号+1，规划 seed=99+编号，场景构建 seed=99。配置、源代码哈希、目标哈希和模型校验都随每个样本保存。评测期间冻结配置。

本轮没有 Native-3 条件，Native-5 与 S-3 的差异同时包含块长度和 S，不能宣称是 S 的单因素收益。独立物理仿真也可能使相同策略的轨迹分叉。报告保留所有实际动作，但不把相对 Native-5 的动作变化单独归因于 S。

官方 [MPC-CEM 默认配置](https://github.com/gaoyuezhou/dino_wm/blob/main/conf/planner/mpc_cem.yaml) 为 horizon=5、n_taken_actions=5；默认 CEM 轮数为 30。本轮沿用试跑的 10 轮、20 推预算及新增成功停止规则，属于固定协议下的方法对照，不是原论文全部设置的严格复现。正式论文附录 A.5 将 Rope 的 0.41 列在 MPC 下，但对执行前缀只用 k 表示。

```bash
python3 experiments/rope_formal20/launch.py --root eval_outputs/rope_formal20 \
  --gpu-map native5=1,fixed3=2,dynamic3=3,fixed3_nowarm=4 --max-pushes 20
eval_outputs/rope_local/venv/bin/python experiments/rope_formal20/report.py \
  --root eval_outputs/rope_formal20 --output reports/rope_formal20
```

完整结束的样本可恢复复用；新独立运行需使用新输出目录。必须等待全部 80 个 episode 完成再汇总。

## 真实仿真截图

原正式实验仅 Native 保存了逐推 RGB；S 组未保存，不能为其追溯导出原执行截图。为满足真实截图要求，另用以下入口独立复跑两对固定目标，保存实际执行时内存中的每推观测。记录操作在规划执行结束后进行，不新增仿真或随机数调用。可视化运行不会混入原 20 样本统计。

```bash
python3 experiments/rope_formal20/capture_launch.py \
  --root eval_outputs/rope_qualitative20 --samples 3 7 --gpus 5 6
eval_outputs/rope_local/venv/bin/python experiments/rope_formal20/visualize.py
```

可视化开始前指定 sample 03（正式运行改善）和 sample 07（正式运行退步），每个条件只跑一次并保留全部结果。图展示独立复跑自己的 CD、每三推截图和实际终点；每一步原图也全部导出。物理仿真波动可能导致复跑与原统计不同，不用原 CD 标注复跑画面。`capture_worker.py` 仅包装原规划器的输出保存，无需修改正在运行的正式实验源文件。

全部评测完成后生成统计图和证据包：

```bash
eval_outputs/rope_local/venv/bin/python experiments/rope_formal20/plot.py \
  --root eval_outputs/rope_formal20 --report reports/rope_formal20
python3 experiments/rope_formal20/bundle.py
```

核查缓存复用时以 `prefix_reuse.json` 的逐推动作记录为准，本轮所有条目均要求 `reused=false`。历史 `frozen_model_audit.json` 中 `native_prefix_reuse` 字段只按 κ>0 标记，不能据此推断实际复用了 Native 轨迹；正式报告额外逐项检查真实复用记录。

## 补充同样本 Native 3/3

新增 `native3` 条件，采用与上述四组完全相同的 20 个目标（原编号 10–29），一张 GPU 顺序运行整组。原四组不重跑、不覆盖。Native-3 与 dynamic3 的规划配置只差 κ=0/2；模型、核心实现、种子、CEM 和停止规则一致。与固定 S 的 strength_mode 标志也不同，但 κ=0 时没有任何 S 惩罚。原入口已有 Native 判定和三步块的实现，本次仅增加命令行选项；差异及源文件快照保存在补充输出目录的 source_snapshot。

```bash
python3 experiments/rope_formal20/launch.py \
  --root eval_outputs/rope_formal20_native3 --gpu-map native3=1 --max-pushes 20
python3 experiments/rope_formal20/status.py --root eval_outputs/rope_formal20_native3
eval_outputs/rope_local/venv/bin/python experiments/rope_formal20/native3_report.py
python3 experiments/rope_formal20/native3_bundle.py
```

补充报告对五组做同目标核验，并保留 Native-3 每一推的真实观测缓存。五组汇总在 `reports/rope_formal20_native3/`，原四组报告作为历史记录保留。全部为独立物理执行，不把新 Native-3 的缓存回填为旧 S 组的执行前缀。
