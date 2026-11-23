> **2026-11-16 有效性更正：** 本报告的启动器将未保存于任务 checkpoint 的 DINO 编码器随机初始化，结果不能用作正确 DINO-WM 的评测或 +S 收益结论。原始数据保留用于审计；原因及修复见 [编码器核查记录](../rope_encoder_audit_20261116/DIAGNOSIS.md)。

# Rope：冻结模型 Native / ECP 停滞记忆试跑

2026-11-10 用户要求：DINO-WM 后续按本机资源安排，不再等待另一台机器；先确认本机双版本，不启动 PushObj 补测。优先检查 ECP 能否用于仿真机械臂 Rope/Granular，本次只启动 Rope，不做任何新训练或在线更新。

## 本机 DINO-WM 双版本已确认

| 项目 | 偏置版本 | 随机版本 |
|---|---|---|
| PushObj 数据 | `/data/zjc/adaJEPA/CL-JEPA/data/pushobj_bias_16k` | `/data/zjc/adaJEPA/CL-JEPA/data/dinowm_pushobj_random16k_20261106` |
| 固定三轮权重 | `CL-JEPA/checkpoints/dinowm_pushobj_bias16k_reference_e3/checkpoints/model_3.pth` | `CL-JEPA/checkpoints/dinowm_pushobj_random16k_orig_e3_20261106/checkpoints/model_3.pth` |
| 训练形状 | T / L / Z / + | I / square / Z / T |
| 限制 | 与随机版本的比较同时改变了形状覆盖及采集配方 | 不能把 I/square 说成该模型的未见形状 |

路径均以 `/data/zjc/adaJEPA` 为根；数据和固定权重文件均已在本机查到。另有 T-only 以及随机 T16k 的 PushT 模型；本次不启动这些模型的新增评测。不能把双版本称为只改变动作偏置的严格单因素对照。

## 预先锁定的 Rope 试跑协议

| 设置 | 取值 |
|---|---|
| 模型 | 本机既有 `rope_dino_h1_fs1_e100_seed0/checkpoints/model_100.pth`，历史训练 epoch100；不训练、不加载持续学习状态 |
| 两组 | 同一冻结模型 Native；同一冻结模型 +S（停滞记忆） |
| 任务 | 旧 `rope_cl_single/stream/chunk_0..4.pkl`，5×10=50 个固定目标，双方相同 |
| 随机性 | 保留旧流的 chunk seeds 99–103；每块内部原环境 seed 映射 `chunk_seed * sample_index + 1`；这是同一套 50 目标试跑，非五个完整测试种子 |
| CEM | 300 candidates、10 iterations、top30；相同初始均值、采样顺序、方差与精度，不增加搜索 |
| 模型/控制时域 | frameskip1；预测5次推动；每次执行5次；最多3次重规划，合计15次推动 |
| 主目标 | 原有 last；视觉/占位本体 MSE，alpha1；不使用粒子状态规划 |
| S | joint latent；kappa2、beta0.25、运动阈值0.05/0.15、帧权重 base2；无 P、无 loss gate、无踢动 |
| 触发时域 | 执行完首个5次推动后才能介入；使用已返回的逐次观测比较最近3次推动的位移；最多在第二、三次计划启用，不新增规划或动作 |
| 代价尺度 | Rope 使用 last，S 的幅度为 D(memory,goal)，不沿用 all-frame 的额外 H+1 除法；固定 kappa，不按结果调参 |
| 评测 | 主指标为15次推动后的粒子 CD，越低越好；报告配对改善/退步、触发率、规划时间；不将恒为假的环境 success 当作任务 SR |
| 运行资源 | 启动时唯一空闲 GPU0；CUDA 按 UUID 指定，EGL 按物理 GPU0；GPU1–7 的 IRIS 保留 |
| 预算 | 两组各50，100个逻辑主评测回合，未改变控制的前缀共享实际 Native 执行；单任务强制触发及集成修正检查另行保留，全部不进入主结果 |

Rope 的 proprio 输入为恒零占位，故本试跑只验证 S，不能称已验证完整 S+P。原 PushObj 的“等待10次重规划”参数在 Rope 的3次规划预算中永远不会启用，因此显式换算成上述推动时间协议。它是本次预先指定的迁移设置，不宣称逐参数完全相同。粒子 CD 仅由评测钩子读取，S 的触发和记忆只读视觉/占位本体观测。

## 正确性与停止条件

CPU 数值检查验证：原 CEM 与加入样本上下文的 CEM 在同一随机流下动作逐位相同；零 kappa 回到原目标；目标处不受排斥；固定上下文不计正则；样本之间不共享记忆；末端目标幅度及5/3推动时域正确。

正式队列前对固定首个任务做 Native/+S 的两计划集成检查。仅该检查强制触发，验证第二次计划可以接入 S、干预前动作与 CD 配对、参数和缓冲区不变。主评测恢复预定0.05/0.15阈值。

每个主任务在首次干预前验证动作与 CD，未触发任务则检查完整轨迹。两组严格核对目标、checkpoint 文件校验值以及运行中全部参数/缓冲区哈希。不因改善不足或无触发扩大预算、改阈值或改种子。数值非有限、配对失败、状态改变或运行失败会停止队列并保留文件。

### 集成阶段的前缀配对修正

最初两次独立物理执行的首计划动作完全相同，但 PyFleX 的 CD 相差约 0.000175，未通过原定严格前缀 CD 核验。主评测当时尚未启动。两次输出均保留在 `eval_outputs/rope_ecp_20261110/integration_before_prefix_reuse/`，不纳入结果；诊断见 `prefix_integrity_update.json`。

随后沿用论文的未受影响轨迹复用原则：先完成每个 Native chunk，保存逐次真实观测、粒子状态、动作和随机状态；+S 在尚未发生停滞干预时重新核对 CEM 动作，并复用对应 Native 执行前缀。已有停滞记忆的任务才从同一个原生边界状态调用原环境继续仿真。所有任务仍运行同样的 CEM，候选随机流按原生边界对齐；不根据 CD 决定复用、不改变阈值。未触发任务完整复用原结果，并在报告中计入。此举消除了干预前的重复仿真漂移，节省的物理运行不转用为更多任务或搜索。

现有 Rope 环境本就每个五推动块调用 `prepare(seed, particle_positions)` 恢复状态，分支沿用同一 API；没有假设跨调用保留额外速度。粒子状态仍只作为普通模拟器初态及 CD 评测输入，规划代价和停滞检测仅使用观测。

强制触发的首个集成任务中，记忆虽然启用，所有采样预测却都在局部核支持域外，附加代价为零。因此还要区分“记忆触发”和“真正改变执行动作”：只要此前没有分支、这次给模拟器的动作又与原生逐位相同，就继续共享原生执行，避免把无动作变化后的物理漂移当成方法增益。首次实际改变动作后才独立分支，此后不再借用原生轨迹；报告分别给出触发数、非零代价与动作改变数。这个修正不调整支持半径或其他方法参数。

正在进行的首个 Native chunk 使用相同的冻结模型、采样器、目标和原生代价，其计算继续保留。队列在该块完成后自动交接到包含上述 +S 分支记录修正的版本，重新核验 +S 集成，再接续完整对照。旧、新源码校验值和交接记录单列保存，不重复运行已经完成的 Native 控制。

## 代码、状态与交付

- 独立接入：`experiments/rope_ecp_20261110/`；运行时只读复用 `/data/zjc/adaJEPA/CL-JEPA-cudagraph` 的模型、数据和 PyFleX 环境，不修改其源码。
- 启动：`/data/zjc/envs/cljepa/bin/python experiments/rope_ecp_20261110/run.py --gpu 0 --run`；已运行时不要重复启动。
- 状态：本目录 `pipeline_status.json`；阶段包括集成检查、主评测和完成打包。
- 原始输出：`/data/zjc/adaJEPA/JEPA-Stall/eval_outputs/rope_ecp_20261110/`。
- 每块完成两组后写 `progress_summary.json`；全部50对完成后生成 `summary.json`、`RESULTS.md`、逐任务 CSV 和含目标/动作/状态/CD/审计的无损归档。
- 完成后自动将本报告目录推送到 `codex/cvpr2027-paper-results-v14`。使用独立稀疏工作副本接到当时远端最新版本，不重写其他论文修改；上传失败单独记录，所有本地结果保留。

这是一次固定旧目标流的探索性迁移试跑，尚未完成三测试种子的确认。Granular 不在本次运行队列中。
