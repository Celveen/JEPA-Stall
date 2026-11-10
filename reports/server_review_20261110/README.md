# 第二篇论文：服务器导入与首次审阅

核查日期：2026-11-10。仓库为 `Celveen/JEPA-Stall`，分支为 `codex/cvpr2027-paper-results-v14`，导入提交为 `adbd03b40b8361b41ae64067c76ce88407d77d18`。本地目录为 `/data/zjc/adaJEPA/JEPA-Stall`。本次阅读稿件、协议、代码和结果归档，并核对相邻项目资源；没有启动新训练、采集或规划评测。

用户已授权后续将补测结果和修改后的论文推送到这个仓库。专用密钥的推送权限已通过 `git push --dry-run` 检查。模型二进制和训练图像仍按现有结果包约定单独保存；这次授权不要求上传这些大文件，也不恢复此前暂缓的 LeWM 权重发布。

## 论文内容与现有证据

论文题目为 **When World Models Stall: Execution-Conditioned Priors for Latent Planning**。主方法 ECP 将执行中的停滞反馈送回动作优化：S 保存停滞观测并对返回这些邻域的预测轨迹加惩罚；P 临时提高本体目标权重。模型可冻结，也可继续 AdaJEPA 更新；预测误差门控制干预时机。交互偏置数据是另一条训练侧改进。

正文、附录、源文件和 PDF 已拉取：

- 稿件：`paper/cvpr2027/paper_v14.pdf`；可编辑入口：`paper/cvpr2027/main.tex`。
- 主实验：`paper/cvpr2027/sections/experiments.tex`；组件与时机：`sections/analysis.tex`。
- 协议、限制与未测项目：`paper/cvpr2027/sections/appendix.tex`。
- 数值归档入口：`reports/publication_v14/README.md`。

| 已有证据 | 当前稿件所报结果 | 本次核对范围 |
|---|---|---|
| PushObj，350 任务 | Released Native：Frozen 43.7%、AdaJEPA 69.4%；数据增强 + ECP：72.3%、76.9% | 阅读正文、组件表、协议及来源清单；未在本次重跑或逐条重算全部 PushObj 轨迹 |
| PushT，原始 T + 六偏移 | 2 checkpoints × 2 更新方式 × 2 ECP 设置 × 7 条件，共 56 组、2,800 回合 | 独立核对 56 份原始成功数组、完成指纹和来源校验值；全部起点/目标一致，1,400 对干预前动作核验通过 |
| Frozen Maze | Native 48/100；S 50/100；S+P 54/100 | 直接读取归档汇总；S 为预先指定主干预，S+P 为组件对照 |
| Adaptive Maze | Native 63/100；gated S 66/100 | 阅读正文、附录和既有配对统计；仅 3 个任务净提升，需保留效应量边界 |
| v14 数值包 | 8 个 tar.gz | 全部文件 SHA-256 与 publication manifest 一致 |

这些已完成结果可以复用。旧的 NEXT_EXPERIMENTS、跨模型审计和 Frozen Maze 启动说明有的早于 v14 完成状态，不能据此重复启动已完成项目。

## 优先补强的证据

### 1. 停滞记忆 S 的独立贡献与跨模型适用性

当前 PushObj 组件表显示：增强数据 Frozen 的 P 与 S+P 均为 72.3%，S 为 62.6%、Native 为 61.1%；增强数据 AdaJEPA 的 P 为 77.1%，S+P 为 76.9%，S 为 76.0%、Native 为 74.9%。Released Frozen 的 P 为 60.0%，S+P 为 59.7%。因此，现有证据支持各组件改善 Native，但不支持 S+P 始终优于 P，也不能把全部主表提升归因于停滞记忆。

建议优先补每个 backbone 的 Native / +S 配对，固定模型、任务、初始化、规划器、精度和控制预算，报告成功率、救回/误伤、触发率及耗时。完整 S+P 可在确有本体通道的模型上另列。当前跨模型排名表主要比较完整控制器，不能替代这种机制验证。

本机 `CL-JEPA-cudagraph/models/lewm.py` 明确没有本体输入，兼容接口返回恒零 proprio。给它乘本体权重不构成有效 P；共同可迁移组件应先用 S，不能静默增加真值状态或新传感分支。

### 2. 独立留出与主方法重复评测

附录明确现有 PushObj/Maze 列表均用于开发；seed 100 与 seed 200 的 PushObj 列表还共享 35 个目标片段和 41 个来源轨迹，不能拼为 700 个独立确认任务。需要锁定控制器后建立来源独立的确认集，检查与训练、验证和开发资料的来源重叠。

主方法默认 3 个测试种子、基线默认 1 个；训练初始化重复与测试种子重复分开报告。已有三个训练初始化的 Native 统计不能当作 ECP 已完成三个初始化或三个测试种子的验证。优先找回已训练 checkpoint，不默认新增训练。

### 3. 低成本补齐组件时机与离线诊断

- Maze 的 adaptive S-only first-stall 对照尚需核实/补齐，才能与 gated S-only 单独比较门控；旧建议是同一 100 个任务。跨硬件复用旧参考前需验证模型重置、随机流及干预前轨迹。
- 优先从已保存动作/状态离线统计物体已到位但推杆未到位、物体尚未完成、导航停滞等失败类型；完整保留失败与救回，而非只挑成功案例。
- 从现有运行时日志汇总规划时间、搜索次数和 ECP 开销；“同为 150 控制步”不表示搜索计算量相等。若要主张比增加搜索更有效，再单列计算量匹配的对照。

### 4. 需要单列预算的训练侧项目

偏置 × 噪声 2×2 因子消融需要严格配对采集和训练，旧提案为 4 次训练、8 组评测、2,800 回合。当前数据实验改变的是完整采集/训练配方，不能单独归因于方向偏置或噪声。此项不是现有 Native/ECP 配对结果的必要重训条件。

训练侧 progress objective 和 Wall 新控制器属于额外研究范围；不因附录有占位符就自动扩成新模型训练、额外任务或参数网格。本次仅列出，不启动。

## DINO-WM 新四组实验：接续前必须区分

**2026-11-10 用户后续修订：** 后续 DINO-WM 使用本机已有随机/偏置双版本，不再依赖另一台机器的进展。本节保留为导入版本的历史来源说明，不是当前任务的前置条件。本次优先运行 Rope 冻结 Native/+S，见 `../rope_ecp_20261110/README.md`；没有启动 DINO-WM PushObj 补测。

第二篇自身已有普通/增强数据 × Native/ECP 的四组实验，两份训练均已完成 epoch 3。v3 使用 200 候选 × 10 CEM 迭代、top30、last 目标、BF16 候选预测、25 控制步时域、每次执行 25、最多 6 次，共 150 控制步；总计四组各 350 回合。ECP warmup 为 50 控制步、运动窗口为 15 控制步，boost 为一个完整计划。旧 30×5 版本已被替代。

**Git 中只是原机器的历史快照：** publication manifest 为 92/140 个完成 chunk，实际 progress 归档也含 92 个完成标记；较晚的独立状态文件为 94/140。它们的路径指向 `/home/appuser/workspace/project/multimodal-test/JEPA-Stall`，不是本机路径。不能据此认定原机现在仍运行，也不能把 94 当作归档中完整可接续的 94 个 chunk。先取原机最新状态与成果，再决定是否补余项。

这两份第二篇 checkpoint 的训练数据为 T/L/Z/+、16k/4k、1,630,266 训练窗口、batch32、3 epochs/152,838 更新；省略 decoder。其权重未随 Git 入库。相邻第一篇的 random16k 模型训练形状为 I/square/Z/T、16k/1.6k，采用完整 52,319,499 参数结构和 decoder；这不是可直接替代的同一实验。不能在接续时静默换权重、改训练分布或重新训练。

## 本机可复用资源与限制

机器可读路径、大小和关键校验值见 `server_resources.json`。资源存在不等于第二篇加载/运行已验证。

| 资源 | 本机核查 | 使用边界 |
|---|---|---|
| 主方法 PushObj seed0、released、T-only | 固定权重和配置存在 | 主历史 seed0 已记 SHA；第二篇模型身份仍需同其来源收据对齐；seed1/2 尚未在已查目录确认 |
| LeWM bias16k epoch10 | 存在，SHA 与第二篇已归档 manifest 完全相同 | 可优先做冻结迁移适配；训练10轮需披露；已有 seed2000 Native 不能直接当作 seed100 配对基线 |
| LeWM random16k epoch3 | 存在，训练完成记录为 30,375 更新 | 训练形状与旧 bias16k 不同；不把与 epoch10 的比较解释为只改变数据；此前权重发布暂缓指令继续保留 |
| DINO-WM reference bias16k epoch3、random PushObj/PushT epoch3 | 文件存在 | 与第二篇专门训练的 ordinary/augmented 两份模型不同，模型和协议不能混写 |
| DDP-WM bias16k / PushT | 权重及代码存在 | 尚未在第二篇接通 S；需相同 checkpoint 的 Native/+S 配对 |
| DreamerV3 / IRIS | T-only 冻结权重和结果存在 | 可复用已有合规结果；不是证明 ECP 跨模型作用的必要新增方法，不默认扩实验 |
| Maze | 已查 CL-JEPA/checkpoints 无 Maze 模型，CL-JEPA/data/diverse_maze 不存在，cljepa 环境无 mujoco_py | 先补依赖与公开 checkpoint；不宣称立即可跑 |

核查期间 GPU0 空闲，GPU1–7 有 IRIS PushObj 评测活进程。该信息只对应核查时刻；启动任何新增作业前需重新查占用，保护现有实验。

## 后续执行顺序

1. 按用户最新要求先运行本机 Rope 冻结 Native/+S，保持旧50目标与15推动预算；不训练，暂不扩到 Granular。
2. DINO-WM 后续使用本机已确认的随机/偏置双版本，披露训练形状差异；不再等待原机。跨模型 Native/+S 接口先核验，再固定后续具体范围和预算。
3. 补来源独立确认与主方法重复评测，完成必要组件时机；离线诊断优先复用现有轨迹。
4. 只有需要回答单因素数据因果问题时，再单列新采集/训练预算。
5. 完整结果核验后更新来源清单、主表、正文/附录和 PDF，并按用户授权推送本分支；不将部分结果填成完整结论。
