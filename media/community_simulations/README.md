# 社区视频真实仿真逐帧素材

**Rope 连续推动视频已补充**：[原两组与新增一组的完整录像、每推 MP4 和连续 PNG 帧包](rope_motion/README.md)。

PushObj、PushT、Rope 和 Maze 各两组，共八组。每个任务提供一组机制开关对比、一组完整方法对比，展示同起点和目标下的执行差异。案例按可见差异筛选，用于视频展示。

| 素材与动态预览 | 对比 | 基线 / Ours 结束步 | 结果 |
|---|---|---|---|
| [pushobj_ablation](pushobj_ablation/preview.gif) | Frozen MPC · ECP off → Ours · Frozen + ECP | 150 / 70 | 预算内未完成 → 成功 |
| [pushobj_methods](pushobj_methods/preview.gif) | AdaJEPA → Ours · Frozen + ECP | 150 / 70 | 预算内未完成 → 成功 |
| [pusht_ablation](pusht_ablation/preview.gif) | LeWM · intervention off → Ours · LeWM + intervention | 150 / 65 | 预算内未完成 → 成功 |
| [pusht_methods](pusht_methods/preview.gif) | AdaJEPA → Ours · LeWM + intervention | 150 / 65 | 预算内未完成 → 成功 |
| [rope_ablation](rope_ablation/preview.gif) | DINO-WM 3/3 · ECP off → Ours · DINO-WM 3/3 + ECP | 20 / 20 | CD 1.7939 → CD 0.1681 |
| [rope_methods](rope_methods/preview.gif) | DINO-WM native 5/5 → Ours · DINO-WM 3/3 + ECP | 20 / 4 | CD 0.7160 → CD 0.0836 |
| [maze_ablation](maze_ablation/preview.gif) | AdaJEPA · ECP off → Ours · AdaJEPA + gated ECP | 150 / 110 | 预算内未完成 → 成功 |
| [maze_methods](maze_methods/preview.gif) | Frozen MPC → Ours · AdaJEPA + gated ECP | 150 / 130 | 预算内未完成 → 成功 |

## 剪辑入口

每组 `baseline/frame_0000.png` 与 `ours/frame_0000.png` 起按编号连续保存原始画面。PushObj、PushT、Maze 每帧对应一个物理控制步；Rope 每帧对应一次完整推动后的真实观测。画面保持仿真器 224×224 原分辨率。

`comparison_frames/frame_*.png` 是 1280×670 的逐步同步展示帧，带方法名称、当前步、独立目标图和实际状态轨迹。`preview.gif` 用于快速选片，`storyboard.jpg` 展示四个过程节点。两方案结束时间不同，展示帧会明确写出 terminal frame held。原始序列在各自真实结束步停止。

轨迹线连接逐控制观测的位置。PushObj 和 PushT 在画面中显示真实 pusher 路径；右侧另附完整路径。Maze 路径使用仿真世界坐标单独展示，Rope 路径是粒子质心的 x/z 轨迹。GIF 控制步按 150ms、推动步按 450ms 播放，剪辑时可自行调速。

## 八组实验各自说明什么

- **PushObj 消融**展示同一 Frozen 模型与规划配置下，ECP 如何让 Z 形物体任务走出停滞并达到目标。350 个配对任务中按最终物体距离改善筛选，chunk16 sample4，阈值 0.45。
- **PushObj 方法对比**展示 Frozen+ECP 与 AdaJEPA 在同一 small-T 目标上的执行差异，chunk24 sample9。
- **PushT 消融**展示 LeWM 干预开关对联合 pusher/object 目标完成的作用，red-goal target6。干预包含 latent memory、proprio 与使用物体位姿和几何的 contact objective。物体在基线末尾已经接近目标，主要差异来自 pusher 的最终对齐。
- **PushT 方法对比**在同一目标上比较 LeWM+干预与 augmented AdaJEPA。实际目标以单独的 Goal 图为准，仿真背景中固定的红色 T 是环境 anchor。
- **Rope 消融**使用 seed101 target15，双方均为 3/3 推动节奏，ECP 从第7次推动开始应用。关闭干预的末端 CD 为 1.7939，开启后为 0.1681。
- **Rope 方法对比**使用独立录制的 target13，比较原生 DINO-WM 5/5 与 Ours 3/3 完整规划流程，CD 从 0.7160 降至 0.0836。该组节奏不同，且 Ours 本次执行没有触发 ECP；ECP 的机制作用由前一组同节奏消融呈现。
- **Maze 消融**使用 medium seed100 target17，比较 AdaJEPA 与 AdaJEPA+gated ECP，干预后在第110控制步完成。
- **Maze 方法对比**使用同一任务集 target13，比较 Frozen MPC 与 AdaJEPA+gated ECP，Ours 在第130控制步完成。

PushObj 与 PushT 的判定同时检查 pusher/object 联合位置与物体角度，Rope 按 CD<0.3 停止，Maze 按位置距离<0.5 停止。各方法名称保留实际模型与规划配置。

## 真实执行与来源

PushT、Rope、Maze 原始 PNG 直接导出已保存的真实仿真 RGB，并逐像素核验。PushObj 以及 PushT 方法对比的 AdaJEPA 行用原始动作在原 Pymunk 仿真器重放，逐五控制步核对原实验日志端点，最大允许误差 0.00051。PushObj 的目标图由原渲染器在保存的目标位姿生成。

每组 `manifest.json` 记录步骤、来源、结果和每张原始 PNG 的 SHA-256。各方案的 `execution.npz` 保存逐步状态，`evidence/` 保存动作、配置与干预记录。顶层 `manifest.json` 汇总源文件校验值，`validation.json` 汇总帧数和原始像素核验。源实验配置中的绝对路径对应原执行机器。

导出使用 `export_frames.py`，核验与索引使用 `verify_and_index.py`。运行需要本仓库已有的记录文件和 `eval_outputs/rope_local/venv` 中的 NumPy、Pillow、PyTorch、Pymunk、Gym 等依赖。PushObj 原始动作与日志在 `reports/frozen_ecp_cost_20261117/raw_results.tar.gz`、`reports/publication_v14/archives/main_pushobj_*.tar.gz`，PushT AdaJEPA 原始记录在 `reports/publication_v14/archives/pusht_visual_complete.tar.gz`。

展示参考 [DINO-WM 项目页](https://dino-wm.github.io/) 的并排方法与独立目标图布局。
