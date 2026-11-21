# Frozen Maze 补充评测

2026-11-09 06:58:35 UTC 在当前 GPU 服务器启动，主调度 PID 3018313。

- 相同 100 个 medium Maze 任务，released checkpoint，30 次 MPC，action repeat 4。
- 已核验 native 基线 48/100；两个重新执行的 native 检查任务与历史动作/状态一致。
- 预先指定主实验 S（停滞记忆），次实验 S+P（组件对照）。两者都使用原有移动触发参数，模型保持冻结。
- 每组重新评测 50 个可能触发的任务；其余 50 个不会启用修正，复用原始结果。所有重新评测都检查干预前动作和状态一致性。
- 输出：`../../eval_outputs/maze_frozen_v10/`；日志 `launcher.log`；完整汇总 `summary.json` 仅在两个实验全部通过验证后生成。
- 新案例 7：Frozen 在第 30 次仍失败，S 第 28 次成功，前 13 次完全相同；原始 MuJoCo 画面已用于图 4。这是一个定性案例，完整成功率等待全量评测。
- 自动填数稿：`../../eval_outputs/maze_frozen_v10/paper_completed_v10/paper.pdf`；状态见 `finalization_status.json`。

与现有 PushT 队列共用 GPU 0、1、3、5、6，不重复启动主调度。图中所有轨迹均来自实际保存的端点，无插值补造运行状态。
