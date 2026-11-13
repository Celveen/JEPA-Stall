# LeWM PushObj contact-guidance experiments

Latest completed experiment: **231/350 (66.0%)** with the contact waypoint included in action optimization, compared with 217/350 (62.0%) for a post-optimization 3–20 px offset. Both use the same enhanced LeWM epoch-10 checkpoint, 350 matched tasks, threshold 0.45, four-replan guidance bursts and at most 150 controls. The model remains frozen.

| Planning condition | Success / 350 | Rate |
|---|---:|---:|
| Native | 215 | 61.43% |
| Intermittent contact offsets | 215 | 61.43% |
| Four-replan offsets, cap 15 px | 218 | 62.29% |
| Four-replan offsets, 3–20 px | 217 | 62.00% |
| Contact objective in optimization, weight 10 | **231** | **66.00%** |

See [the full contact-objective report](lewm_ecp_045_contact_objective/RESULTS.md) for per-shape outcomes, paired rescues/regressions and actual physics contact diagnostics. This deployment uses observed simulator object pose and known shape geometry, as do the matched contact-guidance controls.

The archive includes implementation snapshots, configs, raw execution logs, all executed actions, per-control physics replays, gradient/PD checks and SHA-256 receipts. All 350 outcomes were reproduced by physics replay; inputs and actions before the first trigger match the control arm exactly.

The original checkpoint and input dataset are external dependencies identified in the configs. Repeated rendered target caches and Python bytecode are omitted. No model weights are included.


完整结果见 [连续引导报告](lewm_ecp_045_contact_burst4/RESULTS.md)。

- `lewm_ecp_045_contact_burst4/` 保存新实验的代码、配置、日志、动作、回放状态及核验脚本。
- `lewm_ecp_045_contact/` 保存上一组间歇接触引导的匹配任务对照与物理回放。
- `SHA256SUMS.json` 记录归档文件的 SHA-256。

源运行目录为 `/tmp/lewm_ecp_045_contact_burst4`，归档省略重复图像缓存 `plan_targets.pkl` 与 Python 字节码。任务初始状态、目标状态和全部执行动作保存在 `final_eval_results.npz`，逐步物理状态保存在 `replayed/`。

运行环境为 `/home/appuser/anaconda3/envs/ts/bin/python`。重新测评还需要配置中指定的 checkpoint 和 H7t3 输入数据，均未复制到此归档。汇总、物理回放和诊断脚本可在当前目录结构下直接使用。

新增 [3–20 像素幅度实验](lewm_ecp_045_contact_burst4_bias3_20/RESULTS.md)，以连续 4 次干预、上限 15 像素的一组作为匹配对照。新组完整代码、配置、日志与物理回放保存在对应目录。

新增 [接触目标进入优化实验](lewm_ecp_045_contact_objective/RESULTS.md)，以连续 4 次干预、3–20 像素偏置组作为匹配对照。新组取消优化后偏置，在动作优化中加入执行前缀的接触目标代价，保留代码、配置、日志与物理回放。
