# PushT与六种视觉偏移：运行中

2026-11-09，直接在10.141.224.31上执行。该任务本来已运行在GPU服务器；先前SSH认证障碍来自重复连接同一机器，GPU不可见来自执行沙箱。

- 已通过：16组smoke（32 episode-runs），16对episode的初始/目标状态和介入前动作核验，最大动作差异0。
- 正式作业：56组×50任务=2800 episode-runs；两套权重、Frozen/AdaJEPA、ECP开关、clean与六种偏移，最多30 MPC。
- GPU：0、1、3、5、6；主调度PID2993846；2026-11-09 06:37:24 UTC启动。
- 所有作业日志：../../eval_outputs/pusht_visual_v8/launcher.log；每组原始文件在其attempts目录。
- 动态完成统计：../../eval_outputs/pusht_visual_v8/results.json和results.csv。
- 自动收尾状态：../../eval_outputs/pusht_visual_v8/finalization_status.json。

全部完成后，收尾程序核验1400对episode、汇总rescue/regression，并在eval_outputs/pusht_visual_v8/paper_completed_v10/生成独立的填数版paper.pdf。该文件由锁定的v10源码构建，不覆盖用户随后修改的论文。

原始结果、配置、目标/权重哈希和基线证据保存在reports/pusht_visual_v8/。新增分数完成前保留空缺，不从不同任务列表或smoke套用。

该完整填数稿同时等待 Frozen Maze 配对验证完成，并导入 S 主实验和 S+P 对照结果。
