# PushT clean 缺项核查

同步并检查 SE-JEPA 的全部远端分支及历史 reports-pack 压缩包；该仓库仅作为内部数据来源，本次没有在论文中添加对未发表文章的引用。

| 主表条件 | clean 原始记录 | 成绩 | 处理 |
|---|---|---|---|
| LeWM / GD | 有，50 个 successes | 32/50＝64% | 主表已填写 |
| LeWM / CEM | 未找到匹配记录 | 未测量 | 保留空缺 |
| DDP-WM / CEM | 未找到匹配记录 | 未测量 | 保留空缺 |

两个 CEM 的 VIS50 目录都只有 blur、snp、dark、red_agent、red_object、red_goal 六种扰动。重新核算每种 50 个结果，LeWM 分别为 66%、64%、64%、2%、20%、26%；DDP-WM 分别为 56%、22%、66%、62%、42%、68%，与当前主表一致。LeWM-GD 的 default 记录是另一控制器，不能代填 CEM。PushObj T 形状列的目标与训练设置也不能直接替换 PushT clean。

检查了所有远端分支中 DDP/LeWM 的 clean/default 记录路径；找到的完整 clean 记录均为同一个 LeWM-GD 档案。历史 reports-pack 共 91 个成员，没有 DDP 或 LeWM 条目。分支提交、逐条件计数、GD 文件哈希见 [audit.json](audit.json)。此前的 [baseline_audit.md](../pusht_visual_v8/baseline_audit.md) 也已将两项标为缺测量。

要补齐两项，需要分别对现有 LeWM T16k epoch-10 和 DDP-WM PushT checkpoint 执行 clean CEM，复用主表固定的 50 个目标及文件顺序、原成功规则与 150 个物理步预算。参考已有 VIS50 配置，移除视觉扰动；保留 30 次规划×5 步、预测 25 个物理步、CEM 200 候选／30 精英／10 轮。本次仅做档案核查，没有启动这两个额外 PushT 实验。
