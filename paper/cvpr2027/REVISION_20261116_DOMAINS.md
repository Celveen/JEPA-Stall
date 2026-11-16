# 2026-11-16：启发措辞与四类任务插图

- 摘要改为 “Our training and inference design is inspired by DART's noise-injected data collection and ACT's temporally extended action chunks, respectively.” 引言相应明确 inspired by，保留方法区别和正式引用。
- 本机分支安全快进导入 GitHub 两次新增提交，当前基准为 `9b01e93ef0644e67bbb280efd7c69d528c830ad3`；原有论文修改全部保留。
- Figure 3 第 3 步保留 PushObj、Maze，补充 PushT 颜色偏移与 Rope 真实仿真画面。其余步骤保留已有必要文字和时间条。
- Figure 4 改为 PushObj Z/Square、PushT Red agent/Red goal、Maze Frozen/AdaJEPA、Rope 两组目标，共八组。PushObj 从六组减少到两组。
- PushT 为已完成评测的原生 Pymunk 状态渲染：增强数据 AdaJEPA 的任务 26、23，使用原日志末态和评测颜色；无新增模型推理或物理执行。Rope 使用新提交内原始 RGB：目标 13、17，独立可视化复跑，不计入正式统计。正例 S 未触发，反例触发九次；Native-5 与 S-3 改变动作块，不能解释为 S 单因素收益。图注和附录明确这些区别。
- 数据表格及参考文献源文件不变；没有运行新评测。完整新素材、哈希和配对记录见 `figures/domains_20261116`、`provenance/domain_visuals_20261116.json`。
- 正文 8 页、整篇 20 页、摘要 205 词；45 条参考文献；已编译并渲染检查。无未定义引用、溢出框、Type 3 字体或未嵌入字体。

重建 Figure 3：`python scripts/build_method_figure.py`。
重建 Figure 4：`python scripts/build_domain_figure4.py --paper .`（在论文目录运行）。
随后按 README 编译论文。
