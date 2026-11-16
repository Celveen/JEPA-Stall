# When World Models Stall: Execution-Conditioned Planning in Latent Space

LeWM 的 PushT clean CEM 已补齐为 **72%（36/50）**，方法标签保持 LeWM；配置、权重来源及原始结果见 [完整记录](../../reports/lewm_pusht_clean_20261116/README.md)。

当前稿采用 Execution-Conditioned Planning（ECP）命名。正文对附录证据的引用由 7 处补到 16 处，涵盖额外基线、LeWM 组件、分形状结果、训练种子、动作空间控制、Maze 预算和恢复行为。附录删减重复报数与说明，保留全部实验表格数值；Figure 6、7 共享重复状态并缩为单排，新增真实 PushT 视觉偏移和 Rope 运行序列。完整稿 21 页：正文 8 页，声明与 47 条参考文献 2 页，附录 11 页。摘要保持 196 词。审计见 `provenance/appendix_evidence_revision.json`。

[完整论文](paper.pdf) · [主文](paper_main.pdf) · [附录](paper_appendix.pdf) · [完整源码](paper_source_v14.zip) · [摘要](Abstract.txt) · [可编辑 Figure 3](Figure3.drawio)

## 重建

直接编译使用随稿表格和插图：

```bash
python scripts/build_paper.py --tectonic /path/to/tectonic
```

Figure 3 的矢量 PDF 与 draw.io 来自相同绘图元素，可通过以下命令重建。依赖 PyMuPDF、ReportLab、pypdfium2 和 Pillow；本次使用 Mac 系统 Times New Roman 字体，其他平台可安装相同字体以保持排版。

```bash
python scripts/build_method_figure.py
```

原始输入保存在 `figures/source/fig2_method_v2_input.pdf` 和 `Figure3_v2_input.drawio`。当前输出为 `figures/fig2_method.pdf`、`figures/svg/fig2_method.svg`、`Figure3.drawio` 和 `delivery/Figure3.drawio`。`provenance/method_figure.json` 记录校验和。

实验表以用户 v2 稿为基础，已合并远端 LeWM clean 和 Rope 正式运行结果；旧 `update_comparison_tables.py` 用于此前结果重算，不属于本次排版重建流程，以免覆盖 v2 图注及表格组织。历史审计和修订说明保留供追溯；当前构建信息以 `provenance/build_report.json` 和 `provenance/appendix_evidence_revision.json` 为准。

Figure 4 可用 `python scripts/build_domain_figure4.py --paper .` 重建；素材来源见 `provenance/domain_figure4_manifest.json`；图中 Rope 固定强度可视化与正文动态强度主结果分别标注。

Figure 4 当前为八组完整场景：PushObj · ResNet、PushT · LeWM、Maze · AdaJEPA、Rope · DINO-WM；顶部图例标注 Without ECP / With ECP，灰色边界线已去除。Rope 使用目标 15/24 的新配对截图。Figure 1 顺序为 Frozen、LeWM、AdaJEPA；LeWM 三柱为 47.4%、61.4%、66.0%；可用 `python scripts/build_headline.py --data provenance/headline_data_latest.json` 重建。

## 论文写作约定

- 方法或配置名称首次出现时先解释含义。Frozen 需说明是离线预训练的 ResNet 世界模型、测试时不更新；使用冻结的 LeWM 或 DINO-WM 时明确其模型名称。独立图表注也要能单独读懂，避免把“冻结”误当作骨干网络。
- 实验设置只交代如何开展实验：模型、训练、任务、预算、指标和成功标准，不引用结果表格，不在 Settings 中展示或分析实验结果。
- 仅在帮助读者定位证据时引用图表或附录；相邻文字已引用的，不重复引用。修改后核对目标标签及完整 PDF 的链接，不能为了精简而留下无出处的关键结果。
- 正文 Rope 方法写 DINO-WM (ours)，说明预测并执行三次推动、每推一次观测、五次推动预热；不使用内部的 3/3 简称，成功标准为 CD<0.3。
- 保留数据和图像的实际实验来源；固定强度可视化不能改标为动态 S 主方法。重建时保持主文 8 页，并检查整份 PDF 的排版。

- Table 1 的同方法基线变体连续排列；两行 Ours 连续放在表格最后，并用一条横线与上方基线分开。
- 摘要聚焦问题、方法与关键结果；阈值敏感性等消融细节留在正文，避免在摘要重复。

- 摘要使用 “We propose”；启发关系以 “Inspired by DART and ACT, we …” 表述，说明具体训练与推理选择；报告 Rope 结果时注明 DINO-WM。

附录 Figure 6、7、8、10 可用 `python scripts/build_appendix_figures.py` 从随稿的已核验图片重建；无需运行仿真。Rope 图来自固定强度 S 的独立可视化记录，不改标为动态 S 的正式统计轨迹。

- 正文应引用能支持结论的附录实验，避免机械压低引用数量；本版有 16 处目标引用。
- 完整稿不超过 22 页。图表按信息量设置尺寸，共享相同起点、目标，避免重复展示。
- Overleaf 源码包只包含 CVPR 模板和编译依赖；修订 Markdown、审计文件、原始帧、draw.io 和生成缓存不放入该包。
