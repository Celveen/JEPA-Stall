# When World Models Stall: Execution-Conditioned Planning in Latent Space

当前稿沿用用户提供的 v3 写作版本，并更新 Rope 三种子结果。Rope 主方法为固定强度 S（强度 2）、预热五次推动；动态强度、无额外预热和五推规划放在附录作消融。相同 20 个目标在三个种子下重复运行，每种方法 60 次：CD 0.6865 → 0.4793，成功率 55.0% → 71.7%，平均推动次数 12.68 → 11.30（包含全部成功及失败运行）。全文 21 页：正文 8 页，声明与 47 条参考文献 2 页，附录 11 页；摘要 213 词。

[完整论文](paper.pdf) · [主文](paper_main.pdf) · [附录](paper_appendix.pdf) · [完整源码](paper_source_v14.zip) · [摘要](Abstract.txt) · [可编辑 Figure 3](Figure3.drawio)

## 当前版本

- 已提交的正文、声明与参考文献共 10 页逐像素保留；附录为 11 页。正文引用和参考文献页码回链保持正确，全文有 21 张表、13 张图和 16 个展示公式。
- 附录将实验目的和解读与对应图表一起排版。Figure 8、9 使用完整单栏宽度，Figure 10–13 使用完整跨栏宽度。Figure 11 移除最下方 k、l 两个案例，保留七种 PushObj 形状与三个 Maze 案例；保留区域与原图逐像素一致。
- Figure 10 的两组 Rope 观察对齐第 6、12、20 推，使用相同目标与三推规划节奏。第一组 ECP 成功，第二组获得更低 CD；图注明确给出两组的停止结果。原始素材保持完整视角。
- 方法图去除未嵌入的默认 Helvetica 资源，修正花体 ℛ 的复制映射，图像外观不变；图内文字保持可选择文本，SVG 也使用文本元素。全部图 PDF 和论文 PDF 的字体资源已检查嵌入。
- Overleaf 包只包含 CVPR 模板与实际编译依赖，不含 Markdown、审计文件、原始帧或生成缓存。

本次排版与原图核验见 `provenance/appendix_layout_review.json`，完整构建结果见 `provenance/build_report.json`。实验、字体及引用记录分别见 `provenance/rope_three_seed_revision.json`、`provenance/figure_font_audit.json`、`provenance/reference_audit.json`。

## 重建

直接编译：`python scripts/build_paper.py --tectonic /path/to/tectonic`。

Figure 3：`python scripts/build_method_figure.py`；PDF、SVG 与 draw.io 来自相同绘图元素。依赖 PyMuPDF、ReportLab、pypdfium2 和 Pillow；使用 Mac Times New Roman 字体，字体文件不可用时回退至随 ReportLab 提供的 Vera。

Figure 4：`python scripts/build_domain_figure4.py --paper .`。附录运行图：`python scripts/build_appendix_figures.py`。重建使用已核验的随稿帧，不运行新实验。Rope 帧可用 `python scripts/prepare_rope_matched_visuals.py --repo /path/to/JEPA-Stall` 从原始压缩记录重新导出，来源见 `provenance/rope_fixed_visuals_seed101.json`。

Figure 11、13：`python scripts/build_maze_supplement.py`。Figure 11 的生成器保留 a–j 十个案例，Figure 13 使用既有 MuJoCo 运行记录。

Figure 1 顺序为 Frozen、LeWM、AdaJEPA，LeWM 三柱为 47.4%、61.4%、66.0%；可用 `python scripts/build_headline.py --data provenance/headline_data_latest.json` 重建。LeWM PushT clean CEM 为 72%（36/50），原始结果见 [完整记录](../../reports/lewm_pusht_clean_20261116/README.md)。

## 论文写作与导出约定

- 首次出现时解释方法与配置名称。Frozen 指离线预训练、测试时参数固定的 ResNet 世界模型；其他冻结骨干明确写模型名称。独立图表注需能单独理解。
- 实验设置说明如何开展实验，不分析结果。图表或附录引用用于定位必要证据，避免相邻重复；核对目标与 PDF 链接。
- Rope 主方法写 DINO-WM (ours)：预测并执行三次推动，每推一次观测，五次推动预热、固定记忆强度 2，CD < 0.3 为成功。示例与统计须注明各自实际来源和方法配置。
- Table 1 同方法变体连续排列，最后两行 Ours 用一条横线与基线分开。摘要使用 We propose，保留 DART/ACT 启发关系及代码提供说明。
- 主文 8 页，完整稿不超过 22 页。图中标注保持文本、字体嵌入；截图仅用于真实场景观察。不要增加重复案例或无关的导出说明。
