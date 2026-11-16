# When World Models Stall: Execution-Conditioned Planning in Latent Space

当前稿以用户提供的 CVPR2027 revised_v2 / Overleaf_v2 为准，采用 inspired-by 措辞，并在 Figures 3/4 加入 PushT 视觉偏移和真实 Rope 仿真画面。正文 8 页，声明与 45 条参考文献 2 页，附录 10 页；摘要 205 词。训练端噪声注入的启发改为 DART [23]；DAgger [31] 保留作相关工作，ACT [44] 为时间组织启发，LLM Usage Statement 保留导师版本。表格沿用 v2；Figure 1 补入新发布的 LeWM 随机数据基线，未在本次编辑中运行新实验。详见 [本次修订](REVISION_FIGURES_FULLVIEW.md)。

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

实验表采用用户 v2 快照；旧 `update_comparison_tables.py` 用于此前结果重算，不属于本次排版重建流程，以免覆盖 v2 图注及表格组织。历史审计和修订说明保留供追溯；当前构建信息以 `provenance/build_report.json` 和 `provenance/fullview_revision_20261116.json` 为准。

Figure 4 可用 `python scripts/build_domain_figure4.py --paper .` 重建；素材与两类图像来源的详细区别见本次修订说明。数据表格保持不变。

Figure 4 当前为八组完整场景：PushObj · ResNet、PushT · Red agent/Red goal、Maze · Frozen/AdaJEPA、Rope · DINO-WM；灰色边界线已去除。Rope 使用目标 15/24 的新配对截图。Figure 1 LeWM 三柱为 47.4%、61.4%、66.0%；可用 `python scripts/build_headline.py --data provenance/headline_data_latest.json` 重建。
