# When World Models Stall — v14

CVPR 2027 工作稿。正文 8 页，声明与 43 条参考文献共 2 页，附录 10 页。摘要 183 词。

[完整论文](paper_v14.pdf) · [主文](paper_main.pdf) · [附录](paper_appendix.pdf) · [源码包](paper_source_v14.zip) · [结果与日志](../../reports/publication_v14/README.md)

主表 DINO-WM CEM 的 PushT 行已改为随机 T16k 数据、epoch 3 模型：48 / 40 / 2 / 44 / 40 / 24 / 48%，六偏移均值 33.0%。GD 和 CEM 使用相同模型；CEM 保留其已运行的 300×30 搜索预算和 6×25 环境控制步。证据见 `provenance/dinowm_pusht_cem_e3_v14.json`。

主表将 PushObj 按七种形状展示，每种形状 50 个任务，并加粗 PushObj 和 PushT 的各列最高分。归档基线与干预实验使用不同的起点和目标，任务清单与预算见附录。PushT 的 56 组评测和 Maze 的完成结果均已录入。Figure 3 采用原始的模块布局，使用低饱和配色和圆角 encoder。

## 重建

```bash
python scripts/update_comparison_tables.py --results /path/to/JEPA-Stall/eval_outputs/pusht_visual_v8 --require-complete
python scripts/build_method_figure.py
python scripts/build_paper.py --tectonic /path/to/tectonic
```

重算表格前，按结果包 README 解压 PushT 已完成评测。重生成 Figure 3 需要仓库原始提交 `2de77b4`、`pypdf` 和 `pypdfium2`。直接编译时可使用随稿的表格和插图。`provenance/comparison_v14.json` 记录每种形状的成功数，`provenance/method_figure.json` 记录 Figure 3 的输入和输出校验值。
