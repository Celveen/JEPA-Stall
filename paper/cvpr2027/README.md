# When World Models Stall — v14

CVPR 2027 工作稿，2026-11-10。正文 8 页，声明/参考文献 1 页，附录 10 页；摘要 183 词。

[完整论文](paper_v14.pdf) · [主文](paper_main.pdf) · [附录](paper_appendix.pdf) · [源码包](paper_source_v14.zip) · [结果与日志](../../reports/publication_v14/README.md)

主表 DINO-WM CEM 的 PushT 行已改为随机 T16k 数据、epoch 3 模型：48 / 40 / 2 / 44 / 40 / 24 / 48%，六偏移均值 33.0%。GD 和 CEM 使用相同模型；CEM 保留其已运行的 300×30 搜索预算和 6×25 环境控制步。证据见 `provenance/dinowm_pusht_cem_e3_v14.json`。

官方及增强训练的 Frozen/AdaJEPA、ECP 开关，共 56 组 PushT 评测已经完成并填入第 6 页主表和第 19 页附录。Maze 的完成结果也已录入。PushObj 的 Reference/Paired 是两份不同的任务清单，分别展示；新 DINO-WM PushObj 评测仍运行，独立归档进度。

## 重建

```bash
python scripts/import_dinowm_cem_results.py --repo /path/to/JEPA-Stall
python scripts/update_comparison_tables.py --results /path/to/JEPA-Stall/eval_outputs/pusht_visual_v8 --require-complete
python scripts/build_paper.py --tectonic /path/to/tectonic
```

先按结果包 README 解压已完成评测。以上命令只读取已有结果和构建论文，不启动实验。原图仍为 v12 的真实仿真回放和轨迹。`REVISION_V14.md`、`provenance/integrity_v14.json` 记录本版修改与核验。
