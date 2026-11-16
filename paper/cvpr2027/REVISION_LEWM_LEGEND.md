# Figure 4 图例、LeWM 实景与 Figure 1 排序

- 安全同步 GitHub 的 LeWM clean CEM 72% 结果与 LeWM–PushT 真实配对画面，保留其表格与说明更新。
- Figure 4 顶部增加棕色 Without ECP、青绿色 With ECP 图例。c/d 替换为红色推手目标 8 / 红色背景目标 6 的 LeWM 原始仿真 PNG；基线均失败于 30 次重规划，ECP 分别成功于 28/13 次。
- e 使用已验证的 AdaJEPA 目标 75（与 Figure 2 共享该案例），f 保留 AdaJEPA 目标 28。没有把 Frozen 实验改名。
- 所有场景仍为完整视角。原始 LeWM PNG 字节不变，仅在矢量图中遮住六像素灰色边框，核对边缘没有彩色物体像素。红推手基线末态中推手位于视野外；不补画。
- Figure 1 按 Frozen、LeWM、AdaJEPA 排序，计数与提升幅度不变。
- 删除指定的两段图注说明，精简正文中重复的 contact/geometry 表述；附录集中保留必要的输入、目标定义及图像来源。
- 表格沿用本次同步后的版本；摘要、Figure 3、参考文献未改。正文 8 页，全文 20 页，45 条参考文献；已编译、渲染并检查。

重建 Figure 1：`python scripts/build_headline.py --data provenance/headline_data_latest.json`。
重建 Figure 4：`python scripts/build_domain_figure4.py --paper .`。
导入 LeWM 素材：`python scripts/prepare_lewm_figure4.py --repo /path/to/JEPA-Stall --paper .`。
当前审计：`provenance/lewm_legend_revision.json`。
