# Figure 4 标题、标签与案例修订

- 小标题统一为“数据集 · backbone/model”：PushObj、PushT、Maze 使用 ResNet，Rope 使用 DINO-WM。Frozen/AdaJEPA 区别以及 PushT 颜色偏移类型移入图注。
- Rope 标签改成 Frozen CD: 0.716 / Ours CD: 0.084，颜色与其他案例的基线棕色、方法青色一致。
- 删除原 h 的退步案例，保留目标 13 的改善案例。共七组，下排三组居中。目标 17 原始记录和图像保留在素材与证据清单。
- Rope 改善案例比较 native 5/5 和 fixed-S 3/3 动作节奏，S 未触发；不作为 S 单因素收益，也不加入统计。图注和附录同步。
- 现有 LeWM + ECP 配对实验属于 PushObj；未找到 LeWM–PushT 有/无 ECP 的配对图像或完整轨迹，因此没有把 ResNet 图标成 LeWM。
- 当前还缺第二组 Rope 改善的配对 RGB，以及 LeWM–PushT 的配对执行记录。拿到对应材料后可补全。
- 表格、Figure 3、参考文献保持不变。正文 8 页、整篇 20 页；45 条参考文献；无未定义引用、溢出框或字体嵌入问题。

生成图：`python scripts/build_domain_figure4.py --paper .`。当前审计：`provenance/figure4_backbones_revision.json`。
