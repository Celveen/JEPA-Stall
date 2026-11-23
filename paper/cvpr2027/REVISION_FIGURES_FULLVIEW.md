# 2026-11-16：补齐 LeWM 基线与 Figure 4 全景

- 本机安全快进至 de78d6ec3c95dd999ad771f14ecf02d225683f70，保留现有论文改动。
- Figure 1 补齐 LeWM 随机数据基线：166/350（47.4%）→ 增强数据 215/350（61.4%）→ 增强数据 + ECP 231/350（66.0%），总提升 18.6 个百分点。图例 Base data 在图注区分 Frozen/AdaJEPA 的 released 模型与 LeWM 的 matched random-data 模型。逐任务 CSV 重新计数并核对发布哈希；引言、附录同步。
- Figure 4 恢复八组案例：仅 PushObj 标 ResNet；PushT 恢复 Red agent/Red goal，Maze 恢复 Frozen/AdaJEPA，Rope 保留 DINO-WM。
- 新 Rope 为目标 15/24：Frozen CD 0.236/0.219，Ours CD 0.169/0.147。均为双方成功、Ours 最终误差更低的独立可视化运行。两组 S 触发为 3/0 次；Native 5/5 与 Ours 3/3 的动作节奏不同，不作 S 单因素效果，也不混入统计。
- 所有 Figure 4 面板使用完整原始相机范围，无局部裁剪或放大图。PushObj/PushT 原生渲染时仅跳过四条静态边界线绘制；物体内部像素逐项比对不变，没有改动物体或轨迹、没有推进物理仿真。
- 表格、Figure 3、摘要和参考文献不变。正文 8 页、整篇 20 页、45 条参考文献；已编译并渲染检查。

## 重建

图 1：`python scripts/build_headline.py --data provenance/headline_data_latest.json`

图 4：`python scripts/build_domain_figure4.py --paper .`

需要重建素材时：`python scripts/prepare_fullview_assets.py --repo /path/to/JEPA-Stall --paper .`。依赖原项目的 Pymunk/pygame 渲染环境；不会运行模型评测。

最新审计：`provenance/fullview_revision_20261116.json`。历史修订说明保留供追溯。
