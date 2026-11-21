# 方法名称、实验设置与引用检查

- 全文及独立图表注明确 Frozen：离线预训练的 ResNet 世界模型，测试时权重固定。AdaJEPA 表示在线更新同类世界模型；Native 表示不加 ECP 的原始规划器。Released、Data aug.、Ours 也在实验设置中定义。
- Rope 图中的 Frozen CD 单独解释为参数固定的 DINO-WM 原始规划，未混称 ResNet，也未把已有固定强度图片改标为动态 S。
- Maze 与 Rope 拆为正文 Table 5、Table 6；Rope 行名改为 DINO-WM (ours)，表注注明 CD<0.3 为成功，删除解释箭头方向的句子。
- Experimental setup 只描述模型、训练、规划预算和评价方式，不引用图表、不展示结果。正文指向附录的目标引用从 20 处减为 7 处，保留关键证明、训练细节与消融入口。
- 核对所有保留的交叉引用和参考文献；完整 PDF 的内部链接及对应标签页均检查通过。
- 修正 Rope 观测步与 ResNet 重规划步的区分，以及 terminal 目标下的正则缩放说明。实验数据及图像资产不变。
- 精简重复结果描述，明确 PushT 72%→82% 对应增强数据模型；未来研究的具体计划集中到附录。
- 主文 8 页，参考文献及声明 2 页，附录 11 页；共 21 页、47 条参考文献，摘要 217 词。已编译、逐页渲染并检查，无溢出、缺失引用或尾部空白页。

当前审计：`provenance/clarity_revision.json` 与 `provenance/reference_audit.json`。
后续写作遵循 README 中的“论文写作约定”。
