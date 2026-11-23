# Figure 3 可读性与 DART 归因修订

继续使用用户 v2 / Planning 稿的实验结果和版式。

- 第 4 步补回 Small / repeated move → Activate correction；第 5 步标明 Avoid revisits (S) 与 Boost goal weight (P)，保留持续记忆、三次调用重加权的时间条。
- 删除 Offline 灰色小字。接触方向箭头与扰动箭头配合 Contact bias + noise / Collect transitions，直接展示采集动作加入噪声。
- 训练端的噪声注入思想更接近 DART：摘要、引言与相关工作已修正并新增 DART 引用。本文实际使用 AdaJEPA 的接触偏置采集配方和固定高斯噪声，没有实现 DART 的噪声分布优化。DAgger 保留在相关工作，明确其迭代汇总 learner-visited states 上专家标注的机制。
- ACT 的时间组织启发保留。实验数字、其他插图及导师 LLM Usage Statement 保持不变。
- 正文 8 页，整篇 20 页；摘要 204 词；45 条文献，DART [23]、DAgger [31]、ACT [44]。
- 无未定义引用或溢出框；40 个字体对象均嵌入、无 Type 3。已渲染检查全部页面，并详查 Figure 3。

来源：[DART](https://proceedings.mlr.press/v78/laskey17a.html)、[DAgger](https://proceedings.mlr.press/v15/ross11a.html)。当前校验记录为 `provenance/figure3_clarity_dart.json`，此前修订记录保留作历史追溯。
