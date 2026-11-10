# 论文图形设计与来源审计

采用 `figure-designer` skill，设计参考 LeWM、Temporal Straightening 和 AdaJEPA 的简洁论文图风格，原创布局，不复刻任何参考图。使用 Matplotlib 矢量文字、线条、边框，嵌入原始模拟器生成的真实场景帧。

## Figure 1：动机与运行例子

- 类型：motivated example；范式：matched baseline failure versus intervention success。选择真实匹配任务，直接展示物体到位但推杆没有完成的具体失败；不用训练流程代替动机，也不用小幅收益柱状图作开篇。
- 尺寸：双栏宽 6.875 × 3.0 inch。两行分别 Frozen MPC、Triggered correction，四列 Start、Before correction、Final state、Goal。
- 实体：PushObj `chunk_04_I / sample_5`，相同初始状态、目标、模型；前15次执行动作及状态逐项相等。基线30次失败，T0SP在17次成功。同30次预算，提前成功不重复延展。
- 全部帧来自日志中的真实物理状态，通过原Pymunk绘图器重新渲染。没有执行策略、运行新规划或训练，也未运行物理推进来改变日志状态；只赋值刚体姿态再绘制。原日志保留三位小数，因此应称“logged-state simulator renderings”，不声称是运行当时保存的逐像素截图。
- 展示灰色物体、蓝色推杆以及独立目标观测；没有人为添加目标状态给策略。截图保留完整环境、不裁剪失败区域、不改变原始配色。

## Figure 2：方法概览

- 类型：solution overview；范式：two-level architecture。训练端数据与规划端反馈是两个接口，划分一条细灰线。
- 上层：Interaction-biased data collection → unchanged JEPA+TS learning → encoder/predictor。
- 下层：Observation and goal → latent MPC → execute action prefix → simulator；实际transition形成stall feedback，反馈激活execution-conditioned correction并回到MPC。
- 文字明确proprioceptive reweighting、optional action correction、local memory；没有绘制训练正则有效、通用phase classifier或自动准确识别停滞原因的未证实模块。
- 只有蓝灰基础模块与单一暖色反馈模块；没有图标、渐变、阴影或神经元装饰。

## Figure 3：迷宫真实案例（建议附录）

- 类型：qualitative experiment；范式：同一任务的baseline/intervention序列。
- medium_s100开发集sample75，原AdaJEPA与SZ-switch共用初始模型和前六次执行；基线30次失败，干预27次成功。
- 日志状态重新载入原MuJoCo 2.1模拟器，使用原camera与颜色。渲染分辨率提高到512，原目标与起始观测224图像另行保存用于核验。CPU OSMesa，未使用GPU预算。
- 每一步只set_state再render，没有step或rollout。迷宫布局由原目标生成器恢复，并用存储的start/goal数组逐项校验。两组所有start/goal数组也已校验完全相等。
- 该图是一个案例；是否普遍有效由正文完整100任务统计给出，不从个案推出显著性。

## 通用审计

- 矢量格式：通过。PDF/SVG的文字、框、箭头为矢量；模拟器帧按用户要求保留真实栅格图（这属于场景照片/截图的必要栅格内容，不是将整图栅格化）。
- 字号：通过。按设定双栏宽插入，最小8pt；主体9pt，标题10pt。单栏距离图宽3.25inch，最小8pt。正文不要将双栏图缩成单栏。
- 字体：STIXGeneral与论文Times风格匹配；TrueType子集嵌入PDF。数学字体STIX。
- 色盲友好：标注仅蓝灰与暖橙；不同方法也有行标签、成功/失败文字。真实模拟器颜色未改动。迷宫代理为绿色，但已在文字中说明且通过位置与几何形状辨别。
- 自含caption：通过。各caption首句表达该图支持的具体结论，说明任务、对照、预算与状态重渲染来源。
- 轴范围：Figure1–3无数值坐标轴；补充距离曲线固定横轴0–30、纵轴0–260px，从零开始，未裁剪失败或阈值。
- 无装饰：通过。无3D统计图、渐变、阴影或复杂背景；真实模拟器背景保留。
- 视觉检查：已查看PNG预览，去掉曾与底部时间标签挤占空间的footer，保留caption解释。
- 完整性：源码输入文件SHA256、case IDs、状态值、前缀配对校验、成功/失败及终点距离均见 `source_manifest.json`。`build_figures.py`可重生成。

## 稿件装配检查

无CRITICAL/MAJOR设计问题。主文应将Figure1同一案例引入引言，方法图术语与章节标题保持一致；这些依赖主稿的检查由装配者完成。案例与汇总结果的预算必须都写30次，50次迷宫结果只能在额外预算分析单列。

## 可选 Figure 4：结果概览

两面板分开展示PushObj四种检查点/规划器条件与Maze单独变体；横轴明确是条件，不是四个独立复现。全部数字从稿件provenance/evidence.json读取，预算严格30次。PushObj350共享任务，三组柱分别基线、仅加权、加权+memory；Maze100任务只对比AdaJEPA与SZ-switch memory，无本体boost。共享0–100%轴，不对小幅Maze差异截断放大；不绘制不存在的跨种子方差。caption说明配对共享任务与变体差异。字体最低8pt，主图区6.875×2.75inch，复现脚本build_results.py和结果清单results_figure_manifest.json独立存储。
