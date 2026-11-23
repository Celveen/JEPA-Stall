# Rope 主结果与表格整理

- Table 1 按同方法连续排列：DINO-WM GD/CEM、LeWM GD/CEM、Frozen released/Ours、AdaJEPA released/Ours。删除 released 与 Ours 之间的横线。
- DDP-WM 缺 clean，移出主表；其六种视觉扰动结果保留在附录。检索 SE-JEPA 远端分支及报告包后，找到 R3M 与 DINO CLS 的完整六扰动记录，未找到 clean。原始初始/目标状态与当前 50 任务列表完全一致，六扰动平均成功率为 21.7% / 34.7%。原始 NPZ、配置、来源提交和校验和一并保存。
- 正文 Table 5 合并 Maze 与 Rope，并标注 Success↑、CD↓。Rope 主比较使用相同的三次推动预测及执行长度，每次推动后观测和检测成功。Ours 为动态 S、五次推动预热；不是以“3/3”作为方法名称。
- Rope：DINO-WM native 的平均 CD 0.755、成功率 50%；Ours 为 0.515、60%，平均 CD 降低 31.7%。20 个目标、最多 20 次推动，成功阈值 CD<0.3。
- 五次预测/执行、固定强度、无额外预热放入附录消融。动态 S 的 CD 最低；五次执行 native 的成功率最高，结果完整保留。
- 摘要、引言、实验设置和结论同步更新。Figure 4 的 Rope 图像仍为原有固定强度可视化运行，图注明确变体，没有将其改标为动态 S。
- 所有图像资产不变。附录距离曲线位置前移，避免单张小图独占一页；交叉引用自动更新。
- 主文 8 页，声明及 47 条参考文献 2 页，附录 11 页，总计 21 页；摘要 210 词。编译无未定义引用或溢出，字体全部嵌入，逐页渲染已检查。
- 本次只整理已存在的结果，未运行新实验；LLM Usage Statement 保留导师版本。

当前审计：`provenance/rope_main_revision.json`。新增对比来源：`provenance/additional_baseline_audit.json`。
Rope 原始来源：仓库 `reports/rope_formal20_native3/`。
重建：`python scripts/build_paper.py --tectonic /path/to/tectonic`。
