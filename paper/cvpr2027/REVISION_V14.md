# v14：DINO-WM CEM PushT 更新与发布

主表 DINO-WM CEM 的 PushT 部分改用随机 T16k 数据训练、epoch 3 检查点：Clean 48、Blur 40、S&P 2、Dark 44、Agent 40、Object 24、Goal 48，六偏移均值 33.0%。GD 与 CEM 使用同一模型哈希；PushObj 数值不受此次行内替换影响。

附录同步说明 CEM 300 candidates × 30 iterations、30 elites、6×25 控制步、terminal objective。源码快照目录中的 e10 为历史路径；checkpoint receipt、完成汇总和单次运行协议均标识 epoch 3。

五种条件的完整原始 NPZ 已下载并核对 Git blob、文件 SHA-256、成功数、相同的初始/目标状态及冻结参数。Object/Goal 两种条件在源仓库中只有完成汇总（12/50、24/50），已明确标注证据粒度；不声称对这两项做过逐任务原始复核。

保留全部已完成 Frozen/AdaJEPA 结果和原生仿真图。正文 8 页，声明/参考文献 1 页，附录 10 页，无溢出或未定义引用。实验结果、配置、日志、训练/数据元信息和正在运行的 DINO-WM PushObj 快照分别归档，见 reports/publication_v14。未新增实验。
