# 多布局迷宫：基线与卡住分析

## easy（50 个 episode）

- 冻结规划器 54%，AdaJEPA 68%（仅 AdaJEPA / 仅冻结 = 11/4，p=0.118）
- AdaJEPA 失败 16 个：结束时离目标中位 1.37（成功阈值 0.5）；最后 5 次重规划距离改善中位 0.04，智能体净移动中位 0.35
- 失败类型（最后 5 次重规划）：stall 7，moving, no progress 5，progressing 4
- 潜空间触发（阈值 0.15，未在迷宫上调）：碰到 13 个 episode = 11/16 个失败 + 2 个之后成功的

## medium（50 个 episode）

- 冻结规划器 22%，AdaJEPA 18%（仅 AdaJEPA / 仅冻结 = 3/5，p=0.727）
- AdaJEPA 失败 41 个：结束时离目标中位 1.76（成功阈值 0.5）；最后 5 次重规划距离改善中位 -0.00，智能体净移动中位 0.10
- 失败类型（最后 5 次重规划）：stall 29，moving, no progress 8，progressing 4
- 潜空间触发（阈值 0.15，未在迷宫上调）：碰到 36 个 episode = 36/41 个失败 + 0 个之后成功的

## hard（50 个 episode）

- 冻结规划器 8%，AdaJEPA 4%（仅 AdaJEPA / 仅冻结 = 0/2，p=0.5）
- AdaJEPA 失败 48 个：结束时离目标中位 1.98（成功阈值 0.5）；最后 5 次重规划距离改善中位 -0.01，智能体净移动中位 0.35
- 失败类型（最后 5 次重规划）：stall 23，moving, no progress 22，progressing 3
- 潜空间触发（阈值 0.15，未在迷宫上调）：碰到 42 个 episode = 42/48 个失败 + 0 个之后成功的

