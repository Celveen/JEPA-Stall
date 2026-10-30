# E3：medium_s100，stage1

参照运行：`eval_outputs/stag_maze100_ar4/medium/adajepa/ref`（AdaJEPA，100 个 episode）。各组只重跑参照里触发过卡住的 episode，其余与参照相同。赢 = 参照失败、这一组成功；输 = 参照成功、这一组失败；p 为 McNemar 精确双侧检验。

| 组 | 重跑 | 赢 | 输 | p | 20 次：成功（参照） | 30 次：成功（参照） | 50 次：成功（参照） | 切换的 episode | 切换步中位 |
|---|---:|---:|---:|---:|---|---|---|---:|---:|
| R-trig | 51 | 5 | 4 | 1.000 | 50（51） | 66（63） | 83（82） | 51/51 | 6.0 |
| R-switch | 51 | 4 | 1 | 0.375 | 52（51） | 65（63） | 85（82） | 23/51 | 6.0 |

变化的 episode（E0 类别是参照运行里的结局）：

| 组 | episode | 赢/输 | E0 类别 | 切换步 | 参照成功于 | 这一组成功于 |
|---|---:|---|---|---:|---:|---:|
| R-trig | 0 | 输 | 成功 | 4 | 29 | None |
| R-trig | 4 | 赢 | trap | 17 | None | 42 |
| R-trig | 7 | 赢 | trap | 7 | None | 18 |
| R-trig | 29 | 输 | 成功 | 9 | 31 | None |
| R-trig | 35 | 输 | 成功 | 5 | 36 | None |
| R-trig | 39 | 赢 | trap | 5 | None | 22 |
| R-trig | 51 | 输 | 成功 | 5 | 47 | None |
| R-trig | 68 | 赢 | open | 10 | None | 43 |
| R-trig | 69 | 赢 | trap | 4 | None | 43 |
| R-switch | 4 | 赢 | trap | 24 | None | 49 |
| R-switch | 10 | 赢 | trap | 31 | None | 49 |
| R-switch | 39 | 赢 | trap | 6 | None | 34 |
| R-switch | 80 | 赢 | trap | 6 | None | 28 |
| R-switch | 84 | 输 | 成功 | 6 | 48 | None |
