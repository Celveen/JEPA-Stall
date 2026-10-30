# E3：pushobj_S7，stage1

参照运行：`eval_outputs/escape_S7/none` + `eval_outputs/stag_S7_it50/S0`（AdaJEPA，280 个 episode）。各组只重跑参照里第 10 次之前没成功的 episode，其余与参照相同。赢 = 参照失败、这一组成功；输 = 参照成功、这一组失败；p 为 McNemar 精确双侧检验。

| 组 | 重跑 | 赢 | 输 | p | 20 次：成功（参照） | 30 次：成功（参照） | 50 次：成功（参照） | 切换的 episode | 切换步中位 |
|---|---:|---:|---:|---:|---|---|---|---:|---:|
| R-trig | 78 | 7 | 0 | 0.016 | 219（214） | 222（217） | 224（217） | 65/78 | 14.0 |
| R-switch | 78 | 7 | 0 | 0.016 | 215（214） | 220（217） | 224（217） | 61/78 | 25.0 |

变化的 episode（E0 类别是参照运行里的结局）：

| 组 | episode | 赢/输 | E0 类别 | 切换步 | 参照成功于 | 这一组成功于 |
|---|---:|---|---|---:|---:|---:|
| R-trig | chunk_00_I/3 | 赢 | — | 23 | None | 25 |
| R-trig | chunk_04_small_tee/16 | 赢 | — | 14 | None | 31 |
| R-trig | chunk_05_plus/9 | 赢 | — | 18 | None | 28 |
| R-trig | chunk_06_T/1 | 赢 | — | 17 | None | 18 |
| R-trig | chunk_08_T/12 | 赢 | — | 14 | None | 16 |
| R-trig | chunk_09_I/13 | 赢 | — | 15 | None | 17 |
| R-trig | chunk_09_I/16 | 赢 | — | 21 | None | 43 |
| R-switch | chunk_00_I/3 | 赢 | — | 27 | None | 43 |
| R-switch | chunk_00_I/7 | 赢 | — | 22 | None | 43 |
| R-switch | chunk_04_small_tee/16 | 赢 | — | 33 | None | 35 |
| R-switch | chunk_05_plus/9 | 赢 | — | 25 | None | 27 |
| R-switch | chunk_06_T/1 | 赢 | — | 21 | None | 22 |
| R-switch | chunk_08_T/12 | 赢 | — | 19 | None | 29 |
| R-switch | chunk_09_I/13 | 赢 | — | 37 | None | 38 |
