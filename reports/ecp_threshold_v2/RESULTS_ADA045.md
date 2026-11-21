# Augmented AdaJEPA: ECP threshold ablation on PushObj

All 350 PushObj EVAL50 tasks completed for the augmented AdaJEPA checkpoint (three training epochs). The planner uses 30 replans, five environment controls per replan, and at most 150 controls. Success is measured by the same evaluator and 30-replan cutoff for all four rows. The native score is reconstructed from the original per-iteration logs at that cutoff.

| Planner | Successes / 350 | Success rate | Change from native |
| --- | ---: | ---: | ---: |
| Native GD | 262 | 74.9% | — |
| ECP, upper threshold 0.15 | 269 | 76.9% | +7 tasks / +2.0 points |
| ECP, upper threshold 0.30 | 271 | 77.4% | +9 tasks / +2.6 points |
| ECP, upper threshold 0.45 | **272** | **77.7%** | **+10 tasks / +2.9 points** |

The 0.45 setting triggered on 108/350 tasks (30.9%). Relative to native GD on identical starts and goals, it rescued 11 tasks and regressed one. Relative to the 0.30 setting, it rescued three and regressed two, yielding one additional success. Its median first trigger was at replan 11. The detector's lower threshold was 0.15 and its upper threshold 0.45; the other planner and penalty parameters are recorded in [the protocol](PROTOCOL_ADA045.md).

| Shape | I | L | Square | Z | Small tee | Plus | T |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ECP 0.45 successes / 50 | 27 | 46 | 25 | 49 | 36 | 44 | 45 |

The [paired audit](completed_audit.json) checked all 35 chunks and 350 tasks. It verified source and checkpoint receipts, identical initial and goal states, identical actions through the first ECP intervention, and exact success counts against the evaluator summary. The [task-level table](paired_tasks.csv) records each paired outcome and first trigger. This is an exploratory threshold comparison on the same fixed task set used to inspect earlier settings; the one-task gain over 0.30 should be treated as a small ablation result rather than a separately validated improvement.

The complete [raw archive](ada045_complete_raw.tar.gz) contains per-task actions and states, evaluator logs, configs, receipts, summary, and the audit. Its [manifest](archive_manifest_ada045.json) provides the SHA-256 checksum. The stopped DINO-WM follow-up is documented [separately](CANCELLED_DINO.md) and contributes no result to this table.
