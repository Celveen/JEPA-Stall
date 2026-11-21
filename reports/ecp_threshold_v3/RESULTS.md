# PushObj threshold follow-up: complete paired results

Both conditions completed all 350 EVAL50 tasks (seven shapes, 50 each) with our three-epoch action-biased checkpoint, 30 MPC replans, five controls per replan, and GD 100. The evaluation queue completed 70/70 chunks without task failures. After its coordinator exited with 56 complete chunks, the [resume script](resume.py) revalidated those outputs and ran only the 14 missing Frozen chunks. A CPU-only [audit](completed_audit.json) independently verified the checkpoint hash, source receipts, unchanged frozen model state, identical starts and goals, and action equality through the first ECP intervention on all 700 task runs.

| Model update | Planning setting | Success / 350 | Success rate | ECP triggers |
| --- | --- | ---: | ---: | ---: |
| Frozen | Native | 214 | 61.1% | — |
| Frozen | ECP, lower/upper 0.15/0.45 | **253** | **72.3%** | 136 |
| AdaJEPA | Native | 262 | 74.9% | — |
| AdaJEPA | ECP, lower/upper 0.05/0.15 | 269 | 76.9% | — |
| AdaJEPA | ECP, lower/upper 0.10/0.30 | 271 | 77.4% | 101 |
| AdaJEPA | ECP, lower/upper 0.15/0.45 | **272** | **77.7%** | 108 |
| AdaJEPA | ECP, lower/upper 0.20/0.60 | 271 | 77.4% | 111 |

The new Frozen 0.45 run rescued 39 tasks that its native planner failed and regressed none. The new AdaJEPA 0.60 run rescued 11 native failures and regressed two, for a net gain of nine tasks over native. Compared task by task with AdaJEPA 0.45, AdaJEPA 0.60 rescued three and regressed four; its total is one task lower.

| Condition | I | L | Square | Z | Small tee | Plus | T | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen ECP 0.45 | 23 | 42 | 23 | 46 | 31 | 44 | 44 | 253 |
| AdaJEPA ECP 0.60 | 27 | 46 | 25 | 50 | 35 | 43 | 45 | 271 |

The earlier Frozen ECP 0.15 result, scored at the same 30-replan cutoff, is also 253/350. The new 0.45 run improves two of those tasks and loses two others. That earlier source run allowed 40 replans and was censored at 30; the new run directly caps planning at 30. These comparisons reuse a fixed task set that has already been inspected, so the one-task differences across AdaJEPA thresholds are exploratory, not evidence of a reliably optimal threshold.

Reproducibility: [protocol](PROTOCOL.md), [700-row task table](paired_tasks.csv), [independent audit](completed_audit.json), [full raw output archive](threshold_v3_complete_raw.tar.gz), and [archive checksum manifest](archive_manifest.json). The raw archive retains actions, states, per-iteration logs, resolved configs, trigger records, source and checkpoint receipts, and the complete evaluator summary. Checkpoints are excluded from the archive; the source checkpoint path and hash are in the evaluation manifest.
