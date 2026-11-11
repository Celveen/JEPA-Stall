# Completed DINO-WM PushObj evaluation

Completed 10 November 2026 at 13:11 Beijing time: **140/140 chunks, 1,400 task runs**. Both checkpoints were trained for three epochs and evaluated with frozen weights on the same 350 tasks.

| Training data | Native CEM | CEM + ECP | ECP-triggered tasks |
|---|---:|---:|---:|
| Matched ordinary actions | 24.6% (86/350) | 24.6% (86/350) | 2/350 |
| Action-biased augmentation | 48.9% (171/350) | 48.9% (171/350) | 1/350 |

[Full results, per-shape table, raw data and logs, and reproducible audit](completed_results/README.md).

The protocol executes **25 environment controls after each plan, with at most 6 plans and 150 controls per task**. CEM uses 200 candidates x 10 iterations, 30 elites, BF16 candidate prediction and FP32 objective/search updates. The planning mean is approximately 6.30 seconds; the five-worker queue completed in 2 hours 21 minutes.

ECP uses a 50-control warmup, 15-control movement window, threshold interval 0.05/0.15, penalty coefficient 2 and temporary proprioceptive multiplier 10. All 700 native/ECP pairs have identical targets and pre-intervention actions. Three tasks change their actions after intervention; their success labels remain unchanged.

Separate FP32 checks completed on 14 tasks, with six successes under each precision and no changed success labels. These diagnostic tasks are excluded from benchmark counts and do not establish population-level precision equivalence.

See [PROTOCOL.md](PROTOCOL.md), [pipeline_status.json](pipeline_status.json), [validation_summary.json](validation_summary.json), and [precision_confirmation/summary.json](precision_confirmation/summary.json). Historical v1/v2 results use different deployment protocols and remain separate.

A follow-up authorized on 11 November evaluates the 0.10/0.30 threshold interval and a denser execution cadence. It is ongoing and is not included in these completed results.
