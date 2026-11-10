# Active DINO-WM PushObj evaluation

Started 10 November 2026, 10:50 Beijing time. Both epoch-3 checkpoints are complete; this run evaluates ordinary and augmented training data, each with native planning and ECP.

The shared protocol executes **25 environment controls after each plan, with at most 6 plans and 150 controls per task**. CEM remains 200 candidates x 10 iterations, with 30 elites. All four conditions and FP32 diagnostic controls use this cadence. `conf/dinowm_pushobj_official_cadence.yaml` is the common configuration for subsequent DINO-WM evaluations.

Real-simulator integration passed: paired pre-intervention actions, physical execution limits, early success termination, ECP activation and unchanged checkpoint weights. ECP retains a 50-control warmup and 15-control movement window using observations already returned by the simulator; the temporary boost occupies one complete 25-control plan.

Five main workers run on GPUs 0, 1, 3, 5 and 6. GPU 4 runs 14 full-task FP32 precision checks. The initial live CEM time is approximately 6.4 seconds per plan, or about 38 seconds of search for a task that reaches all six plans. The initial full-queue estimate is **3–4 hours** for 1,400 condition-task evaluations, including overhead, subject to completed-chunk throughput. Relative to the immediately preceding 30-plan configuration, maximum search work per task is reduced by 80%.

See [PROTOCOL.md](PROTOCOL.md), [validation_summary.json](validation_summary.json), [initial_timing_estimate.json](initial_timing_estimate.json), [pipeline_status.json](pipeline_status.json), and [precision_confirmation/status.json](precision_confirmation/status.json). Status files refresh every 30 seconds. Precision comparisons remain diagnostic; no FP32/BF16 accuracy equivalence is claimed.

Old-frequency outputs are retained separately. Benchmark summaries are generated after completion, and partial results have not been added to the paper.
