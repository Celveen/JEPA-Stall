# Active DINO-WM PushObj evaluation

Started 10 November 2026, 10:25 Beijing time. Both epoch-3 training checkpoints are complete; this is evaluation only.

- Budget: 200 CEM candidates, 10 CEM iterations, 30 elites; at most 30 MPC replans, executing 5 environment controls per replan.
- Four conditions: ordinary/native, ordinary/ECP, augmented/native, augmented/ECP; 350 tasks each.
- BF16 candidate prediction; FP32 observation encoding, objective and search updates. All four arms share the numerical protocol.
- Main workers: GPUs 0, 1, 3, 5 and 6. Initial live CEM time is approximately 6.2–6.3 seconds per replan, versus approximately 103–105 seconds in the superseded 300x30 FP32 run. Initial full-run estimate: 16–20 hours, to be recalibrated with completed chunks.
- GPU 4 performs 14 complete FP32 precision spot checks, paired to the first task of each shape for both checkpoints. These are excluded from main totals.

See [PROTOCOL.md](PROTOCOL.md), [validation_summary.json](validation_summary.json), [pipeline_status.json](pipeline_status.json), and [precision_confirmation/status.json](precision_confirmation/status.json). Status files update every 30 seconds. Candidate precision changes action sequences; short integration validation passed, while complete-task precision comparisons are still running.

Old outputs are retained separately. No paper result table has been updated with partial results.
