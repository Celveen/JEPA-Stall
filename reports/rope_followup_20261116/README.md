# Rope completed evaluation results

Ten conditions, ten fixed targets per condition (100 episode-condition results). All runs use the verified official pretrained DINOv2 encoder, frozen Rope predictor, observation after every push, and immediate success stopping at CD < 0.3.

| Condition | Predict / execute | Push cap | S warmup | Mean final CD | Success | S events | Changed tasks | Total executed pushes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Native | 5/1 | 20 | — | 1.004601 | 4/10 | — | — | 157 |
| S | 5/1 | 20 | 3 | 1.079430 | 4/10 | 2 | 1 | 157 |
| S | 5/1 | 20 | 5 | 1.006952 | 4/10 | 3 | 1 | 157 |
| Native | 3/3 | 20 | — | 0.311903 | 8/10 | — | — | 110 |
| Fixed S | 3/3 | 20 | 5 | 0.242980 | 8/10 | 17 | 4 | 111 |
| Dynamic S | 3/3 | 20 | 5 | 0.310912 | 8/10 | 17 | 4 | 108 |
| Native | 5/5 | 20 | — | 0.671488 | 7/10 | — | — | 124 |
| Native | 3/3 | 15 | — | 0.420563 | 7/10 | — | — | 91 |
| Fixed S | 3/3 | 15 | 5 | 0.411715 | 7/10 | 10 | 3 | 91 |
| Dynamic S | 3/3 | 15 | 5 | 0.417772 | 7/10 | 10 | 3 | 94 |

Changed tasks are compared with the Native control in the same protocol and run. Native detector events are diagnostic only and are not counted as S interventions.

## Same-trajectory budget comparison

These are prefixes of the original 20-push executions, not additional independent runs.

| 3/3 condition | CD at 15 / success | CD at 20 / success |
|---|---:|---:|
| native3 | 0.518860 / 6/10 | 0.311903 / 8/10 |
| fixed3 | 0.515425 / 6/10 | 0.242980 / 8/10 |
| dynamic3 | 0.515368 / 6/10 | 0.310912 / 8/10 |

## Interpretation and limits

- The independent 15-push rerun changes only the push cap in configuration, but independent physics can alter later plans. The paired trajectory prefixes isolate the stopping-budget effect: two additional tasks succeed at push 17.
- Fixed and dynamic S use the same strength on every observed trigger in both 3/3 runs (w=1, coefficient 2). Their CD differences do not establish superiority of either strength policy.
- The warmup-3 per-push run has no trigger at pushes 3 or 4. The two warmup policies only diverge after independent simulator feedback; their CD gap is not evidence of a warmup benefit.
- The per-push and chunk implementations also differ in episode isolation and random-stream organization. Their difference is not a single-factor cadence ablation.
- The old 50-target Rope pilot with CD 1.83849134 used an incorrectly initialized encoder and is invalid as a DINO-WM transfer result. The paper appendix still contains that old result and needs a separate manuscript correction; this results update does not rewrite the manuscript or add an unpublished-paper citation.

## Evidence and reports

- [Compact raw evidence](evidence.tar.gz): actual normalized controls and particle states, per-push CD, stopping masks, trigger/cost logs, configurations, and frozen-model/encoder audits. It contains 73 completed output directories (three ten-target batched runs plus 70 individual episodes). Model weight files and image/RNG replay caches are not included.
- [Evidence SHA-256 manifest](evidence_manifest.json); all 800 archived files were verified against their source hashes. Extract into an empty directory to inspect the original eval_outputs layout.
- [Machine-readable summary](summary.json) and [CSV](results.csv).
- [20-push chunk report](../rope_chunk3_strength_20261116/RESULTS.md), [15-push rerun](../rope_chunk3_cap15_20261116/RESULTS.md), and [budget comparison](../rope_chunk3_cap15_20261116/BUDGET_COMPARISON.md).
- [Per-push S warmup 3](../rope_feedback_pretrained_20261116/warmup3/RESULTS.md) and [warmup 5](../rope_feedback_pretrained_20261116/warmup5/RESULTS.md).
- [Encoder diagnosis](../rope_encoder_audit_20261116/DIAGNOSIS.md) and [same-control simulation variation example](../rope_chunk3_cap15_20261116/replicate_variation_example.json).
- [Chunk run instructions](../../experiments/rope_chunk3/README.md) and [per-push run instructions](../../experiments/rope_feedback/README.md).
