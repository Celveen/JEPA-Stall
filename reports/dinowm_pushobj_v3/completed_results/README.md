# DINO-WM PushObj: completed three-epoch evaluation

Completed 10 November 2026 at 13:11 Beijing time. All 140 chunks and 1,400 task runs are complete and have been re-audited from raw files. Each condition uses the same 350 tasks, with 50 for each of seven shapes.

| Training data | Native CEM | CEM + ECP | ECP-triggered tasks |
|---|---:|---:|---:|
| Matched ordinary actions | 86/350 (24.6%) | 86/350 (24.6%) | 2/350 |
| Action-biased data augmentation | 171/350 (48.9%) | 171/350 (48.9%) | 1/350 |

Augmentation adds 85 successes (+24.3 percentage points) under the matched three-epoch training and evaluation protocol. All seven shapes improve. ECP changes the action sequence in the three triggered tasks but yields no rescues or regressions in this run. The present threshold barely activates on this backbone; these outcomes do not establish the benefit of the planning intervention on DINO-WM.

## Per-shape success (%)

| Data / planner | I | L | Square | Z | Small T | Plus | T | Overall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Ordinary / native | 8.0 | 42.0 | 4.0 | 20.0 | 24.0 | 34.0 | 40.0 | 24.6 |
| Ordinary / ECP | 8.0 | 42.0 | 4.0 | 20.0 | 24.0 | 34.0 | 40.0 | 24.6 |
| Augmentation / native | 16.0 | 72.0 | 14.0 | 46.0 | 56.0 | 66.0 | 72.0 | 48.9 |
| Augmentation / ECP | 16.0 | 72.0 | 14.0 | 46.0 | 56.0 | 66.0 | 72.0 | 48.9 |

## Protocol and checks

- Frozen epoch-3 checkpoints; CEM 200 candidates x 10 iterations, 30 elites; BF16 candidate prediction and FP32 encoding/objective/search updates.
- Maximum 6 plans x 25 environment controls = 150 controls; successful episodes terminate early. This matches the requested execution cadence and retains the approved search budget, rather than reproducing every official DINO-WM default.
- ECP movement interval 0.05/0.15, 50-control warmup, 15-control window, penalty coefficient 2, radius coefficient 0.25, temporary proprioceptive multiplier 10. Maximum four detection opportunities.
- All 700 native/ECP pairs have identical start/goal states and identical action prefixes before intervention. Full action sequences are identical in 348/350 ordinary-data and 349/350 augmented-data pairs.
- A separate 14-task FP32/BF16 check gives six successes in each precision with no changed success labels. This diagnostic subset is excluded from benchmark totals and does not establish population-level precision equivalence.

## Files and reproduction

- [summary.json](summary.json): original completed aggregation.
- [by_shape.csv](by_shape.csv): counts and rates for all 28 condition/shape cells.
- [audit.json](audit.json): independently recomputed results, trigger statistics, pairing checks, and raw-file hashes.
- [complete_raw.tar.gz](complete_raw.tar.gz): full numerical records and logs; [archive_manifest.json](archive_manifest.json) records its hash and scope.
- [audit_completed.py](audit_completed.py): CPU-only audit; [package_raw.py](package_raw.py): archive builder.

Extract from the repository root and reproduce the audit:

```bash
tar -xzf reports/dinowm_pushobj_v3/completed_results/complete_raw.tar.gz
python reports/dinowm_pushobj_v3/completed_results/audit_completed.py --repo . --output /tmp/dinowm-v3-audit
```

Raw executed actions and simulation states, per-iteration logs, configurations, completion receipts, timing records, and separate validation diagnostics are included. Model binaries, training images and duplicate rendered observation caches are omitted; checkpoint identity hashes are recorded in the completion receipts.

The earlier `publication_v14` progress archive is a historical partial snapshot, superseded for this experiment by this completed package. It must not be pooled with the completed records. The new 0.30-threshold and increased-detection-frequency follow-up is a separate ongoing experiment and is excluded here.
