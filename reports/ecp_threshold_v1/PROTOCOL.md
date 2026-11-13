# PushObj ECP threshold and detection-frequency validation

Authorized on 2026-11-11: evaluate augmented-data AdaJEPA and DINO-WM with movement threshold 0.30, and an augmented frozen model with more detection opportunities. GPU 2 is excluded.

## Design

The user-specified upper threshold changes from 0.15 to 0.30. The corresponding lower threshold changes proportionally from 0.05 to 0.10. Penalty coefficient 2, radius coefficient 0.25, proprioceptive multiplier 10, 50-control warmup, 15-control observation window, and a 150-control maximum are retained. These are exploratory follow-up evaluations on the existing task set, not independent evidence for a newly calibrated operating point.

| Validation | Checkpoint | Planner | Execution | Threshold interval | Reference |
|---|---|---|---|---|---|
| A | augmented LeWM epoch 3, AdaJEPA online adaptation | GD 100 iterations | 30 plans x 5 controls | 0.10 / 0.30 | existing augmented AdaJEPA native and ECP at 0.15, capped at 30 plans |
| B | augmented DINO-WM epoch 3, frozen | CEM 200 x 10, BF16 candidates | 6 plans x 25 controls | 0.10 / 0.30 | existing v3 augmented native and ECP at 0.15 |
| C | augmented DINO-WM epoch 3, frozen | same CEM | 30 plans x 5 controls | 0.10 / 0.30 | fresh native control at the same execution frequency |

Every condition uses all 350 fixed EVAL50 tasks (seven shapes, fifty each), retaining chunk-local episode indices/seeds. No training is performed. Existing runs are preserved. The user confirmed augmented frozen DINO-WM for validation C. This changes execution frequency and therefore planning computation; the fresh same-frequency native control separates that effect from ECP.

## Implementation and verification plan

1. Add an isolated evaluator and GPU queue under reports/ecp_threshold_v1; reuse existing model, planner, and CEM implementations.
2. Validate resolved configs, checkpoint and target hashes, episode counts, maximum physical controls, and intervention parameters.
3. Execute the first full production chunk for A/B; require exact paired start/goal states and pre-intervention action prefixes against saved native runs before queuing the remaining chunks.
4. Once C is clarified, validate its configuration and run its native/ECP pair with the same targets, preserving 150 controls. Check pre-intervention equality between the new paired runs.
5. Record trigger counts, trigger times, success counts, paired rescues/regressions, intervention strength, runtime, and per-shape outcomes. Report incomplete runs as incomplete.
6. Launch only on GPUs 0,1,3,4,5,6,7 when sufficient memory is available; preserve unrelated processes.
