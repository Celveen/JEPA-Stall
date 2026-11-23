# Completed PushObj ECP threshold and cadence follow-up

Completed 11 November 2026, 16:00 Beijing time. All 140 chunks / 1,400 task runs completed without failed jobs. The raw-file audit verified all 1,050 native/intervention target pairs and action prefixes before intervention. Each condition uses 350 fixed tasks; all checkpoints were trained on augmented data for three epochs.

| Validation | Planner cadence | Native success | ECP at 0.15 | ECP at 0.30 | Triggered at 0.30 | Rescues / regressions vs native |
|---|---|---:|---:|---:|---:|---:|
| A: LeWM + AdaJEPA | 30 plans x 5 controls | 262/350 (74.9%) | 269/350 (76.9%) | 271/350 (77.4%) | 101/350 (28.9%) | 11 / 2 |
| B: frozen DINO-WM | 6 plans x 25 controls | 171/350 (48.9%) | 171/350 (48.9%) | 171/350 (48.9%) | 28/350 (8.0%) | 0 / 0 |
| C: frozen DINO-WM | 30 plans x 5 controls | 72/350 (20.6%) | not evaluated | 72/350 (20.6%) | 181/350 (51.7%) | 1 / 1 |

Both trigger interval endpoints were doubled (0.05/0.15 to 0.10/0.30). Penalty coefficient 2, radius coefficient 0.25 and temporary proprioceptive multiplier 10 were retained. Every condition is capped at 150 environment controls. Increasing planning frequency also increases the search computation and changes the executed action prefix / warm start; the fresh same-frequency native control is therefore required.

## Interpretation

- **AdaJEPA:** relative to ECP at 0.15, the higher threshold rescues three tasks and loses one, adding two successes (+0.57 percentage points). Triggered tasks rise from 90 to 101. Median first trigger advances from replan 14 to 12; 65 of the 90 previously triggered tasks intervene earlier and 25 intervene at the same time. The gain is small and comes from one exploratory task set, rather than an independent validation of a new best operating point.
- **DINO-WM at 6 x 25:** triggered tasks rise from 1 to 28. All 28 have changed action sequences after intervention, but no final success labels change. The intervention is active; additional triggers do not produce additional successful tasks in this run.
- **DINO-WM at 30 x 5:** ECP changes the actions of all 181 triggered tasks, rescues one and loses one. The native control also scores 20.6%, compared with 48.9% under 6 x 25. Thus the large drop is already present in the changed deployment cadence and cannot be assigned to ECP from these results. The cause within the changed planning/execution behavior has not yet been isolated.

The existing 6 x 25 DINO-WM protocol remains the stronger observed deployment configuration. Current evidence supports the action-biased data augmentation benefit on DINO-WM; it does not establish a DINO-WM success-rate gain from ECP.

## Verification and artifacts

- [completed_audit.json](completed_audit.json): outcome counts, actual action changes, first-trigger timing, comparisons with the previous threshold, and source SHA-256 hashes.
- [paired_tasks.csv](paired_tasks.csv): all 1,050 native/intervention task pairs.
- [by_shape.csv](by_shape.csv): all 28 condition/shape cells.
- [analyze_completed.py](analyze_completed.py): CPU-only raw-file audit.
- [pipeline_status.json](pipeline_status.json) and [PROTOCOL.md](PROTOCOL.md): completion record and configuration.
- Original raw records remain under `eval_outputs/ecp_threshold_v1/`; invalid startup checks are separately archived and excluded.

```bash
python reports/ecp_threshold_v1/analyze_completed.py --repo . --output /tmp/ecp-threshold-audit
```

These follow-up runs reuse the fixed task set after inspecting the original outcomes. No new experiment was launched during this audit.
