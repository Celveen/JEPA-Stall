# Completed ECP threshold and cadence validation

All 1,400 task runs are complete. See [RESULTS.md](RESULTS.md) for the full comparison, [completed_audit.json](completed_audit.json) for 1,050 independently checked native/intervention pairs, and [completed_raw.tar.gz](completed_raw.tar.gz) for complete raw results and logs.

| Augmented checkpoint / deployment | Native | ECP threshold 0.30 | Triggered |
|---|---:|---:|---:|
| LeWM + AdaJEPA, 30 x 5 | 74.9% | 77.4% | 101/350 |
| Frozen DINO-WM, 6 x 25 | 48.9% | 48.9% | 28/350 |
| Frozen DINO-WM, 30 x 5 | 20.6% | 20.6% | 181/350 |

The earlier AdaJEPA ECP result at threshold 0.15 is 76.9%; increasing the threshold adds two successes. DINO-WM's denser planning cadence lowers success even in the native control. These exploratory results do not establish a DINO-WM success-rate gain from ECP.

## Recover raw records and repeat the audit

Run from the repository root. The first two archives supply the earlier baselines; the third contains this experiment.

```bash
tar -xzf reports/publication_v14/archives/main_pushobj_adajepa.tar.gz
tar -xzf reports/dinowm_pushobj_v3/completed_results/complete_raw.tar.gz
tar -xzf reports/ecp_threshold_v1/completed_raw.tar.gz
python reports/ecp_threshold_v1/analyze_completed.py --repo . --output /tmp/ecp-threshold-audit
```

[archive_manifest.json](archive_manifest.json) records the archive hash. Every member is listed and hashed inside `ARCHIVE_MANIFEST.json`. Checkpoint binaries and redundant observation caches are excluded. Initial startup checks that loaded the wrong CEM version were stopped, excluded, and replaced; [invalid_preflight_record.json](invalid_preflight_record.json) preserves the reason. The successful benchmark receipts explicitly verify the actual CEM module and BF16 timings.

A subsequent experiment with within-interval detection and an AdaJEPA threshold of 0.45 is separate from this completed package.
