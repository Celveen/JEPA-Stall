# Frozen ECP: paired computational cost

On 350 matched PushObj tasks, ECP raises success from **61.14% to 72.29%** while reducing mean task-loop wall time by **12.06%**. The **2.81%** increase in per-replan latency is offset by **12.42% fewer replans**. There are 39 rescued tasks and no success-to-failure regressions.

| Metric | Without ECP | With ECP | Change |
|---|---:|---:|---:|
| Success | 214/350 (61.14%) | 253/350 (72.29%) | +11.14 percentage points |
| Mean planning time / replan | 747.24 ms | 768.27 ms | +2.81% |
| Mean replans / task | 15.59 | 13.66 | -12.42% |
| Mean task-loop wall time | 15.25 s | 13.41 s | -12.06% |
| Total planning time | 4078.45 s | 3672.32 s | -9.96% |
| P95 task-loop wall time | 31.60 s | 32.49 s | +0.89 s |
| Peak allocated GPU memory | 0.823 GiB | 0.967 GiB | +17.57% |
| Peak reserved GPU memory | 1.342 GiB | 1.469 GiB | +0.127 GiB |
| Model parameters | 34,295,767 | 34,295,767 | 0 |

## Matched protocol

Both arms use the same augmented epoch-3 Frozen checkpoint, 350 fixed tasks, evaluation seed 200, GD100 (learning rate 0.1, horizon 25), and CUDA Graph implementation. The shared cap is 30 replans of five controls. ECP uses the existing 0.15/0.45 thresholds, kappa 2 and proprio boost 10. The no-ECP arm disables escape and omits ECP detector/memory bookkeeping. Each paired ten-task chunk runs sequentially on the same physical RTX 3090; arm order alternates by chunk. Five GPUs run concurrently with shared host CPU. PyTorch is 2.3.0+cu121; per-job receipts record hardware and software details.

Task-loop wall time includes simulator prefix replay and common diagnostics, excluding model loading and the separate physics audit. Per-replan latency includes CUDA Graph capture. No per-episode warmup is removed. Peak memory is PyTorch allocated/reserved memory, not total device memory. This is one timing pass over the task set; the measurements do not provide independent timing-repeat confidence intervals or energy consumption. The improved mean does not extend to P95 task time.

Checkpoint SHA-256: `e80bc83ebafae128834d43256bfbd95c2fb66e4874652679662a131664975fdd`.

## Results and verification

- [summary.json](summary.json): all aggregate measurements and paired changes.
- [per_task_cost.csv](per_task_cost.csv): all 700 episode measurements.
- [PROTOCOL.md](PROTOCOL.md), [manifest.json](manifest.json), [status.json](status.json): measurement definitions, exact job order/configuration paths and completion status.
- [raw_results.tar.gz](raw_results.tar.gz): per-call/episode timers, configs, completion receipts, input targets, actions, replayed states, diagnostic logs and two excluded smoke checks.
- [runtime_sources.tar.gz](runtime_sources.tar.gz): runtime source snapshot, instrumentation, launch scripts, configuration files and fixed target inputs.
- [audit.json](audit.json), [checksums.json](checksums.json), [verify_results.py](verify_results.py): package integrity and reproducible aggregation.

From the repository root, with Python and NumPy installed:

```bash
python reports/frozen_ecp_cost_20261117/verify_results.py
```

The verifier checks all 70 jobs / 700 task outcomes, paired initial/goal states and GPU assignments, unchanged-model/checkpoint receipts, recorded physical-replay success labels, runtime source hashes, and reconstructs the complete summary and per-task table from archived timers. It requires no GPU or checkpoint. Exact historical action matching was verified during execution for all 35 ECP jobs and the four available no-ECP reference jobs; the verifier checks these receipts and does not rerun the simulator or historical comparisons.

To regenerate the tables manually, extract `raw_results.tar.gz` into a scratch directory, copy `analyze.py` and `manifest.json` there, then run `python analyze.py`. For a full GPU rerun, unpack `runtime_sources.tar.gz`, provide the checkpoint and dataset dependencies, and remap recorded absolute paths in the manifest and configs to the local installation before launching. The runtime snapshot preserves the actual experiment code rather than replacing the repository's current planner implementation.
