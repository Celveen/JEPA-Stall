# Frozen ECP cost comparison

Status: complete; 350 paired tasks per arm.

| Metric | Without ECP | With ECP |
|---|---:|---:|
| Success (%) | 61.143 | 72.286 |
| Replans / task | 15.594 | 13.657 |
| Controls / task | 77.971 | 68.286 |
| Episode wall (s / task) | 15.246 | 13.407 |
| Planning (ms / replan) | 747.243 | 768.267 |
| GD (ms / replan) | 747.145 | 752.386 |
| Total planning (s) | 4078.455 | 3672.318 |
| Peak allocated GPU memory (GiB) | 0.823 | 0.967 |
| Peak reserved GPU memory (GiB) | 1.342 | 1.469 |

Rescues: 39; regressions: 0.

One timing pass over 350 fixed tasks; task variation is not timing repeat uncertainty.
GPU-synchronized wall timers; same physical GPU for each paired chunk; alternating order.
Both arms use the same GD100 CUDA Graph backend; per-replan costs include graph capture.
No-ECP uses EscapeMPCPlanner with escape disabled, avoiding inactive ECP detector bookkeeping.
Episode wall includes simulator prefix replay and shared diagnostics, excludes model loading and independent audit.
PyTorch memory peaks include model tensors and graph pools; reserved memory differs from allocated memory.
Five concurrent dedicated benchmark GPUs; CPU is shared. No single-GPU isolation or energy measurement is claimed.
