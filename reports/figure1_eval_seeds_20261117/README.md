# Figure 1: additional evaluation seeds

Completed 17 November 2026. Five Figure 1 conditions now have three evaluation-seed results, with **350 fixed tasks per seed**. This package contains all **3,500 additional-seed task evaluations**, including reused complete batches from the earlier eager-GD queue. It does not contain three independently trained models.

## Results

| Model | Configuration | Original seed successes / 350 | New seed 1 | New seed 2 | Mean ± sample SD (%) |
|---|---|---:|---:|---:|---:|
| Frozen | Augmented data + ECP | 253 | 253 | 253 | **72.29 ± 0.00** |
| LeWM | Augmented data + ECP | 231 | 231 | 231 | **66.00 ± 0.00** |
| AdaJEPA | Base data | 243 | 235 | 244 | **68.76 ± 1.41** |
| AdaJEPA | Augmented data | 262 | 254 | 260 | **73.90 ± 1.19** |
| AdaJEPA | Augmented data + ECP | 272 | 265 | 270 | **76.86 ± 1.03** |

Seeds are **100 / 200 / 300** for Frozen and AdaJEPA and **2000 / 2001 / 2002** for LeWM. Original-seed values come from the existing Figure 1 evidence recorded in [historical_figure1.json](historical_figure1.json). The two additional seeds were evaluated here. Standard deviations use `ddof=1` across the three per-seed success percentages.

For AdaJEPA, augmentation improves the mean by **5.14 percentage points**. Adding ECP improves it by another **2.95 points**, or **8.10 points** over base data. ECP adds **10, 11, and 10** successes over augmented native planning in seeds 100, 200, and 300. The direction of both improvements agrees across all three seeds; these statistics alone are not a significance test.

The other four Figure 1 conditions—Frozen and LeWM with base or augmented data without ECP—were not completed for the additional seeds. They retain their historical single-seed measurements; no three-seed mean or SD is asserted for them. Earlier partial outputs for those conditions are excluded from this package.

## What the seeds measure

The initial states, goal states, task order and checkpoints are fixed. GD uses zero initialization and zero action noise. Frozen and LeWM do not update their weights; their two additional ECP seeds produced identical action arrays and success labels across all chunks. Zero SD here reflects this deterministic evaluation protocol, not robustness across independently trained models or newly sampled tasks.

AdaJEPA performs online adaptation with its predictor in training mode and dropout probability **0.1**. Its episode-seeded dropout changes the updates and subsequent actions. The episode seed is `evaluation_seed * within_chunk_index + 1`; the first task of each chunk therefore has episode seed 1 in every repetition. The task lists are repeated, not independent task draws.

## Protocol and acceleration

- GD100, learning rate 0.1, horizon 25, maximum **30 replans × 5 controls = 150 controls**, staged objective, original checkpoints and task lists.
- ECP uses thresholds 0.15/0.45. LeWM retains contact-objective weight 10 and its geometry-assisted contact target using observed simulator object pose.
- Frozen/AdaJEPA use their original EVAL50 task chunks; LeWM uses its original H7t3 task list. Within-family comparisons preserve the original task identities.
- The CUDA Graph implementation derives from [`Celveen/SE-JEPA`, commit `aa397f3`](https://github.com/Celveen/SE-JEPA/commit/aa397f3f551a8fe5b8015e18514f78c2a2956a98). Each planning call recaptures its current objective, avoiding stale ECP memory/contact tensors. The captured LeWM path includes its action-dependent proprioceptive and contact losses. Adam, learning-rate scheduling and noise draws remain outside the graph.
- **125 real planning calls** across seven validation episodes matched eager GD bit-for-bit in actions and final RNG state, including active ECP memory, LeWM contact interventions and both AdaJEPA checkpoints. These validation episodes are excluded from result counts.
- The previously completed ECP validation measured approximately **3.9–5.4×** planning speedup, including graph capture. This is not an end-to-end benchmark for all conditions.
- Every published job verified unchanged checkpoint files and replayed saved controls through the physical simulator. Frozen/LeWM also verified unchanged model weights.

There are **294 completed batches**. Of these, **40 batches / 720 task evaluations** reuse completed eager-GD results; the remaining **254 batches / 2,780 task evaluations** use CUDA Graph. Reused batches are marked individually in `per_job.csv`. Incomplete batches were not reused. All statistical denominators count each task once.

## Files and verification

| File | Contents |
|---|---|
| [summary.json](summary.json) | Five three-seed summaries and explicit exclusions |
| [per_seed.csv](per_seed.csv) | All 15 condition/seed rows, marking historical versus evaluated results |
| [per_shape_new_seeds.csv](per_shape_new_seeds.csv) | Seven shapes per condition/seed for the additional seeds |
| [per_task_new_seeds.csv](per_task_new_seeds.csv) | All 3,500 additional-seed outcomes, chunk IDs and episode seeds |
| [per_job.csv](per_job.csv) | Batch counts, reuse status, checkpoint hashes and replay verification |
| [audit.json](audit.json) | Identity checks, task-list hashes and between-seed action comparisons |
| [raw_results.tar.gz](raw_results.tar.gz) | Original configs, completion receipts, actions, initial/goal states, replayed state trajectories, and CUDA Graph validation logs |
| [runtime_sources.tar.gz](runtime_sources.tar.gz) | Eager and accelerated Python/config snapshots, original task lists and queue/runner snapshots |
| [checksums.json](checksums.json) | SHA-256 hashes of every published package file |

Recompute counts and sample SD and verify archive contents from a checkout with NumPy installed:

```bash
python reports/figure1_eval_seeds_20261117/verify_results.py
```

The verifier checks file hashes, raw success arrays against receipts and replay records, all per-task rows, three-seed statistics, and all 125 action/RNG parity comparisons. It reads archives in place without extracting them. It checks the existing physical-replay records; it does not rerun the simulator.

To rerun the experiments, use the recorded source snapshot and environment described in [REPRODUCE.md](../../REPRODUCE.md), supply checkpoints matching `audit.json`, and adapt the recorded absolute checkpoint/input/output paths to the new machine. Original paths are retained in raw configs, manifests and runners for provenance. Checkpoint binaries, training image corpora, videos and console progress logs are excluded. The source archive contains the original task-list inputs needed for this evaluation.

This report adds evaluation evidence to the current manuscript branch. Existing single-seed manuscript figures and compiled paper/delivery files are not regenerated by this package.
