# DINO-WM PushObj: approved search budget and accelerated evaluation

The user approved this revision on 10 November 2026: align the new DINO-WM experiment with the existing PushObj **search budget**, then verify inference acceleration. The epoch-3 ordinary-data and augmented-data checkpoints are unchanged. No training is restarted.

## Evaluation settings

| Setting | Previous run | This run |
|---|---:|---:|
| CEM candidates / iteration | 300 | 200 |
| CEM iterations / replan | 30 | 10 |
| Elites | 30 | 30 |
| MPC replan cap | 30 | 30 |
| Prediction horizon (environment controls) | 25 | 25 |
| Executed controls per replan | 5 | 5 |
| Candidate batch size | 16 | 64 |
| Predictor candidate-rollout precision | FP32 | BF16 after implementation validation |
| Model weights, observation encoding, objective and CEM updates | FP32 | FP32 |
| PyTorch memory limit per worker | 4 GiB | 8 GiB |

The budget is 150 environment controls per task; frameskip remains five. The terminal objective (`mode=last`) and existing ECP thresholds, regularization and proprioceptive weighting are retained from the first controlled DINO-WM run. This change aligns candidate count and CEM iterations with the archived PushObj comparison; it does not claim that every other archived setting, checkpoint or target list is identical. TF32 is explicitly disabled. Candidate random numbers are drawn together on CPU, and elites are selected across the entire population, as before.

Four conditions use the same fixed 350 tasks (50 per shape): ordinary/native, ordinary/ECP, augmented/native, augmented/ECP. Parameters remain frozen in every condition. A worker uses one GPU; the GPU pool is 0, 1, 3, 5 and 6. Chunks are interleaved across shapes so early progress includes both seen and unseen objects. This changes scheduling only; sample identities and within-chunk order are unchanged.

## Validation and numerical interpretation

`validate_precision.py` compares complete 200-candidate, 10-iteration CEM searches at identical saved observations and random seeds, on two states per checkpoint. It records full action changes, timing, and both plans rescored by the FP32 model. FP32 took approximately 21.3 seconds per replan; BF16 took approximately 6.3 seconds. The measured FP32-rescored relative-cost changes across these four states ranged from -2.76% to +0.55%. The action sequences are **not identical**.

`run_integration.py` then checks two real simulator starts (I and T) for both checkpoints, both precisions and both ECP settings. Each run uses the full new CEM budget and four MPC replans. ECP activation is forced at replan index two exclusively for these integration checks. Validation requires equal paired start/goal states, bitwise equal native/ECP actions before activation, finite evaluated stagnation penalty, exercised proprioceptive boost, finite actions, and unchanged model hashes. These short runs are not benchmark success results and do not establish FP32/BF16 closed-loop accuracy equivalence.

`check_memory.py` checks the actual ECP penalty with 64 candidates, five predicted transitions and 20 memory entries, plus a 1 GiB model allowance. It also verifies positive penalty at a memory point and zero penalty at a goal outside the repulsion region; an optimized plan may legitimately have zero penalty. The worker memory limit is raised to 8 GiB because the visual-memory distance temporarily expands along candidate, horizon, memory and patch-feature dimensions. Each worker still waits for at least 16,000 MiB free device memory before starting.

`confirm_precision_tasks.py` additionally runs 14 complete FP32 tasks on GPU 4: the first task of each of seven shapes for both checkpoints, without ECP, using the full 30-replan cap. They are matched against the corresponding BF16 main-run outcomes. These checks report rescues/regressions when the pairs finish; they are excluded from benchmark totals and do not by themselves establish population-level accuracy equivalence.

All four final conditions use BF16 candidate rollouts consistently after implementation checks pass; observation encodings, objective reductions and CEM distribution updates remain FP32. The numerical protocol is part of the reported method configuration. It must not be described as bitwise equivalent to the former FP32 run, and outcomes from the two precisions must not be pooled.

## Provenance and progress

- Old outputs remain in `eval_outputs/dinowm_pushobj_3epoch_v1/`; completed and interrupted chunks are preserved.
- Old launchers, source files, logs and stop record are archived under `reports/dinowm_pushobj_v1/superseded_cem300x30/20261110T021312Z/`.
- New outputs use `eval_outputs/dinowm_pushobj_3epoch_v2_cem200x10_bf16/`.
- `pipeline_status.json` is refreshed at least every 30 seconds, including worker PIDs and active output directories.
- Each output includes `cem_timings.jsonl`, configuration, source/checkpoint/target hashes, trajectories, results and a completion marker. Existing completion markers are checked against the full protocol before reuse; partial runs require inspection.
- `summary.json` is generated only after all 140 chunks complete. It reports seen/unseen/all-shape success, paired rescues/regressions and pre-trigger identity checks.

Current paper result tables are unchanged until this evaluation completes.
