# PushT and six visual shifts: completed evaluation

Updated 10 November 2026. All 56 evaluations completed successfully (2,800 episode-runs). Re-reading the raw NPZ files confirms every success count; all 1,400 Native/ECP episode pairs pass pre-intervention action equivalence. The reviewed v13 manuscript now contains every completed result in the main table and full checkpoint/ECP matrix.

Current manuscript: `../../paper/cvpr2027/paper_v13.pdf`. Verified outcomes: `../../eval_outputs/pusht_visual_v8/verification.json` and `summary.md`. The older automatically populated v10 PDF is retained as an archive. No new experiment was started for this import.

## Frozen primary protocol

56 jobs = 2 existing weight families × 7 conditions × 2 model-update regimes × ECP off/on. Each job evaluates the same ordered 50 targets, giving 2,800 episode-runs. The primary target profile is `external50`; the bundled target file has SHA256 `1950b39e5d3ba13b30c14a435a400d84cd3fbe30138cdfed55e98b202d76277b`. These targets exactly match the archived DINO-WM and LeWM baseline tasks. Evaluation seed is 100; episode seeds are `100*i+1`.

| Item | Setting |
|---|---|
| Released weights | `checkpoints/pusht_visual_shift/checkpoints/model_latest.pth` |
| Data-augmented training recipe | `checkpoints/pushobj_T16k_bias_e3_seed0/checkpoints/model_3.pth` |
| Regimes | Frozen; within-episode AdaJEPA, reset before each episode |
| Conditions | Clean, red object, red goal, red agent, Gaussian blur sigma 2, salt/pepper 0.01 each, brightness ×0.9 |
| Physical-action budget | 30 replans ×5 executed actions =150 actions |
| Planning | Horizon25 physical actions (5 latent steps), GD100, LR0.1, unchanged staged objective |
| Success | Combined agent/object position error L2<20px and wrapped angle error<pi/9 |
| ECP | Joint latent memory; window3; trigger starts replan10; gamma0.05/0.15; beta0.25; kappa2; proprio boost×10 for trigger+2 replans |
| ECP-off control | Same planner/diagnostics/per-episode random stream; kappa0 and boost1 |

The two checkpoint recipes also differ in temporal-straightening coefficient (`cos1e-3` released versus `cos1e-2` augmented). Between-checkpoint differences therefore compare complete training recipes; they do not isolate data augmentation alone. No weights are selected or tuned using the new target results.

Fresh salt/pepper noise is keyed by episode identity and evaluator-call index, so ECP off/on see identical noise before their first divergent action. The noise distribution and corruption level are unchanged. The historical evaluator used system entropy for each call, so archived salt/pepper traces are not exact noise-path pairs. The CPU noise check and GPU smoke both passed. All16 smoke episode pairs have identical initial/goal states and identical pre-trigger actions.

## Run directly on the current GPU server

Use the existing `ts` Python environment directly on this host. Do not start a second launcher while the recorded PID is active. The commands below are retained for reproducibility or exact-protocol resume after the current scheduler exits.

```bash
TASK_REPO=/home/appuser/workspace/project/multimodal-test/JEPA-Stall
TASK_PYTHON=/home/appuser/anaconda3/envs/ts/bin/python
TASK_BUNDLE="$TASK_REPO/reports/pusht_visual_v8"

# Read-only assets/config checks; does not execute a model or simulation.
"$TASK_PYTHON" "$TASK_BUNDLE/run_pusht_visual.py" --repo "$TASK_REPO" --output "$TASK_REPO/eval_outputs/pusht_visual_v8" --validate-config

# Wiring/pairing smoke: clean and salt/pepper, 2 targets, separate output.
"$TASK_PYTHON" "$TASK_BUNDLE/run_pusht_visual.py" --repo "$TASK_REPO" --output "$TASK_REPO/eval_outputs/pusht_visual_v8_smoke" --n-evals 2 --conditions clean,snp --gpus auto --execute
"$TASK_PYTHON" "$TASK_BUNDLE/verify_smoke.py" --output "$TASK_REPO/eval_outputs/pusht_visual_v8_smoke"

# Start only after the smoke passes. Repeat the identical command to resume.
"$TASK_PYTHON" "$TASK_BUNDLE/run_pusht_visual.py" --repo "$TASK_REPO" --output "$TASK_REPO/eval_outputs/pusht_visual_v8" --gpus auto --execute
```

`auto` starts one worker per currently idle GPU (<1GiB used and <5% utilization). Explicit occupied GPUs are rejected. The launcher validates target/checkpoint hashes, records source hashes and exact arguments, preserves failed attempts, and skips only exact-protocol completions. SIGINT/SIGTERM terminates its own process groups while retaining completed outputs. Results are written incrementally to JSON/CSV; pending values are null rather than fabricated scores.

All 56 primary configurations and 16 smoke configurations passed Hydra composition. The noise validation passed without executing a world model or simulator. Smoke jobs took52–146seconds for two tasks, depending on early success and condition. Formal jobs use50 tasks, with difficult episodes running the full30-replan budget; completion time will be estimated from the actual queue rather than extrapolated from only easy smoke tasks.

## Archived results and optional legacy extension

`archived_native_results.json` contains 42 verified completed evaluations of the augmented checkpoint on the historical target pool, with raw NPZ hashes and per-condition counts. These are native Frozen/AdaJEPA scores, not ECP results, and use different targets from the primary50. Existing DINO-WM/LeWM scores are imported through the separate baseline audit and must retain their native MPC schedules; DINO-WM uses6×25actions while LeWM and the proposed matrix use30×5, all totaling150actions.

| Historical condition | Frozen (150 tasks) | AdaJEPA (150 tasks) |
|---|---:|---:|
| Clean |82.0%|90.7%|
| Red object |10.0%|24.7%|
| Red goal |18.0%|38.0%|
| Red agent |52.0%|67.3%|
| Blur |58.0%|89.3%|
| Salt/pepper |62.0%|90.0%|
| Dark |75.3%|92.0%|

The optional historical extension is **not scheduled**: `--target-profile legacy_pool --seeds 100,200,300` in a separate output directory yields168jobs. It is a separate task selection and must not be combined with the primary50 or described as additional independent primary tasks.

## Target provenance

The target file comes from the user's provided baseline repository, commit `d4f6cb491ac19d0224485b056fe5b24aa8e50f94`, path `research_snapshots/2026-11-08-pusht-gd/references/lewm/targets_T.pkl`. It is an ordered `random.Random(0).sample(...,50)` subset of the local1000-target pool, whose SHA256 is `f08486da48f7a6961e3d0035b48d9cd13515c0322bbacaf9dd9f973bf7bad624`. Goals are generated by replaying each segment's actions in the simulator, rather than reading its stored terminal state. Internal source links and licensing information are retained in `target_provenance.json`; repository/project names are not manuscript method labels.

The v10 finalizer also imports the fully verified Frozen Maze comparison. It retains all revised method/gallery figures and does not overwrite the reviewed manuscript.
