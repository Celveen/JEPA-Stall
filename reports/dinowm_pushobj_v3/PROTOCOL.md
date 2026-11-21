# DINO-WM PushObj: six plans, 25 controls per plan

The user requested official DINO-WM action execution cadence on 10 November 2026, while retaining the approved PushObj CEM search budget. This revision evaluates the completed ordinary-data and augmented-data epoch-3 checkpoints; it does not retrain them.

## Shared configuration

`conf/dinowm_pushobj_official_cadence.yaml` is merged by every v3 evaluator, including native, ECP and FP32 precision controls. The queue verifies its hash and evaluator source hashes against integration validation before launching.

| Setting | Superseded v2 | Active v3 |
|---|---:|---:|
| CEM candidates / iteration | 200 | 200 |
| CEM iterations / plan | 10 | 10 |
| Elites | 30 | 30 |
| Prediction horizon, environment controls | 25 | 25 |
| Executed controls / plan | 5 | 25 |
| Executed model steps / plan, frameskip 5 | 1 | 5 |
| Maximum MPC plans | 30 | 6 |
| Maximum environment controls / task | 150 | 150 |
| Candidate batch size | 64 | 64 |
| Candidate prediction precision | BF16 | BF16 |
| Weights, observation encoding, objective and CEM updates | FP32 | FP32 |

Successful tasks terminate early. With full-horizon execution there is no unexecuted action suffix to warm-start the next plan. Model and target checkpoints, target identities, normalization and terminal objective remain unchanged.

The execution cadence follows the official [DINO-WM PushT configuration](https://github.com/gaoyuezhou/dino_wm/blob/0a9492fa12044b852ae9e001cc74604b79c8bb0c/conf/plan_pusht.yaml) and its [planning entrypoint](https://github.com/gaoyuezhou/dino_wm/blob/0a9492fa12044b852ae9e001cc74604b79c8bb0c/plan.py). That code interprets its five executed actions as model steps, each containing five environment controls. This repository's entrypoint divides configured physical time values by frameskip, so the equivalent local `n_taken_actions` value is 25. The six-plan cap preserves the requested 150-control budget. Official CEM defaults are 300 candidates and 30 iterations; this study retains the separately approved 200 x 10 search budget. It is not a complete reproduction of all official evaluation settings.

## ECP time interpretation

The controller checks the existing simulated observation history only at planning boundaries. It does not insert extra plans or environment actions.

- Warmup: 50 environment controls (formerly 10 x 5); the first possible intervention is plan index 2, i.e. the third plan.
- Movement window: 15 environment controls. For example, after executing control 50, the detector compares observations at controls 50 and 35. The evaluator already returns the intervening control frames; `physical_stagnation.py` samples them at a five-control stride.
- Temporary proprioceptive boost: the former trigger-plan-plus-two-following-plans duration was 15 controls. Because each new plan executes 25 controls, the duration is rounded up to one complete plan (`boost_replans=0`). A later detection can renew it. Stagnation memories retain their existing persistence.
- Movement thresholds 0.05/0.15, penalty coefficient 2, radius coefficient 0.25 and proprioceptive boost factor 10 are unchanged. Native runs use penalty coefficient 0 and boost factor 1.

These changes express ECP in physical time and accommodate the requested cadence; no test outcome was used to tune thresholds. The new and superseded ECP protocols are recorded separately.

## Conditions and validation

Four conditions use the same 350 fixed PushObj tasks, 50 for each of seven shapes: ordinary/native, ordinary/ECP, augmented/native and augmented/ECP. All parameters remain frozen. Five main workers use GPUs 0, 1, 3, 5 and 6, with an 8 GiB PyTorch memory cap and a 16,000 MiB free-memory start requirement.

`run_integration.py` executes real I/T starts under both checkpoints, with native and forced ECP, full 200 x 10 CEM and the six-plan cap. It verifies paired targets and pre-intervention actions, physical history indices, maximum executed controls, finite penalty and boost execution, and unchanged model hashes. Episodes that succeed before the first intervention exercise the early-termination branch. These integration outcomes are excluded from benchmark totals. The original overly strict assertion that every forced-trigger episode must activate was corrected to allow successful early termination; no model, target or evaluator was changed for that correction.

The unchanged candidate prediction and penalty kernels reuse the hash-linked numerical and memory validation from v2, explicitly labeled as prior evidence in `validation_summary.json`. BF16 changes action sequences and is not claimed to be FP32-equivalent. All four main conditions use the same precision. On GPU 4, `confirm_precision_tasks.py` performs 14 complete native FP32 checks (first task of each shape, both checkpoints), all with the new six-plan cadence, and pairs them with the corresponding BF16 results. They are diagnostic and excluded from main totals.

## Outputs and monitoring

- Main output: `eval_outputs/dinowm_pushobj_3epoch_v3_cem200x10_mpc6x25_bf16/`.
- Status and worker PIDs: `reports/dinowm_pushobj_v3/pipeline_status.json`, refreshed every 30 seconds.
- Precision controls: `reports/dinowm_pushobj_v3/precision_confirmation/`.
- Every task chunk records configuration, source/checkpoint/target hashes, CEM timings, trajectories, ECP episodes and a completion marker. The queue rejects mismatched completions and unexplained partial outputs.
- Final aggregation requires all 140 ten-task chunks and paired pre-intervention checks. It reports seen/unseen/all-shape success and paired rescues/regressions.
- Superseded v2 outputs are retained under their original directory. The stop record and archived launchers are in `reports/dinowm_pushobj_v2/superseded_mpc30x5/20261110T024138Z/`. Old and new frequencies are not pooled.

No paper result table is updated using partial runs.
