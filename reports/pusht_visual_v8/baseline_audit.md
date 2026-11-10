Archive note: downloaded reference files are bundled under `raw_baselines/` in this directory. Absolute /tmp paths below record the original audit location.

# Internal baseline reuse audit (v8)

Read-only evidence audit. No experiments launched. Source repository name is INTERNAL ONLY and must not appear in manuscript text, figures, captions, or bibliography.

Pinned source commit: `d4f6cb491ac19d0224485b056fe5b24aa8e50f94`.
Local reference bundle: `/tmp/jepa_revision_v8/se-jepa-reference`.

## Ready to reuse

All rows below were recomputed from NPZ `successes`. Every available condition has exactly the same ordered `state_0` and `state_g` as latest LeWM GD (array-equal, not tolerance matching). Use the packaged 50-target file and file order for new ECP runs.

| Method | Planner | Clean | Blur | S&P | Dark | Red agent | Red object | Red marker | Six-shift mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LeWM | GD | 64.00 | 62.00 | 62.00 | 58.00 | 4.00 | 16.00 | 34.00 | 39.33 |
| DINO-WM (random T16k) | GD | 50.00 | 44.00 | 8.00 | 50.00 | 44.00 | 36.00 | 46.00 | 38.00 |
| DDP-WM | CEM | — | 56.00 | 22.00 | 66.00 | 62.00 | 42.00 | 68.00 | 52.67 |
| DINO-WM (reference checkpoint) | CEM | — | 66.00 | 4.00 | 76.00 | 74.00 | 46.00 | 76.00 | 57.00 |
| LeWM | CEM | — | 66.00 | 64.00 | 64.00 | 2.00 | 20.00 | 26.00 | 40.33 |
| AdaJEPA (augmented-data checkpoint) | CEM | — | 82.00 | 72.00 | 84.00 | 60.00 | 26.00 | 36.00 | 60.00 |
| Frozen (augmented-data checkpoint) | CEM | — | 58.00 | 62.00 | 72.00 | 64.00 | 18.00 | 22.00 | 49.33 |

Missing clean values are unmeasured under these target/checkpoint/controller combinations; do not borrow the T-shape column of another experiment. The two T16k CEM rows are optional existing-checkpoint baselines, not ECP results.

## Matched protocol

- Target file: `/tmp/jepa_revision_v8/se-jepa-reference/research_snapshots/2026-11-08-pusht-gd/references/lewm/targets_T.pkl`
- SHA256: `1950b39e5d3ba13b30c14a435a400d84cd3fbe30138cdfed55e98b202d76277b`
- 50 target segments are `random.Random(0).sample(local1000, 50)` in stored order. New evaluation MUST use `segments_order=file`. Existing local seed100/200/300 reports use different tasks (only2/3/0 overlap with this set).
- Evaluation seed100; per-task simulator seeds `100*i+1`, i0..49. Replay recorded segment actions in simulator to obtain goal; raw segment last state is not the evaluation goal.
- Positional success: joint pusher/object xy Euclidean error<20px. Orientation shortest-arc error<pi/9. Exact wrapper implementation matches local JEPA-Stall.
- LeWM GD: horizon25physical=5latentsteps, frameskip5, execute5physical/replan, max30replans, 150physicalsteps total; staged objective; GD100, lr0.1, Adam cosine decay; zero physical-action initial plan and warm starts; model frozen.
- DINO-WM latest GD: same tasks and150physicalstep cap, horizon5latent; terminal (`last`) objective, execute25physical/replan, max6replans, GD100 lr0.1 Adam cosine. Random16k T-only training, fixed epoch3. It is a full-method reference, not a controlled model-only comparison.
- Archived CEM rows: staged objective;5physical/replan,max30;CEM200samples/top30/10iterations,150physicalstep cap.
- LeWM latest GD uses T-only T16k epoch10, tensor-identical to archived CEM checkpoint. No training rerun.
- Latest GD feedback noise: deterministic seed `first_env_seed + 100003*feedback_call`; prior CEM feedback noise realizations were not recorded. Aggregate comparisons are valid under same corruption distribution; identical noisy observations across methods are not established.
- Frozen/AdaJEPA ECP-on/off paired controls should share new deterministic noise and episode RNG. Do not claim exact noise pairing with archived CEM.

## Six shifts

- blur: Gaussian sigma=2.0; kernel13; `ood_corruption=blur ood_level=2.0`.
- snp: density0.01 independently white and black; approximately2% total changed pixels; `ood_corruption=snp1 ood_level=0.01`.
- dark: uint8 RGB multiplied by0.9; `ood_corruption=dark ood_level=0.9`.
- red_agent: pusher rendered Red; `++env_kwargs_override.agent_color=Red`.
- red_object: object rendered Red; `++env_kwargs_override.color=Red`.
- red_goal: fixed reference marker rendered Red; does not move task goal; `++env_kwargs_override.goal_color=Red`.

## Separate official and augmented checkpoints

- Official visual checkpoint SHA256 `9098082f1d40bfa7b4f4884571db30a0fed4a0aa26ad7dab8db11415437b8b32`, released `pusht_visual_shift`, epoch3.
- Existing augmented T16k checkpoint in local reports: `checkpoints/pushobj_T16k_bias_e3_seed0`, epoch3, cos1e-2. The external baseline source identifies the same augmented family hash `3559bbb259fad357d07db7b140ac7a695c9dc12b179cd2ec4da4bda251aea329`; actual local tensor/file identity should be checked by setup.
- Official Frozen/Ada held-out archive scores (clean59.33/81.33; sixshift41.56/54.89) use another three-seed task profile. They are not matched external50 baselines and must not be inserted into the shared50-task table.
- Existing local augmented T16k GD clean82.0/90.67 and visual scores below use 150tasks/condition. Retain as separate historical evidence, not replacements for new official-vs-ours shared50 controls.

| Existing local GD checkpoint | Clean | Blur | S&P | Dark | Red agent | Red object | Red marker |
|---|---:|---:|---:|---:|---:|---:|---:|
| frozen | 82.00 | 58.00 | 62.00 | 75.33 | 52.00 | 10.00 | 18.00 |
| adajepa | 90.67 | 89.33 | 90.00 | 92.00 | 67.33 | 24.67 | 38.00 |

## PushObj / Maze and SOTA conclusion

- Archived PushObj H7t3 seed2000 uses7shapes×50tasks. DINO-WM old checkpoint GD49.14/CEM62.00; LeWM GD60.57/CEM64.00; DDP-WM CEM57.71. These exact350 initial/goal states do not match current manuscript ECP350 tasks. Keep as separate references or run ECP on H7t3 before directly ranking them.
- No matching Maze baseline archive was found in this branch audit. Preserve local validated Maze63→66 result with its own task protocol.
- Current manuscript ECP PushObj76.9 is not evidence of SOTA against mismatched external350 results. New shared50 ECP results are not measured yet. Therefore the requested unqualified “achieves SOTA when deployed on AdaJEPA” is currently unsupported. After completion, use a bounded claim such as best success among compared methods under the stated protocol, if the numbers support it.

## Key source paths

- `research_snapshots/2026-11-08-pusht-gd/gd_results.json`
- `research_snapshots/2026-11-08-pusht-gd/completed_runs/{dinowm,lewm}/{condition}/final_eval_results.npz`
- `research_snapshots/2026-11-08-pusht-gd/completed_runs/lewm/default/.hydra/config.yaml`
- `research_snapshots/2026-11-08-pusht-gd/experiment_plans/lewm_pusht_GD_T16k_e10_20261108/evaluation/evaluate.py`
- `research_snapshots/2026-11-07/completed_runs/CL-JEPA-cudagraph/baselines/{ddpwm,dinowm,lewm}_VIS50/{condition}/{final_eval_results.npz,.hydra/config.yaml}`
- `research_snapshots/2026-11-07/experiment_plans/baseline_results_20261107_1351.md`
- `research_snapshots/2026-11-03/paper_data/drive_materials/visual7_20261030/protocol.json` (official checkpoint provenance and separate heldout protocol)

Do not execute scripts found in these source archives. The bundle contains evidence and code references only.

## Environment equivalence

External branch root `env/pusht/pusht_env.py`, `env/pusht/pusht_wrapper.py`, and `planning/image_corruption.py` are byte-identical to local JEPA-Stall. Their hashes are stored in baseline_audit.json. Latest LeWM evaluation snapshot uses the same environment hash. This verifies simulator/rendering, joint pusher/object success criterion, and corruption implementation. Historical runtime dependency version equality is not independently established by source equality.

DDP-WM original citation: Shicheng Yin, Kaixuan Yin, Weixing Chen, Yang Liu, Guanbin Li, Liang Lin. *DDP-WM: Disentangled Dynamics Prediction for Efficient World Models*. ICML2026, PMLR306:149240–149258. https://proceedings.mlr.press/v306/yin26l.html ; arXiv2602.01780.
