# Local baseline and portability evidence audit

Date: 2026-11-06 (Asia/Shanghai). Read-only inspection of JEPA-Stall and adjacent adajepa. No experiments started.

## Finding

The six PushObj bars already reported in manuscript v4 are available under one local protocol. No matched PushObj results for DINO-WM, LeWM, or DDP-WM were found in either repository, and no evidence was found that ECP has been evaluated on those three model families.

| Proposed chart item | Local evidence | What can be plotted now |
|---|---|---|
| Official frozen / data / data + ECP | 153/350, 214/350, 253/350 | 43.7%, 61.1%, 72.3% |
| Official AdaJEPA / data / data + ECP | 243/350, 262/350, 269/350 | 69.4%, 74.9%, 76.9% |
| DINO-WM PushObj | No matching evaluated run found | No numeric bar |
| LeWM PushObj | No matching evaluated run found | No numeric bar |
| DDP-WM PushObj | No model-specific configuration/checkpoint/result found | No numeric bar |
| Ours final under current local protocol | 269/350 with interaction-biased data, AdaJEPA, and S+P | 76.9%, compared only with genuinely matched runs |

The six-bar chart is a comparison of training recipes and update regimes. Frozen and AdaJEPA are two deployment regimes for a common backbone family. They are not two independent world-model architectures.

## Exact local sources

Paths below are relative to `/home/appuser/workspace/project/multimodal-test/JEPA-Stall` unless noted.

- `paper/cvpr2027/sections/experiments.tex`: official checkpoint is reevaluated on 350 PushObj tasks with 30 MPC replans; main table and definitions of the interaction-biased recipe and ECP.
- `paper/cvpr2027/sections/analysis.tex`: component matrix evaluates ECP on both official and interaction-biased checkpoints in Frozen/AdaJEPA deployment modes.
- `checkpoints/pushobj_shape_shift/hydra.yaml`: official PushObj model uses `models.encoder.resnet.SmallResNetGeM`, dim 384; `models.vit.ViTPredictor`, depth 6; TS `cos1e-2`.
- `checkpoints/pushobj_bias_16k_e3_seed0/hydra.yaml`: same SmallResNetGeM encoder and ViTPredictor family, same TS coefficient, trained on interaction-biased data.
- `checkpoints/diversemaze_layout_shift/hydra.yaml`: `SmallResNet`, dim 384, same world-model/predictor family; a task-specific encoder variant does not establish transfer to DINO-WM/LeWM/DDP-WM.
- `reports/wall_dinowm_20261103/report.md` and `summary.json`: the only DINO-WM benchmark results found. Official Wall checkpoint epoch 65; 26/50 (52%) open-loop and 46/50 (92%) MPC. Maximum 10 replans × 25 environment steps, unlike current PushObj 30 × 5. The report expressly states that adaptation, the discriminator, and stall regularization were not run in these experiments.
- `reports/paper2_positioning/wall_stall_proposal.md`: future DINO-WM + regularization/gating study marked pending confirmation and not executed.
- `conf/plan_wall_dinowm.yaml` and `scripts/wall_eval.py`: available DINO-WM evaluation entrypoint is Wall, not PushObj.
- `reports/paper2_positioning/report.md`: LeWM/DINO-WM names and published scores appear in literature notes, including TEMPO's LeWM-based PushObj result. These are literature summaries, not local evaluations.
- `scripts/gen_pushobj_dataset.py`: uses DINO-WM's PushT source trajectories. Data provenance does not make the resulting trained model a DINO-WM checkpoint or a DINO-WM benchmark.

## Search scope and limitations

Inspected filename listings and relevant text content in both repositories' `reports`, `scripts`, `conf`, `checkpoints`, and `eval_outputs`, together with the manuscript provenance. Searched case-insensitive variants of DINO-WM/dinowm, LeWM/LeWorldModel, and DDP-WM. Inspected top-level checkpoint and evaluation directory names. Broad DDP hits refer to distributed data parallel training, not a DDP-WM method. Logs were not exhaustively printed; no checkpoint binaries were deserialized. This finding describes the local repositories checked, not unknown results on other servers.

## Supported wording

- “The gains accumulate in both frozen and adaptive deployment.”
- “The planner improves both the released and interaction-biased checkpoints.”
- “ECP modifies the planning objective without changing the predictive architecture.” (architectural property, not a claim that multiple families have been experimentally validated.)

An empirical claim of applicability to DINO-WM, LeWM, and DDP-WM needs each model to be tested with and without the method. A highest-score comparison needs comparable task definitions, success thresholds, task lists and rollout/planning budgets. A paper-reported score from PushT or a different PushObj protocol cannot support a common-axis SOTA ranking against the current 76.9%.

## Minimal next evidence needed

1. Identify the precise DDP-WM paper/release; the local repo provides no disambiguating citation.
2. Obtain or train valid baseline checkpoints and evaluate all three on the common PushObj task suite and budget. This adds 3 × 350 = 1,050 baseline task evaluations if compatible released checkpoints exist.
3. Add matched +ECP evaluations on each backbone to substantiate portability (another 1,050 tasks). Adaptation and training-data interventions should be explicitly controlled per model.
4. Keep the current six-bar chart now; add the four-method comparison only after real compatible scores are available. Never use zeros, invented heights, or unmatched Wall/PushT scores for missing methods.
