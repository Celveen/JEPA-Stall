# Evidence audit for paper v4

Read-only audit of existing repository evidence on 2026-11-06. No training, planning, or new simulator rollout was run. Raw JSONL records were aggregated by task identity and success by replan 30. Script: `audit_counts.py`; exact machine-readable results: `counts.json`.

## Recommended main comparison

Use a factorial/staged table with columns Frozen and AdaJEPA and rows below. This directly answers the user's requested progression while retaining the planning-only control.

| Configuration | Frozen count/rate | AdaJEPA count/rate |
|---|---:|---:|
| Official released checkpoint, native planner | 153/350 = 43.7% | 243/350 = 69.4% |
| Official released checkpoint + our S+P planner | 209/350 = 59.7% | 256/350 = 73.1% |
| Our training recipe, native planner | 214/350 = 61.1% | 262/350 = 74.9% |
| Our training recipe + our S+P planner | 253/350 = 72.3% | 269/350 = 76.9% |

All four rows share 350 target tasks. Raw roots: `eval_outputs/stag_{frozen,adajepa}_{official,bias16k}_it40/{ref,T0SP}`; evaluations above are censored at 30. Official model is `checkpoints/pushobj_shape_shift`; our model is `checkpoints/pushobj_bias_16k_e3_seed0`, epoch 3. These are **local reevaluations of the official released checkpoint**, not success numbers quoted from the AdaJEPA publication. Caption should say "released checkpoint, reevaluated under our protocol" once; no need repeat caveats in every paragraph.

Exact count-derived gains (percentage points):

- Data recipe vs official: Frozen +61 tasks / +17.4 pp; AdaJEPA +19 / +5.4 pp.
- Planning on our recipe: Frozen +39 / +11.1 pp; AdaJEPA +7 / +2.0 pp.
- Entire pipeline vs official: Frozen +100 / +28.6 pp; AdaJEPA +26 / +7.4 pp.
- Planning alone on official: Frozen +56 / +16.0 pp; AdaJEPA +13 / +3.7 pp.

Round gains from counts, not subtraction of rounded rates: 61.1 to 72.3 is +11.1 pp from 39/350.

Suggested prose: "Our interaction-rich training recipe raises frozen planning success from 43.7% to 61.1% and AdaJEPA from 69.4% to 74.9%. Adding ECP yields 72.3% and 76.9%, respectively, improving the released-checkpoint baselines by 28.6 and 7.4 percentage points. The gains persist with online adaptation, showing that data collection and execution-conditioned planning address complementary parts of the control pipeline."

Training comparison evaluates the complete collection/training recipe: directional action bias, noise, placement, retries, contact selection and filtering. It cannot be presented as a controlled estimate of the directional bias alone. Put precise recipe/matching details in appendix and use "training recipe" or "interaction-rich collection" in main text.

## A dedicated Ablation Studies subsection is supportable now

No new experiments are needed for a substantive ablation section. Organize it by scientific questions, not run IDs or seeds:

1. **What does each planning component contribute?** Four paired checkpoint/adaptation conditions with Native, S, P, S+P. Existing main matrix has 16 cells.
2. **Why condition intervention on execution?** Permanent alpha=10 and initial lookahead versus triggered corrections; add triggered lookahead to show that increasing search alone does not explain recovery.
3. **Where and how strongly should memory act?** Existing 2x3 regularizer-space/strength grid, described below. This is newly surfaced useful evidence, not a new experiment.
4. **When should intervention complement adaptation?** Maze S+P first-stall/delay/gate plus gated S/P decomposition. Freezing controls may stay appendix or supporting analysis; they contain no S/P and must not be described as ECP with adaptation removed.
5. **Does an auxiliary training objective replace execution feedback?** Existing B0–B3 matched retraining screen, perhaps appendix or one short main paragraph, with MSE/selection diagnostic. Do not mix B0's fresh model with historical recipe planner gains.

## Additional valid PushObj ablations

All below use the same 350 tasks and 30-replan outcome. Count audit recomputed from raw logs.

| Our frozen model variant | Success count | Rate | Rescued / regressed vs native |
|---|---:|---:|---:|
| Native | 214 | 61.1 | — |
| S, early trigger window (earliest plan 6) | 219 | 62.6 | 5 / 0 |
| S, standard trigger window (earliest plan 11) | 219 | 62.6 | 5 / 0 |
| P only | 253 | 72.3 | 39 / 0 |
| S+P | 253 | 72.3 | 39 / 0 |
| Permanent alpha=10 | 232 | 66.3 | 50 / 32 |
| Initial lookahead | 221 | 63.1 | 34 / 27 |
| S+P + triggered lookahead | 243 | 69.4 | 29 / 0 |
| Arrived-trigger S | 215 | 61.4 | 2 / 1 |
| Arrived-trigger S+P | 223 | 63.7 | 10 / 1 |
| Arrived-trigger S+P+lookahead | 225 | 64.3 | 12 / 1 |

Source: `eval_outputs/stag_frozen_bias16k_it40/{SZ_early,SZ_mid,T0P,T0SP,ALPHA10,LA_start,T0SPL,D2S,D2SP,D2SPL}`. The arrived-trigger alternatives change the trigger definition and onset together; do not describe them as a clean threshold-only sensitivity sweep. They can remain appendix.

Initial lookahead is a search control, not a matched compute baseline: it uses extra candidate computations. Triggered lookahead likewise consumes additional compute while retaining 30 environment replans. This is useful because the simpler correction has higher task success, but do not claim equal planning FLOPs.

## Existing strength/space grid (newly surfaced for manuscript)

| Memory metric | kappa 0.5 | kappa 2 | kappa 8 |
|---|---:|---:|---:|
| Proprioceptive-only | 263/350 = 75.1 | 262/350 = 74.9 | 263/350 = 75.1 |
| Joint latent | 264/350 = 75.4 | 266/350 = 76.0 | 265/350 = 75.7 |

Native is 262/350 = 74.9. Settings: our biased checkpoint + AdaJEPA, no proprioceptive reweighting; start 10, window 3, gamma 0.05/0.15, beta 0.25, 30 replans. Source: `eval_outputs/stag_EVAL50/{SP_lo,SP_mid,SP_hi,SZ_lo,SZ_mid,SZ_hi}`; paired reference `eval_outputs/escape_EVAL50/none`; selected subsets 97 tasks each. Existing report says 0 missing and 0 pre-trigger action mismatches for all six arms. Independently confirmed reference outcomes AND every action through replan 30 exactly match `eval_outputs/stag_adajepa_bias16k_it40/ref` (0 mismatches). Thus this grid belongs to the main data-recipe AdaJEPA protocol; it is not a different benchmark.

The labels SP_lo/SP_mid/SP_hi in raw folders mean **proprioceptive distance for S**, not simultaneous S+P correction! Relabel these clearly as "proprioceptive memory distance" in manuscript. All six cells have 0 regressions versus native. Latent-memory gains are +2/+4/+3 tasks. Main prose can say latent memory supports a range of regularization strengths and improves more than proprioceptive-only memory, without overstating effect size.

## Maze timing/component evidence

Corrected time scale only: action_repeat=4. 100 medium tasks, five held-out layouts, 30 replans. Native AdaJEPA 63; S+P immediate 66; delayed 10 replans 61; loss gate S+P 65; loss gate S 66; loss gate P 64. Raw roots are `eval_outputs/e3/medium_s100/{R-trig,R-delay,R-switch,SZ-switch,P-switch}` and `eval_outputs/stag_maze100_ar4/medium/adajepa/ref`.

Immediate S+P rescues/regresses 7/4; gated S+P 2/0; gated S 3/0; gated P 1/0; fixed delay 2/4. Full 50-replan data belongs appendix because main budget is 30. Best memory-only 66 and immediate full 66 tie in rate but different paired behavior; do not say gating improves the 30-replan maximum rate. Early model freezing 55, gate freezing 63; these use no S/P.

No standalone SZ-trig/P-trig Maze runs were found (only runner definitions). Full gated S/P decomposition exists.

## Missing sensitivity experiments and practical next runs

Scanned 1,365 saved Hydra configurations with stagnation settings: every beta is 0.25 and every gamma pair is 0.05/0.15. Therefore no beta/radius sweep or trigger-threshold sweep can be claimed from current data. kappa values 0.5, 2, 8 exist in the grid above.

Recommended next requests, if the author wishes to strengthen the paper (do not run without confirmation):

1. **Highest scientific value: matched collection ablation.** Two new matched collection/training conditions with directional contact bias enabled vs disabled, keeping source episodes, collection noise, placements, retries/filtering, sample count, backbone initialization and training schedule fixed. Evaluate Frozen and AdaJEPA with 30 replans. Start with one matched training pair to test direction, then extend to three pairs; three pairs x two recipes x two adaptation modes x 350 tasks = 4,200 task evaluations. This isolates the specific training innovation rather than comparing two historical checkpoints.
2. **Small missing planning comparison:** Maze memory-only at first-stall activation (SZ-trig), matching existing gated memory-only on the same 100 tasks. This isolates gate timing for the component Maze actually uses. Existing runner supports it; preserve 30-replan main reporting. Held-out repeat should follow with hyperparameters fixed.
3. **Small robustness check:** memory radius beta in {0.125,0.25,0.5}, using S alone on Maze and biased PushObj AdaJEPA; reuse existing beta0.25 row, two new arms per environment. Keep kappa2 and all triggers fixed. A trigger sensitivity sweep can be second priority; avoid inflating exploratory grids before confirmatory evaluation.

All testing seed values, target filename identities, training seed bookkeeping, subset replay mechanics and exact McNemar statistics can move to appendix/provenance. Main text only needs task counts, domain shifts, common budget, adaptation setting, comparison definition, and measurement units.

## Configuration cross-check requested during v4 editing

Saved Hydra configurations were compared for every one of the 29 shared grid chunks: joint-latent kappa2 versus the main-table SZ_mid. The only differences are `planner.max_iter` (30 vs 40, both reported at 30), the relocated target filepath, and the `only_samples` lists in five chunks. All model, objective, optimizer, adaptation, action and stagnation fields match. Grid selection used a conservative margin and reran 97 episodes; the later online trigger selection reran 92. The five extra reruns therefore concern selection, not a change of the method.

Both baseline runs are action-identical through replan30. The duplicate SZ_mid runs have identical success/failure outcomes on every task, but their action trajectories are **not** numerically identical: 446 compared action records differ. This is not explained by saved configuration differences, and no causal diagnosis was made. The grid is internally paired to its own baseline and has verified zero pre-trigger discrepancies according to its existing report. It remains valid as a sensitivity study; do not assert duplicate experiment trajectories are bitwise equal.

Suggested concise appendix implementation paragraph:

> We evaluate memory strength at kappa in {0.5, 2, 8} and compute the repulsion distance either in the proprioceptive branch or jointly in visual and proprioceptive latents. All six settings use the same interaction-rich checkpoint, AdaJEPA updates, 350 task instances and 30-replan budget. We fix the history window to 3, normalized-displacement thresholds to (0.05,0.15), and relative squared radius to beta=0.25. Memory activation is allowed after ten completed replans. Proprioceptive reweighting is disabled (boost=1), so the comparison isolates the memory term. The native success rate is 262/350; each candidate replays the potentially affected tasks and retains the identical reference outcome otherwise.

Code anchors: `scripts/stag_run.py:24` strength/space grid; `scripts/stag_run.py:35` fixed trigger/radius settings; `planning/stagnation.py:82` default boost=1; `conf/escape_plan_gd_pushobj.yaml:59` AdaJEPA adaptation configuration; saved configs under `eval_outputs/stag_EVAL50/<arm>/<chunk>/.hydra/config.yaml`.

Highest-priority training follow-up can be a **2x2 bias/noise factorial**, instead of repeating any existing planning grid: bias amplitude {0,9px} x additive noise {0,10px}, with source trajectories, collection start states, generation policy, selection/filtering, sample count, training schedule and initialization paired. Evaluate all four checkpoints with Frozen and AdaJEPA on a fixed 30-replan task set. One training replicate gives 4x2x350=2,800 task evaluations; three replicates gives 8,400. If compute is limited, first authorize the paired bias-off/bias-on comparison at fixed noise10, then expand the factorial. Use a fresh target stream for confirmation, keeping identifiers in appendix. The existing original-checkpoint comparison remains the deployment baseline table, while this factorial supplies the causal evidence for the collection mechanism.
