# Experiment structure for manuscript revision v4

This is an editorial recommendation based on the current manuscript and verified primary papers. No experiments were launched, and all proposed additions below are unmeasured.

## What to borrow from the primary papers

- **Temporal Straightening (TS)** develops representation diagnostics in §5.2, then planning setup/results in §5.3. Its Table 1 organizes comparisons by model configuration and presence/absence of regularization; Appendix B holds detailed planner, dimensionality, and objective comparisons. The transferable principle is to connect each mechanism claim to a controlled comparison and reserve implementation detail for the appendix. Do not copy TS's ordering mechanically: our strongest result is task completion, so lead with it. [TS, §§5.2–5.3 and Appendix B](https://arxiv.org/html/2603.12231v3)
- **LeWM** separates planning evaluation setup (§4.1), efficiency/performance (§4.2), and focused ablations/training behavior (§4.3), before studying representation properties (§5). Detailed ablations sit in Appendix G. The useful precedent is that headline performance and mechanism analysis answer different questions; the latter should substantiate the former without turning the main results into an experiment log. [LeWM, §§4.1–4.3, §5, Appendix G](https://arxiv.org/html/2603.19312v3)
- **AdaJEPA** uses an especially suitable spine: §4.1 Setup → §4.2 Results → §4.3 Ablations. Table 2 pairs Frozen and Adapt rows for each model, displaying gains compactly. Its main setup reports aggregate replication, not arbitrary seed identifiers; numerical implementation settings are in Appendix B. Its default evaluation budget is 20 MPC steps, while Figure 2 also shows 30. The released layout-shift evaluation uses shortest paths of 3–5 cells; ours uses 6–9. Therefore our baseline rows must be labeled matched reevaluations rather than original paper scores. [AdaJEPA, §§4.1–4.3, Table 2, Appendices A.4 and B](https://arxiv.org/html/2606.32026v1)

## Recommended main-text organization

Use the recognizable section title **Experiments**, followed by four subsections. Avoid dividing the story between a broad results section and a separate section that finally reveals the training-data improvement.

### 4.1 Experimental Setup

Keep this to three compact paragraphs: tasks and success criteria; baselines and components; shared comparison protocol. State 350 PushObj tasks, 100 held-out-layout Maze tasks, and a common maximum of 30 replans. Define Frozen as no online parameter updates and AdaJEPA as the official adaptation procedure. Identify the released model and interaction-rich model. Put the task IDs, seed integers, source-overlap counts, optimizer reset details, reuse counts, and scheduling particulars in the appendix.

A suitable final protocol sentence is: “All comparisons use matched tasks and planning budgets; the current evaluation suite was used for development, with complete sampling and replication details in Appendix X.” This retains the evaluation's material status without repeatedly interrupting the findings. Do not describe selected development outcomes as a held-out confirmation test.

### 4.2 Main Results

Lead with a **progressive comparison table** displaying official-model performance, changing training data, and adding our planner. Place Frozen and AdaJEPA side by side: the reader should immediately see the same improvement chain in both regimes.

Suggested columns: Training model/data | Planning correction | Frozen success (%) | AdaJEPA success (%).

Suggested rows, all from the current matched 350-task, 30-replan evaluation:

| Training model/data | Planning correction | Frozen | AdaJEPA |
|---|---|---:|---:|
| Official checkpoint | Native | 43.7 | 69.4 |
| Interaction-rich collection | Native | 61.1 | 74.9 |
| Interaction-rich collection | ECP, S+P | 72.3 | 76.9 |

The primary comparison is the full, consistently defined S+P variant. The component ablation must also show P alone (72.3 / 77.1); do not imply that adding S improves P on every task family. If using P as the main PushObj configuration instead, clearly define that benchmark configuration consistently throughout the abstract, figure, caption, and tables.

Use exact counts before rounding gains: data change gives +17.4 / +5.4 percentage points, followed by S+P's +11.1 / +2.0 points. The total improvements relative to official Frozen/AdaJEPA are +28.6 / +7.4 points. These are within the matched reevaluation, not a claim of beating incomparable published numbers.

Suggested caption:

“**Training and planning contributions compound.** PushObj success (%) on 350 matched tasks with at most 30 MPC replans. The official released checkpoint is evaluated with frozen parameters or the AdaJEPA update rule under the same protocol. Interaction-rich collection improves both regimes; execution-conditioned planning (ECP, S+P) provides a further gain. Full component comparisons appear in Table X.”

Follow with a short Maze paragraph/table establishing the complementary role of memory: adaptive baseline 63%, gated S 66%. Do not make Maze look like it has undergone the PushObj collection intervention; no new offline Maze model was trained.

### 4.3 Ablation Studies

Use an explicitly titled subsection and short claim-oriented paragraphs, rather than scattering the controls across the paper:

1. **Contribution of planning components.** Move the existing four-setting Native/S/P/S+P matrix here. It is already a full component ablation at two representation sources and two update settings. Use one consistent decimal place, bold only the best within each setting, and put exact counts in the appendix unless needed for a specific paired argument.
2. **Conditional versus persistent correction.** Retain the strong frozen PushObj comparison: persistent 10× weighting 66.3%, conditional P 72.3%, native 61.1%. Rescue/harm is useful here because it directly explains the advantage of activation. Use it only in this control table, rather than in every main-results table.
3. **Memory activation and continued adaptation.** Use the existing Maze trigger/gate and freeze controls. State that freeze controls have no S/P correction. Gated S preserves 63 successes and adds three, whereas freezing at the trigger yields 55. Aggregate ties need not imply equivalent behavior; paired counts belong in the table/appendix.
4. **Additional training objectives.** One brief paragraph may refer to B0–B3 in the appendix; these are alternative loss objectives, not ablations of the deployed method. Their conclusion is that directly adding auxiliary prediction/effect objectives did not deliver the execution benefits of the collection-and-planning recipe. Avoid presenting inverse-dynamics loss as an ECP component.

Suggested component caption:

“**Component ablation on PushObj.** Success (%) with the same tasks and 30-replan budget. S denotes memory-based stagnation regularization and P denotes temporary proprioceptive reweighting. Each row fixes the initial checkpoint and online update policy; only the planning correction changes.”

Suggested timing caption:

“**Conditional activation preserves successful executions.** Frozen PushObj planning with the interaction-rich checkpoint. Permanent and conditional weighting use the same 10× multiplier. Rescue/harm counts are paired changes relative to native MPC.”

### 4.4 Qualitative Analysis of Recovery

Explain the two logged simulator cases together: repeated local pusher motion in PushObj and repeated corridor traversal in Maze. Tie the observed behavior directly to S and P, and use the existing distance curve to show departure from a stalled neighborhood. Say that the panels are matched examples with identical prefixes; full-population component attribution comes from the ablation, not from a selected image.

The auxiliary action-space variant can be reduced to one paragraph linking to an appendix table. It uses a different physical trigger and does not contain S, so giving it a long main-text subsection distracts from the central mechanism. Preserve both task-stream results and their overlap accounting in the appendix.

## Seed information: main text versus appendix

**Main text:** number of tasks, pairing, planning budget, reported aggregation unit, number of trained models when displaying mean ± SD, and any material development/selection status. “Three independently trained models” is enough if such a summary appears; numeric initialization identifiers are not needed.

**Appendix:** training seeds 0/1/2; task-sampling seeds 100/200/300; all source/segment overlap counts; per-seed tables; exact counts; confidence or paired-test calculations; current trajectory reuse and prefix checks; hashes/run paths. Do not relabel partly overlapping streams as independent replications, and do not pool them as independent tasks.

The existing historical three-seed training table uses mixed 20/30-replan protocols and different task streams. It belongs in an appendix subsection on replication and should not serve as the headline progressive comparison.

## Highest-value new experiments (proposed only)

1. **Isolate the collection bias.** Train matched models with directional contact bias enabled/disabled while fixing source placement, action-noise scale, trajectory count, quality filter, training windows, and optimization budget. Evaluate native Frozen and AdaJEPA on one fixed confirmation suite. This establishes which part of the collection recipe causes the gain; the released-checkpoint comparison alone evaluates the full recipe. If zero bias severely reduces accepted-trajectory yield, report collection effort separately rather than quietly changing filtering. This requires new data/training and should be confirmed with the user first.
2. **Control for the intervention schedule.** Compare the observed-stagnation trigger with a predetermined or random schedule matched in number and duration of intervention windows. Keep P on the frozen PushObj model and S on the Maze adaptive model. Compare success, baseline-success preservation, and time spent revisiting stalled neighborhoods. This directly tests whether execution evidence is more useful than merely perturbing the planner occasionally. New planning runs only; no offline training required.
3. **Test the memory mechanism against generic motion encouragement.** On Maze, compare S against either a matched generic displacement bonus or a visit penalty without stalled-state selection, fixing coefficient selection and planning compute on development data. Measure final success and revisitation, then freeze settings for confirmation. This substantiates the distinction between task-directed stagnation regularization and simply encouraging movement. The goal-excluding kernel's radius/strength sweep is useful appendix support, but is lower priority than a meaningful behavioral baseline.

Before additional runs, prioritize a fresh source-disjoint confirmation suite for the frozen chosen configurations; this is evaluation confirmation rather than another ablation. No new experiment is required to restructure the paper and expose the existing controls now.
