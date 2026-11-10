# v7 full-manuscript narrative audit

Read all eight section sources, both main-file captions, and `anti-defensive-writing-en/SKILL.md`. This is a read-only manuscript audit; no experiments or shared manuscript edits were made. The recommendations keep the evidence boundary intact while putting the advantage at the start of each paragraph. All existing table values, labels, equations, symbol definitions, and scientifically necessary qualifiers should remain unless a specific prose replacement is given below.

## Highest-value changes

1. Make the central insight **execution history can improve action selection even when the predictive model is fixed**. The data recipe supplies a complementary learning-stage gain. Do not imply that all 28.6 points come from ECP: ECP alone gives 61.1→72.3 on the augmented-data model, whereas data+ECP gives 43.7→72.3 relative to the official frozen model.
2. Present each experiment as an answered question. Prefer “Conditional intervention preserves successful behavior” to “Conditional versus persistent intervention”; prefer “Joint latent memory captures the stalled scene” to “Regularization geometry and strength.” The latter claim should still be tied to the observed ablation rather than stated as a causal proof.
3. The best mechanism evidence is the **39 rescued tasks with all 214 native successes preserved**, compared with 50 rescues and 32 regressions under permanent reweighting. Make this visible in the main analysis. It supports conditional correction much more directly than another long catalogue of percentages.
4. Keep one clear development-status statement in the main setup and full sampling/calibration details in the supplement. Remove project-management voice (“requested,” “historical,” “kept separate from every completed experiment,” “these results come from existing logs”) where it adds no scientific meaning. Provenance can hold history.
5. The four-way component table must retain P=77.1 and S+P=76.9. Do not call combined S+P the best in every setting. The honest positive conclusion is that both mechanisms improve native control and their importance varies with the completion requirement.
6. Replace “Our data” table labels with “Augmented” or “Augmented data” for terminology consistent with the user's requested figure legend. Define once that this is the full contact-biased collection recipe, including noise/placement/filtering, rather than an isolated directional-bias intervention.
7. Do not claim SOTA, cross-family transfer, model convergence detection, guaranteed escape, or universal preservation of successful behavior. Current evidence supports two checkpoints within the same world-model family, frozen/adaptive execution, and observed task-list retention.

## Ready-to-paste main-body blocks

### Related work: replace the three paragraphs, retain section label

```tex
\paragraph{Latent world models for control.}
Learned world models support action selection by predicting future states in a compact representation. World Models introduced learned simulators for behavior learning~\cite{ha2018worldmodels}; PlaNet optimizes latent action trajectories, Dreamer learns behavior from imagined experience, and TD-MPC2 combines latent planning with learned rewards and values~\cite{hafner2019planet,hafner2020dreamer,hansen2024tdmpc2}. Joint-embedding predictive architectures learn dynamics through representation prediction~\cite{assran2023ijepa}. DINO-WM uses pretrained visual features, V-JEPA~2 supports image-goal robot control, and LeWorldModel jointly learns a compact encoder and predictor with Gaussian regularization~\cite{zhou2025dinowm,assran2025vjepa2,maes2026lewm}. We study a complementary control problem: how execution feedback can improve action selection with an existing latent world model.

\paragraph{Representation geometry for planning.}
Value-aware model learning relates prediction to downstream decisions~\cite{farahmand2017valueaware}. Temporal Straightening regularizes latent trajectory curvature for gradient-based planning~\cite{wang2026straightening}; AnisoWM shapes the metric induced by latent distance~\cite{kang2026anisowm}; and D-JEPA learns decision-relevant relations among candidate outcomes and studies visual/proprioceptive fusion~\cite{liu2026djepa}. ECP adapts the use of this geometry during execution. Stalled observations specify where to add a local planning cost, while conditional goal reweighting changes which completion component guides recovery. Frozen-model evaluations isolate these changes to action selection from changes to the representation.

\paragraph{Online adaptation and movement priors.}
MPC repeatedly optimizes actions using new observations~\cite{garcia1989mpc}. AdaJEPA additionally updates the world model from executed transitions, and OpTTA incorporates imagined goal-reaching cost into test-time adaptation~\cite{wang2026adajepa,mete2026optta}. ECP complements model adaptation by conditioning the planning objective on execution history. Behavior and skill priors guide movement~\cite{tirumala2022behaviorpriors,pertsch2021spirl}, while artificial potential fields shape local motion~\cite{khatib1986potential}. Our regularizer constructs a compact latent potential from stalled states encountered within the current episode. At training time, we build on AdaJEPA's contact-biased PushObj collection procedure to examine the complementary value of interaction-rich data. Auxiliary inverse-dynamics supervision~\cite{pathak2017curiosity} is evaluated separately through downstream control.
```

### Method: add an overview paragraph immediately after the section label

```tex
ECP makes observed execution part of the planning objective. A displacement-based signal records stalled configurations, a trajectory regularizer discourages their predicted revisitation, and a temporary change in goal weighting redirects recovery. These corrections act on candidate actions and can accompany either frozen dynamics or online model updates. \Cref{fig:method} shows the resulting controller and its connection to interaction-rich training data.
```

### Method: preserve all displayed mathematics and definitions; replace selected lead-ins/conclusions

Before `eq:rate`, replace the current first sentence with:

```tex
Stagnation is a property of executed motion: the controller repeatedly acts while remaining near the same state. We measure net displacement over $W$ completed replans relative to the remaining goal distance:
```

After the existing definitions of the displacement signal and thresholds, use this final sentence:

```tex
Because the signal measures net displacement, it detects local back-and-forth motion as well as stationary behavior.
```

Before the memory definitions, replace the introductory paragraph with (all symbols retained):

```tex
Execution identifies where the current controller has stalled. We convert this history into a local cost on future predicted trajectories. Before each planning call, every stored observation is re-encoded as memory state $m_j$, where $j$ indexes entries in $\mathcal M_t$ and $w_j$ is its recorded trigger weight. Its goal-scaled amplitude $s_j$ and squared support radius $\rho_j$ are
```

Replace the prose after the memory-weight definitions with:

```tex
The temporal weights $\pi_h$ sum to one and emphasize later predicted states, encouraging the plan to leave the stalled neighborhood by the end of the horizon. We use $\beta=0.25$ and $\kappa=2$; the $1/(H+1)$ scale follows the native all-frame objective. The initial context is excluded because candidate actions cannot change it. Memory persists throughout the episode, allowing the controller to account for earlier stalled configurations when replanning.

The regularizer provides a spatially local correction. Its support radius is half the memory-to-goal distance in the norm induced by $D_{\alpha_0}$. Away from the numerical floor, the goal lies outside each kernel's support; both the penalty and its first derivative vanish beyond that support. Thus the added cost discourages revisiting stalled neighborhoods while retaining the original objective elsewhere. \Cref{sec:properties} gives the derivation and the dependence of the action gradient on the learned dynamics.
```

Temporary reweighting opening (retain equation and all following numerical definitions):

```tex
Spatial memory specifies which configurations to leave. Recovery can also require a change in which goal component receives priority. We therefore pair the regularizer with a temporary increase in proprioceptive goal weight. A positive trigger, with the correction gate open, sets
```

Temporary reweighting post-equation paragraph:

```tex
The multiplier is $b=10$. The interval covers the next three planning calls and is renewed by later triggers. PushObj requires both object placement and pusher positioning; after the object is aligned, increasing the proprioceptive contribution can direct the remaining motion toward the pusher goal. Conditioning this change on execution retains the native weighting during preceding contact and transport. Memory-only, weighting-only, and combined variants isolate the two components.
```

Scheduling subsection opening, before the unchanged loss/gate definitions:

```tex
Model adaptation and planning correction address different parts of control and need not start together. The trigger-only variant sets $g_t=1$ throughout; an empty memory gives zero regularization before the first trigger. For adaptive models, we also use recent prediction improvement to schedule correction.
```

Keep the exact definitions of `ell`, `bar ell`, `q`, `W_g`, `tau_g`, windows, persistent gate, and continuation of updates. End with:

```tex
The gate coordinates correction with ongoing prediction learning; model updates continue after it opens.
```

Data subsection opening, before the unchanged collection equation:

```tex
Interaction-rich training data provide a complementary route to stronger control. Building on AdaJEPA's collection procedure~\cite{wang2026adajepa}, we augment source actions with an object-directed bias and Gaussian noise to increase contact-rich experience:
```

Retain every collection-variable definition and full-recipe qualifier. End with:

```tex
Executed actions are stored with their resulting observations. The full augmentation recipe also includes source placement, retries, jitter, and quality filtering. We retain the original latent-prediction and trajectory-straightening objective~\cite{wang2026straightening}, so the comparison evaluates the collection/training recipe without introducing an auxiliary representation loss. \Cref{sec:implementation} specifies the objective and the 16,000/4,000 training/validation episodes.
```

### Experiments: replace opening and result paragraphs; keep all setup facts and all table entries

Opening:

```tex
Our experiments test three claims: interaction-rich data strengthen the learned controller, execution-conditioned planning adds gains with the same world model, and these gains depend on when and how correction is applied. We evaluate manipulation on PushObj and navigation on Maze, followed by component and scheduling ablations.
```

Keep the setup paragraph's exact tasks, shapes, success predicates, protocol, pairing, and development declaration. Concise final setup sentences:

```tex
Planner comparisons within each checkpoint share tasks and action initialization. All reported task lists were used for development; the data comparison evaluates the complete augmentation and training recipe. Sampling, calibration, replication, and trajectory-reuse details appear in \cref{sec:implementation,sec:replication,sec:reuse}.
```

Result paragraphs:

```tex
\paragraph{Data augmentation strengthens both frozen and adaptive control.}
Interaction-rich training improves the model before any planning correction is applied. With native MPC, the augmented-data checkpoint raises PushObj success from 43.7\% to 61.1\% under frozen dynamics and from 69.4\% to 74.9\% with AdaJEPA (\cref{tab:main}). The gains of 17.4 and 5.4 points show that the data recipe benefits both direct use of the learned dynamics and subsequent online adaptation. Retained training trajectories average 60.6 contact steps, providing sustained experience of action--object interaction; \cref{sec:implementation} gives the full collection procedure.

\paragraph{Execution feedback unlocks further gains with the same model.}
ECP increases success with the augmented-data checkpoint to 72.3\% for frozen planning and 76.9\% with AdaJEPA. These are additional gains of 11.1 and 2.0 points over data augmentation alone, bringing the total improvement over the official baselines to 28.6 and 7.4 points. The frozen comparison isolates an improvement in action selection with fixed model parameters. ECP also improves the released checkpoint directly to 59.7\% and 73.1\%, respectively (\cref{tab:components}), showing that its benefit is present with both evaluated training recipes.

\paragraph{Stagnation regularization improves navigation without offline retraining.}
On Maze, gated regularization raises success from 63\% to 66\% using the same released checkpoint and AdaJEPA updates. It retains all 63 native successes and recovers three additional tasks. This evaluation isolates the planning-side contribution in a second control domain. \Cref{tab:maze} examines the component and scheduling choices that produce this behavior.
```

### Ablations: replace prose paragraphs, preserve all tables intact

```tex
\paragraph{Both components improve native planning.}
Across both checkpoints and update policies, stagnation regularization and temporary goal reweighting each improve native PushObj success (\cref{tab:components}). Reweighting supplies the largest manipulation gains: frozen success rises from 43.7\% to 60.0\% with the official model and from 61.1\% to 72.3\% with the augmented-data model. Combined correction improves the four native baselines by 16.0, 3.7, 11.1, and 2.0 points, respectively. Its result remains within one task of reweighting alone in every setting. The components therefore serve distinct recovery needs: goal-component emphasis dominates the measured PushObj gains, while the Maze controls below isolate the benefit of spatial stagnation memory.

\paragraph{Conditional intervention preserves successful behavior.}
The timing of reweighting is essential to its benefit. With the same $10\times$ multiplier, conditional activation solves 253 tasks versus 232 for permanent reweighting (\cref{tab:timing}). Permanent weighting recovers 50 failures but changes 32 native successes into failures. Conditional weighting instead retains all 214 native successes and recovers 39 failures. The 21-task advantage shows why execution feedback is useful: it applies the change after observed stagnation while preserving the native objective beforehand.

\paragraph{Joint latent memory supports scene-aware correction.}
Joint visual/proprioceptive memory improves on the native 74.9\% success rate at every tested strength, reaching 76.0\% at $\kappa=2$ (\cref{tab:memory-strength}). Proprioceptive-only memory reaches 75.1\%. Including the visual state allows the regularizer to distinguish scene configurations that share an agent position. All six variants retain every native success under the fixed checkpoint, update rule, and trigger. \Cref{sec:additional-controls} provides exact counts and settings.

\paragraph{Correction complements continued adaptation.}
Gated memory achieves 66 Maze successes while retaining all 63 native successes (\cref{tab:maze}). Immediate $S{+}P$ correction also solves 66 tasks, with seven recoveries and four regressions; gated $S{+}P$ solves 65 with two recoveries and no regression. Thus the activation schedule changes which tasks benefit even when aggregate rates coincide. A fixed ten-replan delay gives 61 successes. Freezing adaptation gives 55 successes at the first stall and 63 at the loss gate, with both correction components disabled. These controls favor coordinating the planning objective with ongoing model learning. \Cref{sec:extended} reports extended budgets and paired statistics.

\paragraph{Alternative training and action interfaces.}
The separate action-space correction raises adaptive PushObj success from 74.9\% to 77.4\% (\cref{tab:action}), using a physical-displacement trigger and temporary weighting without latent memory. This result provides a second implementation of execution-conditioned correction. The auxiliary training-objective study in \cref{tab:b} evaluates effect weighting, inverse dynamics, and a positive-effect margin under matched training conditions. None improves task completion over its matched baseline; \cref{tab:diagnostics} reports their prediction and candidate-selection diagnostics.
```

The last paragraph keeps the negative result complete while avoiding letting an off-story training screen become the section's conclusion. If space is tight, retain the last two sentences as the appendix-pointer sentence and put the action-space result first as above.

### Qualitative analysis: stronger behavior-first opening with unchanged numbers

```tex
ECP converts repeated local motion into task completion in the matched executions of \cref{fig:teaser}. In PushObj, the frozen baseline remains within a 3.9-by-2.1-pixel region over replans 15--30. The object is already within 4.0 pixels of its goal, but the pusher remains 45.3 pixels away. Combined correction follows the same first 15 plans, then reduces pusher error to 10.2 pixels and succeeds at replan 17. The joint-error trace in \cref{fig:push-distance} shows this transition from stagnation to completion.

In Maze, the baseline's sampled trajectory over replans 7--30 travels 18.2 units yet has net displacement below 0.001 units, repeatedly revisiting the same corridor. Gated regularization redirects the subsequent trajectory and reaches the goal at replan 27. Both runs share the initial model and task and execute identical actions through replan six. These cases visualize the two recovery mechanisms measured in the ablations: completing the remaining goal component and leaving a repeatedly visited region. Additional rendered states appear in \cref{fig:push-frames,fig:maze-case}.
```

### Conclusion: replacement (~100 words)

```tex
\section{Conclusion}
Execution history provides a useful signal for improving control with a predictive world model. ECP turns observed stagnation into a local trajectory regularizer and a temporary change in goal-component weighting, improving action selection with both frozen and adaptive dynamics. Interaction-rich data supply a complementary training-stage gain. Together, data augmentation and ECP raise PushObj success from 43.7\% to 72.3\% for frozen planning and from 69.4\% to 76.9\% with AdaJEPA under the same protocol. Component ablations and Maze recoveries show how correction timing and spatial memory contribute to this behavior. The resulting approach connects predictive learning with feedback from executed actions.
```

## Captions: shorten methodology/audit boilerplate, retain interpretation controls

`fig:method` first sentence can be “**Learning useful interactions and correcting stalled execution.**” The rest is already accurate; keep simulator-vs-schematic distinction.

`fig:headline`: current caption is clear. Rename “Interaction-rich training and execution feedback improve control” to “**Data augmentation and ECP provide complementary gains.**” Keep 350 tasks/30 replans/released baseline explanation.

`fig:teaser`: keep real saved-state renderings, matched prefixes, and replan-sampled polyline qualifications. It is long, but these distinguish genuine trajectories from cartoons and prevent a false fine-grained trace claim. Move only the detailed inset-displacement explanation to the appendix if needed for layout.

`tab:main`: use “**Data augmentation and ECP improve frozen and adaptive control.**” Current remainder properly defines official reevaluation and gain calculation.

`tab:components`: retain row-wise bold policy, because P slightly exceeds S+P in two rows.

`tab:memory-strength`: “**Joint latent memory improves regularization across the tested strengths.**” Retain disabled-P condition, native baseline, shared trigger/radius.

`tab:maze`: current caption correctly states correction updates continue and freeze controls disable correction; retain these exact distinctions.

## Appendix: exact targeted edits that reduce self-audit voice without hiding material facts

1. Rename `Training-Screen Objectives and Diagnostics` to `Auxiliary Training Objectives and Diagnostics`; keep `sec:training-screen`.
2. Replace appendix opener with: “This supplement specifies the implementation and evaluation protocol, reports additional controls and diagnostics, and documents the simulator renderings. Unmeasured entries in the experiment plan are marked by dashes.”
3. In collection details, retain training/validation split and unestablished released-source ancestry. This is material evaluation context, not optional defensive rhetoric.
4. In regularizer properties, replace “Its effect on executable actions depends on the learned Jacobians. It encourages leaving observed stalled neighborhoods, without guaranteeing a feasible escape from every state.” with “The induced action gradient is determined by the learned Jacobians and favors predicted trajectories that leave the stored neighborhoods.” Keep the exact zero-gradient-at-center and numerical-floor statements: they delimit the mathematical claim.
5. Replace “separates the requested 30-replan view from the available 50-replan traces” with “reports success at 20, 30, and 50 replans from the same execution traces.” Keep non-independence of budgets, delayed-gate definition, exploratory status, source dependence, calibration, and stream overlaps.
6. Replace “These values describe discordance on the existing task lists. They are not corrected for selecting configurations on the same development data, and within-source task dependence is not modeled. We emphasize counts and effect sizes in the main text.” with “These exploratory paired summaries use the development task lists, without configuration-selection correction or a within-source dependence model. The main text reports task counts and effect sizes.”
7. In replication opening, remove “These identifiers determine sampling rather than defining distinct benchmarks.” The table and protocol already define what the identifiers mean. Keep original budgets and training-initialization counts.
8. In the strength grid, replace “These results come from an existing internally paired grid” with “The six conditions use an internally paired evaluation.” Keep identical-baseline actions/outcomes and duplicate-condition post-activation-action distinction; it documents the provenance of numbers reused across tables.
9. In `B2`, replace “This tests action recovery from the learned transition representation; it is not a reproduction of a complete published inverse-dynamics world-model architecture.” with “This controlled auxiliary-head ablation tests action recovery from the learned transition representation.” The scope remains precise without pre-emptively debating unrelated systems.
10. In diagnostic paragraph, retain “candidate-selection measurements” and the actual stall-proxy definition; replace “It is not a direct per-physics-step measurement of stalled duration” with “The unit is therefore a replan-window equivalent in control steps.” Retain that candidate availability is not a closed-loop performance bound, preferably succinctly: “Candidate endpoint success is a diagnostic and does not bound closed-loop MPC success.”
11. Replace simulator provenance paragraph with: “All displayed states were rerendered from logged physical states in the original Pymunk and MuJoCo environments on CPU. Saved positions have three-decimal precision for PushObj and four-decimal precision for Maze. The assets also include the Maze source initial/goal images. Rerendering generated the displayed frames from these states without executing additional planning or action trajectories.” This is clearer than repeated “not X, not Y” and still avoids falsely claiming rollout-time screenshots.
12. Pending table caption: “**Planned additional evaluations.** Dashes denote unmeasured entries. Hyperparameters and selection rules will be fixed before evaluation.” Preserve all unrun rows and values.
13. Replace paragraph after pending table with: “The planned collection factorial will separate directional bias, action noise, and filtering effects. The training-side study will evaluate a goal-conditioned regularizer through both candidate ordering and closed-loop success. Existing auxiliary-objective results are reported in \cref{sec:training-screen}; no training-side stagnation-regularizer gain is included in the main results.” This preserves the absence of such a result without a project-summary discussion of failures.
14. In Artifact Provenance, the first three sentences are useful; final “no additional experimental runs” and “combined PDF...” are artifact-release notes suited to README. Removing them from the paper changes no scientific statement. Keep LLM usage disclosure intact.

## Whole-paper consistency checks after integration

- User-facing term: `Data augmentation`; full scientific definition: interaction-biased collection recipe. Main table can use `Augmented data` to save width. Do not call it synthetic image augmentation.
- “Our method” should distinguish ECP from its data+ECP final configuration whenever a numerical total is given.
- S is a planning-stage trajectory cost. Training retains TS; the B objectives do not establish a successful training-side stagnation regularizer.
- First-stall S+P Maze 66 has 7/4 rescue/regression. Gated S Maze 66 has 3/0. Never substitute one mechanism's count into the other's sentence.
- “Preserves all native successes” must remain tied to its recorded configuration/task set; no general safety or no-regression guarantee.
- More professional does not mean more inflated adjectives: avoid “powerful,” “universal,” “robust” (unless measured), “optimal,” “state of the art,” and “significant” without a valid statistical claim.
- Maintain every equation and symbol definition. The narrative edits should replace redundant prose, not remove executable-method details.
