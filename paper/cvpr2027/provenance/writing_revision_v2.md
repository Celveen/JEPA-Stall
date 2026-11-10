# Revision v2: advantage-led narrative

Read-only review of the current CVPR 2027 manuscript. Applied `anti-defensive-writing-en` to strengthen the central argument while keeping all measured results and the development protocol accurate. The strongest story is **interaction priors at two interfaces: richer interactions during collection, and execution-conditioned preferences during control**. ECP remains the principal new method; the data recipe retains its AdaJEPA attribution.

## Recommended abstract

**153 words by whitespace count; below the requested 200-word maximum.** The 11.1-point gain comes from the exact counts, 39/350, rather than subtracting the rounded displayed rates.

```tex
\begin{abstract}
Successful latent planning depends on both the interactions represented by a world model and the actions favored during execution. We study interaction priors at these two interfaces. Contact-biased collection strengthens the training signal for manipulation, while execution-conditioned planning (ECP) turns recent observations into corrective action preferences. ECP uses normalized latent displacement to trigger temporary proprioceptive reweighting and local memory penalties around stalled states. It preserves the predictive architecture and operates with frozen models or alongside online adaptation. On 350 PushObj tasks under a shared 30-replan budget, conditional reweighting raises frozen-model success from 61.1\% to 72.3\%, an 11.1-percentage-point gain, and exceeds persistent reweighting at 66.3\%. The combined reweighting and memory variant improves four checkpoint/adaptation settings by up to 16.0 points. Maze evaluations additionally show how memory and adaptation timing support recovery in navigation. Together, these results show that interaction-rich training and execution-conditioned preferences provide complementary ways to turn predictive world models into effective controllers.
\end{abstract}
```

## Ten concrete main-text revisions

### 1. Introduction opening: establish the two interfaces immediately

Replace the opening paragraph with:

> A world model learns from the interactions collected during training and serves the action preferences expressed by its planner. These two interfaces determine which behaviors the system can execute successfully. We study interaction priors at both interfaces: contact-biased collection supplies rich manipulation experience, and execution-conditioned planning (ECP) uses recent observations to adjust control when progress stalls. ECP activates temporary goal reweighting and local stagnation memory, recovering successful behavior with the same frozen predictive model.

This establishes the positive contribution before the detailed account of stagnation. The following paragraph can explain fixed visual/proprioceptive priorities using the existing task example.

### 2. Introduction teaser paragraph: foreground the concrete recovery

Replace the paragraph beginning `\Cref{fig:teaser} illustrates...` with:

> \Cref{fig:teaser} shows the benefit of conditioning planning on execution. Starting from the same state and image goal, ECP completes the task at replan 17, whereas native planning remains unsuccessful at replan 30. Both runs use identical frozen model parameters and share their first 15 executed plans. The recovery follows a change in planning preference triggered by observed history.

This retains the unsuccessful baseline as essential evidence, while removing `without attributing...` and `does not need...` qualifications. The exact matched-prefix statement is supported by the existing figure provenance.

### 3. Introduction training paragraph and third contribution: describe their purpose

Replace the paragraph beginning `This perspective also clarifies...` with:

> Training and execution provide complementary places to introduce interaction priors. We examine an interaction-rich collection recipe motivated by AdaJEPA's contact-biased PushObj construction while retaining the JEPA+TS objective. This recipe strengthens frozen planning and supplies a stronger starting point for execution-conditioned correction. A matched auxiliary-objective study then examines how prediction, representation geometry, and action selection relate to completed tasks.

Replace contribution three with:

> An empirical account of interaction priors across data collection and execution, supported by training-seed comparisons, representation diagnostics, and matched downstream planning evaluations.

Keep B0--B3's complete numbers and conclusions in the analysis section. The introduction should identify what that evidence explains, rather than previewing a list of unsuccessful attempts.

### 4. Protocol: state development status once, in a dedicated paragraph

Retain a single clear protocol paragraph in `experiments.tex`, incorporating the material currently repeated in figure captions, the analysis scope subsection, and the conclusion:

> The reported target streams were used for controller selection and are treated as development evaluations. Collection hyperparameters were calibrated using statistics from the released evaluation segments; the collection comparison evaluates the complete recipe. Each planning comparison pairs the initial model, target, and random initialization. The four PushObj settings share one 350-task list, with one checkpoint per recipe; the separate data study uses three training seeds. Source-disjoint confirmation and intervention replication across training seeds are specified in \cref{sec:pending}.

Keep unaffected-trajectory reuse and prefix verification in the following existing paragraph or its appendix reference. This consolidates essential experimental status rather than deleting it. Afterward, remove the repetitive `Scope of the empirical claim` paragraph and repeated `development/exploratory` modifiers elsewhere; continue to avoid significance or generalization claims.

### 5. Method: express operational properties directly

Use the following local replacements:

- Trigger: `The statistic detects low displacement relative to the remaining goal distance. Episode completion is determined by the environment's success predicate.`
- Weighting: `The intervention increases the proprioceptive contribution in response to observed stagnation, using execution history as its activation signal.`
- Memory ending: `The resulting compact-support potential discourages revisits to recorded stalled neighborhoods. Its kernel properties, including the zero derivative at a memory center, are derived in \cref{sec:properties}.`
- Component distinction: `Temporary weighting changes the current goal emphasis, while persistent memory records previously unproductive locations. The component matrix measures their individual and combined effects.`
- Gate: `Successive transition losses provide an execution-dependent scheduling signal for opening the correction gate.`

Keep the existing equations, zero-center derivative, numerical-floor details, and gate's use of different transitions in the technical appendix. Avoid repeatedly contrasting the method with escape guarantees, semantic phase labels, learnability classifiers, or convergence tests that the paper does not claim.

### 6. PushObj results: make conditional activation's advantage explicit

Replace the end of the timing-results paragraph, beginning `Conditional activation gives up...`, with:

> Conditional activation preserves all 214 native successes and recovers 39 additional tasks. Persistent reweighting recovers 50 failures but loses 32 native successes. The 21-task advantage of conditional activation therefore comes from changing priorities selectively while retaining effective native behavior.

Keep the full rescue/harm table. In the component analysis, use:

> Temporary proprioceptive weighting supplies the largest PushObj gain. Memory alone recovers four to seven additional tasks across settings, and the combined variant improves all four settings.

Retain the existing statement that adding memory to weighting changes at most one task. It is the appropriate component attribution, not a weakness to obscure.

### 7. Maze: lead with observed recovery and scheduling behavior

Replace the prose spanning the current `exploratory effect` and freezing-control discussion with:

> Gated memory recovers three additional tasks while preserving all 63 baseline successes at 30 replans. The trigger-only combination reaches the same aggregate success through seven rescues and four harms, showing how the gate changes the executions receiving correction. The exact paired test for gated memory gives $p=0.25$; the complete budget and gate comparisons appear in \cref{tab:maze-full}. With continued model adaptation, the same recorded trajectories reach 87 successes at 50 replans, compared with 82 for AdaJEPA. Freezing adaptation at the trigger or gate yields 55 and 63 successes at 30 replans, respectively, supporting a controller that retains learning while scheduling planning corrections.

This keeps the small effect, paired-test result, unsuccessful freezing controls, and budget distinction. It removes the separate sentence rebutting an unmade cross-layout-generalization claim. Keep the 50-replan comparison supplementary if main-text space is tight.

### 8. Training-data analysis: state what the complete recipe contributes

Replace the paragraph beginning `The collection recipe combines...` after moving calibration/development details to the central protocol paragraph:

> The collection recipe combines directional bias, action noise, source placement, retries, and filtering. Together, these choices produce interaction-rich transitions for the unchanged JEPA+TS learner. The resulting checkpoint reaches 61.1\% frozen success in the shared matrix, compared with 43.7\% for the released checkpoint, and the three-seed study extends the data analysis across training initializations. Collection and execution-conditioned planning thus improve the system at distinct interfaces. \Cref{sec:pending} specifies a matched factorial study of the collection components.

Keep the comparison explicitly attached to the complete recipe. Avoid attributing the full gain to the 9-pixel directional bias alone or presenting its design as a first introduction of contact-biased data.

### 9. Auxiliary-objective analysis: retain the negative evidence as a focused diagnostic

Suggested subsection title:

> Representation diagnostics explain the need for downstream evaluation

Replace the initial result interpretation with:

> Under the matched protocol, JEPA+TS reaches 62.9\% success, effect-pair weighting 62.0\%, inverse dynamics 33.1\%, and positive-effect margins 38.3\%. All three auxiliary configurations have lower observed success than JEPA+TS. Common-candidate diagnostics identify the representation and selection changes accompanying these outcomes.

Replace the B2 diagnostic's final sentence with:

> Lower prediction MSE co-occurs with reduced visual variance, reduced effective rank, and poorer selection even when real candidate trajectories are encoded.

Replace the B3 diagnostic's inferential sentence with:

> B3 increases visual effective rank while reducing proprioceptive variance, exposing a different change in the metric used for candidate selection.

The final analysis can then state:

> These measurements motivate evaluating representation objectives through their induced action preferences and completed tasks. ECP directly incorporates the complementary evidence available during execution: which observed neighborhoods have repeatedly yielded little progress.

Keep the complete B0--B3 table and appendix diagnostics. These formulations state the unfavorable findings clearly, preserve their scientific role, and avoid turning correlations into a causal claim about rank collapse or branch imbalance.

### 10. Conclusion: finish on the capability and unified contribution

Replace the conclusion paragraph with:

> Interaction priors improve latent planning through the experience supplied during training and the preferences applied during execution. ECP uses recent observations to activate temporary proprioceptive reweighting and local stagnation memory, with a complementary bounded action correction. On PushObj, conditional reweighting raises frozen-model success from 61.1\% to 72.3\%, exceeding persistent reweighting at 66.3\%; the combined correction improves all four checkpoint/adaptation settings. The data study further demonstrates the value of interaction-rich collection, and the representation diagnostics connect training choices to downstream action selection. Together, these results establish execution history as a practical control signal for recovering progress with learned world models.

The task budgets and development status remain clearly stated in the central protocol. The conclusion reinforces what was measured and learned, without repeating pending-work lists.
