# Narrative revision: execution history as a planning signal

Prepared 2026-11-06. Read the current manuscript, provenance notes, and the installed anti-defensive-writing skill. The skill is used to foreground the supported contribution; numerical results and material protocol qualifications remain explicit.

## What the two source papers actually do

**TS.** The introduction makes a concrete planning problem visible, then identifies a representation property that would help. Its method develops the model, geometric principle, and training objective; a separate section connects that geometry to planning. Experiments probe the representation before measuring downstream planning. Its figures also move from an intuitive contrast to a mechanism and realized trajectories. The useful pattern is **observable phenomenon → design property → explicit regularizer → mechanism evidence → downstream benefit**. The present paper can follow that reasoning order while using execution history, rather than temporal curvature, as its organizing idea. [Temporal Straightening, v3, introduction and sections 3–5](https://arxiv.org/html/2603.12231v3)

**LeWM.** The introduction commits to one central capability and a compact mechanism. Method separates learning from planning. The next section measures control, efficiency, and training stability; a later section probes physical structure to explain what the model learned. The figures distinguish training from planning with actual observation thumbnails and restrained modules. Its useful pattern is **one central claim → short operational mechanism → decisive practical evaluation → targeted explanatory probes**. Our manuscript should similarly make the ECP objective easy to find and give each experiment one argumentative role. [LeWorldModel, v3, sections 1 and 3–5](https://arxiv.org/html/2603.19312v3)

These are structural lessons, not claims that ECP reproduces either method or inherits their theoretical guarantees. No source sentences or artwork need to be copied.

## One sentence that should govern the manuscript

**Execution history identifies locally unproductive behavior; using that history to reshape the planner's objective restores progress with the same predictive model.**

Contact-rich training is a complementary interface: it supplies useful interactions for the world model. It should strengthen the narrative without competing with ECP for the central novelty claim. Contact bias itself is already present in AdaJEPA's collection procedure. The reported collection result also changes several recipe components together, so keep the existing complete-recipe qualification in the protocol.

## Concrete structure

1. **Introduction.** Open with real repeated-local-motion behavior, explain what execution feedback tells the controller, then introduce the two corrections. Place the same-model recovery evidence before a dense literature discussion. Finish with three contributions ordered mechanism, controlled evidence, training/execution analysis. The current opening about two interfaces is accurate but abstract; it does not immediately give the reader a recognizable phenomenon.
2. **Related Work.** Keep the existing concise organization. TS and LeWM are clearly cited here and in implementation provenance; they are not names for new ECP components.
3. **Execution-Conditioned Stagnation Regularization.** Start with a short overview of the base planner and a single readable objective. Then cover (i) detecting low displacement, (ii) constructing local stagnation regularization, (iii) temporary goal reweighting, and (iv) activation during adaptation. Put the regularizer before weighting so the term requested by the user is no longer buried in a subsection called only “memory.” A short final subsection can cover interaction-biased collection, referring exact recipe details to the appendix. Directional action correction is a separately evaluated variant; place its exact formula in the appendix if space is needed for trajectories.
4. **Recovering Progress in Closed-Loop Planning.** State the shared budget and paired evaluation protocol once. Order evidence as (i) main frozen/adaptive component matrix, (ii) conditional versus persistent correction, (iii) trajectory-level recovery in PushObj and Maze, (iv) gate/adaptation timing. The main matrix must retain S-only, P-only, and S+P so the source of the gain is visible.
5. **Interaction Priors Across Training and Execution.** Lead with stronger frozen planning from the richer data recipe and the three-seed study. Give the auxiliary-objective study one clear role: checking whether proposed representation changes support actual action selection. Keep its exact results accessible, but move detailed diagnostic tables and objective definitions to the appendix if the two-environment trajectory figure needs room. Avoid turning this section into B0→B1→B2→B3 chronology.
6. **Conclusion.** Reinforce that the same model can support better execution when its objective incorporates episode history, and that interaction-rich data complements controller correction.

The supplied abstract and opening drafts are optional replacements, not edits to the shared manuscript. The abstract retains exact reported figures and explicitly says which component generated each headline. It stays below 190 words.

## Explain what “stagnation regularization” already means

The current method already contains an explicit planning regularizer: the memory cost P in the objective and its compact-support kernel. Rename it throughout to **stagnation regularizer**, with **episode-local memory** describing its implementation. Keep the existing S label in tables and define it once as “stagnation regularization.”

A concise explanatory paragraph after the objective can say:

> The stagnation regularizer adds cost when a predicted trajectory returns to a neighborhood where execution has already shown little displacement. Its support is local, and the goal lies outside each recorded neighborhood under the stated radius construction. It therefore adds a preference against repeated visits to observed stalled regions. Temporary goal reweighting acts on a different part of the objective: it changes the relative emphasis on scene agreement and agent positioning.

This is not a global speed reward. Do not describe it as requiring perpetual motion, increasing velocity everywhere, proving escape, or classifying every kind of failure. A radial smooth kernel has zero gradient at its exact center, and finite-memory local shaping does not give a global navigation guarantee. Those facts need not become a repeated defensive paragraph, but they should constrain claims and captions.

The training-side B3 margin is a different mechanism applied to sampled transition embeddings. It is not the same history-conditioned regularizer applied during training, and the recorded B2/B3 results do not establish a successful training version. The training story can be strong and truthful by focusing on interaction-rich data and the distinction between useful interactions, prediction metrics, and completed tasks.

## Make equations readable

- Use a named regularizer, e.g. R_stall(A; M_t), rather than the generic P which is also the table label for proprioceptive weighting. Preserve equation labels if possible. Do not quietly change the implementation.
- Define the candidate action sequence A, prediction horizon H, predicted latent state, visual/proprioceptive blocks, their dimensions, the base weight, and memory set before or immediately after first use.
- Factor the long memory expression into three parts: a per-memory support scale rho_j; a local kernel K_j(z); and a trajectory aggregation with normalized horizon weights. Define every factor next to the expression.
- Describe rho_j as a **squared-distance support scale**, not a Euclidean radius. The support radius is sqrt(beta) times the memory-to-goal norm, subject to the numerical floor.
- Explain [u]_+=max(u,0), clip, the numerical floor epsilon, the stall window W, low/high thresholds, the memory weight w_j, the regularizer strength kappa, the boost factor b, and the boost duration. Symbols should not rely on captions or an appendix for their first definition.
- Keep the base model objective and the ECP correction visually separate. One equation for the native metric, one for the corrected objective, and one compact block for the regularizer is easier to scan than several equally weighted auxiliary variants.
- State that detection uses the base proprioceptive weight and that stored observations are re-encoded with the current model. That prevents ambiguity during adaptation.
- The gate is a measured short-window prediction-loss change, not a proof that learning has converged. Model updates continue after opening it.

## Real trajectories: what each panel must establish

Use recorded states with original simulator renderings and overlay the logged agent trajectory. Show a complete thin trajectory plus an emphasized late segment, with start, goal, and terminal markers. Make time direction legible with a few arrows or sparse replan labels. A close crop around the revisited region can show repeated local motion without fabricating displacement.

The failure panel should support “repeated local motion” directly. For near-static baselines, a zoom or local position-versus-time inset is more informative than claiming oscillation not visible in the trace. If saved states are only at replan boundaries, call the overlay a **replan-state trajectory** and do not imply densely logged control motion.

Each baseline/corrected pair must share the same task and initial model; list the exact correction, budget, trigger/gate step, and outcome. Source verification for this revision identifies the PushObj case as sample 5 under T0SP (S+P; kappa=2, start=10, boost=10), succeeding at replan 17. Call it combined correction; it cannot isolate the contribution of S from P. The Maze sample 75 case uses SZ-switch (S-only; kappa=2, boost=1), succeeding at replan 27, and therefore directly illustrates the regularizer. The aggregate component table remains the relevant mechanism comparison.

Suggested caption pattern:

> **Observed stagnation motivates local correction.** Logged trajectories are overlaid on simulator renders for matched baseline and corrected episodes. The highlighted late segment shows repeated visits near the stalled region. [Exact PushObj component] completes the PushObj task at [...]; [exact Maze component] leaves the revisited region and reaches the Maze goal at [...]. Markers denote [...].

Replace all brackets from verified logs before using. Do not claim the failure visualization establishes the prevalence or cause of all failures.

## Main graphic vocabulary

Replace “JEPA+TS” with **world-model learning** or **encoder + latent predictor** in the top training band. Replace “memory penalty” with **stagnation regularizer** in the feedback branch. Label observations as observed/executed; label rollout and latent path sketches as predicted/schematic. Keep the optional update arrow neutral (“optional model update”), without another method's acronym in the central diagram. Credit the actual JEPA+TS backbone and AdaJEPA update recipe in implementation text and captions where needed.

## Contribution ordering and limits that matter

The main claim is a timing-and-history interface for latent MPC, not first use of potentials, action bias, or latent weighting. The strongest headline is the same-checkpoint conditional-versus-persistent comparison: 61.1% native, 66.3% persistent, 72.3% triggered. S+P improves four settings by up to 16.0 points; that is a combined result, not evidence that S alone accounts for the full gain. Maze S-only at 30 replans improves 63/100 to 66/100. Preserve the development-evaluation protocol, shared task list, one checkpoint per intervention recipe, complete-recipe data comparison, and unrun follow-up experiments in one clear protocol location.
