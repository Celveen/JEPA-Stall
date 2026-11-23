# Factual review of main manuscript

Reviewed `/tmp/jepa_cvpr2027/paper/sections/*.tex` against `evidence.json` and implementation, 2026-11-05. No manuscript files modified.

## Required correction: Maze freeze controls

In `experiments.tex`, the labels `Trigger S+P, freeze` and `Gate S+P, freeze` misdescribe F-trig/F-switch. In `scripts/e3_run.py`, both use `kappa=0`, no proprioceptive boost, and `freeze=True`. They freeze without any planner correction.

Replace the table labels with:

```tex
Freeze at trigger (no $S/P$) &55&3 / 11\\
Freeze at gate (no $S/P$) &63&2 / 2\\
```

Replace the paragraph comparing55to66as if only updates differ with:

> Freezing at the trigger yields55successes versus63for the unchanged adaptive baseline; freezing at the gate yields63. These controls do not support stopping adaptation merely because a stall is detected. The correction variants therefore preserve the opportunity to learn from later transitions.

The55-versus66comparison changes both continued adaptation and S/P correction and cannot isolate the former.

## Required clarification: the gate retains pre-activation memory

In `method.tex`, replace:

> Before activation, alpha_t=alpha_0 and the memory is empty, reproducing the base planner.

with:

> Before activation, alpha_t=alpha_0 and the memory penalty is disabled, reproducing the base planner.

For ungated runs memory starts at the first trigger. For gated runs `SwitchMPCPlanner._post_env_feedback` calls the parent first, so it collects stall observations before the gate opens. `_build_mem` suppresses the penalty until the switch. Add to the gate subsection:

> Stalled observations are accumulated before the gate opens; activation makes this existing memory available to the next plan. Once opened, the gate remains open, while later stall detections continue adding memories and renewing the temporary boost.

The gate does not have to remain satisfied for future corrections. It never closes again, and it does not freeze the model in R/SZ/P runs.

## Timing confirmed; optional precise sentence

The current statement “starts checking after ten completed replans” is correct for main PushObj. At feedback after replan10, `self.iter=9`, `nxt=10`, so the detector can store the first memory and arm a boost. The first changed planning call is replan11. The three-call boost covers11,12,13and can be renewed. Maze E3 starts checking after4completed replans; the3-point/W=3loss comparison can first be evaluated after replan6, so a gated intervention first affects replan7. To remove indexing ambiguity, use:

> The main PushObj trigger is first evaluated after the tenth executed replan, and any resulting correction begins with the eleventh plan.

The directional K2P arm uses the same start10/next-call convention and has no latent-memory component.

## Minor implementation wording

1. Trigger subsection: “All observations in the window are encoded together” is broader than code. Only the current observation, the observation Wreplans earlier, and goal are encoded together. Replace with “The current and lagged observations and the goal are encoded together with the current model.”
2. Collection Eq.(collection): the implementation returns zero direction when norm<=1e-6; it does not divide by max(norm,1e-12) in that edge case. The action-offset variant likewise returns zero for norm<=1e-6. Define a normalized direction u(v)=v/||v|| if||v||>1e-6, else0; useu(v)in both equations, or state this convention in the appendix.
3. Training-screen B2 input includes both real visual and proprioceptive encoded differences. “from visual transitions” should read “from visual/proprioceptive representation differences.”
4. Goal reweighting Eq.(boost): “a trigger is active” should mean “the boost interval is active,” because weighting persists for three plans even if the current stall predicate is false.
5. In Eq.(objective), a gated variant is technically `J0 + g_t*kappa*P`, with g_t=0untilgateopening. It is enough to define P=0whenbranchdisabled or introduceg_t; avoid implying raw nonemptymemoryautomaticallychangespre-switchcost.

## Checked and consistent

- All counts/percentages in main PushObj matrix, timing table, Maze table, action-prior table, B0--B3table, and three-training-seed table match evidence.
- Main combined gains16.0,3.7,11.1,2.0pp are accurate.
- Maze30replan gate-S:66/100,3rescues/0harms,p=.25;50replan82to87is separately identified.
- PushObj physical success is a joint4Dpositionnorm<20px plus wrapped angle<pi/9; Maze positionnorm<.5; repeats and control budgets are correct.
- Trigger distance, weights, alpha1metric, beta .25squaredradius, memorysum, H+1normalization, persistent memory and re-encoding are faithful.
- Memorykernel support and zeroown-gradient atcenter statements are correct for the squared metric; it is C1at support boundary, not generally C2. Calling it “smooth” informally is fine; do not claim infinite differentiability.
- Actionoffset15px,5executedactions,minimum4replaninterval,maximum7events,3planboostmatchK2P_short2.
- Frames and quantitativecasevalues agree with figureagent's pairing audit.
- Development status, cross-row dependence, targetstream overlap, recipeconfounding, and distinctB0checkpoint are stated.
- Candidatecounts/ranks/variances and standardizedpredictionMSE match stored diagnostics; they remaindiagnosticassociations, as text states.
