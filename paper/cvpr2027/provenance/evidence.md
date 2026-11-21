# Verified evidence for manuscript

Read-only CPU aggregation of existing logs; no model inference or new rollouts.

## Main PushObj table: common30-replan budget,350 shared development tasks

| Checkpoint | Planner | Baseline | Memory only | Proprio only | Both |
|---|---|---:|---:|---:|---:|
| official | frozen | 153/350 (43.7%) | 157/350 (44.9%) | 210/350 (60.0%) | 209/350 (59.7%) |
| official | adajepa | 243/350 (69.4%) | 250/350 (71.4%) | 255/350 (72.9%) | 256/350 (73.1%) |
| bias16k | frozen | 214/350 (61.1%) | 219/350 (62.6%) | 253/350 (72.3%) | 253/350 (72.3%) |
| bias16k | adajepa | 262/350 (74.9%) | 266/350 (76.0%) | 270/350 (77.1%) | 269/350 (76.9%) |

Source paths, file hashes and per-arm wins/losses are in evidence.json. These350 tasks are reused in all four rows; rows are not independent studies.

## PushObj bias16k frozen ablation at30 replans

| Method | Success | Rescue / harm | Exact paired p |
|---|---:|---:|---:|
| SZ_mid | 219/350 (62.6%) | 5 / 0 | 0.0625 |
| T0P | 253/350 (72.3%) | 39 / 0 | 3.638e-12 |
| T0SP | 253/350 (72.3%) | 39 / 0 | 3.638e-12 |
| ALPHA10 | 232/350 (66.3%) | 50 / 32 | 0.05981 |
| T0SPL | 243/350 (69.4%) | 29 / 0 | 3.725e-09 |
| LA_start | 221/350 (63.1%) | 34 / 27 | 0.4426 |

## Maze: corrected action_repeat4,100 medium tasks,seed100

| Method |20replans|30replans|50replans|Rescue/harm at30|Rescue/harm at50|p at50|
|---|---:|---:|---:|---:|---:|---:|
|AdaJEPA|51|63|82|—|—|—|
|R-trig|50|66|83|7/4|5/4|1|
|R-switch|52|65|85|2/0|4/1|0.375|
|R-delay|51|61|83|2/4|5/4|1|
|SZ-switch|53|66|87|3/0|6/1|0.125|
|P-switch|52|64|84|1/0|4/2|0.6875|
|F-trig|47|55|60|3/11|2/24|1.049e-05|
|F-switch|53|63|78|2/2|2/6|0.2891|

5 held-out validation layouts,20tasks/layout; BFS start-goal distance6–9blocks. Horizon25control calls (100 base env steps),execute5calls (20base steps) per replan; frameskip5 gives5latent horizonsteps. At30replans control budget150calls/600base steps. No gate stops adaptation.

## Training recipe

a_exec = a_source + 9 * (object_CoG - pusher) / norm(object_CoG - pusher) + 10 * Normal(0,I_2) [pixels]; environment input = a_exec / 100; store executed actions.

Actual source placement,4retry attempts,20px/.3rad jitter,quality gates.16k train/4k val; T/L/Z/+ shapes. Prior report text claiming centroid placement is outdated.

| Training evaluation setting | Frozen (mean ± SD) | AdaJEPA (mean ± SD)|
|---|---:|---:|
|test100_it20|58.57 ± 2.16|70.19 ± 2.33|
|test200_300_it30|57.90 ± 3.45|71.19 ± 2.44|

SD across3training seeds0/1/2. Eval seed100 uses20replans; eval seeds200/300 use30. The released checkpoint is one model, so comparisons are data-recipe associations, not purebias causal experiments.

## Target reuse

{
  "identity": "shape, author ep_idx, segment offset",
  "streams": {
    "S7": {
      "slots": 280,
      "unique_segments": 280,
      "unique_source_episodes": 280
    },
    "EVAL50": {
      "slots": 350,
      "unique_segments": 350,
      "unique_source_episodes": 346
    },
    "EVAL50s200": {
      "slots": 350,
      "unique_segments": 350,
      "unique_source_episodes": 344
    }
  },
  "pairwise": [
    {
      "a": "S7",
      "b": "EVAL50",
      "shared_segments": 0,
      "shared_source_episodes": 0
    },
    {
      "a": "S7",
      "b": "EVAL50s200",
      "shared_segments": 0,
      "shared_source_episodes": 0
    },
    {
      "a": "EVAL50",
      "b": "EVAL50s200",
      "shared_segments": 35,
      "shared_source_episodes": 41
    }
  ]
}

## Method faithful to code

**distance.** D_alpha(z,zprime)=mean((visual-visualprime)^2)+alpha*mean((proprio-proprioprime)^2), alpha=1; latent has visual and proprio branches; proprio may encode velocity.

**trigger.** r_t=sqrt(D(z_t,z_{t-W})/max(D(z_t,z_goal),1e-12)), W=3; w_t=clip((0.15-r_t)/(0.15-0.05),0,1). Store observation and w whenever positive; re-encode all memory observations with current adapted encoder every replan.

**regularizer.** P(A)=sum_h c_h sum_j [w_j D(m_j,g)/(H+1)] [max(0,1-D_sp(z_h,m_j)/(beta D_sp(m_j,g)))]^2 / sum_h c_h; c_h=2^h, beta=.25, kappa=2. Distances/rho clamped at1e-12. Sum over memories (not averaged), persistent memory until episode reset, no deduplication.

**radius.** beta multiplies squared distance; corresponding Euclidean radius is sqrt(beta)=0.5 times goal distance. Goal outside each memory support when beta<1. At exact center a squared radial kernel has zero own gradient; no escape guarantee.

**proprio.** Boost alpha by10 for the next3 plan calls after a trigger, renewed each trigger. Implementation scales both predicted and target proprio bysqrt(10) before original objective. Baseline trigger distance uses alpha1 even during boost.

**switch.** If stalled and (bar_l[t-W]-bar_l[t])/max(bar_l[t-W],1e-12)<theta=.05, activate corrections; bar_l[t]=(l[t-2]+l[t-1]+l[t])/3, W=3, K=1. l is newest not-yet-updated transition prediction MSE. Switch occurs after env feedback and effects begin next plan. Once switched remains active; updates CONTINUE unless freeze control explicitly true.

**objective.** Maze: weighted all-frame objective including context, weights normalized then averaged over T=H+1 frames. PushObj: last-frame objective for MPC iter<H, then same all-frame objective. GD Adam lr.1,100 iterations,zero init, warm starts unexecuted actions.

**action_offset.** K2P uses physical pusher displacement<10px over3 replans, checks start=10 (next plan is eleventh execution), min gap4, max7 escapes. On escape add v/||v||*min(15px,||v||/5) to each of5 executed env actions, v=goal pusher position-current pusher; prop boost10 for kick plus2 replans. This is a separate physical trigger branch, NOT combined with latent-memory main T0SP arm.

**training.** Bias data collection can use simulator object CoG. Deployed latent trigger/regularizer/proprio weight use observed images/proprio and goal observation, no oracle object state. K2P uses pusher goal proprio. Training observer B labels also not the same as online stall trigger.

## Claims and reporting limits

- All intervention comparisons are repeatedly used development targets; no independent confirmatory holdout or new multi-seed intervention result exists.
- PushObj four checkpoint/planner combinations share the same350 tasks; cannot pool as1400 independent episodes or four independent replications.
- Maze best SZ-switch selected among several developer arms:30replans66/100 vs63/100;50replans87/100 vs82/100, p=.125 at50. No statistically significant general Maze improvement demonstrated.
- SZ-switch is penalty-only and has no PushObj counterpart in E3; strongest PushObj result uses proprio reweighting. Do not present per-domain selected maxima as one tested fixed policy.
- Old Maze action_repeat1 failures and corrected action_repeat4 E3 run are different temporal protocols. All arms within E3 use4.
- Training recipe combines action bias9px, noise10px, original-source placement with retries/jitter, contact selection and filtering; historical released-model comparison does not isolate action bias or match all training factors.
- Historical report.md source-placement description is stale; generation_config.json is actual recipe. Bias/noise parameters calibrated with author released evaluation-segment statistics; targets are development data.
- Author training-source/test-source overlap cannot be ruled out; source training/own validation pools are separated, but original author evaluation ancestry is unverified.
- Three historical training seeds exist; four B training-screen arms and intervention studies use one training seed. Historical bias16k seed0 differs from new B0 (no decoder, initialization/runtime differences); do not combine B0 62.86 with old corrected72.29 as a paired gain.
- Negative B2/B3 do not causally establish collapse, goal conflict, or impossibility of representation regularization. Common-candidate diagnostics are fixed candidate choice, not MPC success nor an oracle reachability bound.
- Moving losses are on different transitions; loss gate is not a convergence/overfitting/learnability detector. Freezing controls lower Maze success; adaptation should continue.
- Planning-stage gain is largely pusher-goal completion: old40replan T0SP rescues37/54 object-already-tolerant failures and2/78 object-not-at-goal failures. This breakdown is40budget and should not be silently called30budget.
- Penalty is a compact-support latent repulsion, not demonstrated learned geodesic distance, progress guarantee, globally optimal planning, or novel general potential-field principle.
- B3 positive-effect separation differs from planning-stage stagnation memory regularizer; failure of one does not test the other.

## Explicitly pending experiments

- Independent held-out task stream and Maze seeds with frozen hyperparameters
- Matched collection ablation separating action bias/noise/contact selection while fixing source episodes, sample count and training schedule
- Three-seed intervention replication
- Complete shared intervention-component matrix across both domains
- Task-completion breakdown at matched budgets and long-horizon/contact-specific goals
