# Code-grounded equation and mechanism audit (2026-11-06)

## Deliverables

- `method.tex`: compact replacement method; all core coefficients defined immediately in text, no TS label in method-figure caption, explicit named stagnation regularizer, positive mechanism-led exposition.
- `parameters_appendix.tex`: append subsection under implementation. Defines all notation, exact inherited base objective, feedback/gating order, and action-offset equation moved out of the core method.
- `properties_appendix.tex`: replace current penalty-properties section. Gives kernel and action gradients, goal-exclusion condition, numerical-floor cases, optional proprioceptive kernel.

No code or manuscript edited outside this temporary deliverable directory. No experiment executed.

## Central finding

**The implementation already contains a real planning-side stagnation regularizer.** It is not merely the detector or the temporary proprioceptive reweighting. `planning/stagnation.py:204–226` adds `kappa * _penalty(z_obs_pred)` to the differentiable planning loss. `_penalty` sums compact-support squared-hinge kernels around observed stalled states, over ALL future predicted frames. The main-text term should be called `stagnation regularizer` and written `R_t^stall`, rather than generic `P` or only `local memory`.

Separate roles:
1. Detector `r_t,w_t`: execution-derived signal, not a trainable binary discriminator (`planning/stagnation.py:145–165`).
2. Stagnation regularizer S: penalizes predicted occupation of previously stalled neighborhoods (`planning/stagnation.py:215–226`).
3. Reweighting P: temporarily multiplies existing proprioceptive goal cost by 10 (`planning/stagnation.py:168–169,201–208`).
4. Gate: schedules S and/or P during online learning (`planning/switch.py:52–84`).
5. Action offset: a separately evaluated physical-displacement-triggered action modification (`planning/escape.py:149–155,177–190,271–288`).
6. Training action bias: changes transition collection; it is not evidence that the planner-side regularizer was trained into the model (`scripts/gen_pushobj_dataset.py:127–147,329`; `models/visual_world_model.py:339–377`).

## Mathematical audit

### Metric

Full-latent distance is branch-normalized visual MSE plus alpha times branch-normalized proprioceptive MSE, **not** MSE over their concatenation (`planning/objectives.py:25–31,49–57`; `planning/stagnation.py:150–153,180–185,218–222`). Trigger and memory always use base alpha=1, independent of a temporary alpha=10 planning boost.

### Temporal normalization

- Future latent horizon H=5 (25 control calls grouped into five blocks). Initial context is h=0.
- Base all-frame cost includes h=0..H, normalized weights 2^h/sum_{k=0}^H2^k, and an ADDITIONAL mean over H+1 frames (`planning/objectives.py:43–57`). Thus `J_all=(H+1)^(-1) sum_h omega_h D(...)`.
- Maze always uses all-frame. PushObj uses terminal-only for planning indices 0..4, then all-frame (`planning/objectives.py:60–68`; actual staged config).
- Regularizer excludes context, uses h=1..H, normalized weights 2^h/sum_{k=1}^H2^k, amplitude `D(memory,goal)/(H+1)` (`planning/stagnation.py:218–226`).
- Therefore wording 'a terminal penalty' or 'only penalizes end state' would be incorrect. It is a trajectory-level regularizer on every predicted state, with larger late-horizon weights.
- Source docstring saying 'staying costs (1+kappa) times its objective' needs conditions: full-weight single memory, all-frame cost, no proprio boost, predicted trajectory stays at that memory. Avoid presenting unconditional multiplicative identity.

### Kernel geometry and derivative

The main full-latent kernel uses D in both numerator and goal-distance-based radius. Since D is squared distance, beta=0.25 implies half-distance radius, NOT quarter-distance radius. Numerically actual rho is max(beta*D(m,goal),1e-12). Goal exclusion holds when floor inactive, beta<1; not a universal guarantee at zero/near-zero distances. With Q = diag(1/dv I, alpha0/dp I), derivative inside support is `-4/rho * (1-D/rho) Q(z-m)`. At the center gradient is zero despite maximal penalty. Outside support and at support boundary penalty and gradient are zero. Action gradient also passes through learned rollout Jacobian. These facts support 'discourages revisits / staying near stalled neighborhoods', not 'forces model always to move' or 'guarantees escaping / cannot drift back'.

### Memory and adaptation

`_reset_stag` resets memory every episode (`planning/stagnation.py:99–109`). Each positive trigger appends an observation and unrounded weight (`163–165`); logged weight is separately rounded. No deduplication/decay/memory-size normalization. Re-encode stored images and proprioception each replan (`174–185`) under `@torch.no_grad`; treat memories as constants during action differentiation. Memory remains after weight boost ends and can accumulate repeated states.

`ProbeMPCPlanner._post_env_feedback` first measures newest-transition loss and then updates model (`planning/adajepa_probe.py:145–171`). Stagnation feedback calls its parent first (`planning/stagnation.py:139–148`), so motion is calculated using the post-update encoder. Current, lagged, and goal observations are encoded together. Gate uses pre-update recorded losses but updated-encoder motion. Current paper's vague 'current encoder' was not wrong; new appendix clarifies exact order.

### Trigger, gate, and intervals

- Ratio r is NET displacement across W=3 replans divided by remaining goal norm, not traveled arc length; back-and-forth movement can yield small r (`planning/stagnation.py:145–159`).
- Main PushObj start=10 means check after ten completed plans; earliest intervention is plan 11 (one-based). Maze E3 uses START=4, not config's default7 (`scripts/e3_run.py:54`; `scripts/stag_base_run.py:29–38`).
- Set boost_until=nxt+2 after feedback; subsequent zero-based indices nxt,nxt+1,nxt+2 boost, THREE planning calls total (`planning/stagnation.py:169,201–202`). Do not describe merely 'two replans' based on raw boost_replans=2 config.
- Gate compares two three-point means separated by W_g=3 (`planning/switch.py:62–66`), threshold tau_g=.05, K=1. Needs both windows available and positive stall weight. It remains open permanently after the first qualifying feedback (`71–77`). Prediction-error increase also satisfies q<.05; it is a low-improvement schedule, not 'proof model converged'.
- Memories accumulate before gate, gate suppresses active memory and boost until opening (`planning/switch.py:68–84`). Updates continue in main R/SZ/P arms; freezing only F arms (`scripts/e3_run.py:41–52`).
- It is faithful to write `g_t R_t^stall` if g=1 throughout trigger-only and g=0→1 once for gated. Memory empty before first trigger; alpha also waits on gate. Do not multiply regularizer by current w_t or let g drop when motion resumes: persistent memory would be lost mathematically.

## Training attribution and parameter clarity

Inherited offline objective remains latent observation prediction plus temporal-straightening (`models/visual_world_model.py:339–377`). TS is observed visual displacement curvature only (`270–300`), not a new ECP loss. Main method can call it 'latent-prediction and trajectory-straightening objective' with TS citation; graphics should use 'predictive learning'. Keep explicit TS credit in text/appendix.

Current appendix defines almost all training symbols but should add:
- `d_o=d_v+d_p`;
- `sg` is the stop-gradient operator;
- `hat z_{i+1}^o` is teacher-forced one-step prediction;
- mean includes examples and valid temporal/patch pairs where applicable;
- lambda_TS is inherited regularizer coefficient (0.01 PushObj; 0.001 released Maze config).

Use collection index i rather than t to separate control-call time from MPC replan time. Name bias b_c and noise sigma_c explicitly instead of unexplained constants9 and10. Define identity matrix I_2 and normalized direction with1e-6 zero-direction cutoff. Actual collection primitive is in `scripts/gen_pushobj_dataset.py:127–147`.

## Layout integration notes

`method.tex` is approximately1002 whitespace-delimited words including TeX, and preserves all original section cross-reference labels. It moves the detailed directional action equation to `parameters_appendix.tex`, preserving eq:kick there. No other manuscript uses eq:kick. New labels: sec:parameters, tab:parameters, eq:memoryscale, eq:kernel, eq:timeweights, eq:baseexact, eq:kernelgradient. New method caption has no TS. Replace original properties section to prevent duplicate sec:properties label. Root should compile after insertion; no compile was executed in this bounded audit.
