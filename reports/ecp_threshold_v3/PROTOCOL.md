# PushObj threshold follow-up: AdaJEPA 0.60 and in-house Frozen 0.45

Authorized 13 November 2026. Both conditions use our three-epoch action-biased PushObj checkpoint, `checkpoints/pushobj_bias_16k_e3_seed0/checkpoints/model_3.pth`. The Frozen condition sets `planner.probe.mode=none` and verifies that its model parameters and buffers remain unchanged. DINO-WM is outside this evaluation.

Both conditions use the same 350 fixed EVAL50 PushObj tasks (50 per shape), GD with 100 optimization iterations, a maximum of 30 MPC replans and five environment controls per replan. Each has at most 150 controls. The detector uses a three-replan window, starts after 50 controls, and uses the same latent movement ratio. As in the earlier threshold sweep, the lower threshold is one third of the upper: AdaJEPA 0.20/0.60; Frozen 0.15/0.45. Penalty coefficient 2, radius coefficient 0.25, proprioceptive multiplier 10, and two follow-up replans stay fixed.

Reference comparisons use the corresponding 40-replan runs of the same checkpoint and fixed task set: `eval_outputs/stag_adajepa_bias16k_it40/ref` for AdaJEPA and `eval_outputs/stag_frozen_bias16k_it40/ref` for Frozen. Native success is recalculated at the 30-replan cutoff. Every chunk must match reference starts and goals, and actions must match before the first ECP intervention. The run records checkpoint and source hashes, complete configuration, per-task actions and states, triggers, and paired rescues/regressions. The previously completed AdaJEPA 0.45 result, 272/350, is the direct threshold comparator for AdaJEPA 0.60.

These are exploratory comparisons on the previously inspected fixed task set. Results will be reported as paired task counts and per-shape outcomes after all 350 tasks per condition finish.
