# PushObj augmented AdaJEPA ECP threshold 0.45 ablation

The user narrowed the follow-up on 13 November 2026: stop new DINO-WM evaluation and evaluate only the threshold change on the augmented AdaJEPA checkpoint. The DINO-WM six-by-25 first chunk completed before the cancellation; it and any interrupted outputs are retained separately and excluded from this 350-task result.

The checkpoint is `checkpoints/pushobj_bias_16k_e3_seed0/checkpoints/model_3.pth`, trained for three epochs with action-biased data. The planner remains GD with 100 optimizer iterations and a 30-replan cap, executing five environment controls per replan (150 controls maximum). The ECP movement detector uses a three-replan / 15-control observation window and starts after 50 controls. Its lower/upper thresholds become 0.15/0.45. Penalty coefficient 2, radius coefficient 0.25 and proprioceptive multiplier 10 remain unchanged.

All 350 fixed EVAL50 PushObj tasks are evaluated: 50 for each of seven shapes. The comparison uses the same checkpoint and targets, against the existing native 30-by-5 score 262/350, ECP 0.15 score 269/350 and ECP 0.30 score 271/350. The baseline's source run had a 40-replan cap, so native success labels are recomputed from its per-iteration logs at the 30-replan cutoff. Before publishing the new score, verify identical starts/goals and identical action prefixes through the first intervention, plus the run's checkpoint and source hashes.

This is an exploratory threshold ablation on a fixed task set previously inspected; report raw success counts, per-shape results, triggers, paired rescues/regressions and logs without treating this as a fresh held-out selection set.
