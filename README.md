# Stalls in latent world-model MPC

## CVPR 2027 manuscript and results

The current draft and reproducible result package are on this branch: [paper PDF](paper/cvpr2027/paper_v14.pdf), [editable source](paper/cvpr2027/README.md), and [evaluation data and logs](reports/publication_v14/README.md). The paper includes the completed Frozen/AdaJEPA evaluation and the random-data epoch-3 DINO-WM CEM PushT baseline. The three-epoch DINO-WM PushObj evaluation is complete: [results, raw logs, and reproducible audit](reports/dinowm_pushobj_v3/completed_results/README.md).


Research code for studying why test-time-adapted latent world-model MPC gets stuck ("stalls") and how a
planner can detect and escape it. Experiments run on PushObj (shape shift), the multi-layout point maze, and the standard Wall baseline.

## What is here

| Path | Content |
|---|---|
| `planning/escape.py` | Observation-only stall detector and escape strategies: random or goal-directed kicks, proprio re-weighting (K2P), restarts, a staged object phase, workspace clip |
| `planning/stagnation.py` | Latent stall regulariser: scale-free latent movement trigger, episode memory of stalled states, goal-relative repulsion added to the planning objective |
| `planning/costtogo.py` | Latent cost-to-go head V(z, z_goal) in model steps |
| `scripts/escape_*.py`, `scripts/stag_*.py`, `scripts/ctg_*.py` | Runners, analyses and diagnostics (simulator oracles, solvability ceiling, cost-to-go checks) |
| `planning/adajepa*.py`, `planning/mpc.py`, `planning/gd.py`, `planning/cem.py` | The baseline planners: MPC with gradient or CEM sub-planners, with and without test-time adaptation |
| `models/`, `datasets/`, `env/`, `train.py`, `plan.py` | World-model training and planning infrastructure |
| `data/targets/` | Evaluation target sets (ordered chunks of plan targets) used by the runners |

## Latest results (2026-11-16)

- [Completed Rope formal evaluation and real simulator images](reports/rope_formal20/README.md): four conditions × 20 targets, maximum 20 pushes, stopping at CD < 0.3. Native 5/5: **0.5809, 14/20**; fixed S 3/3: **0.6112, 13/20**; dynamic S 3/3: **0.5152, 12/20**; fixed S without extra warmup: **0.5834, 13/20**. Dynamic S lowers mean CD but no S condition improves success rate. Includes [complete three-push image sequences](reports/rope_formal20/visuals/README.md) from separately recorded executions, favorable and unfavorable examples, and the full raw evidence package.

- [PushT clean-data audit](reports/pusht_clean_audit_20261116/README.md): LeWM-GD clean is **32/50 (64%)** and already populated. No matching LeWM-CEM or DDP-WM-CEM clean archive was found in the reference repository; those two table entries remain unmeasured.

The [complete Rope follow-up summary and raw evidence](reports/rope_followup_20261116/README.md) collect all ten completed conditions, 100 episode-condition results, per-sample measurements, and the encoder/protocol audits.

- [Completed Rope 15-push rerun](reports/rope_chunk3_cap15_20261116/BUDGET_COMPARISON.md): three groups of ten targets at 3/3 cadence give Native CD **0.4206**, fixed S **0.4117**, and dynamic S **0.4178**, all **7/10** successful. Each S arm triggers ten times and changes three tasks' controls; strengths again saturate, so this does not distinguish fixed from dynamic strength. Truncating the earlier 20-push trajectories at 15 yields **6/10**, versus **8/10** after 20: two successes require the extra budget. The independent rerun and same-trajectory budget comparison are reported separately because physics variation can change later plans.

- [Completed Rope ten-target chunk and strength pilot](reports/rope_chunk3_strength_20261116/RESULTS.md): with observation after every push, a 20-push limit, and stopping at CD < 0.3, Native-3 reaches mean CD **0.3119 (8/10)**, fixed-strength S **0.2430 (8/10)**, dynamic-strength S **0.3109 (8/10)**, and Native-5 **0.6715 (7/10)**. Both S variants change actions in four tasks, but all trigger strengths saturate at the same value; their CD difference does **not** establish a fixed-versus-dynamic advantage. [The encoder audit](reports/rope_encoder_audit_20261116/DIAGNOSIS.md) invalidates the old Rope pilot that accidentally used a random DINO encoder. [Protocol and run instructions](experiments/rope_chunk3/README.md).

- Completed Rope per-push replanning controls, ten targets each: Native CD **1.0046**, [S with three-push warmup](reports/rope_feedback_pretrained_20261116/warmup3/RESULTS.md) **1.0794**, and [S with five-push warmup](reports/rope_feedback_pretrained_20261116/warmup5/RESULTS.md) **1.0070**; all solve **4/10**. Each S arm changes one task's actions without improving its final CD. No trigger occurs during the interval distinguished by the warmup settings; their later CD difference follows independent physics branches and is not evidence of a warmup advantage.

- [Completed ECP threshold and cadence follow-up](reports/ecp_threshold_v1/README.md): 1,400 task runs and complete logs. Raising the threshold to 0.30 gives **77.4%** for augmented AdaJEPA (previous ECP: 76.9%); DINO-WM remains **48.9%** at 6 x 25 and **20.6%** at 30 x 5, with a matched native control at each cadence.

- [Completed AdaJEPA 0.45 threshold ablation](reports/ecp_threshold_v2/RESULTS_ADA045.md): **272/350 (77.7%)** on the same PushObj tasks, versus 271/350 (77.4%) at 0.30. The 350-task paired audit and complete raw logs are included; the planned DINO-WM follow-up was stopped before a full evaluation.

- [Completed AdaJEPA 0.60 and in-house Frozen 0.45 follow-up](reports/ecp_threshold_v3/RESULTS.md): both 350-task runs finished and passed a 700-task paired audit. AdaJEPA 0.60 scores **271/350 (77.4%)**, one task below 0.45; Frozen 0.45 scores **253/350 (72.3%)**, 39 tasks above its matched native planner with no regressions. Full logs and configurations are included.

- [Completed DINO-WM PushObj evaluation](reports/dinowm_pushobj_v3/completed_results/README.md): all 1,400 task runs finished. Augmentation improves frozen CEM from **86/350 (24.6%)** to **171/350 (48.9%)**. ECP at the original threshold triggers in only 2 ordinary-data and 1 augmented-data tasks, with no changed success labels. Full logs, per-shape counts and independently verified action-prefix pairs are included.

- [PushObj training A/B screen and complete logs](reports/effect_b_20261103/results.md): all four 3-epoch runs and 1,400 frozen MPC tasks completed. At 30 replans, B0 **220/350**, B1 **217/350**, B2 **116/350**, B3 **134/350**. None improved over B0; original logs, diagnostic data, configs, hashes and the **52.78 GPU-hour** ledger are archived.
- [Wall public-checkpoint baseline](reports/wall_dinowm_20261103/report.md): open-loop CEM **26/50**, capped MPC **46/50**; paired episode outputs and complete success curves included. These runs do not use adaptation or stall regularisation.
- [Switch calibration](reports/switch_calibration_20261102/results.md): lowering the threshold or widening the window did not improve the existing development results; all 387 new episode-arm runs are complete and pairing is verified.
- [Current research narrative](reports/paper2_positioning/narrative.md): the stall regulariser is the core intervention; the gate controls its timing. [The proposed Wall method experiment](reports/paper2_positioning/wall_stall_proposal.md) is pending user confirmation and has not started.

## Setup

```bash
conda env create -f environment.yaml && conda activate ts
```

- Checkpoints live in `checkpoints/<name>/` (`hydra.yaml` + `checkpoints/model_<epoch>.pth`); `plan.py` rebuilds the
  model and environment from that `hydra.yaml`.
- The maze environments need MuJoCo 2.1 (`mujoco-py`, `d4rl`); add `~/.mujoco/mujoco210/bin` and `/usr/lib/nvidia`
  to `LD_LIBRARY_PATH`.
- Runs write to `eval_outputs/` (ignored by git).

Reproducing the paper-2 experiments on another server (assets, checksums, commands in order): [REPRODUCE.md](REPRODUCE.md).

## Typical commands

```bash
# stall regulariser on PushObj: latent traces of a reference run, episode selection, arms, analysis
python scripts/stag_latent_trace.py --ref eval_outputs/escape_EVAL50/none --stream EVAL50 --out eval_outputs/stag_EVAL50/trace.jsonl
python scripts/stag_select.py
python scripts/stag_run.py --arms SZ_lo SZ_mid SZ_hi --gpus 4,5,6,7
python scripts/stag_analyze.py

# simulator oracles and the solvability ceiling of stall states (CPU)
python scripts/stag_oracle.py --variant O2 --workers 16 --out eval_outputs/stag_EVAL50/oracle/O2.jsonl
python scripts/stag_ceiling.py --workers 48 --out eval_outputs/stag_EVAL50/ceiling.jsonl

# cost-to-go head: encode, train, check
python scripts/ctg_encode.py --data data/pushobj_bias_16k/train --out eval_outputs/ctg/bias16k_train
python scripts/ctg_train.py --objective td --out eval_outputs/ctg/V_td.pt
python scripts/ctg_check.py --models td=eval_outputs/ctg/V_td.pt
```

## License

MIT, see [LICENSE](LICENSE). Parts of the world-model and planning infrastructure come from third-party MIT-licensed
code; their notices are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
