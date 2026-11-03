# Stalls in latent world-model MPC

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

## Latest results (2026-11-03)

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
