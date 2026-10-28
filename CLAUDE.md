# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Research code on stalls of test-time-adapted latent world-model MPC: the planner hovers in place and the scene stops
changing short of the goal. It holds the stall detector and escape strategies (`planning/escape.py`), the latent stall
regulariser (`planning/stagnation.py`), the latent cost-to-go head (`planning/costtogo.py`), and the runners,
analyses and simulator diagnostics under `scripts/`. The baseline planners (MPC with GD or CEM, with and without
test-time adaptation) and the world-model code are the infrastructure these build on.

There is no test suite. Verification means a short planning run checked against a reference run: the stall planners
are bit-identical to the baseline until their first trigger, so a smoke test compares per-replan actions.

## Setup

```bash
conda env create -f environment.yaml && conda activate ts
```

- The env `ts` is Python 3.9. `load_ckpt` warms the torch.hub DINOv2 cache inside a try/except; checkpoints with
  SmallResNet encoders (all used here) load regardless.
- Maze envs need MuJoCo 2.1: `LD_LIBRARY_PATH` must include `~/.mujoco/mujoco210/bin` and `/usr/lib/nvidia`
  (`scripts/stag_maze_run.py` adds both). PushObj needs neither.
- `checkpoints/`, `eval_outputs/`, `reports/` and most of `data/` are not tracked; `data/targets/` (evaluation target
  chunks) is. They are real local folders, not links into another checkout: this repo must not depend on the CL-JEPA
  (paper 1) working tree, and nothing here writes into it.

## Architecture

### Two-config layering

`plan.py` reads the plan config (`conf/*plan*.yaml`: planner, adaptation, goals, seeds) and `<model_path>/hydra.yaml`
(the training config inside the checkpoint: architecture, `frameskip`, `num_hist`, `concat_dim`, env). Model and env
are rebuilt from the latter. If `ckpt_base_path` starts with `/` it is the model directory itself (`model_name` is
ignored). Checkpoints pickle whole `nn.Module` objects, so model classes must stay importable and unchanged.

### Planner stack

```
BasePlanner (planning/base_planner.py)       freezes the WM, owns objective_fn and log dumping
  └─ MPCPlanner (planning/mpc.py)            sub-plan → execute n_taken_actions → re-plan
       └─ AdaJEPAMPCPlanner (planning/adajepa_mpc.py)   per-sample planning with a test-time update each replan
            └─ ProbeMPCPlanner (planning/adajepa_probe.py)   per-replan logs and update probes (mode grad = baseline)
                 └─ EscapeMPCPlanner (planning/escape.py)    stall detection, escapes, per-episode seeding
                      └─ StagnationMPCPlanner (planning/stagnation.py)   latent stall regulariser, proprio boost, only_samples
                           └─ LookaheadMPCPlanner (planning/lookahead.py)   macro moves + plan vs the usual plan
  sub_planner: GDPlanner (planning/gd.py) | CEMPlanner (planning/cem.py)
```

- `probe.mode=none` turns the adaptive planner into the frozen per-sample planner (no update), which is the base
  the regulariser arms pair with when the base is "frozen" (`scripts/stag_base_run.py`, `scripts/stag_maze_base_run.py`).
- Adaptation and stall bookkeeping hang off `_post_env_feedback`; objective changes wrap `sub_planner.objective_fn`;
  plan-time changes wrap `sub_planner.plan` (`_sub_plan`).
- `EscapeMPCPlanner._plan_single` seeds the global torch RNG per episode (`seed * 1000003 + episode seed`; dropout
  is active in the update step). That is what makes arms pair exactly with `escape.mode=none` / `stag.kappa=0`.
- Episode seeds are `seed * i + 1` by position `i` in a run, so `stag.only_samples` skips samples without moving them.

### Units

Config values for `goal_H`, `n_taken_actions` and sub-planner `horizon` are env steps; planners work in model steps
(`/ frameskip`). The cost-to-go head counts model steps. The stall regulariser's trigger, radius and strength are
scale-free (latent movement relative to the latent distance to the goal), so the same settings apply to every env.

### Goal sources

| `goal_source` | targets | needs |
|---|---|---|
| `segments` | plan targets replayed in the env (PushObj chunks in `data/targets/<set>/`) | `eval_data_path` = a chunk file |
| `maze_dist` | BFS-distance-controlled start/goal on held-out maze layouts | `eval_data_path` with `val/` |

### Env layer

Planning uses the vectorized batch API (`prepare`, `rollout`, `eval_state`, `update_env`) of `SubprocVectorEnv` /
`SerialVectorEnv`. Success: PushObj joint pusher+object position error < 20 px and angle < 20 deg; maze position < 0.5.
The PushObj pusher is a kinematic body (passes through walls); physics is fully damped (`space.damping = 0`).

### Outputs

Each run writes `logs.json`, `per_iter_logs.json` (per replan: state, success, actions, stall fields),
`escape_episodes.jsonl` / `stag_episodes.jsonl`, `final_eval_results.npz` and `plan_targets.pkl` to its
`hydra.run.dir`, which the runners set explicitly.

## Conventions

- `env/wall/data/` is a source package, not a dataset dir; `.gitignore` keeps it.
- `custom_resolvers.py` registers OmegaConf resolvers used in `hydra.run.dir` templates; import it for its side effect.
- Multiprocess diagnostics set OMP/MKL/OpenBLAS/cv2 threads to 1 and pin one GPU per worker.
