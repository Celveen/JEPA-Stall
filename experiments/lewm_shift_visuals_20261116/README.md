# Frozen LeWM: real red-goal / red-agent paired simulations

Before running, fix the first 10 entries of the existing 50-task `targets_T.pkl`
in file order for both conditions. Seed 100; each native/ours pair shares the
checkpoint, target construction, simulator seeds, GD100 planner, 25-control
horizon, 5-control execution block, and 30-replan (150-control) budget.

Native disables all intervention. Ours reuses the unchanged archived
`lewm_ecp_045_contact_objective` recipe: latent stagnation memory, conditional
analytic pusher objective, and contact-objective optimization (weight 10,
four-replan bursts, at most seven guided replans). The contact component uses
observed simulator object pose and known T geometry. This is not a visual-only
intervention. No post-optimization additive action offset is applied.

These are qualitative GD comparisons, not replacements for the formal clean
CEM50 result. Model names in figures are LeWM; the exact checkpoint and training
epoch remain in the machine-readable reproducibility configuration.

Store every episode outcome and unmodified RGB, states, and actions returned by
the final evaluated simulator execution under the archived resim-from-start
protocol. Crop successful trajectories at the first successful MPC checkpoint;
later padded actions are not part of the presented execution. Both arms use the
native PushT success rule: joint pusher/object xy error <20 px AND wrapped
object angle error <20 degrees. The Rope Chamfer criterion does not apply.

Select the first target in file order where Ours succeeds and Native fails for
each condition. Mark these as selected rescue examples and publish all 10 paired
outcomes per condition, including any regressions. Do not tune parameters to
obtain a particular displayed outcome.

Example invocation from repository root (one arm on one free GPU):

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  eval_outputs/rope_local/venv/bin/python experiments/lewm_shift_visuals_20261116/run.py \
  --condition red_goal --arm native --output eval_outputs/lewm_shift_visuals_20261116/red_goal_native
```

Run the remaining condition/arm combinations on separate free GPUs. For a
pipeline-only check, add `--smoke` and use a different output directory.

After all four runs finish, audit and export the recorded images with:

```bash
MPLCONFIGDIR=/tmp/lewm-pusht-matplotlib eval_outputs/rope_local/venv/bin/python \
  experiments/lewm_shift_visuals_20261116/report.py
```

The report is written to `reports/lewm_shift_visuals_20261116/`. Its CSV contains
every paired outcome, not just the two selected figures. Each selected example
also includes every original RGB frame, a playback GIF, an every-three-control
contact sheet, and measured state-error curves. An archive retains all 40
episodes, including failed runs.
