# Genuine simulator trajectories for the manuscript

Figure `fig1_trajectory_cases.pdf` is 6.875 × 3.8 inches (full text width), with fonts of 8 pt or larger. It uses native Pygame and MuJoCo renderings at saved physical states, with vector trajectory and annotation overlays. No training, model inference, planner invocation, simulator stepping, or new action rollout was performed. Companion SVG and PNG provide editable output and visual preview.

The two rows use the same paired episodes as the earlier manuscript screenshots. They now show all logged MPC endpoint positions, with highlighted later intervals and separate exact-coordinate zooms. The PushObj baseline oscillates in a 4.36 px diameter region during replans 15–30; the Maze baseline repeatedly returns to the same corridor endpoint during replans 7–30.

## Exact conditions

- PushObj: `stag_frozen_bias16k_it40/ref/chunk_04_I`, sample 5, versus `T0SP/chunk_04_I`, sample 5. The correction is **S+P**, with kappa=2 and proprio boost=10. This is not a P-only attribution. Frozen model; first 15 recorded states and action arrays exactly match. Baseline fails at replan 30; corrected succeeds at replan 17. Full original run budget was 40; the figure crops the recorded run at the paper's 30-replan budget.
- Maze: `stag_maze100_ar4/medium/adajepa/ref`, sample 75, versus `e3/medium_s100/SZ-switch`, sample 75. The correction is **gated S only**, with kappa=2 and proprio boost=1. Online adaptation continues in both arms. First 6 recorded states and action arrays exactly match. Baseline fails at replan 30; corrected succeeds at 27. Full original baseline budget was 50; the figure stops at 30.
- Original initial and goal arrays are reloaded and asserted equal across arms. The drawing source does not infer intervention identity from success or `n_escapes`; the original arm definitions and metadata are recorded.

## Sampling and quantitative interpretation

The logs store one physical state at the end of each MPC replan. Dashed connectors link those sampled positions; they are not a reconstruction of the physical path within an executed action prefix. Correspondingly, inset `Path` is the length of the sampled endpoint polyline. `Net` is displacement between the first and last points of the shown interval. Maze connectors can span walls between logged samples; this is why the endpoint representation and dashed connectors are explicitly identified in the figure/caption.

| Task / arm | Interval | Sampled polyline length | Net displacement | Units |
|---|---:|---:|---:|---|
| PushObj baseline | 15–30 | 19.13536 | 0.53468 | px |
| PushObj S+P | 15–17 | 36.11309 | 36.09621 | px |
| Maze baseline | 7–30 | 18.23126 | 0.000283 | simulator units |
| Maze gated S | 7–27 | 16.07752 | 2.60759 | simulator units |

Insets use equal axis scales and their own clearly labeled physical coordinate ranges. PushObj background crops are identical between arms; Maze background crops are identical between arms. Goal stars identify the pusher/agent target. The gray object is the native final simulated object, not a schematic.

## Reproduction

Use the existing `ts` Python environment (NumPy, Matplotlib, Pymunk/Pygame, Gym, mujoco_py). For CPU rendering, the previously prepared isolated package `/tmp/jepa_paper_figures/mujoco_py` contains the OSMesa extension. With its parent on PYTHONPATH, run:

```bash
MUJOCO_PY_FORCE_CPU=1 \
PYTHONPATH=/tmp/jepa_paper_figures \
LD_LIBRARY_PATH=/home/appuser/.mujoco/mujoco210/bin:/usr/lib/x86_64-linux-gnu \
/home/appuser/anaconda3/envs/ts/bin/python build_trajectory.py \
  --repo /home/appuser/workspace/project/multimodal-test/JEPA-Stall \
  --out /tmp/jepa_revision_v3/trajectory
```

On another machine, prepare a local CPU mujoco_py 2.1 package and MuJoCo 2.1.0 installation, then adjust those environment paths. The code keeps simulator implementations and parameters unchanged. Source paths, SHA-256 hashes, all plotted positions, camera transform, pixel-registration checks, and output hashes are in `trajectory_manifest.json`.

## Design audit

- Motivated-example paradigm: matched baseline failure and correction, with a physical behavior visible in each row.
- Native raster simulator scenes + vector paths/text in PDF/SVG. Raster screenshots are required evidence, rather than rasterized vector diagrams.
- Minimum 8 pt font at intended full-width insertion; embedded STIX TrueType fonts.
- Baseline coral, correction teal; markers and dashed sample connectors support grayscale interpretation.
- Honest inset scales, with units and origin meaning explicitly stated. All endpoint data retained through the labeled budget.
- State rendering and endpoint sampling are clearly distinguished from newly executed video or substep path capture.
- No decorative gradients, artificial trajectory smoothing, or invented oscillations.
