# Trajectory figures — v8

The figures use native simulator images at recorded physical states. All path points, initial states, goals, outcome labels, and pair identities come from existing logs. No training, evaluation, action rollout, physics integration, or server job was launched for these figures.

## Deliverables

- `figures/fig1_trajectory_cases.pdf`, 6.875 × 3.75 in: four paired PushObj/Maze panels for Figure 2. Exact late-window endpoint plots retain physical units and report net displacement. Text explicitly distinguishes Frozen versus Ours (frozen) on PushObj and AdaJEPA versus Ours (AdaJEPA) on Maze.
- `figures/fig5_execution_recovery.pdf`, 6.875 × 2.70 in: two PushObj object shapes, each with initial state, Frozen FAILURE, and Ours SUCCESS. All terminal images carry full recorded endpoint paths.
- `figures/figS_pushobj_timeline_v8.pdf`, 6.875 × 2.65 in: optional supplementary two-lane view of the shared PushObj execution through replan 15 before different outcomes.
- Corresponding SVG and PNG assets are in `figures/svg` and `figures/previews`.
- `provenance/trajectory_v8_manifest.json` records source hashes, every endpoint, outcomes, labels, camera projection, frame hashes, and output hashes.
- `provenance/trajectory_extra_frames_v8.json` records the additional T-block frame provenance and the native renderer's exact source hash.

## Case identities and outcomes

| Figure / row | Baseline and corrected log roots | Sample | Identical state/action prefix | Baseline | Ours |
|---|---|---:|---:|---|---|
| Figure 2 PushObj / Figure 5 I-block | `stag_frozen_bias16k_it40/{ref,T0SP}/chunk_04_I` | 5 | 15 replans | Frozen, failure at 30 | Frozen + S+P, success at 17 |
| Figure 2 Maze | `stag_maze100_ar4/medium/adajepa/ref`; `e3/medium_s100/SZ-switch` | 75 | 6 replans | AdaJEPA, failure at 30 | AdaJEPA + gated S, success at 27 |
| Figure 5 T-block | `stag_frozen_bias16k_it40/{ref,T0SP}/chunk_24_small_tee` | 9 | 11 replans | Frozen, failure at 30 | Frozen + S+P, success at 14 |

The T-block is the `small_tee` PushObj shape. It is not a PushT benchmark run. Both PushObj pairs use the same data-augmented model in the frozen regime with and without ECP. Target arrays are exactly equal across each pair. Episode identities are provenance information, not intended main-text exposition.

The plotted pusher-goal error in the I-block case falls from 45.35 px for Frozen to 10.24 px for Ours. In the T-block case it falls from 49.01 px to 4.76 px. These are pusher errors, not full task success thresholds. Success/failure labels are read from the original task outcomes.

Earlier frozen Maze rescue cases were found under `stag_maze100/medium/frozen/{ref,T0SP}`. Those logs precede the corrected action-repeat-4 setup and are not used here. The matched main-result Maze example remains accurately labeled as online-adapted.

## Suggested captions

Figure 2:

> **Execution-conditioned planning escapes local motion loops in manipulation and navigation.** PushObj compares frozen MPC with ECP (S+P), while Maze compares AdaJEPA with gated regularization (S). Native simulator images show recorded states; dashed connectors and arrows indicate the sequence of logged MPC endpoints, with the late window highlighted. Insets report displacement within that window, and stars mark the pusher or agent goal. ECP reaches success at replans 17 and 27, respectively; both baselines fail within the 30-replan budget.

Figure 5:

> **ECP completes stalled PushObj tasks with the model frozen.** Each row compares the same initial/goal pair and model, with native MPC failing at replan 30 and ECP (S+P) succeeding at replan 17 for the I-block and 14 for the T-block. The runs share identical states and actions through replans 15 and 11, respectively. Images are native simulator renders at recorded states; dashed paths connect logged replan endpoints, and stars identify pusher goals.

Shorter alternative for Figure 5:

> **ECP converts frozen-model failures into successful PushObj executions.** The same model and initial/goal pair are used in each row. Native MPC fails at replan 30; ECP (S+P) succeeds at replans 17 and 14. Native simulator frames carry logged MPC endpoint paths; stars denote pusher goals.

## Rendering and integrity details

Every logged endpoint remains at its original coordinates. Round line caps and joins, a restrained white/dark outline, smaller endpoint markers, and a few directional arrowheads improve legibility. No spline, interpolation through additional points, invented detour around maze walls, changed native image colors, or trajectory smoothing is used. Connectors are explicitly dashed: a long connection can span a wall because intermediate physics states were not logged, and it must not be interpreted as a reconstructed continuous physical path.

The original PushObj and Maze final images are reused. The added `small_tee` snapshots use `scripts/build_trajectory_frames_v8.py`, which imports the native Pygame renderer directly, constructs the scene with `_setup()`, assigns each saved pose exactly, and renders. It does not call `reset()`, because the native reset routine includes a physics step. Both `env.step` and `space.step` are replaced with error-raising guards before rendering. Therefore no saved state was advanced or modified by simulation. Rendering needs the existing `ts` environment; figure annotation alone uses standard NumPy/Matplotlib/Pillow.

```bash
/home/appuser/anaconda3/envs/ts/bin/python scripts/build_trajectory_frames_v8.py \
  --repo /path/to/JEPA-Stall --paper /path/to/paper/cvpr2027
python scripts/build_trajectory_v8.py \
  --repo /path/to/JEPA-Stall --paper /path/to/paper/cvpr2027
```

## Figure-designer audit

- Figure types: paired motivated example (Figure 2) and qualitative experimental result (Figure 5).
- Layout: paired baseline/correction scenes, consistent crops within each case, local coordinate insets only in Figure 2; two object shapes in Figure 5.
- Fonts: embedded STIX TrueType, 8 pt minimum at full 6.875 in insertion width. `pdffonts` verified embedded fonts.
- Format: vector PDF/SVG paths, arrows, markers, and text; native raster images are retained as required visual simulation evidence.
- Palette: coral failure, teal success, neutral earlier path, gold goal. Outcome text and X/diamond markers provide non-color encoding.
- Axes: equal physical scale inside each inset; explicit px or simulator units; numerical ranges differ transparently to show true local displacement.
- Self-contained captions: above; no unexplained other-method names in artwork.
- Image inspection: both finished figures checked at intended composition; no clipped labels, overlapping badges, or unreadable tick labels.
- Integrity: initial/goal pairing, state/action prefix, original success label, full endpoint sequence, source hashes, and rendered output hashes checked.
- Severity: zero critical or major issues. Endpoint sampling (rather than substep trajectory logging) remains explicitly stated in the caption and provenance.
