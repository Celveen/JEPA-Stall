# Figure design audit — v11

## 1. Figure type

Figure 2 is a motivated example of execution-conditioned recovery. Figure 4 is qualitative experimental evidence across object geometry and update policy. The supplementary gallery documents the corresponding initial and goal states.

## 2. Paradigm recommendation

Use matched failure/success simulator views with actual execution paths. Each pair fixes task, initial model, and update rule. This supports the claim about conditional recovery directly. Separate displacement insets were removed because they duplicated path information while reducing the scene area. The quantitative multi-method comparison remains a table.

## 3. Layout sketch

Figure 2: 6.875 × 3.25 inches, 2 rows × 4 views. Frozen is the upper row; AdaJEPA is the lower row. Each row pairs native/ECP for PushObj, then native/ECP for Maze. Figure 4: 6.875 × 2.52 inches, 2 rows × 4 matched pairs. Native is left, ECP right. The appendix uses start, goal, native termination and ECP termination views.

All overlays are solid polylines through saved-action simulator replay states. Dots mark every five control calls; arrows are short and aligned with the local path. Identical crops and camera settings are used within a pair.

## 4. Labelling and annotations

Native failure: burnt orange (#B95B38); ECP success: teal (#087D83), with lighter variants on dark Maze backgrounds. Labels state outcomes explicitly. Circle = start, star = agent goal, cross = failure, diamond = success. These shapes and labels retain meaning in grayscale. Main figure text ranges from 8.2 to 9.5 pt at final width. Counts denote terminating MPC replans, as defined in captions.

## 5. Tool suggestion

Implemented with Matplotlib PDF/SVG paths and typography over original Pymunk/MuJoCo raster views. Native simulator images are required by the user and remain raster; trajectory lines, arrows and annotations are vector. No generated or hand-drawn simulation state replaces observed evidence.

## 6. Universal rule audit

- Vector overlays/labels: pass. PDF and editable SVG exported.
- Minimum post-scaling font: pass (8.2 pt in the revised main figures).
- Compact canvas: pass; Figure 2 height reduced, Figure 4 restricted to two rows.
- Encoding without color: pass; explicit outcome text and different endpoint symbols.
- Caption: pass; finding first, pairing, path sampling and replan count defined.
- Geometry: pass; physical path is neither smoothed nor invented; paired crop is identical.
- Chartjunk: pass; no decorative dividers, shadows or extra insets.
- Label overlap: automatically checked and visually inspected in full-size and page-size renders. Supplementary labels repositioned above scenes.

## 7. Integrity gate result

All seven applicable checks pass. The introduction and qualitative discussion were read alongside the figures, verifying the same I-shape and Maze running examples. Ten paired cases were replayed; every recorded endpoint and terminal success condition agrees with its source, within log rounding precision. All checked Maze paths avoid wall interiors. Source file hashes, dense states and output hashes are stored in `replayed_paths_v11.json` and `compact_figures_v11.json`.

The gallery shows selected recoveries, explicitly labelled as such; complete-task success rates are reported separately. Main PushObj method columns keep different task lists separate. No unmeasured result is inferred from a figure or partial evaluation.

## 8. Severity summary

No unresolved critical or major design defect was found in the inspected figures. The three principal corrections are complete: compact layout, genuine dense paths instead of sparse chords, and legible outcome/sampling annotations.
