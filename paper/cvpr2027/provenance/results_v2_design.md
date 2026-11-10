# Result figures, visual revision v2

Two figures have been redrawn from the archived counts and logged physical distances only. No experiment or policy execution was run. The script depends only on Python, NumPy, Matplotlib, and the bundled `plot_data.json`.

```bash
python build_results_v2.py --data plot_data.json --out figures
```

Outputs: `fig4_results.{pdf,svg,png}`, `figS_pushobj_distance.{pdf,svg,png}`, `palette.json`, and `figure_manifest.json`. Grayscale previews are also supplied for inspection. PDF/SVG are the publication versions; PNG is a preview. PDF fonts are embedded TrueType, and SVG text remains editable.

The design uses solid mist-blue, teal, and coral fills, drawing on the restrained palette and white space in the supplied reference and the LeWM/TS figures. The composition is original. All bars use opaque color; no outlined/hollow series remains. Fixed left-to-right order and explicit category labels provide a second encoding beyond color. The time series adds distinct dash styles and square/circle markers.

The results canvas remains6.875×2.75in; the supporting trace is3.25×1.90in. Minimum text size is8pt at those insertion widths. Success axes remain0–100%, and no uncertainty bars are invented. The PushObj paired frozen gain is annotated as+11.1percentage points. Maze is labeled as the separate gated-memory-only variant and retains its true63→66comparison on the same scale.

The distance plot now includes the full starting error of290.43px on a0–300pxaxis. The old0–260pxaxis clipped this true initial value. Curves retain every archived sample and end at their actual endpoints: baseline replan30, correction replan17. The20pxline is a position criterion; the task's separate angle condition is not encoded by that line.

Suggested results caption:

> Temporary planning correction improves frozen PushObj success by11.1percentage points on the contact-biased checkpoint. All PushObj groups use350paired tasks and30replans, with the same left-to-right ordering of native MPC, temporary proprioceptive weighting, and weighting plus stagnation memory. The separate Maze panel uses100tasks and gated memory alone (no proprioceptive boost). Values come from the existing runs; no error bars are inferred.

Visual audit: vector output, embedded fonts, final-size text>=8pt, full solid fills, grayscale-readable ordering/markers, unchanged data, honest axes, and no decorative shadows/gradients. Zero critical or major issues remain after viewing color and grayscale previews.
