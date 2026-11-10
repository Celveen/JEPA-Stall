# First-page headline chart design audit

## Figure type and design

This is an experimental-results headline figure. A grouped bar chart directly compares three stages under each of the two update settings, Frozen and AdaJEPA. The groups represent update settings of the same model family; they do not represent distinct world-model architectures.

The 3.25 × 2.25 inch canvas is sized for one CVPR column. Each group contains Released, Our data, and Our data + ECP in the same left-to-right order as the legend. Solid blue-gray, teal, and warm orange fills match the paper palette. Data labels give each rate to one decimal place. A bracket above each group reports the total improvement from the exact success counts, avoiding arithmetic on rounded labels. A 0–100% vertical axis and light horizontal grid provide a common scale.

Matplotlib is appropriate for this small reproducible statistical chart. A line chart would imply a continuous sequence, and a third group containing external methods would require matched results that have not been established.

## Caption for manuscript integration

> Interaction-rich training and execution-conditioned planning improve both frozen and adaptive control. On the same 350 PushObj tasks with at most 30 MPC replans, our training data and ECP raise success from 43.7% to 72.3% for Frozen and from 69.4% to 76.9% for AdaJEPA. ECP denotes execution-conditioned planning with stagnation memory and proprioceptive reweighting; brackets show gains over the locally reevaluated released checkpoint.

Root should adapt the first sentence to match the manuscript terminology. This caption supports improvements in both update settings, without claiming deployment on different architectures or state-of-the-art performance.

## Verification

- **Vector export:** PDF and SVG are true vector output. PDF is 234 × 162 points. PNG is a preview only.
- **Fonts:** Embedded STIX General regular/bold TrueType subsets verified with `pdffonts`. Smallest type is the 8.2 pt legend at native column size; axes, values, and gains are 8.5–9 pt. Do not scale below the intended 3.25 inch width.
- **Color and ordering:** Three solid fills, always in the legend order and a stable left-to-right series order. Numeric labels and categorical position retain interpretation in grayscale. No hue-only interpretation of the values is required.
- **Axis integrity:** Linear 0–100% success scale, fixed quarter-scale ticks, equal bracket positions, no axis break.
- **Numerical integrity:** Rates recomputed from integer counts and the denominator of 350. Gains are 100/350 = 28.5714 pp and 26/350 = 7.4286 pp. The adaptive final bar is 269/350, i.e. full S+P, rather than the P-only result of 270/350.
- **Uncertainty:** No synthetic error bars; the chart reports observed task-list aggregates rather than independent-run means.
- **Visual inspection:** PNG reviewed at its designed aspect ratio. Labels are separated from bars and annotations; all elements are within the canvas. No hatches, gradients, shadows, or 3D effects.
- **Scope integrity:** No external method scores or SOTA claims are included. The underlying source is the verified v4 result audit/count manifest.

Severity summary: 0 critical, 0 major, 0 minor issues in the standalone figure. The integrated first-page placement and caption should be checked after LaTeX compilation.

## Rebuild

```sh
python build_headline.py --data headline_data.json --out ./output
```

The script writes `fig0_headline.pdf`, `fig0_headline.svg`, `fig0_headline.png`, and `headline_figure_manifest.json`; the manifest contains all computed percentages and file hashes.
