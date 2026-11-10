# Method figure, revision 2

The figure follows the attached SVG's visual language: navy typography, pale blue model modules, pale green feedback modules, actual simulator thumbnails, an encoder glyph, latent tiles, a predictor network icon, and miniature candidate trajectories. The arrangement and all vector geometry are original.

Deliverables: `fig2_method.pdf`, `fig2_method.svg`, `fig2_method.png`, `build_method.py`, and `method_manifest.json`.

- Canvas: 6.875 × 3.0 inches; insert at full text width.
- Fonts: STIX, 8–13 pt; the PDF embeds TrueType subsets.
- Training: contact-biased collection → unchanged JEPA+TS → learned encoder/predictor.
- Execution: current/goal observation → encoder → latent predictor/MPC → executed prefix → environment.
- Feedback: executed transitions → net latent displacement / remaining latent goal distance → temporary proprioceptive weighting and current-episode local stall memory.
- Optional paths: a separate action offset branch, and within-episode model updates that can affect the encoder and predictor.
- No cross-episode persistent memory, maturity gate, or new successful training loss is depicted.
- Thumbnail sources are existing frames under `paper/cvpr2027/figures/frames/`. No new simulator rollout, planner, or training job was run. Small trajectory, transition-token, contact, and weight graphics are explanatory schematics.
- The normalized-stall denominator was checked against `planning/stagnation.py` and the method equation: remaining goal distance, not historical path length.

Reproduce with the repository's `ts` Python environment:

```bash
MPLCONFIGDIR=/tmp/jepa_mpl python build_method.py --frames PATH_TO_PAPER/figures/frames --out OUTPUT_DIRECTORY
```

The script checks pairwise text bounding boxes and records candidates in the manifest. The final output has no overlapping text boxes. PNG and PDF were rendered and visually inspected for labels, arrow routing, and cropping.
