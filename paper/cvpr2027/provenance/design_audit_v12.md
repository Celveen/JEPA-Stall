# Figure design audit — v12

## 1. Figure type
Figure 2: motivated failure examples with recovery. Figure 4: diverse qualitative outcomes on additional tasks. The supplement supplies complete task context.

## 2. Paradigm recommendation
Use matched native/ECP views plus an isolated, enlarged native stall trace. The scene provides physical context, while the local plot reveals repeated motion that is invisible at scene scale. Figure 4 uses different cases to add coverage.

## 3. Layout sketch
Figure 2: 6.875 × 3.38 inches, two rows, four matched cases. Each case contains native failure, local motion and ECP success; there are eight simulator images and four coordinate plots. Figure 4: 6.875 × 2.55 inches, two rows of four paired cases. Local plots retain equal horizontal/vertical scales; scene pairs have identical square crops.

## 4. Labelling and annotations
Native orange, ECP teal; labels explicitly state failure/success. Solid trajectory lines, a dot every five control calls, short arrows aligned with actual integration segments. Orange dashed boxes identify local motion; no trajectory itself is dashed. Local plots show their replan interval and displacement units. Figure fonts are at least 8 pt at final insertion size. All Maze walls are visible, with equal surrounding margin.

## 5. Tool suggestion
Matplotlib PDF/SVG vector annotations over native Pymunk/MuJoCo raster renderings. The camera is centered on full Maze geometry for visualization. No image generation or geometric scene editing is used.

## 6. Universal rule audit
- Vector paths, arrows and text: pass.
- At least 8 pt text at final size: pass.
- Compact two-row main figures: pass.
- Outcome labels and symbols supplement color: pass.
- Captions define local coordinates, units, timing and outcome symbols: pass.
- Physical and axis geometry: pass; no stretch, rotation, smoothing or fabricated paths.
- No missing outer walls: pass; checked both native render bounds and final crops.
- Text collision and clipping: checked programmatically and visually on image previews and PDF pages 2/7.

## 7. Integrity gate result
All applicable gates pass. Running examples agree with the introduction and qualitative section. Two additional Maze pairs reuse recorded full-precision controls; their terminal success and every logged replan state agree. Four Maze cases have no dense points in wall interiors. Initial/goal state identities are disjoint between Figures 2 and 4. The selected failure intervals are labelled and are not claimed to span the entire episode.

## 8. Severity summary
No unresolved critical or major issue found. Corrections completed: geometry-aware centered framing, genuine local trajectory enlargements, and distinct tasks across the two main galleries. Source hashes and numerical local-motion statistics are recorded in the accompanying manifests.
