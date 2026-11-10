# Manuscript v5 final audit

Prepared 2026-11-06. The requested headline figure uses the six already verified PushObj outcomes, with Frozen and AdaJEPA as deployment regimes.

- First-page Figure 1 occupies the right column; the abstract remains in the left column. Placement is verified using PDF word coordinates and independent visual review.
- Figure 1 is a 3.25 × 2.25-inch vector figure with embedded STIX fonts (minimum 8.2 pt), solid fills, numerical labels and a zero-based 0–100% axis. Total gains are computed from counts before rounding: +28.6 and +7.4 pp.
- Figure 2 retains the recorded PushObj and Maze trajectory renderings on page 2. The method diagram is now Figure 3; it contains no external method names. The later redundant overview bar chart is no longer placed in the manuscript; full component tables remain.
- Cross-model sources were independently audited. Published DINO-WM/LeWM PushObj figures use different splits or protocols; DDP-WM's 98% is PushT. No local paired PushObj baseline/+ECP records for those families were found. Missing values remain null in cross_model_pending_v5.json rather than plotted as zero.
- No cross-family deployment or SOTA claim was added. CROSS_MODEL_EXPERIMENTS.md separates comparative ranking from paired portability validation and specifies the additional runs needed. The current six bars show gains across update regimes/checkpoints within one model family.
- All original measured numbers, stagnation equations and main ablation results are preserved. The abstract remains 160 words.
- Final compilation: 7 body pages, 1 reference/statement page, 8 appendix pages, 16 total; no overfull boxes, unresolved references or unembedded fonts.
- No data collection, model training or planning experiment was launched.
