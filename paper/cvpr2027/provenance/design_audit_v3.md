# Manuscript and figures: v3 audit

Prepared 2026-11-06. The manuscript is organized around observed repeated local motion, a history-conditioned planning principle, an explicit stagnation regularizer, and paired recovery evidence. Structural references are documented with primary-paper URLs in narrative_v3.md.

- Abstract: 163 words, page 1. New paired trajectory Figure 1 starts on page 2.
- Figure 1: genuine native simulator state renderings plus exact logged MPC endpoint positions; PushObj S+P and Maze gated S labeled separately. Local insets show full sampled endpoint polylines and net displacement. Dashed connectors do not represent unlogged control-substep paths. Matching task arrays and execution prefixes are verified. Source hashes, coordinate transforms, all drawn states, and metrics are in trajectory_v3_manifest.json.
- Figure 2: Predictive learning, encoder, predictor, and ECP modules. No TS, AdaJEPA, or LeWM label appears in the artwork. Stagnation regularizer and temporary goal reweighting have distinct visual roles. Inherited training and online-update methods remain credited in manuscript text.
- Equations: core objective, trigger, amplitude/support scale, compact-support kernel, and horizon aggregation are separate displays. Symbols are defined at first use and collected in Table 6. Code-grounded equation audit includes file/line references. Full goal objective and kernel/action gradients are in the appendix.
- All plotted results remain unchanged. Detailed B0–B3 result table moved beside its appendix objectives; exact success rates remain in the main discussion.
- Rendered checks: inspected first-page/second-page placement, method diagram, formula column widths, symbol table, result charts, final body page, and supplementary renderings. No overfull boxes or unresolved citations/references. All PDF fonts embedded.
- Main text is 8 pages; statement/references end on page 9; appendix begins on page 10 and spans 7 pages. Combined PDF is 16 pages.
- No new training, planning, or action execution experiments were run. CPU state rendering and log analysis only.
