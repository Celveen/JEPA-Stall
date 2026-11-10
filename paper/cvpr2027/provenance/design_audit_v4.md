# Manuscript v4: final audit

Prepared 2026-11-06. The revision follows Experimental Setup → Comparison with Official Baselines → Ablation Studies → Qualitative Analysis. Primary-paper structural references are documented in experiment_structure_v4.md. Numerical evidence was independently recomputed from existing logs and reviewed in results_audit_v4.md.

- The abstract is 160 words on page 1. Figure 1 remains on page 2; the method graphic is on page 4 and contains no external method names.
- Table 1 on page 6 presents official released/native, our interaction-biased data/native, and our data/ECP as three stages, with Frozen and AdaJEPA side by side. Official rows are matched local reevaluations, not copied publication scores. Full ECP is S+P on PushObj; P-only remains separately labeled.
- Section 4.3 contains four ablation questions: components, conditional versus permanent weighting, memory geometry/strength, and activation timing/continued adaptation. Tables 2–5 provide the corresponding controls. The six-cell geometry/strength study was recovered from existing logs; no new evaluation was run.
- Main-text seed identifiers and task-stream bookkeeping have been removed. Appendix C retains replication, task reuse, calibration, exact counts, additional planning controls, and the development protocol. The main setup keeps a concise disclosure of development-set use and the scope of the full training-recipe comparison.
- Training-data gains concern the complete collection/training recipe. A clean direction-bias/noise factorial and a missing Maze S-only timing control are proposed in NEXT_EXPERIMENTS.md, pending author confirmation.
- Existing real simulator trajectory figures and formulas are retained. All symbols remain defined; the parameter table and full objective are in the appendix.
- Visually inspected the reorganized main results, component/timing tables, new strength table, Maze controls, qualitative plot, and conclusion. No overfull boxes or unresolved references/citations. All fonts are embedded.
- Final layout: 7 body pages, 1 statement/reference page, 8 appendix pages, 16 combined pages. Appendix starts after a page break. No orphaned conclusion page.
- Final independent prose review corrected subject–verb agreement and clarified that planner comparisons hold tasks/action initialization fixed within each checkpoint.
- No training, collection, planning evaluation, or simulator-action experiment was launched for this revision.
