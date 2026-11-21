# Maze Frozen paired supplement

100 medium tasks,30 MPC replans,action_repeat4, released checkpoint. Native48/100 verified from full logs. Primary=S with the existing movement trigger; secondary=S+P component control. Both use start7,W3,gamma0.05/0.15,kappa2,beta0.25. Frozen means probe.mode=none.

50 episodes have a relevant trigger and are rerun; the other50 reuse native outcomes, because the intervention cannot activate. Two native smoke episodes matched the historical actions and states. All reruns must pass pre-trigger action/state equivalence. Output: ../../eval_outputs/maze_frozen_v10/; summary.json is written only after both arms complete and pass verification.

The verified sample7 recovery (native failure30, S success28, identical first13 replans) appears in the v10 paper. `finalize_maze_v10.py` waits for the complete S/SP summary, imports verified counts, and builds a separate `paper_completed_v10/paper.pdf` from a locked source archive. See `eval_outputs/maze_frozen_v10/finalization_status.json` for progress.
