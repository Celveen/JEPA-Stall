# Medium Maze results across three seeds

With the released world model, AdaJEPA+S solved **117 of 150 tasks (78.0%)**, up from **109 of 150 (72.7%)** for AdaJEPA. Mean final position distance dropped from **0.8370 to 0.7450** and mean executed controls from **69.31 to 67.63**. All three metrics improved in every seed. Frozen+S improved on average, with a regression in seed 102.

The comparison uses 50 tasks per seed and condition, balanced across five held-out layouts with shortest-path distances of 6–9 grid cells. Within each seed, all four conditions share the same layouts and start/goal states. The 150 tasks per condition include the original 20 pilot tasks in seed 100. On the remaining 130 tasks, AdaJEPA+S improves from **96 to 103 successes** and from **0.8166 to 0.6815** mean final distance.

| Condition | Successes | Success rate | Mean final distance ↓ | Mean executed controls ↓ | Detector events | Committed memories | Runs with changed actions | Changed / shared controls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Frozen | 92/150 | 61.3% | 1.0044 | 78.61 | — | — | — | — |
| Frozen+S | 97/150 | 64.7% | 0.9295 | 75.86 | 4,366 | 1,157 | 38/150 | 2,791/11,307 |
| AdaJEPA | 109/150 | 72.7% | 0.8370 | 69.31 | — | — | — | — |
| AdaJEPA+S | **117/150** | **78.0%** | **0.7450** | **67.63** | 2,943 | 906 | 48/150 | 2,771/9,979 |

Each cell below reports **successes out of 50 / mean final distance / mean executed controls**.

| Condition | Seed 100 | Seed 101 | Seed 102 |
|---|---:|---:|---:|
| Frozen | 33 / 0.8997 / 75.30 | 29 / 1.1122 / 80.32 | 30 / 1.0012 / 80.20 |
| Frozen+S | 35 / 0.8162 / 72.62 | 34 / 0.8799 / 74.04 | 28 / 1.0924 / 80.92 |
| AdaJEPA | 35 / 0.8045 / 72.54 | 37 / 0.8755 / 71.06 | 37 / 0.8309 / 64.32 |
| AdaJEPA+S | 38 / 0.7893 / 71.30 | 39 / 0.7359 / 69.48 | 40 / 0.7098 / 62.12 |

Each planner predicts 25 controls and executes blocks of up to five. It observes the simulator after every control and stops immediately at a position distance below 0.5, with a limit of 150 controls. The S conditions use thresholds of 0.30/0.10, a 15-control detection window, and dynamic strength `2w`. Frozen and AdaJEPA start checking for stalls at controls 35 and 20, respectively. Each completed block can commit one memory for the next plan. AdaJEPA+S retains the prediction-loss gate and AdaJEPA updates use full blocks only.

A detector event is a control with positive stall weight. Action comparisons use the clipped controls executed by both methods before either stops, with an L2 difference threshold of `1e-6`. The paired audit checks actions and states before S first becomes active. Final distance and control means include every task, including failures.

- [Aggregate CSV](aggregate.csv), [per-seed CSV](by_seed.csv), [all 600 episodes and paired statistics](summary.json), and [checkpoint and run provenance](provenance.json).
- Seed 100: [episode CSV](../maze_medium50_s100_g030_observed/episodes.csv), [summary](../maze_medium50_s100_g030_observed/summary.json), [task manifest](../maze_medium50_s100_g030_observed/goals.npz), [protocol](../maze_medium50_s100_g030_observed/protocol.json), [per-control records](../maze_medium50_s100_g030_observed/per_control.jsonl.gz).
- Seed 101: [episode CSV](../maze_medium50_s101_g030_observed/episodes.csv), [summary](../maze_medium50_s101_g030_observed/summary.json), [task manifest](../maze_medium50_s101_g030_observed/goals.npz), [protocol](../maze_medium50_s101_g030_observed/protocol.json), [per-control records](../maze_medium50_s101_g030_observed/per_control.jsonl.gz).
- Seed 102: [episode CSV](../maze_medium50_s102_g030_observed/episodes.csv), [summary](../maze_medium50_s102_g030_observed/summary.json), [task manifest](../maze_medium50_s102_g030_observed/goals.npz), [protocol](../maze_medium50_s102_g030_observed/protocol.json), [per-control records](../maze_medium50_s102_g030_observed/per_control.jsonl.gz).
