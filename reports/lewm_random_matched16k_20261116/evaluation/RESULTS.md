# Random-data LeWM: matched PushObj evaluation

Final epoch 10; 350 tasks per arm; GD 100 iterations; 30 MPC replans × 5 controls.
All 700 new outcomes verified by independent physics replay. Model parameters and buffers stayed frozen.

| Shape | Random native | Random + ECP | Augmented native | Augmented + ECP |
|---|---:|---:|---:|---:|
| I | 16.00% (8/50) | 18.00% (9/50) | 20.00% (10/50) | 20.00% (10/50) |
| L | 54.00% (27/50) | 66.00% (33/50) | 80.00% (40/50) | 82.00% (41/50) |
| square | 36.00% (18/50) | 42.00% (21/50) | 40.00% (20/50) | 42.00% (21/50) |
| Z | 50.00% (25/50) | 56.00% (28/50) | 74.00% (37/50) | 76.00% (38/50) |
| small_tee | 58.00% (29/50) | 66.00% (33/50) | 72.00% (36/50) | 84.00% (42/50) |
| plus | 56.00% (28/50) | 56.00% (28/50) | 70.00% (35/50) | 74.00% (37/50) |
| T | 62.00% (31/50) | 68.00% (34/50) | 74.00% (37/50) | 84.00% (42/50) |
| All | 47.43% (166/350) | 53.14% (186/350) | 61.43% (215/350) | 66.00% (231/350) |

Random-data ECP paired outcomes: 21 rescues, 1 regressions.

Augmented results are the archived matched runs; they were not rerun or mixed with other task sets.
ECP uses stagnation thresholds 0.15/0.45, memory regularization, and contact optimization weight 10 with conditional analytic proprioceptive guidance. No post-optimization action offset is used.
See `paired_tasks.csv`, `per_shape.csv`, and `full/<arm>/<shape>/replayed_trajectories.npz` for outcomes and trajectories.
