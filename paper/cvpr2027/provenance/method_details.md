# Verified implementation details for the appendix

The descriptions below were checked against executed-run configurations and source code. All counts were reconstructed from existing logs on CPU; no new experiment was run.

## Success criteria and budget accounting

**PushObj.** Let \(p_a,p_o\in\mathbb R^2\) denote the pusher and object positions, and let \(\phi\) denote object orientation. A rollout is successful when
\[
\left\|[p_a-p_a^\star;\,p_o-p_o^\star]\right\|_2<20\ \text{pixels},\qquad
\left|\operatorname{wrap}_{[-\pi,\pi)}(\phi-\phi^\star)\right|<\pi/9.
\]
The position criterion is one joint four-dimensional norm; it is not two independent20-pixel thresholds. Velocity is excluded. A task is counted as solved at the first successful replan boundary. The planner latches success and terminates its per-task rollout; success at an intermediate physics step is not separately counted. Thirty replans execute at most150environment control calls.

**Maze.** Success is \(\|p-p^\star\|_2<0.5\) in maze coordinate units, independent of velocity. The authoritative evaluation predicate is strict `<0.5`; the simulator's sparse reward uses `<=0.5`, which is not the metric used in the tables. The E3 study samples100start–goal tasks round-robin across5held-out layouts,20tasks per layout, with BFS cell distance6–9 and evaluation seed100. Each of5control calls executed per replan repeats its action4times in the base environment. Thirty replans therefore allow150control calls/600base-environment steps;50replans allow250/1000. The planning horizon is25control calls/100base steps. This action-repeat4 protocol is used for both baseline and interventions. Earlier action-repeat1 runs are a separate protocol.

Sources: `env/pusht/pusht_wrapper.py:70`, `env/pointmaze/point_maze_wrapper.py:69`, `planning/evaluator.py:176`, `planning/mpc.py:132`, `datasets/diverse_maze_goals.py:115`, `scripts/e3_run.py:154`.

## World model and offline objective

The PushObj model uses a SmallResNetGeM visual encoder with384output dimensions and a six-layer action-conditioned transformer (16heads, feed-forward width2048, dropout0.1). Proprioception and each5-action block have learned10-dimensional embeddings. Four observations spaced5environment control steps apart form a training window;3observations predict the corresponding three one-step-shifted targets. The action embedding is supplied to the predictor but excluded from prediction loss.

Writing \(z_i^o=[z_i^v;z_i^p]\) for the concatenated visual and proprioceptive representation, the offline objective is
\[
\mathcal L_{\rm WM}=\frac1{3d_o}\sum_{i=0}^{2}
\|\hat z_{i+1}^{o}-\operatorname{sg}(z_{i+1}^{o})\|_2^2
+\lambda_{\rm TS}\mathcal L_{\rm TS},
\qquad
\mathcal L_{\rm TS}=\mathop{\mathrm{mean}}_{i\in\mathcal V}
\left[1-\frac{\langle\Delta z_i^v,\Delta z_{i+1}^v\rangle}
{\|\Delta z_i^v\|_2\|\Delta z_{i+1}^v\|_2}\right].
\]
The code computes cosine similarity with epsilon\(10^{-6}\), includes only adjacent displacements with both norms greater than\(10^{-6}\), and returns0when no pair is valid. TS acts on observed visual embeddings only, not proprioception, action embeddings, or predicted trajectories. Thus the training loss averages visual/proprio coordinates jointly; it is not the separately branch-normalized planning metric \(D_\alpha\). There is no VICReg term in these runs. The training target branch uses stop-gradient, not an EMA target network.

For historical PushObj bias16k checkpoints and B0–B3, \(\lambda_{\rm TS}=0.01\). Training uses3epochs, effective batch64, bf16, Adam for the visual encoder at\(10^{-5}\), and AdamW for the predictor and action/proprio encoders at\(5\times10^{-4}\). The AdamW weight decay is the PyTorch default0.01; Adam weight decay is0. Optimizer betas are the defaults\((0.9,0.999)\). These source paths have no offline learning-rate scheduler. The training split contains16,000trajectories/1,630,266windows; validation contains4,000trajectories/403,951windows. Historical seed0 trains a separate VQ decoder in epoch3 with fully detached encoder inputs and its own Adam optimizer at\(3\times10^{-4}\). Historical seeds1/2 and B0–B3 omit the decoder. No decoder is used for the reported control objectives.

The released Maze model uses SmallResNet (without GeM),384visual dimensions, the same transformer dimensions, and training configuration TS coefficient0.001. Its released checkpoint is reused; no Maze model is trained in this study. **Both released `pushobj_shape_shift/model_latest.pth` and `diversemaze_layout_shift/model_latest.pth` contain checkpoint epoch3**, confirmed by CPU metadata loading. Their Hydra files specify a planned5-epoch training run; do not describe the actual released snapshots as epoch5.

Sources: `models/visual_world_model.py:270`, `models/visual_world_model.py:312`, `models/visual_world_model.py:339`, `models/visual_world_model.py:374`, `train.py:418`, `conf/train_pushobj_bias16k.yaml`, `conf/train_pushobj_bias16k_seeds.yaml`, and each checkpoint's `hydra.yaml`.

## Planning and online adaptation

All main planning comparisons optimize a25-control-step action sequence and execute its first5steps. With frameskip5, the world model predicts5latent steps and executes one latent step per replan. Adam optimizes the action sequence for100iterations with initial learning rate0.1, cosine annealing to0, and no optimization noise. The first sequence initializes to zero in physical action units (then normalizes); later plans warm-start from unexecuted actions and append zero actions to fill the horizon. The frozen arms keep all model parameters fixed.

For Maze, the task cost uses all rollout frames, including the initial context frame, with weights proportional to\(2^h\). For PushObj, the first5MPC plans use the terminal-frame cost; subsequent plans use the weighted all-frame cost. In the all-frame implementation the normalized weighted sum is additionally divided by the number of frames. The memory regularizer's `D(memory, goal)/(H+1)` scaling preserves this convention. Alpha is1before intervention; a proprioceptive intervention multiplies it by10for3plan calls, refreshed by later triggers. The memory penalty has coefficient2and squared-distance radius fraction0.25.

AdaJEPA performs one gradient step after every replan on a replay buffer containing the5most recent execution segments. The objective is a teacher-forced one-step latent prediction MSE, with target stop-gradient and no TS term during online updates. Only the final transformer block plus final transformer normalization and the encoder's final projection submodule are trainable; action and proprio encoders stay fixed. The predictor and encoder learning rates are\(5\times10^{-4}\) and\(10^{-5}\), respectively. Online optimization uses Adam, instantiated afresh for each update call, so Adam moments are not carried across replans. Predictor dropout is active during updating; planning and diagnostic prediction are in evaluation mode. Adapted parameter values and mutable model buffers reset to the pretrained snapshot at the beginning of every task.

The gate compares3-point means of prediction error on the newest, not-yet-trained transition, separated by3replans. Its threshold is a relative decrease below0.05, together with positive stall weight, for one qualifying replan. The gate activates planning corrections on the next plan. Model updates continue after activation in R-switch, SZ-switch, and P-switch; freezing occurs only in the explicitly labeled F controls.

Episode RNG initialization uses `escape.seed *1000003 + episode_seed`, where `episode_seed=evaluation_seed*within_chunk_index+1`. The escape seed is0in the main configurations, so the RNG is seeded directly with the episode seed; it is distinct from the evaluation seed100(or200). Shards retain each episode's original index so subset execution preserves randomization and target assignment.

Sources: `planning/gd.py:24`, `planning/gd.py:53`, `planning/gd.py:89`, `planning/objectives.py:36`, `planning/adajepa.py:83`, `planning/adajepa.py:119`, `planning/adajepa.py:170`, `planning/adajepa_mpc.py:25`, `planning/adajepa_probe.py:281`, `planning/escape.py:115`, `planning/switch.py:52`.

## Paired execution and exact reuse of unaffected trajectories

Only episodes on which an intervention can activate are rerun. Every unselected trajectory is copied from the corresponding baseline, including both successful and unsuccessful trajectories. This is exact computational reuse: the methods execute identical code before their first possible correction, the episode RNG and model reset are fixed, and baseline/arm actions were checked to match before activation. Baseline success by itself is **not** the selection rule: late baseline successes that may be changed by intervention are rerun.

For the PushObj main matrix, the original runs allow40replans; the paper's30-replan results are obtained by truncating these existing traces. Selection uses a stall trigger anywhere in the original40-replan run. Consequently, some rerun episodes may not yet be corrected in the first30replans. The same episode subset is used for SZ_mid, T0P, and T0SP within each row:

| Checkpoint / planner | Total | Rerun per arm | Reused | Reused successful by30 | Reused unsuccessful by30 |
|---|---:|---:|---:|---:|---:|
| Released / frozen |350|195|155|150|5|
| Released / AdaJEPA |350|109|241|234|7|
| Bias16k / frozen |350|134|216|213|3|
| Bias16k / AdaJEPA |350|92|258|255|3|

ALPHA10 and LA_start affect the initial plan, so all350episodes are rerun. They do not use unaffected-trajectory reuse.

For Maze E3,51of100episodes trigger in the50-replan baseline and are rerun for every E3 arm. The other49are reused;45are successful by30replans and47by50. For PushObj S7, all78episodes not solved by replan10are rerun; the202episodes solved by replan10are reused. The50-replan S7 baseline combines its30-replan run with exact-prefix-verified50-replan extensions of the63failures. Each E3 arm preserves the original episode position.

The four PushObj matrix rows use the same350target tasks with different checkpoints/planners. They are not1,400independent tasks. Training-screen B0–B3 instead each rerun all350tasks; these rows use a new matched training pipeline and must not be mixed with old bias16k intervention rows as paired comparisons.

Sources: `scripts/stag_base_run.py:88`, `scripts/stag_base_run.py:133`, `scripts/e3_run.py:82`, `scripts/e3_analyze.py:133`; counts independently reconstructed from the indicated raw paths in `evidence.json`.
