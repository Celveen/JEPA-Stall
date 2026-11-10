# DINO-WM PushObj transfer: three-epoch paired experiment

Status: both three-epoch trainings are complete. Evaluation has moved to the [v3 protocol](../dinowm_pushobj_v3/PROTOCOL.md), with 200x10 CEM and six plans executing 25 controls each. The evaluation section below records the superseded original launch; its logs and metadata remain as provenance.

## Comparison

| Training data | Frozen native CEM | Frozen CEM + ECP |
|---|---|---|
| Matched ordinary actions | ordinary_native | ordinary_ecp |
| Existing action-biased augmentation | augmented_native | augmented_ecp |

Both conditions use identical initial parameter hashes and optimization settings. Model parameters remain frozen throughout evaluation, including the ECP condition. This experiment tests whether data augmentation and planning intervention transfer to DINO-WM under the same limited training budget; it does not reproduce the paper's 100-epoch result.

## Data

The augmented condition reuses `data/pushobj_bias_16k` without alteration: 16,000 training episodes and 4,000 validation episodes, with T, L, Z and plus shapes. Existing augmentation adds a 9-pixel bias toward the block center of gravity and Gaussian noise with standard deviation 10 pixels. Sequence lengths remain those of the existing dataset (49–246 frames); they are not cropped to the official paper's 100-frame collection.

The matched ordinary control is `data/dinowm_pushobj_ordinary_matched16k_v1`. It replays original source actions without this additional bias/noise, retaining each augmented episode's selected source, initial pose and velocity, length, shape and train/validation split. Generation verified all initial states and velocities. Selection and placement rules inherited from the existing augmented collection are held fixed; this control is not represented as the authors' released original PushObj dataset. Its `manifest.json` records source indices and augmented-source metadata hashes.

Each condition contains 1,630,266 training windows and 403,951 validation windows at history 3, prediction 1 and frameskip 5. There are 50,946 optimizer steps per epoch at global batch size 32, or 152,838 steps across three epochs. The preliminary `dinowm_pushobj_matched20k_v1` generation was stopped and is not used.

## Model and optimization

Source: [DINO-WM paper](https://arxiv.org/html/2411.04983v2) and [official implementation](https://github.com/gaoyuezhou/dino_wm), commit `0a9492fa12044b852ae9e001cc74604b79c8bb0c`. Official snapshots and license are included in this directory.

- Frozen DINOv2 ViT-S/14 encoder; 14 × 14 spatial patch tokens, 384 visual channels.
- Six-layer, 16-head predictor; MLP dimension 2048, dropout 0.1; 10-dimensional action and proprioceptive embeddings give model dimension 404.
- Native predictive loss, history 3, one-step prediction, frameskip 5. The detached visualization decoder is omitted.
- AdamW predictor/action/proprioceptive learning rate 5e-4, matching the checked official code. The paper's table lists predictor learning rate 5e-5; the code default is the selected reference here.
- Global batch 32, two DDP ranks, microbatch 8 per rank with gradient accumulation; bf16 precision and SDPA attention are hardware adaptations. Float32 attention forward/backward equivalence passed; see `attention_equivalence.json`.
- Identical initialization and fixed sample order; seed details are reproducibility metadata, not an experimental contribution.
- **Three epochs**, full validation each epoch, archived checkpoints at epochs 1, 2 and 3. Resumable latest checkpoint every 1,000 optimizer steps includes optimizer and per-rank RNG state.

Augmented training uses GPUs 3/5; ordinary training uses GPUs 0/1. The augmented job resumed from optimizer step 10,000. The revised sampler jumps to the exact remaining batches without decoding completed data; its order was checked for both ranks (`resume_sampler_validation.json`). Existing unrelated processes are retained. Each training rank has a 4 GiB PyTorch allocator limit; launch requires at least 16,000 MiB free memory per GPU. Measured training allocation is approximately 1.25 GiB/rank.

## Evaluation

The queue waits for both `complete.json` files to confirm epoch 3, then evaluates the four conditions on the existing `data/targets/EVAL50` benchmark: 35 fixed chunks of 10 episodes, 50 per shape, 350 per condition. These are 1,400 deployments total. Source target hashes are saved in `evaluation_queue.json`. Chunk order, per-chunk seeds and episode ordering remain paired. T/L/Z/plus are seen shapes; I/small_tee/square are unseen. Summary reports seen, unseen and all-shape rates separately.

Native CEM uses 300 candidates, top 30 elites, 30 optimization iterations and a five-model-step (25 physical-step) horizon, with the official terminal objective. Candidate evaluation uses chunks of 16 with cached observation encodings; all candidates participate in a single global elite selection. The unchanged native rollout and CEM produce identical outputs in `planning_validation.json`.

The common deployment budget is **30 MPC replans**, executing one model step (five physical steps) per replan, up to 150 physical steps. This intentionally matches the existing paper experiment budget. The released DINO-WM PushT planner instead executes five model steps per replan; our comparison must not be labeled an exact replication of that deployment protocol.

ECP transfers the existing PushObj settings: joint visual/proprioceptive latent memory, kappa 2, beta 0.25, trigger start 10, window 3, lower/upper movement thresholds 0.05/0.15, proprioceptive multiplier 10, and boost through two subsequent replans. The native arm uses kappa 0 and multiplier 1. No thresholds are fitted to these results. Forced early activation is used only in integration tests.

Integration checks establish matching simulation start/goal states, bitwise identical pre-trigger actions, active proprioceptive intervention, positive stagnation penalty at a stored stalled state, zero penalty at the goal, and unchanged model weights. Smoke success rates are not experiment results. Full-run summaries also record paired rescues/regressions and pre-trigger discrepancies.

## Running artifacts

- `ordinary_train_launcher.json`, `augmented_train_launcher.json`: actual commands, PIDs, GPUs and start times.
- `ordinary_train.log`, `augmented_train.log`: appended training output.
- `checkpoints/dinowm_pushobj_v1/{ordinary,augmented}/status.json`: progress; `metrics.jsonl`: losses; `resume.json`: restored position; `complete.json`: training completion.
- `pipeline_launcher.json`, `pipeline_status.json`: durable queue process and overall progress. The queue fails explicitly if a training process exits without its completion marker.
- `eval_outputs/dinowm_pushobj_3epoch_v1/`: per-condition, per-chunk outputs, configs, logs, model hashes and success arrays. `summary.json` is written only after all 140 chunks complete.

The initial estimate is 8–10 hours for both three-epoch trainings running concurrently, subject to shared GPU and data-loading throughput. Evaluation time is additional. No manuscript result table has been updated with incomplete DINO-WM runs.
