# DINO-WM evaluation performance diagnosis — 10 November 2026

This is an inference microbenchmark on a saved initial observation and goal, not a new task evaluation. Existing four-arm evaluation processes, checkpoints, outputs and scientific settings were not changed. GPU 6 was idle before the bounded diagnostic and is released after it.

## Verified existing run

All four completed first chunks contain 10 episodes and exactly 300 MPC records, with a maximum of 30 replans per episode. Model-parameter displacement is zero. The configured physical horizon 25 is divided by frameskip 5 in `plan.py`, giving 5 model transitions. Each replan executes one model transition, or 5 environment control steps.

| Condition | Wall time (h) | Mean replan (s) | Time in planner | Other time (s) |
|---|---:|---:|---:|---:|
| ordinary_native | 8.789 | 105.155 | 99.705% | 93.18 |
| ordinary_ecp | 8.775 | 104.994 | 99.709% | 91.89 |
| augmented_native | 8.673 | 103.765 | 99.704% | 92.34 |
| augmented_ecp | 8.612 | 103.040 | 99.707% | 90.86 |

## Work per episode

Each replan evaluates 300 candidates in each of 30 CEM iterations, rolling out 5 latent model transitions per candidate. At the 30-replan limit this is 1,350,000 candidate-transition predictions per episode (batched, not 1,350,000 Python calls). The implementation processes 16 candidates per batch, yielding 19 batches per CEM iteration and 2,850 predictor calls per replan. The 10 episodes within a chunk run sequentially; the four conditions run concurrently on GPUs 0, 1, 3 and 5.

Parameters are frozen and CEM runs under `torch.no_grad()`. Observed image encodings are already cached. Freezing weights removes optimization but does not remove forward predictions for different proposed action sequences.

## Fixed-candidate timings

One saved real observation/goal from ordinary_native/chunk_00_I, epoch-3 ordinary checkpoint, identical 300 Gaussian candidates, 5 predicted model transitions, two timed repeats after warm-up on an RTX 3090. Input encoding is kept FP32 and excluded from timing. BF16 autocast applies to candidate rollout; returned costs are compared to the FP32 batch-16 reference. PyTorch 2.3.0+cu121; parameter storage FP32; default CUDA matmul TF32 disabled.

| Precision | Candidate batch | Seconds/CEM iteration | Speedup | Peak allocated GiB | Max cost difference | Elite overlap /30 |
|---|---:|---:|---:|---:|---:|---:|
| fp32 | 16 | 3.213 | 1.00x | 0.436 | 0 | 30 |
| fp32 | 64 | 3.164 | 1.02x | 1.237 | 4.76837e-07 | 30 |
| fp32 | 300 | 3.181 | 1.01x | 4.633 | 4.76837e-07 | 30 |
| tf32 | 16 | 2.586 | 1.24x | 0.437 | 0.000458956 | 30 |
| tf32 | 64 | 2.500 | 1.29x | 1.237 | 0.000525355 | 30 |
| tf32 | 300 | 2.494 | 1.29x | 4.633 | 0.000499249 | 30 |
| bf16 | 16 | 0.997 | 3.22x | 0.359 | 0.00992537 | 29 |
| bf16 | 64 | 0.922 | 3.48x | 0.921 | 0.00864577 | 30 |
| bf16 | 300 | 0.896 | 3.59x | 3.154 | 0.00864577 | 30 |

## Interpretation

Increasing batch size alone barely changes FP32 throughput: 3.213 to 3.181 s. The principal bottleneck is full-precision predictor computation, not low GPU occupancy, simulation, image/video output or adaptation. Profiling one baseline batch attributes about 44.8% of GPU operator time to FP32 memory-efficient attention and about 49.2% to linear matrix multiplications. Existing worker GPUs run at 100% utilization and about 349 W.

BF16 with batch 300 reduces the fixed-candidate computation to 0.896 s (3.59x); batch 64 is close at 0.922 s and uses less memory. Multiplying by 30 CEM iterations gives roughly 27–28 s per replan before non-CEM work, compared with measured live 103–105 s. This is a local throughput projection, not a measured full-task speedup. It would still leave a multi-day full evaluation on four workers; it does not establish a new ETA.

BF16 is not numerically identical: in this observation the maximum candidate-cost deviation is 0.00865 for batch 64/300 and the selected elite set matches. One observation/iteration does not establish equivalence after 30 CEM updates or closed-loop deployment. A precision change must be validated on paired saved states/actions and recorded as a distinct evaluation configuration; incomplete old and new results must not be silently pooled.

Recommended sequence: keep the 30-MPC limit and 300-candidate/30-CEM-iteration search budget, validate an accelerated inference configuration, then apply it consistently to all four conditions with explicit provenance. Reducing CEM iterations is a separate planning-budget experiment and requires confirmation under the user's prior instruction. No such experiment or restart was launched here.
