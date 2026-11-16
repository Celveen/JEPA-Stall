# LeWM: matched random-data training and PushObj evaluation

The newly trained random-data LeWM achieves **47.43%** native PushObj success and
**53.14%** with ECP. The matched augmented-data reference achieves **61.43%** and
**66.00%**, respectively. Each condition uses the same 350 tasks (seven shapes,
50 per shape). All 700 new evaluation outcomes were verified by independent
physics replay, with model parameters and buffers unchanged during planning.

| Training data | Native | With ECP | ECP gain |
|---|---:|---:|---:|
| Random/source-action replay | 166/350 (47.43%) | 186/350 (53.14%) | +5.71 pp |
| Action-biased augmentation | 215/350 (61.43%) | 231/350 (66.00%) | +4.57 pp |

Data augmentation improves native planning by **14.00 percentage points**.
Augmentation plus ECP improves over random-data native planning by **18.57 points**.
Within the new random-data checkpoint, ECP rescues 21 failures and changes one
native success into a failure. The augmented reference has 18 rescues and two
regressions. Augmented results are archived matched evaluations, not new runs.
See [per-shape results](evaluation/RESULTS.md) and [paired tasks](evaluation/paired_tasks.csv).

## Training and protocol

- From scratch, seed 0; 16,000 training and 4,000 validation trajectories.
  Source episode selection, initial states, shapes and lengths match the augmented
  dataset. Random data replays the source actions without the additional 9 px
  contact bias and per-axis 10 px Gaussian noise; it is not newly sampled IID noise.
- Same LeWM architecture and recipe as the enhanced reference: 18,034,478 parameters,
  10 epochs, 127,360 updates, global batch 128, AdamW learning rate 5e-5,
  weight decay 0.001, SIGReg weight 0.09, bf16, and the same calibration/validation budgets.
- Eight RTX 3090 GPUs, 16 windows per rank, SyncBatchNorm and differentiable global
  SIGReg. Global loss/gradient equivalence was checked before training; independent
  per-rank dropout streams mean the random trajectory is not bitwise identical to
  a single-GPU run. See [training details](training/README.md) and [preflight](training/preflight.json).
- Completed 2026-11-16 17:16 CST, after 6.6806 hours. Final checkpoint strictly loads
  into the original evaluator, all parameters/buffers are finite, and latest/final
  weights are tensor-identical. Final validation: loss 0.18494746, prediction loss
  0.00878531, SIGReg 1.95742178.
- Evaluation completed 17:24 CST: 14 jobs, GD 100 iterations, horizon 25 controls,
  at most 30 MPC replans × 5 controls = 150 environment controls per task.
- ECP uses thresholds 0.15/0.45, memory regularization, contact objective weight 10,
  and conditional analytic proprioceptive guidance. Contact targets use observed
  simulator state and known geometry. No post-optimization additive action offset
  is applied in this variant. The first ten replans match between native and ECP.

## Files and verification

- `training/`: recipe, provenance, full per-step/per-epoch metrics, final training
  receipt, distributed preflight checks and completion status.
- `evaluation/`: results, paired/per-shape CSVs, all 14 planning configurations,
  final checkpoint receipt and evaluation status.
- `training_and_evaluation.tar.gz`: complete training/evaluation records, raw console
  logs, executable runtime snapshots, fixed task segments, original outcome arrays,
  action sequences, intervention logs, target observations, replayed state/contact
  trajectories and the exact evaluation source. Python caches and the empty process
  lock are omitted. No results or log lines are truncated.
- `archive_manifest.json`: byte size and SHA-256 for every archived file.
- `SHA256SUMS.json`: checksums for all published files. Every archive member was
  read back and checked against the original local file before publication.

Extract using `tar -xzf training_and_evaluation.tar.gz -C <empty-directory>`.
Scripts and raw configs retain original absolute server paths as execution
provenance; adjust these paths before rerunning elsewhere. Historical training
receipts say planning had not started at the instant training completed; the later
evaluation completion receipt is authoritative for downstream evaluation status.

Large checkpoint binaries and training videos are retained on the server, outside
Git, following the repository checkpoint policy. Final checkpoint SHA-256:
`13b8a6ce0b7a866db0e24ccba8cfe24e64ced5710f5d21ebac34eb0e22397b17`.
The exact checkpoint path is recorded in `training/training_result.json`.

The source run used repository commit `5ab1d1b`; this report is appended to the
latest cloud branch without changing the manuscript or other experiment results.
