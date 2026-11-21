# CVPR 2027 paper and evaluation artifacts — v14

- [Frozen ECP paired cost analysis](../frozen_ecp_cost_20261117/README.md): 350 matched tasks per arm with GD100 + CUDA Graph. ECP increases success from **61.14% to 72.29%**, adds **2.81%** per-replan latency, and reduces mean task-loop wall time by **12.06%**; peak allocated GPU memory rises from **0.823 to 0.967 GiB**. Includes all 700 task measurements, raw actions/replays, configurations, source snapshots and a CPU-only verifier.

The main manuscript is [paper_v14.pdf](../../paper/cvpr2027/paper_v14.pdf); editable source and native simulator figures are in `paper/cvpr2027/`.

## Results included

**Update, 17 November 2026:** [Figure 1 evaluation-seed follow-up](../figure1_eval_seeds_20261117/README.md) adds complete three-seed statistics for all three AdaJEPA conditions and augmented ECP for Frozen/LeWM. AdaJEPA reaches **68.76 ± 1.41% / 73.90 ± 1.19% / 76.86 ± 1.03%** for base / augmentation / augmentation + ECP. The follow-up includes all 3,500 additional-seed outcomes, physical replay, checkpoint/task identities and exact CUDA Graph validation. Four Frozen/LeWM base/augmentation conditions remain single-seed. Existing compiled manuscript figures retain their original single-seed values.

**Update, 11 November 2026:** the DINO-WM PushObj v3 run has completed. Its [complete results and logs](../dinowm_pushobj_v3/completed_results/README.md) supersede the partial snapshot below for that experiment. The original manifest and progress archive retain their historical snapshot dates.

- **PushT and six visual shifts:** all 56 checkpoint/update/ECP/condition evaluations, 2,800 task runs; 1,400 pre-intervention pairs verified.
- **Frozen Maze:** native 48/100, prespecified S 50/100, S+P component control 54/100; completed paired records and inactive-task reuse manifest.
- **DINO-WM PushT CEM:** random T16k, fixed epoch 3, 300 candidates x 30 iterations, 6 x 25 environment controls. Seven scores are 48/40/2/44/40/24/48 percent (clean/blur/salt-pepper/dark/agent/object/goal); six-shift mean 33.0%. Five raw NPZ conditions and two completed-summary conditions are explicitly distinguished in `paper/cvpr2027/provenance/dinowm_pusht_cem_e3_v14.json`. The historical source folder name includes `e10`; checkpoint receipts identify epoch 3.
- **New DINO-WM PushObj experiment:** both three-epoch trainings are complete. The v3 200 x 10 CEM evaluation uses six plans executing 25 controls each and remains in progress at the snapshot time. Partial results are archived separately, not added to the paper's complete benchmark table.

## Reading the archives

`manifest.json` records the snapshot time, classifications and SHA-256 hashes. Each `.tar.gz` includes an `ARCHIVE_MANIFEST.json` listing every member and its content hash. Extract an archive from the repository root to recover original relative paths:

```bash
tar -xzf reports/publication_v14/archives/pusht_visual_complete.tar.gz
```

Archives retain raw success arrays, start/goal states, normalized executed actions, per-iteration/episode logs, configurations and completion evidence. Duplicated rendered observation caches, videos and old generated PDF copies are excluded; the saved actions and simulator provide replay. Model binaries and image training corpora remain outside Git, with source/download instructions, configuration and identity hashes retained. `dataset_and_checkpoint_metadata/` contains collection manifests and training/checkpoint metadata. The main PushObj Frozen/AdaJEPA runs, Maze native/adaptive runs, and additional controller studies are also included in the `main_*` numerical archives. Existing summary reports retain their original tracked report folders.

No new experiment is launched by packaging or paper-building scripts. Historical v1/v2 archives are superseded and must not be aggregated with v3. The publication branch freezes a point-in-time snapshot; live processes continue locally.
