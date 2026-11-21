# Rope 3/3 — new-seed repeat

Four arms, 20 paired tasks each, at most 20 pushes, observation and CD check
after every push, stop immediately at CD <0.3. Each complete three-action block
is preserved unless success terminates execution. CEM remains 300 candidates,
30 elites, 10 iterations. Both model and official DINOv2 backbone are verified
and frozen. Each arm has its own GPU; samples run sequentially on that GPU.

| Arm | Extra warmup | New memory strength |
|---|---:|---|
| Native 3/3 | S disabled | 0 |
| Fixed S 3/3 | 5 pushes | 2 |
| Dynamic S 3/3 | 5 pushes | 2 × detector severity |
| Fixed S 3/3, no extra warmup | 0 pushes | 2 |

The three-push detector window is unchanged. No-warmup can first affect push 4;
five-push warmup can first affect push 7. Fixed versus dynamic describes the
strength of each newly triggered memory, not different observation cadence.

Use base seed 100 instead of the original 99. Keep the same ordered 20 targets
(global IDs 10–29, chunks 1 and 2) to compare seed sensitivity. Scene seed is
100; environment seed is `100 * global_target + 1`; planning seed is
`100 + global_target`. All four arms share these assignments. This is a new
random-seed repeat, not 20 newly generated or previously unseen target pairs.

Reuse the unchanged formal planner implementation. The new worker adds seed
arguments and saves existing RGB/state/action arrays after execution; it does
not add rendering, physics, or RNG calls. All arms execute independently, with
no reuse of Native trajectories. PhysX/Flex execution can diverge even with
matched controls; the report records pre-intervention differences explicitly.

```bash
python3 experiments/rope_seed_repeat/launch.py --seed 100 \
  --root eval_outputs/rope_seed100_20 \
  --report reports/rope_seed100_20 \
  --gpu-map native3=1,fixed3=2,dynamic3=3,fixed3_nowarm=4
```

The launcher automatically audits and summarizes all 80 episodes only after
the four groups finish. Until then, `queue.json` and each episode's
`ecp_progress.json` are the progress records. Report outputs include CD,
successes, trigger counts, actual applied penalties, paired executed-control
differences, comparison with seed 99, and an archive of all episode records.
Source hashes are frozen at launch; a later source change stops the queue.

After completion, reproduce the seed-99/100 comparison with a uniform physical
action-difference threshold (L2 >1e-6) using:

```bash
eval_outputs/rope_local/venv/bin/python experiments/rope_seed_repeat/compare_seeds.py
```

This reads both seeds' original executed controls and verifies their paired
initial/goal states. It distinguishes detector events, memories actually used
in planning, and observed differences from Native, including differences that
occur before intervention or in runs that never apply S.

The completed [three-seed aggregate](../../reports/rope_seeds99_100_101/README.md)
adds seed 101 and compares accuracy, success and mean executed pushes over all
240 runs. Recompute the aggregate from the original episode directories with:

```bash
python experiments/rope_seed_repeat/aggregate_three_seeds.py
```

This requires NumPy and Matplotlib, verifies all paired targets and original
per-seed totals, and writes tables plus a three-panel figure. Seed 99's Native-3
episodes remain in their separate `eval_outputs/rope_formal20_native3` directory.
