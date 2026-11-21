# Frozen: paired ECP computational cost

Full original 350 fixed PushObj tasks, seed 200, augmented epoch-3 checkpoint. Two conditions: frozen native GD versus the same frozen model with ECP (0.15/0.45, kappa 2, proprio boost 10). Both use GD100 and the same CUDA Graph implementation, 30 replans of five controls maximum. Model hashes and physical replay are checked per job.

The off arm uses EscapeMPCPlanner with escape disabled, which removes ECP detector/memory bookkeeping as well as its objective. Its actions are compared exactly with available original frozen-augmented records. The on arm retains the published StagnationMPCPlanner configuration and is compared to all corresponding completed ECP records. Common probe diagnostics remain active in both arms. The instruments only synchronize and time existing calls.

Each of the 35 task chunks runs both arms sequentially on the same physical GPU; order alternates by chunk. Five GPUs (0,1,3,5,6) each process seven paired chunks, one per shape. Other occupied GPUs are excluded. A short off-arm and triggered on-arm smoke check precede the full benchmark; smoke outcomes are excluded. No historical execution times are reused.

Measurements:
- Episode wall time: MPC loop including simulator prefix replay and common diagnostics, excluding checkpoint load, final batch evaluation and independent physics audit.
- Planning latency: existing synchronized MPC planner timing, including GD, graph capture, ECP memory build and objective diagnostic rollout.
- GD latency: synchronized subplanner calls, including graph capture; report mean and p95.
- Separate ECP detector and memory-build timers, plus shared feedback/diagnostic time.
- Actual replans and controls until success or the shared cap; success rate, rescues and regressions.
- Peak allocated and reserved PyTorch GPU memory per episode, plus model parameter count. No new learned parameters are introduced by ECP.
- Cold pipeline time is recorded separately, subtracting the instrument's model-hash audit time.

This is one measured pass over 350 paired tasks, not repeated timing trials. Five jobs may run concurrently on five GPUs and share host CPU resources. No isolated single-GPU latency, energy measurement or statistical timing-repeat confidence interval is claimed. Keep per-replan overhead distinct from total task cost: ECP can alter the number of replans and the success trajectory.

Outputs: `status.json` (queue), `summary.json` (completed pairs only, explicitly partial until all 35 finish), `RESULTS.md`, `per_task_cost.csv`, and per-job `cost_episodes.jsonl`, `cost_calls.jsonl`, configs, completion receipts, action arrays and replayed state arrays. Run `python analyze.py` to regenerate the summary. All code and configuration live in this isolated experiment directory; previous experiment results are unchanged.
