#!/usr/bin/env python3
"""Evaluate existing PushT weights with paired ECP-on/off controls; never train.

Default mode only writes/prints a plan. Use --execute after checking GPU availability.
Completed jobs are resumed by exact content fingerprint; failed attempts stay intact.
"""
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import queue
import shlex
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

ASSETS = {
    "checkpoints/pusht_visual_shift/checkpoints/model_latest.pth": "9098082f1d40bfa7b4f4884571db30a0fed4a0aa26ad7dab8db11415437b8b32",
    "checkpoints/pushobj_T16k_bias_e3_seed0/checkpoints/model_3.pth": "2aceb5a512b6d3b0f3e38507cdad35110734e1105d73a09bb7049b9f4e93ebe6",
}
TARGETS = {
    "external50": ("targets_T.pkl", "file", "1950b39e5d3ba13b30c14a435a400d84cd3fbe30138cdfed55e98b202d76277b"),
    "legacy_pool": ("data/pushobj_eval/val_T/plan_targets.pkl", "random", "f08486da48f7a6961e3d0035b48d9cd13515c0322bbacaf9dd9f973bf7bad624"),
}
MODELS = {
    "released": ("checkpoints/pusht_visual_shift", "latest"),
    "augmented": ("checkpoints/pushobj_T16k_bias_e3_seed0", "3"),
}
CONDITIONS = {
    "clean": [],
    "redBlock": ["++env_kwargs_override.color=Red"],
    "redAnchor": ["++env_kwargs_override.goal_color=Red"],
    "redAgent": ["++env_kwargs_override.agent_color=Red"],
    "blur": ["ood_corruption=blur", "ood_level=2.0"],
    "snp": ["ood_corruption=snp1", "ood_level=0.01"],
    "dark": ["ood_corruption=dark", "ood_level=0.9"],
}
CORE_FILES = [
    "plan.py", "conf/escape_plan_gd_pushobj.yaml", "planning/stagnation.py",
    "planning/escape.py", "planning/adajepa_probe.py", "planning/adajepa.py",
    "planning/adajepa_mpc.py", "planning/mpc.py", "planning/gd.py",
    "planning/evaluator.py", "planning/objectives.py", "planning/image_corruption.py",
    "env/pusht/pusht_env.py", "env/pusht/pusht_wrapper.py",
    "checkpoints/pusht_visual_shift/hydra.yaml",
    "checkpoints/pushobj_T16k_bias_e3_seed0/hydra.yaml",
]
LOCK = threading.Lock()
STOP = threading.Event()
PROCESSES = set()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    tmp.replace(path)


def command(args, job, out):
    ckpt, epoch = MODELS[job["weights"]]
    cmd = [args.python, str(Path(__file__).with_name("evaluate_pusht_visual.py")),
           "--repo", str(args.repo), "--config-name", "escape_plan_gd_pushobj",
           f"ckpt_base_path={args.repo / ckpt}", f"model_epoch={epoch}",
           f"eval_data_path={args.target_path}",
           f"seed={job['seed']}", f"n_evals={args.n_evals}", f"segments_order={args.target_order}",
           "planner.max_iter=30", "+wandb_logging=false", "decode_for_viz=false",
           f"hydra.run.dir={out}", "planner._target_=planning.stagnation.StagnationMPCPlanner",
           f"+planner.probe.mode={'none' if job['base']=='frozen' else 'grad'}",
           "+planner.stag.space=latent", "+planner.stag.start=10", "+planner.stag.window=3",
           "+planner.stag.gamma_lo=0.05", "+planner.stag.gamma_hi=0.15",
           "+planner.stag.beta=0.25", "+planner.stag.alpha=1", "+planner.stag.base=2",
           "+planner.stag.boost_replans=2",
           f"+planner.stag.kappa={2.0 if job['ecp'] else 0.0}",
           f"+planner.stag.prop_boost={10.0 if job['ecp'] else 1.0}"]
    return cmd + CONDITIONS[job["condition"]]


def make_jobs(args, code_hashes):
    jobs = []
    # Clean first, then condition-major: each comparison becomes available together.
    for condition in args.conditions:
        for seed in args.seeds:
            for weights in args.weights:
                for base in ("frozen", "adajepa"):
                    for ecp in (False, True):
                        job = dict(condition=condition, seed=seed, weights=weights, base=base, ecp=ecp)
                        job["id"] = f"{weights}/{condition}/seed{seed}/{base}{'_ecp' if ecp else ''}"
                        job["fingerprint"] = hashlib.sha256(json.dumps({
                            "job": job, "n_evals": args.n_evals, "assets": ASSETS,
                            "target_profile": args.target_profile, "target_sha256": args.target_sha256,
                            "code": code_hashes, "argv": command(args, job, Path("<OUTPUT>")),
                        }, sort_keys=True).encode()).hexdigest()
                        jobs.append(job)
    return jobs


def valid_completion(job_dir, job):
    p = job_dir / "completion.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    if d.get("fingerprint") != job["fingerprint"]:
        raise RuntimeError(f"Completed output has a different protocol: {job_dir}; use a new --output")
    attempt = job_dir / d["attempt"]
    if not (attempt / "final_eval_results.npz").exists() or not (attempt / "logs.json").exists():
        raise RuntimeError(f"Incomplete recorded completion: {job_dir}")
    return d


def available_gpus(requested):
    text = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"
    ], text=True)
    rows = {int(r[0]): (int(r[1]), int(r[2])) for r in csv.reader(text.splitlines())}
    selected = sorted(rows) if requested == "auto" else [int(x) for x in requested.split(",")]
    if len(selected) != len(set(selected)):
        raise RuntimeError("Each GPU may appear only once; this launcher runs one job per free GPU")
    free, busy = [], []
    for gpu in selected:
        if gpu not in rows:
            raise RuntimeError(f"Unknown GPU {gpu}")
        mem, util = rows[gpu]
        (free if mem < 1024 and util < 5 else busy).append(gpu)
    if requested != "auto" and busy:
        raise RuntimeError(f"Requested GPUs are occupied: {busy}; refusing to start")
    if not free:
        raise RuntimeError("No idle GPU (<1GiB used, <5% utilization); no job was started")
    return free, rows


def run_job(args, job, gpu):
    job_dir = args.output / job["id"]
    job_dir.mkdir(parents=True, exist_ok=True)
    with (job_dir / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if valid_completion(job_dir, job):
            return True
        attempts = job_dir / "attempts"
        attempts.mkdir(exist_ok=True)
        number = max([int(p.name) for p in attempts.iterdir() if p.name.isdigit()] + [0]) + 1
        out = attempts / f"{number:03d}"
        out.mkdir()
        cmd = command(args, job, out)
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "WANDB_MODE": "offline",
               "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1",
               "SDL_VIDEODRIVER": "dummy", "PYTHONUNBUFFERED": "1"}
        started = time.monotonic()
        record = {**job, "gpu": gpu, "command": cmd, "started_utc": now(), "status": "running"}
        dump(out / "run_metadata.json", record)
        with LOCK:
            print(f"START GPU={gpu} {job['id']}", flush=True)
        with (out / "console.log").open("w") as log:
            proc = subprocess.Popen(cmd, cwd=args.repo, env=env, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            with LOCK:
                PROCESSES.add(proc)
            rc = proc.wait()
            with LOCK:
                PROCESSES.discard(proc)
        metrics = []
        logs = out / "logs.json"
        if logs.exists():
            for line in logs.read_text().splitlines():
                try:
                    r = json.loads(line)
                    if "final_eval/success_rate" in r:
                        metrics.append(r)
                except json.JSONDecodeError:
                    pass
        ok = rc == 0 and bool(metrics) and (out / "final_eval_results.npz").exists()
        record.update(status="complete" if ok else "failed", exit_code=rc, ended_utc=now(),
                      wall_seconds=time.monotonic() - started)
        dump(out / "run_metadata.json", record)
        if ok:
            dump(job_dir / "completion.json", {
                "fingerprint": job["fingerprint"], "attempt": str(out.relative_to(job_dir)),
                "n": args.n_evals, "success_rate": metrics[-1]["final_eval/success_rate"],
                "wall_seconds": record["wall_seconds"], "completed_utc": now(),
            })
        with LOCK:
            print(f"{'DONE' if ok else 'FAIL'} {job['id']} {record['wall_seconds']:.0f}s", flush=True)
        return ok


def write_summary(args, jobs):
    rows = []
    for job in jobs:
        done = valid_completion(args.output / job["id"], job)
        rows.append({**job, "status": "complete" if done else "pending",
                     "n": done["n"] if done else None,
                     "success_rate": done["success_rate"] if done else None,
                     "result_path": str(args.output / job["id"] / done["attempt"]) if done else None})
    dump(args.output / "results.json", rows)
    with (args.output / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--seeds", default="100", help="100 primary; 100,200,300 optional extension")
    p.add_argument("--target-profile", choices=TARGETS, default="external50")
    p.add_argument("--conditions", default=",".join(CONDITIONS))
    p.add_argument("--weights", default=",".join(MODELS))
    p.add_argument("--n-evals", type=int, default=50, help="Use a separate output for the 2-episode smoke")
    p.add_argument("--gpus", default="auto", help="Auto selects only currently idle GPUs; or comma-separated indices")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--validate-config", action="store_true", help="Hydra --cfg job only: no model/simulator run")
    args = p.parse_args()
    args.repo, args.output = args.repo.resolve(), args.output.resolve()
    relative, args.target_order, args.target_sha256 = TARGETS[args.target_profile]
    args.target_path = (Path(__file__).resolve().parent if args.target_profile == "external50" else args.repo) / relative
    if sha(args.target_path) != args.target_sha256:
        raise RuntimeError(f"Target hash mismatch: {args.target_path}")
    args.seeds = [int(x) for x in args.seeds.split(",")]
    args.conditions, args.weights = args.conditions.split(","), args.weights.split(",")
    if not set(args.conditions) <= set(CONDITIONS) or not set(args.weights) <= set(MODELS):
        p.error("Unknown condition or weights")
    if not 1 <= args.n_evals <= 50 or not args.seeds:
        p.error("n-evals must be between 1 and 50; seeds must be nonempty")
    args.output.mkdir(parents=True, exist_ok=True)
    for relative, expected in ASSETS.items():
        actual = sha(args.repo / relative)
        if actual != expected:
            raise RuntimeError(f"Asset hash differs: {relative}: {actual}")
    code_hashes = {path: sha(args.repo / path) for path in CORE_FILES}
    code_hashes["runner"] = sha(__file__)
    for helper in ("evaluate_pusht_visual.py", "paired_noise.py"):
        code_hashes[helper] = sha(Path(__file__).with_name(helper))
    jobs = make_jobs(args, code_hashes)
    manifest = {"created_utc": now(), "assets": ASSETS, "code": code_hashes,
                "jobs": jobs, "n_jobs": len(jobs), "n_evals": args.n_evals,
                "episode_runs": len(jobs) * args.n_evals, "max_replans": 30,
                "seeds": args.seeds, "repo": str(args.repo),
                "target_profile": args.target_profile, "target_path": str(args.target_path),
                "target_sha256": args.target_sha256, "target_order": args.target_order,
                "baseline": "ECP-off control with identical per-episode RNG and logging",
                "snp_noise": "Fresh noise keyed by episode IDs and evaluation call index; distribution and severity unchanged",
                "training": False, "executed": args.execute}
    # Validate existing completions before updating any summary.
    rows = write_summary(args, jobs)
    dump(args.output / "manifest.json", manifest)
    pending = [j for j, r in zip(jobs, rows) if r["status"] == "pending"]
    print(f"{len(jobs)} jobs; {len(pending)} pending; {len(jobs)*args.n_evals} episode-runs")
    (args.output / "commands.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + "\n".join(
        shlex.join(command(args, j, args.output / j["id"] / "manual")) for j in jobs) + "\n")
    if args.validate_config:
        # Validate every override combination, without executing the decorated main.
        from hydra import compose, initialize_config_dir
        for j in jobs:
            overrides = command(args, j, args.output / "config_only")[6:]
            with initialize_config_dir(config_dir=str(args.repo / "conf"), version_base=None):
                compose(config_name="escape_plan_gd_pushobj", overrides=overrides)
        print(f"Validated all {len(jobs)} Hydra configurations; no experiment executed")
    if not args.execute:
        print("Plan only. Add --execute to run on idle GPUs.")
        return
    if not pending:
        return
    gpus, gpu_snapshot = available_gpus(args.gpus)
    dump(args.output / "gpu_snapshot.json", {"utc": now(), "gpus": gpu_snapshot, "selected": gpus})
    jobs_q = queue.Queue()
    for job in pending:
        jobs_q.put(job)
    outcomes = []
    def stop_running(signum, frame):
        STOP.set()
        # Terminate only subprocesses launched by this invocation; preserve all logs.
        for proc in list(PROCESSES):
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        print("Stopping this launcher; completed outputs remain resumable.", flush=True)
    signal.signal(signal.SIGINT, stop_running)
    signal.signal(signal.SIGTERM, stop_running)
    def worker(gpu):
        while not STOP.is_set():
            try:
                job = jobs_q.get_nowait()
            except queue.Empty:
                return
            try:
                ok = run_job(args, job, gpu)
            except Exception as exc:
                ok = False
                with LOCK:
                    print(f"FAIL {job['id']}: {exc}", flush=True)
            with LOCK:
                outcomes.append(ok)
                write_summary(args, jobs)
            jobs_q.task_done()
    threads = [threading.Thread(target=worker, args=(gpu,)) for gpu in gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    print(f"Finished: {sum(outcomes)} successful, {len(outcomes)-sum(outcomes)} failed")
    if STOP.is_set():
        sys.exit(130)
    if not all(outcomes):
        sys.exit(1)


if __name__ == "__main__":
    main()
