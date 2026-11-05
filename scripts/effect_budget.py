"""Shared hard GPU-hour guard for A/B commands, including failed attempts.

Each guard owns one process group. All guards read the same locked ledger and
charge allocated GPU count x wall time, not utilization. At the global 100-hour
cap every running guard terminates its own group. Never launch another GPU job
for this experiment outside this guard.
"""
import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT/"reports/effect_b_20261103/gpu_budget.json"
RESOURCES = LEDGER.parent/"runtime_resources.json"


def allocation(name, requested):
    """Read per-job overrides at launch, including for an existing supervisor."""
    config = json.loads(RESOURCES.read_text()) if RESOURCES.exists() else {}
    override = config.get("jobs", {}).get(name, {})
    gpus = [str(g) for g in override.get("gpus", requested.split(","))]
    if not gpus or len(set(gpus)) != len(gpus) or not all(g.isdigit() for g in gpus):
        raise ValueError(f"Invalid GPU allocation: {gpus}")
    return gpus, override.get("allowed_existing_processes", {})


def check_occupancy(gpus, allowed):
    text = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits"], text=True)
    cards = {fields[0].strip(): (fields[1].strip(), int(fields[2]))
             for fields in (line.split(",") for line in text.splitlines())}
    text = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"], text=True)
    processes = {}
    for line in text.splitlines():
        uuid, pid = [part.strip() for part in line.split(",")]
        processes.setdefault(uuid, []).append(int(pid))
    accepted = {}
    for gpu in gpus:
        uuid, memory = cards[gpu]
        pids = processes.get(uuid, [])
        if memory > 128 and not pids:
            raise RuntimeError(f"GPU {gpu} has unexplained memory use: {memory} MiB")
        permitted = {int(item["pid"]): item for item in allowed.get(gpu, [])}
        for pid in pids:
            if pid not in permitted:
                raise RuntimeError(f"GPU {gpu} occupied by unapproved PID {pid}")
            proc = Path("/proc")/str(pid)
            start = (proc/"stat").read_text().rsplit(")", 1)[1].split()[19]
            digest = hashlib.sha256((proc/"cmdline").read_bytes()).hexdigest()
            expected = permitted[pid]
            if start != str(expected["start_ticks"]) or digest != expected["cmdline_sha256"]:
                raise RuntimeError(f"GPU {gpu} PID {pid} identity changed")
        if pids:
            accepted[gpu] = {"memory_mib_before_launch": memory, "processes": [permitted[p] for p in pids]}
    return accepted


@contextlib.contextmanager
def locked():
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = json.loads(LEDGER.read_text()) if LEDGER.exists() else {"limit_gpu_hours":100.0,"jobs":[]}
        yield state
        temp = LEDGER.with_suffix(".json.tmp")
        temp.write_text(json.dumps(state,indent=2)+"\n")
        temp.replace(LEDGER)


def used(state):
    now = time.time()
    return sum((j.get("ended_at",now)-j["started_at"])*len(j["gpus"])/3600 for j in state["jobs"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--gpus", required=True)
    ap.add_argument("--max-gpu-hours", type=float, required=True)
    ap.add_argument("command", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.command[1:] if a.command and a.command[0]=="--" else a.command
    gpus, allowed = allocation(a.name, a.gpus)
    accepted = check_occupancy(gpus, allowed)
    start = time.time()
    with locked() as state:
        if any(j["name"]==a.name for j in state["jobs"]):
            raise RuntimeError("Job name already recorded; use a new attempt name")
        if used(state) >= state["limit_gpu_hours"]:
            raise RuntimeError("Global GPU budget exhausted")
        running = {g for j in state["jobs"] if "ended_at" not in j for g in j["gpus"]}
        if running.intersection(gpus):
            raise RuntimeError("Requested GPU already allocated to another experiment job")
        state["jobs"].append({"name":a.name,"gpus":gpus,"command":cmd,
            "requested_gpus":a.gpus,"shared_existing_processes":accepted,
            "started_at":start,"guard_pid":os.getpid(),"max_gpu_hours":a.max_gpu_hours,"status":"running"})
    env = {**os.environ,"CUDA_VISIBLE_DEVICES":",".join(gpus),"WANDB_MODE":"offline",
           "OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1",
           "SDL_VIDEODRIVER":"dummy","PYTHONUNBUFFERED":"1",
           "NCCL_P2P_DISABLE":"1","NCCL_IB_DISABLE":"1"}
    child = None
    reason = "completed"
    code = 1
    stopped = False
    def stop(signum, frame):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)
    try:
        child = subprocess.Popen(cmd,cwd=ROOT,env=env,start_new_session=True)
        with locked() as state:
            next(j for j in state["jobs"] if j["name"]==a.name)["child_pid"] = child.pid
        while child.poll() is None:
            with locked() as state:
                total = used(state)
                state["charged_gpu_hours_as_of_last_check"] = total
                state["last_check"] = time.time()
                limit = state["limit_gpu_hours"]
            job_hours = (time.time()-start)*len(gpus)/3600
            # Leave time for orderly termination before the nominal hard cap.
            if stopped or total >= limit-0.01 or job_hours >= a.max_gpu_hours-0.01:
                reason = "interrupted" if stopped else "budget_limit"
                os.killpg(child.pid,signal.SIGTERM)
                try: child.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL)
                break
            time.sleep(2)
        code = child.wait()
        if code and reason=="completed": reason="failed"
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGKILL)
            child.wait()
        with locked() as state:
            job = next(j for j in state["jobs"] if j["name"]==a.name)
            job.update(ended_at=time.time(),exit_code=code,status=reason)
            job["gpu_hours"] = (job["ended_at"]-start)*len(gpus)/3600
            state["charged_gpu_hours_as_of_last_check"] = used(state)
        print(json.dumps(job),flush=True)
    raise SystemExit(code if reason=="completed" else code or 124)


if __name__ == "__main__": main()
