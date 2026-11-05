"""Durable B0->B3 training with one separate frozen-evaluation GPU queue."""
import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/"reports/effect_b_20261103"
ARMS=("B0","B1","B2","B3")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train-gpus",default="0,1,3,5")
    ap.add_argument("--eval-gpus",default="6")
    a=ap.parse_args()
    if set(a.train_gpus.split(","))&set(a.eval_gpus.split(",")):raise ValueError("Training and evaluation cards overlap")
    smoke=json.loads((REPORTS/"distributed_smoke.json").read_text())
    assert smoke["passed"]
    assert json.loads((REPORTS/"planning_smoke.json").read_text())["passed"]
    cal=json.loads((REPORTS/"calibration/complete.json").read_text())
    assert cal["idm"]["batches"]==cal["effect"]["batches"]==100
    assert (ROOT/"data/pushobj_bias_16k/effect_cache_v1/complete.json").exists()
    candidates=ROOT/"eval_outputs/effect_b_20261103/candidates"
    assert (candidates/"scores_existing_ours.json").exists() and (candidates/"scores_existing_official.json").exists()
    if (REPORTS/"pipeline_status.json").exists():raise RuntimeError("Pipeline state already exists; inspect before resuming")
    state={"status":"running","pid":os.getpid(),"environment":"PushObj","max_replans":30,
        "started_at":time.time(),"train_gpus":a.train_gpus,"eval_gpus":a.eval_gpus,
        "training":{x:"pending" for x in ARMS},"evaluation":{x:"pending" for x in ARMS},"errors":[]}
    lock=threading.Lock()
    def save():
        tmp=REPORTS/"pipeline_status.json.tmp"
        tmp.write_text(json.dumps(state,indent=2)+"\n");tmp.replace(REPORTS/"pipeline_status.json")
    def update(section,key,value):
        with lock:state[section][key]=value;save()
    save()
    def wait_empty(gpus):
        while True:
            text=subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.used","--format=csv,noheader,nounits"],text=True)
            mem={s.split(",")[0].strip():int(s.split(",")[1]) for s in text.splitlines()}
            if all(mem[g]<128 for g in gpus.split(",")):return
            time.sleep(15)
    def run_guard(name,gpus,hours,cmd):
        wait_empty(gpus)
        command=[sys.executable,"-u","scripts/effect_budget.py","--name",name,"--gpus",gpus,
                 "--max-gpu-hours",str(hours),"--",*cmd]
        with (REPORTS/f"{name}.log").open("w") as log:
            return subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
    tasks=queue.Queue()
    def evaluation_worker():
        while True:
            arm=tasks.get()
            if arm is None:return
            update("evaluation",arm,"running")
            code=run_guard(f"eval_{arm}",a.eval_gpus,6,[sys.executable,"-u","scripts/evaluate_effect_b.py","--arm",arm])
            update("evaluation",arm,"complete" if code==0 else f"failed:{code}")
            if code:
                with lock:state["errors"].append(f"Evaluation {arm} exit {code}");save()
    thread=threading.Thread(target=evaluation_worker)
    thread.start()
    try:
        for arm in ARMS:
            update("training",arm,"running")
            cmd=[sys.executable,"-m","torch.distributed.run","--nproc_per_node=4","--master_port=29641",
                "scripts/train_effect_ab.py","--config-name",f"train_effect_{arm.lower()}",
                f"effect_expected_init={REPORTS/'calibration/initialization.json'}"]
            if arm in ("B2","B3"):cmd.append(f"model.effect_lambda={cal['idm' if arm=='B2' else 'effect']['lambda']}")
            code=run_guard(f"train_{arm}",a.train_gpus,18,cmd)
            update("training",arm,"complete" if code==0 else f"failed:{code}")
            if code:
                with lock:state["errors"].append(f"Training {arm} exit {code}");save()
                break
            tasks.put(arm)
    finally:
        tasks.put(None);thread.join()
        with lock:
            state["status"]="complete" if all(x=="complete" for s in ("training","evaluation") for x in state[s].values()) else "incomplete"
            state["ended_at"]=time.time();save()
    print(json.dumps(state,indent=2),flush=True)
    raise SystemExit(0 if state["status"]=="complete" else 1)


if __name__=="__main__":main()
