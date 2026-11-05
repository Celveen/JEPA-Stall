"""Two-replan plumbing check only; not included in performance results."""
import json
import os
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
results={}
for arm in ("B0","B1","B2","B3"):
    ckpt=ROOT/"checkpoints/effect_b_20261103/smoke"/arm
    link=ckpt/"checkpoints/model_smoke.pth"
    if not link.exists():os.link(ckpt/"checkpoints/smoke.pth",link)
    out=ROOT/"eval_outputs/effect_b_20261103/planning_smoke"/arm
    out.mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,"plan.py","--config-name","effect_ab_frozen_pushobj",
         f"ckpt_base_path={ckpt}",f"model_name={arm}","model_epoch=smoke",
         f"eval_data_path={ROOT/'data/targets/EVAL50/chunk_00_I.pkl'}","n_evals=1",
         "planner.max_iter=2",f"hydra.run.dir={out}"]
    with (out/"console.log").open("w") as log:
        subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    rows=[json.loads(l) for l in (out/"per_iter_logs.json").open() if l.strip()]
    assert len(rows)==2 or rows[-1]["success"]
    assert all(r.get("disp_pred",0)==0 and r.get("disp_enc",0)==0 for r in rows)
    results[arm]={"replans":len(rows),"frozen_weights_verified":True,
                  "final_artifact_exists":(out/"final_eval_results.npz").exists()}
    print("PLAN PASS",arm,flush=True)
(ROOT/"reports/effect_b_20261103/planning_smoke.json").write_text(json.dumps({"passed":True,"arms":results},indent=2)+"\n")
