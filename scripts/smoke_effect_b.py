"""Exercise every B arm with the production four-GPU loader/optimizer path."""
import json
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
cal=json.loads((ROOT/"reports/effect_b_20261103/calibration/complete.json").read_text())
for i,arm in enumerate(("B0","B1","B2","B3")):
    cmd=[sys.executable,"-m","torch.distributed.run","--nproc_per_node=4",f"--master_port={29631+i}",
        "scripts/train_effect_ab.py","--config-name",f"train_effect_{arm.lower()}",
        "effect_smoke_steps=2",f"effect_expected_init={ROOT/'reports/effect_b_20261103/calibration/initialization.json'}",
        f"hydra.run.dir={ROOT/'checkpoints/effect_b_20261103/smoke'/arm}"]
    if arm in ("B2","B3"):
        mode="idm" if arm=="B2" else "effect"
        cmd.append(f"model.effect_lambda={cal[mode]['lambda']}")
    print("START",arm,flush=True)
    subprocess.run(cmd,cwd=ROOT,check=True)
    print("PASS",arm,flush=True)
records={arm:json.loads((ROOT/"checkpoints/effect_b_20261103/smoke"/arm/"smoke.json").read_text()) for arm in ("B0","B1","B2","B3")}
initial={arm:json.loads((ROOT/"checkpoints/effect_b_20261103/smoke"/arm/"experiment_initialization.json").read_text()) for arm in records}
for arm in records:
    assert initial[arm]["modules"]==initial["B0"]["modules"]
    assert initial[arm]["samplers"]==initial["B0"]["samplers"]
    assert initial[arm]["dataset_lengths"]==initial["B0"]["dataset_lengths"]
(ROOT/"reports/effect_b_20261103/distributed_smoke.json").write_text(json.dumps({"arms":records,"initializations":initial,"passed":True},indent=2)+"\n")
