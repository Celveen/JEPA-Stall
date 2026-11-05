"""Check two-GPU dispatch and identical frozen actions; separate from formal evaluation."""
import concurrent.futures
import json
import sys
from pathlib import Path
import numpy as np
from evaluate_effect_b import GPUSlots

ROOT=Path(__file__).resolve().parents[1]
slots=GPUSlots(1)
assert slots.gpus==['4','6'],slots.gpus
base=ROOT/'eval_outputs/effect_b_20261103/multigpu_smoke'
ckpt=ROOT/'checkpoints/effect_b_20261103/B0'
def one(i):
    out=base/str(i);out.mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,'plan.py','--config-name','effect_ab_frozen_pushobj',
         f'ckpt_base_path={ckpt}','model_name=B0','model_epoch=3',
         f'eval_data_path={ROOT/"data/targets/EVAL50/chunk_00_I.pkl"}',
         'n_evals=1','planner.max_iter=2',f'hydra.run.dir={out}']
    result,gpu=slots.run(cmd,out)
    assert result.returncode==0,(gpu,result.returncode)
    rows=[json.loads(line) for line in (out/'per_iter_logs.json').read_text().splitlines() if line.strip()]
    assert rows and (len(rows)==2 or rows[-1]['success'])
    assert all(row.get('disp_pred',0)==0 and row.get('disp_enc',0)==0 for row in rows)
    return {'gpu':gpu,'rows':rows}
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:runs=list(pool.map(one,range(2)))
assert {run['gpu'] for run in runs}=={'4','6'}
assert len(runs[0]['rows'])==len(runs[1]['rows'])
for a,b in zip(runs[0]['rows'],runs[1]['rows']):
    for key in ['act','state']:
        np.testing.assert_array_equal(np.asarray(a[key]),np.asarray(b[key]))
    assert a['success']==b['success']
out={'passed':True,'gpus':['4','6'],'replans_per_gpu':len(runs[0]['rows']),
     'identical_actions_and_states':True,'frozen_parameters':True,'formal_result':False}
(ROOT/'reports/effect_b_20261103/multigpu_smoke.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out),flush=True)
