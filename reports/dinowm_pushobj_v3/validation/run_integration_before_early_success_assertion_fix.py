"""Real-simulator checks of the shared 6 x 25-control DINO-WM protocol."""
import hashlib
import json
import os
import pickle
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

R = Path('/home/appuser/workspace/project/multimodal-test/JEPA-Stall')
B = Path(__file__).resolve().parent
V = B / 'validation'
V.mkdir(exist_ok=True)
PY = '/home/appuser/anaconda3/envs/ts/bin/python'
sys.path.insert(0, str(R))
from physical_stagnation import history_before_endpoint

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

# An endpoint at control 50 must compare against control 35, not control -25.
for endpoint in [25, 50, 75, 100, 125, 150]:
    observations = {'proprio': np.arange(endpoint + 1)[None, :, None],
                    'visual': np.arange(endpoint + 1)[None, :, None, None]}
    history = history_before_endpoint(observations, window=3, stride=5)
    for key in observations:
        assert [int(x[key].reshape(-1)[0]) for x in history] == [endpoint - 15, endpoint - 10, endpoint - 5]

source = [R/'data/targets/EVAL50/chunk_00_I.pkl', R/'data/targets/EVAL50/chunk_30_T.pkl']
loaded = [pickle.loads(p.read_bytes()) for p in source]
assert loaded[0]['traj_len'] == loaded[1]['traj_len']
merged = {**loaded[0], 'segments': [x['segments'][0] for x in loaded]}
target = V/'paired_I_T_targets.pkl'
target.write_bytes(pickle.dumps(merged))
(V/'target_provenance.json').write_text(json.dumps({
    'kind': 'integration_only',
    'samples': [{'source': str(p), 'source_sha256': sha(p), 'index': 0} for p in source]
}, indent=2)+'\n')
jobs = {0: ('ordinary', False), 1: ('ordinary', True),
        3: ('augmented', False), 5: ('augmented', True)}

def worker(gpu):
    arm, ecp = jobs[gpu]
    out = V/f'{arm}_bf16_{"ecp" if ecp else "native"}'
    out.mkdir(exist_ok=True)
    cmd = [PY, '-u', str(B/'evaluate.py'), '--repo', str(R), '--checkpoint',
           str(R/'checkpoints/dinowm_pushobj_v1'/arm), '--epoch', '3', '--n-evals', '2',
           '--targets', str(target), '--output', str(out), '--precision', 'bf16',
           '--candidate-batch', '64', '--validation-replans', '6']
    if ecp:
        cmd.extend(['--ecp', '--forced-trigger'])
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'OMP_NUM_THREADS': '2',
           'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '2', 'WANDB_MODE': 'disabled',
           'SDL_VIDEODRIVER': 'dummy', 'SDL_AUDIODRIVER': 'dummy'}
    with (out/'run.log').open('w') as f:
        proc = subprocess.Popen(cmd, cwd=R, env=env, stdout=f, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, start_new_session=True)
        (out/'launcher.json').write_text(json.dumps({
            'pid': proc.pid, 'gpu': gpu, 'command': cmd,
            'started_utc': datetime.now(timezone.utc).isoformat()
        }, indent=2)+'\n')
        code = proc.wait()
    assert code == 0, f'{out}/run.log: exit {code}'
    print(json.dumps({'complete': str(out), 'gpu': gpu}), flush=True)

if '--check-only' not in sys.argv:
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(worker, jobs))

checks = []
for arm in ['ordinary', 'augmented']:
    native = V/f'{arm}_bf16_native'
    ecp = V/f'{arm}_bf16_ecp'
    aa = np.load(native/'final_eval_results.npz')
    bb = np.load(ecp/'final_eval_results.npz')
    ca = json.loads((native/'complete.json').read_text())
    cb = json.loads((ecp/'complete.json').read_text())
    for key in ['state_0', 'state_g']:
        np.testing.assert_array_equal(aa[key], bb[key])
    # Two complete plans precede activation: 10 model steps = 50 controls.
    np.testing.assert_array_equal(aa['actions'][:, :10], bb['actions'][:, :10])
    rows = [json.loads(x) for x in (ecp/'per_iter_logs.json').read_text().splitlines()]
    episodes = [json.loads(x) for x in (ecp/'stag_episodes.jsonl').read_text().splitlines()]
    assert all(x['first_trigger'] == 2 for x in episodes)
    assert any('stag_pen' in x for x in rows)
    assert all(np.isfinite(x['stag_pen']) and x['stag_pen'] >= 0 for x in rows if 'stag_pen' in x)
    assert any(x.get('stag_boost', False) for x in rows)
    for row in rows:
        assert row['elapsed_environment_controls'] in [25, 50, 75, 100, 125, 150]
        assert row['executed_environment_controls'] == 25
        assert row['stag_window_environment_controls'] == 15
    assert ca['frozen_weights_unchanged'] and cb['frozen_weights_unchanged']
    assert ca['model_sha256'] == cb['model_sha256']
    assert np.isfinite(aa['actions']).all() and np.isfinite(bb['actions']).all()
    for data in [aa, bb]:
        assert data['actions'].shape[1] <= 30
    for completion in [ca, cb]:
        assert (completion['cem_candidates'], completion['cem_iterations'], completion['cem_elites']) == (200, 10, 30)
        assert completion['is_validation']
        assert completion['max_mpc_replans'] == 6
        assert completion['executed_physical_steps_per_replan'] == 25
        assert completion['executed_model_steps_per_replan'] == 5
        assert completion['max_environment_controls'] == 150
    checks.append({'arm': arm, 'precision': 'bf16', 'paired_targets_equal': True,
                   'pretrigger_50_controls_identical': True, 'forced_trigger_replan_index': 2,
                   'penalty_evaluated': True, 'proprio_boost_exercised': True,
                   'detector_window_controls': 15, 'weights_unchanged': True,
                   'selected_plan_positive_penalty_records': sum(x.get('stag_pen', 0) > 0 for x in rows),
                   'native_seconds': ca['wall_seconds'], 'ecp_seconds': cb['wall_seconds']})

# These numerical kernels are unchanged. Record prior validation as prior evidence.
previous = R/'reports/dinowm_pushobj_v2'
assert sha(B/'cem_memory.py') == sha(previous/'cem_memory.py')
evidence = []
for path in [previous/'validation/ordinary_precision.json', previous/'validation/augmented_precision.json',
             previous/'memory_validation.json']:
    evidence.append({'path': str(path), 'sha256': sha(path), 'validation': json.loads(path.read_text())})
for entry in evidence[:2]:
    x = entry['validation']
    assert x['frozen_weights_unchanged'] and all(y['actions_finite'] for y in x['rows'])
memory = evidence[-1]['validation']
assert memory['passed'] and memory['stay_penalty'] > 0 and memory['goal_penalty'] == 0
summary = {
    'integration_passed': True, 'bf16_implementation_approved': True,
    'cross_precision_actions_equivalent': False, 'closed_loop_accuracy_equivalence_established': False,
    'scope': 'New real-simulator cadence checks: I/T starts, both checkpoints, native/forced ECP, 200x10 CEM, up to 6 plans executing 25 controls each. Integration-only outcomes are excluded from benchmark totals. FP32 complete-task spot checks are separate.',
    'history_sampling_test_passed': True, 'checks': checks,
    'reused_kernel_validation': evidence,
    'reused_evidence_scope': 'Unchanged candidate rollout, objective and penalty kernels; original precision/memory checks were not rerun. Execution cadence is validated by the new simulator checks above.',
    'source_sha256': {name: sha(B/name) for name in ['evaluate.py', 'cem_memory.py', 'physical_stagnation.py']},
    'protocol_sha256': sha(R/'conf/dinowm_pushobj_official_cadence.yaml'),
    'updated_utc': datetime.now(timezone.utc).isoformat()
}
(B/'validation_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
print(json.dumps(summary, indent=2), flush=True)
