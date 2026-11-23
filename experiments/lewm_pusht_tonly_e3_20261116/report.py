"""Audit completed training/clean evaluation and package primary evidence; no experiments."""
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
TRAIN = REPO / 'eval_outputs/lewm_pusht_tonly_e3_20261116'
EVAL = REPO / 'eval_outputs/lewm_pusht_tonly_e3_clean50'
QUEUE = REPO / 'eval_outputs/lewm_pusht_tonly_e3_clean_queue'
MODEL = REPO / 'checkpoints/lewm_pusht_tonly_e3_4gpu_20261116'
OUT = REPO / 'reports/lewm_pusht_clean_20261116'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    training = json.loads((TRAIN / 'status.json').read_text())
    queue = json.loads((QUEUE / 'status.json').read_text())
    done = json.loads((EVAL / 'complete.json').read_text())
    frozen = json.loads((EVAL / 'frozen_model_check.json').read_text())
    cfg = yaml.safe_load((EVAL / 'config.yaml').read_text())
    assert training['status'] == queue['status'] == done['status'] == 'complete'
    assert not done['diagnostic_only'] and done['episodes'] == 50
    assert sha(MODEL / 'checkpoints/model_3.pth') == done['checkpoint_sha256'] == training['result']['checkpoint_sha256']
    assert sha(MODEL / 'hydra.yaml') == done['hydra_sha256'] == training['result']['hydra_sha256']
    assert sha(ROOT / 'evaluate_clean.py') == done['evaluation_runner_sha256']
    payload = torch.load(MODEL / 'checkpoints/model_3.pth', map_location='cpu', weights_only=True)
    assert payload['epoch'] == 3 and payload['step'] == 38277
    digest = hashlib.sha256()
    for name, tensor in payload['model'].items():
        assert torch.isfinite(tensor).all()
        digest.update(name.encode()); digest.update(tensor.contiguous().numpy().tobytes())
    assert frozen['unchanged'] and digest.hexdigest() == frozen['weights_sha256']
    assert cfg['n_evals'] == 50 and cfg['seed'] == 100 and cfg['segments_order'] == 'file'
    assert cfg['goal_source'] == 'segments' and cfg['ood_corruption'] is None and cfg['env_kwargs_override'] is None
    assert cfg['planner']['max_iter'] == 30 and cfg['planner']['n_taken_actions'] == 5
    assert cfg['planner']['sub_planner']['horizon'] == 25
    for name, expected in done['source_sha256'].items():
        assert sha(EVAL / 'source_snapshot' / name) == expected
    with np.load(EVAL / 'final_eval_results.npz') as raw:
        successes, lengths = raw['successes'], raw['action_len']
        assert successes.shape == lengths.shape == (50,)
        assert successes.sum() == done['successes'] == 36
        assert done['success_rate'] == float(successes.mean()) == .72
        assert np.array_equal(np.isfinite(lengths), successes)
        assert np.all((lengths[successes] >= 1) & (lengths[successes] <= 30))
        assert raw['actions'].shape == (50, 30, 10) and np.isfinite(raw['actions']).all()
        previous = REPO / 'reports/pusht_visual_v8/raw_baselines/research_snapshots/2026-11-08-pusht-gd/completed_runs/lewm/default/final_eval_results.npz'
        with np.load(previous) as reference:
            for key in ('state_0', 'state_g'):
                np.testing.assert_array_equal(raw[key], reference[key])
    metrics = [json.loads(line) for line in (MODEL / 'metrics.jsonl').read_text().splitlines()]
    validation = [row for row in metrics if 'val' in row]
    assert [row['step'] for row in validation] == [12759, 25518, 38277]
    OUT.mkdir(parents=True, exist_ok=True)
    summary = dict(method='LeWM', planner='CEM', condition='clean', successes=36, episodes=50,
        success_percent=72., train_seconds=training['ended_unix']-training['started_unix'],
        evaluation_seconds=queue['ended_unix']-queue['evaluation_started_unix'],
        validation_by_epoch=validation, checkpoint_sha256=done['checkpoint_sha256'],
        target_sha256=done['target_sha256'], paired_initial_and_goal_states_exact=True,
        frozen_weights_verified=True, raw_npz_sha256=sha(EVAL/'final_eval_results.npz'),
        checkpoint_provenance='Clean uses the newly trained T-only model; historical shifted-condition scores retain their archived checkpoint.')
    (OUT/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    with (OUT/'per_sample.csv').open('w') as stream:
        writer=csv.DictWriter(stream, fieldnames=['sample','env_seed','success','first_success_replan','controls_to_success'],lineterminator='\n')
        writer.writeheader()
        for i, success in enumerate(successes):
            writer.writerow(dict(sample=i,env_seed=100*i+1,success=bool(success),
                first_success_replan=int(lengths[i]) if success else '',controls_to_success=int(5*lengths[i]) if success else ''))
    for source,name in [(EVAL/'complete.json','evaluation_receipt.json'),(TRAIN/'training_result.json','checkpoint_receipt.json'),
                        (ROOT/'data_audit.json','data_audit.json'),(ROOT/'preflight.json','preflight.json'),
                        (MODEL/'hydra.yaml','training_config.yaml'),(EVAL/'config.yaml','evaluation_config.yaml')]:
        shutil.copyfile(source,OUT/name)
    members=[]
    for prefix,folder in [('evaluation',EVAL),('training_run',TRAIN)]:
        members.extend((p,f'{prefix}/{p.relative_to(folder)}') for p in sorted(folder.rglob('*')) if p.is_file())
    members.extend([(MODEL/'metrics.jsonl','training/metrics.jsonl'),(MODEL/'hydra.yaml','training/hydra.yaml'),
                    (QUEUE/'status.json','evaluation_queue/status.json'),(QUEUE/'evaluate.log','evaluation_queue/evaluate.log')])
    members.extend((p,f'implementation/{p.relative_to(ROOT)}') for p in sorted(ROOT.rglob('*'))
                   if p.is_file() and '__pycache__' not in p.parts)
    archive=OUT/'evidence.tar.gz'; manifest=[]
    with tarfile.open(archive,'w:gz') as tar:
        for path,name in members:
            content=path.read_bytes(); info=tarfile.TarInfo(name);info.size=len(content);info.mode=0o644
            tar.addfile(info,io.BytesIO(content))
            manifest.append(dict(path=name,bytes=len(content),sha256=hashlib.sha256(content).hexdigest()))
    with tarfile.open(archive,'r:gz') as tar:
        assert tar.getnames()==[row['path'] for row in manifest]
        for row in manifest:
            assert hashlib.sha256(tar.extractfile(row['path']).read()).hexdigest()==row['sha256']
    (OUT/'evidence_manifest.json').write_text(json.dumps(dict(archive=archive.name,sha256=sha(archive),members=manifest),indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
