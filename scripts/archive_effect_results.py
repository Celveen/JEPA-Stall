"""Validate and archive the completed A/B screen without running GPU experiments."""
import argparse
import concurrent.futures
import csv
import gzip
import hashlib
import json
import shutil
import tarfile
from pathlib import Path

import numpy as np
import yaml

ARMS = ('B0', 'B1', 'B2', 'B3')


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def validate(root):
    report = root/'reports/effect_b_20261103'
    status = read(report/'pipeline_status.json')
    assert status['status'] == 'complete' and not status['errors']
    budget = read(report/'gpu_budget.json')
    assert all('ended_at' in job for job in budget['jobs'])
    gpu_hours = sum(job['gpu_hours'] for job in budget['jobs'])
    assert gpu_hours < budget['limit_gpu_hours'] == 100
    blocks = read(root/'data/targets/EVAL50/manifest.json')['blocks']
    assert len(blocks) == 35 and sum(b['n'] for b in blocks) == 350
    candidate_path = root/'eval_outputs/effect_b_20261103/candidates/candidates.npz'
    candidate_sha = digest(candidate_path)
    summaries, init_reference, target_reference = {}, None, {}
    planner_reference = None
    all_episodes = {}
    for arm in ARMS:
        result = read(report/f'eval_{arm}.json')
        episodes = {e['key']: e for e in result['episodes']}
        assert len(episodes) == result['tasks'] == 350
        train = root/'checkpoints/effect_b_20261103'/arm
        init = read(train/'experiment_initialization.json')
        shared_init = {k: init[k] for k in ('modules', 'samplers', 'dataset_lengths', 'world_size', 'effective_batch_size')}
        if init_reference is None:
            init_reference = shared_init
        assert shared_init == init_reference
        assert init['effective_batch_size'] == 64 and init['world_size'] == 4
        metrics = lines(train/'metrics.jsonl')
        assert [m['epoch'] for m in metrics] == [1, 2, 3]
        for b in blocks:
            out = root/'eval_outputs/effect_b_20261103'/arm/'frozen30'/b['chunk']
            final = np.load(out/'final_eval_results.npz')
            successes = final['successes']
            assert successes.shape == (b['n'],)
            target = (final['state_0'], final['state_g'])
            if arm == 'B0':
                target_reference[b['chunk']] = target
            else:
                for actual, expected in zip(target, target_reference[b['chunk']]):
                    np.testing.assert_array_equal(actual, expected)
            config = yaml.safe_load((out/'.hydra/config.yaml').read_text())
            planner = config['planner']
            if planner_reference is None:
                planner_reference = planner
            assert planner == planner_reference
            assert config['seed'] == 100 and config['model_epoch'] == 3
            assert Path(config['ckpt_base_path']).name == arm
            assert Path(config['eval_data_path']).name == b['chunk']+'.pkl'
            assert planner['max_iter'] == 30 and planner['n_taken_actions'] == 5
            assert planner['probe']['mode'] == planner['escape']['mode'] == 'none'
            assert planner['stag']['kappa'] == 0 and planner['stag']['prop_boost'] == 1
            history = lines(out/'per_iter_logs.json')
            logs = lines(out/'logs.json')
            rate = [row['final_eval/success_rate'] for row in logs if 'final_eval/success_rate' in row][-1]
            assert np.isclose(rate, successes.mean())
            for i in range(b['n']):
                e = episodes[f"{b['chunk']}:{i}"]
                rows = sorted((row for row in history if row['sample_idx'] == i), key=lambda row: row['step'])
                assert all(1 <= row['step'] <= 30 for row in rows)
                assert len({row['step'] for row in rows}) == len(rows)
                first = next((row['step'] for row in rows if row['success']), None)
                if not rows:
                    st, goal = target[0][i], target[1][i]
                    assert np.linalg.norm(st[:4]-goal[:4]) < 20
                    assert abs((st[4]-goal[4]+np.pi) % (2*np.pi)-np.pi) < np.pi/9
                    first = 0
                if first is None:
                    assert len(rows) == 30
                assert e['first_success_replan'] == first
                assert e['success30'] == bool(successes[i]) == (first is not None)
                assert e['success20'] == (first is not None and first <= 20)
                assert e['replans'] == len(rows)
        for horizon in (20, 30):
            assert sum(e[f'success{horizon}'] for e in episodes.values()) == result['groups']['all'][f'success{horizon}']
        if arm != 'B0':
            base = all_episodes['B0']
            for horizon in (20, 30):
                key = f'success{horizon}'
                rescued = sorted(k for k, e in episodes.items() if e[key] and not base[k][key])
                harmed = sorted(k for k, e in episodes.items() if not e[key] and base[k][key])
                recorded = result['paired_vs_B0'][str(horizon)]
                assert rescued == sorted(recorded['rescued']) and harmed == sorted(recorded['harmed'])
        all_episodes[arm] = episodes
        scores = read(root/'eval_outputs/effect_b_20261103/candidates'/f'scores_{arm}.json')
        assert scores['candidate_sha256'] == candidate_sha
        assert scores['decisions'] == len(scores['records']) == 64
        summaries[arm] = {
            'groups': result['groups'],
            'paired_vs_B0': {k: {name: len(ids) for name, ids in value.items()} for k, value in result.get('paired_vs_B0', {}).items()},
            'candidate_diagnostics': {k: v for k, v in scores.items() if k != 'records'},
            'final_training_metrics': metrics[-1],
            'checkpoint': {'path': str((train/'checkpoints/model_3.pth').relative_to(root)),
                           'sha256': digest(train/'checkpoints/model_3.pth')},
        }
    return {'client_archive_date': '2026-11-05', 'environment': 'PushObj', 'training_seed': 0,
            'epochs': 3, 'tasks_per_arm': 350, 'total_tasks': 1400, 'max_replans': 30,
            'gpu_hours': gpu_hours, 'arms': summaries,
            'verification': {'passed': True, 'chunks_checked': 140, 'raw_successes_match_reports': True,
                             'all_initial_modules_and_sampler_hashes_match': True,
                             'paired_start_goal_states_identical': True, 'planner_configs_identical': True,
                             'candidate_archive_sha256': candidate_sha}}


def archive(root, output, name, sources):
    sources = sorted(set(sources))
    records = [{'path': str(p.relative_to(root)), 'bytes': p.stat().st_size, 'sha256': digest(p)} for p in sources]
    path = output/(name+'.tar.gz')
    with path.open('wb') as handle, gzip.GzipFile(filename='', mode='wb', compresslevel=6, mtime=0, fileobj=handle) as gz:
        with tarfile.open(mode='w|', fileobj=gz) as tar:
            for source, record in zip(sources, records):
                item = tar.gettarinfo(str(source), arcname=record['path'])
                item.uid = item.gid = item.mtime = 0
                item.uname = item.gname = ''
                with source.open('rb') as member:
                    tar.addfile(item, member)
    with tarfile.open(path, 'r:gz') as tar:
        actual = tar.getmembers()
        assert [item.name for item in actual] == [item['path'] for item in records]
        for member, record in zip(actual, records):
            assert hashlib.sha256(tar.extractfile(member).read()).hexdigest() == record['sha256']
    print(f'ARCHIVED {name}: {len(sources)} files, {path.stat().st_size/2**20:.2f} MiB', flush=True)
    return {'file': path.name, 'bytes': path.stat().st_size, 'sha256': digest(path),
            'verified_after_compression': True, 'members': records}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = ap.parse_args()
    root = args.root.resolve()
    report = root/'reports/effect_b_20261103'
    summary = validate(root)
    save(report/'summary.json', summary)
    print('VALIDATED 140 chunks / 1400 tasks and all paired results', flush=True)
    with (report/'summary.csv').open('w') as handle:
        fields = ['arm','tasks','success20','success30','rate20','rate30','seen_rate30','unseen_rate30','delta30_pp','rescued30','harmed30','stall_proxy_env_steps']
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n"); writer.writeheader()
        for arm, result in summary['arms'].items():
            groups = result['groups']; all_ = groups['all']; paired = result['paired_vs_B0'].get('30', {})
            writer.writerow({'arm':arm,'tasks':all_['n'],'success20':all_['success20'],'success30':all_['success30'],
                'rate20':all_['rate20'],'rate30':all_['rate30'],'seen_rate30':groups['seen']['rate30'],
                'unseen_rate30':groups['unseen']['rate30'],'delta30_pp':100*(all_['rate30']-summary['arms']['B0']['groups']['all']['rate30']),
                'rescued30':paired.get('rescued',0),'harmed30':paired.get('harmed',0),'stall_proxy_env_steps':all_['mean_physical_stall_env_steps']})
    for arm in ARMS:
        dest = report/'training'/arm; dest.mkdir(parents=True, exist_ok=True)
        for name in ['metrics.jsonl','metrics_iter.jsonl','experiment_initialization.json','hydra.yaml']:
            source = root/'checkpoints/effect_b_20261103'/arm/name
            if source.exists(): shutil.copy2(source, dest/name)
    diagnostics = report/'diagnostics'; diagnostics.mkdir(exist_ok=True)
    for source in (root/'eval_outputs/effect_b_20261103/candidates').iterdir():
        if source.is_file(): shutil.copy2(source, diagnostics/source.name)
    cache = report/'cache'; cache.mkdir(exist_ok=True)
    shutil.copy2(root/'data/pushobj_bias_16k/effect_cache_v1/complete.json', cache/'complete.json')
    output = report/'archives'; output.mkdir(exist_ok=True)
    groups = {
        'console_logs': [p for directory in [root/'reports/effect_ab_20261103', report] for p in directory.rglob('*.log')],
        'training_runtime': [p for p in (root/'checkpoints/effect_b_20261103').rglob('*') if p.is_file() and p.suffix != '.pth'],
        'cache_metadata': list((root/'data/pushobj_bias_16k/effect_cache_v1').glob('*.jsonl')),
        'evaluation_smokes': [p for name in ['planning_smoke','multigpu_smoke'] for p in (root/'eval_outputs/effect_b_20261103'/name).rglob('*') if p.is_file()],
    }
    for arm in ARMS:
        groups['evaluation_'+arm] = [p for p in (root/'eval_outputs/effect_b_20261103'/arm/'frozen30').rglob('*') if p.is_file()]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        pending = [pool.submit(archive, root, output, name, sources) for name, sources in groups.items()]
        manifests = [job.result() for job in pending]
    save(output/'manifest.json', {'format':'tar.gz; paths relative to repository root', 'lossless':True,
        'model_checkpoints_included':False, 'checkpoint_hashes':'../summary.json', 'archives':manifests})
    print(json.dumps({'archive_count':len(manifests),'compressed_MiB':sum(x['bytes'] for x in manifests)/2**20,
                      'gpu_hours':summary['gpu_hours'],'verification':summary['verification']}), flush=True)


if __name__ == '__main__':
    main()
