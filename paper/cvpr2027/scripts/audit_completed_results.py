from pathlib import Path
import hashlib,json
import numpy as np
from datetime import datetime,timezone
import argparse
parser=argparse.ArgumentParser(description='Recompute completed result counts and verify paired trajectories; no experiments.')
parser.add_argument('--repo',type=Path,required=True)
r=parser.parse_args().repo
paper=Path(__file__).resolve().parents[1];root=r/'eval_outputs/pusht_visual_v8'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
m=json.loads((root/'manifest.json').read_text());verified=json.loads((root/'verification.json').read_text())
assert m['n_jobs']==56 and m['n_evals']==50 and verified['status']=='passed'
old={x['id']:x for x in verified['rows']};data={};hashes={};common=None
for job in m['jobs']:
    folder=root/job['id'];done=json.loads((folder/'completion.json').read_text())
    assert done['fingerprint']==job['fingerprint'] and done['n']==50
    attempt=folder/done['attempt'];raw=attempt/'final_eval_results.npz'
    hashes[job['id']]=sha(raw);assert hashes[job['id']]==old[job['id']]['npz_sha256']
    with np.load(raw) as d:
        arr={k:d[k] for k in ['successes','state_0','state_g','actions']}
    assert len(arr['successes'])==50 and int(arr['successes'].sum())==old[job['id']]['successes']
    assert abs(arr['successes'].mean()-done['success_rate'])<1e-7
    if common is None:common=arr
    for k in ['state_0','state_g']:np.testing.assert_array_equal(arr[k],common[k])
    data[job['id']]=(arr,attempt)
pairs=[]
for job in m['jobs']:
    if not job['ecp']:continue
    arr,attempt=data[job['id']];base,_=data[job['id'].removesuffix('_ecp')]
    episodes=[json.loads(s) for s in (attempt/'stag_episodes.jsonl').read_text().splitlines()]
    assert len(episodes)==50 and sorted(x['sample_idx'] for x in episodes)==list(range(50))
    for ep in episodes:
        i=ep['sample_idx'];first=ep['first_trigger'];n=min(arr['actions'].shape[1],base['actions'].shape[1])
        if first is not None:n=min(first,n)
        np.testing.assert_allclose(arr['actions'][i,:n],base['actions'][i,:n],atol=1e-5,rtol=1e-5)
    a=base['successes'].astype(bool);b=arr['successes'].astype(bool)
    pairs.append({'id':job['id'],'n':50,'native_successes':int(a.sum()),'ecp_successes':int(b.sum()),'rescues':int((~a&b).sum()),'regressions':int((a&~b).sum())})
# Recompute the five imported PushObj reference rows from the stored outcomes.
ref=json.loads((paper/'provenance/pushobj_baseline_data_v11.json').read_text());reference_states=None;reference_rows=[]
for row in ref['rows']:
    parts=[]
    for item in row['files']:
        path=r/item['path'];assert sha(path)==item['sha256']
        with np.load(path) as d:parts.append({k:d[k] for k in ['successes','state_0','state_g']})
    joined={k:np.concatenate([x[k] for x in parts]) for k in parts[0]}
    assert len(joined['successes'])==350 and int(joined['successes'].sum())==row['successes']
    if reference_states is None:reference_states=joined
    for k in ['state_0','state_g']:np.testing.assert_array_equal(joined[k],reference_states[k])
    reference_rows.append({k:row[k] for k in ['method','planner','n','successes','success_percent']})
# The PushT imported rows have exactly the same starts and goals as the new matrix.
audit=json.loads((r/'reports/pusht_visual_v8/baseline_audit.json').read_text())
archive=r/'reports/pusht_visual_v8/raw_baselines';baseline_checks=[]
for row in audit['external_baseline_rows']:
    for relative in row['source_files']:
        raw=archive/relative;condition=raw.parent.name
        with np.load(raw) as d:
            for k in ['state_0','state_g']:np.testing.assert_array_equal(d[k],common[k])
            assert len(d['successes'])==50
            score=100*float(d['successes'].mean())
            assert abs(score-row['scores_percent'][condition])<1e-6
        baseline_checks.append({'file':relative,'sha256':sha(raw),'success_percent':score})
maze=json.loads((r/'eval_outputs/maze_frozen_v10/summary.json').read_text())
assert maze['baseline_successes']==48 and maze['n']==100
for arm in maze['arms'].values():
    assert arm['pretrigger_pairs_verified']==50 and arm['untriggered_reused']==50
    assert arm['successes']==48+len(arm['rescues'])-len(arm['regressions'])
report={'checked_utc':datetime.now(timezone.utc).isoformat(),'new_experiments_started':False,'primary_jobs_complete':56,'episode_runs':2800,'paired_episodes_verified':1400,'all_primary_targets_equal':True,'raw_result_sha256':hashes,'native_ecp_pairs':pairs,'pushobj_reference_rows':reference_rows,'pushobj_reference_states_match_within_list':True,'pusht_imported_runs_verified':len(baseline_checks),'pusht_imported_files':baseline_checks,'maze_summary':maze,'missing_cells_note':'Three archived CEM clean-PushT measurements unavailable; cross-list PushObj blanks are not incomplete primary jobs. New DINO-WM v3 remains in progress and is not imported.'}
(paper/'provenance/results_audit_v13.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['raw_result_sha256','native_ecp_pairs','pushobj_reference_rows','pusht_imported_files','maze_summary']},indent=2))
