import csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent

def analyze():
    m=json.loads((ROOT/'manifest.json').read_text());rows=[];calls=[];jobs=[];paired=0
    for pair in m['pairs']:
        paths={arm:ROOT/'runs'/pair['chunk']/arm for arm in ('off','on')}
        if not all((p/'complete.json').exists() for p in paths.values()):continue
        receipts={a:json.loads((p/'complete.json').read_text()) for a,p in paths.items()}
        assert receipts['off']['gpu']==receipts['on']['gpu']
        with np.load(paths['off']/'final_eval_results.npz') as a,np.load(paths['on']/'final_eval_results.npz') as b:
            np.testing.assert_array_equal(a['state_0'],b['state_0']);np.testing.assert_array_equal(a['state_g'],b['state_g'])
        for arm,path in paths.items():
            rec=receipts[arm];assert rec['physics_replay_verified'] and rec['model_unchanged'] and not rec['smoke']
            jobs.append(rec)
            eps=[json.loads(l) for l in (path/'cost_episodes.jsonl').read_text().splitlines()]
            assert len(eps)==rec['tasks']==pair['n'] and sum(r['success'] for r in eps)==rec['successes']
            for row in eps:rows.append(dict(arm=arm,chunk=pair['chunk'],shape=pair['shape'],gpu=rec['gpu'],**row))
            for line in (path/'cost_calls.jsonl').read_text().splitlines():
                calls.append(dict(arm=arm,chunk=pair['chunk'],**json.loads(line)))
        paired+=1
    if not rows:return
    arms={}
    for arm in ('off','on'):
        r=[x for x in rows if x['arm']==arm];j=[x for x in jobs if x['arm']==arm]
        gd=[x for x in calls if x['arm']==arm and x['metric']=='gd']
        replans=sum(x['replans'] for x in r)
        assert len(gd)==replans
        arms[arm]=dict(tasks=len(r),successes=sum(x['success'] for x in r),
            success_percent=100*sum(x['success'] for x in r)/len(r),
            replans=replans,mean_replans=float(np.mean([x['replans'] for x in r])),
            mean_controls=float(np.mean([x['controls'] for x in r])),
            total_episode_wall_seconds=sum(x['episode_wall_seconds'] for x in r),
            mean_episode_wall_seconds=float(np.mean([x['episode_wall_seconds'] for x in r])),
            median_episode_wall_seconds=float(np.median([x['episode_wall_seconds'] for x in r])),
            p95_episode_wall_seconds=float(np.percentile([x['episode_wall_seconds'] for x in r],95)),
            total_planner_seconds=sum(x['planner_seconds'] for x in r),
            mean_planner_ms_per_replan=1000*sum(x['planner_seconds'] for x in r)/replans,
            total_gd_seconds=sum(x['gd_seconds'] for x in r),
            mean_gd_ms_per_replan=1000*sum(x['gd_seconds'] for x in r)/replans,
            p95_gd_ms_per_replan=1000*float(np.percentile([x['seconds'] for x in gd],95)),
            common_diagnostic_feedback_seconds=sum(x['common_feedback_seconds'] for x in r),
            ecp_memory_build_seconds=sum(x['ecp_build_memory_seconds'] for x in r),
            ecp_detector_seconds=sum(x['ecp_detector_seconds'] for x in r),
            ecp_active_replans=sum(x['ecp_active_replans'] for x in r),
            peak_allocated_gib=max(x['peak_allocated_bytes'] for x in r)/2**30,
            peak_reserved_gib=max(x['peak_reserved_bytes'] for x in r)/2**30,
            cold_pipeline_seconds_excluding_hash=sum(x['planning_main_excluding_hash_seconds'] for x in j),
            model_parameters=j[0]['model_parameters'])
    off={(r['chunk'],r['sample_index']):r for r in rows if r['arm']=='off'}
    on={(r['chunk'],r['sample_index']):r for r in rows if r['arm']=='on'}
    assert off.keys()==on.keys()
    gains=dict(rescues=sum(not off[k]['success'] and on[k]['success'] for k in off),
               regressions=sum(off[k]['success'] and not on[k]['success'] for k in off),
               success_gain_pp=arms['on']['success_percent']-arms['off']['success_percent'])
    for metric in ('mean_episode_wall_seconds','mean_planner_ms_per_replan','mean_gd_ms_per_replan','total_planner_seconds','mean_replans','peak_allocated_gib'):
        gains[metric+'_change_percent']=100*(arms['on'][metric]/arms['off'][metric]-1)
    summary=dict(state='complete' if paired==len(m['pairs']) else 'partial',completed_pairs=paired,total_pairs=len(m['pairs']),
                 paired_tasks_per_arm=len(off),arms=arms,comparison=gains,
                 notes=['One timing pass over 350 fixed tasks; task variation is not timing repeat uncertainty.',
                        'GPU-synchronized wall timers; same physical GPU for each paired chunk; alternating order.',
                        'Both arms use the same GD100 CUDA Graph backend; per-replan costs include graph capture.',
                        'No-ECP uses EscapeMPCPlanner with escape disabled, avoiding inactive ECP detector bookkeeping.',
                        'Episode wall includes simulator prefix replay and shared diagnostics, excludes model loading and independent audit.',
                        'PyTorch memory peaks include model tensors and graph pools; reserved memory differs from allocated memory.',
                        'Five concurrent dedicated benchmark GPUs; CPU is shared. No single-GPU isolation or energy measurement is claimed.'])
    (ROOT/'summary.json.tmp').write_text(json.dumps(summary,indent=2)+'\n');(ROOT/'summary.json.tmp').replace(ROOT/'summary.json')
    with (ROOT/'per_task_cost.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    labels=[('Success (%)','success_percent'),('Replans / task','mean_replans'),('Controls / task','mean_controls'),
            ('Episode wall (s / task)','mean_episode_wall_seconds'),('Planning (ms / replan)','mean_planner_ms_per_replan'),
            ('GD (ms / replan)','mean_gd_ms_per_replan'),('Total planning (s)','total_planner_seconds'),
            ('Peak allocated GPU memory (GiB)','peak_allocated_gib'),('Peak reserved GPU memory (GiB)','peak_reserved_gib')]
    lines=['# Frozen ECP cost comparison','',f'Status: {summary["state"]}; {len(off)} paired tasks per arm.','',
           '| Metric | Without ECP | With ECP |','|---|---:|---:|']
    for label,key in labels:lines.append(f'| {label} | {arms["off"][key]:.3f} | {arms["on"][key]:.3f} |')
    lines+=['',f'Rescues: {gains["rescues"]}; regressions: {gains["regressions"]}.','']+summary['notes']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':analyze()
