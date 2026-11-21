from pathlib import Path
import json,hashlib,shutil
import numpy as np
import argparse
parser=argparse.ArgumentParser(description='Import and verify recorded DINO-WM PushT CEM results; no experiments.')
parser.add_argument('--repo',type=Path,required=True)
r=parser.parse_args().repo;s=Path(__file__).resolve().parents[1];download=r/'reports/pusht_visual_v8/raw_baselines'
base=r/'reports/pusht_visual_v8/raw_baselines/research_snapshots/2026-11-08-pusht-gd/source_pipeline_metadata/dinowm'
meta=json.loads((base/'summary.json').read_text());receipt=json.loads((base/'evaluation_plan_receipt.json').read_text());status=json.loads((base/'pipeline_status.json').read_text())
assert meta['epoch']==3 and meta['checkpoint_sha256']=='4cba2153c3851dd4f9fabc7ad67fd8384df11fc65422bf9d9dfc7000d48fa3a4'
assert status['stage']=='complete' and all(x['status']=='complete' for x in status['jobs'].values())
assert receipt['sha256']==meta['evaluation_plan_sha256']
plan=receipt['plan'];assert plan['planners']['cem']['num_samples']==300 and plan['planners']['cem']['opt_steps']==30
assert all(x['target_sha256']=='1950b39e5d3ba13b30c14a435a400d84cd3fbe30138cdfed55e98b202d76277b' for x in plan['tasks']['pusht']['conditions'])
primary=np.load(r/'reports/pusht_visual_v8/raw_baselines/research_snapshots/2026-11-08-pusht-gd/completed_runs/dinowm/default/final_eval_results.npz')
order=['default','blur','snp','dark','red_agent','red_object','red_goal'];raw=[];scores=[]
for c in order:
    row=meta['by_planner']['cem']['conditions'][c];assert row['episodes']==50
    assert abs(row['success_rate']*50-row['successes'])<1e-6
    score=row['successes']*2;scores.append(score)
    matches=list(download.glob('research_snapshots/**/cem/'+c+'/final_eval_results.npz'))
    item={'condition':c,'n':50,'successes':row['successes'],'success_percent':score,'source_kind':'completed_run_summary'}
    if matches:
        assert len(matches)==1;p=matches[0];a=np.load(p)
        for k in ['state_0','state_g']:np.testing.assert_array_equal(a[k],primary[k])
        assert len(a['successes'])==50 and int(a['successes'].sum())==row['successes']
        conf=json.loads((p.parent/'protocol.json').read_text());weights=json.loads((p.parent/'frozen_weights_check.json').read_text())
        assert weights['unchanged'] and weights['before_sha256']==weights['after_sha256']
        assert [conf[k] for k in ['model_epoch','episodes','max_replans','executed_physical_steps_per_replan','candidates','optimizer_iterations']]==[3,50,6,25,300,30]
        item.update(source_kind='raw_npz_and_completed_summary',raw_path=str(p.relative_to(download)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),exact_states_match=True)
    else:
        assert c in ['red_object','red_goal'];item['raw_archive_available']=False
    raw.append(item)
assert scores==[48,40,2,44,40,24,48] and sum(scores[1:])/6==33.0
new=json.loads((s/'provenance/visual_baseline_data_v8.json').read_text());oldrow=None
for row in new['rows']:
    if row['method']=='DINO-WM' and row['planner']=='CEM':
        oldrow=dict(row);row.update(success_percent=scores,training='random T16k',epoch=3,checkpoint_sha256=meta['checkpoint_sha256'],evidence='dinowm_pusht_cem_e3_v14.json')
new['protocol_note']='150 physical controls. Both DINO-WM PushT rows use the same random T16k epoch-3 checkpoint, terminal objective, and 6x25 execution. DINO-WM CEM uses 300x30, while other CEM rows use 200x10 and 30x5. Shared task list; each method retains its recorded noise realization.'
new['new_ecp_results']='All 56 primary jobs complete; provenance/comparison_v14.json'
new['detailed_audit']='dinowm_pusht_cem_e3_v14.json (new CEM); original v8 audit retained for other baselines'
(s/'provenance/visual_baseline_data_v14.json').write_text(json.dumps(new,indent=2)+'\n')
report={'source_commit':'d4f6cb491ac19d0224485b056fe5b24aa8e50f94','training':'random T16k','epoch':3,'checkpoint_sha256':meta['checkpoint_sha256'],'evaluation_plan_sha256':meta['evaluation_plan_sha256'],'target_sha256':plan['tasks']['pusht']['conditions'][0]['target_sha256'],'protocol':meta['controller'],'cem':plan['planners']['cem'],'replaced_pushT_row':oldrow,'conditions':raw,'six_shift_mean_percent':33.0,'all_conditions_complete':True,'raw_npz_conditions_verified':5,'summary_only_conditions':['red_object','red_goal'],'summary_evidence':{'path':str((base/'summary.json').relative_to(r)),'sha256':hashlib.sha256((base/'summary.json').read_bytes()).hexdigest()},'raw_archive_root':'reports/pusht_visual_v8/raw_baselines','note':'The directory retains an old e10 name; checkpoint receipt and per-run protocol verify epoch 3. Red-object/red-goal are backed by completed 50-task aggregate records; their raw NPZ files are absent from the source snapshot.'}
(s/'provenance/dinowm_pusht_cem_e3_v14.json').write_text(json.dumps(report,indent=2)+'\n')
p=s/'scripts/update_comparison_tables.py';t=p.read_text().replace('visual_baseline_data_v8.json','visual_baseline_data_v14.json').replace('comparison_v13.json','comparison_v14.json');p.write_text(t)
p=s/'sections/appendix.tex';t=p.read_text();old='The archived DINO-WM GD implementation instead executes 25 steps per replan for up to six replans. The archived CEM runs use 30 replans, five executed physical steps, 200 candidates, 30 elites, and ten optimizer iterations. DINO-WM uses a random-data T16k checkpoint for GD and a released checkpoint for CEM; LeWM uses the same checkpoint for both planners.'
newtext='Both DINO-WM PushT planners use the same random-data T16k checkpoint at epoch three, a terminal goal objective, and up to six replans executing 25 physical steps each. DINO-WM CEM uses 300 candidates, 30 elites, and 30 optimizer iterations; its GD counterpart uses 100 gradient steps. LeWM and DDP-WM CEM retain 30 replans, five executed physical steps, 200 candidates, 30 elites, and ten optimizer iterations. LeWM uses the same checkpoint for both planners.'
assert old in t or newtext in t;t=t.replace(old,newtext);p.write_text(t)
print('Verified new row:',scores,'mean',sum(scores[1:])/6)
