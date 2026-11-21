import json, pickle, hashlib, collections, math, pathlib, statistics

ROOT=pathlib.Path('/home/appuser/workspace/project/multimodal-test/JEPA-Stall')
OUT=pathlib.Path('/tmp/jepa_paper_evidence')
def read(p): return json.loads((ROOT/p).read_text())
def source(p): return {'path':p,'sha256':hashlib.sha256((ROOT/p).read_bytes()).hexdigest()}
def mcnemar(w,l):
    if w+l==0:return 1.
    return min(1.,2*sum(math.comb(w+l,i) for i in range(min(w,l)+1))/2**(w+l))
def records(root,chunked=False):
    out=collections.defaultdict(dict)
    for f in sorted((ROOT/root).glob('*/per_iter_logs.json')):
        for line in f.open():
            if line.strip():
                r=json.loads(line);key=(f.parent.name,r['sample_idx']) if chunked else r['sample_idx']
                assert r['step'] not in out[key],(root,key,r['step'])
                out[key][r['step']]=r
    return out
def success(st,b): return any(r['success'] and s<=b for s,r in st.items())
def compare(ref,arm,b):
    merged=dict(ref);merged.update(arm)
    assert set(merged)==set(ref)
    base={k:success(v,b) for k,v in ref.items()};cur={k:success(v,b) for k,v in merged.items()}
    w=sum(cur[k] and not base[k] for k in ref);l=sum(base[k] and not cur[k] for k in ref)
    return {'n':len(ref),'budget_replans':b,'baseline_success':sum(base.values()),'success':sum(cur.values()),'wins':w,'losses':l,'p_exact_unadjusted':mcnemar(w,l),'rerun':len(arm)}

e={'date':'2026-11-05','root':str(ROOT),'audit':'Read-only CPU aggregation of existing logs; no model inference or new rollouts.'}
e['pushobj_main30']=[]
for model in ['official','bias16k']:
    for base in ['frozen','adajepa']:
        run=f'eval_outputs/stag_{base}_{model}_it40';ref=records(run+'/ref',True)
        assert len(ref)==350
        summary=f'reports/stag_step1/{"official_frozen" if model=="official" and base=="frozen" else "official_adajepa" if model=="official" else "bias16k_T0P" if base=="frozen" else "bias16k_adajepa"}_it30.json'
        reported=read(summary)
        row={'model':model,'base':base,'source':source(summary),'raw_reference':run+'/ref','arms':{}}
        row['baseline_success']=sum(success(v,30) for v in ref.values())
        assert row['baseline_success']==round(reported['base_success']*350)
        for name in ['SZ_mid','T0P','T0SP']:
            r=compare(ref,records(run+'/'+name,True),30)
            assert r['success']==round(reported['arms'][name]['success']*350)
            assert [r['wins'],r['losses']]==reported['arms'][name]['vs_base'][:2]
            r['seen_success']=round(reported['arms'][name]['seen']*200)
            r['unseen_success']=round(reported['arms'][name]['unseen']*150)
            row['arms'][name]=r
        e['pushobj_main30'].append(row)
ref=records('eval_outputs/stag_frozen_bias16k_it40/ref',True)
e['pushobj_ablation30']={a:compare(ref,records('eval_outputs/stag_frozen_bias16k_it40/'+a,True),30) for a in ['SZ_mid','T0P','T0SP','ALPHA10','T0SPL','LA_start']}
e['pushobj_ablation30_source']=source('reports/stag_step2/bias16k_it30.json')
e['maze_e3']={'source':source('reports/e3/medium_s100/stage2.json'),'raw_reference':'eval_outputs/stag_maze100_ar4/medium/adajepa/ref','n':100,'seed':100,'checkpoint':'checkpoints/diversemaze_layout_shift','layouts':5,'tasks_per_layout':20,'bfs_distance_blocks':[6,9],'action_repeat':4,'environment_calls_per_replan':5,'base_environment_steps_per_replan':20,'planning_horizon_control_calls':25,'planning_horizon_base_steps':100,'frameskip':5,'model_steps_executed_per_replan':1,'model_horizon':5,'raw_budget':50,'arms':{}}
ref=records(e['maze_e3']['raw_reference']);assert len(ref)==100
for name in ['R-trig','R-switch','R-delay','SZ-switch','P-switch','F-trig','F-switch']:
    raw=records('eval_outputs/e3/medium_s100/'+name)
    e['maze_e3']['arms'][name]={str(b):compare(ref,raw,b) for b in [20,30,50]}
e['pushobj_s7']={'source':source('reports/e3/pushobj_S7/stage1.json'),'arms':{}}
ref=dict(records('eval_outputs/escape_S7/none',True));ref.update(records('eval_outputs/stag_S7_it50/S0',True));assert len(ref)==280
for name in ['R-trig','R-switch']:
    raw=records('eval_outputs/e3/pushobj_S7/'+name,True)
    e['pushobj_s7']['arms'][name]={str(b):compare(ref,raw,b) for b in [20,30,50]}
e['action_offset30']={}
for stream in ['escape_EVAL50_r30','escape_EVAL50s200']:
    p=f'reports/{stream}/summary.json';d=read(p)
    e['action_offset30'][stream]={'source':source(p),'n':d['n'],'arms':{k:{q:v[q] for q in ['success','only_arm','only_none','p','seen','unseen','budget']} for k,v in d['arms'].items()}}

targets={}
for name in ['S7','EVAL50','EVAL50s200']:
    ids=[]
    for p in sorted((ROOT/'data/targets'/name).glob('*.pkl')):
        d=pickle.load(p.open('rb'))
        ids.extend((str(s['shape']),int(s['ep_idx']),int(s['offset'])) for s in d['segments'])
    targets[name]=ids
e['target_overlap']={'identity':'shape, author ep_idx, segment offset','streams':{k:{'slots':len(v),'unique_segments':len(set(v)),'unique_source_episodes':len(set((a,b) for a,b,c in v))} for k,v in targets.items()},'pairwise':[]}
for a,b in [('S7','EVAL50'),('S7','EVAL50s200'),('EVAL50','EVAL50s200')]:
    e['target_overlap']['pairwise'].append({'a':a,'b':b,'shared_segments':len(set(targets[a])&set(targets[b])),'shared_source_episodes':len(set((x,y) for x,y,z in targets[a])&set((x,y) for x,y,z in targets[b]))})

d=read('reports/pushobj_bias_16k/train_seed_summary.json')
e['training_data_recipe']={'config':read('data/pushobj_bias_16k/generation_config.json'),'source':source('data/pushobj_bias_16k/generation_config.json'),'formula':'a_exec = a_source + 9 * (object_CoG - pusher) / norm(object_CoG - pusher) + 10 * Normal(0,I_2) [pixels]; environment input = a_exec / 100; store executed actions.','training_seed_results':{seed:{setting:v['all'] for setting,v in vals.items() if setting!='vs_released'} for seed,vals in d['per_seed'].items()},'across_training_seeds':{setting:{g:vals[g] for g in ['seen','unseen','all']} for setting,vals in d['across'].items()},'released':d['released'],'statistics_source':source('reports/pushobj_bias_16k/train_seed_summary.json')}
e['b_training_screen']={'source':source('reports/effect_b_20261103/summary.json'),'arms':{k:{'groups':v['groups'],'paired_vs_B0':v.get('paired_vs_B0'),'candidate_diagnostics':v['candidate_diagnostics']} for k,v in read('reports/effect_b_20261103/summary.json')['arms'].items()},'settings':'seed0, 3epochs, effective batch64, TS0.01, no decoder, identical initial backbone parameters and sampler order; EVAL50 development targets, 30 replans, frozen model; no planning correction.','calibration':read('reports/effect_b_20261103/calibration/complete.json'),'B1_formula':'Per5-step prediction pair, w=1+I[visual_effect OR proprio_effect], normalized by global batch mean; replace JEPA prediction MSE by weighted MSE; TS unchanged. Coverage98.10% in final training epoch.','B2_formula':'L_IDM=mean||g([meanpatch(v[t+1]-v[t]),p[t+1]-p[t]])-a[t:t+5]||^2; g=MLP394->256->10 with GELU; use real encoded frames at0,5,10,15 and each corresponding5executed actions. No action embedding input. lambda=.418538541733931.','B3_formula':'For branches visual/proprio, layer-normalize each per-sample embedding without affine parameters (eps1e-5). D_b(i,j)=mean((LN(z_b[i])-LN(z_b[j]))^2), NOT RMS distance. Loss=global positive-branch-pair mean of [max(0,.2-D_b(i,j))]^2 for pairs(0,5),(5,10),(10,15),(0,15). No negative/static contraction term. lambda=2.4677111194988774.'}

e['method']={
'distance':'D_alpha(z,zprime)=mean((visual-visualprime)^2)+alpha*mean((proprio-proprioprime)^2), alpha=1; latent has visual and proprio branches; proprio may encode velocity.',
'trigger':'r_t=sqrt(D(z_t,z_{t-W})/max(D(z_t,z_goal),1e-12)), W=3; w_t=clip((0.15-r_t)/(0.15-0.05),0,1). Store observation and w whenever positive; re-encode all memory observations with current adapted encoder every replan.',
'regularizer':'P(A)=sum_h c_h sum_j [w_j D(m_j,g)/(H+1)] [max(0,1-D_sp(z_h,m_j)/(beta D_sp(m_j,g)))]^2 / sum_h c_h; c_h=2^h, beta=.25, kappa=2. Distances/rho clamped at1e-12. Sum over memories (not averaged), persistent memory until episode reset, no deduplication.',
'radius':'beta multiplies squared distance; corresponding Euclidean radius is sqrt(beta)=0.5 times goal distance. Goal outside each memory support when beta<1. At exact center a squared radial kernel has zero own gradient; no escape guarantee.',
'proprio':'Boost alpha by10 for the next3 plan calls after a trigger, renewed each trigger. Implementation scales both predicted and target proprio bysqrt(10) before original objective. Baseline trigger distance uses alpha1 even during boost.',
'switch':'If stalled and (bar_l[t-W]-bar_l[t])/max(bar_l[t-W],1e-12)<theta=.05, activate corrections; bar_l[t]=(l[t-2]+l[t-1]+l[t])/3, W=3, K=1. l is newest not-yet-updated transition prediction MSE. Switch occurs after env feedback and effects begin next plan. Once switched remains active; updates CONTINUE unless freeze control explicitly true.',
'objective':'Maze: weighted all-frame objective including context, weights normalized then averaged over T=H+1 frames. PushObj: last-frame objective for MPC iter<H, then same all-frame objective. GD Adam lr.1,100 iterations,zero init, warm starts unexecuted actions.',
'action_offset':'K2P uses physical pusher displacement<10px over3 replans, checks start=10 (next plan is eleventh execution), min gap4, max7 escapes. On escape add v/||v||*min(15px,||v||/5) to each of5 executed env actions, v=goal pusher position-current pusher; prop boost10 for kick plus2 replans. This is a separate physical trigger branch, NOT combined with latent-memory main T0SP arm.',
'training':'Bias data collection can use simulator object CoG. Deployed latent trigger/regularizer/proprio weight use observed images/proprio and goal observation, no oracle object state. K2P uses pusher goal proprio. Training observer B labels also not the same as online stall trigger.',
'provenance':[source(p) for p in ['planning/stagnation.py','planning/switch.py','planning/escape.py','planning/objectives.py','scripts/e3_run.py','scripts/stag_base_run.py','scripts/gen_pushobj_dataset.py']]
}
e['limitations']=[
'All intervention comparisons are repeatedly used development targets; no independent confirmatory holdout or new multi-seed intervention result exists.',
'PushObj four checkpoint/planner combinations share the same350 tasks; cannot pool as1400 independent episodes or four independent replications.',
'Maze best SZ-switch selected among several developer arms:30replans66/100 vs63/100;50replans87/100 vs82/100, p=.125 at50. No statistically significant general Maze improvement demonstrated.',
'SZ-switch is penalty-only and has no PushObj counterpart in E3; strongest PushObj result uses proprio reweighting. Do not present per-domain selected maxima as one tested fixed policy.',
'Old Maze action_repeat1 failures and corrected action_repeat4 E3 run are different temporal protocols. All arms within E3 use4.',
'Training recipe combines action bias9px, noise10px, original-source placement with retries/jitter, contact selection and filtering; historical released-model comparison does not isolate action bias or match all training factors.',
'Historical report.md source-placement description is stale; generation_config.json is actual recipe. Bias/noise parameters calibrated with author released evaluation-segment statistics; targets are development data.',
'Author training-source/test-source overlap cannot be ruled out; source training/own validation pools are separated, but original author evaluation ancestry is unverified.',
'Three historical training seeds exist; four B training-screen arms and intervention studies use one training seed. Historical bias16k seed0 differs from new B0 (no decoder, initialization/runtime differences); do not combine B0 62.86 with old corrected72.29 as a paired gain.',
'Negative B2/B3 do not causally establish collapse, goal conflict, or impossibility of representation regularization. Common-candidate diagnostics are fixed candidate choice, not MPC success nor an oracle reachability bound.',
'Moving losses are on different transitions; loss gate is not a convergence/overfitting/learnability detector. Freezing controls lower Maze success; adaptation should continue.',
'Planning-stage gain is largely pusher-goal completion: old40replan T0SP rescues37/54 object-already-tolerant failures and2/78 object-not-at-goal failures. This breakdown is40budget and should not be silently called30budget.',
'Penalty is a compact-support latent repulsion, not demonstrated learned geodesic distance, progress guarantee, globally optimal planning, or novel general potential-field principle.',
'B3 positive-effect separation differs from planning-stage stagnation memory regularizer; failure of one does not test the other.'
]
e['pending_experiments']=['Independent held-out task stream and Maze seeds with frozen hyperparameters','Matched collection ablation separating action bias/noise/contact selection while fixing source episodes, sample count and training schedule','Three-seed intervention replication','Complete shared intervention-component matrix across both domains','Task-completion breakdown at matched budgets and long-horizon/contact-specific goals']
(OUT/'evidence.json').write_text(json.dumps(e,indent=2,ensure_ascii=False))

md=['# Verified evidence for manuscript','',e['audit'],'','## Main PushObj table: common30-replan budget,350 shared development tasks','', '| Checkpoint | Planner | Baseline | Memory only | Proprio only | Both |','|---|---|---:|---:|---:|---:|']
for r in e['pushobj_main30']:
    nums=[r['baseline_success']]+[r['arms'][k]['success'] for k in ['SZ_mid','T0P','T0SP']]
    md.append('| '+r['model']+' | '+r['base']+' | '+' | '.join(f'{n}/350 ({100*n/350:.1f}%)' for n in nums)+' |')
md+=['','Source paths, file hashes and per-arm wins/losses are in evidence.json. These350 tasks are reused in all four rows; rows are not independent studies.','', '## PushObj bias16k frozen ablation at30 replans','', '| Method | Success | Rescue / harm | Exact paired p |','|---|---:|---:|---:|']
for k,v in e['pushobj_ablation30'].items():md.append(f"| {k} | {v['success']}/350 ({100*v['success']/350:.1f}%) | {v['wins']} / {v['losses']} | {v['p_exact_unadjusted']:.4g} |")
md+=['','## Maze: corrected action_repeat4,100 medium tasks,seed100','', '| Method |20replans|30replans|50replans|Rescue/harm at30|Rescue/harm at50|p at50|','|---|---:|---:|---:|---:|---:|---:|','|AdaJEPA|51|63|82|—|—|—|']
for k,v in e['maze_e3']['arms'].items():md.append(f"|{k}|{v['20']['success']}|{v['30']['success']}|{v['50']['success']}|{v['30']['wins']}/{v['30']['losses']}|{v['50']['wins']}/{v['50']['losses']}|{v['50']['p_exact_unadjusted']:.4g}|")
md+=['','5 held-out validation layouts,20tasks/layout; BFS start-goal distance6–9blocks. Horizon25control calls (100 base env steps),execute5calls (20base steps) per replan; frameskip5 gives5latent horizonsteps. At30replans control budget150calls/600base steps. No gate stops adaptation.','', '## Training recipe','',e['training_data_recipe']['formula'],'','Actual source placement,4retry attempts,20px/.3rad jitter,quality gates.16k train/4k val; T/L/Z/+ shapes. Prior report text claiming centroid placement is outdated.','', '| Training evaluation setting | Frozen (mean ± SD) | AdaJEPA (mean ± SD)|','|---|---:|---:|']
for k,v in e['training_data_recipe']['across_training_seeds'].items():
    q=v['all'];md.append(f"|{k}|{100*q['frozen_mean']:.2f} ± {100*q['frozen_sd']:.2f}|{100*q['adajepa_mean']:.2f} ± {100*q['adajepa_sd']:.2f}|")
md+=['','SD across3training seeds0/1/2. Eval seed100 uses20replans; eval seeds200/300 use30. The released checkpoint is one model, so comparisons are data-recipe associations, not purebias causal experiments.','', '## Target reuse','',json.dumps(e['target_overlap'],indent=2),'','## Method faithful to code','']
for k,v in e['method'].items():
    if k!='provenance':md += [f'**{k}.** {v}','']
md+=['## Claims and reporting limits','']+[f'- {x}' for x in e['limitations']]
md+=['','## Explicitly pending experiments','']+[f'- {x}' for x in e['pending_experiments']]
(OUT/'evidence.md').write_text('\n'.join(md)+'\n')
print(json.dumps({'output':str(OUT),'push30':[(r['model'],r['base'],r['baseline_success'],{k:v['success'] for k,v in r['arms'].items()}) for r in e['pushobj_main30']],'maze30':{k:[v['30']['success'],v['30']['wins'],v['30']['losses'],v['30']['p_exact_unadjusted']] for k,v in e['maze_e3']['arms'].items()},'overlap':e['target_overlap']},indent=2))
