"""Audit the added Native-3 baseline and compare all five matched-target conditions."""
import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from report import report as audit_original

ARMS = ('native5', 'native3', 'fixed3', 'dynamic3', 'fixed3_nowarm')
LABELS = ('Native 5/5', 'Native 3/3', 'Fixed S 3/3', 'Dynamic S 3/3', 'Fixed S, no warmup')


def csv_write(path, rows):
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main(original, addition, output):
    original, addition, output = Path(original), Path(addition), Path(output)
    assert json.loads((addition/'queue.json').read_text())['status'] == 'complete'
    with tempfile.TemporaryDirectory(prefix='rope-original-audit-') as directory:
        previous = audit_original(original, directory)
    source = addition/'source_snapshot'
    before, after = (source/'worker_before.py').read_bytes(), (source/'worker_native3.py').read_bytes()
    assert after == before.replace(b"choices=('native5', 'fixed3',", b"choices=('native5', 'native3', 'fixed3',", 1)
    worker_key = 'experiments/rope_formal20/worker.py'
    old_sha, new_sha = hashlib.sha256(before).hexdigest(), hashlib.sha256(after).hexdigest()
    data, rows = {}, []
    for arm in ARMS:
        for i in range(20):
            p = (addition if arm == 'native3' else original)/arm/f'sample_{i:02d}'
            receipt = json.loads((p/'complete.json').read_text())
            frozen = json.loads((p/'frozen_model_audit.json').read_text())
            encoder = json.loads((p/'encoder_audit.json').read_text())
            chunk = json.loads((p/'chunk_audit.json').read_text())
            records = json.loads((p/'ecp_records.json').read_text())
            costs = json.loads((p/'ecp_costs.json').read_text())
            reuse = json.loads((p/'prefix_reuse.json').read_text())
            with np.load(p/'rope_metrics.npz') as f: metrics = dict(f)
            with np.load(p/'final_eval_results.npz') as f: final = dict(f)
            cfg = receipt['config']['planner']
            native = arm.startswith('native')
            h = 5 if arm == 'native5' else 3
            assert cfg['chunk_size'] == cfg['sub_planner']['horizon'] == h
            assert cfg['max_iter'] == 20 and cfg['success_cd'] == .3 and cfg['n_taken_actions'] == 1
            assert cfg['sub_planner']['num_samples'] == 300 and cfg['sub_planner']['topk'] == 30
            assert cfg['sub_planner']['opt_steps'] == 10
            assert cfg['reuse_actions'] and not cfg['resim_from_start']
            assert cfg['ecp']['kappa'] == (0 if native else 2)
            assert cfg['ecp']['warmup_pushes'] == (0 if arm == 'fixed3_nowarm' else 5)
            assert cfg['ecp']['independent_execution'] and 'reference_dir' not in cfg['ecp']
            assert not any(r['reused'] for r in reuse)
            assert frozen['unchanged'] and frozen['all_parameters_frozen'] and frozen['all_modules_eval']
            assert encoder['matches_official_pretrained_weights'] and encoder['verified_tensors'] == 175
            assert receipt['global_sample'] == i+10
            assert receipt['target_chunk'] == (i+10)//10 and receipt['target_index'] == i%10
            assert cfg['episode_seed'] == 99*(i+10)+1 and cfg['planning_seed'] == 99+i+10
            pushes, cd = int(metrics['actual_pushes'][0]), float(metrics['final_cd'][0])
            assert np.isfinite(cd) and 0 <= pushes <= 20
            assert bool(metrics['successes'][0]) == (cd < .3) and (cd < .3 or pushes == 20)
            timeline = np.r_[metrics['initial_cd'][0], metrics['cd_by_replan'][0]]
            assert len(timeline) == pushes+1 and final['actions'].shape[1] == pushes
            hits = np.flatnonzero(timeline < .3)
            assert not len(hits) or hits[0] == pushes
            assert chunk['optimizer_push_indices_zero_based'] == list(range(0,pushes,h))
            first = int(metrics['first_intervention'][0])
            if native:
                assert first == -1 and not costs
            if arm == 'native3':
                with np.load(p/'native_execution_cache.npz') as cache:
                    assert cache['obs_visual'].shape[1] == pushes+1
                    np.testing.assert_array_equal(cache['states'][0,-1], metrics['final_states'][0])
                    physical = final['actions'] * np.asarray(receipt['action_std'],dtype=np.float32) + np.asarray(receipt['action_mean'],dtype=np.float32)
                    np.testing.assert_array_equal(cache['physical_actions'], physical)
            events = [] if native else [r for r in records if r.get('severity_weight',0) > 0]
            used = [r for r in events if any(q >= r['completed_pushes'] for q in chunk['optimizer_push_indices_zero_based'])]
            row = dict(arm=arm,sample=i,global_sample=i+10,initial_cd=float(timeline[0]),final_cd=cd,
                success=cd<.3,actual_pushes=pushes,optimizer_calls=chunk['optimizer_calls'],
                S_trigger_events=len(events),used_trigger_memories=len(used),
                first_S_application_push=first+1 if first>=0 else None,
                positive_penalty_evaluations=sum(v['positive_candidates'] for v in costs.values()),
                used_fractional_strength_memories=sum(0 < r['new_memory_strength'] < 2 for r in used),
                planning_s=sum(r['planning_s'] for r in records),elapsed_s=frozen['elapsed_s'])
            rows.append(row)
            data[arm,i] = dict(path=p,receipt=receipt,frozen=frozen,metrics=metrics,final=final,timeline=timeline,row=row)
    for (arm,i), d in data.items():
        ref = data['dynamic3',i]
        for key in ('checkpoint_sha256','encoder_sha256','full_target_sha256','action_mean','action_std','torch'):
            assert d['receipt'][key] == ref['receipt'][key], key
        assert d['frozen']['parameters_and_buffers_before'] == ref['frozen']['parameters_and_buffers_before']
        for key in ('state_0','state_g'):
            np.testing.assert_array_equal(d['final'][key],ref['final'][key])
        actual_sources = d['receipt']['implementation_sha256']
        assert actual_sources.keys() == ref['receipt']['implementation_sha256'].keys()
        for key, value in actual_sources.items():
            expected = (new_sha if arm=='native3' else old_sha) if key==worker_key else ref['receipt']['implementation_sha256'][key]
            assert value == expected, key
        if arm == 'native3':
            left, right = copy.deepcopy(d['receipt']['config']), copy.deepcopy(ref['receipt']['config'])
            for cfg in (left,right):
                cfg.pop('saved_folder'); cfg.pop('goal_file_path'); cfg['planner']['ecp'].pop('kappa')
            assert left == right, 'Native3 differs from dynamic3 beyond kappa and output/target-file paths'
    summary = dict(status='complete',samples_per_condition=20,maximum_pushes=20,success_rule='CD < 0.3',
                   target_chunks=[1,2],formal_episodes=100,conditions={},paired_vs_native3={})
    for arm in ARMS:
        selected = [r for r in rows if r['arm']==arm]
        summary['conditions'][arm] = dict(mean_cd=float(np.mean([r['final_cd'] for r in selected])),
            std_cd_across_targets=float(np.std([r['final_cd'] for r in selected],ddof=1)),
            successes=sum(r['success'] for r in selected),total_pushes=sum(r['actual_pushes'] for r in selected),
            optimizer_calls=sum(r['optimizer_calls'] for r in selected),S_trigger_events=sum(r['S_trigger_events'] for r in selected),
            triggered_samples=sum(r['S_trigger_events']>0 for r in selected),
            applied_samples=sum(r['first_S_application_push'] is not None for r in selected),
            used_trigger_memories=sum(r['used_trigger_memories'] for r in selected),
            used_fractional_strength_memories=sum(r['used_fractional_strength_memories'] for r in selected),
            positive_penalty_evaluations=sum(r['positive_penalty_evaluations'] for r in selected),
            planning_s_sum=sum(r['planning_s'] for r in selected),elapsed_s_sum=sum(r['elapsed_s'] for r in selected))
        if arm != 'native3':
            for key in ('mean_cd','successes','total_pushes','S_trigger_events'):
                assert summary['conditions'][arm][key] == previous['conditions'][arm][key], 'Historical result changed'
    comparisons = []
    baseline = np.asarray([data['native3',i]['row']['final_cd'] for i in range(20)])
    for arm in ('native5','fixed3','dynamic3','fixed3_nowarm'):
        cds = np.asarray([data[arm,i]['row']['final_cd'] for i in range(20)])
        summary['paired_vs_native3'][arm] = dict(mean_cd_delta=float(np.mean(cds-baseline)),
            lower_cd=int((cds<baseline).sum()),higher_cd=int((cds>baseline).sum()),equal_cd=int((cds==baseline).sum()),
            gained_successes=int(((cds<.3)&(baseline>=.3)).sum()),lost_successes=int(((cds>=.3)&(baseline<.3)).sum()))
        for i in range(20):
            l,r = data['native3',i],data[arm,i]
            std,mean = [np.asarray(l['receipt'][key],dtype=np.float32) for key in ('action_std','action_mean')]
            a,b = [d['final']['actions'][0]*std+mean for d in (l,r)]
            n = min(len(a),len(b)); delta=b[:n]-a[:n]
            changed=np.flatnonzero(np.any(delta!=0,axis=1))
            cd_changed=np.flatnonzero(l['timeline'][1:n+1]!=r['timeline'][1:n+1])
            first=int(changed[0])+1 if len(changed) else None
            applied=r['row']['first_S_application_push']
            comparisons.append(dict(left='native3',right=arm,sample=i,global_sample=i+10,
                native3_cd=float(baseline[i]),other_cd=float(cds[i]),cd_delta=float(cds[i]-baseline[i]),
                common_pushes=n,changed_executed_pushes=len(changed),
                complete_controls_identical=a.shape==b.shape and np.array_equal(a,b),
                first_control_difference_push=first,first_cd_difference_push=int(cd_changed[0])+1 if len(cd_changed) else None,
                first_S_application_push=applied,
                control_difference_before_first_S=bool(first is not None and (applied is None or first<applied)) if not arm.startswith('native') else None,
                maximum_physical_control_L2=float(np.linalg.norm(delta,axis=1).max()) if n else 0.))
    summary['audit'] = dict(identical_ordered_initial_and_goal_states=True,identical_frozen_model=True,
        same_planner_core_sources=True,worker_only_change='CLI accepts native3',
        native3_vs_dynamic3_config_difference='kappa 0 vs 2; output and target-file paths differ',
        all_native3_actual_frames_saved=True,independent_physics=True)
    output.mkdir(parents=True,exist_ok=True)
    csv_write(output/'per_sample.csv',rows);csv_write(output/'action_comparisons.csv',comparisons)
    csv_write(output/'cd_by_push.csv', [dict(arm=arm,sample=i,global_sample=i+10,
        push=step,cd=float(cd),actual_final=step==len(d['timeline'])-1)
        for (arm,i),d in data.items() for step,cd in enumerate(d['timeline'])])
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# Rope：补齐同样本 Native 3/3', '',
        '五组各 20 个相同目标（原编号 10–29），最多 20 推，每推观测，CD < 0.3 即停。原四组结果完整保留，仅新增 Native-3。', '',
        '| 条件 | 平均 CD↓ | 成功 | 实际总推数 | S 触发次数 | 实际施加 S 的目标 |',
        '|---|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        r=summary['conditions'][arm]
        lines.append(f"| {arm} | {r['mean_cd']:.6f} | {r['successes']}/20 | {r['total_pushes']} | {r['S_trigger_events']} | {r['applied_samples']} |")
    lines += ['', '固定 S 与动态 S 的预热为 5 推；fixed3_nowarm 无额外预热，但仍需三推检测窗口。', '',
        '## 与 Native-3 的同目标比较', '',
        '| 条件 | 平均 CD 差（本条件 − Native-3）↓ | CD 更低 / 更高 | 新增成功 / 丢失成功 |',
        '|---|---:|---:|---:|']
    for arm,r in summary['paired_vs_native3'].items():
        lines.append(f"| {arm} | {r['mean_cd_delta']:+.6f} | {r['lower_cd']} / {r['higher_cd']} | {r['gained_successes']} / {r['lost_successes']} |")
    lines += ['', 'Native-5 对 Native-3 比较块长度；Native-3 对三个 S 条件比较同样 3/3 节奏下加入 S 的表现。目标、模型、种子、CEM 候选/迭代、成功规则与预算均匹配。', '',
        '各条件独立执行物理仿真；相同种子不能消除数值波动。动作在 S 介入前可能已分叉，因此本表是单次独立执行的同协议对照，不能把每一个动作差异都归因于 S。', '',
        '| 与 Native-3 比较 | 动作序列不同的目标 | 共有推数中动作不同 | S 介入前已有动作差异的目标 |',
        '|---|---:|---:|---:|']
    for arm in ('native5','fixed3','dynamic3','fixed3_nowarm'):
        selected=[r for r in comparisons if r['right']==arm]
        early='—' if arm=='native5' else str(sum(r['control_difference_before_first_S'] for r in selected))
        lines.append(f"| {arm} | {sum(not r['complete_controls_identical'] for r in selected)}/20 | {sum(r['changed_executed_pushes'] for r in selected)}/{sum(r['common_pushes'] for r in selected)} | {early} |")
    lines += ['', '未实际施加 S 而已有动作差异的目标也计入最后一列；逐样本首个动作/CD 差异与首个 S 使用时点见 action_comparisons.csv。', '',
        '![五组正式结果](results.png)', '',
        '[全部逐样本结果](per_sample.csv) · [动作比较](action_comparisons.csv) · [汇总与校验](summary.json)', '',
        '原四组和可视化实例见 [原始结果包](../rope_formal20/README.md)。其中的截图来自单独可视化复跑，不混入本表。本次 Native-3 的实际逐推画面全部随补充证据包保存。', '']
    (output/'RESULTS.md').write_text('\n'.join(lines))
    plot(data,output)
    return summary


def plot(data, output):
    os.environ.setdefault('MPLCONFIGDIR','/tmp/jepa-stall-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=('#555555','#111111','#1976b9','#cf6b19','#7c509e')
    fig,axes=plt.subplots(1,2,figsize=(13,4.6),constrained_layout=True)
    for j,(arm,label,color) in enumerate(zip(ARMS,LABELS,colors)):
        rows=[data[arm,i]['row'] for i in range(20)]
        cds=[r['final_cd'] for r in rows]
        axes[0].scatter(j+np.linspace(-.13,.13,20),cds,color=color,s=22,alpha=.8)
        axes[0].scatter(j,np.mean(cds),color=color,marker='D',s=90,edgecolor='white',zorder=4)
        curve=[100*np.mean([r['success'] and r['actual_pushes']<=k for r in rows]) for k in range(21)]
        axes[1].step(range(21),curve,where='post',label=label,color=color,linewidth=2)
    axes[0].axhline(.3,color='#777777',linestyle=':',linewidth=1)
    axes[0].set(xticks=range(5),xticklabels=LABELS,ylabel='Final Chamfer distance',title='All 20 matched targets; diamonds show means',ylim=(0,None))
    axes[0].tick_params(axis='x',labelsize=8,rotation=12)
    axes[1].set(xlabel='Completed pushes',ylabel='Success (%)',title='Success at CD < 0.3',xticks=range(0,21,5),yticks=range(0,101,20),ylim=(0,100))
    axes[1].legend(frameon=False,fontsize=8)
    for ax in axes:
        ax.grid(axis='y',alpha=.18);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Rope: matched Native 3/3 control added; 20 targets per condition, 20-push limit',fontsize=11)
    for suffix in ('png','pdf'):fig.savefig(output/f'results.{suffix}',dpi=200)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original',type=Path,default=Path('eval_outputs/rope_formal20'))
    parser.add_argument('--addition',type=Path,default=Path('eval_outputs/rope_formal20_native3'))
    parser.add_argument('--output',type=Path,default=Path('reports/rope_formal20_native3'))
    args=parser.parse_args()
    print(json.dumps(main(args.original,args.addition,args.output),indent=2))
