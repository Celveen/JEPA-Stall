#!/usr/bin/env python3
"""Plot measured threshold outcomes and empirical Rope success-by-push curves.

Use --import-repo once to verify and bundle counts from the archived task CSVs.
Normal builds require only provenance/planning_curve_data.json. No experiments.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/jepa-planning-curves')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--import-repo', type=Path)
args = parser.parse_args()
p = Path(__file__).resolve().parents[1]
data_path = p/'provenance/planning_curve_data.json'

if args.import_repo:
    repo = args.import_repo
    source_hashes = {}
    def rows(name):
        source = repo/name
        source_hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
        return list(csv.DictReader(source.open()))

    threshold = rows('reports/ecp_threshold_v2/paired_tasks.csv')
    assert len(threshold) == 350
    keys = [(r['chunk'], r['sample_idx']) for r in threshold]
    assert len(set(keys)) == 350
    counts = [sum(int(r[f'gamma{g}_success']) for r in threshold) for g in ['015','030','045']]
    followup = [r for r in rows('reports/ecp_threshold_v3/paired_tasks.csv')
                if r['condition'] == 'adajepa_aug_g060']
    keyed = {(r['chunk'], r['sample_idx']): r for r in followup}
    assert len(keyed) == 350 and set(keyed) == set(keys)
    for r in threshold:
        other = keyed[r['chunk'], r['sample_idx']]
        assert r['native_success'] == other['native_success']
        assert r['gamma045_success'] == other['ada045_success']
    counts.append(sum(int(r['ecp_success']) for r in followup))
    native = sum(int(r['native_success']) for r in threshold)
    assert native == 262 and counts == [269, 271, 272, 271]

    rope = rows('reports/rope_seeds99_100_101/per_run.csv')
    assert len(rope) == 240
    arms = ['native3', 'fixed3', 'dynamic3', 'fixed3_nowarm']
    assert set(r['arm'] for r in rope) == set(arms)
    curves = {}
    target_lists = {}
    for arm in arms:
        selected = [r for r in rope if r['arm'] == arm]
        assert len(selected) == 60
        per_seed = {}
        for seed in ['99','100','101']:
            subset = [r for r in selected if r['seed'] == seed]
            assert len(subset) == 20
            targets = sorted(int(r['global_sample']) for r in subset)
            assert len(set(targets)) == 20
            if seed not in target_lists:
                target_lists[seed] = targets
            assert targets == target_lists[seed]
            for r in subset:
                assert r['success'] in ('True','False')
                assert (r['success'] == 'True') == (float(r['final_cd']) < .3)
                assert 1 <= int(r['pushes']) <= 20
                assert r['success'] == 'True' or int(r['pushes']) == 20
            per_seed[seed] = [sum(r['success'] == 'True' and int(r['pushes']) <= k
                                  for r in subset) for k in range(21)]
        cumulative = [sum(v[k] for v in per_seed.values()) for k in range(21)]
        assert cumulative == sorted(cumulative)
        curves[arm] = {'runs': 60, 'success_counts': cumulative,
                      'per_seed_success_counts': per_seed,
                      'mean_executed_pushes': sum(int(r['pushes']) for r in selected)/60}
    assert [curves[a]['success_counts'][-1] for a in arms] == [33,43,38,42]
    data = {'source_sha256': source_hashes, 'new_experiments_run': False,
            'threshold': {'upper_thresholds': [.15,.30,.45,.60], 'tasks':350,
                          'native_successes': native, 'ecp_successes':counts},
            'rope': {'push_budgets':list(range(21)), 'success_criterion':'CD < 0.3',
                     'same_20_targets_repeated_across_three_seeds':True,
                     'curves':curves},
            'plotting': {'threshold':'straight segments connect measured points only',
                         'rope':'empirical cumulative success; right-continuous steps',
                         'smoothing':False, 'extrapolation':False}}
    data_path.write_text(json.dumps(data,indent=2)+'\n')

data = json.loads(data_path.read_text())
plt.rcParams.update({'font.family':'serif','font.serif':['STIXGeneral'],
    'mathtext.fontset':'stix','font.size':9, 'axes.labelsize':9,
    'xtick.labelsize':8, 'ytick.labelsize':8, 'legend.fontsize':8,
    'pdf.fonttype':42, 'ps.fonttype':42, 'svg.fonttype':'none',
    'axes.linewidth':.6, 'savefig.facecolor':'white'})
BROWN, TEAL = '#B95B38', '#087D83'
def style(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(length=2.5, width=.6)
    ax.grid(axis='y',color='#DCE2E5',linewidth=.45,zorder=0)
    ax.set_axisbelow(True)

def save(fig, stem):
    for suffix in ['pdf','svg','png']:
        kw = {'metadata':{'CreationDate':None,'ModDate':None}} if suffix=='pdf' else {}
        fig.savefig(p/f'figures/{stem}.{suffix}',dpi=240,**kw)
        if suffix == 'svg':
            target = p/f'figures/{stem}.svg'
            target.write_text('\n'.join(line.rstrip() for line in target.read_text().splitlines())+'\n')
    plt.close(fig)

t=data['threshold']
x=np.array(t['upper_thresholds']); y=np.array(t['ecp_successes'])*100/t['tasks']
native=t['native_successes']*100/t['tasks']
fig,ax=plt.subplots(figsize=(3.35,2.00))
fig.subplots_adjust(left=.15,right=.97,bottom=.24,top=.94)
style(ax)
ax.axhline(native,color=BROWN,ls=(0,(4,2)),lw=1.3)
ax.plot(x,y,color=TEAL,marker='o',ms=4,lw=1.6,zorder=3)
for xx,yy in zip(x,y):
    ax.annotate(f'{yy:.1f}',(xx,yy),xytext=(0,6),textcoords='offset points',
                ha='center',color=TEAL,fontsize=8)
ax.text(.155,75.1,f'Native: {native:.1f}%',color=BROWN,fontsize=8)
ax.text(.60,76.35,'ECP',color=TEAL,ha='right',fontsize=8)
ax.set(xlim=(.12,.63),ylim=(74,79),xticks=x,yticks=[74,76,78],
       xlabel=r'Upper trigger threshold $\gamma_{\rm hi}$',ylabel='Success (%)')
ax.set_xticklabels(['0.15','0.30','0.45','0.60'])
save(fig,'fig_threshold_sensitivity')

fig,ax=plt.subplots(figsize=(3.35,2.3))
fig.subplots_adjust(left=.15,right=.97,bottom=.20,top=.73)
style(ax)
specs=[('native3','Native',BROWN,'--',1.4),
       ('fixed3','Ours: fixed S, warm-up 5',TEAL,'-',1.8),
       ('dynamic3','Dynamic S, warm-up 5','#5668A6','-.',1.2),
       ('fixed3_nowarm','Fixed S, no warm-up','#9A6A9E',':',1.5)]
handles=[]
for arm,label,color,ls,lw in specs:
    curve=data['rope']['curves'][arm]
    y=np.array(curve['success_counts'])*100/curve['runs']
    line,=ax.step(data['rope']['push_budgets'],y,where='post',label=label,
                 color=color,linestyle=ls,linewidth=lw,zorder=4 if arm=='fixed3' else 3)
    handles.append(line)
ax.legend(handles=[handles[i] for i in [0,2,1,3]],loc='lower left',
          bbox_to_anchor=(-.035,1.02),ncol=2,frameon=False,
          handlelength=1.8,columnspacing=.8,handletextpad=.5,fontsize=7.1)
ax.set(xlim=(0,20),ylim=(0,80),xticks=[0,5,10,15,20],yticks=[0,20,40,60,80],
       xlabel='Executed pushes',ylabel='Cumulative success (%)')
save(fig,'figS_rope_success_budget')
print('Verified outcomes and created two vector plots with embedded, selectable text.')
