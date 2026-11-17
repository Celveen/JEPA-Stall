#!/usr/bin/env python3
"""Plot measured Rope CD at matched push budgets; no simulator runs.

--import-repo verifies every point against original NPZ archives and bundles the
observed sequences. Normal builds use only provenance/rope_cd_curve_data.json.
"""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile

os.environ.setdefault('MPLCONFIGDIR', '/tmp/jepa-rope-cd-curve')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PAPER = Path(__file__).resolve().parents[1]
DATA = PAPER/'provenance/rope_cd_curve_data.json'
ARMS = ['native3', 'fixed3']
SEEDS = [99, 100, 101]


def import_data(repo):
    hashes = {}
    def source(name):
        p = repo/name
        hashes[name] = hashlib.sha256(p.read_bytes()).hexdigest()
        return p
    def csv_rows(name):
        with source(name).open() as f:
            return list(csv.DictReader(f))

    endpoints = {(int(r['seed']), r['arm'], int(r['sample'])): r
                 for r in csv_rows('reports/rope_seeds99_100_101/per_run.csv')}
    csv_paths = {99:'reports/rope_formal20_native3/cd_by_push.csv',
                 100:'reports/rope_seed100_20/cd_by_push.csv',
                 101:'reports/rope_seed101_20/cd_by_push.csv'}
    records = []
    initial_values = {}
    metrics_verified = 0
    for seed in SEEDS:
        rows = csv_rows(csv_paths[seed])
        for arm in ARMS:
            if seed == 99:
                archive = ('reports/rope_formal20_native3/native3_evidence.tar.gz'
                           if arm == 'native3' else 'reports/rope_formal20/evidence.tar.gz')
                prefix = 'supplement' if arm == 'native3' else 'formal'
                prefix += f'/{arm}'
            else:
                archive = f'reports/rope_seed{seed}_20/{arm}_evidence.tar.gz'
                prefix = arm
            with tarfile.open(source(archive)) as tar:
                for sample in range(20):
                    endpoint = endpoints[seed, arm, sample]
                    assert int(endpoint['global_sample']) == sample+10
                    selected = sorted([r for r in rows if r['arm']==arm and int(r['sample'])==sample],
                                      key=lambda r: int(r['push']))
                    pushes = int(endpoint['pushes'])
                    assert [int(r['push']) for r in selected] == list(range(pushes+1))
                    cds = np.array([float(r['cd']) for r in selected])
                    member = f'{prefix}/sample_{sample:02}/rope_metrics.npz'
                    with np.load(io.BytesIO(tar.extractfile(member).read()), allow_pickle=False) as z:
                        assert int(z['actual_pushes'].item()) == pushes
                        assert np.array_equal(z['completed_pushes_by_replan'], np.arange(1,pushes+1))
                        assert z['executed_mask'].all()
                        raw = np.r_[z['initial_cd'].ravel(), z['cd_by_replan'].ravel()]
                        assert np.array_equal(raw, cds), (seed,arm,sample)
                        assert float(z['final_cd'].item()) == cds[-1] == float(endpoint['final_cd'])
                        assert float(z['success_cd'].item()) == .3
                        assert bool(z['successes'].item()) == (endpoint['success']=='True') == (cds[-1]<.3)
                    assert np.isfinite(cds).all() and (cds>=0).all()
                    assert (cds[:-1]>=.3).all()
                    assert pushes == 20 or cds[-1] < .3
                    if sample not in initial_values:
                        initial_values[sample] = cds[0]
                    assert initial_values[sample] == cds[0]
                    records.append({'seed':seed,'arm':arm,'sample':sample,'global_sample':sample+10,
                                    'pushes':pushes,'success':bool(cds[-1]<.3),'observed_cd':cds.tolist()})
                    metrics_verified += len(cds)
    assert len(records)==120
    curves = {}
    for arm in ARMS:
        seed_means = []
        for seed in SEEDS:
            selected = [r for r in records if r['arm']==arm and r['seed']==seed]
            assert len(selected)==20
            # A stopped successful run retains its observed terminal state.
            # Do not discard solved episodes or pad them with an invented zero.
            held = [r['observed_cd']+[r['observed_cd'][-1]]*(20-r['pushes']) for r in selected]
            seed_means.append(np.mean(held,axis=0))
        mean = np.mean(seed_means,axis=0)
        sd = np.std(seed_means,axis=0,ddof=1)
        curves[arm] = {'seed_means':np.array(seed_means).tolist(),'mean_cd':mean.tolist(),
                       'seed_sd':sd.tolist(),'first_mean_cd_below_0_7':int(np.flatnonzero(mean<.7)[0]),
                       'trapezoidal_area_0_to_20':float(np.trapezoid(mean))}
    assert np.isclose(curves['native3']['mean_cd'][-1], .6864581634387983)
    assert np.isclose(curves['fixed3']['mean_cd'][-1], .4792646551396854)
    assert [curves[a]['first_mean_cd_below_0_7'] for a in ARMS] == [14,12]
    data = {'source_sha256':hashes,'new_experiments_run':False,'seeds':SEEDS,
            'distinct_targets':20,'runs_per_method':60,'push_budgets':list(range(21)),
            'observed_cd_values_verified_against_raw_npz':metrics_verified,
            'stopping_rule':'Stop at first CD < 0.3 or 20 executed pushes.',
            'aggregation':'Equal-weight mean over 20 targets and then over three seeds.',
            'stopped_runs':'Carry terminal observed CD forward at subsequent push budgets.',
            'band':'Sample SD (ddof=1) of three seed-level means; not a confidence interval.',
            'plotting':'Straight segments between integer push budgets; no smoothing or monotonic fit.',
            'interpretation':'Descriptive controller comparison; independent physics can diverge before intervention.',
            'curves':curves,'runs':records}
    DATA.write_text(json.dumps(data,indent=2)+'\n')
    print(f'Verified {metrics_verified} observed CD values from 120 runs against raw metrics.')
    return data


def plot(data):
    plt.rcParams.update({'font.family':'serif','font.serif':['STIXGeneral'],
        'mathtext.fontset':'stix','font.size':9,'axes.labelsize':9,
        'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':7.8,
        'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
        'axes.linewidth':.6,'savefig.facecolor':'white'})
    fig,ax=plt.subplots(figsize=(3.35,1.5))
    fig.subplots_adjust(left=.14,right=.98,bottom=.27,top=.82)
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(length=2.5,width=.6)
    ax.grid(axis='y',color='#DCE2E5',linewidth=.45,zorder=0)
    ax.set_axisbelow(True)
    for arm,label,color,ls in [('native3','DINO-WM (native)','#B95B38','--'),
                               ('fixed3','DINO-WM (ours)','#087D83','-')]:
        c=data['curves'][arm];mean=np.array(c['mean_cd']);sd=np.array(c['seed_sd'])
        ax.fill_between(data['push_budgets'],mean-sd,mean+sd,color=color,alpha=.13,linewidth=0)
        ax.plot(data['push_budgets'],mean,color=color,ls=ls,lw=1.4,label=label,
                marker='o',markersize=2.7,markevery=[0,5,10,15,20],zorder=3)
    ax.legend(loc='lower left',bbox_to_anchor=(-.02,1.02),ncol=2,frameon=False,
              handlelength=1.6,handletextpad=.45,columnspacing=1.0,borderaxespad=0)
    ax.set(xlim=(5,20),ylim=(0,1.8),xticks=[5,10,15,20],yticks=[0,.5,1.,1.5],
           xlabel='Pushes',ylabel='Mean CD')
    for suffix in ['pdf','svg','png']:
        options={'metadata':{'CreationDate':None,'ModDate':None}} if suffix=='pdf' else {}
        fig.savefig(PAPER/f'figures/fig_rope_cd_pushes.{suffix}',dpi=240,**options)
        if suffix == 'svg':
            target = PAPER/f'figures/fig_rope_cd_pushes.{suffix}'
            target.write_text('\n'.join(line.rstrip() for line in target.read_text().splitlines())+'\n')
    plt.close(fig)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--import-repo',type=Path)
    args=parser.parse_args()
    data=import_data(args.import_repo) if args.import_repo else json.loads(DATA.read_text())
    plot(data)
    print('Created Rope CD curve with embedded fonts and selectable labels.')
