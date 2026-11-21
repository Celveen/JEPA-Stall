#!/usr/bin/env python3
"""Redraw two figures from archived numbers only. No experiment is executed.

Portable usage: python build_results_v2.py --data plot_data.json --out ./figures
Requires numpy and matplotlib. Bundled plot_data.json retains source hashes.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PALETTE = {
    'native_fill': '#A8B9C8',
    'weighting_fill': '#477C82',
    'memory_fill': '#D99270',
    'native_line': '#62849F',
    'correction_line': '#B56E4F',
    'ink': '#2C4C63',
    'grid': '#E6EBEF',
    'muted': '#798791',
}

def style():
    plt.rcParams.update({
        'font.family':'serif', 'font.serif':['STIXGeneral','DejaVu Serif'],
        'mathtext.fontset':'stix', 'font.size':9,
        'axes.labelsize':9, 'xtick.labelsize':8.5, 'ytick.labelsize':8.5,
        'text.color':PALETTE['ink'], 'axes.labelcolor':PALETTE['ink'],
        'xtick.color':PALETTE['ink'], 'ytick.color':PALETTE['ink'],
        'axes.edgecolor':PALETTE['ink'], 'axes.linewidth':0.65,
        'pdf.fonttype':42, 'ps.fonttype':42, 'svg.fonttype':'none',
        'savefig.facecolor':'white', 'figure.facecolor':'white',
    })

def save(fig, stem, out):
    for ext in ('pdf','svg','png'):
        fig.savefig(out/f'{stem}.{ext}', dpi=250)
    plt.close(fig)

def clean(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(axis='both', length=2.5, width=.65, pad=3)
    ax.set_axisbelow(True)

def results(data,out):
    p=data['pushobj']; m=data['maze']
    assert p['n']==350 and data['budget_replans']==30
    assert m['n']==100 and m['budget_replans']==30
    fig=plt.figure(figsize=(6.875,2.75))
    ax=fig.add_axes([.075,.265,.638,.545])
    mx=fig.add_axes([.816,.265,.166,.545])
    x=np.array([0.,1.10,2.40,3.50]); width=.25
    specs=[('native','Native MPC',PALETTE['native_fill']),
           ('P','+ weighting',PALETTE['weighting_fill']),
           ('PS','+ weighting + memory',PALETTE['memory_fill'])]
    for j,(key,label,color) in enumerate(specs):
        vals=np.asarray(p[key])/p['n']*100
        bars=ax.bar(x+(j-1)*width, vals, width-.025,
                    facecolor=color, edgecolor='white',linewidth=.4,
                    label=label,zorder=3)
        for bar,v in zip(bars,vals):
            ax.text(bar.get_x()+bar.get_width()/2,v+1.7,f'{v:.1f}',
                    ha='center',va='bottom',fontsize=8.1)
    ax.set_xticks(x,['Released\nFrozen','Contact-biased\nFrozen',
                     'Released\nAdaJEPA','Contact-biased\nAdaJEPA'])
    ax.set_ylabel('Success (%)',labelpad=5)
    ax.set_xlim(-.57,4.07)
    ax.set_ylim(0,100);ax.set_yticks([0,25,50,75,100])
    ax.grid(axis='y',color=PALETTE['grid'],linewidth=.6)
    ax.legend(loc='upper left',bbox_to_anchor=(-.025,1.34),ncol=3,
              frameon=False,fontsize=8.6,handlelength=1.05,
              columnspacing=1.3,handletextpad=.45,borderpad=0)
    ax.text(0,1.12,'(a) PushObj',transform=ax.transAxes,fontsize=10,
            fontweight='bold',ha='left')
    ax.text(.37,1.12,'350 paired tasks',transform=ax.transAxes,
            fontsize=9,ha='left')
    # Exact paired difference: 253/350 - 214/350 = 11.142857... pp.
    delta=(p['PS'][1]-p['native'][1])/p['n']*100
    left,right=x[1]-width,x[1]+width
    ax.plot([left,left,right,right],[83,86,86,83],
            color=PALETTE['ink'],linewidth=.75,clip_on=False)
    ax.text(x[1],88,f'+{delta:.1f} pp',ha='center',va='bottom',
            fontsize=8.5,fontweight='bold')
    vals=[m['baseline_success'],m['success']]
    bars=mx.bar([0,1],vals,.54,
                facecolor=PALETTE['native_fill'],edgecolor='white',
                linewidth=.4,zorder=3)
    bars[1].set_facecolor(PALETTE['memory_fill'])
    for bar,value in zip(bars,vals):
        mx.text(bar.get_x()+bar.get_width()/2,value+2,f'{value:.0f}',
                ha='center',va='bottom',fontsize=9)
    mx.set_xticks([0,1],['AdaJEPA','+ gated\nmemory'],fontsize=8.3)
    mx.set_ylim(0,100);mx.set_xlim(-.6,1.6);mx.set_yticks([0,25,50,75,100])
    mx.tick_params(labelleft=False)
    mx.grid(axis='y',color=PALETTE['grid'],linewidth=.6)
    mx.text(0,1.12,'(b) Maze: 100 tasks',transform=mx.transAxes,fontsize=9.2,
            fontweight='bold')
    mx.text(.5,-.36,'Memory only',transform=mx.transAxes,fontsize=8.5,
            ha='center',va='center')
    for a in (ax,mx):clean(a)
    fig.text(.392,.065,'Checkpoint / planner',ha='center',fontsize=9)
    save(fig,'fig4_results',out)

def distance(data,out):
    d=data['distance_trace']; traces=d['traces']; b=traces['ref']; c=traces['T0SP']
    fig,ax=plt.subplots(figsize=(3.25,1.90))
    fig.subplots_adjust(left=.185,right=.975,bottom=.255,top=.965)
    ax.plot(b['replans'],b['position_error_px'],
            color=PALETTE['native_line'],lw=1.35,ls=(0,(3.5,2.2)),
            marker='s',markersize=2.6,markevery=[0,5,10,15,20,25,30],
            markeredgewidth=.45,label='Frozen MPC',zorder=3)
    ax.plot(c['replans'],c['position_error_px'],
            color=PALETTE['correction_line'],lw=1.5,ls='-',
            marker='o',markersize=2.7,markevery=[0,5,10,15,17],
            markeredgewidth=.45,label='Triggered correction',zorder=4)
    ax.axhline(d['threshold_px'],color=PALETTE['muted'],lw=.7,
               ls=(0,(1.2,2)),zorder=1)
    ax.axvline(d['trigger_after_replan'],color=PALETTE['muted'],
               lw=.7,ls=(0,(1.2,2)),zorder=1)
    ax.text(15.65,170,'Trigger',fontsize=8.1,color=PALETTE['muted'])
    # Endpoint labels distinguish the two lines even in grayscale.
    ax.text(29.5,57,'45.5 px',fontsize=8.1,ha='right',va='bottom',
            color=PALETTE['native_line'])
    ax.annotate('11.0 px',xy=(17,c['position_error_px'][-1]),
                xytext=(20.4,2),fontsize=8.1,ha='left',va='bottom',
                color=PALETTE['correction_line'],
                arrowprops={'arrowstyle':'-','lw':.6,
                            'color':PALETTE['correction_line']})
    ax.text(1,24,'20 px',fontsize=8.0,ha='left',va='bottom',
            color=PALETTE['muted'])
    ymax=max(300.,np.ceil(max(b['position_error_px'])/50)*50)
    ax.set_xlim(0,30);ax.set_ylim(0,ymax)
    ax.set_xticks([0,10,20,30]);ax.set_yticks([0,100,200,300])
    ax.set_xlabel('MPC replan',labelpad=4)
    ax.set_ylabel('Joint position error (px)',labelpad=5)
    ax.legend(frameon=False,loc='upper right',fontsize=8.3,
              handlelength=1.8,handletextpad=.5,borderpad=.15,labelspacing=.4)
    clean(ax)
    save(fig,'figS_pushobj_distance',out)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',type=Path,default=Path(__file__).with_name('plot_data.json'))
    ap.add_argument('--out',type=Path,default=Path(__file__).resolve().parent)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    data=json.loads(a.data.read_text());style();results(data,a.out);distance(data,a.out)
    (a.out/'palette.json').write_text(json.dumps(PALETTE,indent=2)+'\n')
    manifest={'source_data':a.data.name,
              'source_data_sha256':hashlib.sha256(a.data.read_bytes()).hexdigest(),
              'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'palette':PALETTE,'bar_fill':'solid for every bar; no hollow bars',
              'results_canvas_inches':[6.875,2.75],
              'distance_canvas_inches':[3.25,1.90],
              'results_y_limits_percent':[0,100],
              'minimum_font_pt':8.0,
              'no_error_bars':'single recorded intervention runs; no invented uncertainty',
              'grayscale_encoding':'fixed left-to-right series order and labels; curve styles plus square/circle markers',
              'maze_variant':'gated memory only; no proprioceptive boost; not highlighted as a large gain',
              'count_validation':data['pushobj'],
              'changes':'Visual rendering only; underlying archived counts and trace samples unchanged.'}
    (a.out/'figure_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'out':str(a.out),'figures':['fig4_results','figS_pushobj_distance']},indent=2))

if __name__=='__main__':main()
