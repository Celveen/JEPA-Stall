#!/usr/bin/env python3
"""Polish genuine logged trajectories and native simulator frames for paper v8.

Read-only evidence use: no simulator stepping, model inference, training, or
planning. Native state renders are reused; annotations remain vector graphics.
Dashed segments connect replan endpoints, not unobserved physics substeps.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/tmp/jepa_trajectory_v8_mpl')
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle, Circle

P = argparse.ArgumentParser()
P.add_argument('--repo', type=Path, required=True)
P.add_argument('--paper', type=Path, required=True)
args = P.parse_args()
R, O = args.repo, args.paper
F, V = O/'figures', O/'provenance'
for d in [F/'previews', F/'svg', V]: d.mkdir(parents=True, exist_ok=True)
INK = '#243D51'
MUTED = '#61798C'
HISTORY = '#8299AA'
FAIL = '#BF603D'
OURS = '#137F86'
LIGHT_FAIL = '#ECA88F'
LIGHT_OURS = '#69D5CA'
GOLD = '#E9B950'
BORDER = '#CCD6DE'
plt.rcParams.update({'font.family':'serif', 'font.serif':['STIXGeneral'],
    'mathtext.fontset':'stix', 'font.size':8, 'pdf.fonttype':42,
    'ps.fonttype':42, 'svg.fonttype':'none', 'axes.linewidth':.55,
    'savefig.facecolor':'white'})

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
old = json.loads((V/'trajectory_v3_manifest.json').read_text())
source = json.loads((V/'source_manifest.json').read_text())
manifest = {'version':8, 'evidence':'Native simulator images rendered previously at recorded states. Every trajectory point is copied from an existing MPC log; no new execution.',
 'sampling':'One endpoint per MPC replan. Dashed lines and their arrows indicate sequence of sampled endpoints, not the substep physical path. No spline smoothing or invented wall avoidance.',
 'sources':dict(old['sources']), 'frame_sources':{}, 'cases':{},
 'coordinate_transforms':old['coordinate_transforms'], 'outputs':{}}
for rel,expected in old['sources'].items():
    path=R/rel
    assert path.exists(), str(path)
    assert sha(path)==expected, 'Evidence source changed: '+rel

cases={}
for key in ['pushobj','maze']:
    c=old['cases'][key]
    d={'key':key, 'goal':np.array(source['cases'][key]['states']['goal'][:2]),
       'late':15 if key=='pushobj' else 7}
    for mode in ['base','ours']:
        q=np.array(c['states'][mode])
        d[mode]=q
        # Explicitly recheck outcomes and all endpoint positions from the logs.
        folder=R/c['baseline_path' if mode=='base' else 'correction_path']
        rows={}
        for path in sorted(folder.rglob('per_iter_logs.json')):
            for line in path.read_text().splitlines():
                row=json.loads(line)
                if row['sample_idx']==c['sample_idx'] and row['step']<=30:
                    rows[row['step']]=row
        assert len(rows)==len(q)-1
        for j in sorted(rows):np.testing.assert_array_equal(q[j],rows[j]['state'])
        assert bool(rows[max(rows)]['success'])==(mode=='ours')
        image_path=F/'trajectory_frames'/f'{key}_{mode}_final.png'
        manifest['frame_sources'][str(image_path.relative_to(O))]=sha(image_path)
        d[mode+'_image']=np.array(Image.open(image_path))
    if key=='pushobj':
        d['project']=lambda xy: np.asarray(xy)
    else:
        t=old['coordinate_transforms']['maze']
        scale=t['scale_pixels_per_world_unit']; offset=np.array(t['body_offset_xy']); camera=np.array(t['camera_position'][:2])
        d['project']=lambda xy: 256+scale*(np.asarray(xy)+offset-camera)
    cases[key]=d
    manifest['cases'][key]=dict(c)
    manifest['cases'][key]['explicit_outcome_check']={'baseline':False,'ours':True}
    manifest['cases'][key]['display_labels']={'baseline':'Frozen' if key=='pushobj' else 'AdaJEPA','ours':'Ours (frozen)' if key=='pushobj' else 'Ours (AdaJEPA)'}


def halo(line,dark=False,width=2):
    line.set_path_effects([pe.Stroke(linewidth=width,foreground=INK if dark else 'white',alpha=.8),pe.Normal()])


def sequence_path(ax,xy,color,dark=False,lw=1.2,arrows=(),markers=True,z=5):
    """Keep the exact observed polyline; round caps are display styling only."""
    xy=np.asarray(xy)
    if len(xy)<2:return
    line,=ax.plot(xy[:,0],xy[:,1],color=color,lw=lw,ls=(0,(3.2,1.8)),
        dash_capstyle='round',solid_joinstyle='round',zorder=z)
    halo(line,dark,lw+1.05)
    if markers:
        ax.scatter(xy[:,0],xy[:,1],s=3.7,color=color,edgecolors='white' if not dark else INK,linewidths=.25,zorder=z+.1)
    for i in arrows:
        if i<0:i+=len(xy)-1
        if not 0<=i<len(xy)-1:continue
        a,b=xy[i],xy[i+1]
        if np.linalg.norm(b-a)<1e-6:continue
        start=a+.34*(b-a);end=a+.74*(b-a)
        ar=FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=6.7,
            linewidth=.7,color=color,zorder=z+1,shrinkA=0,shrinkB=0)
        ar.set_path_effects([pe.Stroke(linewidth=1.5,foreground=INK if dark else 'white',alpha=.8),pe.Normal()])
        ax.add_patch(ar)


def markers(ax,xy,goal,dark=False,mode='ours',start=True,end=True,goalsize=63):
    color=FAIL if mode=='base' else OURS
    if start:ax.scatter(*xy[0],s=24,marker='o',facecolors='white',edgecolors=INK,linewidths=.9,zorder=12)
    if end:ax.scatter(*xy[-1],s=32,marker='X' if mode=='base' else 'D',facecolors=color,edgecolors='white',linewidths=.7,zorder=13)
    ax.scatter(*goal,s=goalsize,marker='*',facecolors=GOLD,edgecolors=INK,linewidths=.6,zorder=14)


def scene(ax,d,mode,upto=None,image_override=None,full=True):
    key=d['key'];dark=key=='maze';q=d[mode] if upto is None else d[mode][:upto+1]
    xy=d['project'](q[:,:2]);goal=d['project'](d['goal'])
    ax.imshow(d[mode+'_image'] if image_override is None else image_override)
    histcolor='#E3E5D9' if dark else HISTORY
    latecolor=(LIGHT_FAIL if mode=='base' else LIGHT_OURS) if dark else (FAIL if mode=='base' else OURS)
    # Background history is lighter and subordinate to the observed late window.
    sequence_path(ax,xy,histcolor,dark,lw=.85,arrows=([0,4] if key=='pushobj' else [0]),markers=True,z=3)
    late=d['late']
    if len(xy)>late:
        sequence_path(ax,xy[late:],latecolor,dark,lw=1.45,arrows=([0] if key=='pushobj' else ([3,4,-2] if mode=='ours' else [14])),markers=True,z=6)
    markers(ax,xy,goal,dark,mode,start=full,end=full)
    if key=='pushobj':ax.set_xlim(5,285);ax.set_ylim(385,25)
    else:ax.set_xlim(16,450);ax.set_ylim(494,64)
    ax.set_xticks([]);ax.set_yticks([])
    for sp in ax.spines.values():sp.set_color(BORDER);sp.set_linewidth(.6)
    return xy


def badge(fig,x,y,word,color,ha='right',fontsize=8):
    fig.text(x,y,word,color=color,fontsize=fontsize,weight='bold',ha=ha,va='center',
      bbox={'boxstyle':'round,pad=.27,rounding_size=.16','facecolor':'#F8EEE9' if color==FAIL else '#EAF4F3','edgecolor':'none'})


def save(fig,name,w,h):
    for ext,folder in [('pdf',F),('svg',F/'svg'),('png',F/'previews')]:
        p=folder/f'{name}.{ext}';fig.savefig(p,dpi=280 if ext=='png' else None)
        manifest['outputs'][str(p.relative_to(O))]={'sha256':sha(p),'width_inches':w,'height_inches':h,'minimum_font_pt':8}
    plt.close(fig)

# Figure 2: paired native scenes, with a local endpoint plot beside each scene.
w,h=6.875,3.75
fig=plt.figure(figsize=(w,h))
for row,key in enumerate(['pushobj','maze']):
    d=cases[key]
    ytop=.947-row*.46;bot=ytop-.382
    for col,mode in enumerate(['base','ours']):
        x=.025+col*.50
        label=('PushObj · Frozen' if key=='pushobj' else 'Maze · AdaJEPA') if mode=='base' else ('PushObj · Ours (frozen)' if key=='pushobj' else 'Maze · Ours (AdaJEPA)')
        fig.text(x,ytop,f'({chr(97+row*2+col)}) {label}',fontsize=8.7,color=INK,weight='bold',va='center')
        badge(fig,x+.450,ytop,'FAILURE' if mode=='base' else 'SUCCESS',FAIL if mode=='base' else OURS)
        ax=fig.add_axes([x,bot,.225,.342])
        xy=scene(ax,d,mode)
        # Position inset in an unambiguous separate coordinate system.
        ix=x+.267;iy=bot+.098;iw=.164;ih=.206
        ia=fig.add_axes([ix,iy,iw,ih])
        q=d[mode][d['late']:,:2];q=q-q[0]
        color=FAIL if mode=='base' else OURS
        sequence_path(ia,q,color,False,lw=1.05,arrows=([0] if mode=='ours' and key=='pushobj' else []),z=3)
        ia.scatter(0,0,s=16,facecolors='white',edgecolors=INK,linewidths=.7,zorder=6)
        ia.scatter(*q[-1],s=23,marker='X' if mode=='base' else 'D',facecolors=color,edgecolors='white',linewidths=.5,zorder=7)
        ia.set_aspect('equal',adjustable='box');ia.invert_yaxis()
        ia.spines[['top','right']].set_visible(False)
        ia.spines[['left','bottom']].set_color(BORDER)
        ia.tick_params(labelsize=8,length=2,pad=1)
        if key=='pushobj' and mode=='base':
            ia.set_xlim(-2.3,2.9);ia.set_ylim(2.0,-1.8);ia.set_xticks([-2,0,2]);ia.set_yticks([-1,1])
        elif key=='pushobj':
            ia.set_xlim(-3,21);ia.set_ylim(3,-38);ia.set_xticks([0,20]);ia.set_yticks([-30,0])
        elif mode=='base':
            ia.set_xlim(-.6,1);ia.set_ylim(.28,-3.1);ia.set_xticks([0]);ia.set_yticks([-3,0])
        else:
            ia.set_xlim(-4.3,1.7);ia.set_ylim(3.6,-.8);ia.set_xticks([-4,0]);ia.set_yticks([0,3])
        unit='px' if key=='pushobj' else 'units'
        ia.set_xlabel(r'$\Delta x$ ('+unit+')',labelpad=.3,fontsize=8)
        ia.set_ylabel(r'$\Delta y$',labelpad=.3,fontsize=8)
        n=len(d[mode])-1
        fig.text(ix+iw/2,bot+.327,f'Replans {d["late"]}–{n}',fontsize=8,color=MUTED,ha='center')
        stats=old['cases'][key]['late_window_statistics'][mode]
        net=stats['net_displacement'];netstr=f'{net:.2f}' if net>=.001 else '<0.001'
        fig.text(ix+iw/2,bot+.006,f'Net: {netstr} {unit}',fontsize=8,color=color,ha='center',weight='bold')
        fig.text(x+.1125,bot-.023,f'Replan {n}',fontsize=8,color=MUTED,ha='center')
handles=[Line2D([0],[0],color=HISTORY,lw=.9,ls=(0,(3,2)),marker='.',ms=3,label='Earlier endpoints'),
 Line2D([0],[0],color=OURS,lw=1.3,ls=(0,(3,2)),marker='.',ms=3,label='Late endpoints'),
 Line2D([0],[0],marker='o',color='none',mfc='white',mec=INK,ms=4.5,label='Start'),
 Line2D([0],[0],marker='X',color='none',mfc=FAIL,mec='white',ms=5.5,label='Failure'),
 Line2D([0],[0],marker='D',color='none',mfc=OURS,mec='white',ms=4.5,label='Success'),
 Line2D([0],[0],marker='*',color='none',mfc=GOLD,mec=INK,ms=7,label='Goal')]
fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.001),ncol=6,frameon=False,fontsize=8,handlelength=1.4,columnspacing=.9,handletextpad=.4)
save(fig,'fig1_trajectory_cases',w,h)


manifest['separator_rule_removed']=True
manifest['version']=10
(V/'trajectory_v10_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
