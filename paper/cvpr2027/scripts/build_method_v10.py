#!/usr/bin/env python3
"""Original vector overview with continuous, explicitly routed arrow paths."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path as MPath
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle, Polygon
P=argparse.ArgumentParser();P.add_argument('--paper',type=Path,required=True);a=P.parse_args();F=a.paper/'figures'
plt.rcParams.update({'font.family':'serif','font.serif':['STIXGeneral'],'mathtext.fontset':'stix','font.size':8,'pdf.fonttype':42,'svg.fonttype':'none'})
W,H=468,234
fig=plt.figure(figsize=(W/72,H/72));ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,W),ylim=(0,H));ax.axis('off')
INK='#25495E';TEAL='#117E84';ORANGE='#BA633E';MUTED='#627784';PALE='#EBF2F6';GREEN='#EDF7F5';EDGE='#BCCDD6';sources={};texts=[];edges=[]
def text(x,y,s,size=8.5,ha='center',color=INK,bold=False):
 t=ax.text(x,y,s,fontsize=size,ha=ha,va='center',color=color,weight='bold' if bold else 'normal',linespacing=1.1,zorder=10);texts.append(t);return t
def box(x,y,w,h,fill='white',edge=EDGE,lw=.8):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=3.5',fc=fill,ec=edge,lw=lw,zorder=1))
def arrow(points,color=INK,lw=1.65,name='',head=8):
 pts=np.asarray(points,float);codes=[MPath.MOVETO]+[MPath.LINETO]*(len(pts)-1)
 patch=FancyArrowPatch(path=MPath(pts,codes),arrowstyle='-|>',mutation_scale=head,linewidth=lw,color=color,capstyle='round',joinstyle='round',zorder=6)
 ax.add_patch(patch);edges.append({'name':name,'points':pts.tolist(),'line_width_pt':lw,'head_size':head,'continuous_path':True})
def thumb(name,x,y,size):
 p=F/'frames'/name;im=np.array(Image.open(p));ax.imshow(im,extent=(x,x+size,y,y+size),zorder=2,interpolation='lanczos');sources[name]=hashlib.sha256(p.read_bytes()).hexdigest()
 ax.add_patch(Rectangle((x,y),size,size,fc='none',ec=EDGE,lw=.6,zorder=3))
# Offline data and learning form a separate, compact band.
box(3,190,462,41,fill='#F5F7F9',edge='#D8E1E6',lw=.6)
text(11,222,'OFFLINE TRAINING',8,ha='left',bold=True)
ax.add_patch(Circle((18,203),3,fc='#5384B9',ec=INK,lw=.7));ax.add_patch(Rectangle((34,195),7,17,fc='#90A8BA',ec=INK,lw=.7));arrow([(23,203),(31,203)],ORANGE,1.5,head=6)
text(50,203,'Interaction-biased\ndata augmentation',8.5,ha='left')
arrow([(144,203),(162,203)],name='augmented trajectories to training')
box(169,195,119,24,PALE,INK)
text(228.5,211,'Predictive learning',8.7,bold=True);text(228.5,201,'observations + actions',8)
arrow([(294,203),(312,203)],name='trained parameters')
box(319,195,137,24,'white',INK)
text(387.5,211,'Learned world model',8.7,bold=True);text(387.5,201,r'Encoder $E_\theta$ + predictor $P_\theta$',8)
arrow([(387,191),(387,181),(128,181),(128,150)],lw=1.5,name='initialize planning model')
arrow([(256,181),(256,169)],lw=1.5,name='initialize predictor')
text(11,179,'ONLINE CONTROL',8,ha='left',bold=True)
# A single forward row: observations -> encoding -> predictive planning -> execution.
box(4,95,84,73)
text(46,159,'1  State and goal',8.5,bold=True)
thumb('push_start.png',10,115,31);thumb('push_goal.png',50,115,31)
text(25.5,104,r'$o_t$',9);text(65.5,104,r'$o_g$',9)
arrow([(91,131),(102,131)],name='observations to encoder')
ax.add_patch(Polygon([(107,149),(150,143),(150,118),(107,112)],fc=PALE,ec=INK,lw=1.15,zorder=2))
text(128,131,r'$E_\theta$',14)
text(128,103,'Encode',8.5)
arrow([(153,131),(166,131)],name='latent observations')
text(159,149,r'$z_t,z_g$',8.5)
box(170,95,174,73,PALE,INK,1.0)
text(257,159,'2  Predict and optimize actions',9,bold=True)
text(190,137,r'$P_\theta$',12)
arrow([(205,136),(219,136)],lw=1.5,name='latent rollouts')
ax.plot([225,239,253,270,286,304],[124,139,144,146,144,142],color='#ADBDC7',lw=1.0)
ax.plot([225,242,256,276,292,309],[124,120,116,115,117,119],color='#ADBDC7',lw=1.0)
arrow([(225,124),(241,130),(259,132),(280,138),(309,145)],lw=1.9,name='selected latent rollout',head=7)
ax.scatter([225],[124],s=9,c=INK,zorder=7);ax.scatter([316],[147],s=38,marker='*',c=TEAL,zorder=8)
text(323,134,r'$z_g$',8)
text(257,109,r'$J_{\rm goal}(A;\alpha_t)+\kappa g_t\mathcal{R}^{\rm stall}_t(A)$',10)
arrow([(348,131),(362,131)],lw=1.9,name='execute optimized prefix')
text(356,145,r'$a_t$',9)
box(366,95,98,73)
text(415,159,'3  Execute first block',8.6,bold=True)
thumb('push_before.png',377,116,29);thumb('maze_original_start.png',423,116,29)
text(391.5,105,'PushObj',8);text(437.5,105,'Maze',8)
# Feedback is a consistent right-to-left row, returning only to the action cost.
arrow([(415,92),(415,74)],TEAL,1.8,name='realized transition to feedback')
box(353,10,111,62,GREEN,TEAL)
text(408.5,62,'Executed transition',8.8,bold=True,color=TEAL)
thumb('push_before.png',364,25,25);thumb('push_baseline_end.png',426,25,25)
arrow([(393,37),(421,37)],TEAL,1.5,name='observed motion',head=7)
text(407,46,r'$a_t$',8)
text(376.5,18,r'$o_t$',8);text(438.5,18,r'$o_{t+1}$',8)
arrow([(350,40),(336,40)],TEAL,1.8,name='feedback to stall detection')
box(224,10,109,62,GREEN,TEAL)
text(278.5,62,'4  Detect stagnation',8.8,bold=True,color=TEAL)
text(278.5,45,'Net movement',8.7)
ax.plot([240,317],[37,37],color=TEAL,lw=.8)
text(278.5,29,'Distance to goal',8.7)
text(278.5,16,'Small ratio → activate',8,color=TEAL)
arrow([(221,40),(209,40)],TEAL,1.8,name='activation of ECP')
box(4,10,202,62,GREEN,TEAL)
text(105,62,'5  Correct the planning objective',8.8,bold=True,color=TEAL)
text(61,47,'Stagnation memory (S)',8.2,bold=True,color=TEAL)
ax.add_patch(Circle((30,28),9,fc='#D5E9E5',ec=TEAL,lw=.8))
ax.scatter([26,29,33],[29,25,31],s=7,color=TEAL,zorder=4)
arrow([(13,21),(19,37),(35,39),(50,34),(64,25)],TEAL,1.4,name='avoid stalled neighborhood schematic',head=6)
text(91,28,'Avoid revisits',8,color=TEAL)
text(157,47,'Goal weighting (P)',8.2,bold=True,color=TEAL)
for x,hh,c in [(132,10,INK),(142,3,ORANGE),(174,10,INK),(184,18,ORANGE)]:ax.add_patch(Rectangle((x,18),6,hh,fc=c,ec='none',zorder=3))
arrow([(152,26),(168,26)],TEAL,1.3,name='temporary proprioceptive reweighting',head=6)
# One uninterrupted elbow connects the correction to the cost it modifies.
arrow([(105,75),(105,84),(257,84),(257,93)],TEAL,1.8,name='ECP changes action objective')
text(313,82,'Next plan',8,color=TEAL)
text(105,4,'World model: frozen or online-adapted',8,color=MUTED)
text(364,4,'Replan from each new observation',8,color=MUTED)
fig.canvas.draw();renderer=fig.canvas.get_renderer();overlaps=[]
for i,t in enumerate(texts):
 for u in texts[i+1:]:
  b=t.get_window_extent(renderer);c=u.get_window_extent(renderer)
  if b.overlaps(c):overlaps.append([t.get_text(),u.get_text()])
for ext,sub in [('pdf',''),('svg','svg'),('png','previews')]:
 path=F/sub/f'fig2_method.{ext}';path.parent.mkdir(exist_ok=True);fig.savefig(path,dpi=280)
m={'version':10,'canvas_inches':[W/72,H/72],'minimum_font_pt':8,'minimum_font_in_manuscript_pt':8,
 'changes':['single continuous path per arrow; no independently shrunk last segment','consistent arrow heads and1.5–1.9pt main strokes','three separated phases; five numbered online stages','online adaptation shown as a model regime, not as an ECP parameter update','independent action-offset ablation removed from the central method diagram'],
 'arrows':edges,'text_overlap_candidates':overlaps,'native_frame_hashes':sources,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(a.paper/'provenance/method_v10_manifest.json').write_text(json.dumps(m,indent=2)+'\n');print(json.dumps({'overlaps':overlaps,'arrows':len(edges)}))
