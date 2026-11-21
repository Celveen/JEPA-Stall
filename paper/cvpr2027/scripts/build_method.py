#!/usr/bin/env python3
"""Execution-conditioned planning overview. All geometry is original vector art.
Thumbnails are pre-existing simulator-state renderings. No experiments are run.
"""
from pathlib import Path
import argparse, json, hashlib
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Polygon, Circle, Arc
AP=argparse.ArgumentParser();AP.add_argument('--frames',type=Path,default=Path('/home/appuser/workspace/project/multimodal-test/JEPA-Stall/paper/cvpr2027/figures/frames'));AP.add_argument('--out',type=Path,default=Path(__file__).resolve().parent);a=AP.parse_args();a.out.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'serif','font.serif':['STIXGeneral'],'mathtext.fontset':'stix','font.size':8.5,'pdf.fonttype':42,'ps.fonttype':42})
W,H=495,216
f=plt.figure(figsize=(W/72,H/72));ax=f.add_axes([0,0,1,1]);ax.set_xlim(0,W);ax.set_ylim(0,H);ax.axis('off')
INK='#2C4C63';BLUE='#2C4C63';MUTED='#8E9AA3';PALE='#EBF3F8';GREEN='#477C82';PGREEN='#EEF5EF';ORANGE='#B56E4F';GRAY='#F5F7F8'
texts=[];sources={}
def text(x,y,s,sz=8.5,ha='center',va='center',c=INK,bold=False):
 t=ax.text(x,y,s,ha=ha,va=va,fontsize=sz,color=c,weight='bold' if bold else 'normal',linespacing=1.03,zorder=10);texts.append(t);return t
def box(x,y,w,h,fill='white',edge=MUTED,rad=3,lw=.8,ls='-'):
 p=FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={rad}',facecolor=fill,edgecolor=edge,linewidth=lw,linestyle=ls,zorder=1);ax.add_patch(p);return p
def arrow(x1,y1,x2,y2,c=INK,lw=1.,ls='-',ms=6,conn='arc3,rad=0'):
 ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=ms,linewidth=lw,color=c,linestyle=ls,connectionstyle=conn,zorder=5))
def path(points,c=INK,lw=1.,ls='-',head=True):
 points=np.array(points);ax.plot(points[:-1,0],points[:-1,1],c=c,lw=lw,ls=ls,zorder=4)
 if head:arrow(*points[-2],*points[-1],c,lw,ls)
 else:ax.plot(points[-2:,0],points[-2:,1],c=c,lw=lw,ls=ls,zorder=4)
def thumb(name,x,y,w,h):
 p=a.frames/name;arr=np.asarray(Image.open(p));ax.imshow(arr,extent=[x,x+w,y,y+h],aspect='auto',interpolation='lanczos',zorder=3)
 ax.add_patch(Rectangle((x,y),w,h,fill=False,edgecolor='#C3CCD2',lw=.45,zorder=4));sources[name]=hashlib.sha256(p.read_bytes()).hexdigest()
def tiles(x,y,scale=3):
 cs=[BLUE,'#D4E7EE',MUTED]
 for j in range(3):
  for i in range(3):ax.add_patch(Rectangle((x+i*(scale+1),y+j*(scale+1)),scale,scale,fc=cs[(i+2*j)%3],ec='none',zorder=3))
def network(x,y,w=20,h=19):
 ys=[[y+h*.25,y+h*.75],[y,y+h*.5,y+h],[y+h*.25,y+h*.75]]
 for k in range(2):
  for yy in ys[k]:
   for yy2 in ys[k+1]:ax.plot([x+k*w/2,x+(k+1)*w/2],[yy,yy2],c=MUTED,lw=.55,zorder=2)
 for k in range(3):
  for yy in ys[k]:ax.add_patch(Circle((x+k*w/2,yy),1.6,fc='white',ec=BLUE,lw=.75,zorder=4))

# Training interface: a compact strip, no extra representation loss.
box(3,174,489,39,fill=GRAY,edge='#D6DEE3',rad=4,lw=.6)
text(11,205,'TRAINING',8.5,ha='left',bold=True)
# Schematic contact-biased collection: two pusher-action/object pairs.
ax.add_patch(Rectangle((10,179),31,18,fc='white',ec='#CDD5DB',lw=.6))
ax.add_patch(Polygon([[28,181],[34,181],[34,193],[28,193]],fc='#93A8B7',ec=BLUE,lw=.5))
ax.add_patch(Circle((17,186),2.5,fc='#4E7FD2',ec=BLUE,lw=.5));arrow(20,186,27,186,ORANGE,.9,ms=5)
text(49,188,'Contact-biased\ntransitions',8.5,ha='left')
arrow(119,188,143,188,BLUE)
box(149,179,113,26,fill=PALE,edge=BLUE,rad=3,lw=.7)
text(205,197,'Predictive learning',9,bold=True)
text(205,185,'image + action sequences',8)
arrow(266,188,295,188,BLUE)
box(299,179,184,26,fill='white',edge=BLUE,rad=3,lw=.7)
text(310,192,'Learned world model',8.5,ha='left')
text(440,192,r'$E_\theta$',11,c=BLUE);arrow(450,192,462,192,BLUE,.8);text(473,192,r'$P_\theta$',11,c=BLUE)
text(391,183,'encoder + predictor',8)
# Initialization arrow routed above the optional action branch.
path([(391,178),(391,169),(247,169),(247,151)],BLUE,.85)
text(10,160,'PLANNING',8.5,ha='left',bold=True)

# Actual visual inputs.
box(3,87,82,65,fill='white',edge='#C3CDD4',rad=3,lw=.6)
text(44,146,'Observation + goal',8.5)
thumb('push_start.png',8,107,33,33);thumb('push_goal.png',47,107,33,33)
text(24.5,98,r'$o_t$',9);text(63.5,98,r'$o_g$',9)
arrow(86,121,99,121,INK,1.2)
# Encoder glyph and explicitly separate visual/proprioceptive code.
ax.add_patch(Polygon([[101,139],[132,133],[132,105],[101,99]],fc=PALE,ec=BLUE,lw=1.0,zorder=2))
text(116.5,119,r'$E_\theta$',13,c=BLUE)
text(116.5,90,'Encoder',8.5)
arrow(134,121,146,121,INK,1.)
tiles(150,116,3)
text(156,138,r'$z_t,z_g$',9,c=BLUE)
arrow(165,121,181,121,INK,1.)

# Latent prediction and action optimization as a distinct model block.
box(182,86,151,65,fill=PALE,edge=BLUE,rad=4,lw=.8)
text(257.5,143,'Latent MPC',9.5,bold=True,c=BLUE)
network(192,112,22,18);text(203,103,r'$P_\theta$',10,c=BLUE)
arrow(217,121,229,121,BLUE,.9)
# Three candidate trajectories and a selected path in latent space.
ax.plot([233,248,264,284,310],[110,126,129,131,129],c='#A6B5BF',lw=.7,zorder=3)
ax.plot([233,250,270,288,310],[110,110,105,104,103],c='#A6B5BF',lw=.7,zorder=3)
path([(233,110),(248,116),(266,119),(286,123),(310,126)],BLUE,1.4)
ax.add_patch(Circle((233,110),1.4,fc=BLUE,ec=BLUE,zorder=5));ax.scatter([313],[129],marker='*',s=26,color=GREEN,zorder=5)
text(234,101,r'$z_t$',8.5,c=BLUE);text(316,137,r'$z_g$',8.5,c=GREEN)
text(278,91,'Optimize actions',8.5)
arrow(335,121,349,121,INK,1.2)
# Executed prefix symbol.
text(372,139,'Execute',9)
for i in range(4):ax.add_patch(Rectangle((352+i*9,115),7,10,fc=BLUE if i==0 else '#E1E9EE',ec=BLUE if i==0 else MUTED,lw=.5))
text(372,105,r'$a_t$',10)
arrow(390,121,409,121,INK,1.2)

# Independent action-offset branch: the dashed edge is explicitly optional.
box(340,149,69,15,fill='white',edge=ORANGE,rad=3,lw=.7,ls=(0,(3,2)))
text(374.5,156.5,'Action offset',8,c=ORANGE)
arrow(374.5,148,374.5,143,ORANGE,.8,ls=(0,(3,2)),ms=5)
text(449,157,'Dashed: optional',8,c=MUTED)

# Native environment thumbnails; no changed colors or generative imagery.
box(411,87,81,65,fill='white',edge='#C3CDD4',rad=3,lw=.6)
text(451.5,146,'Environment',8.5)
thumb('push_before.png',416,107,32,32);thumb('maze_original_start.png',454,107,32,32)
text(432,98,'PushObj',8);text(470,98,'Maze',8)
arrow(451.5,85,451.5,70,GREEN,1.1)

# Feedback route, scoped to the current episode.
box(405,10,87,59,fill=PGREEN,edge=GREEN,rad=4,lw=.8)
text(448.5,60,'Execution feedback',8.5,bold=True,c=GREEN)
# Tiny transition sketch with image tokens and executed action.
box(414,29,19,16,fill='white',edge=MUTED,rad=1,lw=.6)
box(466,29,19,16,fill='white',edge=MUTED,rad=1,lw=.6)
tiles(419,33,2);tiles(471,33,2)
arrow(437,37,461,37,GREEN,.9)
text(448.5,46,r'$a_t$',8.5,c=GREEN)
text(423.5,21,r'$o_t$',8.5);text(475.5,21,r'$o_{t+1}$',8.5)
arrow(403,39.5,390,39.5,GREEN,1.1)

box(279,10,109,59,fill=PGREEN,edge=GREEN,rad=4,lw=.8)
text(333.5,60,'Normalized stall signal',8.5,bold=True,c=GREEN)
# History with small net movement and a longer path: visualizes the ratio.
xs=np.array([289,295,301,304,300,295,291]);ys=np.array([31,38,32,26,22,24,29])
ax.plot(xs,ys,color=GREEN,lw=.95,marker='o',markersize=1.9,zorder=4)
arrow(289,31,291,29,ORANGE,.85,ms=4)
ax.plot([291,305],[29,47],c=MUTED,lw=.6,ls=(0,(2,2)),zorder=2);ax.scatter([305],[47],marker='*',s=22,color=GREEN,zorder=5)
text(345,35,'net movement',8.5);ax.plot([313,377],[29,29],c=GREEN,lw=.65,zorder=4);text(345,22,'goal distance',8.5)
arrow(277,39.5,266,39.5,GREEN,1.1)

# Two distinct interventions, with local memory explicitly episode-scoped.
box(79,10,185,59,fill=PGREEN,edge=GREEN,rad=4,lw=.8)
ax.plot([172,172],[16,63],c='#C1D5CA',lw=.6,zorder=2)
text(126,60,'Goal reweighting',8.5,bold=True,c=GREEN)
# Schematic component weights. Only the proprioceptive component changes.
for x0,heights in [(94,[14,2]),(137,[14,20])]:
 for j,hh in enumerate(heights):ax.add_patch(Rectangle((x0+j*10,27),6,hh,fc=BLUE if j==0 else '#D99270',ec='none',zorder=3))
arrow(119,35,130,35,GREEN,.8,ms=5)
text(100,22,r'$v\quad p$',8);text(143,22,r'$v\quad p$',8)
text(126,14,'temporary',8,c=GREEN)
text(218.5,60,'Stagnation regularizer',8,bold=True,c=GREEN)
ax.add_patch(Circle((204,34),8.5,fc='#DEE9E2',ec=GREEN,lw=.65,ls=(0,(2,2)),zorder=2))
ax.scatter([199,203,207,204],[34,31,35,38],s=8,c=GREEN,zorder=4)
path([(187,28),(190,42),(204,47),(219,45),(242,35)],BLUE,1.0)
text(218.5,18,'episode-local memory',8,c=GREEN)
path([(245,70),(245,79),(268,79),(268,85)],GREEN,1.0)
text(302,78,'planning cost',8,c=GREEN)

# Optional within-episode online update, separate from feedback correction.
box(3,10,66,59,fill='white',edge=BLUE,rad=4,lw=.8,ls=(0,(3,2)))
text(36,60,'Online update',8.5,c=BLUE)
ax.add_patch(Arc((36,36),20,18,theta1=20,theta2=320,lw=1.1,color=BLUE,zorder=4))
arrow(45,30,46,36,BLUE,.9,ms=5)
text(36,36,r'$\theta$',11,c=BLUE)
text(36,18,'optional',8,c=BLUE)
path([(405,13),(400,13),(400,3),(36,3),(36,9)],BLUE,.7,ls=(0,(3,2)))
path([(36,70),(36,78),(202,78),(202,85)],BLUE,.7,ls=(0,(3,2)))
path([(95,78),(95,111),(100,111)],BLUE,.7,ls=(0,(3,2)))

# Draw all labels as editable text; report possible text intersections for inspection.
f.canvas.draw();ren=f.canvas.get_renderer();overlaps=[]
for i,t in enumerate(texts):
 b=t.get_window_extent(ren)
 for u in texts[i+1:]:
  c=u.get_window_extent(ren)
  if b.overlaps(c):
   inter=max(0,min(b.x1,c.x1)-max(b.x0,c.x0))*max(0,min(b.y1,c.y1)-max(b.y0,c.y0))
   if inter>1:overlaps.append([t.get_text(),u.get_text(),round(inter,1)])
for ext in ['pdf','svg','png']:f.savefig(a.out/f'fig2_method.{ext}',dpi=240)
manifest={'canvas_inches':[W/72,H/72],'minimum_font_pt':8,'reference_style':'User-provided mainfig1.svg; original layout and component schematics, no copied mechanism','real_thumbnails_sha256':sources,'scope':['training contact-biased collection','inherited predictive training objective credited in manuscript','shared encoder/predictor','latent MPC','normalized realized movement proxy','temporary proprioceptive weights','current-episode local latent memory','optional independent action offset','optional within-episode model updates'],'excluded':['persistent cross-episode memory','maturity gate','trained phase classifier','successful B2/B3 representation regularizer'],'text_bbox_overlap_candidates':overlaps,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(a.out/'method_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'text_overlap_candidates':overlaps,'outputs':['fig2_method.pdf','fig2_method.svg','fig2_method.png']},indent=2))
