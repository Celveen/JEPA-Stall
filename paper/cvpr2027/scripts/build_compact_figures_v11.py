#!/usr/bin/env python3
"""Compact, true-simulation trajectory figures at final CVPR column dimensions."""
import argparse,json,hashlib,os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/jepa_v11_mpl')
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
p=argparse.ArgumentParser();p.add_argument('--paper',type=Path,required=True);a=p.parse_args();O=a.paper;F=O/'figures';V=O/'provenance'
data=json.loads((V/'replayed_paths_v11.json').read_text());cases={x['key']:x for x in data['cases']}
plt.rcParams.update({'font.family':'serif','font.serif':['STIXGeneral'],'font.size':9,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
INK='#243D51';FAIL='#B95B38';OURS='#087D83';GOLD='#E7B33C';MUTED='#5F707D'
manifest={'revision':11,'source':'replayed_paths_v11.json','path_type':'solid polylines through recorded simulator integration steps','marker_spacing':'every five executed control calls','smoothing':False,'outputs':{}}

def project(c,xy):
 if c['task']=='PushObj':return np.asarray(xy)
 t=c['projection'];return 256+t['scale_pixels_per_world_unit']*(np.asarray(xy)+np.array(t['body_offset_xy'])-np.array(t['camera_position'][:2]))

def panel(ax,c,mode,small=False):
 ax.imshow(np.array(Image.open(O/c['frames'][mode]['path'])))
 x0,y0,x1,y1=c['crop'];ax.set(xlim=(x0,x1),ylim=(y1,y0));ax.set_xticks([]);ax.set_yticks([])
 for sp in ax.spines.values():sp.set_visible(False)
 dark=c['task']=='Maze';color=('#F1AC8D' if mode=='base' else '#81DBCE') if dark else (FAIL if mode=='base' else OURS)
 if mode in ['base','ours']:
  xy=project(c,np.array(c['paths'][mode])[:,:2]);dots=project(c,np.array(c['control_states'][mode])[5::5,:2]);lw=.75 if small else 1.0
  line,=ax.plot(xy[:,0],xy[:,1],color=color,lw=lw,solid_capstyle='round',solid_joinstyle='round',zorder=4)
  line.set_path_effects([pe.Stroke(linewidth=lw+.8,foreground=INK if dark else 'white',alpha=.85),pe.Normal()])
  ax.scatter(dots[:,0],dots[:,1],s=3.0 if small else 5.0,color=color,edgecolors=INK if dark else 'white',linewidths=.22,zorder=6)
  # Put short directional arrows on the actual local tangent, never across a turn.
  cumulative=np.r_[0,np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))];chosen=[]
  for q in ([.28,.66] if small else [.2,.43,.7]):
   i=int(np.searchsorted(cumulative,q*cumulative[-1]));j=min(i+1,len(xy)-1)
   while j<len(xy)-1 and np.linalg.norm(xy[j]-xy[i])<3:j+=1
   delta=xy[j]-xy[i];norm=float(np.linalg.norm(delta))
   if norm<.3 or any(np.linalg.norm(xy[i]-x)<18 for x in chosen):continue
   chosen.append(xy[i]);start=xy[i];end=xy[j]
   ar=FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=5 if small else 6,linewidth=.7,color=color,shrinkA=0,shrinkB=0,zorder=8)
   ar.set_path_effects([pe.Stroke(linewidth=1.35,foreground=INK if dark else 'white'),pe.Normal()]);ax.add_patch(ar)
  ax.scatter(*xy[0],s=13 if small else 22,marker='o',fc='white',ec=INK,lw=.65,zorder=9)
  ax.scatter(*xy[-1],s=20 if small else 33,marker='X' if mode=='base' else 'D',fc=color,ec='white',lw=.6,zorder=10)
 ax.scatter(*project(c,np.array(c['goal'])[:2]),s=32 if small else 47,marker='*',fc=GOLD,ec=INK,lw=.5,zorder=11)

def legend(fig,y=.018,small=False):
 handles=[Line2D([0],[0],color=OURS,lw=1,marker='.',ms=3,label='Path; dot every 5 controls'),Line2D([0],[0],marker='o',color='none',mfc='white',mec=INK,ms=4,label='Start'),Line2D([0],[0],marker='*',color='none',mfc=GOLD,mec=INK,ms=6,label='Goal'),Line2D([0],[0],marker='X',color='none',mfc=FAIL,mec='white',ms=5,label='Failure'),Line2D([0],[0],marker='D',color='none',mfc=OURS,mec='white',ms=4,label='Success')]
 fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,y),ncol=5,frameon=False,fontsize=8.4,handlelength=1.3,handletextpad=.35,columnspacing=.9,borderaxespad=0)

def save(fig,name,rows,panels):
 fig.canvas.draw();renderer=fig.canvas.get_renderer();texts=[t for t in fig.texts if t.get_text()]
 boxes=[t.get_window_extent(renderer) for t in texts]
 overlaps=[(texts[i].get_text(),texts[j].get_text()) for i in range(len(boxes)) for j in range(i+1,len(boxes)) if boxes[i].overlaps(boxes[j])]
 assert not overlaps,(name,overlaps)
 for ext,sub in [('pdf',''),('svg','svg'),('png','previews')]:
  path=F/sub/f'{name}.{ext}';path.parent.mkdir(exist_ok=True);fig.savefig(path,dpi=300)
  manifest['outputs'][str(path.relative_to(O))]={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'size_inches':fig.get_size_inches().tolist(),'rows':rows,'simulation_panels':panels,'overlapping_labels':0}
 plt.close(fig)

# Eight panels: four frozen, four adaptive; two matched domain pairs per row.
fig=plt.figure(figsize=(6.875,3.25))
for row,keys in enumerate([['I','maze_frozen'],['pushobj_ada','maze_ada']]):
 top=.98-row*.50;bottom=.56-row*.50
 for pair,key in enumerate(keys):
  c=cases[key];x=.012+pair*.5
  fig.text(x+.238,top,('PushObj' if c['task']=='PushObj' else 'Maze')+' · '+c['baseline'],fontsize=9.5,weight='bold',color=INK,ha='center',va='top')
  for j,mode in enumerate(['base','ours']):
   left=x+j*.242;ax=fig.add_axes([left,bottom,.23,.355]);panel(ax,c,mode)
   n=len(c['states'][mode])-1;label=('Native: failure' if mode=='base' else 'ECP: success')+f' ({n})'
   fig.text(left+.115,bottom-.015,label,fontsize=8.8,color=FAIL if mode=='base' else OURS,ha='center',va='top')
save(fig,'fig1_trajectory_cases',2,8)

# Eight paired recoveries in two rows. The I-shape case is already in Figure 2.
fig=plt.figure(figsize=(6.875,2.52))
order=['L','Z','T','plus','square','small_tee','maze_frozen','maze_ada']
for k,key in enumerate(order):
 c=cases[key];row,col=divmod(k,4);x=.009+col*.25;top=.985-row*.51;bottom=.58-row*.50
 title=c['label']+' · Frozen' if c['task']=='PushObj' else 'Maze · '+c['baseline']
 fig.text(x+.118,top,f'({chr(97+k)}) {title}',fontsize=8.7,weight='bold',color=INK,ha='center',va='top')
 for j,mode in enumerate(['base','ours']):
  left=x+j*.121;ax=fig.add_axes([left,bottom,.114,.305]);panel(ax,c,mode,small=True)
  n=len(c['states'][mode])-1
  fig.text(left+.057,bottom-.023,('Fail' if mode=='base' else 'Success')+f' · {n}',fontsize=8.2,color=FAIL if mode=='base' else OURS,ha='center',va='top')
save(fig,'fig5_execution_recovery',2,16)

# The supplementary views preserve start and goal frames and use the same dense paths.
old=json.loads((V/'gallery_v10_manifest.json').read_text());original={x['key']:x for x in old['cases']}
fig=plt.figure(figsize=(6.875,6.7))
for k,key in enumerate(['I','L','Z','T','plus','square','small_tee','maze_frozen','maze_ada']):
 c=cases[key];c['frames'].update({m:original[key]['frames'][m] for m in ['start','goal']})
 row,col=divmod(k,2);x=.012+col*.5;top=.982-row*.194;bottom=top-.171
 name='Maze' if c['task']=='Maze' else c['label']
 fig.text(x,top,f'({chr(97+k)}) {name} · {c["baseline"]} / ECP',fontsize=9,color=INK,weight='bold',va='top')
 for j,mode in enumerate(['start','goal','base','ours']):
  left=x+j*.121;ax=fig.add_axes([left,bottom,.114,.112]);panel(ax,c,mode,small=True)
  label={'start':'Start','goal':'Goal','base':'Failure','ours':'ECP success'}[mode]
  fig.text(left+.057,top-.027,label,fontsize=8.3,color=INK if mode in ['start','goal'] else (FAIL if mode=='base' else OURS),ha='center',va='top')
legend(fig,.001,True);save(fig,'figS_diverse_execution_v11',5,36)
(V/'compact_figures_v11.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Built compact Figure 2 (8 panels), Figure 4 (2 rows), and supplementary trajectories')
