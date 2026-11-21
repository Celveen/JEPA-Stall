"""Native paired scenes, isolated stall traces, and disjoint recovery examples."""
import argparse,json,hashlib,os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/jepa_v12_mpl')
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch,Rectangle
from matplotlib.ticker import MaxNLocator

p=argparse.ArgumentParser();p.add_argument('--paper',type=Path,required=True);a=p.parse_args();O=a.paper;F=O/'figures';V=O/'provenance'
source=json.loads((V/'replayed_paths_v12.json').read_text());cases={x['key']:x for x in source['cases']}
INK='#243D51';FAIL='#B95B38';OURS='#087D83';GOLD='#E7B33C'
plt.rcParams.update({'font.family':'serif','font.serif':['STIXGeneral'],'font.size':8.5,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white','axes.unicode_minus':False})
main=['I','maze_frozen','pushobj_ada','maze_ada']
gallery=['L','Z','T','plus','square','small_tee','maze_frozen_alt','maze_ada_alt']
assert not set(main)&set(gallery)
def identity(c):return np.r_[c['start'],c['goal']].tobytes()
assert not {identity(cases[k]) for k in main}&{identity(cases[k]) for k in gallery}
manifest={'revision':12,'source':'replayed_paths_v12.json','figure2_case_keys':main,'figure4_case_keys':gallery,'shared_case_keys':[],'shared_initial_goal_pairs':0,'paths':'solid physics replay','dots_every_controls':5,'inset_statistics':{},'outputs':{}}

def project(c,xy):
 xy=np.asarray(xy)
 if c['task']=='PushObj':return xy
 t=c['projection'];return 256+t['scale_pixels_per_world_unit']*(xy+np.array(t['body_offset_xy'])-np.array(t['camera_position'][:2]))

def stroke(ax,xy,color,width,dark=False,arrow_size=5,fractions=(.25,.6),dotxy=None):
 xy=np.asarray(xy);line,=ax.plot(xy[:,0],xy[:,1],color=color,lw=width,solid_capstyle='round',solid_joinstyle='round',zorder=4)
 line.set_path_effects([pe.Stroke(linewidth=width+.65,foreground=INK if dark else 'white',alpha=.85),pe.Normal()])
 if dotxy is not None:
  ax.scatter(dotxy[:,0],dotxy[:,1],s=3.8,color=color,ec=INK if dark else 'white',lw=.22,zorder=6)
 steps=np.linalg.norm(np.diff(xy,axis=0),axis=1);dist=np.r_[0,np.cumsum(steps)]
 if not dist[-1]:return
 # Arrow endpoints stay on the same integration-step segment.
 for q in fractions:
  i=min(int(np.searchsorted(dist,q*dist[-1])),len(xy)-2)
  if steps[i]<1e-10:continue
  ar=FancyArrowPatch(xy[i],xy[i+1],arrowstyle='-|>',mutation_scale=arrow_size,lw=.6,color=color,shrinkA=0,shrinkB=0,zorder=8)
  ar.set_path_effects([pe.Stroke(linewidth=1.2,foreground=INK if dark else 'white'),pe.Normal()]);ax.add_patch(ar)

def scene(ax,c,mode,small=False,roi=None):
 ax.imshow(np.array(Image.open(O/c['frames'][mode]['path'])))
 x0,y0,x1,y1=c['crop'];ax.set(xlim=(x0,x1),ylim=(y1,y0));ax.set_xticks([]);ax.set_yticks([]);ax.set_aspect('equal')
 for sp in ax.spines.values():sp.set_visible(False)
 dark=c['task']=='Maze';color=('#F1AC8D' if mode=='base' else '#81DBCE') if dark else (FAIL if mode=='base' else OURS)
 if mode in ['base','ours']:
  xy=project(c,np.array(c['paths'][mode])[:,:2]);dots=project(c,np.array(c['control_states'][mode])[5::5,:2])
  stroke(ax,xy,color,.70 if small else .85,dark=dark,arrow_size=4.7 if small else 5.3,dotxy=dots)
  ax.scatter(*xy[0],s=11 if small else 16,marker='o',fc='white',ec=INK,lw=.65,zorder=9)
  ax.scatter(*xy[-1],s=18 if small else 24,marker='X' if mode=='base' else 'D',fc=color,ec='white',lw=.55,zorder=10)
 ax.scatter(*project(c,np.array(c['goal'])[:2]),s=26 if small else 34,marker='*',fc=GOLD,ec=INK,lw=.5,zorder=11)
 if roi is not None:
  points=project(c,roi);lo=points.min(0);hi=points.max(0);margin=(x1-x0)*.015
  ax.add_patch(Rectangle(lo-margin,*(hi-lo+2*margin),fill=False,ec=color,lw=.8,ls=(0,(2,1)),zorder=12))

def inset(ax,c,first,last):
 ends=c['control_end_indices']['base'];path=np.array(c['paths']['base'])[ends[first*5]:ends[last*5]+1,:2]
 q=path-path[0];ctl=np.array(c['control_states']['base'])[first*5:last*5+1:5,:2]-path[0]
 stroke(ax,q,FAIL,.75,arrow_size=5.2,fractions=(.12,.4,.72),dotxy=ctl)
 lo=q.min(0);hi=q.max(0);center=(lo+hi)/2;span=max(hi-lo)*1.28
 ax.set_xlim(center[0]-span/2,center[0]+span/2);ax.set_ylim(center[1]+span/2,center[1]-span/2);ax.set_aspect('equal')
 ax.xaxis.set_major_locator(MaxNLocator(nbins=2));ax.yaxis.set_major_locator(MaxNLocator(nbins=2))
 ax.tick_params(axis='both',labelsize=8,length=2,pad=1)
 for side in ['top','right']:ax.spines[side].set_visible(False)
 for side in ['left','bottom']:ax.spines[side].set_color(FAIL);ax.spines[side].set_linewidth(.55)
 unit='px' if c['task']=='PushObj' else 'u'
 ax.set_xlabel(r'$\Delta x$ ('+unit+')',fontsize=8,labelpad=.3)
 ax.set_ylabel(r'$\Delta y$',fontsize=8,labelpad=.3)
 manifest['inset_statistics'][c['key']]={'first_replan':first,'last_replan':last,'unit':'pixels' if c['task']=='PushObj' else 'world units','dense_samples':len(path),'extent_xy':np.ptp(path,0).tolist(),'path_length':float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum()),'net_displacement':float(np.linalg.norm(q[-1])),'reference_position':path[0].tolist(),'scale':'equal x/y, linear','uses_dense_physics_states':True}
 return path

def axes_inches(fig,x,y,w,h):return fig.add_axes([x/fig.get_figwidth(),y/fig.get_figheight(),w/fig.get_figwidth(),h/fig.get_figheight()])
def text_inches(fig,x,y,s,**kw):return fig.text(x/fig.get_figwidth(),y/fig.get_figheight(),s,**kw)

def save(fig,name,rows,panels):
 fig.canvas.draw();renderer=fig.canvas.get_renderer();ts=[x for x in fig.texts if x.get_text()];bs=[x.get_window_extent(renderer) for x in ts]
 collisions=[(ts[i].get_text(),ts[j].get_text()) for i in range(len(bs)) for j in range(i+1,len(bs)) if bs[i].overlaps(bs[j])]
 assert not collisions,(name,collisions)
 for ext,folder in [('pdf',''),('svg','svg'),('png','previews')]:
  target=F/folder/f'{name}.{ext}';target.parent.mkdir(exist_ok=True);fig.savefig(target,dpi=350)
  manifest['outputs'][str(target.relative_to(O))]={'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'size_inches':fig.get_size_inches().tolist(),'rows':rows,'simulation_panels':panels,'text_collisions':0}
 plt.close(fig)

# Two rows of matched cases: 8 simulator views and 4 isolated native motion plots.
fig=plt.figure(figsize=(6.875,3.38));intervals={'I':(15,30),'maze_frozen':(3,13),'pushobj_ada':(20,30),'maze_ada':(7,30)}
for k,key in enumerate(main):
 row,col=divmod(k,2);c=cases[key];x=col*3.4375;top=3.34-row*1.67;bottom=top-1.39
 text_inches(fig,x+1.72,top,f'({chr(97+k)}) {c["task"]} · {c["baseline"]}',fontsize=9.4,weight='bold',color=INK,ha='center',va='top')
 for offset,label,color in [(.53,'Native: failure',FAIL),(1.74,'Local motion',FAIL),(2.92,'ECP: success',OURS)]:
  text_inches(fig,x+offset,top-.23,label,fontsize=8.3,color=color,ha='center',va='top')
 first,last=intervals[key];ia=axes_inches(fig,x+1.35,bottom+.12,.78,.78);region=inset(ia,c,first,last)
 text_inches(fig,x+1.74,top-.38,f'{first}–{last} replans',fontsize=8,color=INK,ha='center',va='top')
 for offset,mode in [(.03,'base'),(2.42,'ours')]:
  ax=axes_inches(fig,x+offset,bottom,1.0,1.0);scene(ax,c,mode,roi=region if mode=='base' else None)
  text_inches(fig,x+offset+.5,bottom-.065,f'{len(c["states"][mode])-1} replans',fontsize=8.2,color=FAIL if mode=='base' else OURS,ha='center',va='top')
save(fig,'fig1_trajectory_cases',2,8)

# Eight different paired tasks, with no repeated initial/goal pair from Figure 2.
fig=plt.figure(figsize=(6.875,2.55))
for k,key in enumerate(gallery):
 c=cases[key];row,col=divmod(k,4);x=.009+col*.25;top=.985-row*.51;bottom=.58-row*.50
 title=c['label']+' · Frozen' if c['task']=='PushObj' else 'Maze · '+c['baseline']
 fig.text(x+.118,top,f'({chr(97+k)}) {title}',fontsize=8.7,weight='bold',color=INK,ha='center',va='top')
 for j,mode in enumerate(['base','ours']):
  left=x+j*.121;ax=fig.add_axes([left,bottom,.114,.305]);scene(ax,c,mode,small=True)
  fig.text(left+.057,bottom-.023,('Fail' if mode=='base' else 'Success')+f' · {len(c["states"][mode])-1}',fontsize=8.2,color=FAIL if mode=='base' else OURS,ha='center',va='top')
save(fig,'fig5_execution_recovery',2,16)

# Supplement documents full start/goal states of all distinct Maze cases and seven shapes.
supp=['I','L','Z','T','plus','square','small_tee','maze_frozen','maze_ada','maze_frozen_alt','maze_ada_alt']
fig=plt.figure(figsize=(6.875,7.1))
for k,key in enumerate(supp):
 c=cases[key];row,col=divmod(k,2);x=.012+col*.5;top=.99-row*.164;bottom=top-.153
 name='Maze' if c['task']=='Maze' else c['label']
 fig.text(x,top,f'({chr(97+k)}) {name} · {c["baseline"]} / ECP',fontsize=8.6,color=INK,weight='bold',va='top')
 for j,mode in enumerate(['start','goal','base','ours']):
  left=x+j*.121;ax=fig.add_axes([left,bottom,.114,.1]);scene(ax,c,mode,small=True)
  label={'start':'Start','goal':'Goal','base':'Failure','ours':'ECP success'}[mode]
  fig.text(left+.057,top-.023,label,fontsize=8.0,color=INK if mode in ['start','goal'] else (FAIL if mode=='base' else OURS),ha='center',va='top')
save(fig,'figS_diverse_execution_v12',6,44)
(V/'figures_v12_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Built eight scene + four local-motion Figure 2; disjoint eight-pair Figure 4; full-state supplement.')
