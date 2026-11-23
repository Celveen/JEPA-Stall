#!/usr/bin/env python3
"""Render diverse, paired recoveries from actual MPC logs; never run dynamics."""
import argparse,hashlib,importlib.util,json,os
from pathlib import Path
os.environ.setdefault('SDL_VIDEODRIVER','dummy');os.environ.setdefault('SDL_AUDIODRIVER','dummy');os.environ.setdefault('MPLCONFIGDIR','/tmp/jepa_gallery_mpl')
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
P=argparse.ArgumentParser();P.add_argument('--repo',type=Path,required=True);P.add_argument('--paper',type=Path,required=True);a=P.parse_args();R=a.repo;O=a.paper;F=O/'figures';V=O/'provenance';D=F/'gallery_v10';D.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'serif','font.serif':['STIXGeneral'],'mathtext.fontset':'stix','font.size':8,'pdf.fonttype':42,'svg.fonttype':'none'})
INK='#243D51';FAIL='#BC613E';OURS='#117E84';GOLD='#E6B249';GRAY='#637786'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest={'revision':10,'evidence':'Native simulator renderings at states from actual evaluated executions; vector overlays connect saved MPC endpoints. No model inference or simulator stepping.','selection':'One recorded Frozen-to-ECP rescue per shape; preserve the two previously published I/small-T cases, then select the first rescued sample in sorted shard/index order. Maze uses the previously verified AdaJEPA pair and the first completed, verified frozen-memory rescue in the new evaluation.','sources':{},'cases':[],'outputs':{}}
def source(p):manifest['sources'][str(p.relative_to(R))]=sha(p)
def load(folder,i):
 out={}
 for p in folder.rglob('per_iter_logs.json'):
  for line in p.read_text().splitlines():
   x=json.loads(line)
   if x['sample_idx']==i and x['step']<=30:
    assert x['step'] not in out
    out[x['step']]=x
 return out
spec=importlib.util.spec_from_file_location('native_push_renderer',R/'env/pusht/pusht_env.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);source(R/'env/pusht/pusht_env.py')
selected=[('I','chunk_04_I',5),('L','chunk_06_L',9),('Z','chunk_15_Z',6),('T','chunk_30_T',6),('plus','chunk_27_plus',2),('square','chunk_11_square',7),('small_tee','chunk_24_small_tee',9)]
cases=[]
for shape,chunk,i in selected:
 folder=R/'eval_outputs/stag_frozen_bias16k_it40';br=folder/'ref'/chunk;orr=folder/'T0SP'/chunk
 with np.load(br/'final_eval_results.npz') as nb:sb=nb['state_0'].copy();gb=nb['state_g'].copy()
 with np.load(orr/'final_eval_results.npz') as no:np.testing.assert_array_equal(sb,no['state_0']);np.testing.assert_array_equal(gb,no['state_g'])
 b=load(br,i);o=load(orr,i);assert not b[30]['success'] and o[max(o)]['success']
 prefix=0
 for t in range(1,min(max(b),max(o))+1):
  if b[t]['state']!=o[t]['state'] or b[t]['act']!=o[t]['act']:break
  prefix=t
 c={'key':shape,'label':{'small_tee':'Small-T','plus':'Plus','square':'Square'}.get(shape,shape),'task':'PushObj','sample_idx':i,'baseline':'Frozen','intervention':'ECP (S+P)','baseline_dir':str(br.relative_to(R)),'ecp_dir':str(orr.relative_to(R)),'start':sb[i].tolist(),'goal':gb[i].tolist(),'states':{'base':[sb[i].tolist()]+[b[t]['state'] for t in sorted(b)],'ours':[sb[i].tolist()]+[o[t]['state'] for t in sorted(o)]},'baseline_success':False,'ours_success':True,'matching_prefix_replans':prefix,'frames':{}}
 for q in [br,orr]:
  for f in ['final_eval_results.npz','per_iter_logs.json','stag_episodes.jsonl']:source(q/f)
 e=mod.PushTEnv(shape={"plus":"+"}.get(shape,shape),with_target=False,with_velocity=True,render_size=512);e._setup()
 def no_step(*args,**kwargs):raise RuntimeError('Dynamics/rollouts are forbidden during figure generation')
 e.space.step=no_step;e.step=no_step
 extents=[]
 for label,s in [('start',sb[i]),('goal',gb[i]),('base',np.array(c['states']['base'][-1])),('ours',np.array(c['states']['ours'][-1]))]:
  e.block.angle=float(s[4]);e.block.position=tuple(s[2:4]);e.agent.position=tuple(s[:2]);e.agent.velocity=tuple(s[5:7]);e.space.reindex_shapes_for_body(e.block);e.space.reindex_shapes_for_body(e.agent)
  assert np.max(np.abs(np.r_[e.agent.position,e.block.position,e.block.angle]-s[:5]))<1e-8
  im=e.render('rgb_array');p=D/f'{shape}_{label}.png';Image.fromarray(im).save(p);c['frames'][label]={'path':str(p.relative_to(O)),'sha256':sha(p)}
  for sh in e.block.shapes:
   bb=sh.bb;extents += [[bb.left,bb.bottom],[bb.right,bb.top]]
 xy=np.concatenate([np.array(c['states'][m])[:,:2] for m in ['base','ours']]);allxy=np.vstack([xy,gb[i][:2],extents]);lo=allxy.min(0)-22;hi=allxy.max(0)+22;size=min(512,max(hi-lo));center=(lo+hi)/2;lo=np.clip(center-size/2,0,512-size);hi=lo+size;c['crop']=[*lo.tolist(),*hi.tolist()]
 cases.append(c)
# Native MuJoCo frames for the verified navigation pair.
v8=json.loads((V/'trajectory_v8_manifest.json').read_text());old=v8['cases']['maze'];sm=json.loads((V/'source_manifest.json').read_text())['cases']['maze']
c={'key':'maze_ada','label':'Maze','task':'Maze','sample_idx':75,'baseline':'AdaJEPA','intervention':'ECP (gated S)','baseline_dir':old['baseline_path'],'ecp_dir':old['correction_path'],'start':old['states']['base'][0],'goal':sm['states']['goal'],'states':old['states'],'baseline_success':False,'ours_success':True,'matching_prefix_replans':6,'crop':[16,64,450,498],'projection':v8['coordinate_transforms']['maze'],'frames':{}}
for mode,p in [('start',F/'frames/maze_start.png'),('goal',F/'frames/maze_goal.png'),('base',F/'trajectory_frames/maze_base_final.png'),('ours',F/'trajectory_frames/maze_ours_final.png')]:c['frames'][mode]={'path':str(p.relative_to(O)),'sha256':sha(p)}
for mode,folder in [('base',R/c['baseline_dir']),('ours',R/c['ecp_dir'])]:
 rr=load(folder,75);assert bool(rr[max(rr)]['success'])==(mode=='ours');np.testing.assert_array_equal(np.array(c['states'][mode])[1:],np.array([rr[t]['state'] for t in sorted(rr)]))
cases.append(c)
frozen_case=V/'maze_frozen_case_v10.json'
if frozen_case.exists():
 c=json.loads(frozen_case.read_text())
 for info in c['frames'].values():assert sha(O/info['path'])==info['sha256']
 cases.append(c)

def project(c,xy):
 if c['task']=='PushObj':return np.asarray(xy)
 t=c['projection'];return 256+t['scale_pixels_per_world_unit']*(np.asarray(xy)+np.array(t['body_offset_xy'])-np.array(t['camera_position'][:2]))
def panel(ax,c,mode,path=True):
 im=np.array(Image.open(O/c['frames'][mode]['path']));ax.imshow(im)
 x0,y0,x1,y1=c['crop'];ax.set(xlim=(x0,x1),ylim=(y1,y0));ax.set_xticks([]);ax.set_yticks([])
 for sp in ax.spines.values():sp.set_visible(False)
 dark=c['task']=='Maze';color=('#F4B397' if mode=='base' else '#8CE0D2') if dark else (FAIL if mode=='base' else OURS)
 if path and mode in ['base','ours']:
  xy=project(c,np.array(c['states'][mode])[:,:2]);line,=ax.plot(xy[:,0],xy[:,1],color=color,lw=1.0,ls=(0,(2.7,1.5)),dash_capstyle='round',solid_joinstyle='round',zorder=4)
  line.set_path_effects([pe.Stroke(linewidth=1.9,foreground=INK if dark else 'white',alpha=.8),pe.Normal()])
  ax.scatter(*xy[0],s=10,marker='o',fc='white',ec=INK,lw=.55,zorder=7)
  ax.scatter(*xy[-1],s=17,marker='X' if mode=='base' else 'D',fc=color,ec='white',lw=.5,zorder=8)
 if mode!='goal':ax.scatter(*project(c,np.array(c['goal'])[:2]),s=30,marker='*',fc=GOLD,ec=INK,lw=.4,zorder=9)
def save(fig,name):
 fig.canvas.draw();renderer=fig.canvas.get_renderer()
 labels=[t for t in fig.texts+sum([list(ax.texts)+[ax.title] for ax in fig.axes],[]) if t.get_visible() and t.get_text()]
 boxes=[t.get_window_extent(renderer) for t in labels]
 collisions=[(labels[i].get_text(),labels[j].get_text()) for i in range(len(boxes)) for j in range(i+1,len(boxes)) if boxes[i].overlaps(boxes[j])]
 outside=[labels[i].get_text() for i,b in enumerate(boxes) if b.x0<0 or b.y0<0 or b.x1>fig.bbox.width or b.y1>fig.bbox.height]
 assert not collisions,(name,collisions)
 assert not outside,(name,outside)
 manifest.setdefault('layout_checks',{})[name]={'text_labels':len(labels),'overlapping_labels':0,'clipped_labels':0}
 for ext,sub in [('pdf',''),('svg','svg'),('png','previews')]:
  p=F/sub/f'{name}.{ext}';fig.savefig(p,dpi=320);manifest['outputs'][str(p.relative_to(O))]=sha(p)
 plt.close(fig)
# Nine matched cases in three rows. Case headings and method labels occupy separate lines.
fig=plt.figure(figsize=(6.5,4.6))
for k,c in enumerate(cases):
 row,col=divmod(k,3);x=.012+col*(1/3);y=.680-row*.320
 fig.text(x+.155,y+.285,f'({chr(97+k)}) {c["label"]}',fontsize=8.5,color=INK,weight='bold',ha='center')
 for j,mode in enumerate(['base','ours']):
  ax=fig.add_axes([x+j*.158,y+.030,.152,.215]);panel(ax,c,mode)
  title=c['baseline'] if mode=='base' else 'Ours'
  fig.text(x+j*.158+.076,y+.251,title,fontsize=8,color=INK,ha='center')
  n=len(c['states'][mode])-1
  fig.text(x+j*.158+.076,y+.005,('Fail' if mode=='base' else 'Success')+f' · {n}',fontsize=8,color=FAIL if mode=='base' else OURS,ha='center',weight='bold')
fig.text(.012,.015,'Same model + task per pair. Dashed lines: logged MPC endpoints.',fontsize=8,color=GRAY)
fig.text(.99,.015,'Star: agent goal',fontsize=8,color=GRAY,ha='right')
save(fig,'fig5_execution_recovery')
# Larger DINO-WM-style Start / Goal / Native / Ours panels for every shape.
fig=plt.figure(figsize=(6.5,6.95))
for k,c in enumerate(cases):
 row,col=divmod(k,2);x=.015+col*.5;y=.799-row*.194
 fig.text(x,y+.184,f'({chr(97+k)}) {c["label"]} · {c["baseline"]} / ECP',fontsize=8.5,color=INK,weight='bold')
 for j,mode in enumerate(['start','goal','base','ours']):
  ax=fig.add_axes([x+j*.119,y+.023,.113,.130]);panel(ax,c,mode,path=mode in ['base','ours'])
  ax.set_title({'start':'Start','goal':'Goal','base':c['baseline'],'ours':'Ours'}[mode],fontsize=8,color=INK,pad=2)
  if mode in ['base','ours']:
   fig.text(x+j*.119+.0565,y+.010,('Failure' if mode=='base' else 'Success'),fontsize=8,color=FAIL if mode=='base' else OURS,ha='center')
save(fig,'figS_diverse_execution_v10')
manifest['cases']=cases;manifest['generator_sha256']=sha(__file__)
(V/'gallery_v10_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'cases':[(c['key'],len(c['states']['ours'])-1) for c in cases],'paired_native_failure_ecp_success':len(cases),'renders':'native simulators, no dynamics executed'},indent=2))
