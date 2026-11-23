#!/usr/bin/env python3
"""Publish logged MPC endpoint trajectories over native simulator state renders.
No inference, planning, training, stepping, or synthetic trajectories are used.
Camera projection is recovered from the exact MuJoCo rendering context.
"""
import argparse,hashlib,json,os,sys,pickle
from pathlib import Path
os.environ.setdefault('SDL_VIDEODRIVER','dummy')
os.environ.setdefault('SDL_AUDIODRIVER','dummy')
os.environ.setdefault('MPLCONFIGDIR','/tmp/jepa_trajectory_mpl')
os.environ.setdefault('MESA_SHADER_CACHE_DIR','/tmp/jepa_trajectory_mesa')
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle,ConnectionPatch
import matplotlib.patheffects as pe

ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();R=args.repo;O=args.out;O.mkdir(parents=True,exist_ok=True);(O/'frames').mkdir(exist_ok=True);sys.path.insert(0,str(R))
plt.rcParams.update({'font.family':'serif','font.serif':['STIXGeneral'],'mathtext.fontset':'stix','font.size':8,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.linewidth':.5,'savefig.facecolor':'white'})
INK='#2C4C63';BASE='#B56E4F';OURS='#477C82';HISTORY='#879CAB';CREAM='#F8E9D8'
manifest={'format_version':3,'rendering':'Original PushT and MuJoCo renderers at recorded states; no simulator step or new model/planning run. Vector annotations over native raster images.','sampling':'One logged physical state at the end of each MPC replan; straight dashed connectors are not substep physical trajectories.','budget':30,'state_precision':{'pushobj':3,'maze':4},'sources':{},'cases':{},'coordinate_transforms':{},'outputs':{}}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def source(p):manifest['sources'][str(p.relative_to(R))]=sha(p)
def rows(folder,i):
 d={}
 for p in sorted(folder.rglob('per_iter_logs.json')):
  rr=[json.loads(l) for l in p.read_text().splitlines() if l.strip()]
  rr=[r for r in rr if r['sample_idx']==i and r['step']<=30]
  if rr:source(p);d.update({r['step']:r for r in rr})
 return d
old=R/'paper/cvpr2027/provenance/source_manifest.json';source(old);olddata=json.loads(old.read_text())
for rel in ['env/pusht/pusht_env.py','env/pointmaze/maze_model.py','env/pointmaze/point_maze_wrapper.py','scripts/stag_base_run.py','scripts/e3_run.py']:source(R/rel)
cases={}
for task,key,base,correct,i,match,late in [
 ('PushObj','pushobj','eval_outputs/stag_frozen_bias16k_it40/ref/chunk_04_I','eval_outputs/stag_frozen_bias16k_it40/T0SP/chunk_04_I',5,15,15),
 ('Maze','maze','eval_outputs/stag_maze100_ar4/medium/adajepa/ref','eval_outputs/e3/medium_s100/SZ-switch',75,6,7)]:
 rr=rows(R/base,i);aa=rows(R/correct,i);s0=np.array(olddata['cases'][key]['states']['initial']);goal=np.array(olddata['cases'][key]['states']['goal'])
 for t in range(1,match+1):assert rr[t]['state']==aa[t]['state'] and rr[t]['act']==aa[t]['act']
 assert not rr[30]['success'] and aa[max(aa)]['success']
 d={'task':task,'key':key,'base':rr,'ours':aa,'s0':s0,'goal':goal,'late':late,'match':match,'spec':olddata['cases'][key].get('maze_spec')}
 cases[key]=d;mc={'sample_idx':i,'baseline_path':base,'correction_path':correct,'baseline_condition':'frozen MPC' if key=='pushobj' else 'online-adapted MPC','correction':'S+P' if key=='pushobj' else 'gated S only','matching_action_and_state_prefix_replans':match,'states':{},'late_window_statistics':{}}
 for mode,rs in [('base',rr),('ours',aa)]:
  p=np.array([s0]+[rs[t]['state'] for t in sorted(rs)]);d[mode+'_states']=p;d[mode+'_xy']=p[:,:2]
  mc['states'][mode]=p.tolist();pts=p[late:,:2];diff=np.diff(pts,axis=0)
  mc['late_window_statistics'][mode]={'first_replan':late,'last_replan':len(p)-1,'endpoint_polyline_length':float(np.linalg.norm(diff,axis=1).sum()),'net_displacement':float(np.linalg.norm(pts[-1]-pts[0])),'x_span':float(np.ptp(pts[:,0])),'y_span':float(np.ptp(pts[:,1])),'diameter':float(np.max(np.linalg.norm(pts[:,None]-pts[None,:],axis=-1))),'unit':'pixels' if key=='pushobj' else 'simulator units'}
 manifest['cases'][key]=mc
# Fresh pairing checks against original target arrays, not only an older manifest.
pr=R/'eval_outputs/stag_frozen_bias16k_it40/ref/chunk_04_I/final_eval_results.npz'
pa=R/'eval_outputs/stag_frozen_bias16k_it40/T0SP/chunk_04_I/final_eval_results.npz'
source(pr);source(pa);rz=np.load(pr);az=np.load(pa)
for key in ['state_0','state_g']:np.testing.assert_array_equal(rz[key],az[key])
np.testing.assert_array_equal(cases['pushobj']['s0'],rz['state_0'][5]);np.testing.assert_array_equal(cases['pushobj']['goal'],rz['state_g'][5])
mr=R/'eval_outputs/stag_maze100_ar4/medium/adajepa/ref/shard01/plan_targets.pkl'
ma=R/'eval_outputs/e3/medium_s100/SZ-switch/shard04/plan_targets.pkl'
source(mr);source(ma);rt=pickle.load(open(mr,'rb'));at=pickle.load(open(ma,'rb'))
for key in ['state_0','state_g']:np.testing.assert_array_equal(rt[key],at[key])
np.testing.assert_array_equal(cases['maze']['s0'],rt['state_0'][75]);np.testing.assert_array_equal(cases['maze']['goal'],rt['state_g'][75])
for key,rel in [('pushobj','eval_outputs/stag_frozen_bias16k_it40/T0SP/chunk_04_I/stag_meta.json'),('maze','eval_outputs/e3/medium_s100/SZ-switch/shard04/stag_meta.json')]:
 source(R/rel);meta=json.loads((R/rel).read_text());manifest['cases'][key]['intervention_metadata']=meta
 assert meta['kappa']==2 and meta['prop_boost']==(10 if key=='pushobj' else 1)
 manifest['cases'][key]['original_target_array_pairing_verified']=True

# Render PushObj by direct pose assignment, without stepping physics.
from env.pusht.pusht_env import PushTEnv
p=cases['pushobj'];e=PushTEnv(shape='I',with_target=False,with_velocity=True,render_size=512);e.reset()
for mode in ['base','ours']:
 s=p[mode+'_states'][-1];e.block.angle=float(s[4]);e.block.position=tuple(s[2:4]);e.agent.position=tuple(s[:2]);e.agent.velocity=tuple(s[5:7]);e.space.reindex_shapes_for_body(e.block);e.space.reindex_shapes_for_body(e.agent)
 assert np.max(np.abs(np.r_[e.agent.position,e.block.position,e.block.angle]-s[:5]))<1e-8
 im=e.render('rgb_array');p[mode+'_image']=im;Image.fromarray(im).save(O/'frames'/f'pushobj_{mode}_final.png')
p['project']=lambda xy:np.asarray(xy)
manifest['coordinate_transforms']['pushobj']={'formula':'image_x=x, image_y=y (Pygame native 512x512 coordinates)','crop_xyxy':[10,20,300,390],'equal_axis_scale':True}
# Recover the native free camera rather than estimating registration from pixels.
import gym,env
m=cases['maze'];e=gym.make('diverse_maze',maze_spec=m['spec'],action_repeat=4).unwrapped;e.prepare(0,m['s0'])
for mode in ['base','ours']:
 s=m[mode+'_states'][-1];e.set_state(s[:2],s[2:4]);e.set_marker();im=e.sim.render(512,512);m[mode+'_image']=im;Image.fromarray(im).save(O/'frames'/f'maze_{mode}_final.png')
c=e.sim.render_contexts[0];cam0=c.scn.camera[0];cam1=c.scn.camera[1];pos=.5*(np.asarray(cam0.pos)+np.asarray(cam1.pos));near=float(cam0.frustum_near);top=float(cam0.frustum_top);scale=256*near/(top*float(pos[2]));offset=e.sim.data.get_geom_xpos('particle_geom')[:2]-e.sim.data.qpos[:2]
assert np.allclose(c.cam.azimuth,90) and np.allclose(c.cam.elevation,-90)
def maze_project(xy):return 256+scale*(np.asarray(xy)+offset-pos[:2])
m['project']=maze_project
manifest['coordinate_transforms']['maze']={'formula':'pixel_xy=256 + s*(qpos_xy + body_offset_xy - camera_xy). Native MuJoCo array is bottom-up; displayed without flip, preserving original observations.','camera_lookat':c.cam.lookat.tolist(),'camera_position':pos.tolist(),'azimuth':float(c.cam.azimuth),'elevation':float(c.cam.elevation),'fovy':float(e.model.vis.global_.fovy),'scale_pixels_per_world_unit':scale,'body_offset_xy':offset.tolist(),'crop_xyxy':[16,64,450,494],'equal_axis_scale':True}
# Registration check against visible sphere centers in original non-occluded frames.
checks=[]
for name,s in [('maze_start',m['s0']),('maze_goal',m['goal'])]:
 src=R/f'paper/cvpr2027/figures/frames/{name}.png';source(src);im=np.array(Image.open(src));mask=(im[:,:,1]>1.3*im[:,:,0])&(im[:,:,1]>1.3*im[:,:,2])&(im[:,:,1]>65);ys,xs=np.where(mask);observed=np.array([xs.mean(),ys.mean()]);pred=maze_project(s[:2]);err=float(np.linalg.norm(observed-pred));assert err<1.5
 checks.append({'frame':name,'observed_green_pixel_centroid':observed.tolist(),'projected_center':pred.tolist(),'error_pixels':err})
manifest['coordinate_transforms']['maze']['registration_checks']=checks

# Four panels: native scene on left, exact late-window position zoom on right.
f=plt.figure(figsize=(6.875,3.8));axes=[]
for row,key in enumerate(['pushobj','maze']):
 d=cases[key];top_y=.915-row*.46;bot=top_y-.354
 for col,mode in enumerate(['base','ours']):
  x=.035+col*.50;w=.223;h=.354;clr=BASE if mode=='base' else OURS;n=len(d[mode+'_xy'])-1
  label=('Frozen MPC' if key=='pushobj' else 'Online-adapted MPC') if mode=='base' else ('ECP: regularization + weighting (S+P)' if key=='pushobj' else 'ECP: gated regularization (S)')
  f.text(x,top_y+.042,f"({chr(97+2*row+col)}) {d['task']}",fontsize=9,color=INK,weight='bold')
  f.text(x,top_y+.008,label,fontsize=8.1,color=INK)
  ax=f.add_axes([x,bot,w,h]);axes.append(ax);ax.imshow(d[mode+'_image']);xy=d['project'](d[mode+'_xy']);goal=d['project'](d['goal'][:2]);late=d['late']
  # Visible line outlines on both bright PushObj and dark Maze native backgrounds.
  histclr=HISTORY if key=='pushobj' else CREAM
  line,=ax.plot(xy[:,0],xy[:,1],color=histclr,lw=1.05,ls=(0,(2,1.4)),marker='.',ms=2.6,zorder=3)
  line.set_path_effects([pe.Stroke(linewidth=1.9,foreground='white' if key=='pushobj' else INK,alpha=.7),pe.Normal()])
  line,=ax.plot(xy[late:,0],xy[late:,1],color=clr if key=='pushobj' else ('#F4A688' if mode=='base' else '#83D3CA'),lw=1.25,ls=(0,(2,1.3)),marker='.',ms=3,zorder=4)
  line.set_path_effects([pe.Stroke(linewidth=2.0,foreground='white' if key=='pushobj' else INK,alpha=.8),pe.Normal()])
  ax.plot(*xy[0],marker='o',ms=5,mfc='white',mec=INK,mew=1,zorder=7);ax.plot(*xy[-1],marker='X',ms=5,mfc=clr,mec='white',mew=.7,zorder=8);ax.plot(*goal,marker='*',ms=8,mfc='#E5B442',mec=INK,mew=.6,zorder=8)
  if key=='pushobj':ax.set_xlim(10,300);ax.set_ylim(390,20)
  else:ax.set_xlim(16,450);ax.set_ylim(494,64)
  ax.set_xticks([]);ax.set_yticks([])
  for sp in ax.spines.values():sp.set_color('#CED7DD');sp.set_linewidth(.6)
  # Side inset intentionally uses data coordinates with explicit units and scale.
  ix=x+.255;iy=bot+.135;iw=.181;ih=.175;ia=f.add_axes([ix,iy,iw,ih]);q=d[mode+'_xy'][late:];origin=q[0];qq=q-origin
  ia.plot(qq[:,0],qq[:,1],color=clr,lw=1,ls=(0,(2,1.3)),marker='o',ms=2.3,mec='white',mew=.25)
  ia.plot(0,0,'o',ms=4,mfc='white',mec=INK,mew=.8);ia.plot(*qq[-1],'X',ms=5,mfc=clr,mec='white',mew=.5)
  ia.set_aspect('equal',adjustable='box');ia.invert_yaxis();ia.tick_params(labelsize=8,length=2,pad=1);ia.spines[['top','right']].set_visible(False);ia.spines[['bottom','left']].set_color('#B2BDC5')
  unit='px' if key=='pushobj' else 'units';ia.set_xlabel(r'$\Delta x$ ('+unit+')',fontsize=8,labelpad=1);ia.set_ylabel(r'$\Delta y$',fontsize=8,labelpad=0)
  f.text(ix+iw/2,bot+.326,f'Replans {late}–{n}',ha='center',fontsize=8,color=INK)
  stats=manifest['cases'][key]['late_window_statistics'][mode]
  if key=='pushobj' and mode=='base':ia.set_xlim(-2.1,2.8);ia.set_ylim(2.2,-2.2);ia.set_xticks([-2,0,2]);ia.set_yticks([-2,0,2])
  elif key=='pushobj':ia.set_xlim(-3,21);ia.set_ylim(3,-38);ia.set_xticks([0,20]);ia.set_yticks([-30,0])
  elif mode=='base':ia.set_xlim(-1.3,1.3);ia.set_ylim(.35,-3.2);ia.set_xticks([-1,1]);ia.set_yticks([-3,0])
  else:ia.set_xlim(qq[:,0].min()-.4,qq[:,0].max()+.4);ia.set_ylim(qq[:,1].max()+.4,qq[:,1].min()-.4);ia.set_xticks([-4,0]);ia.set_yticks([0,2])
  # Each inset reports the logged-endpoint length; no claim of substep arc length.
  path=stats['endpoint_polyline_length'];net=stats['net_displacement'];nettxt=f'{net:.2f}' if key=='pushobj' else (f'{net:.4f}' if net<.001 else f'{net:.2f}')
  f.text(ix+iw/2,bot+.005,f'Path {path:.1f}; net {nettxt} {unit}',ha='center',fontsize=8,color=INK)
  f.text(x+w/2,bot-.027,f"{'Failure' if mode=='base' else 'Success'} at replan {n}",ha='center',fontsize=8.5,color=clr,weight='bold')
# Thin separator and compact shared legend, with shape encoding independent of color.
f.lines.append(Line2D([.025,.975],[.523,.523],transform=f.transFigure,color='#DCE3E7',lw=.6))
handles=[Line2D([0],[0],color=HISTORY,lw=1,ls='--',marker='.',label='MPC endpoints'),Line2D([0],[0],color=BASE,lw=1.3,ls='--',marker='.',label='Late window'),Line2D([0],[0],marker='o',color='none',mfc='white',mec=INK,ms=5,label='Start'),Line2D([0],[0],marker='X',color='none',mfc=OURS,mec='white',ms=6,label='End'),Line2D([0],[0],marker='*',color='none',mfc='#E5B442',mec=INK,ms=8,label='Agent goal')]
f.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,-.005),ncol=5,frameon=False,fontsize=8,handlelength=1.6,columnspacing=1.25,handletextpad=.45)
for ext in ['pdf','svg','png']:
 p=O/f'fig1_trajectory_cases.{ext}';f.savefig(p,dpi=250 if ext=='png' else None);manifest['outputs'][p.name]={'sha256':sha(p),'width_inches':6.875,'height_inches':3.8,'minimum_font_pt':8}
plt.close(f);manifest['generator_sha256']=sha(Path(__file__));(O/'trajectory_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest['cases'],indent=2))
