#!/usr/bin/env python3
"""Render logged simulator states and build original paper figures.

No policy, world-model inference, planning, training, or action rollout is run.
PushObj poses are assigned directly without a physics step. Maze uses set_state.
Run in the existing ts environment. For CPU-only MuJoCo, use an isolated copy
of mujoco_py on PYTHONPATH and MUJOCO_PY_FORCE_CPU=1 with OSMesa available.
"""
import os
os.environ.setdefault('SDL_VIDEODRIVER','dummy')
os.environ.setdefault('SDL_AUDIODRIVER','dummy')
os.environ.setdefault('MPLCONFIGDIR','/tmp/jepa_paper_mpl')
os.environ.setdefault('MESA_SHADER_CACHE_DIR','/tmp/jepa_paper_mesa')
import argparse, hashlib, json, pickle, sys
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

AP=argparse.ArgumentParser()
AP.add_argument('--repo',type=Path,default=Path('/home/appuser/workspace/project/multimodal-test/JEPA-Stall'))
AP.add_argument('--out',type=Path,default=Path(__file__).resolve().parent)
AP.add_argument('--skip-maze',action='store_true')
AP.add_argument('--only-method',action='store_true')
args=AP.parse_args();ROOT=args.repo;OUT=args.out;OUT.mkdir(exist_ok=True,parents=True)
(OUT/'frames').mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT))
plt.rcParams.update({'font.family':'serif','font.serif':['STIXGeneral','DejaVu Serif'],'mathtext.fontset':'stix','font.size':9,'pdf.fonttype':42,'ps.fonttype':42,'axes.linewidth':.6,'savefig.facecolor':'white'})
BLUE='#406782';ORANGE='#B55C27';GRAY='#59636B';LIGHT='#EDF2F5';WARM='#F7EEE6'
manifest={'format_version':1,'rendering':'original simulator renderers at logged physical states; not newly executed rollouts','state_precision':'PushObj logs rounded to 3 decimals; Maze logs rounded to 4 decimals','sources':{},'cases':{},'figures':{}}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def source(p):
 p=Path(p);manifest['sources'][str(p.relative_to(ROOT))]=sha(p)
def load_rows(folder,i):
 rows=[]
 for p in sorted(folder.rglob('per_iter_logs.json')):
  aa=[json.loads(l) for l in p.read_text().splitlines() if l.strip()]
  bb=[r for r in aa if r['sample_idx']==i]
  if bb:source(p);rows+=bb
 return {r['step']:r for r in rows}
def frame_save(im,name):
 p=OUT/'frames'/f'{name}.png';Image.fromarray(im).save(p);return im

def render_push():
 from env.pusht.pusht_env import PushTEnv
 root=ROOT/'eval_outputs/stag_frozen_bias16k_it40';chunk='chunk_04_I';i=5
 rp=root/'ref'/chunk;ip=root/'T0SP'/chunk
 r=load_rows(rp,i);a=load_rows(ip,i)
 for p in [rp/'final_eval_results.npz',ip/'final_eval_results.npz']:source(p)
 rz=np.load(rp/'final_eval_results.npz');az=np.load(ip/'final_eval_results.npz')
 # Rerun files retain full batch order for this historical experiment.
 np.testing.assert_array_equal(rz['state_0'],az['state_0']);np.testing.assert_array_equal(rz['state_g'],az['state_g'])
 assert not r[30]['success'] and a[17]['success']
 for t in range(1,16):
  assert r[t]['state']==a[t]['state'] and r[t]['act']==a[t]['act']
 s0=rz['state_0'][i];g=rz['state_g'][i]
 e=PushTEnv(shape='I',with_target=False,with_velocity=True,render_size=512);e.reset()
 def render(s,name):
  e.block.angle=float(s[4]);e.block.position=tuple(float(v) for v in s[2:4]);e.agent.position=tuple(float(v) for v in s[:2]);e.agent.velocity=tuple(float(v) for v in s[5:7]);e.space.reindex_shapes_for_body(e.block);e.space.reindex_shapes_for_body(e.agent)
  assert np.max(np.abs(np.r_[e.agent.position,e.block.position,e.block.angle]-np.array(s[:5])))<1e-8
  return frame_save(e.render('rgb_array'),name)
 ims=[[render(s0,'push_start'),render(r[15]['state'],'push_before'),render(r[30]['state'],'push_baseline_end'),render(g,'push_goal')],
      [render(s0,'push_start'),render(a[15]['state'],'push_before'),render(a[17]['state'],'push_corrected_end'),render(g,'push_goal')]]
 manifest['cases']['pushobj']={'task':f'{chunk}/sample_{i}','baseline':'stag_frozen_bias16k_it40/ref','intervention':'stag_frozen_bias16k_it40/T0SP','budget':30,'baseline_steps':[0,15,30],'intervention_steps':[0,15,17],'matched_initial_and_goal':True,'actions_states_equal_through_replan':15,'baseline_success_at30':False,'intervention_success_at17':True,'baseline_final_dist4_px':r[30]['dist4'],'baseline_final_object_error_px':r[30]['dist_obj'],'baseline_final_pusher_error_px':r[30]['dist_agent'],'intervention_final_dist4_px':a[17]['dist4'],'intervention_final_object_error_px':a[17]['dist_obj'],'intervention_final_pusher_error_px':a[17]['dist_agent'],'states':{'initial':s0.tolist(),'goal':g.tolist(),'pre_intervention':r[15]['state'],'baseline_final':r[30]['state'],'intervention_final':a[17]['state']}}
 draw_sequence(ims, ['Frozen MPC','Triggered correction'], ['Start','Before correction','Final state','Goal'], [['$t=0$','$t=15$','$t=30$; failure','Target observation'],['$t=0$','$t=15$','$t=17$; success','Target observation']], 'fig1_pushobj_motivation', 'Blue: pusher. Gray: object. Same start, goal, model, and first 15 executed plans.')
 # A paired distance trace from exact reported distances, not from re-rendering.
 f,ax=plt.subplots(figsize=(3.25,1.75));f.subplots_adjust(left=.18,right=.97,bottom=.26,top=.91)
 ax.plot([0]+list(r),[np.linalg.norm(s0[:4]-g[:4])]+[r[t]['dist4'] for t in r],color=GRAY,lw=1.3,ls='--',label='Frozen MPC')
 ax.plot([0]+list(a),[np.linalg.norm(s0[:4]-g[:4])]+[a[t]['dist4'] for t in a],color=BLUE,lw=1.5,label='Triggered correction')
 ax.axhline(20,color=ORANGE,lw=.8,ls=':');ax.axvline(15,color=GRAY,lw=.7,ls=':');ax.set_xlim(0,30);ax.set_ylim(0,260);ax.set_xlabel('MPC replan');ax.set_ylabel('Joint position error (px)');ax.set_xticks([0,10,20,30]);ax.spines[['top','right']].set_visible(False);ax.legend(frameon=False,fontsize=8,loc='upper right');save(f,'figS_pushobj_distance')

def render_maze():
 import torch,gym,env
 from datasets.diverse_maze_goals import generate_diverse_maze_goals
 rb=ROOT/'eval_outputs/stag_maze100_ar4/medium/adajepa/ref';ab=ROOT/'eval_outputs/e3/medium_s100/SZ-switch';i=75
 r=load_rows(rb,i);a=load_rows(ab,i)
 pp=ab/'shard04/plan_targets.pkl';source(pp);targets=pickle.load(open(pp,'rb'));s0=targets['state_0'][i];g=targets['state_g'][i]
 rpp=next(rb.glob('shard*/plan_targets.pkl'));source(rpp);rt=pickle.load(open(rpp,'rb'));np.testing.assert_array_equal(rt['state_0'],targets['state_0']);np.testing.assert_array_equal(rt['state_g'],targets['state_g'])
 gend=generate_diverse_maze_goals(ROOT/'data/diverse_maze/val',100,100,min_block_radius=6,max_block_radius=9)
 source(ROOT/'data/diverse_maze/val/maze_specs.pth')
 np.testing.assert_allclose(s0,gend['start_states'][i],atol=1e-6);np.testing.assert_allclose(g,gend['goal_states'][i],atol=1e-6)
 for t in range(1,7):assert r[t]['state']==a[t]['state'] and r[t]['act']==a[t]['act']
 assert not r[30]['success'] and a[27]['success']
 e=gym.make('diverse_maze',maze_spec=gend['map_specs'][i],action_repeat=4).unwrapped;e.prepare(0,s0)
 def render(s,name):
  e.set_state(np.array(s[:2],dtype=np.float64),np.array(s[2:4],dtype=np.float64));e.set_marker();return frame_save(e.sim.render(512,512),name)
 ims=[[render(s0,'maze_start'),render(r[6]['state'],'maze_before'),render(r[30]['state'],'maze_baseline_end'),render(g,'maze_goal')],
      [render(s0,'maze_start'),render(a[6]['state'],'maze_before'),render(a[27]['state'],'maze_corrected_end'),render(g,'maze_goal')]]
 for name, arr in [('maze_original_start',targets['obs_0']['visual'][i,0]),('maze_original_goal',targets['obs_g']['visual'][i,0])]:frame_save(arr,name)
 manifest['cases']['maze']={'sample_idx':75,'split':'medium_s100 development','map_index':int(gend['map_indices'][i]),'maze_spec':gend['map_specs'][i],'baseline':'stag_maze100_ar4/medium/adajepa/ref','intervention':'e3/medium_s100/SZ-switch','budget':30,'baseline_steps':[0,6,30],'intervention_steps':[0,6,27],'actions_states_equal_through_replan':6,'baseline_success_at30':False,'intervention_success_at27':True,'baseline_final_distance':r[30]['dist_pos'],'intervention_final_distance':a[27]['dist_pos'],'state_based_rendering':'MuJoCo 2.1.0 OSMesa CPU, original native camera and colors, higher raster resolution (512 rather than 224)','states':{'initial':s0.tolist(),'goal':g.tolist(),'pre_intervention':r[6]['state'],'baseline_final':r[30]['state'],'intervention_final':a[27]['state']}}
 draw_sequence(ims,['AdaJEPA','AdaJEPA + memory'],['Start','Switch point','Final state','Goal'],[['$t=0$','$t=6$','$t=30$; failure','Target observation'],['$t=0$','$t=6$','$t=27$; success','Target observation']],'fig3_maze_case','Green sphere: agent. Same start, goal, model, and first 6 executed plans.')

def save(f,name):
 for ext in ['pdf','svg']:f.savefig(OUT/f'{name}.{ext}',bbox_inches=None)
 f.savefig(OUT/f'{name}.png',dpi=200)
 manifest['figures'][name]={'width_inches':float(f.get_size_inches()[0]),'height_inches':float(f.get_size_inches()[1]),'minimum_font_pt':8,'pdf':name+'.pdf','preview':name+'.png'}
 plt.close(f)

def draw_sequence(ims,rowlabels,colheads,subtitles,name,note):
 f=plt.figure(figsize=(6.875,3.0))
 left=.14;w=.198;h=.385;gap=.016;ys=[.535,.09]
 for c,head in enumerate(colheads):f.text(left+c*(w+gap)+w/2,.972,head,ha='center',va='top',fontsize=10)
 for rr in range(2):
  f.text(.011,ys[rr]+h*.5,rowlabels[rr].replace(' ','\n',1),ha='left',va='center',color=GRAY if rr==0 else BLUE,fontsize=9.5)
  for cc in range(4):
   ax=f.add_axes([left+cc*(w+gap),ys[rr],w,h]);ax.imshow(ims[rr][cc]);ax.set_axis_off()
   f.text(left+cc*(w+gap)+w/2,ys[rr]-.013,subtitles[rr][cc],ha='center',va='top',fontsize=8.5,color=ORANGE if cc==2 and rr==0 else BLUE if cc==2 else '#222222')
 save(f,name)

def method():
 f=plt.figure(figsize=(6.875,3.35));ax=f.add_axes([0,0,1,1]);ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
 def box(x,y,w,h,text,fill='white',edge=GRAY,fontsize=9):
  b=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.005,rounding_size=0.008',linewidth=.85,edgecolor=edge,facecolor=fill);ax.add_patch(b);ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fontsize,color='#222222',linespacing=1.2);return(x,y,w,h)
 def arrow(p,q,color=GRAY,style='-',connection='arc3,rad=0'):
  ax.add_patch(FancyArrowPatch(p,q,arrowstyle='-|>',mutation_scale=10,lw=.9,color=color,linestyle=style,connectionstyle=connection))
 ax.text(.025,.965,'Training: interaction coverage',fontsize=10,va='top',weight='bold')
 box(.025,.745,.25,.14,'Interaction-biased\ndata collection',LIGHT,BLUE)
 box(.355,.745,.24,.14,'Unchanged\nJEPA + TS learning')
 box(.695,.745,.28,.14,'World model\nencoder + predictor',LIGHT,BLUE)
 arrow((.28,.815),(.35,.815));arrow((.60,.815),(.69,.815))
 ax.text(.315,.835,'data',ha='center',fontsize=8)
 ax.plot([.025,.975],[.675,.675],color='#D4DADF',lw=.7)
 ax.text(.025,.64,'Planning: execution-dependent correction',fontsize=10,va='top',weight='bold')
 box(.025,.415,.19,.135,'Observation\nand goal')
 box(.30,.415,.20,.135,'Latent MPC',LIGHT,BLUE)
 box(.59,.415,.18,.135,'Execute\naction prefix')
 box(.835,.415,.14,.135,'Simulator')
 arrow((.22,.482),(.295,.482));arrow((.505,.482),(.585,.482));arrow((.775,.482),(.83,.482))
 # Explicit learned-model interface; does not imply new training losses.
 arrow((.835,.74),(.835,.585));arrow((.835,.585),(.40,.585));arrow((.40,.585),(.40,.555))
 ax.text(.615,.596,'latent predictions',fontsize=8,ha='center')
 box(.685,.08,.29,.17,'Observed transitions\nand stall feedback',WARM,ORANGE)
 box(.025,.045,.50,.24,'Execution-conditioned correction\nTemporary proprioceptive reweighting\nOptional action correction and local memory',WARM,ORANGE,9)
 arrow((.905,.41),(.905,.255),ORANGE)
 arrow((.68,.165),(.53,.165),ORANGE)
 arrow((.40,.29),(.40,.41),ORANGE)
 ax.text(.44,.35,'planning objective / actions',fontsize=8,va='center')
 save(f,'fig2_method')

if args.only_method:
 manifest=json.loads((OUT/'source_manifest.json').read_text())
 method()
else:
 render_push();method()
 if not args.skip_maze:render_maze()
source(ROOT/'env/pusht/pusht_env.py');source(ROOT/'env/pointmaze/maze_model.py');source(ROOT/'env/pointmaze/point_maze_wrapper.py')
manifest['generator_sha256']=sha(Path(__file__))
(OUT/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'figures':list(manifest['figures']),'sources':len(manifest['sources'])},indent=2))
