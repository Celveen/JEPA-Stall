#!/usr/bin/env python3
"""Replay saved controls, verify logged endpoints, and export dense simulator paths.
No model loading, action optimization, training, or new benchmark measurement.
"""
import argparse,hashlib,json,os,sys
from pathlib import Path
os.environ.setdefault('SDL_VIDEODRIVER','dummy');os.environ.setdefault('SDL_AUDIODRIVER','dummy')
os.environ.setdefault('MUJOCO_PY_FORCE_CPU','1');os.environ.setdefault('MESA_SHADER_CACHE_DIR','/tmp/jepa_v11_mesa')
p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--paper',type=Path,required=True);a=p.parse_args();R=a.repo;O=a.paper
sys.path.insert(0,str(R))
import numpy as np
import torch
from PIL import Image
import gym,env
from datasets.pusht_dset import ACTION_MEAN,ACTION_STD
from datasets.point_maze_dset import PointMazeDataset
from datasets.diverse_maze_goals import generate_diverse_maze_goals
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
V=O/'provenance';D=O/'figures/replay_v11';D.mkdir(exist_ok=True)
old=json.loads((V/'gallery_v10_manifest.json').read_text());cases=old['cases']

def records(root,i):
 rows={};paths=[]
 for path in sorted(root.rglob('per_iter_logs.json')):
  chosen=[json.loads(l) for l in path.read_text().splitlines() if l.strip()]
  chosen=[x for x in chosen if x['sample_idx']==i and x['step']<=30]
  if chosen:
   paths.append(path)
   for x in chosen:assert x['step'] not in rows;rows[x['step']]=x
 assert len(paths)==1 and sorted(rows)==list(range(1,max(rows)+1)),(root,i,paths)
 return rows,paths[0]

# Add an adaptive manipulation recovery, completing the 2 domains x 2 update rules.
chunk='chunk_32_T';i=6;br=R/'eval_outputs/stag_adajepa_bias16k_it40/ref'/chunk;orr=R/'eval_outputs/stag_adajepa_bias16k_it40/T0SP'/chunk
b,_=records(br,i);o,_=records(orr,i);nb=np.load(br/'final_eval_results.npz');no=np.load(orr/'final_eval_results.npz')
for key in ['state_0','state_g']:np.testing.assert_array_equal(nb[key],no[key])
assert not b[30]['success'] and o[max(o)]['success']
prefix=0
for t in range(1,min(max(b),max(o))+1):
 if b[t]['act']!=o[t]['act'] or b[t]['state']!=o[t]['state']:break
 prefix=t
cases.append({'key':'pushobj_ada','label':'T','task':'PushObj','sample_idx':i,'baseline':'AdaJEPA','intervention':'ECP (S+P)','baseline_dir':str(br.relative_to(R)),'ecp_dir':str(orr.relative_to(R)),'start':nb['state_0'][i].tolist(),'goal':nb['state_g'][i].tolist(),'baseline_success':False,'ours_success':True,'matching_prefix_replans':prefix,'states':{'base':[nb['state_0'][i].tolist()]+[b[t]['state'] for t in sorted(b)],'ours':[nb['state_0'][i].tolist()]+[o[t]['state'] for t in sorted(o)]}})
g=generate_diverse_maze_goals(R/'data/diverse_maze/val',100,100,min_block_radius=6,max_block_radius=9)
train=R/'data/diverse_maze/train';statsroot=train if (train/'states.pth').exists() else R/'data/diverse_maze/val'
ds=PointMazeDataset(data_path=statsroot,normalize_action=True)
manifest={'revision':11,'operation':'Deterministic replay of saved full-precision executed controls; no model inference or new planning. All plotted substeps come from the original simulator.','sources':{},'cases':[]}
for c in cases:
 i=c['sample_idx'];s0=np.array(c['start']);goal=np.array(c['goal']);task=c['task'];data={'frames':{},'paths':{},'control_states':{},'control_end_indices':{},'endpoint_errors':{},'wall_core_crossings':{}}
 for mode,key in [('base','baseline_dir'),('ours','ecp_dir')]:
  rows,path=records(R/c[key],i);npz=path.parent/'final_eval_results.npz';raw=np.load(npz);acts=raw['actions'][i,:len(rows)]
  np.testing.assert_array_equal(raw['state_0'][i],s0);np.testing.assert_array_equal(raw['state_g'][i],goal)
  for j,row in rows.items():np.testing.assert_allclose(acts[j-1].ravel(),row['act'],atol=5.01e-5,rtol=0)
  for q in [path,npz]:manifest['sources'][str(q.relative_to(R))]=sha(q)
  act=torch.tensor(acts.reshape(-1,2),dtype=torch.float32)
  if task=='PushObj':
   e=gym.make('pushobj',with_velocity=True,with_target=True).unwrapped;e.update_env({'shape':'T' if c['key']=='pushobj_ada' else {'plus':'+'}.get(c['key'],c['key'])})
   controls=(act*ACTION_STD+ACTION_MEAN).numpy()
  else:
   np.testing.assert_array_equal(g['start_states'][i],s0)
   e=gym.make('diverse_maze',maze_spec=g['map_specs'][i],action_repeat=4).unwrapped
   controls=(act*ds.action_std+ds.action_mean).numpy()
  _,state=e.prepare(100*i+1,s0)
  dense=[np.asarray(state).tolist()];boundaries=[0];control_states=[np.asarray(state).tolist()]
  if task=='PushObj':
   original=e.space.step
   def traced(dt):
    original(dt);dense.append([*e.agent.position,*e.block.position,e.block.angle,*e.agent.velocity])
   e.space.step=traced
  else:
   original=e.do_simulation
   def traced(ctrl,n_frames):
    for _ in range(n_frames):
     original(ctrl,1);dense.append(np.r_[e.sim.data.qpos[:2],e.sim.data.qvel[:2]].tolist())
   e.do_simulation=traced
  for control in controls:
   _,_,_,info=e.step(control);boundaries.append(len(dense)-1);control_states.append(np.asarray(info['state']).tolist())
  endpoints=np.asarray(control_states)[5::5];reference=np.array([rows[t]['state'] for t in sorted(rows)])
  err=float(np.max(np.abs(endpoints-reference)));tol=.00051 if task=='PushObj' else .000051
  assert err<=tol,(c['key'],mode,err,tol)
  assert bool(e.eval_state(goal,np.asarray(control_states)[-1])['success'])==(mode=='ours')
  if task=='PushObj':
   import pygame
   e.render_size=512;e.goal_color=pygame.Color('White');im=e.render('rgb_array');c['projection']={'kind':'pushobj_identity'}
  else:
   e.set_marker();im=e.sim.render(512,512)
   ctx=e.sim.render_contexts[0];cam0,cam1=ctx.scn.camera[:2];pos=.5*(np.asarray(cam0.pos)+np.asarray(cam1.pos));scale=256*float(cam0.frustum_near)/(float(cam0.frustum_top)*float(pos[2]));offset=e.sim.data.get_geom_xpos('particle_geom')[:2]-e.sim.data.qpos[:2]
   c['projection']={'kind':'maze_camera','scale_pixels_per_world_unit':scale,'body_offset_xy':offset.tolist(),'camera_position':pos.tolist()}
   xy=np.array(dense)[:,:2]+offset;cross=0;wall_names=[]
   for gi,name in enumerate(e.model.geom_names):
    if name and 'wall' in name:
     center=e.model.geom_pos[gi,:2];half=e.model.geom_size[gi,:2];cross+=int(np.any(np.all(np.abs(xy-center)<half-1e-5,axis=1)));wall_names.append(name)
   assert wall_names and cross==0,(c['key'],mode,cross,wall_names)
   data['wall_core_crossings'][mode]=cross
  img=D/f'{c["key"]}_{mode}.png';Image.fromarray(im).save(img)
  data['frames'][mode]={'path':str(img.relative_to(O)),'sha256':sha(img)};data['paths'][mode]=dense;data['control_states'][mode]=control_states;data['control_end_indices'][mode]=boundaries;data['endpoint_errors'][mode]=err
  print(json.dumps({'case':c['key'],'arm':mode,'replans':len(rows),'physics_samples':len(dense),'max_logged_endpoint_error':err}),flush=True)
  e.close()
 c.update(data)
 # Use a common square crop for each matched pair, including complete trajectories and goal.
 if task=='PushObj':
  xy=np.vstack([np.array(data['paths'][mode])[:,:2] for mode in ['base','ours']]+[goal[:2][None,:]])
  extents=np.vstack([np.array(data['paths'][mode])[:,2:4] for mode in ['base','ours']]+[goal[2:4][None,:]])
  lo=np.minimum(xy.min(0)-22,extents.min(0)-95);hi=np.maximum(xy.max(0)+22,extents.max(0)+95);side=min(512,float(max(hi-lo)));center=(lo+hi)/2;lo=np.clip(center-side/2,0,512-side);c['crop']=[*lo.tolist(),*(lo+side).tolist()]
 else:c['crop']=[16,64,450,498]
 manifest['cases'].append(c)
(V/'replayed_paths_v11.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Saved dense simulation paths for',len(cases),'matched cases',flush=True)
