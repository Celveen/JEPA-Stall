"""Use recorded trajectories to correct framing and add nonoverlapping Maze examples."""
import os,sys,json,hashlib,argparse
from pathlib import Path
os.environ.setdefault('SDL_VIDEODRIVER','dummy');os.environ.setdefault('SDL_AUDIODRIVER','dummy')
os.environ.setdefault('MUJOCO_PY_FORCE_CPU','1');os.environ.setdefault('MESA_SHADER_CACHE_DIR','/tmp/jepa_v12_mesa')
p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--paper',type=Path,required=True);a=p.parse_args();R=a.repo;O=a.paper
sys.path.insert(0,str(R))
import numpy as np,torch,gym,env
from PIL import Image
from datasets.diverse_maze_goals import generate_diverse_maze_goals
from datasets.point_maze_dset import PointMazeDataset
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
data=json.loads((O/'provenance/replayed_paths_v11.json').read_text());data['revision']=12
gallery=json.loads((O/'provenance/gallery_v10_manifest.json').read_text())
old={c['key']:c for c in gallery['cases']}
for c in data['cases']:
 if c['key'] in old:
  c['frames'].update({k:old[c['key']]['frames'][k] for k in ['start','goal']})
g=generate_diverse_maze_goals(R/'data/diverse_maze/val',100,100,min_block_radius=6,max_block_radius=9)
ds=PointMazeDataset(data_path=R/'data/diverse_maze/val',normalize_action=True)
D=O/'figures/frames_v12';D.mkdir(exist_ok=True)
def records(root,i):
 rows={};paths=[]
 for path in root.rglob('per_iter_logs.json'):
  rr=[json.loads(s) for s in path.read_text().splitlines() if s.strip()]
  selected=[x for x in rr if x['sample_idx']==i and x['step']<=30]
  if selected:
   paths.append(path)
   for x in selected:assert x['step'] not in rows;rows[x['step']]=x
 assert len(paths)==1 and sorted(rows)==list(range(1,max(rows)+1))
 return rows,paths[0]
for key,i,base,ours,baseline in [
 ('maze_frozen_alt',60,'eval_outputs/stag_maze100_ar4/medium/frozen/ref','eval_outputs/maze_frozen_v10/S','Frozen'),
 ('maze_ada_alt',28,'eval_outputs/stag_maze100_ar4/medium/adajepa/ref','eval_outputs/e3/medium_s100/SZ-switch','AdaJEPA')]:
 b,bp=records(R/base,i);o,op=records(R/ours,i);nb=np.load(bp.parent/'final_eval_results.npz');no=np.load(op.parent/'final_eval_results.npz')
 for k in ['state_0','state_g']:np.testing.assert_array_equal(nb[k][i],no[k][i])
 assert not b[max(b)]['success'] and o[max(o)]['success']
 s0=nb['state_0'][i].tolist();goal=nb['state_g'][i].tolist();prefix=0
 for t in range(1,min(max(b),max(o))+1):
  if b[t]['state']!=o[t]['state'] or b[t]['act']!=o[t]['act']:break
  prefix=t
 c=dict(key=key,label='Maze',task='Maze',sample_idx=i,baseline=baseline,intervention='ECP (S)',baseline_dir=base,ecp_dir=ours,start=s0,goal=goal,baseline_success=False,ours_success=True,matching_prefix_replans=prefix,frames={},states={},paths={},control_states={},control_end_indices={},endpoint_errors={},wall_core_crossings={})
 for mode,rows,path,raw in [('base',b,bp,nb),('ours',o,op,no)]:
  c['states'][mode]=[s0]+[rows[t]['state'] for t in sorted(rows)]
  acts=raw['actions'][i,:len(rows)]
  for j,row in rows.items():np.testing.assert_allclose(acts[j-1].ravel(),row['act'],atol=5.01e-5,rtol=0)
  for f in [path,path.parent/'final_eval_results.npz']:data['sources'][str(f.relative_to(R))]=sha(f)
  controls=(torch.tensor(acts.reshape(-1,2),dtype=torch.float32)*ds.action_std+ds.action_mean).numpy()
  e=gym.make('diverse_maze',maze_spec=g['map_specs'][i],action_repeat=4).unwrapped
  _,state=e.prepare(100*i+1,np.array(s0));dense=[np.asarray(state).tolist()];ends=[0];cs=[np.asarray(state).tolist()]
  original=e.do_simulation
  def traced(ctrl,n):
   for _ in range(n):original(ctrl,1);dense.append(np.r_[e.sim.data.qpos[:2],e.sim.data.qvel[:2]].tolist())
  e.do_simulation=traced
  for control in controls:
   _,_,_,info=e.step(control);ends.append(len(dense)-1);cs.append(np.asarray(info['state']).tolist())
  err=float(np.abs(np.array(cs)[5::5]-np.array([rows[t]['state'] for t in sorted(rows)])).max());assert err<=.000051
  assert bool(e.eval_state(np.array(goal),np.array(cs[-1]))['success'])==(mode=='ours')
  c['paths'][mode]=dense;c['control_states'][mode]=cs;c['control_end_indices'][mode]=ends;c['endpoint_errors'][mode]=err
  e.close()
 data['cases'].append(c)
for c in data['cases']:
 if c['task']=='Maze':
  i=c['sample_idx'];e=gym.make('diverse_maze',maze_spec=g['map_specs'][i],action_repeat=4).unwrapped;e.prepare(100*i+1,np.array(c['start']))
  wall_ids=[j for j,n in enumerate(e.model.geom_names) if n and 'wall' in n]
  wall_lo=(e.model.geom_pos[wall_ids,:2]-e.model.geom_size[wall_ids,:2]).min(0)
  wall_hi=(e.model.geom_pos[wall_ids,:2]+e.model.geom_size[wall_ids,:2]).max(0)
  ctx=e.sim.render_contexts[0]
  ctx.cam.lookat[:]=[*((wall_lo+wall_hi)/2),0]
  ctx.cam.distance=max(float(ctx.cam.distance),float(max(wall_hi-wall_lo)+1)/(2*np.tan(np.deg2rad(45)/2)))
  # The body translation is constant; dynamic geom_xpos may lag a final integration step.
  offset=e.model.body_pos[e.model.body_name2id('particle'),:2].copy()
  for mode,s in [('start',c['start']),('goal',c['goal']),('base',c['control_states']['base'][-1]),('ours',c['control_states']['ours'][-1])]:
   s=np.asarray(s);e.set_state(s[:2],s[2:4]);e.sim.forward();e.set_marker();im=e.sim.render(512,512)
   path=D/f'{c["key"]}_{mode}.png';Image.fromarray(im).save(path);c['frames'][mode]={'path':str(path.relative_to(O)),'sha256':sha(path)}
  camera=e.sim.render_contexts[0].scn.camera;pos=.5*(np.asarray(camera[0].pos)+np.asarray(camera[1].pos));scale=256*float(camera[0].frustum_near)/(float(camera[0].frustum_top)*float(pos[2]))
  c['projection']={'kind':'maze_camera','scale_pixels_per_world_unit':scale,'body_offset_xy':offset.tolist(),'camera_position':pos.tolist(),'offset_source':'model.body_pos[particle]','framing':'native top-down renderer centered on complete wall bounds for visualization'}
  for mode in ['base','ours']:
   xy=np.array(c['paths'][mode])[:,:2]+offset;hits=0;n=0
   for gi,name in enumerate(e.model.geom_names):
    if name and 'wall' in name:
     n+=1;center=e.model.geom_pos[gi,:2];half=e.model.geom_size[gi,:2];hits+=int(np.any(np.all(np.abs(xy-center)<half-1e-5,axis=1)))
   assert n and hits==0,(c['key'],mode,hits);c['wall_core_crossings'][mode]=hits
  e.close()
  # Preserve complete outer walls with an equal margin for both arms.
  bounds=[]
  for f in c['frames'].values():
   im=np.array(Image.open(O/f['path']));mask=(im[:,:,0]>1.3*im[:,:,2])&(im[:,:,0]>100)&(im[:,:,1]>65);yy,xx=np.where(mask)
   assert min(xx.min(),yy.min())>=18 and max(xx.max(),yy.max())<494, 'Native render must include complete walls and margin'
   bounds.extend([[xx.min(),yy.min()],[xx.max(),yy.max()]])
  xy=np.array(bounds);lo=xy.min(0)-18;hi=xy.max(0)+18
 else:
  bounds=[]
  for f in c['frames'].values():
   im=np.array(Image.open(O/f['path'])).astype(float);mask=(im[:,:,2]>im[:,:,0]*1.025)&(im[:,:,2]>im[:,:,1]*1.025);yy,xx=np.where(mask)
   assert len(xx);bounds.extend([[xx.min(),yy.min()],[xx.max(),yy.max()]])
  xy=np.vstack([np.array(bounds),np.array(c['paths']['base'])[:,:2],np.array(c['paths']['ours'])[:,:2],np.array(c['goal'])[:2]])
  lo=xy.min(0)-22;hi=xy.max(0)+22
 center=(lo+hi)/2;side=float(max(hi-lo));assert side<=512
 lo=np.clip(center-side/2,0,512-side);c['crop']=[*lo.tolist(),*(lo+side).tolist()]
 c['crop_policy']='Square union of full native geometry and displayed paths, with equal paired margins; no stretching or rotation.'
 print(c['key'],c['crop'],c['endpoint_errors'],flush=True)
data['operation']='Existing saved-action replay and native state rendering only; no model inference or new benchmark runs.'
(O/'provenance/replayed_paths_v12.json').write_text(json.dumps(data,indent=2)+'\n')
