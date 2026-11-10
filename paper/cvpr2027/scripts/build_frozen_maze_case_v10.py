#!/usr/bin/env python3
"""Render a completed frozen Maze paired episode; no trajectory is re-executed."""
import argparse,hashlib,json,os,pickle,sys
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/jepa_maze_v10_mpl');os.environ.setdefault('MESA_SHADER_CACHE_DIR','/tmp/jepa_maze_v10_mesa')
import numpy as np
from PIL import Image
ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--paper',type=Path,required=True);ap.add_argument('--sample',type=int,default=7);a=ap.parse_args();R=a.repo;O=a.paper;i=a.sample
sys.path.insert(0,str(R))
import gym,env
from datasets.diverse_maze_goals import generate_diverse_maze_goals
br=R/'eval_outputs/stag_maze100_ar4/medium/frozen/ref';ar=R/'eval_outputs/maze_frozen_v10/S'
def load(root):
 rr={};paths=[]
 for p in root.rglob('per_iter_logs.json'):
  selected=[]
  for line in p.read_text().splitlines():
   x=json.loads(line)
   if x['sample_idx']==i and x['step']<=30:selected.append(x)
  if selected:
   paths.append(str(p.relative_to(R)))
   for x in selected:assert x['step'] not in rr;rr[x['step']]=x
 return rr,paths
b,bpaths=load(br);o,opaths=load(ar)
assert not b[30]['success'] and o[max(o)]['success']
npth=R/opaths[0];npth=npth.parent/'plan_targets.pkl'
with npth.open('rb') as f:nt=pickle.load(f)
with (br/'shard00/plan_targets.pkl').open('rb') as f:bt=pickle.load(f)
for k in ['state_0','state_g']:np.testing.assert_array_equal(nt[k],bt[k])
s0=bt['state_0'][i];goal=bt['state_g'][i]
meta=[json.loads(l) for l in (npth.parent/'stag_episodes.jsonl').read_text().splitlines() if json.loads(l)['sample_idx']==i][0]
t=meta['first_trigger'];assert t is not None
for j in range(1,t+1):
 np.testing.assert_array_equal(b[j]['state'],o[j]['state']);np.testing.assert_array_equal(b[j]['act'],o[j]['act'])
g=generate_diverse_maze_goals(R/'data/diverse_maze/val',100,100,min_block_radius=6,max_block_radius=9)
np.testing.assert_allclose(s0,g['start_states'][i],atol=1e-6);np.testing.assert_allclose(goal,g['goal_states'][i],atol=1e-6)
e=gym.make('diverse_maze',maze_spec=g['map_specs'][i],action_repeat=4).unwrapped;e.prepare(0,s0)
def no_step(*args,**kwargs):raise RuntimeError('No action execution is permitted for rendering logged states')
e.step=no_step
D=O/'figures/gallery_v10';D.mkdir(exist_ok=True)
c={'key':'maze_frozen','label':'Maze (frozen)','task':'Maze','sample_idx':i,'baseline':'Frozen','intervention':'ECP (S)','baseline_dir':str(br.relative_to(R)),'ecp_dir':str(ar.relative_to(R)),'start':s0.tolist(),'goal':goal.tolist(),'states':{'base':[s0.tolist()]+[b[j]['state'] for j in sorted(b)],'ours':[s0.tolist()]+[o[j]['state'] for j in sorted(o)]},'baseline_success':False,'ours_success':True,'matching_prefix_replans':t,'crop':[16,64,450,498],'frames':{},'baseline_log_paths':bpaths,'ecp_log_paths':opaths,'completed_episode_metadata':meta,'map_index':int(g['map_indices'][i]),'maze_spec':g['map_specs'][i]}
for mode,s in [('start',s0),('goal',goal),('base',b[30]['state']),('ours',o[max(o)]['state'])]:
 s=np.asarray(s);e.set_state(s[:2],s[2:4]);e.set_marker();im=e.sim.render(512,512);p=D/f'maze_frozen_{mode}.png';Image.fromarray(im).save(p);c['frames'][mode]={'path':str(p.relative_to(O)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
ctx=e.sim.render_contexts[0];p0=ctx.scn.camera[0];p1=ctx.scn.camera[1];pos=.5*(np.asarray(p0.pos)+np.asarray(p1.pos));scale=256*float(p0.frustum_near)/(float(p0.frustum_top)*float(pos[2]));offset=e.sim.data.get_geom_xpos('particle_geom')[:2]-e.sim.data.qpos[:2]
c['projection']={'formula':'pixel_xy=256+s*(qpos_xy+body_offset_xy-camera_xy)','scale_pixels_per_world_unit':scale,'body_offset_xy':offset.tolist(),'camera_position':pos.tolist()}
# Preserve immutable per-episode evidence while other episodes append to their shard.
snap={'baseline_rows':[b[j] for j in sorted(b)],'ecp_rows':[o[j] for j in sorted(o)],'sample_idx':i,'first_trigger':t,'state_0':s0.tolist(),'state_g':goal.tolist()}
p=O/'provenance/maze_frozen_case_v10_records.json';p.write_text(json.dumps(snap,indent=2)+'\n');c['selected_records_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
(O/'provenance/maze_frozen_case_v10.json').write_text(json.dumps(c,indent=2)+'\n');print(json.dumps({'sample':i,'baseline_failure_replan':30,'ecp_success_replan':max(o),'identical_prefix':t,'native_renderer':'MuJoCo'},indent=2))
