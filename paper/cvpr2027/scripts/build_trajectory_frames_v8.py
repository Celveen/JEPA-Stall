#!/usr/bin/env python3
"""Render saved PushObj states directly; never run actions or physics steps."""
import argparse, hashlib, importlib.util, json, os
from pathlib import Path
os.environ.setdefault('SDL_VIDEODRIVER','dummy')
os.environ.setdefault('SDL_AUDIODRIVER','dummy')
os.environ.setdefault('MPLCONFIGDIR','/tmp/jepa_native_frames_mpl')
import numpy as np
from PIL import Image
ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--paper',type=Path,required=True);args=ap.parse_args();R=args.repo;O=args.paper
src=R/'env/pusht/pusht_env.py'
spec=importlib.util.spec_from_file_location('native_pushobj_renderer',src)
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
F=O/'figures'/'trajectory_frames';F.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
m={'rendering':'Native PushTEnv renderer from source; direct pose assignment after scene construction. No reset, action, env.step, or physics step.','sources':{str(src.relative_to(R)):sha(src)},'cases':{}}
for chunk,i,shape,label in [('chunk_24_small_tee',9,'small_tee','pushobj_tee')]:
 folder=R/'eval_outputs/stag_frozen_bias16k_it40'
 br=folder/'ref'/chunk;orr=folder/'T0SP'/chunk
 npz1=br/'final_eval_results.npz';npz2=orr/'final_eval_results.npz'
 b=np.load(npz1);o=np.load(npz2)
 for key in ['state_0','state_g']:np.testing.assert_array_equal(b[key],o[key])
 for p in [npz1,npz2,br/'per_iter_logs.json',orr/'per_iter_logs.json',orr/'stag_meta.json']:m['sources'][str(p.relative_to(R))]=sha(p)
 s0=b['state_0'][i];goal=b['state_g'][i];rows={}
 for mode,path in [('base',br),('ours',orr)]:
  allrows=[json.loads(l) for l in (path/'per_iter_logs.json').read_text().splitlines()]
  rows[mode]={q['step']:q for q in allrows if q['sample_idx']==i and q['step']<=30}
 assert not rows['base'][30]['success'];assert rows['ours'][max(rows['ours'])]['success']
 match=0
 for j in range(1,min(max(rows['base']),max(rows['ours']))+1):
  if rows['base'][j]['state']!=rows['ours'][j]['state'] or rows['base'][j]['act']!=rows['ours'][j]['act']:break
  match=j
 states={mode:np.array([s0]+[rr[k]['state'] for k in sorted(rr)]) for mode,rr in rows.items()}
 c={'shape':shape,'sample_idx':i,'baseline_path':str(br.relative_to(R)),'correction_path':str(orr.relative_to(R)),'goal':goal.tolist(),'states':{k:v.tolist() for k,v in states.items()},'matching_action_and_state_prefix_replans':match,'baseline_condition':'frozen MPC','correction':'S+P','original_target_array_pairing_verified':True,'explicit_outcome_check':{'baseline':False,'ours':True},'frames':{}}
 e=mod.PushTEnv(shape=shape,with_target=False,with_velocity=True,render_size=512)
 e._setup() # Construct scene only. reset() would call _set_state()/space.step().
 def prohibited(*a,**kw):raise RuntimeError('Physics or action execution is forbidden for figure rendering')
 e.space.step=prohibited;e.step=prohibited
 for mode,s in [('start',s0),('before',states['base'][match]),('base',states['base'][-1]),('ours',states['ours'][-1])]:
  e.block.angle=float(s[4]);e.block.position=tuple(s[2:4]);e.agent.position=tuple(s[:2]);e.agent.velocity=tuple(s[5:7]);e.space.reindex_shapes_for_body(e.block);e.space.reindex_shapes_for_body(e.agent)
  assert np.max(np.abs(np.r_[e.agent.position,e.block.position,e.block.angle]-s[:5]))<1e-8
  img=e.render('rgb_array');p=F/f'{label}_{mode}.png';Image.fromarray(img).save(p)
  c['frames'][mode]={'path':str(p.relative_to(O)),'sha256':sha(p)}
 m['cases'][label]=c
m['generator_sha256']=sha(__file__)
(O/'provenance'/'trajectory_extra_frames_v8.json').write_text(json.dumps(m,indent=2)+'\n')
print(json.dumps({k:{'matching_prefix':v['matching_action_and_state_prefix_replans'],'steps':{kk:len(vv)-1 for kk,vv in v['states'].items()}} for k,v in m['cases'].items()}))
