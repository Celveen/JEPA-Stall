#!/usr/bin/env python3
"""Matched plain/augmented PushObj: same sources, starts, shapes and100-frame lengths."""
import argparse,hashlib,json,os,pickle,sys,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
P=argparse.ArgumentParser();P.add_argument('--repo',type=Path,required=True);P.add_argument('--output',type=Path,required=True);P.add_argument('--train-per-shape',type=int,default=4500);P.add_argument('--val-per-shape',type=int,default=500);P.add_argument('--workers',type=int,default=24);P.add_argument('--seed',type=int,default=1109)
SHAPES=['T','L','Z','+']; G=None; SOURCE=None

def init(repo):
 global G,SOURCE
 torch.set_num_threads(1);sys.path.insert(0,str(repo));sys.path.insert(0,str(repo/'scripts'))
 import gen_pushobj_dataset as G
 SOURCE=G.load_source(repo/'data/pusht_noise_meta');G._init_worker()

def run(job):
 shape,source,seed,folder,attempt_idx=job
 S,A,V,L=SOURCE
 state=np.r_[S[source,0],V[source,0]];rng=np.random.RandomState(seed)
 # Shared physical placement; choose it before either rollout.
 for k in range(5):
  init=state.copy()
  if k:init[2:4]+=rng.normal(0,20,2);init[4]+=rng.normal(0,.3)
  if G._placement_ok(shape,init,seed,3):break
 else:return {'valid':False,'reason':'initial_pose'}
 results={}
 for arm,bias,noise in [('ordinary',0,0),('augmented',9,10)]:
  frames,states,contacts,actions=G._replay(shape,init,A[source,:100],seed,bias,noise,seed*31)
  disp=np.linalg.norm(np.diff(states[:,2:4],axis=0),axis=1)
  if not np.isfinite(states).all() or ((states[:,:2]<0)|(states[:,:2]>512)).any() or disp.max(initial=0)>80:
   return {'valid':False,'reason':arm+'_invalid_rollout'}
  results[arm]=(frames,states,actions,contacts)
 # Same candidate acceptance policy in both arms; no success/contact filtering.
 paths={}
 import imageio
 for arm,(frames,states,actions,contacts) in results.items():
  p=Path(folder)/arm/f'{shape.replace("+","plus")}_{attempt_idx:05d}.mp4';p.parent.mkdir(parents=True,exist_ok=True)
  writer=imageio.get_writer(p,fps=10,codec='libx264',ffmpeg_params=['-threads','1'])
  for f in frames:writer.append_data(f)
  writer.close();paths[arm]=str(p)
 return {'valid':True,'source_index':source,'shape':shape,'seed':seed,'init_state':init.tolist(),'paths':paths,'data':{arm:{'states':x[1][:,:5].astype(np.float32),'velocities':x[1][:,5:7].astype(np.float32),'rel_actions':x[2].astype(np.float32),'contacts':x[3]} for arm,x in results.items()}}

def main(a):
 a.output.mkdir(parents=True,exist_ok=True)
 if (a.output/'manifest.json').exists():raise RuntimeError('Complete dataset already exists; reuse it rather than regenerate.')
 sys.path.insert(0,str(a.repo/'scripts'));from gen_pushobj_dataset import load_source
 S,A,V,L=load_source(a.repo/'data/pusht_noise_meta');eligible=(L>=100)&np.array([not ((s[:int(n),:2]<0)|(s[:int(n),:2]>512)).any() for s,n in zip(S,L)])
 manifest={'seed':a.seed,'shapes':SHAPES,'frames_per_trajectory':100,'split_source_boundary':16000,'augmentation':{'bias_toward_block_cog_px':9,'isotropic_action_noise_std_px':10},'ordinary':{'bias_px':0,'additional_noise_px':0},'selection':'same source, initial pose and candidate acceptance in both arms; no contact filtering; paired rollout numerical/workspace quality check','source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (a.repo/'data/pusht_noise_meta/train').glob('*') if p.is_file()},'splits':{}}
 start=time.monotonic()
 with ProcessPoolExecutor(a.workers,initializer=init,initargs=(a.repo,)) as pool:
  for split,target in [('train',a.train_per_shape),('val',a.val_per_shape)]:
   ids=np.where(eligible & ((np.arange(len(L))<16000) if split=='train' else (np.arange(len(L))>=16000)))[0]
   records=[];rejected=0
   for si,shape in enumerate(SHAPES):
    sources=np.random.RandomState(a.seed+si+(100 if split=='val' else 0)).permutation(ids)
    selected=[];used=0
    while len(selected)<target:
     batch=sources[used:used+min(256,target-len(selected)+32)]
     if not len(batch):raise RuntimeError(f'Not enough paired valid sources: {split}/{shape} {len(selected)}/{target}')
     jobs=[(shape,int(src),a.seed*100000+int(src)*7+si,str(a.output/'staging'/split),used+i) for i,src in enumerate(batch)]
     for result in pool.map(run,jobs):
      if result['valid'] and len(selected)<target:selected.append(result)
      elif result['valid']:
       for p in result['paths'].values():Path(p).unlink()
      else:rejected+=1
     used+=len(batch)
     (a.output/'progress.json').write_text(json.dumps({'split':split,'shape':shape,'accepted':len(selected),'target_per_shape':target,'rejected_pairs':rejected,'seconds':time.monotonic()-start},indent=2)+'\n')
     print(split,shape,len(selected),'/',target,'seconds',round(time.monotonic()-start),flush=True)
    records.extend(selected)
   provenance=[]
   for arm in ['ordinary','augmented']:
    dst=a.output/arm/split;(dst/'obses').mkdir(parents=True,exist_ok=True)
    tensors={key:torch.from_numpy(np.stack([v['data'][arm][key] for v in records])) for key in ['states','velocities','rel_actions']}
    for key,t in tensors.items():torch.save(t,dst/f'{key}.pth')
    for name,obj in [('seq_lengths',[100]*len(records)),('shapes',[v['shape'] for v in records])]:
     with (dst/f'{name}.pkl').open('wb') as f:pickle.dump(obj,f)
    for i,v in enumerate(records):Path(v['paths'][arm]).replace(dst/'obses'/f'episode_{i:03d}.mp4')
    prov=[{'episode':i,**{k:v[k] for k in ['shape','source_index','seed','init_state']},'contact_steps':v['data'][arm]['contacts']} for i,v in enumerate(records)]
    (dst/'provenance.json').write_text(json.dumps(prov)+'\n')
   for key in ['states','velocities']:
    x=torch.load(a.output/'ordinary'/split/f'{key}.pth',weights_only=True);y=torch.load(a.output/'augmented'/split/f'{key}.pth',weights_only=True);assert torch.equal(x[:,0],y[:,0])
   manifest['splits'][split]={'episodes_per_arm':len(records),'unique_sources':len(set(v['source_index'] for v in records)),'source_indices':sorted(set(v['source_index'] for v in records)),'rejected_pairs':rejected,'shape_counts':{s:target for s in SHAPES}}
   del records
 assert not set(manifest['splits']['train']['source_indices'])&set(manifest['splits']['val']['source_indices'])
 manifest['wall_seconds']=time.monotonic()-start;manifest['status']='complete'
 for arm in ['ordinary','augmented']:(a.output/arm/'generation_config.json').write_text(json.dumps(manifest,indent=2)+'\n')
 (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print('MATCHED DATA COMPLETE',flush=True)
if __name__=='__main__':main(P.parse_args())
