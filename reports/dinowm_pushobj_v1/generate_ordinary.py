#!/usr/bin/env python3
"""Reconstruct only the plain replay matched to the existing augmented dataset."""
import argparse,hashlib,json,os,pickle,sys,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
P=argparse.ArgumentParser();P.add_argument('--repo',type=Path,required=True);P.add_argument('--augmented',type=Path,required=True);P.add_argument('--output',type=Path,required=True);P.add_argument('--workers',type=int,default=24)
G=None;SRC=None

def init(repo):
 global G,SRC
 torch.set_num_threads(1);sys.path.insert(0,str(repo));sys.path.insert(0,str(repo/'scripts'));import gen_pushobj_dataset as G
 SRC=G.load_source(repo/'data/pusht_noise_meta');G._init_worker()

def run(job):
 record,velocity,path=job;S,A,V,L=SRC;i=record['source_index'];shape=record['shape'];state=np.r_[record['init_state'],velocity];length=int(record['length'])
 seed=int(i*7+['T','L','Z','+'].index(shape));frames,states,contacts,acts=G._replay(shape,state,A[i,:length],seed,0,0)
 assert np.isfinite(states).all() and len(states)==length
 import imageio
 writer=imageio.get_writer(path,fps=10,codec='libx264',ffmpeg_params=['-threads','1'])
 for frame in frames:writer.append_data(frame)
 writer.close()
 return record['episode'],states.astype(np.float32),acts.astype(np.float32),contacts

def main(a):
 a.output.mkdir(parents=True,exist_ok=True);start=time.monotonic();audit={'status':'running','augmented_reused_without_changes':str(a.augmented),'ordinary_additional_action_bias_px':0,'ordinary_additional_noise_px':0,'matching':'same source episode, initial pose/velocity, length and shape; augmentation dataset selection/placement rules held fixed','splits':{}}
 with ProcessPoolExecutor(a.workers,initializer=init,initargs=(a.repo,)) as pool:
  for split in ['train','val']:
   src=a.augmented/split;dst=a.output/split;(dst/'obses').mkdir(parents=True,exist_ok=True)
   records=json.loads((src/'provenance.json').read_text());velocity=torch.load(src/'velocities.pth',weights_only=True)[:,0].numpy();starts=torch.load(src/'states.pth',weights_only=True)[:,0].numpy();lengths=pickle.load(open(src/'seq_lengths.pkl','rb'));n=len(records);maxlen=max(lengths)
   states=torch.zeros(n,maxlen,5);vels=torch.zeros(n,maxlen,2);acts=torch.zeros(n,maxlen,2);prov=[None]*n
   jobs=[(r,velocity[i],str(dst/'obses'/f'episode_{i:03d}.mp4')) for i,r in enumerate(records)]
   for done,(i,s,ac,c) in enumerate(pool.map(run,jobs),1):
    t=len(s);states[i,:t]=torch.from_numpy(s[:,:5]);vels[i,:t]=torch.from_numpy(s[:,5:]);acts[i,:t]=torch.from_numpy(ac)
    np.testing.assert_allclose(s[0,:5],starts[i],atol=1e-5,rtol=0);np.testing.assert_allclose(s[0,5:],velocity[i],atol=1e-5,rtol=0)
    prov[i]={**records[i],'contact_steps':c,'reconstruction':'original source actions, no added bias/noise'}
    if done%200==0 or done==n:
     row={'status':'generating','split':split,'complete':done,'total':n,'elapsed_seconds':time.monotonic()-start};(a.output/'progress.json').write_text(json.dumps(row,indent=2)+'\n');print(json.dumps(row),flush=True)
   for key,val in [('states',states),('velocities',vels),('rel_actions',acts)]:torch.save(val,dst/f'{key}.pth')
   import shutil
   for name in ['seq_lengths.pkl','shapes.pkl']:shutil.copy2(src/name,dst/name)
   (dst/'provenance.json').write_text(json.dumps(prov)+'\n')
   audit['splits'][split]={'episodes':n,'windows':sum(int(l)-19 for l in lengths),'sources':sorted(set(r['source_index'] for r in records)),'augmented_metadata_sha256':{name:hashlib.sha256((src/name).read_bytes()).hexdigest() for name in ['states.pth','velocities.pth','rel_actions.pth','provenance.json','seq_lengths.pkl','shapes.pkl']}}
 assert not set(audit['splits']['train']['sources'])&set(audit['splits']['val']['sources'])
 audit.update(status='complete',wall_seconds=time.monotonic()-start,all_initial_states_and_velocities_matched=True)
 (a.output/'generation_config.json').write_text(json.dumps(audit,indent=2)+'\n');(a.output/'manifest.json').write_text(json.dumps(audit,indent=2)+'\n');print('ORDINARY MATCHED DATA COMPLETE',flush=True)
if __name__=='__main__':main(P.parse_args())
