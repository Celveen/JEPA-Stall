#!/usr/bin/env python3
"""DINO-WM native model/loss and AdamW recipe, with DDP, microbatches and resumable logs."""
import argparse,contextlib,copy,hashlib,json,math,os,random,sys,time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,DistributedSampler,BatchSampler,SequentialSampler
from omegaconf import OmegaConf
P=argparse.ArgumentParser();P.add_argument('--repo',type=Path,required=True);P.add_argument('--data',type=Path,required=True);P.add_argument('--output',type=Path,required=True);P.add_argument('--epochs',type=int,default=3);P.add_argument('--batch',type=int,default=32);P.add_argument('--microbatch',type=int,default=8);P.add_argument('--workers',type=int,default=6);P.add_argument('--seed',type=int,default=0);P.add_argument('--precision',choices=['bf16','fp32'],default='bf16');P.add_argument('--max-steps',type=int,default=0)

class ResumeBatchSampler(BatchSampler):
 start_batch=0
 def __iter__(self):
  for i, batch in enumerate(super().__iter__()):
   if i>=self.start_batch:yield batch

def atomic(path,obj):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)
def tensor_hash(module):
 h=hashlib.sha256()
 for k,v in module.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def main(a):
 rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0));torch.cuda.set_device(local)
 if world>1:dist.init_process_group('nccl')
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 # Leave at least half each card to already-running evaluations.
 torch.cuda.set_per_process_memory_fraction(4*2**30 / torch.cuda.get_device_properties(local).total_memory,local)
 assert a.batch%world==0 and (a.batch//world)%a.microbatch==0
 random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
 a.output.mkdir(parents=True,exist_ok=True);sys.path.insert(0,str(a.repo))
 from datasets.pusht_dset import load_pusht_slice_train_val
 from datasets.img_transforms import default_transform
 from native_models.dino import DinoV2Encoder
 from native_models.proprio import ProprioceptiveEmbedding
 from native_models.vit import ViTPredictor
 from native_models.visual_world_model import VWorldModel
 datasets,_=load_pusht_slice_train_val(default_transform(224),n_rollout=None,data_path=str(a.data),num_hist=3,num_pred=1,frameskip=5,with_velocity=True)
 encoder=DinoV2Encoder('dinov2_vits14','x_norm_patchtokens')
 for p in encoder.parameters():p.requires_grad_(False)
 prop=ProprioceptiveEmbedding(num_frames=1,in_chans=4,emb_dim=10,tubelet_size=1,use_3d_pos=False)
 action=ProprioceptiveEmbedding(num_frames=1,in_chans=10,emb_dim=10,tubelet_size=1,use_3d_pos=False)
 predictor=ViTPredictor(num_patches=196,num_frames=3,dim=404,depth=6,heads=16,mlp_dim=2048,dropout=.1,emb_dropout=0,pool='mean')
 wm=VWorldModel(224,3,1,encoder,prop,action,None,predictor,proprio_dim=10,action_dim=10,concat_dim=1,num_action_repeat=1,num_proprio_repeat=1,train_encoder=False,train_predictor=True,train_decoder=False).cuda(local)
 opt=torch.optim.AdamW(wm.predictor.parameters(),lr=5e-4);oa=torch.optim.AdamW(list(wm.proprio_encoder.parameters())+list(wm.action_encoder.parameters()),lr=5e-4)
 initial={'encoder':tensor_hash(encoder),'predictor':tensor_hash(predictor),'action_encoder':tensor_hash(action),'proprio_encoder':tensor_hash(prop)}
 cfg={'env':{'name':'pushobj','args':[],'kwargs':{'with_velocity':True,'with_target':True},'dataset':{'data_path':str(a.data),'transform':{'_target_':'datasets.img_transforms.default_transform','img_size':224}}},'encoder':{'_target_':'native_models.dino.DinoV2Encoder','name':'dinov2_vits14','feature_key':'x_norm_patchtokens'},'model':{'_target_':'native_models.visual_world_model.VWorldModel','image_size':224,'num_hist':3,'num_pred':1,'train_encoder':False,'train_predictor':True,'train_decoder':False},'img_size':224,'frameskip':5,'num_hist':3,'num_pred':1,'has_predictor':True,'has_decoder':False,'concat_dim':1,'proprio_emb_dim':10,'action_emb_dim':10,'num_action_repeat':1,'num_proprio_repeat':1,'training':{'epochs':a.epochs,'batch_size':a.batch,'predictor_lr':5e-4,'action_encoder_lr':5e-4,'seed':a.seed,'precision':a.precision},'native_commit':'0a9492fa12044b852ae9e001cc74604b79c8bb0c'}
 if rank==0:
  (a.output/'hydra.yaml').write_text(OmegaConf.to_yaml(cfg));atomic(a.output/'initialization.json',initial)
 batch=a.batch//world
 loaders={}
 for key,ds in datasets.items():
  sampler=DistributedSampler(ds,num_replicas=world,rank=rank,shuffle=False) if world>1 else SequentialSampler(ds)
  loaders[key]=DataLoader(ds,batch_sampler=ResumeBatchSampler(sampler,batch_size=batch,drop_last=False),num_workers=a.workers,pin_memory=True,persistent_workers=a.workers>0,generator=torch.Generator().manual_seed(a.seed+rank+123))
 start_epoch=1;start_batch=0;step=0;resume=a.output/'checkpoints/model_latest.pth'
 if resume.exists():
  c=torch.load(resume,map_location='cpu',weights_only=False)
  for name in ['encoder','predictor','action_encoder','proprio_encoder']:getattr(wm,name).load_state_dict(c[name].state_dict())
  opt.load_state_dict(c['predictor_optimizer']);oa.load_state_dict(c['action_encoder_optimizer']);start_epoch=c['next_epoch'];start_batch=c['next_batch'];step=c['global_step']
  rng=c['rng'][rank];random.setstate(rng['python']);np.random.set_state(rng['numpy']);torch.set_rng_state(rng['torch']);torch.cuda.set_rng_state(rng['cuda'],local)
 if rank==0:
  atomic(a.output/'resume.json',{'start_epoch':start_epoch,'start_batch':start_batch,'global_step':step,'target_epochs':a.epochs,'resumed':resume.exists()})
 model=DDP(wm,device_ids=[local],broadcast_buffers=False) if world>1 else wm
 autocast=lambda:torch.autocast('cuda',dtype=torch.bfloat16,enabled=a.precision=='bf16')
 start=time.monotonic()
 def save(epoch,next_epoch,next_batch,archive=False):
  rng={'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state(local)}
  rr=[None]*world
  if world>1:dist.all_gather_object(rr,rng)
  else:rr=[rng]
  if rank==0:
   dst=a.output/'checkpoints';dst.mkdir(exist_ok=True)
   c={name:getattr(wm,name) for name in ['encoder','predictor','action_encoder','proprio_encoder']}
   c.update(epoch=epoch,next_epoch=next_epoch,next_batch=next_batch,global_step=step,predictor_optimizer=opt.state_dict(),action_encoder_optimizer=oa.state_dict(),rng=rr,protocol=cfg)
   torch.save(c,dst/'model_latest.tmp');(dst/'model_latest.tmp').replace(dst/'model_latest.pth')
   if archive:
    import shutil;shutil.copy2(dst/'model_latest.pth',dst/f'model_{epoch}.pth')
  if world>1:dist.barrier()
 for epoch in range(start_epoch,a.epochs+1):
  model.train();wm.encoder.eval();t_epoch=time.monotonic();total=0.;count=0
  offset=start_batch if epoch==start_epoch else 0
  loaders['train'].batch_sampler.start_batch=offset
  for bi,(obs,act,state) in enumerate(loaders['train'],start=offset):
   obs={k:v.cuda(local,non_blocking=True) for k,v in obs.items()};act=act.cuda(local,non_blocking=True);n=len(act);opt.zero_grad(set_to_none=True);oa.zero_grad(set_to_none=True);ls=0.
   for j in range(0,n,a.microbatch):
    end=min(j+a.microbatch,n);sync=model.no_sync() if world>1 and end<n else contextlib.nullcontext()
    with sync,autocast():loss=model({k:v[j:end] for k,v in obs.items()},act[j:end])[3];weighted=loss*((end-j)/n)
    weighted.backward();ls+=float(loss.detach())*(end-j)
   assert math.isfinite(ls),'Nonfinite training loss'
   opt.step();oa.step();step+=1;total+=ls;count+=n
   if rank==0 and (step<=3 or step%100==0):
    row={'phase':'train','epoch':epoch,'batch':bi+1,'batches':len(loaders['train']),'global_step':step,'loss':ls/n,'seconds':time.monotonic()-start,'max_gpu_memory_gb':torch.cuda.max_memory_allocated()/2**30}
    with (a.output/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    atomic(a.output/'status.json',{'status':'training',**row});print(json.dumps(row),flush=True)
   if step%1000==0:save(epoch-1,epoch,bi+1)
   if a.max_steps and step>=a.max_steps:
    save(0,epoch,bi+1,archive=True)
    if rank==0:
     assert tensor_hash(encoder)==initial['encoder'];atomic(a.output/'smoke_complete.json',{'steps':step,'frozen_encoder_unchanged':True,'finite_loss':True,'max_gpu_memory_gb':torch.cuda.max_memory_allocated()/2**30})
    if world>1:dist.destroy_process_group()
    return
  model.eval();vl=torch.zeros(2,device=local)
  with torch.no_grad():
   for obs,act,state in loaders['valid']:
    for j in range(0,len(act),a.microbatch):
     oa_batch={k:v[j:j+a.microbatch].cuda(local,non_blocking=True) for k,v in obs.items()};aa=act[j:j+a.microbatch].cuda(local,non_blocking=True)
     with autocast():loss=wm(oa_batch,aa)[3]
     vl[0]+=loss.float()*len(aa);vl[1]+=len(aa)
  if world>1:dist.all_reduce(vl)
  assert torch.isfinite(vl).all()
  save(epoch,epoch+1,0,archive=True)
  if rank==0:
   row={'phase':'validation','epoch':epoch,'global_step':step,'validation_loss':float(vl[0]/vl[1]),'epoch_seconds':time.monotonic()-t_epoch}
   with (a.output/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
   atomic(a.output/'status.json',{'status':'complete' if epoch==a.epochs else 'training',**row});print(json.dumps(row),flush=True)
 if rank==0:
  assert tensor_hash(encoder)==initial['encoder'];atomic(a.output/'complete.json',{'epoch':a.epochs,'global_step':step,'frozen_encoder_unchanged':True,'checkpoint':str(a.output/f'checkpoints/model_{a.epochs}.pth')})
 if world>1:dist.destroy_process_group()
if __name__=='__main__':main(P.parse_args())
