"""T-only LeWM recipe with global batch 128 distributed across four GPUs.

Rank zero uses the original WindowBatcher and scatters its unchanged batches.
SyncBatchNorm and differentiable global SIGReg preserve the full-batch objective.
"""
import json
import sys
import time
from pathlib import Path
import hydra
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from omegaconf import OmegaConf
from datasets.window_stream import WindowBatcher
from models.lewm import build_lewm
from train_lewm import lr_at
from distributed_training import initialize, GlobalSIGReg, TrainingModel, scatter_batch, reduce_losses, normalized_visual


@torch.no_grad()
def recalibrate(model, source, transform, device, batches):
    world_model = model.module.world_model
    layers = [m for m in world_model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    momenta = [m.momentum for m in layers]
    for layer in layers:
        layer.reset_running_stats(); layer.momentum = None
    model.train()
    iterator = iter(source) if dist.get_rank() == 0 else None
    for _ in range(batches):
        visual, actions = next(iterator) if iterator is not None else (None,None)
        visual, actions = scatter_batch(visual,actions,device,uint8=True)
        visual = normalized_visual(visual,transform)
        hs = world_model.history_size
        world_model.predict(world_model.embed(visual)[:,:hs],world_model.action_encoder(torch.nan_to_num(actions,0.))[:,:hs])
    for layer,momentum in zip(layers,momenta):layer.momentum=momentum


@torch.no_grad()
def evaluate(model, source, device, batches):
    model.eval()
    total = torch.zeros(3,device=device)
    iterator = iter(source) if dist.get_rank() == 0 else None
    for _ in range(batches):
        if iterator is not None:
            obs,actions,*_=next(iterator);visual=obs['visual']
        else:visual,actions=None,None
        visual,actions=scatter_batch(visual,actions,device,uint8=False)
        with torch.autocast('cuda',dtype=torch.bfloat16):values=model(visual,actions)
        total+=reduce_losses(values)
    model.train()
    return dict(zip(['loss','pred_loss','sigreg_loss'],(total/batches).tolist()))


def main():
    rank,world,device=initialize()
    assert world==4
    cfg=OmegaConf.load(sys.argv[1])
    assert cfg.training.epochs == 3 and cfg.training.batch_size == 128
    torch.manual_seed(cfg.training.seed);np.random.seed(cfg.training.seed)
    output=Path(cfg.saved_folder)
    metadata=[None]
    train_loader=bn_loader=val_loader=None
    if rank==0:
        (output/'checkpoints').mkdir(parents=True,exist_ok=True)
        datasets,_=hydra.utils.call(cfg.env.dataset,num_hist=cfg.num_hist,num_pred=cfg.num_pred,frameskip=cfg.frameskip)
        train_set,val_set=datasets['train'],datasets['valid']
        train_loader=WindowBatcher(train_set,cfg.training.batch_size,cfg.training.num_workers,cfg.training.seed)
        bn_loader=WindowBatcher(train_set,cfg.training.batch_size,cfg.training.num_workers,cfg.training.seed+1)
        val_loader=torch.utils.data.DataLoader(val_set,batch_size=cfg.training.batch_size,shuffle=True,
            num_workers=cfg.training.num_workers,pin_memory=True,generator=torch.Generator().manual_seed(0))
        metadata[0]=(len(train_set),len(val_set),len(train_loader),int(train_set.action_dim))
    dist.broadcast_object_list(metadata,src=0,device=device)
    train_windows,val_windows,steps_per_epoch,action_dim=metadata[0]
    assert train_windows==1633205 and steps_per_epoch==12759 and val_windows==101784
    OmegaConf.update(cfg,'model.action_dim',action_dim,force_add=True)
    if rank==0:OmegaConf.save(cfg,output/'hydra.yaml')
    transform=hydra.utils.call(cfg.env.dataset.transform)
    native=build_lewm(cfg.model).to(device)
    native=torch.nn.SyncBatchNorm.convert_sync_batchnorm(native)
    module=TrainingModel(native,GlobalSIGReg(**cfg.training.sigreg).to(device),cfg.training.sigreg_weight)
    model=DDP(module,device_ids=[rank],output_device=rank)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg.training.lr,weight_decay=cfg.training.weight_decay)
    # Distinct per-rank dropout streams; model initialization remains seed 0.
    torch.cuda.manual_seed(cfg.training.seed+rank)
    total_steps=cfg.training.epochs*steps_per_epoch
    step=0;t0=time.time()
    logfile=(output/'metrics.jsonl').open('a') if rank==0 else None
    model.train()
    if rank==0:
        print(f'train windows {train_windows}, val windows {val_windows}, action_dim {action_dim}',flush=True)
        print(f'{world} GPUs; global batch {cfg.training.batch_size}; parameters {sum(p.numel() for p in model.parameters())}',flush=True)
    for epoch in range(1,cfg.training.epochs+1):
        run=torch.zeros(3,device=device);nrun=0
        iterator=iter(train_loader) if rank==0 else None
        for _ in range(steps_per_epoch):
            visual,actions=next(iterator) if iterator is not None else (None,None)
            visual,actions=scatter_batch(visual,actions,device,uint8=True)
            visual=normalized_visual(visual,transform)
            for group in optimizer.param_groups:group['lr']=lr_at(step,total_steps,cfg.training.lr)
            with torch.autocast('cuda',dtype=torch.bfloat16):losses=model(visual,actions)
            assert all(torch.isfinite(value).item() for value in losses), 'Non-finite loss'
            optimizer.zero_grad(set_to_none=True);losses[0].backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.training.grad_clip)
            assert torch.isfinite(norm).item(),'Non-finite training gradient'
            optimizer.step();step+=1;run+=reduce_losses(losses);nrun+=1
            if step%cfg.training.log_every==0:
                if rank==0:
                    row={'step':step,'epoch':epoch,'lr':optimizer.param_groups[0]['lr'],
                         **dict(zip(['loss','pred_loss','sigreg_loss'],(run/nrun).tolist())),
                         'windows_per_s':step*cfg.training.batch_size/(time.time()-t0)}
                    print(json.dumps(row),flush=True);logfile.write(json.dumps(row)+'\n');logfile.flush()
                run.zero_();nrun=0
        recalibrate(model,bn_loader,transform,device,cfg.training.bn_batches)
        validation=evaluate(model,val_loader,device,cfg.training.val_batches)
        if rank==0:
            row={'epoch':epoch,'step':step,'val':validation,'hours':(time.time()-t0)/3600}
            print(json.dumps(row),flush=True);logfile.write(json.dumps(row)+'\n');logfile.flush()
            payload={'model':native.state_dict(),'epoch':epoch,'step':step}
            torch.save(payload,output/'checkpoints'/f'model_{epoch}.pth')
            torch.save(payload,output/'checkpoints/model_latest.pth')
        dist.barrier()
    if rank==0:print('done',flush=True)
    dist.destroy_process_group()


if __name__=='__main__':main()
