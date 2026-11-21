"""Verify distributed loss/gradients/BN against the full-batch reference."""
from pathlib import Path
import copy
import ctypes
import gc
import json
import sys
import time
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'runtime'))
import hydra
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from omegaconf import OmegaConf
from models.lewm import build_lewm,SIGReg
from distributed_training import initialize,GlobalSIGReg,TrainingModel,scatter_batch,reduce_losses,normalized_visual


def main():
    print('Initializing four-GPU communication', flush=True)
    rank,world,device=initialize();assert world==4
    print(f'Rank {rank}: initialized; building model', flush=True)
    cfg=OmegaConf.load(ROOT/'config.yaml');cfg.model.action_dim=10
    torch.manual_seed(0)
    native=build_lewm(cfg.model).to(device)
    plain=copy.deepcopy(native) if rank==0 else None
    native=torch.nn.SyncBatchNorm.convert_sync_batchnorm(native)
    dropout=[(m,m.p) for m in native.modules() if isinstance(m,torch.nn.Dropout)]
    attention_dropout=[(m,m.dropout) for m in native.modules() if isinstance(getattr(m,'dropout',None),(float,int))]
    for module,_ in dropout:module.p=0.
    for module,_ in attention_dropout:module.dropout=0.
    if plain is not None:
        for module in plain.modules():
            if isinstance(module,torch.nn.Dropout):module.p=0.
            if isinstance(getattr(module,'dropout',None),(float,int)):module.dropout=0.
    model=DDP(TrainingModel(native,GlobalSIGReg(**cfg.training.sigreg).to(device),cfg.training.sigreg_weight),device_ids=[rank])
    print(f'Rank {rank}: DDP ready', flush=True)
    transform=hydra.utils.call(cfg.env.dataset.transform)
    if rank==0:
        generator=torch.Generator().manual_seed(321)
        raw=torch.randint(0,256,(128,4,128,128,3),dtype=torch.uint8,generator=generator)
        acts=torch.randn(128,4,10,generator=generator)
        full=normalized_visual(raw[:16].to(device),transform)
        torch.cuda.manual_seed(987)
        reference=plain.loss(full,acts[:16].to(device),SIGReg(**cfg.training.sigreg).to(device),cfg.training.sigreg_weight)
        reference[0].backward()
        reference_values=[float(x) for x in reference]
    else:raw=acts=None
    dist.barrier()
    visual,action=scatter_batch(raw[:16] if rank==0 else None,acts[:16] if rank==0 else None,device,uint8=True)
    visual=normalized_visual(visual,transform)
    torch.cuda.manual_seed(987)
    values=model(visual,action);averages=reduce_losses(values)
    values[0].backward()
    metrics={}
    if rank==0:
        np.testing.assert_allclose(reference_values,averages.tolist(),rtol=3e-4,atol=3e-5)
        numerator=denominator=maximum=0.
        params=dict(native.named_parameters())
        for name,param in plain.named_parameters():
            other=params[name]
            assert (param.grad is None)==(other.grad is None),name
            if param.grad is not None:
                delta=(param.grad-other.grad).float()
                numerator+=float(delta.square().sum());denominator+=float(param.grad.float().square().sum())
                maximum=max(maximum,float(delta.abs().max()))
        relative=(numerator/max(denominator,1e-30))**.5
        assert relative<3e-3,relative
        buffers=dict(native.named_buffers())
        for name,value in plain.named_buffers():
            torch.testing.assert_close(value,buffers[name],rtol=3e-4,atol=3e-5)
        plain.load_state_dict(native.state_dict(),strict=True)
        metrics={'reference_losses':reference_values,'distributed_losses':averages.tolist(),
                 'gradient_relative_l2_error':relative,'gradient_max_absolute_error':maximum,
                 'global_batchnorm_buffers_match':True,'native_checkpoint_strict_load':True}
        del plain,params,reference,full;gc.collect();torch.cuda.empty_cache()
    dist.barrier()
    for module,probability in dropout:module.p=probability
    for module,probability in attention_dropout:module.dropout=probability
    model.zero_grad(set_to_none=True)
    visual,action=scatter_batch(raw,acts,device,uint8=True)
    visual=normalized_visual(visual,transform)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg.training.lr,weight_decay=cfg.training.weight_decay)
    torch.cuda.manual_seed(cfg.training.seed+rank)
    durations=[]
    torch.cuda.reset_peak_memory_stats(device)
    for step in range(20):
        dist.barrier();torch.cuda.synchronize();start=time.time()
        with torch.autocast('cuda',dtype=torch.bfloat16):losses=model(visual,action)
        assert all(torch.isfinite(x).item() for x in losses)
        optimizer.zero_grad(set_to_none=True);losses[0].backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.training.grad_clip)
        assert torch.isfinite(norm).item();optimizer.step();torch.cuda.synchronize()
        duration=torch.tensor(time.time()-start,device=device)
        dist.all_reduce(duration,op=dist.ReduceOp.MAX)
        durations.append(float(duration))
    memory=[None]*world
    dist.all_gather_object(memory,torch.cuda.max_memory_allocated(device)/1024**3)
    from train_lewm_distributed import recalibrate,evaluate
    calibration=[(raw,acts)]*2 if rank==0 else None
    recalibrate(model,calibration,transform,device,2)
    validation=[({'visual':normalized_visual(raw.to(device),transform).cpu()},acts)]*2 if rank==0 else None
    validation_result=evaluate(model,validation,device,2)
    assert all(np.isfinite(v) for v in validation_result.values())
    if rank==0:
        runtime_version=ctypes.c_int()
        assert ctypes.CDLL('libnccl.so.2').ncclGetVersion(ctypes.byref(runtime_version)) == 0
        result={'passed':True,'gpu_count':world,'implementation':'DDP + global SyncBatchNorm + differentiable global SIGReg',
                'torch':torch.__version__,'cuda':torch.version.cuda,'nccl_compiled':list(torch.cuda.nccl.version()),
                'nccl_runtime':runtime_version.value,
                'deterministic_equivalence_test_dropout_disabled':True,'training_dropout_restored':True,
                **metrics,'global_batch':128,'local_batch':128//world,'bf16_forward_backward_finite_all_ranks':True,
                'distributed_bn_recalibration_and_validation_smoke_passed':True,
                'smoke_validation_losses':validation_result,
                'step_seconds':durations,'steady_step_seconds_median':float(np.median(durations[5:])),
                'peak_memory_gib_by_rank':memory,'input':'synthetic pixels/actions; dataset integrity audited separately'}
        (ROOT/'preflight.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2),flush=True)
    dist.destroy_process_group()


if __name__=='__main__':main()
