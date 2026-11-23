"""Check worst-case ECP memory (20 stall memories at a 30-replan budget)."""
import json
import sys
from pathlib import Path
import torch

R=Path('/home/appuser/workspace/project/multimodal-test/JEPA-Stall')
sys.path.insert(0,str(R))
from planning.stagnation import StagnationMPCPlanner
torch.set_num_threads(2)
torch.cuda.set_per_process_memory_fraction(8*2**30/torch.cuda.get_device_properties(0).total_memory)
torch.manual_seed(7)
# Reserve more than persistent model/encoder tensors to make the allocation check conservative.
reserve=torch.empty(2**28,dtype=torch.float32,device='cuda')
p=object.__new__(StagnationMPCPlanner)
p.stag_space='latent';p.stag_alpha=1.;p.stag_base=2.
p._mem={'v':torch.randn(20,196*384,device='cuda'),'p':torch.randn(20,10,device='cuda'),'rho':torch.ones(20,device='cuda'),'stay':torch.ones(20,device='cuda'),'w':torch.ones(20,device='cuda')}
z={'visual':torch.randn(64,6,196,384,device='cuda'),'proprio':torch.randn(64,6,10,device='cuda')}
with torch.no_grad():
    torch.cuda.reset_peak_memory_stats();pen=p._penalty(z);torch.cuda.synchronize()
    assert torch.isfinite(pen).all()
    report={'passed':True,'candidate_batch':64,'horizon_model_steps':5,'memory_entries':20,'reserved_model_allowance_gib':1,'memory_limit_gib':8,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'finite_penalty':True}
    # Check the actual regularizer at a memory point and at a goal outside its support.
    mv=z['visual'][:1,0].flatten(1);mp=z['proprio'][:1,0]
    p._mem={'v':mv,'p':mp,'rho':torch.tensor([.5],device='cuda'),'stay':torch.tensor([2.],device='cuda'),'w':torch.ones(1,device='cuda')}
    stay={'visual':z['visual'][:1,:1].expand(1,6,196,384),'proprio':z['proprio'][:1,:1].expand(1,6,10)}
    goal={k:v+1 for k,v in stay.items()}
    report['stay_penalty']=float(p._penalty(stay)[0]);report['goal_penalty']=float(p._penalty(goal)[0])
    assert report['stay_penalty']>0 and report['goal_penalty']==0
Path(__file__).with_name('memory_validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
