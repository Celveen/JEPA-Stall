import importlib.util,json,sys,torch
from pathlib import Path
from native_models import vit
# Original official implementation, only remove its hard-coded CUDA allocation for this CPU check.
s=(Path(__file__).parent/'official/vit.py').read_text().replace(".to('cuda')",'')
ns={};exec(compile(s,'official_vit.py','exec'),ns)
torch.manual_seed(9)
a=ns['ViTPredictor'](num_patches=4,num_frames=3,dim=20,depth=2,heads=2,mlp_dim=32,dropout=0).eval()
b=vit.ViTPredictor(num_patches=4,num_frames=3,dim=20,depth=2,heads=2,mlp_dim=32,dropout=0).eval();b.load_state_dict(a.state_dict())
x=torch.randn(2,12,20,requires_grad=True);y=x.detach().clone().requires_grad_(True)
o=a(x);p=b(y);o.square().mean().backward();p.square().mean().backward()
f=float((o-p).abs().max());g=float((x.grad-y.grad).abs().max());torch.testing.assert_close(o,p,atol=1e-5,rtol=1e-5);torch.testing.assert_close(x.grad,y.grad,atol=1e-5,rtol=1e-5)
r={'forward_max_abs':f,'input_gradient_max_abs':g,'mask':'full within-frame attention, causal across frames','status':'passed'}
(Path(__file__).parent/'attention_equivalence.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
