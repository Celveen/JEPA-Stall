"""Bounded inference profiling on saved observations; no simulation or training."""
import contextlib
import json
import pickle
import statistics
import sys
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

ROOT = Path('/home/appuser/workspace/project/multimodal-test/JEPA-Stall')
SRC = ROOT / 'reports/dinowm_pushobj_v1'
OUT = Path('/tmp/dinowm_perf_diagnosis')
OUT.mkdir(exist_ok=True)
sys.path[:0] = [str(SRC), str(ROOT)]
from plan import load_model
from preprocessor import Preprocessor
from datasets.pusht_dset import ACTION_MEAN, ACTION_STD, STATE_MEAN, STATE_STD, PROPRIO_MEAN, PROPRIO_STD
from datasets.img_transforms import default_transform
from cem_memory import rollout_cached
from planning.objectives import create_objective_fn

torch.set_num_threads(2)
torch.manual_seed(1)
torch.cuda.set_per_process_memory_fraction(0.6)
ckpt = ROOT / 'checkpoints/dinowm_pushobj_v1/ordinary'
wm = load_model(ckpt / 'checkpoints/model_3.pth', OmegaConf.load(ckpt / 'hydra.yaml'), 1, 'cuda')
wm.eval()
for p in wm.parameters():
    p.requires_grad_(False)
raw = pickle.loads((ROOT / 'eval_outputs/dinowm_pushobj_3epoch_v1/ordinary_native/chunk_00_I/plan_targets.pkl').read_bytes())
prep = Preprocessor(ACTION_MEAN, ACTION_STD, STATE_MEAN[:5], STATE_STD[:5], PROPRIO_MEAN[:4], PROPRIO_STD[:4], default_transform(224))
obs = {k: v.cuda() for k, v in prep.transform_obs({k: v[:1] for k, v in raw['obs_0'].items()}).items()}
goal = {k: v.cuda() for k, v in prep.transform_obs({k: v[:1] for k, v in raw['obs_g'].items()}).items()}
objective = create_objective_fn(alpha=1, base=2, mode='last')
report = {'kind': 'fixed_observation_inference_profile_not_task_evaluation', 'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(), 'default_matmul_tf32': torch.backends.cuda.matmul.allow_tf32, 'default_cudnn_tf32': torch.backends.cudnn.allow_tf32, 'parameter_dtypes': sorted({str(p.dtype) for p in wm.parameters()}), 'horizon_model_steps': 5, 'candidates': 300, 'results': []}
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
with torch.no_grad():
    z0, zg = wm.encode_obs(obs), wm.encode_obs(goal)
    actions = torch.randn(300, 5, 10).cuda()
    actions[0] = 0
    def run(batch, precision):
        costs = []
        context = torch.autocast('cuda', dtype=torch.bfloat16) if precision == 'bf16' else contextlib.nullcontext()
        with context:
            for start in range(0, len(actions), batch):
                act = actions[start:start + batch]
                n = len(act)
                zz = {k: v.expand(n, *v.shape[1:]) for k, v in z0.items()}
                gg = {k: v.expand(n, *v.shape[1:]) for k, v in zg.items()}
                pred, _ = rollout_cached(wm, zz, act)
                costs.append(objective(pred, gg))
        return torch.cat(costs)
    reference = None
    for precision, batch in [('fp32', 16), ('fp32', 64), ('fp32', 300), ('tf32', 16), ('tf32', 64), ('tf32', 300), ('bf16', 16), ('bf16', 64), ('bf16', 300)]:
        torch.backends.cuda.matmul.allow_tf32 = precision == 'tf32'
        torch.backends.cudnn.allow_tf32 = precision == 'tf32'
        try:
            run(batch, precision)
            torch.cuda.synchronize()
            times = []
            torch.cuda.reset_peak_memory_stats()
            for _ in range(2):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                costs = run(batch, precision)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)
            if reference is None:
                reference = costs.clone()
            ref_elites = torch.argsort(reference)[:30]
            elites = torch.argsort(costs)[:30]
            row = {'precision': precision, 'batch': batch, 'seconds': times, 'median_s': statistics.median(times), 'max_allocated_gib': torch.cuda.max_memory_allocated() / 2**30, 'cost_max_abs_error': float((reference - costs).abs().max()), 'cost_mean_abs_error': float((reference - costs).abs().mean()), 'elite_overlap_of_30': len(set(ref_elites.tolist()) & set(elites.tolist())), 'best_candidate_same': int(ref_elites[0]) == int(elites[0])}
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            row = {'precision': precision, 'batch': batch, 'error': 'CUDA OOM under 60% memory cap'}
        report['results'].append(row)
        (OUT / 'profile.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(row), flush=True)
    # Profile one baseline rollout to identify actual attention and GEMM kernels.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as prof:
        zz = {k: v.expand(16, *v.shape[1:]) for k, v in z0.items()}
        rollout_cached(wm, zz, actions[:16])
        torch.cuda.synchronize()
    table = prof.key_averages().table(sort_by='self_cuda_time_total', row_limit=18)
    (OUT / 'operators.txt').write_text(table)
    print(table, flush=True)
