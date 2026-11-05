"""100 fixed training minibatches; no optimizer update or validation tuning."""
import os
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[k] = "1"
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
import numpy as np
import torch
from hydra import initialize_config_dir, compose
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import custom_resolvers
from scripts.effect_training_utils import build_model, module_hash, CalibrationWindows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, default=100)
    ap.add_argument("--microbatch", type=int, default=16)
    ap.add_argument("--out", default="reports/effect_b_20261103/calibration")
    args = ap.parse_args()
    out = ROOT/args.out
    out.mkdir(parents=True, exist_ok=True)
    if (out/"complete.json").exists():
        raise RuntimeError("Calibration already frozen; do not overwrite")
    with initialize_config_dir(config_dir=str(ROOT/"conf"), version_base=None):
        cfg = compose(config_name="train_effect_b0")
    data = CalibrationWindows(count=args.batches*64)
    ids_json = json.dumps(data.ids)
    (out/"train_window_ids.json").write_text(ids_json+"\n")
    results = {}
    started = time.monotonic()
    for mode in ("idm", "effect"):
        wm = build_model(cfg, mode).to("cuda:0")
        wm.train()
        names = ("encoder", "predictor", "action_encoder", "proprio_encoder")
        hashes = {k: module_hash(getattr(wm,k)) for k in names}
        if (out/"initialization.json").exists():
            assert hashes == json.loads((out/"initialization.json").read_text())["modules"]
        else:
            (out/"initialization.json").write_text(json.dumps({"modules": hashes}, indent=2)+"\n")
        params = [p for p in wm.encoder.parameters() if p.requires_grad]
        rng_before = {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
                      "numpy": np.random.get_state(), "python": random.getstate()}
        loader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=False, num_workers=4,
            pin_memory=True, generator=torch.Generator().manual_seed(101))
        records = []
        for i, (obs, act) in enumerate(loader):
            gbase = [torch.zeros_like(p, dtype=torch.float32) for p in params]
            gaux = [torch.zeros_like(p, dtype=torch.float32) for p in params]
            aux_values, base_values, active = [], [], []
            full_positive_count = float(obs["effect_labels"].sum())
            for start in range(0, 64, args.microbatch):
                end = min(64, start+args.microbatch)
                ob = {k:v[start:end].cuda(non_blocking=True) for k,v in obs.items()}
                ac = act[start:end].cuda(non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    _, _, _, _, lc = wm(ob, ac)
                gb = torch.autograd.grad(lc["z_loss_unweighted"], params, retain_graph=True)
                ga = torch.autograd.grad(lc["aux_loss"], params)
                scale_base = (end-start)/64
                scale_aux = (float(ob["effect_labels"].sum())/max(full_positive_count,1)
                             if mode == "effect" else scale_base)
                for dst, src in zip(gbase, gb): dst.add_(src.float(), alpha=scale_base)
                for dst, src in zip(gaux, ga): dst.add_(src.float(), alpha=scale_aux)
                base_values.append(float(lc["z_loss_unweighted"].detach())*scale_base)
                aux_values.append(float(lc["aux_loss"].detach())*scale_aux)
                if mode == "effect": active.append(float(lc["effect_active_fraction"].detach())*scale_aux)
                del gb, ga, lc, ob, ac
            bn = float(torch.sqrt(sum(g.square().sum() for g in gbase)))
            an = float(torch.sqrt(sum(g.square().sum() for g in gaux)))
            records.append({"batch": i, "jepa_encoder_grad_norm": bn, "aux_encoder_grad_norm": an,
                "jepa_loss": sum(base_values), "aux_loss": sum(aux_values),
                "margin_active_fraction": sum(active) if active else None})
            if (i+1)%10==0:
                print(f"{mode}: {i+1}/{args.batches} batches; JEPA grad={bn:.6g}, aux grad={an:.6g}", flush=True)
            del gbase, gaux
        base_norm = np.mean([r["jepa_encoder_grad_norm"] for r in records])
        aux_norm = np.mean([r["aux_encoder_grad_norm"] for r in records])
        if not np.isfinite(base_norm+aux_norm) or aux_norm < 1e-8 or base_norm < 1e-8:
            raise RuntimeError(f"Invalid gradient calibration for {mode}: {base_norm}, {aux_norm}")
        lam = float(.1*base_norm/aux_norm)
        assert hashes == {k: module_hash(getattr(wm,k)) for k in names}, "Calibration changed model state"
        torch.set_rng_state(rng_before["torch"])
        torch.cuda.set_rng_state_all(rng_before["cuda"])
        np.random.set_state(rng_before["numpy"])
        random.setstate(rng_before["python"])
        torch.save(rng_before, out/f"{mode}_rng_restored.pth")
        info = {"mode":mode, "lambda":lam, "batches":args.batches, "batch_size":64,
            "microbatch":args.microbatch, "gradient_parameters":"visual encoder only",
            "mean_jepa_encoder_grad_norm":float(base_norm), "mean_aux_encoder_grad_norm":float(aux_norm),
            "target_gradient_ratio":.1, "state_unchanged":True, "rng_restored":True,
            "train_windows_sha256":hashlib.sha256(ids_json.encode()).hexdigest(),
            "unique_source_windows":len(set(data.ids)), "window_draws":len(data.ids),
            "aux_parameters":sum(p.numel() for p in wm.idm_head.parameters()) if wm.idm_head else 0,
            "idm_head_linear_forward_MACs_per_pair":394*256+256*10 if mode=="idm" else 0,
            "records":records}
        (out/f"{mode}.json").write_text(json.dumps(info,indent=2)+"\n")
        results[mode] = {k:v for k,v in info.items() if k!="records"}
        print(f"{mode} fixed lambda={lam:.8g}", flush=True)
        del wm
        torch.cuda.empty_cache()
    results["wall_seconds"] = time.monotonic()-started
    (out/"complete.json").write_text(json.dumps(results,indent=2)+"\n")


if __name__ == "__main__": main()
