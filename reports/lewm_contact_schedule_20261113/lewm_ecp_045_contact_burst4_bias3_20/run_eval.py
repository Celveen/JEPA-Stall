"""Evaluate four consecutive contact-guidance replans per trigger burst."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from omegaconf import OmegaConf
import torch

ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT.with_name("lewm_ecp_045_contact_burst4")
SHAPES = ("I", "L", "square", "Z", "small_tee", "plus", "T")
CONTACT = {
    "position_tol_px": 20,
    "angle_tol_deg": 20,
    "staging_clearance_px": 6,
    "workspace_px": [22, 489],
}

sys.path.insert(0, str(ROOT / "source"))
import plan


def model_digest(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", choices=SHAPES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--replans", type=int, default=30)
    args = parser.parse_args()
    output = args.output.resolve()

    cfg = OmegaConf.load(REFERENCE / "full" / args.shape / "config.yaml")
    cfg.n_evals = args.limit
    cfg.planner.max_iter = args.replans
    cfg.saved_folder = str(output)
    cfg.planner._target_ = "planning.lewm_contact_burst.LeWMContactBurstMPCPlanner"
    cfg.planner.contact = {**CONTACT, "shape": args.shape}
    cfg.planner.burst = {"replans": 4}
    cfg.planner.bias.cap_px = 20.0
    cfg.planner.bias.floor_px = 3.0

    output.mkdir(parents=True)
    OmegaConf.save(cfg, output / "config.yaml")
    original_load = plan.load_model
    loaded = {}

    def load_frozen(*args, **kwargs):
        model = original_load(*args, **kwargs)
        model.requires_grad_(False).eval()
        loaded.update(model=model, digest=model_digest(model))
        return model

    plan.load_model = load_frozen
    torch.set_num_threads(2)
    os.chdir(output)
    start = time.monotonic()
    plan.planning_main(OmegaConf.to_container(cfg, resolve=True))
    if model_digest(loaded["model"]) != loaded["digest"]:
        raise RuntimeError("LeWM weights or buffers changed during evaluation")
    with np.load("final_eval_results.npz") as results:
        summary = {
            "condition": "stagnation_regularizer_with_contact_burst4_bias3_20",
            "successes": int(results["successes"].sum()),
            "tasks": args.limit,
            "model_state_sha256": loaded["digest"],
            "model_unchanged": True,
            "wall_seconds": time.monotonic() - start,
        }
    Path("summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
