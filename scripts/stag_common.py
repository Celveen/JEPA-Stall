"""Shared helpers for the stall-regulariser scripts (calibration, oracle diagnostic, analysis).

Builds the world model, preprocessor and objective exactly as plan.py does for goal_source=segments,
reads stream targets and reference-run logs, and renders PushObj states.
"""
import collections
import json
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

ACTION_SCALE = 100.0  # env action x 100 = pixel offset of the pusher's target


def load_world_model(ckpt="checkpoints/pushobj_bias_16k_e3_seed0", epoch="3", device="cuda:0"):
    """(wm, preprocessor) as plan.py builds them for segment targets; wm in eval mode, frozen."""
    import hydra
    from omegaconf import OmegaConf
    from types import SimpleNamespace
    from datasets.img_transforms import default_transform
    from datasets.pusht_dset import ACTION_MEAN, ACTION_STD, PROPRIO_MEAN, PROPRIO_STD, STATE_MEAN, STATE_STD
    from plan import load_model
    from preprocessor import Preprocessor

    model_path = (ROOT / ckpt).resolve()
    cfg = OmegaConf.load(model_path / "hydra.yaml")
    tcfg = cfg.env.dataset.get("transform", None)
    transform = hydra.utils.instantiate(tcfg) if tcfg is not None else default_transform()
    dset = SimpleNamespace(action_mean=ACTION_MEAN, action_std=ACTION_STD, state_mean=STATE_MEAN[:5],
                           state_std=STATE_STD[:5], proprio_mean=PROPRIO_MEAN[:4], proprio_std=PROPRIO_STD[:4],
                           transform=transform)
    wm = load_model(model_path / "checkpoints" / f"model_{epoch}.pth", cfg, cfg.num_action_repeat, torch.device(device))
    wm.eval()
    for p in wm.parameters():
        p.requires_grad = False
    pre = Preprocessor(action_mean=dset.action_mean, action_std=dset.action_std, state_mean=dset.state_mean,
                       state_std=dset.state_std, proprio_mean=dset.proprio_mean, proprio_std=dset.proprio_std,
                       transform=dset.transform)
    return wm, pre, cfg


@torch.no_grad()
def encode(wm, pre, visual, proprio, device="cuda:0", batch=256):
    """Encode frames (N, H, W, C) uint8 and proprio (N, 4); returns visual (N, P*D) and proprio (N, Dp)."""
    from utils import move_to_device
    vs, ps = [], []
    for i in range(0, len(visual), batch):
        obs = {"visual": visual[i : i + batch, None], "proprio": proprio[i : i + batch, None].astype(np.float32)}
        z = wm.encode_obs(move_to_device(pre.transform_obs(obs), torch.device(device)))
        vs.append(z["visual"][:, 0].flatten(1).float().cpu())
        ps.append(z["proprio"][:, 0].flatten(1).float().cpu())
    return torch.cat(vs), torch.cat(ps)


def stream_targets(stream, chunk):
    """Segments of one chunk: list of dicts with init state (7,), actions (T, 2) in pixels, shape."""
    d = pickle.load(open(ROOT / "data/targets" / stream / f"{chunk}.pkl", "rb"))
    out = []
    for s in d["segments"]:
        st = np.zeros(7, dtype=np.float64)
        st[: len(s["states"][0])] = s["states"][0]
        out.append({"init": st, "actions": np.asarray(s["actions"], dtype=np.float64), "shape": s.get("shape", d["shape"])})
    return out


_ENVS = {}


def make_env(shape):
    """One PushObj env per shape (as plan.py builds it: with_velocity=True), reused across calls."""
    if shape not in _ENVS:
        from env.pusht.pusht_wrapper import PushTWrapper
        env = PushTWrapper(with_velocity=True)
        env.update_env({"shape": shape})
        _ENVS[shape] = env
    return _ENVS[shape]


def render(shape, state, seed=0):
    """Observation (visual (H, W, C) uint8, proprio (4,)) and the settled state at a 7-dim state."""
    obs, st = make_env(shape).prepare(seed, np.asarray(state, dtype=np.float64))
    return obs, np.asarray(st, dtype=np.float64)


def goal_state(seg, seed=0):
    """Final state (and observation) of a target segment replayed from its initial state, as plan.py does."""
    env = make_env(seg["shape"])
    obses, states = env.rollout(seed, seg["init"], seg["actions"] / ACTION_SCALE)
    return states[-1].astype(np.float64), {k: v[-1] for k, v in obses.items()}


def ref_episodes(ref, max_iter=30):
    """{(chunk, sample_idx): records sorted by step} of a reference planning run, truncated at max_iter."""
    eps = {}
    for ch in sorted(p for p in (ROOT / ref).iterdir() if (p / "per_iter_logs.json").exists()):
        by = collections.defaultdict(list)
        for line in (ch / "per_iter_logs.json").open():
            if line.strip():
                r = json.loads(line)
                if r["step"] <= max_iter:
                    by[r["sample_idx"]].append(r)
        for i, rs in by.items():
            eps[(ch.name, i)] = sorted(rs, key=lambda r: r["step"])
    return eps
