"""Shared initialization and fixed TRAIN-only calibration observations."""
import hashlib
import json
import random
from pathlib import Path
import hydra
import numpy as np
import torch
from decord import VideoReader
import decord
from torch.utils.data import Dataset
from datasets.img_transforms import default_transform
from datasets.pusht_dset import ACTION_MEAN, ACTION_STD, PROPRIO_MEAN, PROPRIO_STD
from utils import seed

ROOT = Path(__file__).resolve().parents[1]


def module_hash(module):
    h = hashlib.sha256()
    for k, v in getattr(module, "module", module).state_dict().items():
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def build_model(cfg, mode):
    # Same order and dimensions as Trainer.init_models. Dataset loading/slicing
    # consumes NumPy RNG only; it does not advance PyTorch initialization RNG.
    seed(cfg.training.seed)
    encoder = hydra.utils.instantiate(cfg.encoder)
    proprio = hydra.utils.instantiate(cfg.proprio_encoder, in_chans=4, emb_dim=cfg.proprio_emb_dim)
    action = hydra.utils.instantiate(cfg.action_encoder, in_chans=10, emb_dim=cfg.action_emb_dim)
    predictor = hydra.utils.instantiate(cfg.predictor, num_patches=1, num_frames=cfg.num_hist,
        dim=encoder.emb_dim + proprio.emb_dim*cfg.num_proprio_repeat + action.emb_dim*cfg.num_action_repeat)
    model = hydra.utils.instantiate(cfg.model, encoder=encoder, proprio_encoder=proprio,
        action_encoder=action, predictor=predictor, decoder=None, proprio_dim=proprio.emb_dim,
        action_dim=action.emb_dim, concat_dim=cfg.concat_dim, num_action_repeat=cfg.num_action_repeat,
        num_proprio_repeat=cfg.num_proprio_repeat, straighten=cfg.training.straighten,
        stop_grad=cfg.training.stop_grad, effect_mode=mode, effect_lambda=0.0)
    return model


class CalibrationWindows(Dataset):
    def __init__(self, count=6400):
        self.records = [json.loads(l) for l in (ROOT/"reports/effect_ab_20261103/train_observer.jsonl").open()]
        rng = np.random.default_rng(0)
        order = []
        while len(order) < count:
            order.extend(rng.permutation(len(self.records)).tolist())
        self.order = order[:count]
        data = ROOT/"data/pushobj_bias_16k/train"
        xy = torch.load(data/"states.pth", weights_only=True)[..., :2].float().clone()
        vel = torch.load(data/"velocities.pth", weights_only=True).float()
        self.proprio = (torch.cat([xy, vel], -1)-PROPRIO_MEAN)/PROPRIO_STD
        self.actions = (torch.load(data/"rel_actions.pth", weights_only=True).float()/100-ACTION_MEAN)/ACTION_STD
        self.data = data
        self.transform = default_transform()
        self.ids = [self.records[i]["sample"]["id"] for i in self.order]

    def __len__(self):
        return len(self.order)

    def __getitem__(self, idx):
        decord.bridge.set_bridge("torch")
        r = self.records[self.order[idx]]
        e, s = r["sample"]["episode"], r["sample"]["start"]
        indices = [s, s+5, s+10, s+15]
        image = VideoReader(str(self.data/"obses"/f"episode_{e:03d}.mp4"), num_threads=1).get_batch(indices)
        image = self.transform(image.permute(0, 3, 1, 2).float()/255)
        labels = torch.tensor([[p["labels"][b] == 1 for b in ("visual", "proprio")] for p in r["pairs"]])
        obs = {"visual": image, "proprio": self.proprio[e, indices], "effect_labels": labels}
        return obs, self.actions[e, s:s+20].reshape(4, 10)
