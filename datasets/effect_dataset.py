"""Positive observation evidence cache for the approved PushObj B experiment."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from datasets.pusht_dset import load_pusht_slice_train_val


class EffectSlices(Dataset):
    def __init__(self, base, cache_path):
        self.base = base
        self.cache_path = str(cache_path)
        self.labels = np.load(cache_path, mmap_mode="r")
        if self.labels.dtype != np.bool_ or self.labels.shape[-2:] != (2, 2):
            raise ValueError("Expected bool cache [episode,time,horizon5/15,visual/proprio]")
        for key in ("slices", "proprio_dim", "action_dim", "state_dim"):
            setattr(self, key, getattr(base, key))
        if base.num_frames != 4 or base.frameskip != 5:
            raise ValueError("Effect cache is registered for four frames, frameskip five")
        if self.labels.shape[0] != len(base.dataset):
            raise ValueError("Cache/dataset episode count mismatch")

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        obs, act, state = self.base[idx]
        ep, start, _ = self.base.slices[idx]
        labels = np.stack([self.labels[ep, start, 0], self.labels[ep, start+5, 0],
                           self.labels[ep, start+10, 0], self.labels[ep, start, 1]])
        obs["effect_labels"] = torch.from_numpy(labels)
        return obs, act, state


def load_effect_pusht(effect_cache, **kwargs):
    cache = Path(effect_cache)
    meta = json.loads((cache / "complete.json").read_text())
    if Path(meta["data_path"]).resolve() != Path(kwargs["data_path"]).resolve():
        raise ValueError("Cache provenance does not match training data")
    datasets, trajectories = load_pusht_slice_train_val(**kwargs)
    for part, split in (("train", "train"), ("valid", "val")):
        datasets[part] = EffectSlices(datasets[part], cache / f"{split}.npy")
    return datasets, trajectories
