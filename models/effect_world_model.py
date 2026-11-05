"""Training-only controls for PushObj B; rollout/native planning is inherited.

Action embeddings are removed with separate_emb before BOTH auxiliaries. The
original forward path is returned unchanged for B0 or inference without labels.
"""
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from models.visual_world_model import VWorldModel


def make_idm_head(in_dim, action_dim, hidden=256, seed=1701):
    # Additional parameters must not advance the shared backbone/dropout RNG.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        return nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, action_dim))


def distributed_mean_denominator(total, count):
    """Global mean/count, detached: DDP will average the local gradients."""
    values = torch.stack([total.detach().float(), count.detach().float()])
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(values)
    return values


def masked_mean(values, mask):
    mask = mask.to(values.dtype)
    count = mask.sum()
    if dist.is_available() and dist.is_initialized():
        denominator = count.detach().clone()
        dist.all_reduce(denominator)
        return (values*mask).sum() * dist.get_world_size() / denominator.clamp_min(1)
    return (values*mask).sum() / count.clamp_min(1)


def effect_distances(z_obs):
    out = []
    for branch in ("visual", "proprio"):
        z = z_obs[branch].float().flatten(2)
        z = F.layer_norm(z, (z.shape[-1],), eps=1e-5)
        differences = torch.stack([z[:, 1]-z[:, 0], z[:, 2]-z[:, 1],
                                   z[:, 3]-z[:, 2], z[:, 3]-z[:, 0]], dim=1)
        out.append(differences.square().mean(-1))
    return torch.stack(out, -1)  # B, pairs(01,12,23,03), branch(V,P)


class EffectWorldModel(VWorldModel):
    def __init__(self, *args, effect_mode="none", effect_lambda=0.0,
                 effect_margin=0.2, idm_hidden=256, idm_action_dim=10, **kwargs):
        super().__init__(*args, **kwargs)
        if effect_mode not in ("none", "reweight", "idm", "effect"):
            raise ValueError(effect_mode)
        self.effect_mode = effect_mode
        self.effect_lambda = float(effect_lambda)
        self.effect_margin = float(effect_margin)
        self._collect_effect = False
        self._last_effect_z = None
        self.idm_head = None
        if effect_mode == "idm":
            encoder = getattr(self.encoder, "module", self.encoder)
            self.idm_head = make_idm_head(encoder.emb_dim + self.proprio_dim // self.num_proprio_repeat,
                                          idm_action_dim, idm_hidden)

    def encode(self, obs, act):
        z = super().encode(obs, act)
        if self._collect_effect:
            self._last_effect_z = z
        return z

    def forward(self, obs, act):
        enabled = self.effect_mode != "none" and "effect_labels" in obs
        if self.training and self.effect_mode != "none" and not enabled:
            raise ValueError("B auxiliary training requires the registered observation cache")
        if not enabled:
            return super().forward(obs, act)
        self._collect_effect = True
        try:
            z_pred, vp, vr, loss, components = super().forward(obs, act)
            z = self._last_effect_z
        finally:
            self._collect_effect = False
            self._last_effect_z = None
        if z.shape[1] != 4 or self.num_hist != 3 or self.num_pred != 1:
            raise ValueError("Registered B controls require a four-frame training window")
        labels = obs["effect_labels"].bool()
        if labels.shape != (z.shape[0], 4, 2):
            raise ValueError("Expected effect labels [B,4,2]")
        components["z_loss_unweighted"] = components["z_loss"]
        if self.effect_mode == "reweight":
            target = z[:, 1:].detach() if self.stop_grad else z[:, 1:]
            per_pair = (self.visual_prop(z_pred).float() - self.visual_prop(target).float()).square().flatten(2).mean(-1)
            weights = 1.0 + labels[:, :3].any(-1).float()
            total, count = distributed_mean_denominator(weights.sum(), weights.new_tensor(weights.numel()))
            weights = weights / (total / count)
            weighted = (per_pair * weights).mean()
            loss = loss - components["z_loss"] + weighted
            components["z_loss"] = weighted
            components["effect_pair_coverage"] = labels[:, :3].any(-1).float().mean()
        else:
            z_obs, _ = self.separate_emb(z)  # NO action token/dimensions
            if self.effect_mode == "idm":
                visual = z_obs["visual"].float().mean(dim=2)
                proprio = z_obs["proprio"].float()
                delta = torch.cat([visual[:, 1:]-visual[:, :-1], proprio[:, 1:]-proprio[:, :-1]], -1)
                action_hat = self.idm_head(delta)
                aux = F.mse_loss(action_hat.float(), act[:, :3].float())
            else:
                distances = effect_distances(z_obs)
                aux = masked_mean(F.relu(self.effect_margin - distances).square(), labels)
                components["effect_active_fraction"] = masked_mean((distances < self.effect_margin).float(), labels)
                components["effect_distance_visual"] = masked_mean(distances[..., 0], labels[..., 0])
                components["effect_distance_proprio"] = masked_mean(distances[..., 1], labels[..., 1])
            components["aux_loss"] = aux
            components["aux_loss_scaled"] = aux * self.effect_lambda
            loss = loss + aux * self.effect_lambda
        components["loss"] = loss
        return z_pred, vp, vr, loss, components
