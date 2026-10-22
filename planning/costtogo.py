"""Cost-to-go head for latent planning: V(z, z_goal) ~ model steps from z to z_goal.

V reads the base world model's latents (visual and proprio parts of an encoded or predicted frame),
standardised with the training latents' statistics, and outputs a non-negative number of model steps
(one model step = frameskip env steps). It is trained offline once on the world model's own training
trajectories (scripts/ctg_train.py) and never changes the world model:

    mc  hindsight regression: pairs (t, t + k) of one trajectory, target k
    td  goal-conditioned TD in cost form, V(s, g) <- 1 + V(s', g), with a low expectile so that V
        approaches the fewest steps the data supports (it can stitch trajectories)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CostToGo(nn.Module):
    def __init__(self, dim_v=384, dim_p=10, hidden=512, depth=3):
        super().__init__()
        d = dim_v + dim_p
        self.register_buffer("mean", torch.zeros(d))
        self.register_buffer("std", torch.ones(d))
        layers, w = [], 3 * d
        for _ in range(depth):
            layers += [nn.Linear(w, hidden), nn.LayerNorm(hidden), nn.SiLU()]
            w = hidden
        layers.append(nn.Linear(w, 1))
        self.net = nn.Sequential(*layers)
        self.dim_v, self.dim_p = dim_v, dim_p

    def set_stats(self, mean, std):
        self.mean.copy_(torch.as_tensor(mean, dtype=self.mean.dtype))
        self.std.copy_(torch.as_tensor(std, dtype=self.std.dtype).clamp_min(1e-6))

    def forward(self, zv, zp, gv, gp):
        """zv (..., dim_v), zp (..., dim_p), goal likewise (broadcastable); returns model steps (...)."""
        z = (torch.cat([zv, zp], -1) - self.mean) / self.std
        g = (torch.cat([gv, gp], -1) - self.mean) / self.std
        z, g = torch.broadcast_tensors(z, g)
        return F.softplus(self.net(torch.cat([z, g, z - g], -1))).squeeze(-1)


def load_costtogo(path, device="cpu"):
    ck = torch.load(path, map_location=device)
    m = CostToGo(**ck["config"]).to(device)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m
