"""Global-batch equivalents for distributed LeWM training."""
import os
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.nn.functional import all_gather
from models.lewm import SIGReg


def initialize():
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl')
    return rank, dist.get_world_size(), torch.device('cuda', rank)


class GlobalSIGReg(SIGReg):
    def forward(self, local):
        # Backward sums the gradients from all identical global regularizers;
        # DDP's subsequent averaging restores the single-global-batch gradient.
        proj = torch.cat(all_gather(local.contiguous()), dim=1)
        if dist.get_rank() == 0:
            projection = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        else:
            projection = torch.empty(proj.size(-1), self.num_proj, device=proj.device)
        dist.broadcast(projection, src=0)
        projection = projection / projection.norm(p=2, dim=0)
        x_t = (proj @ projection).unsqueeze(-1) * self.t
        error = (x_t.cos().mean(-3)-self.phi).square() + x_t.sin().mean(-3).square()
        return ((error @ self.weights) * proj.size(-2)).mean()


class TrainingModel(nn.Module):
    def __init__(self, model, sigreg, weight):
        super().__init__()
        self.world_model = model
        self.sigreg = sigreg
        self.weight = weight

    def forward(self, visual, actions):
        return self.world_model.loss(visual, actions, self.sigreg, self.weight)


def scatter_batch(visual, actions, device, uint8=True):
    rank, world = dist.get_rank(), dist.get_world_size()
    description = [tuple(visual.shape) if rank == 0 else None,
                   tuple(actions.shape) if rank == 0 else None]
    dist.broadcast_object_list(description, src=0, device=device)
    vis_shape, act_shape = description
    assert vis_shape[0] % world == 0 and act_shape[0] == vis_shape[0]
    visual_local = torch.empty((vis_shape[0]//world,)+vis_shape[1:],
                               dtype=torch.uint8 if uint8 else torch.float32, device=device)
    actions_local = torch.empty((act_shape[0]//world,)+act_shape[1:], dtype=torch.float32, device=device)
    if rank == 0:
        visual = visual.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True)
    dist.scatter(visual_local, list(visual.chunk(world)) if rank == 0 else None, src=0)
    dist.scatter(actions_local, list(actions.chunk(world)) if rank == 0 else None, src=0)
    return visual_local, actions_local


def reduce_losses(losses):
    values = torch.stack([value.detach().float() for value in losses])
    dist.all_reduce(values)
    return values / dist.get_world_size()


def normalized_visual(visual, transform):
    batch = visual.size(0)
    visual = visual.permute(0,1,4,2,3).float().div_(255.)
    return transform(visual.flatten(0,1)).unflatten(0,(batch,-1))
