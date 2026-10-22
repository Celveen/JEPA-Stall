"""Train the cost-to-go head V(z, z_goal) (planning/costtogo.py) on encoded training trajectories.

Inputs come from scripts/ctg_encode.py. Units are model steps (5 env steps). Two objectives:

    mc  hindsight regression: state s and a later frame g of the same trajectory, target (g - s) / 5
    td  goal-conditioned TD in cost form: target 0 for g = s, (g - s) / 5 for a goal less than one model
        step ahead, otherwise 1 + V_target(s + 5 env steps, g); low expectile (0.1) so V approaches the
        fewest steps the data supports, plus V <= (g - s) / 5 for same-trajectory goals (the data's own
        path is an upper bound). Goals: 20% s itself, 50% later frames of the trajectory, 30% random frames.

A share of the states (pred_frac) is replaced by the world model's own predictions of that frame, the
input V sees during planning. Validation (every eval_every steps) on another split: Spearman between V
and the true step count for same-trajectory pairs, on encoded and on predicted states.

    python scripts/ctg_train.py --objective td --out eval_outputs/ctg/V_td.pt --device cuda:0
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from planning.costtogo import CostToGo  # noqa: E402

FS = 5


class Split:
    """Encoded trajectories of one split on a device."""

    def __init__(self, path, device):
        p = ROOT / path
        self.vis = torch.from_numpy(np.load(p / "vis.npy")).to(device)
        self.prop = torch.from_numpy(np.load(p / "prop.npy")).to(device)
        self.start = torch.from_numpy(np.load(p / "ep_start.npy")).long().to(device)
        self.len = torch.from_numpy(np.load(p / "ep_len.npy")).long().to(device)
        self.pvis = torch.from_numpy(np.load(p / "pred_vis.npy")).to(device)
        self.pprop = torch.from_numpy(np.load(p / "pred_prop.npy")).to(device)
        self.pidx = torch.from_numpy(np.load(p / "pred_idx.npy")).long().to(device)
        self.n = len(self.vis)
        self.shapes = json.loads((p / "shapes.json").read_text())
        self.ep_of = torch.repeat_interleave(torch.arange(len(self.start), device=device), self.len)

    def z(self, idx):
        return self.vis[idx].float(), self.prop[idx].float()

    def sample_states(self, n, pred_frac, gen, need_next=False):
        """Frame indices s and their input latents (encoded, or the world model's prediction of s)."""
        dev = self.vis.device
        n_pred = int(n * pred_frac)
        s = torch.randint(0, self.n, (n - n_pred,), device=dev, generator=gen)
        j = torch.randint(0, len(self.pidx), (n_pred,), device=dev, generator=gen)
        s = torch.cat([s, self.pidx[j]])
        zv, zp = self.z(s)
        zv[n - n_pred:], zp[n - n_pred:] = self.pvis[j].float(), self.pprop[j].float()
        if need_next:  # keep states with a frame one model step later in the same trajectory
            e = self.ep_of[s]
            ok = s + FS < self.start[e] + self.len[e]
            s, zv, zp = s[ok], zv[ok], zp[ok]
        return s, zv, zp

    def future(self, s, gen):
        """A uniformly drawn frame of the same trajectory at or after s."""
        e = self.ep_of[s]
        end = self.start[e] + self.len[e]
        u = torch.rand(len(s), device=s.device, generator=gen)
        return s + (u * (end - s).float()).long().clamp(max=(end - s - 1))


@torch.no_grad()
def spearman(a, b):
    ra = a.argsort().argsort().float()
    rb = b.argsort().argsort().float()
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra * rb).sum() / (ra.norm() * rb.norm() + 1e-9))


@torch.no_grad()
def validate(model, val, gen, n=20000):
    out = {}
    for name, frac in (("enc", 0.0), ("pred", 1.0)):
        s, zv, zp = val.sample_states(n, frac, gen)
        g = val.future(s, gen)
        k = (g - s).float() / FS
        gv, gp = val.z(g)
        v = model(zv, zp, gv, gp)
        out[name] = {"spearman": round(spearman(v, k), 4), "mae": round(float((v - k).abs().mean()), 3),
                     "bias": round(float((v - k).mean()), 3)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="eval_outputs/ctg/bias16k_train")
    ap.add_argument("--val", default="eval_outputs/ctg/bias16k_val")
    ap.add_argument("--objective", choices=("mc", "td"), required=True)
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--pred-frac", type=float, default=0.3)
    ap.add_argument("--max-steps", type=float, default=60.0)
    ap.add_argument("--expectile", type=float, default=0.1)
    ap.add_argument("--tau", type=float, default=0.005, help="target network update rate (td)")
    ap.add_argument("--eval-every", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    dev = torch.device(a.device)
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    tr, va = Split(a.train, dev), Split(a.val, dev)
    model = CostToGo().to(dev)
    sub = torch.randint(0, tr.n, (200000,), device=dev, generator=gen)
    allz = torch.cat(tr.z(sub), -1)
    model.set_stats(allz.mean(0), allz.std(0))
    target = CostToGo().to(dev)
    target.load_state_dict(model.state_dict())
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    log, t0 = [], time.time()
    for it in range(1, a.steps + 1):
        if a.objective == "mc":
            s, zv, zp = tr.sample_states(a.batch, a.pred_frac, gen)
            g = tr.future(s, gen)
            y = ((g - s).float() / FS).clamp(max=a.max_steps)
            gv, gp = tr.z(g)
            loss = F.huber_loss(model(zv, zp, gv, gp), y)
        else:
            s, zv, zp = tr.sample_states(a.batch, a.pred_frac, gen, need_next=True)
            n = len(s)
            u = torch.rand(n, device=dev, generator=gen)
            g = torch.where(u < 0.2, s, tr.future(s, gen))
            rnd = u >= 0.7
            g[rnd] = torch.randint(0, tr.n, (int(rnd.sum()),), device=dev, generator=gen)
            same = tr.ep_of[g] == tr.ep_of[s]
            ahead = (g - s).float() / FS
            gv, gp = tr.z(g)
            with torch.no_grad():
                nv, np_ = tr.z(s + FS)
                boot = 1 + target(nv, np_, gv, gp)
                y = torch.where(same & (ahead >= 0) & (ahead < 1), ahead, boot).clamp(max=a.max_steps)
            v = model(zv, zp, gv, gp)
            d = y - v
            w = torch.where(d < 0, 1 - a.expectile, a.expectile)
            loss = (w * d.pow(2)).mean()
            upper = same & (ahead >= 0)  # the data's own path bounds the fewest steps from above
            loss = loss + F.relu(v[upper] - ahead[upper]).pow(2).mean()
            with torch.no_grad():
                for pt, pm in zip(target.parameters(), model.parameters()):
                    pt.lerp_(pm, a.tau)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if it % a.eval_every == 0 or it == a.steps:
            model.eval()
            vgen = torch.Generator(device=dev).manual_seed(123)
            rec = {"step": it, "loss": round(float(loss), 4), "val": validate(model, va, vgen),
                   "minutes": round((time.time() - t0) / 60, 1)}
            model.train()
            log.append(rec)
            print(json.dumps(rec), flush=True)
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": {"dim_v": model.dim_v, "dim_p": model.dim_p}, "state_dict": model.state_dict(),
                "args": vars(a), "log": log}, out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
