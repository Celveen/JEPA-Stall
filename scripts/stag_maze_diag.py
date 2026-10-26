"""Offline diagnosis of maze stalls: does the world model believe the agent can move where it is stuck?

Reads the reference run of the frozen per-sample planner (scripts/stag_maze_base_run.py). Each replan's start state
is the logged state (position and velocity) and its executed action the logged one, so every replan can be
re-rendered and re-run in the simulator. On stalled replans (the movement trigger's ratio < gamma_hi, the rule the
planner uses) and, as a control, on moving replans:

  1-step   the model's prediction for the action actually executed vs where the agent actually went
  replan   a fresh GD plan from that observation (the planner's horizon, objective and optimiser, zero
           initialisation): the positions the model predicts along it vs the positions the simulator reaches with
           the same actions, and whether the predicted path crosses a wall

Predicted positions are read off the predicted proprio embedding by inverting the proprio encoder (a 1x1 conv and
a LayerNorm on the normalized position and velocity) with least squares from several starts; the same read-out on
the encoded true next state gives the read-out error, reported alongside. Positions are in maze cells
(x = row, y = column); success needs < 0.5 from the goal.

    LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia python scripts/stag_maze_diag.py --level easy
"""
import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

LEVELS = {"easy": (3, 5), "medium": (6, 9), "hard": (10, 14)}


def build(ckpt, data, device):
    """(wm, preprocessor, cfg) as plan.py builds them for the maze (normalization stats from the dataset)."""
    import hydra
    from omegaconf import OmegaConf
    from plan import load_model
    from preprocessor import Preprocessor

    cfg = OmegaConf.load(ROOT / ckpt / "hydra.yaml")
    cfg.env.dataset.data_path = str((ROOT / data).resolve())
    _, dset = hydra.utils.call(cfg.env.dataset, num_hist=cfg.num_hist, num_pred=cfg.num_pred, frameskip=cfg.frameskip)
    dset = dset["valid"]
    wm = load_model(ROOT / ckpt / "checkpoints" / "model_latest.pth", cfg, cfg.num_action_repeat, torch.device(device))
    wm.eval()
    for p in wm.parameters():
        p.requires_grad = False
    pre = Preprocessor(action_mean=dset.action_mean, action_std=dset.action_std, state_mean=dset.state_mean,
                       state_std=dset.state_std, proprio_mean=dset.proprio_mean, proprio_std=dset.proprio_std,
                       transform=dset.transform)
    return wm, pre, cfg


class ProprioReadout:
    """The proprio (position, velocity) an embedding stands for: least-squares inversion of the proprio encoder."""

    def __init__(self, wm, pre, device):
        self.enc, self.device = wm.proprio_encoder, device
        self.mean = torch.as_tensor(np.asarray(pre.proprio_mean), dtype=torch.float32, device=device)
        self.std = torch.as_tensor(np.asarray(pre.proprio_std), dtype=torch.float32, device=device)

    def decode(self, e, starts=16, steps=400, residual=False):
        e = e.float().to(self.device).flatten(1)
        n = len(e)
        gen = torch.Generator().manual_seed(0)
        p = (torch.randn(n * starts, 1, 4, generator=gen) * 1.5).to(self.device).requires_grad_(True)
        tgt = e.repeat_interleave(starts, 0)
        opt = torch.optim.Adam([p], lr=0.05)
        for _ in range(steps):
            opt.zero_grad()
            ((self.enc(p)[:, 0] - tgt) ** 2).sum().backward()
            opt.step()
        with torch.no_grad():
            res = ((self.enc(p)[:, 0] - tgt) ** 2).sum(1).view(n, starts)
            best = p[:, 0].view(n, starts, 4)[torch.arange(n), res.argmin(1)]
        out = (best * self.std + self.mean).cpu().numpy()
        return (out, res.min(1).values.cpu().numpy()) if residual else out


class Maze:
    """One layout: simulator and open-cell grid."""

    def __init__(self, spec, wm, pre, device):
        import gym
        import env  # noqa: F401  (registers diverse_maze)
        from datasets.diverse_maze_goals import _parse_open_grid

        self.env = gym.make("diverse_maze", maze_spec=spec).unwrapped
        self.env.prepare(0, np.zeros(4))  # sets up the render context and camera
        _, _, self.open = _parse_open_grid(spec)
        self.wm, self.pre, self.device = wm, pre, device

    def render(self, xy, vel=(0.0, 0.0)):
        self.env.set_state(np.asarray(xy, dtype=np.float64), np.asarray(vel, dtype=np.float64))
        self.env.set_marker()
        return self.env._render_frame()

    @torch.no_grad()
    def encode(self, frames, proprio, batch=256):
        from utils import move_to_device
        vs, ps = [], []
        for i in range(0, len(frames), batch):
            obs = {"visual": frames[i:i + batch, None], "proprio": proprio[i:i + batch, None].astype(np.float32)}
            z = self.wm.encode_obs(move_to_device(self.pre.transform_obs(obs), torch.device(self.device)))
            vs.append(z["visual"][:, 0].flatten(1))
            ps.append(z["proprio"][:, 0].flatten(1))
        return torch.cat(vs), torch.cat(ps)

    def crosses_wall(self, a, b, n=25):
        for t in np.linspace(0, 1, n):
            r, c = np.rint(a + t * (b - a)).astype(int)
            if not (0 <= r < self.open.shape[0] and 0 <= c < self.open.shape[1]) or not self.open[r, c]:
                return True
        return False

    def rollout(self, state, raw_actions):
        """Positions after every env step from a full state (pos, vel) under raw actions (T, 2)."""
        self.env.set_state(np.asarray(state[:2], dtype=np.float64), np.asarray(state[2:4], dtype=np.float64))
        out = []
        for a in raw_actions:
            _, _, _, info = self.env.step(a)
            out.append(np.array(info["state"][:4], dtype=np.float64))
        return np.array(out)


def dist(za, zb, alpha=1.0):
    """The planning objective's metric between single-frame latent dicts (MSE visual + alpha MSE proprio)."""
    mv = float(((za["visual"].flatten(1) - zb["visual"].flatten(1)) ** 2).mean())
    mp = float(((za["proprio"].flatten(1) - zb["proprio"].flatten(1)) ** 2).mean())
    return mv + alpha * mp, mv, mp


def latent(wm, pre, device, frame, proprio):
    from utils import move_to_device
    obs = {"visual": frame[None, None], "proprio": np.asarray(proprio, dtype=np.float32)[None, None]}
    return move_to_device(pre.transform_obs(obs), torch.device(device))


def gd_plan(wm, objective, obs0, z_g, H, A, zero_act, opt_steps=100, lr=0.1):
    """The planner's GD (Adam, cosine schedule) from zero env actions; returns actions (1, H, A) and predicted latents."""
    acts = zero_act.expand(1, H, A).clone().requires_grad_(True)
    opt = torch.optim.Adam([acts], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=opt_steps, eta_min=0.0)
    for _ in range(opt_steps):
        opt.zero_grad()
        z, _ = wm.rollout(obs_0=obs0, act=acts)
        objective(z, z_g).sum().backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        z, _ = wm.rollout(obs_0=obs0, act=acts)
    return acts.detach(), z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", choices=list(LEVELS), required=True)
    ap.add_argument("--root", default="eval_outputs/stag_maze100")
    ap.add_argument("--base", default="frozen")
    ap.add_argument("--ckpt", default="checkpoints/diversemaze_layout_shift")
    ap.add_argument("--data", default="data/diverse_maze")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--per-episode", type=int, default=4, help="stalled and moving replans sampled per episode")
    ap.add_argument("--gamma-hi", type=float, default=0.15)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-eps", type=int, default=0, help="first N logged episodes only (smoke tests)")
    ap.add_argument("--out", default="eval_outputs/stag_maze_diag")
    a = ap.parse_args()

    from datasets.diverse_maze_goals import generate_diverse_maze_goals
    from einops import rearrange
    from planning.objectives import create_objective_fn

    wm, pre, cfg = build(a.ckpt, a.data, a.device)
    ro = ProprioReadout(wm, pre, a.device)
    fs = cfg.frameskip
    pcfg = __import__("omegaconf").OmegaConf.load(ROOT / "conf/stag_plan_gd_diversemaze.yaml")
    H = pcfg.planner.sub_planner.horizon // fs
    objective = create_objective_fn(alpha=pcfg.objective.alpha, base=pcfg.objective.base, mode=pcfg.objective.mode)
    lo, hi = LEVELS[a.level]
    goals = generate_diverse_maze_goals(val_data_path=str(ROOT / a.data / "val"), n_evals=a.n, seed=a.seed,
                                        min_block_radius=lo, max_block_radius=hi)
    eps = collections.defaultdict(list)
    for f in sorted((ROOT / a.root / a.level / a.base / "ref").glob("shard*/per_iter_logs.json")):
        for line in f.open():
            if line.strip():
                r = json.loads(line)
                eps[r["sample_idx"]].append(r)
    mazes, rng = {}, np.random.default_rng(0)
    out_dir = ROOT / a.out
    out_dir.mkdir(parents=True, exist_ok=True)
    fout = (out_dir / f"{a.level}_{a.base}.jsonl").open("w")
    act_mean = torch.as_tensor(np.asarray(pre.action_mean), dtype=torch.float32)
    act_std = torch.as_tensor(np.asarray(pre.action_std), dtype=torch.float32)
    zero = rearrange(pre.normalize_actions(torch.zeros(1, 1, fs, 2)), "b t f d -> b t (f d)").to(a.device).float()
    for i in sorted(eps)[: a.max_eps or None]:
        rs = sorted(eps[i], key=lambda r: r["step"])
        by = {r["step"]: r for r in rs}
        spec = goals["map_specs"][i]
        key = int(goals["map_indices"][i])
        if key not in mazes:
            mazes[key] = Maze(spec, wm, pre, a.device)
        mz = mazes[key]
        start, goal = goals["start_states"][i], goals["goal_states"][i]
        state = {0: np.asarray(start, dtype=np.float64)}
        state.update({k: np.asarray(r["state"], dtype=np.float64) for k, r in by.items()})
        # replan k starts in state[k], executes by[k + 1]["act"] and ends in state[k + 1]
        cand = [k for k in by if by[k].get("stag_r") is not None and (k + 1) in by]
        stalled = [k for k in cand if by[k]["stag_r"] < a.gamma_hi]
        moving = [k for k in cand if by[k]["stag_r"] >= a.gamma_hi]
        pick = lambda ks: sorted(rng.choice(ks, size=min(a.per_episode, len(ks)), replace=False).tolist()) if ks else []  # noqa: E731
        g_frame = mz.render(goal[:2])
        z_g = wm.encode_obs(latent(wm, pre, a.device, g_frame, np.r_[goal[:2], 0.0, 0.0]))
        for kind, ks in (("stalled", pick(stalled)), ("moving", pick(moving))):
            for k in ks:
                s0, s1 = state[k], state[k + 1]
                frame0, frame1 = mz.render(s0[:2], s0[2:4]), mz.render(s1[:2], s1[2:4])
                obs0 = latent(wm, pre, a.device, frame0, s0)
                act = torch.as_tensor(by[k + 1]["act"], dtype=torch.float32, device=a.device).view(1, 1, -1)
                with torch.no_grad():
                    z1p, _ = wm.rollout(obs_0=obs0, act=act)
                zv1, zp1 = mz.encode(frame1[None], s1[None])
                dec, res = ro.decode(torch.cat([z1p["proprio"][:, 1].flatten(1), zp1]), residual=True)
                p_pred1, p_true1 = dec[:, :2]
                z0 = {"visual": z1p["visual"][:, 0], "proprio": z1p["proprio"][:, 0]}
                zh1 = {"visual": z1p["visual"][:, 1], "proprio": z1p["proprio"][:, 1]}
                zt1 = {"visual": zv1, "proprio": zp1}
                zg = {"visual": z_g["visual"][:, 0], "proprio": z_g["proprio"][:, 0]}
                acts, zp = gd_plan(wm, objective, obs0, z_g, H, zero.shape[-1], zero)
                p_plan = ro.decode(zp["proprio"][0, 1:])[:, :2]  # predicted positions after model steps 1..H
                raw = (rearrange(acts.cpu(), "b t (f d) -> (b t f) d", f=fs) * act_std + act_mean).numpy()
                sim_full = mz.rollout(s0, raw)[fs - 1::fs]  # true states after model steps 1..H
                sim = sim_full[:, :2]
                zve, zpe = mz.encode(mz.render(sim_full[-1][:2], sim_full[-1][2:4])[None], sim_full[-1][None])
                zend_pred = {"visual": zp["visual"][:, -1], "proprio": zp["proprio"][:, -1]}
                zend_true = {"visual": zve, "proprio": zpe}
                path = [s0[:2]] + list(p_plan)
                rec = {
                    "sample_idx": i, "replan": k, "kind": kind, "stag_r": by[k]["stag_r"], "layout": key,
                    "pos": s0[:2].round(4).tolist(), "goal": np.asarray(goal[:2]).round(4).tolist(),
                    "dist_goal": float(np.linalg.norm(s0[:2] - goal[:2])),
                    "step_true": float(np.linalg.norm(s1[:2] - s0[:2])),
                    "step_pred": float(np.linalg.norm(p_pred1 - s0[:2])),
                    "step_pred_err": float(np.linalg.norm(p_pred1 - s1[:2])),
                    "readout_err": float(np.linalg.norm(p_true1 - s1[:2])),
                    "plan_end_pred": p_plan[-1].round(3).tolist(), "plan_end_true": sim[-1].round(3).tolist(),
                    "plan_gain_pred": float(np.linalg.norm(s0[:2] - goal[:2]) - np.linalg.norm(p_plan[-1] - goal[:2])),
                    "plan_gain_true": float(np.linalg.norm(s0[:2] - goal[:2]) - np.linalg.norm(sim[-1] - goal[:2])),
                    "plan_end_err": float(np.linalg.norm(p_plan[-1] - sim[-1])),
                    "plan_move_pred": float(np.linalg.norm(p_plan[-1] - s0[:2])),
                    "plan_move_true": float(np.linalg.norm(sim[-1] - s0[:2])),
                    "plan_crosses_wall": any(mz.crosses_wall(path[j], path[j + 1]) for j in range(len(path) - 1)),
                    "step_crosses_wall": mz.crosses_wall(s0[:2], p_pred1),
                    "readout_res_pred": float(res[0]), "readout_res_true": float(res[1]),
                    # the planner's own metric (MSE visual + alpha MSE proprio)
                    "lat_err1": dist(zh1, zt1), "lat_move1_true": dist(zt1, z0), "lat_move1_pred": dist(zh1, z0),
                    "lat_goal_now": dist(z0, zg), "lat_goal_end_pred": dist(zend_pred, zg),
                    "lat_goal_end_true": dist(zend_true, zg),
                }
                fout.write(json.dumps(rec) + "\n")
                fout.flush()
        print(f"episode {i}: {len(stalled)} stalled / {len(moving)} moving replans", flush=True)
    fout.close()


if __name__ == "__main__":
    main()
