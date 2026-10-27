"""E1: does a TEMPO-style offline temporal distance learned from trajectories transfer to unseen maze layouts?

The released maze data holds 5 layouts (the world model's held-out ones) x 50 random-walk trajectories of 100 steps,
states and actions only. A temporal distance F (TEMPO, arXiv 2610.04988: an MLP on the frozen latent, trained so
that ||F(z_t) - F(z_t+k)|| regresses k / kappa on same-trajectory pairs, hinge on pairs > K apart) is trained
leave-one-layout-out (train on 4 layouts, test on the 5th) and, as the control, in-layout (40 trajectories of every
layout). Both are evaluated on the same points of every layout, so their gap is what an unseen layout costs.

  dyn     Do the data and the evaluation env share dynamics? Every data transition is replayed in the env from the
          data state with the data action (teacher forcing), with m env steps per data step and with or without
          the env's velocity clip, then a density / damping sweep at the best m; data positions are checked
          against the env's free space.
  encode  Renders every data state, a 0.1 lattice of every layout's free space and the E0 goals, and encodes them
          with the frozen encoder (proprio velocity set to 0: a single frame carries no velocity, as in TEMPO,
          and the data's velocity scale differs from the env's).
  train   F for every setting (lolo<L>, inlayout) x K x seed.
  train_geo  Diagnostic oracle, privileged: the same F supervised with the true geodesic distance between lattice
          points of the training layouts. Separates a weak training signal (trajectory time of fast random walks)
          from a frozen latent that cannot express walls.
  eval    A ranking: Spearman of a cost against geodesic distance on lattice pairs, all and wall-separated pairs
            (geodesic - Euclidean > 1).
          B correction: on lattice triples (s1, s2, g) whose geodesic distances differ by > 0.5 and that the
            Euclidean distance orders wrongly, how often the cost orders them right.
          C traps: at the E0 trap states (2-trap with a route rise, first trigger or last replan), does the argmin
            of the cost within geodesic reach 2 make geodesic progress >= eps (escape)?
          D harm: the same test at the trigger states of later successes.
          E greedy: repeated reach-2 jumps to the cost's argmin from the 100 E0 starts: reach the goal?
          Costs: the planner's latent metric (MSE visual + alpha MSE proprio), d_F, and TEMPO's blend
          (1 - w) latent + w (dbar d_F)^2 with dbar the median kappa-step latent displacement of the training data.
          C-E use, for each episode, the F of the fold that held its layout out (or the in-layout F).

    LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia python scripts/stag_maze_tempo.py --stages dyn
    CUDA_VISIBLE_DEVICES=5 LD_LIBRARY_PATH=... python scripts/stag_maze_tempo.py --stages encode train eval summary
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
from scipy.sparse.csgraph import dijkstra  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from stag_maze_trap import EPS, LEVELS, Geo  # noqa: E402

N_LAYOUTS, N_TRAIN_IN = 5, 40  # in-layout control: first 40 trajectories of every layout


def load_val(a):
    d = ROOT / a.data / "val"
    ld = lambda f: torch.load(d / f"{f}.pth", weights_only=False)  # noqa: E731
    return ld("states").numpy(), ld("actions").numpy(), ld("map_indices").numpy(), ld("maze_specs")


# ---------------------------------------------------------------------------------------------------- dyn
def stage_dyn(a, out_dir):
    import gym
    import env  # noqa: F401  (registers diverse_maze)

    S, A, M, specs = load_val(a)
    n, T = S.shape[:2]
    geos = {k: Geo(specs[k]) for k in specs}
    clear = np.concatenate([geos[int(M[i])].clearance(S[i, :, :2]) for i in range(n)])
    res = {"n_traj": int(n), "steps": int(T), "data_step_median": float(np.median(np.linalg.norm(np.diff(S[..., :2], axis=1), axis=-1))),
           "data_speed_median": float(np.median(np.linalg.norm(S[..., 2:], axis=-1))),
           "data_speed_p99": float(np.percentile(np.linalg.norm(S[..., 2:], axis=-1), 99)),
           "frac_speed_over_5": float((np.abs(S[..., 2:]) > 5).any(-1).mean()),
           "frac_pos_in_wall": float((clear < -0.02).mean()), "clearance_min": float(clear.min())}
    rng = np.random.default_rng(0)
    pairs = [(i, t) for i in range(n) for t in range(T - 1)]
    pairs = [pairs[j] for j in rng.choice(len(pairs), size=min(a.dyn_n, len(pairs)), replace=False)]

    def replay(m, clip, density=1.0, damping=1.0):
        envs = {}
        pe, ve, ratio = [], [], []
        for i, t in pairs:
            k = int(M[i])
            if k not in envs:
                envs[k] = gym.make("diverse_maze", maze_spec=specs[k], density_scale=density, damping_scale=damping).unwrapped
                if not clip:
                    envs[k].clip_velocity = lambda: None  # instance attribute shadows the method used in step()
            e = envs[k]
            e.set_state(S[i, t, :2].astype(np.float64), S[i, t, 2:].astype(np.float64))
            for _ in range(m):
                _, _, _, info = e.step(A[i, t].astype(np.float64))
            s1 = np.asarray(info["state"][:4], dtype=np.float64)
            pe.append(np.linalg.norm(s1[:2] - S[i, t + 1, :2]))
            ve.append(np.linalg.norm(s1[2:] - S[i, t + 1, 2:]))
            dd = np.linalg.norm(S[i, t + 1, :2] - S[i, t, :2])
            if dd > 0.05:
                ratio.append(np.linalg.norm(s1[:2] - S[i, t, :2]) / dd)
        return {"m": m, "clip": clip, "density": density, "damping": damping, "pos_err_median": float(np.median(pe)),
                "vel_err_median": float(np.median(ve)), "disp_ratio_median": float(np.median(ratio))}

    res["variants"] = [replay(m, clip) for clip in (True, False) for m in range(1, 7)]
    best = min(res["variants"], key=lambda v: v["pos_err_median"])
    res["sweep"] = [replay(best["m"], best["clip"], dn, dp) for dn in (0.25, 0.5, 1.0) for dp in (0.25, 0.5, 1.0)]
    (out_dir / "dyn.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("variants", "sweep")}, indent=1))
    for v in res["variants"] + res["sweep"]:
        print(v)


# ---------------------------------------------------------------------------------------------------- encode
def lattice(geo, stride):
    ij = np.argwhere(geo.idx >= 0)  # same C order as the node ids
    keep = (ij % stride == 0).all(1)
    return np.nonzero(keep)[0]


def stage_encode(a, out_dir):
    from datasets.diverse_maze_goals import generate_diverse_maze_goals
    from stag_maze_diag import Maze, build

    S, _, M, specs = load_val(a)
    wm, pre, _ = build(a.ckpt, a.data, a.device)
    geos = {k: Geo(specs[k]) for k in specs}
    mazes = {k: Maze(specs[k], wm, pre, a.device) for k in specs}

    def enc(k, xy):
        zv, zp = [], []
        for s in range(0, len(xy), 512):
            frames = np.stack([mazes[k].render(p) for p in xy[s:s + 512]])
            v, p = mazes[k].encode(frames, np.c_[xy[s:s + 512], np.zeros((len(frames), 2))])
            zv.append(v.cpu().numpy())
            zp.append(p.cpu().numpy())
        return np.concatenate(zv), np.concatenate(zp)

    out = {}
    zv, zp = zip(*[enc(int(M[i]), S[i, :, :2]) for i in range(len(S))])
    out["data_zv"], out["data_zp"] = np.stack(zv), np.stack(zp)  # (n, T, D)
    stride = int(round(a.lattice / geos[0].h))
    for k in specs:
        nodes = lattice(geos[k], stride)
        out[f"lat{k}_nodes"] = nodes
        out[f"lat{k}_zv"], out[f"lat{k}_zp"] = enc(k, geos[k].pos[nodes])
    lo, hi = LEVELS[a.level]
    goals = generate_diverse_maze_goals(val_data_path=str(ROOT / a.data / "val"), n_evals=a.n, seed=a.seed,
                                        min_block_radius=lo, max_block_radius=hi)
    gl = np.asarray(goals["map_indices"], dtype=int)
    gx = np.asarray(goals["goal_states"])[:, :2]
    gz = [enc(int(gl[i]), gx[i:i + 1]) for i in range(len(gx))]
    out["goal_zv"], out["goal_zp"] = np.concatenate([z[0] for z in gz]), np.concatenate([z[1] for z in gz])
    out["goal_xy"], out["goal_layout"] = gx, gl
    out["start_xy"] = np.asarray(goals["start_states"])[:, :2]
    np.savez_compressed(out_dir / "latents.npz", **out)
    print({k: v.shape for k, v in out.items()})


# ---------------------------------------------------------------------------------------------------- train
class F(torch.nn.Module):
    def __init__(self, d_in, mean, std, width=256, d_out=64):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32))
        self.net = torch.nn.Sequential(torch.nn.Linear(d_in, width), torch.nn.SiLU(), torch.nn.Linear(width, width),
                                       torch.nn.SiLU(), torch.nn.Linear(width, d_out))

    def forward(self, z):
        return self.net((z - self.mean) / self.std)


def settings():
    return [f"lolo{k}" for k in range(N_LAYOUTS)] + ["inlayout"]


def train_trajs(setting, M):
    if setting == "inlayout":
        return np.concatenate([np.nonzero(M == k)[0][:N_TRAIN_IN] for k in range(N_LAYOUTS)])
    return np.nonzero(M != int(setting[4:]))[0]


def lattice_geodesics(geo, nodes):
    """Geodesic distances between all lattice points of one layout."""
    return dijkstra(geo.G, directed=False, indices=nodes)[:, nodes]


def stage_train_geo(a, out_dir):
    """Diagnostic oracle: F supervised with the true geodesic distance between lattice points of the training layouts
    (privileged). It separates a weak training signal from a latent that cannot express walls."""
    _, _, _, specs = load_val(a)
    L = np.load(out_dir / "latents.npz")
    dev = torch.device(a.device)
    lat = {k: (np.concatenate([L[f"lat{k}_zv"], L[f"lat{k}_zp"]], -1).astype(np.float32),
               lattice_geodesics(Geo(specs[k]), L[f"lat{k}_nodes"]).astype(np.float32)) for k in specs}
    (out_dir / "models").mkdir(exist_ok=True)
    for setting in settings():
        lay = [k for k in specs if setting == "inlayout" or k != int(setting[4:])]
        Zs = [torch.as_tensor(lat[k][0], device=dev) for k in lay]
        Gs = [torch.as_tensor(lat[k][1], device=dev) for k in lay]
        flat = np.concatenate([lat[k][0] for k in lay])
        for seed in a.seeds:
            torch.manual_seed(seed)
            g = torch.Generator(device=dev).manual_seed(seed)
            f = F(flat.shape[-1], flat.mean(0), flat.std(0) + 1e-6).to(dev)
            opt = torch.optim.Adam(f.parameters(), lr=1e-3)
            for step in range(a.steps):
                j = step % len(lay)
                n = len(Zs[j])
                x, y = torch.randint(n, (2, a.batch), device=dev, generator=g)
                d = (f(Zs[j][x]) - f(Zs[j][y])).norm(dim=-1)
                loss = torch.nn.functional.smooth_l1_loss(d, Gs[j][x, y])
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(f.parameters(), 10.0)
                opt.step()
            torch.save(f.state_dict(), out_dir / "models" / f"geo_{setting}_s{seed}.pt")
            print("geo", setting, seed, round(float(loss), 4))


def stage_train(a, out_dir):
    S, _, M, _ = load_val(a)
    L = np.load(out_dir / "latents.npz")
    Z = np.concatenate([L["data_zv"], L["data_zp"]], -1).astype(np.float32)  # (n, T, D)
    dev = torch.device(a.device)
    (out_dir / "models").mkdir(exist_ok=True)
    T = Z.shape[1]
    for setting in settings():
        tr = train_trajs(setting, M)
        Zt = torch.as_tensor(Z[tr], device=dev)
        flat = Z[tr].reshape(-1, Z.shape[-1])
        for K in a.K:
            for seed in a.seeds:
                torch.manual_seed(seed)
                g = torch.Generator(device=dev).manual_seed(seed)
                f = F(Z.shape[-1], flat.mean(0), flat.std(0) + 1e-6).to(dev)
                opt = torch.optim.Adam(f.parameters(), lr=1e-3)
                for _ in range(a.steps):
                    b = torch.randint(len(tr), (a.batch,), device=dev, generator=g)
                    k = torch.randint(1, K + 1, (a.batch,), device=dev, generator=g)
                    t0 = (torch.rand(a.batch, device=dev, generator=g) * (T - k)).long()
                    d = (f(Zt[b, t0]) - f(Zt[b, t0 + k])).norm(dim=-1)
                    loss = torch.nn.functional.smooth_l1_loss(d, k.float() / a.kappa)
                    if K < T - 1:  # hinge: pairs more than K apart at least K / kappa apart
                        kf = torch.randint(K + 1, T, (a.batch,), device=dev, generator=g)
                        t1 = (torch.rand(a.batch, device=dev, generator=g) * (T - kf)).long()
                        df = (f(Zt[b, t1]) - f(Zt[b, t1 + kf])).norm(dim=-1)
                        loss = loss + 0.5 * torch.relu(K / a.kappa - df).mean()
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(f.parameters(), 10.0)
                    opt.step()
                torch.save(f.state_dict(), out_dir / "models" / f"{setting}_K{K}_s{seed}.pt")
                print(setting, K, seed, round(float(loss), 4))


# ---------------------------------------------------------------------------------------------------- eval
def latent_cost(zv, zp, gv, gp, alpha):
    return ((zv - gv) ** 2).mean(-1) + alpha * ((zp - gp) ** 2).mean(-1)


def trap_states(a):
    """E0 trap states (2-trap with a route rise at the first trigger or the last replan) and later successes' triggers."""
    geom = [json.loads(x) for x in (ROOT / a.trap_dir / f"{a.level}_geom.jsonl").open()]
    traps, succ = [], []
    for r in geom:
        pr = r["per_replan"]
        if not r["success"]:
            for w, k in (("trig", r["trig"]), ("end", r["T"])):
                if k is not None and pr[str(k)]["trap2"] and pr[str(k)]["barrier"] > EPS:
                    traps.append({"base": r["base"], "i": r["i"], "which": w, "k": k, "cls": r["cls"]})
        elif r["trig"] is not None:
            succ.append({"base": r["base"], "i": r["i"], "which": "trig", "k": r["trig"]})
    return traps, succ


def episode_states(a):
    """(base, i) -> {replan: state} from the reference runs, as stag_maze_trap reads them."""
    from stag_maze_trap import load_eps
    out = {}
    for base in ("frozen", "adajepa"):
        for i, rs in load_eps(ROOT / a.root / a.level / base / "ref").items():
            out[(base, i)] = {r["step"]: np.asarray(r["state"], dtype=np.float64) for r in rs}
    return out


def stage_eval(a, out_dir):
    S, _, M, specs = load_val(a)
    L = np.load(out_dir / "latents.npz")
    dev = torch.device(a.device)
    geos = {k: Geo(specs[k]) for k in specs}
    D = L["data_zv"].shape[-1] + L["data_zp"].shape[-1]
    traps, succ = trap_states(a)
    est = episode_states(a)
    rng = np.random.default_rng(0)
    # fixed evaluation samples per layout: lattice goals for A/B, geodesic matrices
    samp = {}
    for k in specs:
        nodes = L[f"lat{k}_nodes"]
        gi = rng.choice(len(nodes), size=min(a.n_goals, len(nodes)), replace=False)
        dg = dijkstra(geos[k].G, directed=False, indices=nodes[gi])[:, nodes]  # (n_goals, n_lat)
        pos = geos[k].pos[nodes]
        eu = np.linalg.norm(pos[None] - pos[gi][:, None], axis=-1)
        trip = []
        for gj in range(len(gi)):
            s1, s2 = rng.integers(len(nodes), size=(2, a.n_trip))
            ok = (np.abs(dg[gj, s1] - dg[gj, s2]) > 0.5) & ((dg[gj, s1] < dg[gj, s2]) != (eu[gj, s1] < eu[gj, s2]))
            trip += [(gj, x, y) for x, y in zip(s1[ok], s2[ok])]
        samp[k] = {"gi": gi, "dg": dg, "eu": eu, "trip": np.array(trip)}
    rows = []
    models = [("time", K) for K in a.K if "time" in a.targets] + ([("geo", 0)] if "geo" in a.targets else [])
    for setting in settings():
        for target, K in models:
            for seed in a.seeds:
                f = F(D, np.zeros(D), np.ones(D)).to(dev)
                name = f"{setting}_K{K}_s{seed}.pt" if target == "time" else f"geo_{setting}_s{seed}.pt"
                f.load_state_dict(torch.load(out_dir / "models" / name, map_location=dev))
                f.eval()
                emb = lambda zv, zp: f(torch.as_tensor(np.concatenate([zv, zp], -1), dtype=torch.float32, device=dev)).detach().cpu().numpy()  # noqa: E731
                tr, kap = train_trajs(setting, M), int(a.kappa)
                v, p = L["data_zv"][tr], L["data_zp"][tr]
                step = np.sqrt(latent_cost(v[:, kap:], p[:, kap:], v[:, :-kap], p[:, :-kap], a.alpha))
                if target == "geo":  # d_F counts cells: latent displacement per cell (kappa-step pairs are rarely wall-separated)
                    step = step / np.maximum(np.linalg.norm(S[tr, kap:, :2] - S[tr, :-kap, :2], axis=-1), 1e-3)
                dbar = float(np.median(step))
                eval_layouts = [int(setting[4:])] if setting.startswith("lolo") else list(specs)
                for k in eval_layouts:
                    rec = {"target": target, "setting": setting, "K": K, "seed": seed, "layout": k, "dbar": dbar}
                    zv, zp = L[f"lat{k}_zv"], L[f"lat{k}_zp"]
                    E = emb(zv, zp)
                    sm = samp[k]
                    gi = sm["gi"]
                    lat = latent_cost(zv[None], zp[None], zv[gi][:, None], zp[gi][:, None], a.alpha)  # (n_goals, n_lat)
                    dF = np.linalg.norm(E[None] - E[gi][:, None], axis=-1)
                    costs = {"latent": lat, "F": dF, "blend": (1 - a.w) * lat + a.w * (dbar * dF) ** 2, "euclid": sm["eu"]}
                    off = sm["dg"] > 0  # drop each goal paired with itself
                    sep = (sm["dg"] - sm["eu"] > 1.0) & off
                    for name, c in costs.items():
                        rec[f"A_{name}"] = float(spearmanr(c[off], sm["dg"][off])[0])
                        rec[f"Asep_{name}"] = float(spearmanr(c[sep], sm["dg"][sep])[0]) if sep.sum() > 10 else None
                        t = sm["trip"]
                        if len(t):
                            right = (c[t[:, 0], t[:, 1]] < c[t[:, 0], t[:, 2]]) == (sm["dg"][t[:, 0], t[:, 1]] < sm["dg"][t[:, 0], t[:, 2]])
                            rec[f"B_{name}"] = float(right.mean())
                    rec["n_sep"], rec["n_trip"] = int(sep.sum()), int(len(sm["trip"]))
                    rows.append(rec)
                # C, D, E on the E0 episodes of the layouts this F is evaluated on
                gl = L["goal_layout"]
                for kind, states in (("C", traps), ("D", succ)):
                    for s in states:
                        k = int(gl[s["i"]])
                        if k not in eval_layouts:
                            continue
                        res = local_test(a, geos[k], L, k, s["i"], est[(s["base"], s["i"])][s["k"]][:2], emb, dbar)
                        rows.append({"target": target, "setting": setting, "K": K, "seed": seed, "layout": k, "kind": kind, **s, **res})
                for i in range(len(gl)):
                    k = int(gl[i])
                    if k not in eval_layouts:
                        continue
                    rows.append({"target": target, "setting": setting, "K": K, "seed": seed, "layout": k, "kind": "E", "i": i,
                                 **greedy_test(a, geos[k], L, k, i, emb, dbar)})
                print(target, setting, K, seed, "done", flush=True)
    with (out_dir / "eval.jsonl").open("w") as fo:
        for r in rows:
            fo.write(json.dumps(r, default=float) + "\n")


def _lat_costs(a, L, k, i, emb, dbar):
    """Per lattice node of layout k: the three costs to the goal of E0 episode i."""
    zv, zp = L[f"lat{k}_zv"], L[f"lat{k}_zp"]
    gv, gp = L["goal_zv"][i], L["goal_zp"][i]
    lat = latent_cost(zv, zp, gv, gp, a.alpha)
    dF = np.linalg.norm(emb(zv, zp) - emb(gv[None], gp[None]), axis=-1)
    return {"latent": lat, "F": dF, "blend": (1 - a.w) * lat + a.w * (dbar * dF) ** 2}


def local_test(a, geo, L, k, i, p, emb, dbar):
    """Argmin of every cost within geodesic reach 2 of p (lattice points): geodesic progress and escape."""
    g = L["goal_xy"][i]
    dg, _ = geo.from_goal(g)
    nodes = L[f"lat{k}_nodes"]
    d = dijkstra(geo.G, directed=False, indices=geo.node(p), limit=2.0)[nodes]
    m = np.isfinite(d)
    g0 = float(dg[geo.node(p)])
    out = {}
    costs = _lat_costs(a, L, k, i, emb, dbar)
    costs["euclid"] = np.linalg.norm(geo.pos[nodes] - g, axis=1)
    for name, c in costs.items():
        q = nodes[m][np.argmin(c[m])]
        prog = g0 - float(dg[q])
        out[f"prog_{name}"], out[f"esc_{name}"] = round(prog, 4), bool(prog >= EPS)
    return out


def greedy_test(a, geo, L, k, i, emb, dbar, r=2.0, it=200):
    g, nodes = L["goal_xy"][i], L[f"lat{k}_nodes"]
    costs = _lat_costs(a, L, k, i, emb, dbar)
    costs["euclid"] = np.linalg.norm(geo.pos[nodes] - g, axis=1)
    pos = geo.pos[nodes]
    out = {}
    for name, c in costs.items():
        cur = int(np.argmin(np.linalg.norm(pos - L["start_xy"][i], axis=1)))
        for _ in range(it):
            d = dijkstra(geo.G, directed=False, indices=nodes[cur], limit=r)[nodes]
            m = np.nonzero(np.isfinite(d))[0]
            nxt = int(m[np.argmin(c[m])])
            if c[nxt] >= c[cur] - 1e-9:
                break
            cur = nxt
        out[f"reach_{name}"] = bool(np.linalg.norm(pos[cur] - g) < 0.5)
    return out


# ---------------------------------------------------------------------------------------------------- summary
def summarize(a, out_dir, rep_dir):
    rows = [json.loads(x) for x in (out_dir / "eval.jsonl").open()]
    dyn = json.loads((out_dir / "dyn.json").read_text()) if (out_dir / "dyn.json").exists() else None
    md, S = ["# E1：TEMPO 式离线距离能否推广到没见过的迷宫布局（汇总）", ""], {}
    if dyn:
        md += ["## 0. 数据与环境的动力学", "",
               f"- 数据：{dyn['n_traj']} 条 × {dyn['steps']} 步；每步位移中位数 {dyn['data_step_median']:.2f} 格；"
               f"速度中位数 {dyn['data_speed_median']:.2f}，p99 {dyn['data_speed_p99']:.2f}；"
               f"有分量超过 5 的状态 {dyn['frac_speed_over_5']:.1%}；位置落在环境墙内（间隙 < −0.02）{dyn['frac_pos_in_wall']:.2%}（最小间隙 {dyn['clearance_min']:.3f}）",
               "", "| 每个数据步对应的环境步 | 速度截断 | 密度 | 阻尼 | 位置误差（中位数） | 速度误差（中位数） | 位移比 环境/数据（中位数） |",
               "|---:|---|---:|---:|---:|---:|---:|"]
        for v in dyn["variants"] + dyn["sweep"]:
            md.append(f"| {v['m']} | {'有' if v['clip'] else '无'} | {v['density']} | {v['damping']} | {v['pos_err_median']:.3f} | "
                      f"{v['vel_err_median']:.3f} | {v['disp_ratio_median']:.2f} |")
        md.append("")
    glob_ = [r for r in rows if "kind" not in r]
    grp = lambda s: "inlayout" if s == "inlayout" else "lolo"  # noqa: E731
    gname = lambda g: "布局内" if g == "inlayout" else "留一布局"  # noqa: E731
    keys = [("time", K) for K in a.K if "time" in a.targets] + ([("geo", 0)] if "geo" in a.targets else [])
    mlabel = lambda t, K: f"TEMPO 式 K={K}" if t == "time" else "测地监督（特权诊断）"  # noqa: E731
    md += ["## 1. 全局排序（A）与纠正率（B）", "",
           "每个布局 100 个格点目标 × 全部格点；被墙隔开 = 测地 − 欧氏 > 1；B 只看欧氏距离排错的三元组。均值取遍布局与种子。"
           "欧氏和潜代价与 F 无关，每组重复列出。", "",
           "| 设置 | 距离 F | 代价 | A：Spearman（全部） | A：Spearman（被墙隔开） | B：纠正率 |", "|---|---|---|---:|---:|---:|"]
    for g in ("inlayout", "lolo"):
        for t, K in keys:
            R = [r for r in glob_ if grp(r["setting"]) == g and r["target"] == t and r["K"] == K]
            for name in ("euclid", "latent", "F", "blend"):
                av = lambda key: float(np.mean([r[key] for r in R if r.get(key) is not None]))  # noqa: E731
                S[f"{g}/{t}{K}/{name}"] = {"A": av(f"A_{name}"), "Asep": av(f"Asep_{name}"), "B": av(f"B_{name}")}
                md.append(f"| {gname(g)} | {mlabel(t, K)} | {name} | {av(f'A_{name}'):.3f} | {av(f'Asep_{name}'):.3f} | {av(f'B_{name}'):.3f} |")
    Kstar = max(a.K, key=lambda K: S[f"inlayout/time{K}/blend"]["Asep"])
    S["K_star"] = Kstar
    md += ["", f"TEMPO 式距离判断用的 K = {Kstar}（布局内设置下混合代价在被墙隔开的对上 Spearman 最高；只按布局内选）。", ""]
    show = [("time", Kstar)] + ([("geo", 0)] if "geo" in a.targets else [])
    loc = [r for r in rows if r.get("kind") in ("C", "D")]
    md += ["## 2. 陷阱状态上能否走出来（C）与正常状态上是否伤害（D）", "",
           "可达半径 2 以内代价最小的格点，测地距离下降 ≥ 0.25 算走出来。各种子分别计数后取均值（括号里是各种子的计数）。", "",
           "| 设置 | 距离 F | 状态 | 个数 | 欧氏 | 潜代价 | 只用 d_F | 混合 |", "|---|---|---|---:|---:|---:|---:|---:|"]
    for t, K in show:
        for g in ("inlayout", "lolo"):
            for kind, sel, label in (("C", lambda r: True, "陷阱状态（全部）"),
                                     ("C", lambda r: r["base"] == "adajepa", "陷阱状态（AdaJEPA）"),
                                     ("D", lambda r: True, "后来成功的触发状态")):
                R = [r for r in loc if grp(r["setting"]) == g and r["target"] == t and r["K"] == K and r["kind"] == kind and sel(r)]
                if not R:
                    continue
                per_seed = {s: [r for r in R if r["seed"] == s] for s in a.seeds}
                n = len(per_seed[a.seeds[0]])
                cells = []
                for name in ("euclid", "latent", "F", "blend"):
                    cnt = [sum(r[f"esc_{name}"] for r in per_seed[s]) for s in a.seeds]
                    cells.append(f"{np.mean(cnt):.1f}（{'/'.join(map(str, cnt))}）")
                    S[f"{g}/{t}{K}/{kind}/{label}/{name}"] = {"n": n, "counts": cnt}
                md.append(f"| {gname(g)} | {mlabel(t, K)} | {label} | {n} | " + " | ".join(cells) + " |")
    md += ["", "### AdaJEPA 的陷阱 episode（取每个 episode 第一个陷阱状态）", "",
           "| 设置 | 距离 F | episode 数 | 潜代价走出 | 只用 d_F 走出 | 混合走出 |", "|---|---|---:|---:|---:|---:|"]
    for t, K in show:
        for g in ("inlayout", "lolo"):
            cnts = {name: [] for name in ("latent", "F", "blend")}
            for s in a.seeds:
                R = [r for r in loc if grp(r["setting"]) == g and r["target"] == t and r["K"] == K and r["kind"] == "C"
                     and r["base"] == "adajepa" and r["seed"] == s]
                first = {}
                for r in sorted(R, key=lambda r: r["k"]):
                    first.setdefault(r["i"], r)
                for name in cnts:
                    cnts[name].append(sum(r[f"esc_{name}"] for r in first.values()))
                n_ep = len(first)
            S[f"{g}/{t}{K}/adajepa_trap_episodes"] = {"n": n_ep, **cnts}
            md.append(f"| {gname(g)} | {mlabel(t, K)} | {n_ep} | " +
                      " | ".join(f"{np.mean(v):.1f}（{'/'.join(map(str, v))}）" for v in cnts.values()) + " |")
    ge = [r for r in rows if r.get("kind") == "E"]
    md += ["", "## 3. 贪心可达（E）：从 E0 的 100 个起点用代价做可达半径 2 的贪心下降", "",
           "| 设置 | 距离 F | 欧氏 | 潜代价 | 只用 d_F | 混合 |", "|---|---|---:|---:|---:|---:|"]
    for t, K in show:
        for g in ("inlayout", "lolo"):
            cells = []
            for name in ("euclid", "latent", "F", "blend"):
                cnt = [sum(r[f"reach_{name}"] for r in ge if grp(r["setting"]) == g and r["target"] == t and r["K"] == K
                           and r["seed"] == s) for s in a.seeds]
                cells.append(f"{np.mean(cnt):.1f}（{'/'.join(map(str, cnt))}）")
                S[f"{g}/{t}{K}/E/{name}"] = cnt
            md.append(f"| {gname(g)} | {mlabel(t, K)} | " + " | ".join(cells) + " |")
    rep_dir.mkdir(parents=True, exist_ok=True)
    (rep_dir / "summary.md").write_text("\n".join(md) + "\n")
    (rep_dir / "summary.json").write_text(json.dumps(S, indent=1, default=float))
    print("\n".join(md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="+", default=["encode", "train", "train_geo", "eval", "summary"],
                    choices=["dyn", "encode", "train", "train_geo", "eval", "summary"])
    ap.add_argument("--targets", nargs="+", default=["time", "geo"], choices=["time", "geo"],
                    help="F trained on trajectory time (TEMPO) and / or on true geodesics (privileged diagnostic)")
    ap.add_argument("--level", choices=list(LEVELS), default="easy")
    ap.add_argument("--root", default="eval_outputs/stag_maze100")
    ap.add_argument("--trap-dir", default="eval_outputs/stag_maze_trap")
    ap.add_argument("--data", default="data/diverse_maze")
    ap.add_argument("--ckpt", default="checkpoints/diversemaze_layout_shift")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--dyn-n", type=int, default=6000, help="data transitions replayed per dynamics variant")
    ap.add_argument("--lattice", type=float, default=0.1)
    ap.add_argument("--K", type=int, nargs="+", default=[10, 20, 40])
    ap.add_argument("--kappa", type=float, default=5.0, help="data steps per unit of d_F (one model step)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--w", type=float, default=0.5, help="TEMPO blend weight")
    ap.add_argument("--alpha", type=float, default=1.0, help="objective.alpha of the plan config")
    ap.add_argument("--n-goals", type=int, default=100)
    ap.add_argument("--n-trip", type=int, default=400, help="triples drawn per lattice goal (before filtering)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="eval_outputs/stag_maze_tempo")
    ap.add_argument("--report", default="reports/stag_maze_tempo")
    a = ap.parse_args()
    out_dir = ROOT / a.out
    out_dir.mkdir(parents=True, exist_ok=True)
    for stage, fn in (("dyn", stage_dyn), ("encode", stage_encode), ("train", stage_train), ("train_geo", stage_train_geo),
                      ("eval", stage_eval)):
        if stage in a.stages:
            fn(a, out_dir)
    if "summary" in a.stages:
        summarize(a, out_dir, ROOT / a.report)


if __name__ == "__main__":
    main()
