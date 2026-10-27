"""Are the maze failures Euclidean traps, and could they be solved within the replans left?

Reads the reference runs of the frozen per-sample planner and of AdaJEPA (scripts/stag_maze_base_run.py) and the
layouts those runs used (the maze_dist goals, regenerated with the runs' seed). Positions are in the env's qpos units.
Cell k spans [k - 0.7, k + 0.3]: the agent body sits at qpos + 1.2 and the wall blocks at cell + 1
(env/pointmaze/maze_model.py), and the agent is a disk of radius 0.1. Free space is therefore a fine grid of the
points the disk fits on, and geodesic distances come from Dijkstra on that grid (8-connected, no corner cutting).

  geom    For every logged state: Euclidean and geodesic distance to the goal; the barrier along the shortest path
          (the largest rise of the Euclidean distance after a dip, i.e. how far the real route must first move away
          from the goal); for r = 1, 2, 3, whether the free point with the smallest Euclidean distance within
          geodesic radius r makes geodesic progress (< eps: an r-trap, i.e. greedily minimising the Euclidean
          distance within reach r makes no progress along the real route). Per start, whether repeated greedy jumps
          of reach r end at the goal. Each failure is classed, in this order: near miss (geodesic < 1 from the goal
          at the end), trap (an r-trap with barrier > eps at the first trigger or at the end), progressing
          (geodesic progress > 0.3 over the last 5 replans), or stuck in the open.
  sim     From the first stall trigger of every failure, and from every start as a check of the controller, a
          privileged scripted controller (pure pursuit along the geodesic path, PD on position, actions clipped to
          the env's [-1, 1]) runs for the replans left x n_taken_actions actions, each held for --action-repeat env
          steps as in the runs. Success (< 0.5) is checked at replan boundaries, as in the runs.
  latent  At the trigger and last states of the failures, and at the trigger states of later successes, the
          planner's own single-frame metric (MSE visual + alpha MSE proprio, initial model, zero velocity) is
          evaluated on rendered points of the same geodesic balls. Its argmin is tested for geodesic progress exactly
          as the Euclidean argmin is in geom, on the same points.

    LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia python scripts/stag_maze_trap.py --stages geom sim
    CUDA_VISIBLE_DEVICES=5 LD_LIBRARY_PATH=... python scripts/stag_maze_trap.py --stages latent summary
"""
import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from scipy.sparse import csr_matrix  # noqa: E402
from scipy.sparse.csgraph import dijkstra  # noqa: E402

LEVELS = {"easy": (3, 5), "medium": (6, 9), "hard": (10, 14)}
BASES = ("frozen", "adajepa")
OFF, RADIUS = 0.2, 0.1  # qpos of cell k's centre is k - OFF; agent radius
EPS = 0.25  # progress / rise counted, in maze units (success needs < 0.5)
RS = (1, 2, 3)  # geodesic reach of the local test
NEAR, PROG, WIN = 1.0, 0.3, 5  # near-miss distance; geodesic progress over the last WIN replans


class Geo:
    """Free space of one layout on a fine grid, geodesic distances and the local trap test."""

    def __init__(self, spec, h=0.05):
        from datasets.diverse_maze_goals import _parse_open_grid

        _, _, self.open = _parse_open_grid(spec)
        rows, cols = self.open.shape
        self.h = h
        self.walls = np.argwhere(~self.open).astype(np.float64)
        xs = -0.5 - OFF + h / 2 + h * np.arange(int(round(rows / h)))
        ys = -0.5 - OFF + h / 2 + h * np.arange(int(round(cols / h)))
        self.x0, self.y0 = xs[0], ys[0]
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        free = self.clearance(np.stack([X, Y], -1)) > 0
        nx, ny = free.shape
        idx = -np.ones((nx, ny), dtype=np.int64)
        idx[free] = np.arange(free.sum())
        self.idx, self.pos = idx, np.stack([X[free], Y[free]], -1)
        src, dst, w = [], [], []
        for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
            a = idx[0:nx - dx, max(0, -dy):ny - max(0, dy)]
            b = idx[dx:nx, max(dy, 0):ny + min(dy, 0)]
            ok = (a >= 0) & (b >= 0)
            if dx and dy:  # no corner cutting
                ok &= (idx[dx:nx, max(0, -dy):ny - max(0, dy)] >= 0) & (idx[0:nx - dx, max(dy, 0):ny + min(dy, 0)] >= 0)
            src.append(a[ok])
            dst.append(b[ok])
            w.append(np.full(ok.sum(), h * np.hypot(dx, dy)))
        n = len(self.pos)
        self.G = csr_matrix((np.concatenate(w), (np.concatenate(src), np.concatenate(dst))), shape=(n, n))
        self._goal = {}

    def clearance(self, p):
        """Distance from points (..., 2) to the nearest wall block minus the agent radius (<= 0: the disk hits a wall)."""
        lo = self.walls - 0.5 - OFF
        p = np.asarray(p, dtype=np.float64)[..., None, :]
        d = np.maximum(np.maximum(lo - p, p - (lo + 1.0)), 0.0)
        return np.sqrt((d ** 2).sum(-1)).min(-1) - RADIUS

    def node(self, p):
        i = int(np.clip(round((p[0] - self.x0) / self.h), 0, self.idx.shape[0] - 1))
        j = int(np.clip(round((p[1] - self.y0) / self.h), 0, self.idx.shape[1] - 1))
        if self.idx[i, j] >= 0:
            return int(self.idx[i, j])
        return int(np.argmin(((self.pos - p) ** 2).sum(1)))

    def from_goal(self, g):
        key = (round(float(g[0]), 6), round(float(g[1]), 6))
        if key not in self._goal:
            self._goal[key] = dijkstra(self.G, directed=False, indices=self.node(g), return_predecessors=True)
        return self._goal[key]

    def path(self, n, g):
        """Nodes of the shortest path from node n to the goal node."""
        _, pred = self.from_goal(g)
        out = [n]
        while pred[out[-1]] >= 0:
            out.append(int(pred[out[-1]]))
        return out

    def ball(self, p, r=max(RS)):
        d = dijkstra(self.G, directed=False, indices=self.node(p), limit=r)
        keep = np.isfinite(d)
        return np.nonzero(keep)[0], d[keep]

    def greedy(self, p, g, r, it=200):
        """Repeated jumps to the Euclidean argmin within geodesic reach r (a perfect-model planner of reach r that
        minimises the Euclidean distance): its end point and whether that is within 0.5 of the goal."""
        p, g = np.asarray(p[:2], dtype=np.float64), np.asarray(g[:2], dtype=np.float64)
        for _ in range(it):
            nodes, d = self.ball(p, r)
            q = self.pos[nodes[np.argmin(np.linalg.norm(self.pos[nodes] - g, axis=1))]]
            if np.linalg.norm(q - g) > np.linalg.norm(p - g) - 1e-3:
                break
            p = q
        return bool(np.linalg.norm(p - g) < 0.5), round(float(np.linalg.norm(p - g)), 4)

    def blocked(self, a, b, n=60):
        t = np.linspace(0.0, 1.0, n)[:, None]
        return bool((self.clearance(a + t * (b - a)) <= 0).any())

    def query(self, p, g):
        p, g = np.asarray(p[:2], dtype=np.float64), np.asarray(g[:2], dtype=np.float64)
        dg, _ = self.from_goal(g)
        n0 = self.node(p)
        e0, g0 = float(np.linalg.norm(p - g)), float(dg[n0])
        pp = self.pos[self.path(n0, g)]
        ep = np.linalg.norm(pp - g, axis=1)
        arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pp, axis=0), axis=1))]
        below = np.nonzero(ep < e0 - EPS)[0]
        u = (g - p) / max(e0, 1e-9)
        rise = np.r_[e0, ep]
        out = {"e": round(e0, 4), "geo": round(g0, 4),
               "barrier": round(float((rise - np.minimum.accumulate(rise)).max()), 4),
               "above": round(float(ep.max() - e0), 4),
               "l_esc": round(float(arc[below[0]] if len(below) else arc[-1]), 4),
               "los_blocked": self.blocked(p, g), "clear": round(float(self.clearance(p)), 4),
               "pressed": bool(self.clearance(p) < 0.05 and self.clearance(p + 0.2 * u) <= 0)}
        nodes, d = self.ball(p)
        ek = np.linalg.norm(self.pos[nodes] - g, axis=1)
        for r in RS:
            m = d <= r
            q = nodes[m][np.argmin(ek[m])]
            prog = g0 - float(dg[q])
            out[f"prog{r}"], out[f"trap{r}"] = round(prog, 4), bool(prog < EPS)
        return out


def load_eps(level_dir):
    eps = collections.defaultdict(list)
    for f in sorted(level_dir.glob("shard*/per_iter_logs.json")):
        for line in f.open():
            if line.strip():
                r = json.loads(line)
                eps[r["sample_idx"]].append(r)
    return {i: sorted(rs, key=lambda r: r["step"]) for i, rs in eps.items()}


def episodes(a, base, goals):
    """Per episode: states by replan (0 = start), success replan, first trigger, layout."""
    out = []
    for i, rs in load_eps(ROOT / a.root / a.level / base / "ref").items():
        st = {0: np.asarray(goals["start_states"][i], dtype=np.float64)}
        st.update({r["step"]: np.asarray(r["state"], dtype=np.float64) for r in rs})
        succ = [r["step"] for r in rs if r["success"]]
        trig = [r["step"] for r in rs if r.get("stag_r") is not None and r["stag_r"] < a.gamma_hi]
        out.append({"i": i, "states": st, "T": max(st), "t_succ": succ[0] if succ else None,
                    "trig": trig[0] if trig else None, "goal": np.asarray(goals["goal_states"][i], dtype=np.float64),
                    "layout": int(goals["map_indices"][i]), "spec": goals["map_specs"][i],
                    "block_dist": int(goals["block_dists"][i])})
    return sorted(out, key=lambda e: e["i"])


def classify(ep, q, r=2):
    """Class of a failure from the per-replan geometry q (dict replan -> query): near the goal along the route; in an
    r-trap with a real rise on the route, at the first trigger or at the end; still progressing; or stuck in the open."""
    T, k = ep["T"], ep["trig"]
    if q[T]["geo"] < NEAR:
        return "near_miss"
    if any(s is not None and q[s][f"trap{r}"] and q[s]["barrier"] > EPS for s in (k, T)):
        return "trap"
    if q[max(0, T - WIN)]["geo"] - q[T]["geo"] > PROG:
        return "progressing"
    return "open"


def stage_geom(a, eps, geos, fout):
    for base in BASES:
        for ep in eps[base]:
            geo, xy = geos[ep["layout"]], ep["states"]
            q = {k: geo.query(s, ep["goal"]) for k, s in xy.items()}
            T, fail = ep["T"], ep["t_succ"] is None
            rec = {"base": base, "i": ep["i"], "layout": ep["layout"], "block_dist": ep["block_dist"],
                   "success": not fail, "t_succ": ep["t_succ"], "T": T, "trig": ep["trig"],
                   "start": q[0], "end": q[T], "trig_q": q[ep["trig"]] if ep["trig"] is not None else None,
                   "greedy": {r: geo.greedy(xy[0], ep["goal"], r) for r in RS},
                   "cls": classify(ep, q) if fail else None,
                   "cls_r": {r: classify(ep, q, r) for r in RS} if fail else None,
                   "moving_end": bool(np.linalg.norm(xy[T][:2] - xy[max(0, T - WIN)][:2]) > PROG),
                   "per_replan": {k: {kk: q[k][kk] for kk in ("e", "geo", "trap1", "trap2", "trap3", "barrier")}
                                  for k in sorted(q)}}
            fout.write(json.dumps(rec) + "\n")


def pursue(env, geo, s, g, steps, fs, kp, kd, look=0.4):
    """Privileged controller from full state s: env steps until success (checked every fs steps) or None."""
    env.set_state(np.asarray(s[:2], dtype=np.float64), np.asarray(s[2:4], dtype=np.float64))
    pos, vel = np.asarray(s[:2], dtype=np.float64), np.asarray(s[2:4], dtype=np.float64)
    for t in range(steps):
        pp = geo.pos[geo.path(geo.node(pos), g)]
        arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pp, axis=0), axis=1))]
        tgt = g if arc[-1] < look else pp[np.searchsorted(arc, look)]
        _, _, _, info = env.step(np.clip(kp * (tgt - pos) - kd * vel, -1.0, 1.0))
        pos, vel = np.asarray(info["state"][:2], dtype=np.float64), np.asarray(info["state"][2:4], dtype=np.float64)
        if (t + 1) % fs == 0 and np.linalg.norm(pos - g) < 0.5:
            return t + 1
    return None


def stage_sim(a, eps, geos, fout):
    import gym
    import env  # noqa: F401  (registers diverse_maze)

    envs = {}
    for base in BASES:
        for ep in eps[base]:
            if ep["layout"] not in envs:
                envs[ep["layout"]] = gym.make("diverse_maze", maze_spec=ep["spec"],
                                              action_repeat=a.action_repeat).unwrapped
            e, geo, g, T = envs[ep["layout"]], geos[ep["layout"]], ep["goal"][:2], a.max_iter
            rec = {"base": base, "i": ep["i"], "success": ep["t_succ"] is not None, "trig": ep["trig"]}
            t = pursue(e, geo, ep["states"][0], g, T * a.n_taken, a.n_taken, a.kp, a.kd)
            rec["from_start"] = None if t is None else t / a.n_taken
            if ep["t_succ"] is None and ep["trig"] is not None:
                k = ep["trig"]
                t = pursue(e, geo, ep["states"][k], g, (T - k) * a.n_taken, a.n_taken, a.kp, a.kd)
                rec["left"] = T - k
                rec["from_trig"] = None if t is None else t / a.n_taken
            fout.write(json.dumps(rec) + "\n")


def stage_latent(a, eps, geos, fout):
    import torch
    from stag_maze_diag import Maze, build

    wm, pre, _ = build(a.ckpt, a.data, a.device)
    mazes = {}
    stride = int(round(a.latent_step / geos[next(iter(geos))].h))
    for base in BASES:
        for ep in eps[base]:
            fail = ep["t_succ"] is None
            which = [("trig", ep["trig"]), ("end", ep["T"])] if fail else [("trig", ep["trig"])]
            which = [(w, k) for w, k in which if k is not None]
            if not which:
                continue
            if ep["layout"] not in mazes:
                mazes[ep["layout"]] = Maze(ep["spec"], wm, pre, a.device)
            mz, geo, g = mazes[ep["layout"]], geos[ep["layout"]], ep["goal"][:2]
            dg, _ = geo.from_goal(g)
            gv, gp = mz.encode(mz.render(g)[None], np.r_[g, 0.0, 0.0][None])
            for w, k in which:
                p = ep["states"][k][:2]
                nodes, d = geo.ball(p)
                ij = np.stack([np.rint((geo.pos[nodes, 0] - geo.x0) / geo.h), np.rint((geo.pos[nodes, 1] - geo.y0) / geo.h)], 1)
                keep = (ij % stride == 0).all(1)
                n0 = geo.node(p)
                keep |= nodes == n0
                nodes, d = nodes[keep], d[keep]
                pts = geo.pos[nodes]
                frames = np.stack([mz.render(x) for x in pts])
                zv, zp = mz.encode(frames, np.c_[pts, np.zeros((len(pts), 2))])
                with torch.no_grad():
                    c = (((zv - gv) ** 2).mean(1) + a.alpha * ((zp - gp) ** 2).mean(1)).cpu().numpy()
                ek = np.linalg.norm(pts - g, axis=1)
                g0 = float(dg[n0])
                rec = {"base": base, "i": ep["i"], "success": not fail, "which": w, "k": k, "n_pts": int(len(pts))}
                for r in RS:
                    m = d <= r
                    for name, f in (("lat", c), ("euc", ek)):
                        q = nodes[m][np.argmin(f[m])]
                        prog = g0 - float(dg[q])
                        rec[f"{name}_prog{r}"], rec[f"{name}_trap{r}"] = round(prog, 4), bool(prog < EPS)
                    if m.sum() > 2:
                        rk = lambda v: np.argsort(np.argsort(v))  # noqa: E731
                        rec[f"spearman{r}"] = round(float(np.corrcoef(rk(c[m]), rk(ek[m]))[0, 1]), 4)
                fout.write(json.dumps(rec) + "\n")


def fisher(t):
    from scipy.stats import fisher_exact
    return float(fisher_exact(t)[1])


def summarize(a, out_dir, rep_dir):
    def rd(stage):
        f = out_dir / f"{a.level}_{stage}.jsonl"
        return [json.loads(x) for x in f.open()] if f.exists() else []

    geom, sim, lat = rd("geom"), rd("sim"), rd("latent")
    simk = {(r["base"], r["i"]): r for r in sim}
    name = {"frozen": "冻结规划器", "adajepa": "AdaJEPA"}
    S, md = {}, [f"# 迷宫 {a.level}：失败是不是欧氏陷阱（E0 汇总）", ""]
    md += ["位置单位是迷宫格，成功要求离目标 < 0.5。r 陷阱：测地距离 r 以内欧氏距离最小的可达点，测地距离下降 < "
           f"{EPS}。屏障 > {EPS} 表示沿最短路径欧氏距离必须先升高。", ""]
    bmax = max((r["start"]["barrier"] for r in geom), default=0.0)
    n_bar = len({r["i"] for r in geom if r["start"]["barrier"] > EPS})
    intro = (f"起点最短路径上欧氏距离的最大升高只有 {bmax:.2f}（< {EPS}）：这一档没有一个起点需要先远离目标。"
             "因此按起点分组改用两种量：" if n_bar == 0 else
             f"{n_bar} 个起点的最短路径上欧氏距离要先升高 > {EPS}（最大 {bmax:.2f}）。按起点分组用三种量：屏障是否 > {EPS}；")
    md += ["## 1. 哪些 episode 更容易失败（按起点分组）", "",
           intro + "从起点做贪心欧氏下降（完美模型、可达半径 r，每次跳到 r 以内欧氏距离最小的点）"
           "能否到达目标；最短路径比直线长多少。", "",
           "| 基准 | 分组 | 组内 episode | 组内失败 | 组外失败 | Fisher p |", "|---|---|---:|---:|---:|---:|"]
    groups = [(f"起点屏障 > {EPS}", lambda x: x["start"]["barrier"] > EPS)] if n_bar else []
    groups += [(f"贪心欧氏下降（r={r}）到不了目标", lambda x, r=r: not x["greedy"][str(r)][0]) for r in RS]
    groups.append(("最短路径比直线长 > 0.5", lambda x: x["start"]["geo"] - x["start"]["e"] > 0.5))
    for base in BASES:
        G = [r for r in geom if r["base"] == base]
        S[base] = {"n": len(G), "fail": sum(not r["success"] for r in G), "groups": {}}
        for gname, sel in groups:
            ins, out = [r for r in G if sel(r)], [r for r in G if not sel(r)]
            fi, fo = sum(not r["success"] for r in ins), sum(not r["success"] for r in out)
            p = fisher([[fi, len(ins) - fi], [fo, len(out) - fo]])
            S[base]["groups"][gname] = {"n_in": len(ins), "fail_in": fi, "n_out": len(out), "fail_out": fo, "fisher_p": p}
            md.append(f"| {name[base]} | {gname} | {len(ins)} | {fi}/{len(ins)} | {fo}/{len(out)} | {p:.3g} |")
    md += ["", "## 2. 失败分类", "",
           "依次判断：差一点 = 最后离目标的测地距离 < 1；陷阱 = 第一次触发时或最后处在 2 陷阱，且最短路径上欧氏距离要升高 > 0.25；"
           "还在推进 = 最后 5 次重规划测地距离下降 > 0.3；其余为卡在开阔处。", "",
           "| 基准 | 失败 | 差一点 | 欧氏陷阱 | 还在推进 | 卡在开阔处 | 最后被墙隔开（测地 − 欧氏 > 1）且贴墙朝向目标 | 陷阱个数：1 陷阱 / 3 陷阱 |",
           "|---|---:|---:|---:|---:|---:|---:|---|"]
    for base in BASES:
        F = [r for r in geom if r["base"] == base and not r["success"]]
        c = collections.Counter(r["cls"] for r in F)
        sep = sum(r["end"]["geo"] - r["end"]["e"] > 1.0 and r["end"]["pressed"] for r in F)
        t1, t3 = (sum(r["cls_r"][str(k)] == "trap" for r in F) for k in (1, 3))
        S[base]["cls"] = dict(c)
        S[base]["trap_r1"], S[base]["trap_r3"], S[base]["end_wall_separated_pressed"] = t1, t3, sep
        md.append(f"| {name[base]} | {len(F)} | {c['near_miss']} | {c['trap']} | {c['progressing']} | {c['open']} | "
                  f"{sep}/{len(F)} | {t1} / {t3} |")
    md += ["", "## 3. 第一次卡住触发时", "",
           "| 基准 | 失败中触发过 | 触发时在 2 陷阱 | 后来成功中触发过 | 其中触发时在 2 陷阱 |", "|---|---:|---:|---:|---:|"]
    for base in BASES:
        G = [r for r in geom if r["base"] == base and r["trig_q"] is not None]
        F, Sx = [r for r in G if not r["success"]], [r for r in G if r["success"]]
        ft, st = sum(r["trig_q"]["trap2"] for r in F), sum(r["trig_q"]["trap2"] for r in Sx)
        S[base].update({"trig_fail": len(F), "trig_fail_trap2": ft, "trig_succ": len(Sx), "trig_succ_trap2": st})
        md.append(f"| {name[base]} | {len(F)} | {ft} | {len(Sx)} | {st} |")
    md += ["", "## 4. 成功的 episode 是否也进过陷阱（对照）", "",
           "| 基准 | 成功 | 曾在 2 陷阱 | 失败 | 曾在 2 陷阱 | 失败的重规划中处在 2 陷阱的比例（中位数） |", "|---|---:|---:|---:|---:|---:|"]
    for base in BASES:
        G = [r for r in geom if r["base"] == base]
        vis = lambda r: any(v["trap2"] for k, v in r["per_replan"].items() if int(k) > 0)  # noqa: E731
        Sx, F = [r for r in G if r["success"]], [r for r in G if not r["success"]]
        frac = [np.mean([v["trap2"] for k, v in r["per_replan"].items() if int(k) > 0]) for r in F]
        S[base].update({"succ_visit_trap2": sum(map(vis, Sx)), "fail_visit_trap2": sum(map(vis, F)),
                        "fail_trap2_frac_median": float(np.median(frac)) if frac else None})
        md.append(f"| {name[base]} | {len(Sx)} | {S[base]['succ_visit_trap2']} | {len(F)} | {S[base]['fail_visit_trap2']} | "
                  f"{np.median(frac) if frac else float('nan'):.2f} |")
    if sim:
        md += ["", "## 5. 剩余预算内能否走完（有特权的脚本控制器）", "",
               "| 基准 | 从起点（检查控制器） | 从起点所需重规划（中位数） | 失败中从触发处可解 | 其中卡在 2 陷阱 | 卡在开阔处 | 差一点 | 还在推进 |",
               "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for base in BASES:
            R = [r for r in sim if r["base"] == base]
            ok = [r["from_start"] for r in R if r["from_start"] is not None]
            F = [r for r in geom if r["base"] == base and not r["success"] and r["trig"] is not None]
            cnt = collections.defaultdict(lambda: [0, 0])
            for r in F:
                s_ = simk.get((base, r["i"]), {})
                cnt[r["cls"]][1] += 1
                cnt[r["cls"]][0] += s_.get("from_trig") is not None
            tot = [sum(v[0] for v in cnt.values()), sum(v[1] for v in cnt.values())]
            S[base]["sim"] = {"from_start_ok": len(ok), "n": len(R), "from_start_median": float(np.median(ok)) if ok else None,
                              "from_trig": {k: v for k, v in cnt.items()}, "from_trig_total": tot}
            f = lambda k: f"{cnt[k][0]}/{cnt[k][1]}"  # noqa: E731
            md.append(f"| {name[base]} | {len(ok)}/{len(R)} | {np.median(ok) if ok else float('nan'):.0f} | {tot[0]}/{tot[1]} | "
                      f"{f('trap')} | {f('open')} | {f('near_miss')} | {f('progressing')} |")
        for base in BASES:
            F = [r for r in geom if r["base"] == base and not r["success"]]
            tf = sum(r["cls"] == "trap" and simk.get((base, r["i"]), {}).get("from_trig") is not None for r in F)
            S[base]["trap_and_feasible"] = tf
            md.append("")
            md.append(f"- {name[base]}：卡在 2 陷阱且从触发处可解 **{tf}/{len(F)}**（{tf / max(len(F), 1):.0%}）")
    if lat:
        md += ["", "## 6. 规划器自己的潜代价是否也是陷阱（初始模型，零速度渲染）", "",
               "| 基准 | 状态 | 个数 | 欧氏 2 陷阱 | 潜代价 2 陷阱 | 两者一致 | 1 / 3 陷阱一致 | 潜代价与欧氏距离的 Spearman（r=2，中位数） |",
               "|---|---|---:|---:|---:|---:|---|---:|"]
        groups = (("失败·触发时", lambda r: not r["success"] and r["which"] == "trig"),
                  ("失败·最后", lambda r: not r["success"] and r["which"] == "end"),
                  ("后来成功·触发时", lambda r: r["success"]))
        S["latent"] = {}
        for base in BASES:
            for gname, sel in groups:
                L = [r for r in lat if r["base"] == base and sel(r)]
                if not L:
                    continue
                agree = lambda r_: sum(x[f"lat_trap{r_}"] == x[f"euc_trap{r_}"] for x in L)  # noqa: E731
                sp = [x["spearman2"] for x in L if "spearman2" in x]
                S["latent"][f"{base}/{gname}"] = {"n": len(L), "euc_trap2": sum(x["euc_trap2"] for x in L),
                                                  "lat_trap2": sum(x["lat_trap2"] for x in L), "agree2": agree(2),
                                                  "agree1": agree(1), "agree3": agree(3),
                                                  "spearman2_median": float(np.median(sp)) if sp else None}
                md.append(f"| {name[base]} | {gname} | {len(L)} | {S['latent'][f'{base}/{gname}']['euc_trap2']} | "
                          f"{S['latent'][f'{base}/{gname}']['lat_trap2']} | {agree(2)}/{len(L)} | {agree(1)}/{len(L)} · {agree(3)}/{len(L)} | "
                          f"{np.median(sp) if sp else float('nan'):.2f} |")
    rep_dir.mkdir(parents=True, exist_ok=True)
    (rep_dir / f"summary_{a.level}.md").write_text("\n".join(md) + "\n")
    (rep_dir / f"summary_{a.level}.json").write_text(json.dumps(S, indent=1, default=float))
    print("\n".join(md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="+", default=["geom", "sim", "summary"], choices=["geom", "sim", "latent", "summary"])
    ap.add_argument("--level", choices=list(LEVELS), default="easy")
    ap.add_argument("--root", default="eval_outputs/stag_maze100")
    ap.add_argument("--data", default="data/diverse_maze")
    ap.add_argument("--ckpt", default="checkpoints/diversemaze_layout_shift")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--gamma-hi", type=float, default=0.15)
    ap.add_argument("--max-iter", type=int, default=50)
    ap.add_argument("--n-taken", type=int, default=5, help="actions per replan (n_taken_actions)")
    ap.add_argument("--action-repeat", type=int, default=1, help="env steps per action, as in the runs' env_kwargs_override")
    ap.add_argument("--kp", type=float, default=10.0)
    ap.add_argument("--kd", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=1.0, help="objective.alpha of the plan config")
    ap.add_argument("--latent-step", type=float, default=0.2, help="spacing of the rendered points (maze units)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="eval_outputs/stag_maze_trap")
    ap.add_argument("--report", default="reports/stag_maze_trap")
    a = ap.parse_args()

    from datasets.diverse_maze_goals import generate_diverse_maze_goals

    lo, hi = LEVELS[a.level]
    goals = generate_diverse_maze_goals(val_data_path=str(ROOT / a.data / "val"), n_evals=a.n, seed=a.seed,
                                        min_block_radius=lo, max_block_radius=hi)
    eps = {base: episodes(a, base, goals) for base in BASES}
    geos = {}
    for base in BASES:
        for ep in eps[base]:
            if ep["layout"] not in geos:
                geos[ep["layout"]] = Geo(ep["spec"])
    out_dir = ROOT / a.out
    out_dir.mkdir(parents=True, exist_ok=True)
    for stage, fn in (("geom", stage_geom), ("sim", stage_sim), ("latent", stage_latent)):
        if stage in a.stages:
            with (out_dir / f"{a.level}_{stage}.jsonl").open("w") as fout:
                fn(a, eps, geos, fout)
    if "summary" in a.stages:
        summarize(a, out_dir, ROOT / a.report)


if __name__ == "__main__":
    main()
