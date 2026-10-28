"""D0: offline model-or-ruler attribution from the logs of scripts/attr_run.py (reports/paper2_positioning/d0_plan.md).

    --check    every (episode, replan) of a run against its reference run: executed actions, success and the stall
               trigger's ratio must be identical, and every planned replan must carry the attribution fields
    --labels   privileged labels per replan, for evaluation only (maze runs; GPU): eval_outputs/attr_d0/labels_<run>.jsonl
               - the position the model predicted for the executed step (proprio read-out of attr_zp1, as in
                 scripts/stag_maze_diag.py) and the geodesic distances to the goal of the replan's start, end and
                 predicted end position (scripts/stag_maze_trap.py's grid; a point in a wall takes its nearest free node)
               - whether the end position is an r-trap of the planner's own latent cost: the cheapest rendered point
                 within geodesic radius r makes < EPS geodesic progress (initial encoder, zero velocity, as E0 does),
                 and the same test for the Euclidean distance
    --analyze  windows, verdicts, labels and the D0 criteria -> reports/attr_d0/<name>.{md,json}

Window = the 3 replans the stall trigger looks at: steps k-2..k, where record k carries the trigger's ratio. Per replan,
with the planner's per-frame cost C = cv + alpha cp:

    p = (c - C(zh_1)) / c      promise of the executed step
    d = (c - C(z_next)) / c    delivery
    h = (c - min_h C(zh_h)) / c   headroom of the executed plan

and window values are their means. Verdicts (d0_plan.md 2.2), in this order:

    M  p >= p_min and d < tau p                 the model promised, reality did not deliver
    R  rule "plan":  h < h_min and |d| < p_min   the plan sees nothing better and nothing happens
       rule "local": |p| < p_min and |d| < p_min the executed step promises nothing and delivers nothing
    N  p >= p_min and d >= tau p                 the promise was kept
    U  otherwise
    S  an R window whose restart check found a plan with an objective lower than the executed one by >= h_min
       (relative to the executed plan's objective)

Labels (evaluation only): M* = the model over-promised geodesic progress by > M_OVER cells summed over the window;
R* = not M* and the window's end position is a latent-cost 2-trap; progress = not M*, >= PROG cells of geodesic
progress over the window and no success inside it. PushObj: R* = the first trigger of a failure that the O1 oracle
(simulator + the same latent objective, CEM) could not solve either.

    python scripts/attr_d0.py --check --runs A1 A2
    CUDA_VISIBLE_DEVICES=0 python scripts/attr_d0.py --labels --runs A1 A2
    python scripts/attr_d0.py --analyze --runs A1 A2 --name dev
"""
import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from attr_run import MAZE, MAZE_N, MAZE_SEED, REFS, ROOT  # noqa: E402
from stag_maze_base_run import LEVELS  # noqa: E402

sys.path.insert(0, str(ROOT))

ATTR_FIELDS = ("attr_cv", "attr_cp", "attr_nv", "attr_np", "attr_J", "attr_zp1")
W = 3
ALPHA = 1.0
DEFAULT = {"tau": 0.5, "p_min": 0.02, "h_min": 0.05}  # prior thresholds (d0_plan.md 2.3)
GRID = {"tau": (0.3, 0.5, 0.7), "p_min": (0.01, 0.02, 0.05), "h_min": (0.02, 0.05, 0.1)}
M_OVER, PROG, TRAP_R = 0.5, 0.5, 2  # label settings (maze cells); sensitivity in the report
O1_PATH = "eval_outputs/stag_EVAL50/oracle/O1.jsonl"
E0 = {"A2": "eval_outputs/stag_maze_trap_ar4/easy_geom.jsonl", "A4": "eval_outputs/stag_maze_trap_ar4/medium_geom.jsonl"}


# ---------------------------------------------------------------------- logs
def records(run_dir, maze):
    """{(chunk, sample_idx, step): record} of all shards / chunks under run_dir."""
    out = {}
    for f in sorted(Path(run_dir).glob("*/per_iter_logs.json")):
        chunk = None if maze else f.parent.name
        for line in f.open():
            if line.strip():
                r = json.loads(line)
                out[(chunk, r["sample_idx"], r["step"])] = r
    return out


def episodes(run_dir, maze):
    eps = collections.defaultdict(list)
    for (c, i, _), r in records(run_dir, maze).items():
        eps[(c, i)].append(r)
    return {k: sorted(v, key=lambda r: r["step"]) for k, v in sorted(eps.items(), key=lambda kv: (kv[0][0] or "", kv[0][1]))}


def check(run, root):
    maze = run in MAZE
    new, ref = records(ROOT / root / run, maze), records(ROOT / REFS[run], maze)
    bad, missing_attr = [], 0
    for k, r in new.items():
        q = ref.get(k)
        if q is None:
            bad.append((k, "not in reference"))
            continue
        for f in ("act", "success", "stag_r"):
            if (f in q or f != "stag_r") and r.get(f) != q.get(f):  # escape-planner references log no stag_r
                bad.append((k, f))
        missing_attr += any(f not in r for f in ATTR_FIELDS)
    eps = {(c, i) for c, i, _ in new}
    n_rs = sum("attr_rs" in r for r in new.values())
    print(f"{run}: {len(eps)} episodes, {len(new)} replans; mismatches {len(bad)}; "
          f"replans without attribution fields {missing_attr}; restart checks {n_rs}")
    for k, f in bad[:10]:
        print("   ", k, f)
    return not bad and not missing_attr


# ---------------------------------------------------------------------- labels (maze)
def maze_labels(run, a):
    import torch
    from datasets.diverse_maze_goals import generate_diverse_maze_goals
    from stag_maze_diag import Maze, ProprioReadout, build
    from stag_maze_trap import EPS, RS, Geo

    level = MAZE[run][0]
    lo, hi = LEVELS[level]
    goals = generate_diverse_maze_goals(val_data_path=str(ROOT / a.maze_data / "val"), n_evals=MAZE_N, seed=MAZE_SEED,
                                        min_block_radius=lo, max_block_radius=hi)
    eps = episodes(ROOT / a.root / run, maze=True)
    wm, pre, _ = build(a.maze_ckpt, a.maze_data, a.device)
    ro = ProprioReadout(wm, pre, a.device)
    flat = [(i, r) for (_, i), rs in eps.items() for r in rs]
    zp = torch.tensor([r["attr_zp1"] for _, r in flat], dtype=torch.float32)
    pred, res = [], []
    for s in range(0, len(zp), 1024):
        p, e = ro.decode(zp[s:s + 1024], residual=True)
        pred.append(p)
        res.append(e)
    pred, res = np.concatenate(pred), np.concatenate(res)

    geos, mazes, rmax = {}, {}, max(RS)
    rows, cand = [], collections.defaultdict(set)  # cand: layout -> nodes whose latent cost is needed
    for k, (i, r) in enumerate(flat):
        lay = int(goals["map_indices"][i])
        if lay not in geos:
            geos[lay] = Geo(goals["map_specs"][i])
        geo = geos[lay]
        stride = int(round(a.latent_step / geo.h))
        g = np.asarray(goals["goal_states"][i][:2], dtype=np.float64)
        dg, _ = geo.from_goal(g)
        prev = [x for _, x in flat[max(0, k - 1):k] if x["sample_idx"] == i and x["step"] == r["step"] - 1]
        s0 = np.asarray(prev[0]["state"][:2] if prev else goals["start_states"][i][:2], dtype=np.float64)
        s1, sp = np.asarray(r["state"][:2], dtype=np.float64), pred[k][:2]
        n1 = geo.node(s1)
        nodes, dist = geo.ball(s1, rmax)
        ij = np.stack([np.rint((geo.pos[nodes, 0] - geo.x0) / geo.h), np.rint((geo.pos[nodes, 1] - geo.y0) / geo.h)], 1)
        keep = (ij % stride == 0).all(1) | (nodes == n1)
        nodes, dist = nodes[keep], dist[keep]
        cand[lay].update(nodes.tolist())
        ek = np.linalg.norm(geo.pos[nodes] - g, axis=1)
        row = {"i": i, "step": r["step"], "layout": lay, "dg_s": float(dg[geo.node(s0)]), "dg_e": float(dg[n1]),
               "dg_p": float(dg[geo.node(sp)]), "res": float(res[k]), "pred": [round(float(x), 4) for x in pred[k]],
               "err": float(np.linalg.norm(sp - s1))}
        for rr in RS:
            m = dist <= rr
            row[f"euc_prog{rr}"] = float(dg[n1] - dg[nodes[m][np.argmin(ek[m])]])
        rows.append((row, nodes, dist, g))

    # latent cost of every candidate node: render (zero velocity) and encode once per layout, initial encoder
    zcache = {}
    for lay, nodes in cand.items():
        i0 = next(i for i, _ in flat if int(goals["map_indices"][i]) == lay)
        if lay not in mazes:
            mazes[lay] = Maze(goals["map_specs"][i0], wm, pre, a.device)
        nodes = np.array(sorted(nodes))
        pts = geos[lay].pos[nodes]
        zv, zp_ = [], []
        for s in range(0, len(pts), 256):
            fr = np.stack([mazes[lay].render(x) for x in pts[s:s + 256]])
            v, p = mazes[lay].encode(fr, np.c_[pts[s:s + 256], np.zeros((len(fr), 2))])
            zv.append(v.cpu())
            zp_.append(p.cpu())
        zcache[lay] = (nodes, torch.cat(zv), torch.cat(zp_))
        print(f"{run}: layout {lay}: {len(nodes)} rendered points", flush=True)
    cost = {}  # (layout, goal) -> {node: latent cost}
    for row, nodes, dist, g in rows:
        key = (row["layout"], tuple(np.round(g, 6)))
        if key not in cost:
            lay = row["layout"]
            nn, zv, zp_ = zcache[lay]
            fr = mazes[lay].render(g)[None]
            gv, gp = mazes[lay].encode(fr, np.r_[g, 0.0, 0.0][None])
            with torch.no_grad():
                c = (((zv.to(gv.device) - gv) ** 2).mean(1) + ALPHA * ((zp_.to(gp.device) - gp) ** 2).mean(1)).cpu().numpy()
            cost[key] = dict(zip(nn.tolist(), c))
        geo = geos[row["layout"]]
        dg, _ = geo.from_goal(g)
        c = np.array([cost[key][n] for n in nodes.tolist()])
        for rr in RS:
            m = dist <= rr
            row[f"lat_prog{rr}"] = float(row["dg_e"] - dg[nodes[m][np.argmin(c[m])]])
        row["eps"] = EPS
    out = ROOT / a.root / f"labels_{run}.jsonl"
    with out.open("w") as f:
        for row, *_ in rows:
            f.write(json.dumps(row) + "\n")
    print(f"{run}: {len(rows)} replans labelled -> {out.relative_to(ROOT)}", flush=True)


# ---------------------------------------------------------------------- windows
def rec_q(r):
    C = [v + ALPHA * p for v, p in zip(r["attr_cv"], r["attr_cp"])]
    c, n = C[0], r["attr_nv"] + ALPHA * r["attr_np"]
    cp0, cp1, npp = r["attr_cp"][0], r["attr_cp"][1], r["attr_np"]
    q = {"c": c, "p": (c - C[1]) / c, "d": (c - n) / c, "h": (c - min(C[1:])) / c,
         "pp": (cp0 - cp1) / cp0, "dp": (cp0 - npp) / cp0, "mse": r.get("loss_new_cur")}
    rs = r.get("attr_rs")
    if rs:
        q["s_gain"] = (rs["J_exec"] - min(rs["J"])) / max(rs["J_exec"], 1e-12)
    if "attr_c0_init" in r:
        c0 = r["attr_c0_init"][0] + ALPHA * r["attr_c0_init"][1]
        q["drift"] = abs(c - c0) / c0
    return q


def windows(run, a):
    """All 3-replan windows of a run with their quantities and labels."""
    maze = run in MAZE
    eps = episodes(ROOT / a.root / run, maze)
    lab = {}
    if maze:
        for line in (ROOT / a.root / f"labels_{run}.jsonl").open():
            x = json.loads(line)
            lab[(x["i"], x["step"])] = x
    o1 = {}
    if not maze:
        for line in (ROOT / O1_PATH).open():
            x = json.loads(line)
            o1[(x["chunk"], x["sample_idx"])] = x
    out = []
    for (chunk, i), rs in eps.items():
        by = {r["step"]: r for r in rs}
        steps = sorted(by)
        T = steps[-1]
        succ = any(r["success"] for r in rs)
        trig = [s for s in steps if (by[s].get("stag_w") or 0) > 0]
        first = trig[0] if trig else None
        for k in steps:
            if not all(s in by for s in (k - 2, k - 1, k)):
                continue
            ws = [by[s] for s in (k - 2, k - 1, k)]
            qs = [rec_q(r) for r in ws]
            w = {"run": run, "chunk": chunk, "i": i, "k": k, "T": T, "ep_success": succ,
                 "trig": (by[k].get("stag_w") or 0) > 0, "first": k == first, "last": k == T,
                 "any_success": any(r["success"] for r in ws)}
            for f in ("p", "d", "h", "pp", "dp"):
                w[f] = float(np.mean([q[f] for q in qs]))
            m = [q["mse"] for q in qs if q["mse"] is not None]
            w["mse"] = float(np.mean(m)) if m else None
            sg = [q["s_gain"] for q in qs if "s_gain" in q]
            w["s_gain"] = max(sg) if sg else None
            dr = [q["drift"] for q in qs if "drift" in q]
            w["drift"] = max(dr) if dr else None
            if maze:
                ls = [lab[(i, s)] for s in (k - 2, k - 1, k)]
                w["over"] = sum(x["dg_e"] - x["dg_p"] for x in ls)
                w["act_prog"] = ls[0]["dg_s"] - ls[-1]["dg_e"]
                w["dg_end"] = ls[-1]["dg_e"]
                for rr in (1, 2, 3):
                    w[f"lat_trap{rr}"] = ls[-1][f"lat_prog{rr}"] < ls[-1]["eps"]
                    w[f"euc_trap{rr}"] = ls[-1][f"euc_prog{rr}"] < ls[-1]["eps"]
            else:
                x = o1.get((chunk, i))
                w["o1"] = None if x is None or x["start_step"] != k else {
                    "solved": bool(x["success"]), "pusher_only": x["obj_err0"] < 20 and x["ang0_deg"] < 20}
                d4 = [by[s].get("dist4") for s in (k - 3, k) if s in by]
                w["d4_prog"] = d4[0] - d4[1] if len(d4) == 2 else None
            out.append(w)
    return out


def label(w, m_over=M_OVER, trap_r=TRAP_R, prog=PROG):
    """Privileged class of a maze window: M*, R*, progress or other (None for PushObj windows)."""
    if "over" not in w:
        return None
    if w["over"] > m_over:
        return "M*"
    if w[f"lat_trap{trap_r}"] and w["trig"]:
        return "R*"
    if w["act_prog"] >= prog and not w["any_success"]:
        return "prog"
    return "other"


def verdict(w, tau, p_min, h_min, rule="plan", restart=True):
    p, d, h = w["p"], w["d"], w["h"]
    if p >= p_min and d < tau * p:
        return "M"
    flat = (h < h_min) if rule == "plan" else (abs(p) < p_min)
    if flat and abs(d) < p_min:
        if restart and w["s_gain"] is not None and w["s_gain"] >= h_min:
            return "S"
        return "R"
    if p >= p_min:
        return "N"
    return "U"


# ---------------------------------------------------------------------- statistics
def frac(ws, ok, n_boot=2000, seed=0):
    """Fraction of windows with ok(w), and a 95% bootstrap interval over episodes."""
    if not ws:
        return {"k": 0, "n": 0, "f": None, "lo": None, "hi": None}
    by = collections.defaultdict(list)
    for w in ws:
        by[(w["run"], w["chunk"], w["i"])].append(bool(ok(w)))
    groups = list(by.values())
    k, n = sum(sum(g) for g in groups), sum(len(g) for g in groups)
    if not n_boot:
        return {"k": int(k), "n": int(n), "f": k / n, "lo": None, "hi": None}
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        pick = rng.integers(len(groups), size=len(groups))
        kk = sum(sum(groups[j]) for j in pick)
        nn = sum(len(groups[j]) for j in pick)
        bs.append(kk / nn)
    return {"k": int(k), "n": int(n), "f": k / n, "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5))}


def auc(pos, neg):
    """P(score of a positive > score of a negative), ties count half."""
    pos, neg = np.asarray([x for x in pos if x is not None]), np.asarray([x for x in neg if x is not None])
    if len(pos) == 0 or len(neg) == 0:
        return None
    allv = np.concatenate([pos, neg])
    ranks = np.argsort(np.argsort(allv, kind="mergesort"), kind="mergesort") + 1.0
    for v in np.unique(allv):  # average ranks of ties
        m = allv == v
        ranks[m] = ranks[m].mean()
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def fmt(x):
    if x["f"] is None:
        return "—"
    return f"{x['k']}/{x['n']} = {100 * x['f']:.0f}% [{100 * x['lo']:.0f}, {100 * x['hi']:.0f}]"


def q3(v):
    v = [x for x in v if x is not None]
    if not v:
        return "—"
    a, b, c = np.percentile(v, [25, 50, 75])
    return f"{b:.3f} ({a:.3f}–{c:.3f})"


def traps(run):
    """E0's trap failures of a run: {episode: window step} (first trigger if the trap is there, else the end)."""
    if run not in E0:
        return {}
    out = {}
    for line in (ROOT / E0[run]).open():
        x = json.loads(line)
        if x["base"] == "adajepa" and not x["success"] and x["cls"] == "trap":
            q = x["trig_q"]
            at_trig = x["trig"] is not None and q is not None and q["trap2"] and q["barrier"] > 0.25
            out[x["i"]] = x["trig"] if at_trig else x["T"]
    return out


# ---------------------------------------------------------------------- report
def analyze(a):
    W_ = {run: windows(run, a) for run in a.runs}
    th = dict(DEFAULT)
    th.update({k: v for k, v in (("tau", a.tau), ("p_min", a.p_min), ("h_min", a.h_min)) if v is not None})
    L, S = [], {"thresholds": th, "runs": {}}
    L.append(f"# D0 判别器离线检验：{a.name}\n")
    L.append(f"阈值 τ = {th['tau']}，p_min = {th['p_min']}，h_min = {th['h_min']}。标签：M\\* 高估测地推进 > {M_OVER} 格；"
             f"R\\* = 非 M\\*、触发窗口、终点是潜代价 {TRAP_R} 陷阱；推进 = 非 M\\*、窗口内测地推进 ≥ {PROG} 格、窗口内未成功。"
             "区间是按 episode 做 bootstrap 的 95% 区间。\n")
    for run, ws in W_.items():
        maze = run in MAZE
        S["runs"][run] = R = {}
        trig = [w for w in ws if w["trig"]]
        L.append(f"## {run}\n")
        n_ep = len({(w['chunk'], w['i']) for w in ws})
        L.append(f"{n_ep} 个 episode，{len(ws)} 个窗口，其中触发窗口 {len(trig)} 个。\n")
        # distributions
        L.append("| 窗口 | 个数 | p̄ | d̄ | h̄ | d̄/p̄（p̄ ≥ p_min） |")
        L.append("|---|---:|---|---|---|---|")
        groups = [("触发·失败", [w for w in trig if not w["ep_success"]]),
                  ("触发·后来成功", [w for w in trig if w["ep_success"]])]
        if maze:
            for lb in ("M*", "R*", "prog", "other"):
                groups.append((f"标签 {lb}", [w for w in ws if label(w) == lb]))
        else:
            groups.append(("O1 解不开的首次触发", [w for w in ws if w["o1"] and not w["o1"]["solved"]]))
            groups.append(("O1 解开的首次触发", [w for w in ws if w["o1"] and w["o1"]["solved"]]))
            groups.append(("未触发、联合误差下降 ≥ 10", [w for w in ws if not w["trig"] and (w["d4_prog"] or 0) >= 10]))
        for name, g in groups:
            L.append(f"| {name} | {len(g)} | {q3([w['p'] for w in g])} | {q3([w['d'] for w in g])} | {q3([w['h'] for w in g])} | "
                     f"{q3([w['d'] / w['p'] for w in g if w['p'] >= th['p_min']])} |")
        L.append("")
        R["dist"] = {name: {"n": len(g), **{f: float(np.median([w[f] for w in g])) if g else None for f in ("p", "d", "h")}}
                     for name, g in groups}
        # verdicts
        for rule in ("plan", "local"):
            L.append(f"判决（R 规则：{rule}）：\n")
            cats = ("M", "R", "S", "N", "U")
            L.append("| 窗口 | " + " | ".join(cats) + " |")
            L.append("|---|" + "---:|" * len(cats))
            for name, g in groups:
                c = collections.Counter(verdict(w, rule=rule, **th) for w in g)
                L.append(f"| {name} | " + " | ".join(str(c.get(x, 0)) for x in cats) + " |")
            L.append("")
        # AUCs (maze)
        if maze:
            mpos = [w for w in ws if label(w) == "M*"]
            mneg = [w for w in ws if label(w) in ("R*", "prog", "other")]
            rpos = [w for w in trig if label(w) == "R*"]
            rneg = [w for w in trig if label(w) != "R*"]
            au = {"M: p̄ − d̄": auc([w["p"] - w["d"] for w in mpos], [w["p"] - w["d"] for w in mneg]),
                  "M: 本体 p̄ − d̄": auc([w["pp"] - w["dp"] for w in mpos], [w["pp"] - w["dp"] for w in mneg]),
                  "M: 总潜空间单步误差": auc([w["mse"] for w in mpos], [w["mse"] for w in mneg]),
                  "R: −h̄（触发窗口）": auc([-w["h"] for w in rpos], [-w["h"] for w in rneg]),
                  "R: −(|p̄| + |d̄|)（触发窗口）": auc([-abs(w["p"]) - abs(w["d"]) for w in rpos],
                                                   [-abs(w["p"]) - abs(w["d"]) for w in rneg])}
            R["auc"] = au
            L.append(f"AUC（M\\* {len(mpos)} 对 非 M\\* {len(mneg)}；R\\* {len(rpos)} 对 其余触发窗口 {len(rneg)}）：" +
                     "；".join(f"{k} {v:.2f}" if v is not None else f"{k} —" for k, v in au.items()) + "\n")
            dr = [w["drift"] for w in ws if w["drift"] is not None]
            if dr:
                L.append(f"编码器漂移（初始和决策时编码器下 c 的相对差，窗口内最大）：中位 {np.median(dr):.4f}，"
                         f"95 分位 {np.percentile(dr, 95):.4f}，最大 {max(dr):.4f}。\n")
                R["drift"] = [float(np.median(dr)), float(np.percentile(dr, 95)), float(max(dr))]
            prog = [w for w in ws if label(w) == "prog" and w["p"] >= th["p_min"]]
            if prog:
                m = float(np.median([w["d"] / w["p"] for w in prog]))
                R["optimism_m"] = m
                L.append(f"乐观偏差：推进窗口（{len(prog)} 个）上 d̄/p̄ 的中位数 m = {m:.2f}。\n")
            ss = [w["s_gain"] for w in trig if w["s_gain"] is not None]
            if ss:
                L.append(f"重启检查（触发窗口里做过的 {len(ss)} 个）：最好的随机重启比执行的计划目标低的相对幅度，"
                         f"中位 {np.median(ss):.3f}，≥ h_min 的 {sum(x >= th['h_min'] for x in ss)} 个。\n")
    # criteria (and their dev analogues)
    L.append("## 标准\n")
    crit = {}
    for rule in ("plan", "local"):
        v = lambda w, rule=rule: verdict(w, rule=rule, **th)  # noqa: E731
        rows = []
        for run in a.runs:
            ws = W_[run]
            if run in MAZE:
                fm = [w for w in ws if w["first"] and label(w) == "M*"]
                rows.append((f"{run}：带 M\\* 的首次触发窗口判 M（C-a 型）", frac(fm, lambda w: v(w) == "M")))
                pr = [w for w in ws if label(w) == "prog"]
                rows.append((f"{run}：推进窗口判 M 或 R（C-c 型，误报）", frac(pr, lambda w: v(w) in ("M", "R"))))
                tp = traps(run)
                tw = [w for w in ws if w["i"] in tp and w["k"] == tp[w["i"]]]
                rows.append((f"{run}：E0 陷阱失败判 R（C-b 型）", frac(tw, lambda w: v(w) == "R")))
            else:
                ru = [w for w in ws if w["o1"] and not w["o1"]["solved"]]
                rows.append((f"{run}：O1 解不开的首次触发判 R（C-e 型）", frac(ru, lambda w: v(w) == "R")))
        mz = [w for run in a.runs if run in MAZE for w in W_[run] if w["trig"] and v(w) == "R"]
        rows.append(("迷宫各运行：判 R 的触发窗口中带 M\\*（C-d 型，纯度）", frac(mz, lambda w: label(w) == "M*")))
        crit[rule] = {name: x for name, x in rows}
        L.append(f"R 规则：{rule}\n")
        L.append("| 检验 | 结果 |")
        L.append("|---|---|")
        for name, x in rows:
            L.append(f"| {name} | {fmt(x)} |")
        L.append("")
    S["criteria"] = crit
    # sensitivity
    L.append("## 阈值敏感性（R 规则：plan / local）\n")
    L.append("| τ | p_min | h_min | C-a 型 | C-c 型 | C-d 型（plan / local） |")
    L.append("|---:|---:|---:|---|---|---|")
    maze_runs = [r for r in a.runs if r in MAZE]
    for tau in GRID["tau"]:
        for p_min in GRID["p_min"]:
            for h_min in GRID["h_min"]:
                t = {"tau": tau, "p_min": p_min, "h_min": h_min}
                fm = [w for r in maze_runs for w in W_[r] if w["first"] and label(w) == "M*"]
                pr = [w for r in maze_runs for w in W_[r] if label(w) == "prog"]
                ca = frac(fm, lambda w: verdict(w, **t) == "M", n_boot=0) if fm else None
                cc = frac(pr, lambda w: verdict(w, **t) in ("M", "R"), n_boot=0) if pr else None
                cd = []
                for rule in ("plan", "local"):
                    mz = [w for r in maze_runs for w in W_[r] if w["trig"] and verdict(w, rule=rule, **t) == "R"]
                    x = frac(mz, lambda w: label(w) == "M*", n_boot=0)
                    cd.append(f"{x['k']}/{x['n']}")
                f_ = lambda x: f"{x['k']}/{x['n']}" if x else "—"  # noqa: E731
                L.append(f"| {tau} | {p_min} | {h_min} | {f_(ca)} | {f_(cc)} | {' / '.join(cd)} |")
    L.append("")
    # label sensitivity (thresholds as above)
    L.append("## 标签敏感性（阈值同上）\n")
    L.append("| M\\* 阈值（格） | 陷阱半径 | M\\* / R\\* / 推进 窗口数 | C-a 型 | C-c 型 | C-d 型（plan / local） |")
    L.append("|---:|---:|---|---|---|---|")
    for m_over in (0.3, 0.5, 1.0):
        for trap_r in (1, 2, 3):
            lb = lambda w, m=m_over, t=trap_r: label(w, m_over=m, trap_r=t)  # noqa: E731
            ws = [w for r in maze_runs for w in W_[r]]
            n = collections.Counter(lb(w) for w in ws)
            ca = frac([w for w in ws if w["first"] and lb(w) == "M*"], lambda w: verdict(w, **th) == "M", n_boot=0)
            cc = frac([w for w in ws if lb(w) == "prog"], lambda w: verdict(w, **th) in ("M", "R"), n_boot=0)
            cd = [frac([w for w in ws if w["trig"] and verdict(w, rule=rule, **th) == "R"], lambda w: lb(w) == "M*", n_boot=0)
                  for rule in ("plan", "local")]
            f_ = lambda x: f"{x['k']}/{x['n']}" if x["n"] else "—"  # noqa: E731
            L.append(f"| {m_over} | {trap_r} | {n['M*']} / {n['R*']} / {n['prog']} | {f_(ca)} | {f_(cc)} | "
                     f"{f_(cd[0])} / {f_(cd[1])} |")
    L.append("")
    out = ROOT / a.report
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.name}.md").write_text("\n".join(L))
    (out / f"{a.name}.json").write_text(json.dumps(S, indent=1, default=float))
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--labels", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--runs", nargs="+", default=["A1", "A2"])
    ap.add_argument("--root", default="eval_outputs/attr_d0")
    ap.add_argument("--report", default="reports/attr_d0")
    ap.add_argument("--name", default="dev")
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--p-min", type=float, default=None)
    ap.add_argument("--h-min", type=float, default=None)
    ap.add_argument("--maze-ckpt", default="checkpoints/diversemaze_layout_shift")
    ap.add_argument("--maze-data", default="data/diverse_maze")
    ap.add_argument("--latent-step", type=float, default=0.2, help="spacing of the rendered points (maze units), as E0")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    if a.check:
        ok = [check(run, a.root) for run in a.runs]
        sys.exit(0 if all(ok) else 1)
    if a.labels:
        for run in a.runs:
            if run in MAZE:
                maze_labels(run, a)
    if a.analyze:
        analyze(a)


if __name__ == "__main__":
    main()
