"""E1: when does fixing the model stop helping? Offline characterisation of the switch signal on dev data
(reports/paper2_positioning/narrative.md, v3, section 6). Reads existing AdaJEPA logs only; nothing is planned.

Switch signal, per replan s (record step s), label-free and in latent units:

    armed(s)    the stall trigger fires: r = d(z_s, z_{s-3}) / d(z_s, z_goal) < 0.15 from its start replan on
                (maze: the logged stag_w > 0; PushObj: the offline latent trace, as scripts/stag_select.py does)
    plateau(s)  AdaJEPA's prediction loss L stopped falling: with L3_s the mean of L over replans s-2..s,
                (L3_{s-W} - L3_s) / L3_{s-W} < theta
    switch      the first s at which armed and plateau have both held for K consecutive replans

L is the loss on AdaJEPA's own replay buffer before that replan's update (loss_buf_cur, primary) or on the newest,
not yet trained, transition (loss_new_cur, secondary). Per (signal, W, theta, K) and data set:

    F  failed episodes that triggered: share switched with >= 10 replans of the budget left
    S  later-successful episodes that triggered: share switched before their success (false switches)
    H  success hazard per replan at risk: after the first trigger but before the switch, and after the switch

Criteria fixed beforehand (narrative v3): F >= 70%, S <= 25%, and the after-switch hazard clearly below the before-switch
one (here: at most half). Selection: among settings meeting F and S on BOTH data sets, the lowest mean S, then the
earliest switch. Maze only (privileged, evaluation): the model's one-step position error read off its prediction
(eval_outputs/attr_d0/labels_A2.jsonl) at the first trigger, at the switch and at the end, and whether that error was
still falling at the switch (a "false plateau" of the loss).

Control without any loss: switch D replans after the first trigger if the trigger is still armed then. The report also
shows the median loss (relative to the first trigger) and the privileged error at k replans after the first trigger.

    python scripts/e1_switch.py                      # maze medium (main), PushObj S7
    python scripts/e1_switch.py --maze-run eval_outputs/attr_d0/A2 --maze-labels eval_outputs/attr_d0/labels_A2.jsonl \
        --maze-name "迷宫 easy 修正设定 AdaJEPA" --out reports/e1_switch/easy
    python scripts/e1_switch.py --push-ext eval_outputs/stag_S7_it50/S0 --push-budget 50 --out reports/e1_switch/s7_it50
"""
import argparse
import collections
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

GAMMA_HI = 0.15
GRID = {"W": (3, 5), "theta": (0.05, 0.1, 0.2), "K": (1, 2, 3)}
SIGNALS = ("loss_buf_cur", "loss_new_cur")
LEFT = 10  # replans of budget that must remain at a failed episode's switch
DELAYS = (0, 3, 5, 8, 10, 15)  # time-only control
CRIT = {"F": 0.70, "S": 0.25, "H": 0.5}
FALSE_PLATEAU = 0.2  # privileged error still fell by > 20% over the last W replans at the switch


def maze(a):
    """Episodes of the maze dev run: {key: {"steps": {s: record}, "succ": s or None, "armed": set, "err": {s: cells}}}."""
    err = {}
    if a.maze_labels:  # privileged labels exist only for runs logged with planning/attribution.py
        for line in (ROOT / a.maze_labels).open():
            x = json.loads(line)
            err[(x["i"], x["step"])] = x["err"]
    eps = collections.defaultdict(dict)
    for f in sorted((ROOT / a.maze_run).glob("*/per_iter_logs.json")):
        for line in f.open():
            r = json.loads(line)
            eps[r["sample_idx"]][r["step"]] = r
    out = {}
    for i, st in eps.items():
        succ = [s for s, r in st.items() if r["success"]]
        out[("maze", i)] = {"steps": st, "succ": min(succ) if succ else None,
                            "armed": {s for s, r in st.items() if (r.get("stag_w") or 0) > 0},
                            "err": {s: err[(i, s)] for s in st if (i, s) in err}}
    return out, a.maze_budget


def pushobj(a):
    trace = {}
    for line in (ROOT / a.push_trace).open():
        x = json.loads(line)
        trace[(x["chunk"], x["sample_idx"])] = x
    ext = collections.defaultdict(dict)  # episodes rerun with a longer budget (scripts/s7_extend.py) replace the reference's
    if a.push_ext:
        for f in sorted((ROOT / a.push_ext).glob("*/per_iter_logs.json")):
            for line in f.open():
                if line.strip():
                    r = json.loads(line)
                    ext[(f.parent.name, r["sample_idx"])][r["step"]] = r
    out = {}
    for f in sorted((ROOT / a.push_run).glob("*/per_iter_logs.json")):
        chunk = f.parent.name
        eps = collections.defaultdict(dict)
        for line in f.open():
            r = json.loads(line)
            eps[r["sample_idx"]][r["step"]] = r
        for i, st in eps.items():
            if (chunk, i) in ext:  # logged by StagnationMPCPlanner (kappa 0): the trigger as the planner computes it
                st = ext[(chunk, i)]
                armed = {s for s, r in st.items() if (r.get("stag_w") or 0) > 0}
            else:
                tr = trace[(chunk, i)]
                armed = {x["s"] for x in tr["steps"]
                         if a.push_start <= x["s"] < a.push_budget and "dW" in x and x["dW"] / max(x["dg"], 1e-12) < GAMMA_HI}
            succ = [s for s, r in st.items() if r["success"]]
            out[(chunk, i)] = {"steps": st, "succ": min(succ) if succ else None, "armed": armed, "err": {}}
    return out, a.push_budget


def switch_step(ep, sig, W, theta, K):
    st = ep["steps"]
    L = {s: r[sig] for s, r in st.items() if r.get(sig) is not None}
    L3 = {s: np.mean([L[t] for t in (s - 2, s - 1, s)]) for s in L if all(t in L for t in (s - 2, s - 1, s))}
    run = 0
    for s in sorted(st):
        ok = s in ep["armed"] and s in L3 and (s - W) in L3 and (L3[s - W] - L3[s]) / max(L3[s - W], 1e-12) < theta
        run = run + 1 if ok else 0
        if run >= K:
            return s
    return None


def evaluate(eps, budget, sig, W, theta, K):
    F_n = F_k = S_n = S_k = 0
    risk = {"pre": [0, 0], "post": [0, 0]}  # [replans at risk, successes]
    sw_steps, rows = [], []
    for key, ep in eps.items():
        if not ep["armed"]:
            continue
        first = min(ep["armed"])
        sw = switch_step(ep, sig, W, theta, K)
        T = max(ep["steps"])
        if ep["succ"] is None:
            F_n += 1
            F_k += sw is not None and sw <= budget - LEFT
        elif ep["succ"] > first:
            S_n += 1
            S_k += sw is not None and sw < ep["succ"]
        end = ep["succ"] if ep["succ"] is not None else T
        for s in range(first + 1, end + 1):  # replans planned after the trigger fired
            phase = "post" if sw is not None and s > sw else "pre"
            risk[phase][0] += 1
            risk[phase][1] += ep["succ"] == s
        if sw is not None:
            sw_steps.append(sw)
        rows.append({"key": key, "first": first, "switch": sw, "succ": ep["succ"], "T": T})
    hz = {p: (v[1] / v[0] if v[0] else None) for p, v in risk.items()}
    return {"F": F_k / F_n if F_n else None, "F_k": F_k, "F_n": F_n, "S": S_k / S_n if S_n else None, "S_k": S_k,
            "S_n": S_n, "hz_pre": hz["pre"], "hz_post": hz["post"], "risk": risk,
            "sw_median": float(np.median(sw_steps)) if sw_steps else None, "rows": rows}


def evaluate_delay(eps, budget, D):
    """F and S for the time-only control: switch at the first armed replan >= first trigger + D."""
    F_n = F_k = S_n = S_k = 0
    for ep in eps.values():
        if not ep["armed"]:
            continue
        first = min(ep["armed"])
        sw = next((s for s in sorted(ep["armed"]) if s >= first + D), None)
        if ep["succ"] is None:
            F_n += 1
            F_k += sw is not None and sw <= budget - LEFT
        elif ep["succ"] > first:
            S_n += 1
            S_k += sw is not None and sw < ep["succ"]
    return {"F": F_k / F_n if F_n else None, "F_k": F_k, "F_n": F_n, "S": S_k / S_n if S_n else None, "S_k": S_k, "S_n": S_n}


def privileged(eps, rows, W):
    """Maze: position error at the first trigger, at the switch and at the end; false plateaus at the switch."""
    at = {"trig": [], "switch": [], "end": []}
    fp = n = 0
    for r in rows:
        ep = eps[r["key"]]
        e = ep["err"]
        e3 = {s: np.mean([e[t] for t in (s - 2, s - 1, s)]) for s in e if all(t in e for t in (s - 2, s - 1, s))}
        if r["first"] in e3:
            at["trig"].append(e3[r["first"]])
        if r["T"] in e3:
            at["end"].append(e3[r["T"]])
        sw = r["switch"]
        if sw is not None and sw in e3:
            at["switch"].append(e3[sw])
            if (sw - W) in e3:
                n += 1
                fp += (e3[sw - W] - e3[sw]) / max(e3[sw - W], 1e-12) > FALSE_PLATEAU
    med = {k: (float(np.median(v)) if v else None) for k, v in at.items()}
    acc = float(np.mean(np.array(at["switch"]) <= 0.5)) if at["switch"] else None
    return {"median_err": med, "switch_err_le_0.5": acc, "false_plateau": fp, "n_checked": n}


def curves(eps, sig, max_k=20):
    """Median L (relative to its value at the first trigger) and privileged error by replans since the first trigger."""
    out = {}
    for grp in ("fail", "succ"):
        rel, err = collections.defaultdict(list), collections.defaultdict(list)
        for ep in eps.values():
            if not ep["armed"] or (grp == "fail") != (ep["succ"] is None):
                continue
            first = min(ep["armed"])
            L0 = ep["steps"][first].get(sig)
            if not L0:
                continue
            for k in range(0, max_k + 1, 5):
                s = first + k
                if s in ep["steps"] and ep["steps"][s].get(sig) is not None:
                    rel[k].append(ep["steps"][s][sig] / L0)
                    if s in ep["err"]:
                        err[k].append(ep["err"][s])
        out[grp] = {k: (float(np.median(rel[k])), len(rel[k]), float(np.median(err[k])) if err[k] else None)
                    for k in sorted(rel)}
    return out


def pct(x):
    return "—" if x is None else f"{100 * x:.0f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maze-run", default="eval_outputs/stag_maze100_ar4/medium/adajepa/ref")
    ap.add_argument("--maze-labels", default="", help="privileged labels (e.g. eval_outputs/attr_d0/labels_A2.jsonl)")
    ap.add_argument("--maze-name", default="迷宫 medium 修正设定 AdaJEPA")
    ap.add_argument("--maze-budget", type=int, default=50)
    ap.add_argument("--push-run", default="eval_outputs/escape_S7/none")
    ap.add_argument("--push-trace", default="eval_outputs/stag_S7/trace.jsonl")
    ap.add_argument("--push-start", type=int, default=10)
    ap.add_argument("--push-budget", type=int, default=30)
    ap.add_argument("--push-ext", default="", help="rerun of the reference's failures with a longer budget "
                    "(eval_outputs/stag_S7_it50/S0, with --push-budget 50)")
    ap.add_argument("--out", default="reports/e1_switch")
    a = ap.parse_args()

    data = {a.maze_name: maze(a), "PushObj S7 AdaJEPA（我们的权重）": pushobj(a)}
    res = {}
    for sig, W, theta, K in itertools.product(SIGNALS, GRID["W"], GRID["theta"], GRID["K"]):
        res[(sig, W, theta, K)] = {name: evaluate(eps, budget, sig, W, theta, K) for name, (eps, budget) in data.items()}

    def meets(r):
        return all(x["F"] is not None and x["F"] >= CRIT["F"] and x["S"] is not None and x["S"] <= CRIT["S"] for x in r.values())

    ok = [k for k, r in res.items() if meets(r)]
    best = min(ok, key=lambda k: (np.mean([x["S"] for x in res[k].values()]),
                                  np.mean([x["sw_median"] or 99 for x in res[k].values()]))) if ok else None

    L = ["# E1：修模型什么时候不再有用（开发集，离线）\n",
         f"标准（事先定）：F ≥ {pct(CRIT['F'])}（失败且触发过的 episode，预算用完前至少还剩 {LEFT} 次时切换）；"
         f"S ≤ {pct(CRIT['S'])}（触发后才成功的 episode，在成功前被切换）；切换后的成功率 ≤ 切换前的一半（按重规划计）。"
         "两个数据集用同一组参数。\n"]
    for name, (eps, budget) in data.items():
        trig = [e for e in eps.values() if e["armed"]]
        L.append(f"- {name}：{len(eps)} 个 episode，预算 {budget} 次；触发过 {len(trig)} 个，"
                 f"其中失败 {sum(e['succ'] is None for e in trig)} 个，触发后才成功 {sum(e['succ'] is not None and e['succ'] > min(e['armed']) for e in trig)} 个。")
    L.append("")
    L.append("## 全部参数\n")
    L.append("| 信号 | W | θ | K | " + " | ".join(f"{n[:8]} F / S / 成功率 前→后 / 切换中位" for n in data) + " | 达标 |")
    L.append("|---|---:|---:|---:|" + "---|" * len(data) + "---|")
    for k, r in res.items():
        cells = []
        for name in data:
            x = r[name]
            hz = (f"{x['hz_pre']:.3f}→{x['hz_post']:.3f}" if x["hz_pre"] is not None and x["hz_post"] is not None else "—")
            cells.append(f"{pct(x['F'])} ({x['F_k']}/{x['F_n']}) / {pct(x['S'])} ({x['S_k']}/{x['S_n']}) / {hz} / "
                         f"{x['sw_median'] if x['sw_median'] is not None else '—'}")
        L.append(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | " + " | ".join(cells) + f" | {'✓' if k in ok else ''} |")
    L.append("")
    summary = {"criteria": CRIT, "n_meeting_F_S": len(ok), "best": None}
    if best is None:
        L.append("**没有一组参数在两个数据集上同时满足 F 和 S。**\n")
    else:
        sig, W, theta, K = best
        L.append(f"## 选中的参数：信号 {sig}，W = {W}，θ = {theta}，K = {K}\n")
        L.append("（规则：两个数据集都满足 F 和 S 的参数中，平均 S 最低，其次切换最早。）\n")
        summary["best"] = {"signal": sig, "W": W, "theta": theta, "K": K, "data": {}}
        for name, (eps, budget) in data.items():
            x = res[best][name]
            hz_ok = x["hz_pre"] and x["hz_post"] is not None and x["hz_post"] <= CRIT["H"] * x["hz_pre"]
            L.append(f"- {name}：F {pct(x['F'])}（{x['F_k']}/{x['F_n']}），S {pct(x['S'])}（{x['S_k']}/{x['S_n']}），"
                     f"成功率每次重规划 切换前 {x['hz_pre']:.3f}（{x['risk']['pre'][1]}/{x['risk']['pre'][0]}）→ 切换后 "
                     f"{x['hz_post']:.3f}（{x['risk']['post'][1]}/{x['risk']['post'][0]}）{'，达标' if hz_ok else '，未达标'}；"
                     f"切换步中位 {x['sw_median']}。")
            summary["best"]["data"][name] = {k: v for k, v in x.items() if k != "rows"}
            if eps is data[a.maze_name][0] and any(e["err"] for e in eps.values()):
                pv = privileged(eps, x["rows"], W)
                summary["best"]["privileged"] = pv
                m = pv["median_err"]
                L.append(f"  - 特权核对（模型单步位置误差，3 次平均，格）：首次触发时 {m['trig']:.2f}，切换时 "
                         f"{m['switch']:.2f}，结束时 {m['end']:.2f}；切换时误差 ≤ 0.5 格的占 {pct(pv['switch_err_le_0.5'])}；"
                         f"切换时特权误差在最近 {W} 次里仍下降 > {int(100 * FALSE_PLATEAU)}%（假平台）的 {pv['false_plateau']}/{pv['n_checked']}。")
        L.append("")
    # control: switch D replans after the first trigger if the trigger is armed then (still stuck), no loss at all
    L.append("## 对照：只看时间（首次触发后再过 D 次，仍卡着就切换）\n")
    L.append("| D | " + " | ".join(f"{n[:8]} F / S" for n in data) + " |")
    L.append("|---:|" + "---|" * len(data))
    summary["delay"] = {}
    for D in DELAYS:
        cells = []
        for name, (eps, budget) in data.items():
            x = evaluate_delay(eps, budget, D)
            summary["delay"].setdefault(name, {})[D] = x
            cells.append(f"{pct(x['F'])} ({x['F_k']}/{x['F_n']}) / {pct(x['S'])} ({x['S_k']}/{x['S_n']})")
        L.append(f"| {D} | " + " | ".join(cells) + " |")
    L.append("")
    for name, (eps, _) in data.items():
        lag = [e["succ"] - min(e["armed"]) for e in eps.values() if e["armed"] and e["succ"] is not None and e["succ"] > min(e["armed"])]
        if lag:
            q = np.percentile(lag, [25, 50, 75])
            L.append(f"- {name}：触发后才成功的 episode，在首次触发后第 {q[0]:.0f} / {q[1]:.0f} / {q[2]:.0f} 次成功"
                     f"（四分位，n = {len(lag)}，最晚 {max(lag)}）。")
    L.append("")
    # curves
    L.append("## 触发之后损失和位置误差怎么变（中位数；损失相对首次触发时）\n")
    for name, (eps, _) in data.items():
        cv = curves(eps, SIGNALS[0])
        L.append(f"{name}（{SIGNALS[0]}）：\n")
        L.append("| 触发后第 k 次 | 失败：相对损失（n） | 失败：位置误差 | 后来成功：相对损失（n） | 后来成功：位置误差 |")
        L.append("|---:|---|---|---|---|")
        for k in sorted(set(cv["fail"]) | set(cv["succ"])):
            f_, s_ = cv["fail"].get(k), cv["succ"].get(k)
            ff = lambda v: "—" if v is None else f"{v[0]:.2f}（{v[1]}）"  # noqa: E731
            fe = lambda v: "—" if v is None or v[2] is None else f"{v[2]:.2f}"  # noqa: E731
            L.append(f"| {k} | {ff(f_)} | {fe(f_)} | {ff(s_)} | {fe(s_)} |")
        L.append("")
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "tables.md").write_text("\n".join(L))
    summary["grid"] = {"|".join(map(str, k)): {n: {kk: vv for kk, vv in x.items() if kk != "rows"} for n, x in r.items()}
                       for k, r in res.items()}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print("\n".join(L))


if __name__ == "__main__":
    main()
