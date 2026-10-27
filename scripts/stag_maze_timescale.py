"""Maze easy with each planned action held for 4 env steps, the timescale of the diverse_maze data.

The data step is exactly 4 env steps (scripts/stag_maze_tempo.py, stage dyn), while the released config executes each
action for 1 env step. `++env_kwargs_override.action_repeat=4` holds each action for 4 env steps inside step(), so
the planner, the frames it sees, AdaJEPA's segments and the success checks are unchanged; only the time an action
lasts is. Runs come from scripts/stag_maze_base_run.py (--root eval_outputs/stag_maze100_ar4); the originals are in
eval_outputs/stag_maze100.

  replay   checks through the planning env: from data states, one step() per data action with action_repeat 1 and 4
  compare  a short run against the reference run: per-replan actions and states of the same episodes
  targets  start and goal states (and start/goal frames) of the new runs against the originals
  summary  success by replan budget for the four runs, paired McNemar tests, the initial model's one-step loss on the
           executed segments, movement per replan, stall triggers

    LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia python scripts/stag_maze_timescale.py --stages replay
    python scripts/stag_maze_timescale.py --stages compare --run eval_outputs/stag_maze_timescale/smoke_ar1/easy/adajepa/ref
    python scripts/stag_maze_timescale.py --stages targets summary
"""
import argparse
import collections
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

BASES = ("frozen", "adajepa")
RUNS = {"old": "eval_outputs/stag_maze100", "new": "eval_outputs/stag_maze100_ar4"}
NAME = {("frozen", "old"): "冻结·原始", ("adajepa", "old"): "AdaJEPA·原始",
        ("frozen", "new"): "冻结·修正", ("adajepa", "new"): "AdaJEPA·修正"}
BUDGETS = (5, 10, 12, 20, 30, 40, 50)


def load_eps(run_dir):
    eps = collections.defaultdict(list)
    for f in sorted(Path(run_dir).glob("shard*/per_iter_logs.json")):
        for line in f.open():
            if line.strip():
                r = json.loads(line)
                eps[r["sample_idx"]].append(r)
    return {i: sorted(rs, key=lambda r: r["step"]) for i, rs in eps.items()}


def stage_replay(a, out_dir):
    import gym
    import torch
    import env  # noqa: F401  (registers diverse_maze)

    d = ROOT / a.data / "val"
    ld = lambda f: torch.load(d / f"{f}.pth", weights_only=False)  # noqa: E731
    S, A, M, specs = ld("states").numpy(), ld("actions").numpy(), ld("map_indices").numpy(), ld("maze_specs")
    rng = np.random.default_rng(0)
    pairs = [(i, t) for i in range(S.shape[0]) for t in range(S.shape[1] - 1)]
    pairs = [pairs[j] for j in rng.choice(len(pairs), size=a.replay_n, replace=False)]
    res = {}
    for rep in (1, 4):
        envs, pe, ve, ratio = {}, [], [], []
        for i, t in pairs:
            k = int(M[i])
            if k not in envs:
                envs[k] = gym.make("diverse_maze", maze_spec=specs[k], action_repeat=rep).unwrapped
            e = envs[k]
            e.set_state(S[i, t, :2].astype(np.float64), S[i, t, 2:].astype(np.float64))
            _, _, _, info = e.step(A[i, t].astype(np.float64))
            s1 = np.asarray(info["state"][:4], dtype=np.float64)
            pe.append(np.linalg.norm(s1[:2] - S[i, t + 1, :2]))
            ve.append(np.linalg.norm(s1[2:] - S[i, t + 1, 2:]))
            dd = np.linalg.norm(S[i, t + 1, :2] - S[i, t, :2])
            if dd > 0.05:
                ratio.append(np.linalg.norm(s1[:2] - S[i, t, :2]) / dd)
        res[f"repeat{rep}"] = {"pos_err_median": float(np.median(pe)), "pos_err_p99": float(np.percentile(pe, 99)),
                               "vel_err_median": float(np.median(ve)), "disp_ratio_median": float(np.median(ratio))}
    res["n"] = len(pairs)
    res["pass"] = res["repeat4"]["pos_err_median"] < 1e-3
    (out_dir / "replay.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def stage_compare(a, out_dir):
    run, ref = load_eps(ROOT / a.run), load_eps(ROOT / a.ref)
    worst = {"act": 0.0, "state": 0.0}
    n = 0
    for i, rs in run.items():
        rr = {r["step"]: r for r in ref[i]}
        for r in rs:
            for k in worst:
                worst[k] = max(worst[k], float(np.abs(np.asarray(r[k]) - np.asarray(rr[r["step"]][k])).max()))
            n += 1
    res = {"run": a.run, "ref": a.ref, "episodes": sorted(run), "replans": n, "max_abs_diff": worst,
           "identical": all(v == 0.0 for v in worst.values())}
    (out_dir / f"compare_{Path(a.run).parts[-3]}_{Path(a.run).parts[-2]}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def stage_targets(a, out_dir):
    res = {}
    for base in BASES:
        t = {}
        for tag, root in RUNS.items():
            f = sorted((ROOT / root / a.level / base / "ref").glob("shard*/plan_targets.pkl"))[0]
            t[tag] = pickle.load(f.open("rb"))
        res[base] = {k: float(np.abs(np.asarray(t["old"][k]) - np.asarray(t["new"][k])).max()) for k in ("state_0", "state_g")}
        for k in ("obs_0", "obs_g"):
            res[base][k] = float(np.abs(np.asarray(t["old"][k]["visual"], dtype=np.float64) -
                                        np.asarray(t["new"][k]["visual"], dtype=np.float64)).max())
    res["identical"] = all(v == 0.0 for b in BASES for v in res[b].values())
    (out_dir / "targets.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def mcnemar(x, y):
    """Exact McNemar on paired booleans: (only x, only y, two-sided p)."""
    from scipy.stats import binomtest

    b, c = int(np.sum(x & ~y)), int(np.sum(~x & y))
    return b, c, float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0


def summarize(a, out_dir, rep_dir):
    E = {(base, tag): load_eps(ROOT / root / a.level / base / "ref") for base in BASES for tag, root in RUNS.items()}
    idx = sorted(set.intersection(*(set(v) for v in E.values())))
    tsucc = {}
    for key, eps in E.items():
        tsucc[key] = np.array([next((r["step"] for r in eps[i] if r["success"]), np.inf) for i in idx])
    ok = lambda key, b: tsucc[key] <= b  # noqa: E731
    S = {"n": len(idx), "success": {}, "mcnemar": {}, "loss": {}, "move": {}, "trigger": {}}
    md = [f"# 迷宫 {a.level}：每个动作执行 4 个环境步（时间尺度修正）汇总", "",
          f"{len(idx)} 个 episode，四组运行在同样的起终点和 episode 种子上配对。预算按重规划次数计；"
          "修正后每次重规划 = 5 个动作 × 4 个环境步 = 20 个环境步，原始为 5 个环境步。", "",
          "## 1. 成功率随重规划预算", "",
          "| 运行 | " + " | ".join(f"≤{b}" for b in BUDGETS) + " | 成功所需重规划（中位数） |",
          "|---|" + "---:|" * (len(BUDGETS) + 1)]
    for key in E:
        t = tsucc[key]
        S["success"][NAME[key]] = {str(b): int(ok(key, b).sum()) for b in BUDGETS}
        med = float(np.median(t[np.isfinite(t)])) if np.isfinite(t).any() else float("nan")
        S["success"][NAME[key]]["median_replans"] = med
        md.append(f"| {NAME[key]} | " + " | ".join(str(int(ok(key, b).sum())) for b in BUDGETS) + f" | {med:.0f} |")
    md += ["", "按物理时间对齐：修正后 12 次重规划 = 240 个环境步，原始 50 次 = 250 个环境步。", "",
           "## 2. 配对比较（精确 McNemar）", "",
           "| 比较 | 预算 | 成功率 | 只有前者成功 | 只有后者成功 | p |", "|---|---|---|---:|---:|---:|"]
    comps = [(("frozen", "new"), 50, ("frozen", "old"), 50), (("adajepa", "new"), 50, ("frozen", "new"), 50),
             (("frozen", "new"), 50, ("adajepa", "old"), 50), (("adajepa", "new"), 50, ("adajepa", "old"), 50),
             (("adajepa", "old"), 50, ("frozen", "old"), 50),
             (("frozen", "new"), 12, ("frozen", "old"), 50), (("adajepa", "new"), 12, ("adajepa", "old"), 50)]
    for k1, b1, k2, b2 in comps:
        x, y = ok(k1, b1), ok(k2, b2)
        b, c, p = mcnemar(x, y)
        lab = f"{NAME[k1]} vs {NAME[k2]}"
        S["mcnemar"][f"{lab} @{b1}/{b2}"] = {"succ": [int(x.sum()), int(y.sum())], "only_first": b, "only_second": c, "p": p}
        md.append(f"| {lab} | {b1} / {b2} | {int(x.sum())} / {int(y.sum())} | {b} | {c} | {p:.3g} |")
    # per-element MSE over the 384 visual + 10 proprio channels (action channels dropped); split back into the two parts
    vis_d, prop_d = 384, 10
    part = {"总": lambda r, s: r[f"loss_new_{s}"], "视觉": lambda r, s: r[f"loss_new_vis_{s}"],
            "本体": lambda r, s: ((vis_d + prop_d) * r[f"loss_new_{s}"] - vis_d * r[f"loss_new_vis_{s}"]) / prop_d}
    md += ["", "## 3. 动力学误差：模型对刚执行片段的单步预测损失", "",
           "初始（未更新）模型对每次重规划刚执行的片段、最新一帧的单步潜预测 MSE（`loss_new_init`），按 384 个视觉通道和 "
           "10 个本体通道拆开；“当前模型”是 AdaJEPA 更新前的模型（冻结组与初始模型相同）。第 1 次重规划还没有任何更新，"
           "两种基准相同。", "",
           "| 运行 | 部分 | 初始模型·第 1 次重规划 | 初始模型·全部重规划 | 当前模型·全部重规划 |", "|---|---|---:|---:|---:|"]
    for key, eps in E.items():
        S["loss"][NAME[key]] = {}
        for pname, f in part.items():
            first = [f(eps[i][0], "init") for i in idx if eps[i] and eps[i][0]["step"] == 1]
            allr = [f(r, "init") for i in idx for r in eps[i]]
            cur = [f(r, "cur") for i in idx for r in eps[i]]
            S["loss"][NAME[key]][pname] = {"first": float(np.median(first)), "all": float(np.median(allr)),
                                           "cur_all": float(np.median(cur))}
            md.append(f"| {NAME[key]} | {pname} | {np.median(first):.4f} | {np.median(allr):.4f} | {np.median(cur):.4f} |")
    md.append("")
    for base in BASES:
        for pname, f in part.items():
            o = {i: f(rs[0], "init") for i, rs in E[(base, "old")].items() if rs[0]["step"] == 1}
            n_ = {i: f(rs[0], "init") for i, rs in E[(base, "new")].items() if rs[0]["step"] == 1}
            ratio = [n_[i] / o[i] for i in idx if i in o and i in n_ and o[i] > 0]
            S["loss"][f"{base}_first_ratio_new_over_old_{pname}"] = float(np.median(ratio))
            md.append(f"- {NAME[(base, 'new')]} / {NAME[(base, 'old')]}，第 1 次重规划{pname}损失逐 episode 比值的中位数："
                      f"**{np.median(ratio):.2f}**")
    md += ["", "## 4. 每次重规划的移动和卡住触发", "",
           "| 运行 | 每次重规划的位移（中位数，格） | 触发过（stag_r < 0.15，第 4 次重规划起） | 其中失败 |", "|---|---:|---:|---:|"]
    for key, eps in E.items():
        mv, trig, trig_fail = [], 0, 0
        for j, i in enumerate(idx):
            st = [np.asarray(r["state"][:2]) for r in eps[i]]
            mv += [float(np.linalg.norm(st[k + 1] - st[k])) for k in range(len(st) - 1)]
            hit = any(r.get("stag_r") is not None and r["stag_r"] < a.gamma_hi for r in eps[i])
            trig += hit
            trig_fail += hit and not np.isfinite(tsucc[key][j])
        S["move"][NAME[key]] = float(np.median(mv))
        S["trigger"][NAME[key]] = {"fired": trig, "fired_fail": trig_fail}
        md.append(f"| {NAME[key]} | {np.median(mv):.3f} | {trig} | {trig_fail} |")
    rep_dir.mkdir(parents=True, exist_ok=True)
    (rep_dir / f"summary_{a.level}.md").write_text("\n".join(md) + "\n")
    (rep_dir / f"summary_{a.level}.json").write_text(json.dumps(S, indent=1))
    np.savez(out_dir / f"tsucc_{a.level}.npz", idx=np.array(idx), **{f"{b}_{t}": tsucc[(b, t)] for b, t in E})
    print("\n".join(md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="+", default=["summary"], choices=["replay", "compare", "targets", "summary"])
    ap.add_argument("--level", default="easy")
    ap.add_argument("--data", default="data/diverse_maze")
    ap.add_argument("--replay-n", type=int, default=3000)
    ap.add_argument("--run", help="compare: run dir (…/<level>/<base>/ref)")
    ap.add_argument("--ref", help="compare: reference dir (default: the same base and level in the original runs)")
    ap.add_argument("--gamma-hi", type=float, default=0.15)
    ap.add_argument("--out", default="eval_outputs/stag_maze_timescale")
    ap.add_argument("--report", default="reports/stag_maze_timescale")
    a = ap.parse_args()
    if a.run and not a.ref:
        p = Path(a.run).parts
        a.ref = str(Path(RUNS["old"]) / p[-3] / p[-2] / "ref")
    out_dir = ROOT / a.out
    out_dir.mkdir(parents=True, exist_ok=True)
    for stage, fn in (("replay", stage_replay), ("compare", stage_compare), ("targets", stage_targets)):
        if stage in a.stages:
            fn(a, out_dir)
    if "summary" in a.stages:
        summarize(a, out_dir, ROOT / a.report)


if __name__ == "__main__":
    main()
