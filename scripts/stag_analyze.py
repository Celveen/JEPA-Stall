"""Analyse stall-regulariser arms (scripts/stag_run.py) against AdaJEPA and K2P on the same episodes.

Episodes not selected by scripts/stag_select.py keep the AdaJEPA reference outcome (identical by
construction). Checks first that every rerun episode reproduces the reference exactly up to the
replan on which its penalty switches on (and entirely when it never switches on). Then:

  success   merged over all episodes, exact paired McNemar vs AdaJEPA and vs K2P_short2
  leave     after the first trigger, the pusher gets > 15 px away from where it stalled within 3 replans
  object    the object error falls by > 5 px within 10 replans of the trigger; the pusher reaches
            the far side of the object (cos(pusher - object, object goal - object) < -0.3)
  harm      selected episodes that AdaJEPA solves after the trigger and the arm does not
(state-based evaluation metrics; the method itself only sees latents)

    python scripts/stag_analyze.py --arms SP_lo SP_mid SP_hi SZ_lo SZ_mid SZ_hi
"""
import argparse
import collections
import json
from math import comb
from pathlib import Path

import numpy as np

from stag_common import ROOT, goal_state, ref_episodes, stream_targets

UNSEEN = {"I", "square", "small_tee"}


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)


def load_arm(root, max_iter):
    eps, meta = {}, {}
    for ch in sorted(p for p in root.iterdir() if (p / "per_iter_logs.json").exists()):
        by = collections.defaultdict(list)
        for line in (ch / "per_iter_logs.json").open():
            if line.strip():
                r = json.loads(line)
                if r["step"] <= max_iter:
                    by[r["sample_idx"]].append(r)
        for i, rs in by.items():
            eps[(ch.name, i)] = sorted(rs, key=lambda r: r["step"])
        if (ch / "stag_episodes.jsonl").exists():
            for line in (ch / "stag_episodes.jsonl").open():
                d = json.loads(line)
                meta[(ch.name, d["sample_idx"])] = d
    return eps, meta


def succ_step(rs):
    return next((r["step"] for r in rs if r["success"]), None)


def side(s, g):
    u, v = np.asarray(s[:2]) - np.asarray(s[2:4]), np.asarray(g[2:4]) - np.asarray(s[2:4])
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9))


def diag(rs, t0, g):
    """Diagnostics after the trigger at record step t0 (state after t0 replans)."""
    by = {r["step"]: r for r in rs}
    if t0 not in by:
        return None
    p0, o0 = np.array(by[t0]["state"][:2]), by[t0]["dist_obj"]
    nxt = [by[s] for s in range(t0 + 1, t0 + 11) if s in by]
    if not nxt:
        return None
    leave = any(np.linalg.norm(np.array(r["state"][:2]) - p0) > 15 for r in nxt[:3])
    end = nxt[-1]
    solved = any(r["success"] for r in nxt)
    return {"leave": leave, "obj_better": solved or (o0 - end["dist_obj"] > 5),
            "obj_moved": any(np.linalg.norm(np.array(r["state"][2:4]) - np.array(by[t0]["state"][2:4])) > 5 for r in nxt),
            "behind": any(side(r["state"], g) < -0.3 for r in nxt), "front0": side(by[t0]["state"], g) > 0.3}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", default="EVAL50")
    ap.add_argument("--root", default="eval_outputs/stag_EVAL50")
    ap.add_argument("--ref", default="eval_outputs/escape_EVAL50/none")
    ap.add_argument("--k2p", default="eval_outputs/escape_EVAL50_r30/K2P_short2")
    ap.add_argument("--arms", nargs="+", default=["SP_lo", "SP_mid", "SP_hi", "SZ_lo", "SZ_mid", "SZ_hi"])
    ap.add_argument("--max-iter", type=int, default=30)
    ap.add_argument("--out", default="reports/stag_EVAL50/summary.json")
    a = ap.parse_args()

    root = ROOT / a.root
    touched = json.loads((root / "touched.json").read_text())
    sel = {(r["chunk"], r["sample_idx"]) for r in touched["episodes"]}
    ref = ref_episodes(a.ref, a.max_iter)
    k2p = ref_episodes(a.k2p, a.max_iter)
    keys = sorted(ref)
    shapes, goals = {}, {}
    for ch, i in keys:
        if ch not in shapes:
            shapes[ch] = stream_targets(a.stream, ch)
    for k in sel:
        goals[k] = goal_state(shapes[k[0]][k[1]])[0]
    shape_of = {k: shapes[k[0]][k[1]]["shape"] for k in keys}
    ok_ref = {k: succ_step(ref[k]) is not None for k in keys}
    ok_k2p = {k: succ_step(k2p[k]) is not None for k in keys}
    res = {"n": len(keys), "selected": len(sel), "AdaJEPA": float(np.mean([ok_ref[k] for k in keys])),
           "K2P_short2": float(np.mean([ok_k2p[k] for k in keys])), "arms": {}}
    lines = [f"# 停滞正则（潜空间版）EVAL50 seed 100 结果", "",
             f"350 个 episode，重跑其中 {len(sel)} 个（其余与 AdaJEPA 逐位相同，沿用其结果）。", "",
             "| 组 | 成功率 | 与 AdaJEPA（仅该组/仅 AdaJEPA，p） | 与 K2P_short2（p） | 没见过/见过 | 触发 episode | 离开 | 物体改善 | 绕到背后 | 误伤 |",
             "|---|---:|---|---|---|---:|---:|---:|---:|---:|"]
    for arm in a.arms:
        if not (root / arm).exists():
            continue
        eps, meta = load_arm(root / arm, a.max_iter)
        missing = [k for k in sel if k not in eps]
        ok = {k: (succ_step(eps[k]) is not None) if k in eps else ok_ref[k] for k in keys}
        # exactness: identical to the reference up to the replan the penalty switches on
        bad, trig = [], {}
        for k in sel & set(eps):
            m = meta.get(k, {})
            t = m.get("first_trigger")  # planning replan (0-based) from which the penalty is on = its record step
            trig[k] = t
            lim = t if t is not None else 10 ** 9  # records up to step t were planned without the penalty
            for r0, r1 in zip(ref[k], eps[k]):
                if r1["step"] > lim:
                    break
                if r0["act"] != r1["act"]:
                    bad.append(k)
                    break
            if t is None and len(ref[k]) != len(eps[k]):
                bad.append(k)
        fired = [k for k in sel & set(eps) if trig.get(k) is not None]
        d_arm = {k: diag(eps[k], trig[k], goals[k]) for k in fired}
        d_ref = {k: diag(ref[k], trig[k], goals[k]) for k in fired}
        frac = lambda d, f: np.mean([v[f] for v in d.values() if v is not None]) if d else float("nan")
        harm = sum(1 for k in fired if ok_ref[k] and not ok[k])
        gain = sum(1 for k in fired if ok[k] and not ok_ref[k])
        b = sum(ok[k] and not ok_ref[k] for k in keys)
        c = sum(ok_ref[k] and not ok[k] for k in keys)
        b2 = sum(ok[k] and not ok_k2p[k] for k in keys)
        c2 = sum(ok_k2p[k] and not ok[k] for k in keys)
        sr = float(np.mean([ok[k] for k in keys]))
        un = np.mean([ok[k] for k in keys if shape_of[k] in UNSEEN])
        se = np.mean([ok[k] for k in keys if shape_of[k] not in UNSEEN])
        res["arms"][arm] = {"success": sr, "vs_AdaJEPA": [b, c, mcnemar(b, c)], "vs_K2P": [b2, c2, mcnemar(b2, c2)],
                            "unseen": un, "seen": se, "missing": len(missing), "not_identical_before_trigger": len(bad),
                            "fired": len(fired), "gain": gain, "harm": harm,
                            "leave": [frac(d_arm, "leave"), frac(d_ref, "leave")],
                            "obj_better": [frac(d_arm, "obj_better"), frac(d_ref, "obj_better")],
                            "behind": [frac(d_arm, "behind"), frac(d_ref, "behind")],
                            "obj_moved": [frac(d_arm, "obj_moved"), frac(d_ref, "obj_moved")]}
        lines.append(f"| {arm} | {sr:.1%} | {sr - res['AdaJEPA']:+.1%}（{b}/{c}，p={mcnemar(b, c):.3g}） | "
                     f"{sr - res['K2P_short2']:+.1%}（{b2}/{c2}，p={mcnemar(b2, c2):.3g}） | {un:.1%} / {se:.1%} | {len(fired)} | "
                     f"{frac(d_arm, 'leave'):.0%}（{frac(d_ref, 'leave'):.0%}） | {frac(d_arm, 'obj_better'):.0%}（{frac(d_ref, 'obj_better'):.0%}） | "
                     f"{frac(d_arm, 'behind'):.0%}（{frac(d_ref, 'behind'):.0%}） | {harm} |")
        if missing or bad:
            lines.append(f"|  ⚠ {arm}: 缺 {len(missing)} 个 episode；{len(bad)} 个在开罚前与 AdaJEPA 不一致 |||||||||")
    lines += ["", f"参照：AdaJEPA {res['AdaJEPA']:.1%}，K2P_short2 {res['K2P_short2']:.1%}。"
              "括号里是 AdaJEPA 在同一批 episode、同一时刻之后的对应比例。",
              "离开 = 触发后 3 次重规划内推杆离开卡住点 15 px 以上；物体改善 = 10 次重规划内物体误差下降超过 5 px（或成功）；"
              "绕到背后 = 推杆到达物体背离目标的一侧；误伤 = AdaJEPA 能成功、该组失败。"]
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
