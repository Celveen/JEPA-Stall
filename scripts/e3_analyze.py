"""E3 / E2 analysis: each arm of scripts/e3_run.py against the AdaJEPA reference run, episode by episode.

    --check   bit-identity: every replan up to an episode's switch must equal the reference (actions, success,
              trigger ratio); with --same A B, two arms must be identical everywhere (R-trig vs T0SP-stag)
    default   per arm: wins (reference failed, arm succeeded) and losses, exact two-sided McNemar test, successes at
              budgets 20 / 30 / 50 over all episodes (episodes not rerun count as the reference), switch steps, and
              the E0 class of every changed episode -> reports/e3/<level>_s<seed>/<name>.{md,json}

    python scripts/e3_analyze.py --check --arms R-trig R-switch --root eval_outputs/e3_smoke --same R-trig T0SP-stag
    python scripts/e3_analyze.py --arms R-trig R-switch --name stage1
    python scripts/e3_analyze.py --env pushobj --arms R-trig R-switch --name stage1

PushObj (S7, our weights): episodes are keyed by (chunk, sample_idx); the reference is the 30-replan AdaJEPA run with
its 63 failures replaced by their 50-replan rerun (scripts/s7_extend.py), and has no E0 classes.
"""
import argparse
import collections
import json
import sys
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BUDGETS = (20, 30, 50)
E0 = {"medium": "eval_outputs/stag_maze_trap_ar4/medium_geom.jsonl"}
PUSH_REFS = {"S7": ["eval_outputs/escape_S7/none", "eval_outputs/stag_S7_it50/S0"]}  # later runs replace episodes


def records(run_dir, chunked=False):
    """{episode: {step: record}}; episode = sample_idx (maze shards) or (chunk, sample_idx) (PushObj chunks)."""
    out = collections.defaultdict(dict)
    for f in sorted(Path(run_dir).glob("*/per_iter_logs.json")):
        for line in f.open():
            if line.strip():
                r = json.loads(line)
                out[(f.parent.name, r["sample_idx"]) if chunked else r["sample_idx"]][r["step"]] = r
    return out


def episodes(run_dir, chunked=False):
    out = {}
    for f in sorted(Path(run_dir).glob("*/stag_episodes.jsonl")):
        for line in f.open():
            if line.strip():
                e = json.loads(line)
                out[(f.parent.name, e["sample_idx"]) if chunked else e["sample_idx"]] = e
    return out


def reference(env, refs):
    """Reference records; with several runs, a later run's episodes replace the earlier run's."""
    out = {}
    for d in refs:
        out.update(records(ROOT / d, env == "pushobj"))
    return out


def ep_name(k):
    return f"{k[0]}/{k[1]}" if isinstance(k, tuple) else k


def first_success(steps):
    s = [k for k, r in steps.items() if r["success"]]
    return min(s) if s else None


def mcnemar(w, l):
    n = w + l
    if n == 0:
        return 1.0
    k = min(w, l)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def check(a):
    ref, ch = reference(a.env, a.ref), a.env == "pushobj"
    ok = True
    for arm in a.arms:
        rec, eps = records(ROOT / a.root / a.tag / arm, ch), episodes(ROOT / a.root / a.tag / arm, ch)
        bad = n = 0
        for i, steps in rec.items():
            e = eps.get(i, {})
            sw = e.get("switch_step") if "switch_step" in e else e.get("first_trigger")  # StagnationMPCPlanner arms
            last = sw if sw is not None else max(steps)
            for s in range(1, last + 1):
                n += 1
                q, r = ref[i].get(s), steps.get(s)
                # stag_r only where the reference logged it (PushObj's 30-replan run planned without the trigger)
                if q is None or r is None or any(r.get(f) != q.get(f) for f in ("act", "success", "loss_new_cur")) or \
                        ("stag_r" in q and r.get("stag_r") != q["stag_r"]):
                    bad += 1
        print(f"{arm}: {len(rec)} episodes; {n} replans up to the switch compared with the reference; mismatches {bad}; "
              f"switch steps {[eps.get(i, {}).get('switch_step') for i in sorted(rec)]}")
        ok &= bad == 0
    if a.same:
        A, B = (records(ROOT / a.root / a.tag / x, ch) for x in a.same)
        diff = sum(1 for i in A for s in set(A[i]) | set(B.get(i, {}))
                   if A[i].get(s, {}).get("act") != B.get(i, {}).get(s, {}).get("act"))
        print(f"{a.same[0]} vs {a.same[1]}: {sum(len(v) for v in A.values())} replans; differing actions {diff}")
        ok &= diff == 0
    sys.exit(0 if ok else 1)


def e0_class(level):
    if level not in E0:
        return {}
    out = {}
    for line in (ROOT / E0[level]).open():
        x = json.loads(line)
        if x["base"] == "adajepa":
            out[x["i"]] = "成功" if x["success"] else x["cls"]
    return out


def analyze(a):
    ref, ch = reference(a.env, a.ref), a.env == "pushobj"
    n_all = len(ref)
    ref_fs = {i: first_success(st) for i, st in ref.items()}
    cls = e0_class(a.level) if a.env == "maze" else {}
    L = [f"# E3：{a.tag}，{a.name}\n",
         f"参照运行：{' + '.join(f'`{r}`' for r in a.ref)}（AdaJEPA，{n_all} 个 episode）。"
         + ("各组只重跑参照里触发过卡住的 episode" if a.env == "maze" else "各组只重跑参照里第 10 次之前没成功的 episode")
         + "，其余与参照相同。"
         "赢 = 参照失败、这一组成功；输 = 参照成功、这一组失败；p 为 McNemar 精确双侧检验。\n",
         "| 组 | 重跑 | 赢 | 输 | p | " + " | ".join(f"{b} 次：成功（参照）" for b in BUDGETS) + " | 切换的 episode | 切换步中位 |",
         "|---|---:|---:|---:|---:|" + "---|" * len(BUDGETS) + "---:|---:|"]
    S, detail = {}, []
    for arm in a.arms:
        d = ROOT / a.root / a.tag / arm
        rec, eps = records(d, ch), episodes(d, ch)
        fs = dict(ref_fs)
        fs.update({i: first_success(st) for i, st in rec.items()})
        wins = [i for i in rec if ref_fs[i] is None and fs[i] is not None]
        losses = [i for i in rec if ref_fs[i] is not None and fs[i] is None]
        p = mcnemar(len(wins), len(losses))
        bud = {b: (sum(v is not None and v <= b for v in fs.values()), sum(v is not None and v <= b for v in ref_fs.values()))
               for b in BUDGETS}
        sw = [eps[i]["switch_step"] for i in rec if i in eps and eps[i].get("switch_step") is not None]
        L.append(f"| {arm} | {len(rec)} | {len(wins)} | {len(losses)} | {p:.3f} | " +
                 " | ".join(f"{x}（{y}）" for x, y in bud.values()) +
                 f" | {len(sw)}/{len(rec)} | {np.median(sw) if sw else '—'} |")
        S[arm] = {"rerun": len(rec), "wins": [ep_name(i) for i in wins], "losses": [ep_name(i) for i in losses], "p": p,
                  "budgets": bud,
                  "switched": len(sw), "switch_median": float(np.median(sw)) if sw else None}
        for i in sorted(wins + losses):
            detail.append(f"| {arm} | {ep_name(i)} | {'赢' if i in wins else '输'} | {cls.get(i, '—')} | "
                          f"{eps.get(i, {}).get('switch_step')} | {ref_fs[i]} | {fs[i]} |")
    L.append("")
    if detail:
        L += ["变化的 episode（E0 类别是参照运行里的结局）：\n",
              "| 组 | episode | 赢/输 | E0 类别 | 切换步 | 参照成功于 | 这一组成功于 |", "|---|---:|---|---|---:|---:|---:|"] + detail
    out = ROOT / a.report / a.tag
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.name}.md").write_text("\n".join(L) + "\n")
    (out / f"{a.name}.json").write_text(json.dumps(S, indent=1))
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--env", default="maze", choices=("maze", "pushobj"))
    ap.add_argument("--level", default="medium")
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--stream", default="S7", help="PushObj target set")
    ap.add_argument("--ref", nargs="+", default=None,
                    help="default eval_outputs/stag_maze100_ar4/<level>/adajepa/ref (maze), PUSH_REFS[stream] (PushObj)")
    ap.add_argument("--root", default="eval_outputs/e3")
    ap.add_argument("--report", default="reports/e3")
    ap.add_argument("--name", default="stage1")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--same", nargs=2, default=None)
    a = ap.parse_args()
    if a.env == "maze":
        a.ref = a.ref or [f"eval_outputs/stag_maze100_ar4/{a.level}/adajepa/ref"]
        a.tag = f"{a.level}_s{a.seed}"
    else:
        a.ref = a.ref or PUSH_REFS[a.stream]
        a.tag = f"pushobj_{a.stream}"
    check(a) if a.check else analyze(a)


if __name__ == "__main__":
    main()
