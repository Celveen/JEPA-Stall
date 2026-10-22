"""Offline feasibility checks F1 and F2 for the cost-to-go head.

F1  accuracy: on encoded splits (seen shapes: the base model's val split; unseen shapes: held-out
    trajectories generated only for this check), for same-trajectory pairs (s, g later): Spearman between
    V(s, g) and the true model-step count, mean absolute error and bias, on encoded states and on the
    world model's predictions of them; and how often V falls from one model step to the next on the way
    to the trajectory's last frame.
F2  direction: at AdaJEPA's object-off stall states (oracle diagnostic, scripts/stag_oracle.py), the
    pusher is placed around the object in `dirs` directions just clear of it (object untouched), the
    scenes are rendered and encoded, and each scorer (V of every model, and the planning objective's
    latent distance to the goal) picks its best placement. "behind" = the placement lies on the object's
    far side from its goal (cos(pusher - object, object goal - object) < -0.3), where a push moves the
    object toward its goal; chance = the share of such placements on the ring.

    python scripts/ctg_check.py --models mc=eval_outputs/ctg/V_mc.pt td=eval_outputs/ctg/V_td.pt
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
from ctg_train import FS, Split, spearman  # noqa: E402
from planning.costtogo import load_costtogo  # noqa: E402
from stag_common import encode, goal_state, load_world_model, ref_episodes, render, stream_targets  # noqa: E402

UNSEEN = {"I", "square", "small_tee"}


@torch.no_grad()
def f1(models, split, gen, n=20000):
    res = {}
    shapes = np.array(split.shapes)
    for name, frac in (("enc", 0.0), ("pred", 1.0)):
        s, zv, zp = split.sample_states(n, frac, gen)
        g = split.future(s, gen)
        k = (g - s).float() / FS
        gv, gp = split.z(g)
        sh = shapes[split.ep_of[s].cpu().numpy()]
        for m, model in models.items():
            v = model(zv, zp, gv, gp)
            r = {"all": {"spearman": round(spearman(v, k), 3), "mae": round(float((v - k).abs().mean()), 2),
                         "bias": round(float((v - k).mean()), 2)}}
            for shp in sorted(set(sh)):
                mk = torch.as_tensor(sh == shp, device=v.device)
                r[shp] = {"spearman": round(spearman(v[mk], k[mk]), 3), "mae": round(float((v[mk] - k[mk]).abs().mean()), 2)}
            res.setdefault(m, {})[name] = r
    # monotone progress toward each trajectory's last frame
    eps = torch.randperm(len(split.start), generator=torch.Generator().manual_seed(0))[:500]
    for m, model in models.items():
        dec, tot = 0, 0
        for e in eps.tolist():
            st, L = int(split.start[e]), int(split.len[e])
            idx = torch.arange(st, st + L, FS, device=split.vis.device)
            last = torch.tensor([st + L - 1], device=idx.device)
            v = model(*split.z(idx), *split.z(last))
            dec += int((v[1:] < v[:-1]).sum())
            tot += len(v) - 1
        res[m]["monotone"] = round(dec / max(tot, 1), 3)
    return res


def ring_states(shape, st, dirs, lo=12.0, hi=220.0, step=4.0):
    """Pusher placements around the object, just clear of it, with the object exactly where it is."""
    _, st0 = render(shape, np.r_[st[:5], 0.0, 0.0])
    out = []
    for k in range(dirs):
        u = np.array([np.cos(2 * np.pi * k / dirs), np.sin(2 * np.pi * k / dirs)])
        for d in np.arange(lo, hi, step):
            s = np.r_[st0[2:4] + d * u, st0[2:5], 0.0, 0.0]
            if s[:2].min() < 22 or s[:2].max() > 489:
                break
            obs, s1 = render(shape, s)
            if (np.abs(s1[2:4] - st0[2:4]).max() < 0.3 and abs(s1[4] - st0[4]) < 0.003
                    and np.linalg.norm(s1[:2] - s[:2]) < 1.0):
                obs, s1 = render(shape, np.r_[st0[2:4] + (d + step) * u, st0[2:5], 0.0, 0.0])
                out.append((obs, s1))
                break
    return out


def side(s, g):
    u, v = s[:2] - s[2:4], g[2:4] - s[2:4]
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9))


@torch.no_grad()
def f2(models, wm, pre, a, device):
    rows = [json.loads(line) for line in (ROOT / a.oracle).open()]
    ref = ref_episodes(a.ref, 30)
    stalls = [r for r in rows if not (r["obj_err0"] < 20 and r["ang0_deg"] < 20)]  # object off
    tally = {m: [] for m in list(models) + ["latent_dist"]}
    chance, per = [], []
    for r in stalls:
        seg = stream_targets(a.stream, r["chunk"])[r["sample_idx"]]
        g, gobs = goal_state(seg)
        st = np.asarray(next(x for x in ref[(r["chunk"], r["sample_idx"])] if x["step"] == r["start_step"])["state"], dtype=np.float64)
        ring = ring_states(seg["shape"], st, a.dirs)
        if len(ring) < a.dirs // 2:
            continue
        cur, _ = render(seg["shape"], np.r_[st[:5], 0.0, 0.0])
        frames = [o["visual"] for o, _ in ring] + [cur["visual"], gobs["visual"]]
        props = [np.r_[s[:2], 0.0, 0.0] for _, s in ring] + [np.r_[st[:2], st[5:7]], np.asarray(gobs["proprio"], dtype=np.float64)]
        zv, zp = encode(wm, pre, np.stack(frames), np.stack(props), device=device)
        zv, zp = zv.to(device), zp.to(device)
        gv, gp = zv[-1:], zp[-1:]
        sides = np.array([side(s, g) for _, s in ring])
        behind = sides < -0.3
        chance.append(float(behind.mean()))
        scores = {m: model(zv[:-1], zp[:-1], gv, gp).cpu().numpy() for m, model in models.items()}
        scores["latent_dist"] = (((zv[:-1] - gv) ** 2).mean(1) + a.alpha * ((zp[:-1] - gp) ** 2).mean(1)).cpu().numpy()
        rec = {"chunk": r["chunk"], "sample_idx": r["sample_idx"], "shape": seg["shape"], "n_ring": len(ring),
               "chance": float(behind.mean())}
        for m, sc in scores.items():
            ring_sc, cur_sc = sc[:-1], sc[-1]
            best = int(np.argmin(ring_sc))
            tally[m].append((bool(behind[best]), bool(ring_sc[best] < cur_sc), seg["shape"] in UNSEEN))
            rec[m] = {"best_side": round(float(sides[best]), 3), "behind": bool(behind[best]),
                      "beats_stall": bool(ring_sc[best] < cur_sc),
                      "rank_corr_with_side": round(float(np.corrcoef(ring_sc, sides)[0, 1]), 3)}
        per.append(rec)
    summ = {"n": len(per), "chance_behind": round(float(np.mean(chance)), 3)}
    for m, t in tally.items():
        t = np.array(t)
        summ[m] = {"behind": round(float(t[:, 0].mean()), 3),
                   "behind_unseen": round(float(t[t[:, 2], 0].mean()), 3) if t[:, 2].any() else None,
                   "behind_seen": round(float(t[~t[:, 2], 0].mean()), 3) if (~t[:, 2]).any() else None,
                   "best_ring_beats_stall": round(float(t[:, 1].mean()), 3)}
    return summ, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="name=path")
    ap.add_argument("--seen", default="eval_outputs/ctg/bias16k_val")
    ap.add_argument("--unseen", default="eval_outputs/ctg/unseen_val")
    ap.add_argument("--oracle", default="eval_outputs/stag_EVAL50/oracle/O2.jsonl")
    ap.add_argument("--ref", default="eval_outputs/escape_EVAL50/none")
    ap.add_argument("--stream", default="EVAL50")
    ap.add_argument("--dirs", type=int, default=24)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--skip", nargs="*", default=[], help="f1 and/or f2")
    ap.add_argument("--out", default="reports/ctg_offline/checks.json")
    a = ap.parse_args()
    dev = torch.device(a.device)
    models = {kv.split("=")[0]: load_costtogo(ROOT / kv.split("=")[1], dev) for kv in a.models}
    res = {}
    if "f1" not in a.skip:
        gen = torch.Generator(device=dev).manual_seed(0)
        res["F1"] = {name: f1(models, Split(path, dev), gen) for name, path in (("seen", a.seen), ("unseen", a.unseen))
                     if (ROOT / path / "vis.npy").exists()}
        print(json.dumps(res["F1"], indent=1), flush=True)
    if "f2" not in a.skip:
        wm, pre, _ = load_world_model(device=a.device)
        res["F2"], per = f2(models, wm, pre, a, a.device)
        print(json.dumps(res["F2"], indent=1), flush=True)
        res["F2_per_state"] = per
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
