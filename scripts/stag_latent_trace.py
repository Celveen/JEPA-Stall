"""Latent-space motion traces of a reference planning run, for the stall regulariser's trigger.

Every logged state of every episode (plus the initial state and the goal) is rendered and encoded
with the base model, and per replan s the trace records, in the planning objective's metric
d(z, z') = sqrt(MSE_visual + alpha * MSE_proprio) (and in the proprio latent alone):

    dW    d(z_s, z_{s-W})              latent movement over the last W replans
    dg    d(z_s, z_goal)               distance left to the goal
    d1    d(z_s, z_{s-1})              one-replan step
    cos   cosine of consecutive latent velocities z_s - z_{s-1}, z_{s-1} - z_{s-2}
          (temporal straightening's curvature; negative when the scene goes back and forth)

Frames are rendered at the logged positions with zero velocity (the frame only depends on positions;
a velocity would drift the pusher during the env's settling step) and the proprio input is the logged
position and velocity, so the encoded observation matches what the planner saw.
Pixel-space stall flags (pusher moved < 10 / 15 px over W replans) are kept for comparison only.

    python scripts/stag_latent_trace.py --ref eval_outputs/escape_S7/none --stream S7 --out eval_outputs/stag_S7/trace.jsonl
"""
import argparse
import json

from stag_common import ROOT, encode, goal_state, load_world_model, ref_episodes, render, stream_targets

import numpy as np  # noqa: E402
import torch  # noqa: E402


def frame(shape, st):
    s = np.asarray(st, dtype=np.float64).copy()
    s[5:] = 0.0
    obs, _ = render(shape, s)
    return obs["visual"], np.r_[np.asarray(st, dtype=np.float64)[:2], np.asarray(st, dtype=np.float64)[5:7]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--stream", required=True)
    ap.add_argument("--max-iter", type=int, default=30)
    ap.add_argument("--window", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    wm, pre, _ = load_world_model()
    eps = ref_episodes(a.ref, a.max_iter)
    targets = {}
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    W = a.window
    with out.open("w") as f:
        for (ch, i), rs in sorted(eps.items()):
            if ch not in targets:
                targets[ch] = stream_targets(a.stream, ch)
            seg = targets[ch][i]
            g, gobs = goal_state(seg)
            states = [seg["init"]] + [np.asarray(r["state"], dtype=np.float64) for r in rs]
            vis, prop = zip(*[frame(seg["shape"], st) for st in states])
            zv, zp = encode(wm, pre, np.stack(list(vis) + [gobs["visual"]]),
                            np.stack(list(prop) + [np.asarray(gobs["proprio"], dtype=np.float64)]))
            zv, zp = zv.double(), zp.double()
            zgv, zgp = zv[-1], zp[-1]
            zv, zp = zv[:-1], zp[:-1]

            def d(i1, i2=None):
                av, ap_ = zv[i1], zp[i1]
                bv, bp = (zgv, zgp) if i2 is None else (zv[i2], zp[i2])
                return (float(((av - bv) ** 2).mean() + a.alpha * ((ap_ - bp) ** 2).mean()) ** 0.5,
                        float(((ap_ - bp) ** 2).mean()) ** 0.5)

            rows = []
            for s in range(1, len(states)):
                row = {"s": s, "success": bool(rs[s - 1]["success"]), "d1": d(s, s - 1)[0], "dg": d(s)[0],
                       "dg_p": d(s)[1]}
                if s >= W:
                    dw, dwp = d(s, s - W)
                    row.update({"dW": dw, "dW_p": dwp,
                                "px": float(np.linalg.norm(states[s][:2] - states[s - W][:2]))})
                if s >= 2:
                    v1 = torch.cat([zv[s] - zv[s - 1], a.alpha ** 0.5 * (zp[s] - zp[s - 1])])
                    v0 = torch.cat([zv[s - 1] - zv[s - 2], a.alpha ** 0.5 * (zp[s - 1] - zp[s - 2])])
                    row["cos"] = float(torch.nn.functional.cosine_similarity(v1, v0, dim=0))
                rows.append({k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()})
            succ = next((r["step"] for r in rs if r["success"]), None)
            f.write(json.dumps({"chunk": ch, "sample_idx": i, "shape": seg["shape"], "success_step": succ,
                                "dg0": round(d(0)[0], 6), "steps": rows}) + "\n")
    print(f"{len(eps)} episodes -> {out}")


if __name__ == "__main__":
    main()
