"""Oracle diagnostic: where does AdaJEPA's stall come from? (decides whether a training-time fix is warranted)

Each AdaJEPA failure is restarted from its first stall state (latent trigger, scripts/stag_select.py) and
planned to the end of its replan budget with the world model replaced by the true simulator, using CEM:

    O1  simulator + the planning objective in latent space (base-model encodings of the rendered frames;
        the objective's frame weights over the H/5 + 1 replan boundaries), horizon 25 env steps
    O2  simulator + the true state cost: joint pusher/object position error + angle error (the success
        test's 20 px <-> 20 deg), same frame weights, horizon 25
    O3  O2 with a 50-step horizon

    O4_<V>_l<lam>  O1 plus a terminal cost-to-go mu * V(z_end, z_goal) from a trained head (--ctg name=path),
        mu = lam * D(z_now, z_goal) / max(V(z_now, z_goal), 1): at the current state the terminal term equals
        lam times the current objective (F3 of the cost-to-go plan)

    O1 succeeds       -> the stall comes from the model's predictions (a training-side fix could help)
    O1 fails, O2 not  -> the latent objective misleads (training-side: shape the latent metric)
    O2 fails, O3 not  -> the task cost is myopic at this horizon (planner side: look beyond the horizon)

Execution replays the same simulator (5 env steps per replan); success uses the env's own test.
CEM: N samples per iteration (the warm start and the do-nothing plan always included), K iterations,
E elites, best sample executed; the warm start is the previous best plan shifted by one replan.

    python scripts/stag_oracle.py --variant O2 --workers 32 --out eval_outputs/stag_EVAL50/oracle/O2.jsonl
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):  # one core per worker process
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

from stag_common import ROOT, goal_state, ref_episodes, stream_targets

ANG_PX = 20.0 / (np.pi / 9)  # 20 deg of angle error counts like 20 px of position error
VARIANTS = {"O1": ("latent", 25), "O2": ("state", 25), "O3": ("state", 50)}


def variant(name):
    """(kind, horizon, cost-to-go model name or None, lam) of a variant name."""
    if name.startswith("O4_"):
        _, model, lam = name.split("_")
        return "latent", 25, model, float(lam.lstrip("l"))
    return VARIANTS[name] + (None, 0.0)
_WM = {}  # per worker process: device -> (wm, preprocessor)
_CTG = {}  # per worker process: (device, path) -> cost-to-go head


def angdiff(a, b):
    return np.abs((a - b + np.pi) % (2 * np.pi) - np.pi)


class Sim:
    """Physics of one PushObj env without rendering or reward: exactly env.step's PD loop."""

    def __init__(self, shape, state):
        from stag_common import make_env
        from pymunk.vec2d import Vec2d
        self.Vec2d = Vec2d
        self.env = make_env(shape)
        self.env.prepare(0, np.asarray(state, dtype=np.float64))  # fresh space at the start state

    def restore(self, st):
        e = self.env
        e.block.velocity, e.block.angular_velocity = (0, 0), 0
        e._set_state(np.asarray(st[:7], dtype=np.float64))  # as the env's reset does (one settling step)

    def state(self):
        return np.asarray(self.env._get_obs(), dtype=np.float64)

    def step(self, a_px):
        e, dt = self.env, 1.0 / self.env.sim_hz
        target = e.agent.position + self.Vec2d(float(a_px[0]), float(a_px[1]))
        for _ in range(e.sim_hz // e.control_hz):
            e.agent.velocity += (e.k_p * (target - e.agent.position) + e.k_v * (self.Vec2d(0, 0) - e.agent.velocity)) * dt
            e.space.step(dt)

    def rollout(self, st, acts, every=5, frames=False):
        """States (and frames) at the replan boundaries of an action sequence (H, 2) from state st."""
        self.restore(st)
        sts, frs = [self.state()], []
        if frames:
            frs.append(self.env._render_frame("rgb_array"))
        for t, a in enumerate(acts, 1):
            self.step(a)
            if t % every == 0:
                sts.append(self.state())
                if frames:
                    frs.append(self.env._render_frame("rgb_array"))
        return np.stack(sts), frs


def run_episode(job):
    import cv2
    import torch
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    a, ep = job["args"], job["ep"]
    kind, H, ctg_name, lam = variant(a["variant"])
    rng = np.random.default_rng([a["seed"], ep["key_hash"]])
    seg = stream_targets(a["stream"], ep["chunk"])[ep["sample_idx"]]
    g, gobs = goal_state(seg)
    sim = Sim(seg["shape"], ep["state"])
    T = H // 5 + 1
    wts = 2.0 ** np.arange(T)
    wts /= wts.sum()
    if kind == "latent":
        from stag_common import encode, load_world_model
        dev = f"cuda:{(mp.current_process()._identity[0] - 1) % job['n_gpus']}"  # one GPU per worker process
        if dev not in _WM:
            _WM[dev] = load_world_model(device=dev)[:2]
        wm, pre = _WM[dev]
        zgv, zgp = encode(wm, pre, gobs["visual"][None], np.asarray(gobs["proprio"], dtype=np.float64)[None], device=dev)
        V = None
        if ctg_name is not None:
            import torch
            from planning.costtogo import load_costtogo
            path = str(ROOT / dict(kv.split("=") for kv in a["ctg"])[ctg_name])
            if (dev, path) not in _CTG:
                _CTG[(dev, path)] = load_costtogo(path, dev)
            V = _CTG[(dev, path)]
            gvd, gpd = zgv.to(dev), zgp.to(dev)

    def cost(cands, st):
        if kind == "state":
            out = []
            for A in cands:
                S, _ = sim.rollout(st, A)
                c = np.linalg.norm(S[:, :4] - g[:4], axis=1) + ANG_PX * angdiff(S[:, 4], g[4])
                out.append(float((wts * c).sum()))
            return np.array(out)
        vis, prop = [], []
        for A in cands:
            S, F = sim.rollout(st, A, frames=True)
            vis += F
            prop += [np.r_[s[:2], s[5:7]] for s in S]
        zv, zp = encode(wm, pre, np.stack(vis), np.stack(prop), device=dev, batch=64)
        c = ((zv - zgv) ** 2).mean(1) + a["alpha"] * ((zp - zgp) ** 2).mean(1)
        total = (c.view(len(cands), T).double().numpy() * wts).sum(1)
        if V is not None:
            with torch.no_grad():
                ev = zv.view(len(cands), T, -1)[:, -1].to(dev)  # last replan boundary of each candidate
                ep_ = zp.view(len(cands), T, -1)[:, -1].to(dev)
                v_end = V(ev, ep_, gvd, gpd).double().cpu().numpy()
                v_now = float(V(zv[:1].to(dev), zp[:1].to(dev), gvd, gpd))
            stay = float(c.view(len(cands), T)[0, 0])  # D(z_now, z_goal): frame 0 is the current state
            mu = lam * stay / max(v_now, 1.0)
            cost.last = {"v_now": round(v_now, 3), "mu": round(mu, 6)}
            total = total + mu * v_end
        return total

    st = np.asarray(ep["state"], dtype=np.float64)
    sim.restore(st)
    st = sim.state()
    mean = np.zeros((H, 2))
    log, success, t0 = [], False, time.time()
    for k in range(ep["budget"]):
        m, sd = mean.copy(), np.full((H, 2), a["sigma0"])
        best_c, best_a, stay_c = np.inf, mean, None
        for it in range(a["iters"]):
            cands = m + sd * rng.standard_normal((a["n"], H, 2))
            cands[0], cands[1] = mean, 0.0  # warm start, do nothing
            c = cost(cands, st)
            if it == 0:
                stay_c = float(c[1])
            order = np.argsort(c)
            if c[order[0]] < best_c:
                best_c, best_a = float(c[order[0]]), cands[order[0]].copy()
            el = cands[order[: a["elites"]]]
            m, sd = el.mean(0), np.maximum(el.std(0), a["sigma_min"])
        sim.restore(st)
        for x in best_a[:5]:
            sim.step(x)
        st = sim.state()
        pos_err = float(np.linalg.norm(st[:4] - g[:4]))
        ang = float(angdiff(st[4], g[4]))
        success = pos_err < 20 and ang < np.pi / 9
        log.append({**getattr(cost, "last", {}), "k": k + 1, "best": round(best_c, 5), "stay": round(stay_c, 5), "pos_err": round(pos_err, 2),
                    "obj_err": round(float(np.linalg.norm(st[2:4] - g[2:4])), 2), "ang_deg": round(np.degrees(ang), 2),
                    "state": np.round(st, 2).tolist()})
        if success:
            break
        mean = np.concatenate([best_a[5:], np.zeros((5, 2))])
    s0 = np.asarray(ep["state"], dtype=np.float64)

    def side(s):  # cosine between (pusher - object) and (object goal - object): > 0 pusher in front, < 0 behind
        u, v = s[:2] - s[2:4], g[2:4] - s[2:4]
        return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9))
    return {"chunk": ep["chunk"], "sample_idx": ep["sample_idx"], "variant": a["variant"], "start_step": ep["start_step"],
            "budget": ep["budget"], "success": bool(success), "replans": len(log),
            "obj_err0": round(float(np.linalg.norm(s0[2:4] - g[2:4])), 2), "ang0_deg": round(float(np.degrees(angdiff(s0[4], g[4]))), 2),
            "side0": round(side(s0), 3), "side_min": round(min(side(np.asarray(r["state"])) for r in log), 3),
            "seconds": round(time.time() - t0, 1), "log": log}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, help="O1, O2, O3 or O4_<ctg name>_l<lam>")
    ap.add_argument("--ctg", nargs="*", default=[], help="name=path of cost-to-go heads for O4 variants")
    ap.add_argument("--stream", default="EVAL50")
    ap.add_argument("--ref", default="eval_outputs/escape_EVAL50/none")
    ap.add_argument("--touched", default="eval_outputs/stag_EVAL50/touched.json")
    ap.add_argument("--max-iter", type=int, default=30)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--elites", type=int, default=32)
    ap.add_argument("--sigma0", type=float, default=25.0, help="initial CEM std (px per env step)")
    ap.add_argument("--sigma-min", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--gpus", default="4,5,6,7", help="O1 only")
    ap.add_argument("--limit", type=int, default=0, help="first N episodes only (smoke tests)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    touched = json.loads((ROOT / a.touched).read_text())
    eps = ref_episodes(a.ref, a.max_iter)
    jobs, gpus = [], a.gpus.split(",")
    for row in touched["episodes"]:
        s = row["first_trigger_offline"]
        if row["ref_success_step"] is not None or s is None:
            continue  # AdaJEPA failures whose (base-encoder) trigger fires
        rec = next(r for r in eps[(row["chunk"], row["sample_idx"])] if r["step"] == s)
        ep = {"chunk": row["chunk"], "sample_idx": row["sample_idx"], "start_step": s, "budget": a.max_iter - s,
              "state": rec["state"], "key_hash": int(row["chunk"].split("_")[1]) * 100 + row["sample_idx"]}
        jobs.append({"args": vars(a), "ep": ep, "n_gpus": len(gpus)})
    if a.limit:
        jobs = jobs[: a.limit]
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {(r["chunk"], r["sample_idx"]) for r in map(json.loads, out.open())}
    jobs = [j for j in jobs if (j["ep"]["chunk"], j["ep"]["sample_idx"]) not in done]
    if variant(a.variant)[0] == "latent":
        os.environ["CUDA_VISIBLE_DEVICES"] = a.gpus
    print(f"{a.variant}: {len(jobs)} episodes to plan ({len(done)} done) with {a.workers} workers", flush=True)
    with mp.get_context("spawn").Pool(a.workers) as pool, out.open("a") as f:
        for res in pool.imap_unordered(run_episode, jobs):
            f.write(json.dumps(res) + "\n")
            f.flush()
            print(f"{res['chunk']}/{res['sample_idx']}: success={res['success']} replans={res['replans']} "
                  f"obj_err {res['obj_err0']}->{res['log'][-1]['obj_err']} ({res['seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
