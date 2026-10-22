"""Solvability ceiling of AdaJEPA's stall states (diagnostic only).

Two privileged solvers restart every stall state of the oracle diagnostic (scripts/stag_oracle.py rows:
AdaJEPA failures at their first trigger) in the true simulator with the true state, and count the env
steps they need to pass the env's own success test, checked every 5 env steps as in the planner:

    P1  long-horizon search: CEM over `p1_h` env steps parameterised per replan (one offset held for 5 env
        steps), cost = closest approach to the goal along the rollout + 0.25 x final error (true joint
        pusher/object position error + 20 px per 20 deg of angle), MPC executing one replan at a time
    P2  go-around-and-push macro search: contact points around the object (`dirs`), an approach around a
        safe circle (both ways when the straight path would hit the object), push directions toward the
        object's goal (0, +-20, +-40 deg) and along the contact normal (0, +-30, +-60 deg), push lengths;
        every macro is simulated in closed loop from the current state and the one leaving the object
        closest to its goal pose (plus 0.05 per env step) is executed; before every macro the solver
        checks whether bringing the pusher straight home would already pass, and it stops (pusher home)
        when no macro improves the object pose

Neither is a method: they use the simulator, the true state and (P2) the task's structure, only to tell
whether the stalls can be solved within the episode's remaining replans and within `max_replans`.
Both are deterministic and budget-independent, so one run with max_replans answers both budgets.

    python scripts/stag_ceiling.py --workers 48 --out eval_outputs/stag_EVAL50/ceiling.jsonl
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import multiprocessing as mp  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

from stag_common import ROOT, goal_state, ref_episodes, stream_targets  # noqa: E402
from stag_oracle import ANG_PX, Sim, angdiff  # noqa: E402

PUSHER_R = 15.0
LO, HI = 22.0, 489.0  # pusher centre bounds: walls at 5 / 506 (radius 2) + pusher radius


def cost_joint(S, g):
    return np.linalg.norm(S[..., :4] - g[:4], axis=-1) + ANG_PX * angdiff(S[..., 4], g[4])


def cost_obj(s, g):
    return float(np.linalg.norm(s[2:4] - g[2:4]) + ANG_PX * angdiff(s[4], g[4]))


def success(s, g):
    return bool(np.linalg.norm(s[:4] - g[:4]) < 20 and angdiff(s[4], g[4]) < np.pi / 9)


class ExactSim(Sim):
    """Sim with exact snapshots (velocities included, no settling step), so candidates and the executed
    trajectory branch from the same physical state."""

    def snap(self):
        a, b = self.env.agent, self.env.block
        return (a.position, a.velocity, b.position, b.angle, b.velocity, b.angular_velocity)

    def load(self, s):
        a, b = self.env.agent, self.env.block
        a.position, a.velocity = s[0], s[1]
        b.angle, b.position = s[3], s[2]
        b.velocity, b.angular_velocity = s[4], s[5]


class Run:
    """The executed trajectory: env steps used, success checked at every 5-step boundary."""

    def __init__(self, sim, g, budget):
        self.sim, self.g, self.budget = sim, g, budget
        self.steps, self.solved_at = 0, None

    def execute(self, acts):
        for a in acts:
            if self.done:
                return
            self.sim.step(a)
            self.steps += 1
            if self.steps % 5 == 0 and success(self.sim.state(), self.g):
                self.solved_at = self.steps

    def idle_to_boundary(self):
        while not self.done and self.steps % 5:
            self.execute([np.zeros(2)])

    @property
    def done(self):
        return self.solved_at is not None or self.steps >= self.budget


# ---------------------------------------------------------------------------------------------- P1
def solve_p1(sim, g, run, a, rng):
    D = a["p1_h"] // 5
    mean = np.zeros((D, 2))
    while not run.done:
        base, st = sim.snap(), sim.state()
        m, sd = mean.copy(), np.full((D, 2), a["p1_sigma0"])
        best_c, best = np.inf, mean
        for _ in range(a["p1_iters"]):
            cands = m + sd * rng.standard_normal((a["p1_n"], D, 2))
            cands[0], cands[1] = mean, 0.0
            costs = np.empty(len(cands))
            for i, A in enumerate(cands):
                sim.load(base)
                S = []
                for x in A:
                    for _ in range(5):
                        sim.step(x)
                    S.append(sim.state())
                c = cost_joint(np.stack(S), g)
                costs[i] = c.min() + 0.25 * c[-1]
            order = np.argsort(costs)
            if costs[order[0]] < best_c:
                best_c, best = float(costs[order[0]]), cands[order[0]].copy()
            el = cands[order[: a["p1_elites"]]]
            m, sd = el.mean(0), np.maximum(el.std(0), a["p1_sigma_min"])
        sim.load(base)
        run.execute([best[0]] * 5)
        mean = np.concatenate([best[1:], np.zeros((1, 2))])
    return {}


# ---------------------------------------------------------------------------------------------- P2
def geometry(env):
    from shapely.ops import unary_union
    from env.pusht.pusht_env import pymunk_to_shapely
    poly = unary_union(pymunk_to_shapely(env.block, env.block.shapes))
    c = np.array(env.block.position, dtype=np.float64)
    r = max(np.linalg.norm(np.array(p) - c) for p in poly.exterior.coords) if poly.geom_type == "Polygon" else \
        max(np.linalg.norm(np.array(p) - c) for gm in poly.geoms for p in gm.exterior.coords)
    return poly, c, r


def contact_point(poly, c, u):
    """Pusher centre just clear of the object along direction u from its position."""
    from shapely.geometry import Point
    d = 1.0
    while Point(*(c + d * u)).buffer(PUSHER_R).intersects(poly) and d < 400:
        d += 2.0
    return c + (d + 2.0) * u


def clear_segment(p, q, poly):
    from shapely.geometry import LineString
    return LineString([tuple(p), tuple(q)]).distance(poly) > PUSHER_R + 1


def approach(pos, target, poly, c, r_safe, side):
    """Waypoints from pos to target that stay clear of the object (around the safe circle if needed)."""
    if clear_segment(pos, target, poly):
        return [target]
    a0 = math.atan2(*(pos - c)[::-1])
    a1 = math.atan2(*(target - c)[::-1])
    delta = (a1 - a0) % (2 * math.pi)
    if side < 0:
        delta -= 2 * math.pi
    n = max(2, int(abs(delta) * r_safe / 12))
    pts = [c + r_safe * np.array([math.cos(a0 + delta * k / n), math.sin(a0 + delta * k / n)]) for k in range(n + 1)]
    return [np.clip(p, LO, HI) for p in pts] + [target]


def follow(sim, waypoints, max_steps, step_px=12.0, tol=3.0):
    acts = []
    for w in waypoints:
        while len(acts) < max_steps:
            v = np.asarray(w) - np.array(sim.env.agent.position)
            if np.linalg.norm(v) < tol:
                break
            a = v * min(1.0, step_px / (np.linalg.norm(v) + 1e-9))
            sim.step(a)
            acts.append(a)
    return acts


def push_variants(a):
    for fam, offs in (("goal", (0, 20, -20, 40, -40)), ("normal", (0, 30, -30, 60, -60))):
        for off in offs:
            for L in a["push_lengths"]:
                yield fam, off, L


def go_home(sim, g, a):
    """Actions that take the pusher to its own goal position around the object (shorter way round)."""
    poly, c, r = geometry(sim.env)
    pos = np.array(sim.env.agent.position, dtype=np.float64)
    base, best = sim.snap(), None
    for side in (1, -1):
        sim.load(base)
        acts = follow(sim, approach(pos, np.clip(g[:2], LO, HI), poly, c, r + PUSHER_R + 8, side), a["max_macro_steps"])
        if best is None or len(acts) < len(best):
            best = acts
    sim.load(base)
    return best


def home_passes(sim, g, run, acts):
    """Would going home now pass the success test at the next replan boundaries (simulated, then undone)?"""
    base = sim.snap()
    steps, ok = run.steps, False
    for x in list(acts) + [np.zeros(2)] * 5:
        sim.step(x)
        steps += 1
        if steps % 5 == 0 and success(sim.state(), g):
            ok = True
            break
    sim.load(base)
    return ok


def solve_p2(sim, g, run, a, rng):
    chosen = []
    while not run.done:
        home = go_home(sim, g, a)
        if home_passes(sim, g, run, home):  # object close enough already: just bring the pusher home
            break
        st = sim.state()
        poly, c, r = geometry(sim.env)
        r_safe = r + PUSHER_R + 8
        pos = np.array(sim.env.agent.position, dtype=np.float64)
        to_goal = g[2:4] - c
        goal_ang = math.atan2(to_goal[1], to_goal[0]) if np.linalg.norm(to_goal) > 1 else None
        base, c0 = sim.snap(), cost_obj(st, g)
        best = (np.inf, None, None)
        for k in range(a["dirs"]):
            phi = 2 * math.pi * k / a["dirs"]
            pc = contact_point(poly, c, np.array([math.cos(phi), math.sin(phi)]))
            if pc.min() < LO or pc.max() > HI:
                continue
            for side in ((1,) if clear_segment(pos, pc, poly) else (1, -1)):
                sim.load(base)
                appr = follow(sim, approach(pos, pc, poly, c, r_safe, side), a["max_macro_steps"])
                at_contact = sim.snap()  # every push variant branches from the end of this approach
                for fam, off, L in push_variants(a):
                    if fam == "goal" and goal_ang is None:
                        continue
                    psi = (goal_ang if fam == "goal" else phi + math.pi) + math.radians(off)
                    push_to = np.clip(pc + (L + PUSHER_R + 2) * np.array([math.cos(psi), math.sin(psi)]), LO, HI)
                    sim.load(at_contact)
                    acts = appr + follow(sim, [push_to], a["max_macro_steps"])
                    score = cost_obj(sim.state(), g) + 0.05 * len(acts)
                    if score < best[0]:
                        best = (score, acts, (round(math.degrees(phi)), fam, off, L, side))
        sim.load(base)
        if best[1] is None or best[0] >= c0 - 1.0:
            break  # no macro improves the object pose
        run.execute(best[1])
        chosen.append(best[2])
    # the pusher returns to its own goal position (around the object), then waits for a boundary
    if not run.done:
        run.execute(go_home(sim, g, a))
        run.idle_to_boundary()
    return {"macros": chosen}


def run_job(job):
    import cv2
    import torch
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    a, ep = job["args"], job["ep"]
    seg = stream_targets(a["stream"], ep["chunk"])[ep["sample_idx"]]
    g, _ = goal_state(seg)
    sim = ExactSim(seg["shape"], ep["state"])
    run = Run(sim, g, 5 * a["max_replans"])
    t0 = time.time()
    rng = np.random.default_rng([a["seed"], ep["key"]])
    extra = (solve_p1 if job["method"] == "P1" else solve_p2)(sim, g, run, a, rng)
    s = sim.state()
    rep = None if run.solved_at is None else run.solved_at // 5
    return {"chunk": ep["chunk"], "sample_idx": ep["sample_idx"], "shape": seg["shape"], "method": job["method"],
            "kind": ep["kind"], "budget_R": ep["budget"], "solved_replans": rep,
            "within_R": rep is not None and rep <= ep["budget"], "within_max": rep is not None,
            "obj_err_end": round(float(np.linalg.norm(s[2:4] - g[2:4])), 2),
            "ang_end_deg": round(float(np.degrees(angdiff(s[4], g[4]))), 2),
            "pusher_err_end": round(float(np.linalg.norm(s[:2] - g[:2])), 2),
            "steps": run.steps, "seconds": round(time.time() - t0, 1), **extra}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["P1", "P2"])
    ap.add_argument("--stream", default="EVAL50")
    ap.add_argument("--ref", default="eval_outputs/escape_EVAL50/none")
    ap.add_argument("--oracle", default="eval_outputs/stag_EVAL50/oracle/O2.jsonl", help="the stall states")
    ap.add_argument("--max-replans", type=int, default=60)
    ap.add_argument("--p1-h", type=int, default=100)
    ap.add_argument("--p1-n", type=int, default=256)
    ap.add_argument("--p1-iters", type=int, default=6)
    ap.add_argument("--p1-elites", type=int, default=32)
    ap.add_argument("--p1-sigma0", type=float, default=20.0)
    ap.add_argument("--p1-sigma-min", type=float, default=2.0)
    ap.add_argument("--dirs", type=int, default=24)
    ap.add_argument("--push-lengths", nargs="+", type=float, default=[10, 20, 30, 50])
    ap.add_argument("--max-macro-steps", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ref = ref_episodes(a.ref, 30)
    eps = []
    for r in map(json.loads, (ROOT / a.oracle).open()):
        st = next(x for x in ref[(r["chunk"], r["sample_idx"])] if x["step"] == r["start_step"])["state"]
        kind = "pusher-only" if r["obj_err0"] < 20 and r["ang0_deg"] < 20 else "object-off"
        eps.append({"chunk": r["chunk"], "sample_idx": r["sample_idx"], "state": st, "budget": r["budget"], "kind": kind,
                    "key": int(r["chunk"].split("_")[1]) * 100 + r["sample_idx"]})
    if a.limit:
        eps = eps[: a.limit]
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {(r["method"], r["chunk"], r["sample_idx"]) for r in map(json.loads, out.open())}
    jobs = [{"args": vars(a), "ep": ep, "method": m} for m in a.methods for ep in eps
            if (m, ep["chunk"], ep["sample_idx"]) not in done]
    jobs.sort(key=lambda j: j["method"] != "P1")  # the long P1 jobs first
    print(f"{len(jobs)} jobs ({len(done)} done) on {a.workers} workers", flush=True)
    with mp.get_context("spawn").Pool(a.workers) as pool, out.open("a") as f:
        for res in pool.imap_unordered(run_job, jobs):
            f.write(json.dumps(res) + "\n")
            f.flush()
            print(f"{res['method']} {res['chunk']}/{res['sample_idx']} ({res['kind']}): solved at replan "
                  f"{res['solved_replans']} (budget {res['budget_R']}), obj {res['obj_err_end']} px / {res['ang_end_deg']} deg, "
                  f"{res['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
