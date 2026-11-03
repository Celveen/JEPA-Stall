"""Development experiment: calibrate cost-repair timing while AdaJEPA keeps learning.

replay is an offline diagnostic, never a prediction of intervention success.
prepare writes exact closed-loop jobs. run requires CUDA and verifies pairing.
All new runs live separately from E1/E3; no stopping of model updates is used.
"""
import argparse
import ast
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
from types import SimpleNamespace

VARIANTS = {
    "R-switch": {"mode": "loss_new", "W": 3, "theta": 0.05, "K": 1, "freeze": False},
    "R-w5-t05": {"mode": "loss_new", "W": 5, "theta": 0.05, "K": 1, "freeze": False},
    "R-w3-t01": {"mode": "loss_new", "W": 3, "theta": 0.01, "K": 1, "freeze": False},
    "R-w5-t01": {"mode": "loss_new", "W": 5, "theta": 0.01, "K": 1, "freeze": False},
}
NEW_ARMS = tuple(k for k in VARIANTS if k != "R-switch")
DEFAULT_ROOT = "eval_outputs/switch_calibration_20261102"
DEFAULT_REPORT = "reports/switch_calibration_20261102"
TAGS = {"maze": "medium_s100", "pushobj": "pushobj_S7"}
OLD_REPORTS = {"maze": "reports/e3/medium_s100/stage2.json", "pushobj": "reports/e3/pushobj_S7/stage1.json"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def wilson(k, n):
    if not n:
        return None
    z = 1.959963984540054
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    w = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0, c - w), min(1, c + w)]


def load_modules(repo):
    sys.path.insert(0, str(repo / "scripts"))
    return importlib.import_module("e1_switch"), importlib.import_module("e3_run")


def replay(a, e1):
    cfg = SimpleNamespace(
        maze_labels=None, maze_run="eval_outputs/stag_maze100_ar4/medium/adajepa/ref", maze_budget=50,
        push_trace="eval_outputs/stag_S7/trace.jsonl", push_run="eval_outputs/escape_S7/none",
        push_ext="eval_outputs/stag_S7_it50/S0", push_start=10, push_budget=50,
    )
    data = {"maze": e1.maze(cfg), "pushobj": e1.pushobj(cfg)}
    source_paths = [cfg.maze_run, cfg.push_run, cfg.push_ext]
    paths = [p for d in source_paths for p in sorted((a.repo / d).glob("*/per_iter_logs.json"))]
    paths += [a.repo / cfg.push_trace, a.repo / "scripts/e1_switch.py", a.repo / "planning/switch.py"]
    # Exercise the actual online gate without importing the simulator or allocating a GPU.
    source = ast.parse((a.repo / "planning/switch.py").read_text())
    cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == "SwitchMPCPlanner")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_switch_due")
    namespace = {"np": e1.np}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "planning/switch.py", "exec"), namespace)
    online_gate = namespace["_switch_due"]
    compared = 0
    summary = {"kind": "offline_reference_replay", "budget": 50, "model_updates": "continue",
               "variants": VARIANTS, "datasets": {},
               "source_sha256": {str(p.relative_to(a.repo)): digest(p) for p in paths}}
    old = json.loads((a.repo / "reports/e1_switch/s7_it50/summary.json").read_text())
    old_names = {"maze": "迷宫 medium 修正设定 AdaJEPA", "pushobj": "PushObj S7 AdaJEPA（我们的权重）"}
    lines = ["# 切换参数验证：离线回放", "",
             "固定候选：同一最新步预测误差，W∈{3,5}，θ∈{0.05,0.01}，K=1。模型全程更新。",
             "这是基线轨迹上的信号回放，不是新增闭环成功率，也不证明模型已学不动。",
             "F：触发后最终失败的任务中，至少剩10次预算时切换的比例。",
             "S：触发后原本能成功的任务中，在成功前切换的比例；这不等于实际误伤。",
             "95%区间按任务数计算 Wilson 区间；同一任务的多个窗口不当成独立样本。", "",
             "| 环境 | 候选 | F | S | 切换步中位 | F 95%区间 | S 95%区间 |",
             "|---|---|---|---|---:|---|---|"]
    for env, (eps, budget) in data.items():
        if len(eps) != (100 if env == "maze" else 280):
            raise RuntimeError(f"Incomplete reference for {env}: {len(eps)} episodes")
        result = {}
        for name, v in VARIANTS.items():
            for ep in eps.values():
                gate = SimpleNamespace(sw_mode="loss_new", sw_W=v["W"], sw_theta=v["theta"], sw_K=v["K"],
                                       _loss_new={}, _sw_run=0)
                online_step = None
                for step, record in sorted(ep["steps"].items()):
                    if online_gate(gate, {**record, "stag_w": float(step in ep["armed"])}):
                        online_step = step
                        break
                expected = e1.switch_step(ep, "loss_new_cur", v["W"], v["theta"], v["K"])
                if online_step != expected:
                    raise RuntimeError(f"Online/offline gate mismatch: {env} {name}")
                compared += 1
            x = e1.evaluate(eps, budget, "loss_new_cur", v["W"], v["theta"], v["K"])
            for metric in ("F", "S"):
                x[metric + "_wilson95"] = wilson(x[metric + "_k"], x[metric + "_n"])
            x["switches_before_success_or_timeout"] = sum(
                r["switch"] is not None and r["switch"] < (r["succ"] or r["T"]) for r in x["rows"])
            if v["theta"] == 0.05:
                prev = old["grid"][f"loss_new_cur|{v['W']}|0.05|1"][old_names[env]]
                for key in ("F_k", "F_n", "S_k", "S_n", "sw_median", "risk"):
                    if x[key] != prev[key]:
                        raise RuntimeError(f"Existing E1 regression mismatch: {env} {name} {key}")
            result[name] = x
            interval = lambda k: "–".join(f"{100*t:.1f}%" for t in x[k + "_wilson95"])
            lines.append(f"| {env} | {name} | {x['F_k']}/{x['F_n']} ({100*x['F']:.1f}%) | "
                         f"{x['S_k']}/{x['S_n']} ({100*x['S']:.1f}%) | {x['sw_median']} | "
                         f"{interval('F')} | {interval('S')} |")
        # Smaller theta must never cause an earlier switch for the same W/K on a fixed trajectory.
        for early, late in (("R-switch", "R-w3-t01"), ("R-w5-t05", "R-w5-t01")):
            for before, after in zip(result[early]["rows"], result[late]["rows"]):
                assert before["key"] == after["key"]
                assert after["switch"] is None or (before["switch"] is not None and after["switch"] >= before["switch"])
        summary["datasets"][env] = {"n_episodes": len(eps), "results": result}
    summary["online_gate_checks"] = {"episode_parameter_pairs": compared, "mismatches": 0,
        "scope": "Same trigger inputs; PushObj offline latent trigger approximation is not revalidated here"}
    lines += ["", f"在线切换函数与离线计算一致：{compared} 个任务×参数组合，0处不一致（使用相同触发输入）。",
              "PushObj 旧基线部分采用离线潜轨迹触发；其与实时触发的细微差异仍须闭环配对检查。",
              "闭环结论：待新运行完成。固定参数不根据本回放结果继续扩展搜索。",
              "PushObj 中触发后成功的分母仅4；不能据此宣称稳定的误切换控制。",
              "基线回放只诊断切换时间，选择最终参数必须使用闭环净救回、误伤和20/30/50次成功率。"]
    dump(a.report / "offline.json", summary)
    (a.report / "offline.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def run_args(a, env):
    return SimpleNamespace(env=env, stream="S7", arms=a.arms, level="medium", seed=100,
        ref=None, smoke=a.smoke, episodes=None, per_shard=9, gpus=a.gpus, jobs=a.jobs,
        root=str(a.root), ckpt="checkpoints/diversemaze_layout_shift", data="data/diverse_maze",
        min_free_gb=40.0, stagger=10.0)


def prepare(a, runner):
    runner.ARMS.update({k: (dict(runner.T0SP), dict(v)) for k, v in VARIANTS.items()})
    manifest = {"schema": 1, "kind": "closed_loop_development", "budget": 50,
        "model_updates": "continue", "smoke": a.smoke, "variants": {k: VARIANTS[k] for k in a.arms},
        "repo_head": subprocess.check_output(["git", "-C", str(a.repo), "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {p: digest(a.repo / p) for p in
                          ["planning/switch.py", "scripts/e3_run.py", *OLD_REPORTS.values()]},
        "jobs": []}
    # Collect all expected jobs, including completed ones, for completeness auditing.
    old_finished, old_active = runner.finished, runner.active
    runner.finished, runner.active = lambda _: False, lambda _: False
    try:
        for env in a.env:
            q = queue.Queue()
            args = run_args(a, env)
            (runner.maze_jobs if env == "maze" else runner.pushobj_jobs)(args, q)
            while not q.empty():
                out, command = q.get_nowait()
                cmd = [str(c) for c in command]
                ids = next(c.split("=", 1)[1] for c in cmd if c.startswith("++planner.stag.only_samples="))
                manifest["jobs"].append({"env": env, "arm": out.parent.name, "out": str(out),
                    "samples": json.loads(ids), "command": cmd})
    finally:
        runner.finished, runner.active = old_finished, old_active
    if not manifest["jobs"]:
        raise RuntimeError("No episodes selected; check reference assets")
    dest = a.report / ("smoke_manifest.json" if a.smoke else "manifest.json")
    if dest.exists():
        previous = json.loads(dest.read_text())
        # Never silently reuse an output directory for a different experiment.
        for key in ("variants", "jobs", "source_sha256", "smoke"):
            if previous[key] != manifest[key]:
                raise RuntimeError(f"Manifest changed ({key}); use a fresh --root and --report")
    dump(dest, manifest)
    print(f"Prepared {len(manifest['jobs'])} jobs, "
          f"{sum(len(j['samples']) for j in manifest['jobs'])} episode-arm runs; {dest}")
    return manifest


def completeness(manifest, runner):
    errors = []
    for job in manifest["jobs"]:
        out = Path(job["out"])
        if not runner.finished(out):
            errors.append(f"unfinished: {out}")
            continue
        f = out / "stag_episodes.jsonl"
        rows = [json.loads(s) for s in f.read_text().splitlines() if s.strip()] if f.exists() else []
        ids = [r["sample_idx"] for r in rows]
        if len(ids) != len(set(ids)) or set(ids) != set(job["samples"]):
            errors.append(f"missing/duplicate episode results: {out}")
        if any(r.get("freeze") is not False for r in rows):
            errors.append(f"missing/incorrect model-update policy: {out}")
    return errors


def analyze(a, manifest, runner):
    errors = completeness(manifest, runner)
    if errors:
        dump(a.report / "closed_loop_status.json", {"complete": False, "errors": errors})
        raise RuntimeError(f"Refusing partial success-rate report: {len(errors)} incomplete jobs")
    outputs = {}
    for env in a.env:
        common = [sys.executable, "-B", str(a.repo / "scripts/e3_analyze.py"), "--env", env,
                  "--arms", *a.arms, "--root", str(a.root)]
        checked = subprocess.run(common + ["--check"], cwd=a.repo, text=True, capture_output=True)
        (a.report / f"{env}_pairing.txt").write_text(checked.stdout + checked.stderr)
        if checked.returncode:
            raise RuntimeError(f"Pre-switch pairing failed for {env}; rerun reference on this hardware")
        if not a.smoke:
            # Safe after every expected selected episode has finished and passed pairing.
            subprocess.run(common + ["--report", str(a.report / "closed_loop"), "--name", "candidates"],
                           cwd=a.repo, check=True)
        outputs[env] = {"complete": True, "pre_switch_pairing": "passed"}
    dump(a.report / ("smoke_status.json" if a.smoke else "closed_loop_status.json"),
         {"complete": True, "smoke": a.smoke, "environments": outputs,
          "variants": manifest["variants"], "source_sha256": manifest["source_sha256"]})
    if not a.smoke and set(a.env) == {"maze", "pushobj"} and set(NEW_ARMS).issubset(a.arms):
        compare(a)


def compare(a):
    """Compare shared/per-environment calibration using actual outcomes, never offline F/S."""
    data = {}
    lines = ["# 闭环开发集比较", "", "模型全程更新；所有候选保持同一 T0SP 代价修正。",
             "统一参数与分环境选择来自同一四组候选，都是开发集选择结果，尚未经过独立测试。",
             "先按50次成功率，再按误伤更少、20/30次成功率更高排序；统一参数对两个环境等权。", "",
             "| 环境 | 候选 | 20次成功 | 30次成功 | 50次成功 | 救回 | 误伤 | 配对p（未校正） |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for env in ("maze", "pushobj"):
        old = json.loads((a.repo / OLD_REPORTS[env]).read_text())
        new = json.loads((a.report / "closed_loop" / TAGS[env] / "candidates.json").read_text())
        data[env] = {"R-switch": old["R-switch"], **{k: new[k] for k in NEW_ARMS}}
        n = 100 if env == "maze" else 280
        for arm, row in {"R-trig (control)": old["R-trig"], **data[env]}.items():
            rates = [row["budgets"][str(b)][0] / n for b in (20, 30, 50)]
            lines.append(f"| {env} | {arm} | {100*rates[0]:.1f}% | {100*rates[1]:.1f}% | "
                         f"{100*rates[2]:.1f}% | {len(row['wins'])} | {len(row['losses'])} | {row['p']:.4g} |")

    def score(env, arm):
        r, n = data[env][arm], (100 if env == "maze" else 280)
        return (r["budgets"]["50"][0] / n, -len(r["losses"]) / n,
                (r["budgets"]["20"][0] + r["budgets"]["30"][0]) / (2 * n))

    shared = max(VARIANTS, key=lambda arm: tuple(sum(score(e, arm)[i] for e in data) / 2 for i in range(3)))
    separate = {env: max(VARIANTS, key=lambda arm: score(env, arm)) for env in data}
    lines += ["", f"开发集统一参数候选：{shared}。", f"开发集分别校准候选：{separate}。",
              "以上选择不得作为独立测试成绩。下一阶段使用未参与开发的新任务/种子，并报告多种子不确定性。"]
    (a.report / "closed_loop_comparison.md").write_text("\n".join(lines) + "\n")
    dump(a.report / "development_selection.json", {"independent_test_complete": False,
         "shared": shared, "per_environment": separate, "metrics": data})


def run(a, manifest, runner):
    import torch
    gpus = a.gpus.split(",")
    if not torch.cuda.is_available():
        dump(a.report / "runtime_status.json", {"state": "blocked_no_cuda", "torch": torch.__version__,
             "closed_loop_started": False, "reason": "No CUDA device is visible in this execution environment"})
        raise RuntimeError("No CUDA device; manifest saved, no planning jobs started")
    if a.jobs > 2 * len(gpus) or any(not g.isdigit() or int(g) >= torch.cuda.device_count() for g in gpus):
        raise ValueError("Require visible GPU IDs and at most two workers per GPU")
    q = queue.Queue()
    for job in manifest["jobs"]:
        out = Path(job["out"])
        if runner.finished(out):
            continue
        if out.exists() and any(out.iterdir()):
            raise RuntimeError(f"Incomplete output exists: {out}; inspect it before restarting to avoid appended logs")
        q.put((out, job["command"]))
    if q.empty():
        return analyze(a, manifest, runner)
    if not a.smoke:
        smoke = a.report / "smoke_status.json"
        status = json.loads(smoke.read_text()) if smoke.exists() else {}
        if (not status.get("complete") or status.get("variants") != manifest["variants"]
                or status.get("source_sha256") != manifest["source_sha256"]
                or set(status.get("environments", {})) != set(a.env)):
            raise RuntimeError("Run --smoke 2 successfully before the full experiment")
    args = run_args(a, a.env[0])
    results = []
    dump(a.report / "runtime_status.json", {"state": "running", "closed_loop_started": True,
                                           "jobs": q.qsize(), "gpus": gpus})
    threads = [threading.Thread(target=runner.run_worker, args=(gpus[i % len(gpus)], q, args, results))
               for i in range(min(a.jobs, q.qsize()))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if not results or not all(results):
        dump(a.report / "runtime_status.json", {"state": "failed", "results": results})
        raise RuntimeError("One or more planning jobs failed; inspect console logs")
    analyze(a, manifest, runner)
    dump(a.report / "runtime_status.json", {"state": "complete", "smoke": a.smoke,
                                           "closed_loop_started": True})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("phase", choices=("replay", "prepare", "run", "analyze"))
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--report", type=Path, default=Path(DEFAULT_REPORT))
    ap.add_argument("--root", type=Path, default=Path(DEFAULT_ROOT))
    ap.add_argument("--env", nargs="+", choices=("maze", "pushobj"), default=["maze", "pushobj"])
    ap.add_argument("--arms", nargs="+", choices=list(VARIANTS), default=list(NEW_ARMS))
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--gpus", default="0,1,4,5,6,7")
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    a.repo = a.repo.resolve()
    a.report = (a.repo / a.report).resolve()
    a.root = (a.repo / a.root).resolve()
    if a.smoke < 0 or a.jobs < 1 or len(set(a.arms)) != len(a.arms) or len(set(a.env)) != len(a.env):
        ap.error("Require non-negative smoke size, positive jobs, and unique arms/environments")
    if a.smoke:
        a.root = a.root.with_name(a.root.name + "_smoke")
    # Imports should not modify the research checkout during offline analysis.
    sys.dont_write_bytecode = True
    e1, runner = load_modules(a.repo)
    if a.phase == "replay":
        return replay(a, e1)
    manifest = prepare(a, runner)
    if a.phase == "run":
        return run(a, manifest, runner)
    if a.phase == "analyze":
        return analyze(a, manifest, runner)


if __name__ == "__main__":
    main()
