"""Check paired inputs, unchanged pre-trigger actions, and burst scheduling."""

from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.with_name("lewm_ecp_045_contact")
SHAPES = ("I", "L", "square", "Z", "small_tee", "plus", "T")


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def audit(shape, rows):
    output, prior = ROOT / "full" / shape, PREVIOUS / "full" / shape
    config = OmegaConf.to_container(OmegaConf.load(output / "config.yaml"), resolve=True)
    reference = OmegaConf.to_container(OmegaConf.load(prior / "config.yaml"), resolve=True)
    assert config["planner"].pop("burst") == {"replans": 4}
    config["saved_folder"] = reference["saved_folder"]
    config["planner"]["_target_"] = reference["planner"]["_target_"]
    assert config == reference, "Unexpected configuration change"
    bias, contact = config["planner"]["bias"], config["planner"]["contact"]
    summary = json.loads((output / "summary.json").read_text())
    old_summary = json.loads((prior / "summary.json").read_text())
    assert summary["model_unchanged"]
    assert summary["model_state_sha256"] == old_summary["model_state_sha256"]
    triggers = {r["replan"]: r for r in records(output / "lewm_ecp_replans.jsonl")}
    episodes = records(output / "lewm_ecp_episodes.jsonl")
    applied, phase_tasks, phases = {}, {}, Counter()
    continuation_without_trigger = 0
    for event in records(output / "lewm_action_bias.jsonl"):
        replan = event["replan"]
        for action in event["actions"]:
            i = action["sample_idx"]
            start, end = action["burst_start_replan"], action["burst_end_replan"]
            assert start <= replan <= end == start + 3
            assert triggers[start]["weight"][i] > 0
            fired = triggers[replan]["weight"][i] > 0
            assert action["detector_trigger_for_current_plan"] == fired
            continuation_without_trigger += not fired
            history = applied.setdefault(i, [])
            if history:
                last_replan, last_start = history[-1]
                assert replan > last_replan
                assert start == last_start or replan - last_replan >= bias["min_gap"]
            history.append((replan, start))
            assert len(history) <= bias["max_interventions"]
            ready = (action["object_position_error_px"] < contact["position_tol_px"]
                     and action["object_angle_error_deg"] < contact["angle_tol_deg"])
            phase = action["phase"]
            assert phase in ("pusher_goal", "approach_contact", "push")
            assert ready == (phase == "pusher_goal")
            direction = np.asarray(action["direction_px"])
            expected = direction * min(1 / 5, bias["cap_px"] / np.linalg.norm(direction))
            np.testing.assert_allclose(action["measured_offset_px_per_control"],
                                       np.broadcast_to(expected, (5, 2)), atol=1e-4)
            phases[phase] += 1
            phase_tasks.setdefault(phase, set()).add(i)
    with np.load(output / "final_eval_results.npz") as run, np.load(prior / "final_eval_results.npz") as old:
        for key in ("state_0", "state_g"):
            np.testing.assert_array_equal(run[key], old[key])
        assert run["actions"].shape == (50, 30, 10)
        assert np.isfinite(run["actions"]).all()
        assert len(episodes) == len(rows) == 50
        assert int(run["successes"].sum()) == summary["successes"]
        for i, row in enumerate(rows):
            trigger = episodes[i]["first_trigger"]
            assert trigger == (int(row["first_proprio_replan"]) if row["first_proprio_replan"] else None)
            # The first triggered plan uses the same guidance in both arms.
            unchanged = trigger + 1 if trigger is not None else 30
            np.testing.assert_array_equal(run["actions"][i, :unchanged], old["actions"][i, :unchanged])
            history = applied.get(i, [])
            assert all(r < run["action_len"][i] for r, _ in history)
            row["burst4"] = int(run["successes"][i])
            row["burst4_bias_count"] = len(history)
            row["burst4_burst_count"] = len({s for _, s in history})
            row["burst4_bias_replans"] = " ".join(str(r) for r, _ in history)
    return phases, Counter({k: len(v) for k, v in phase_tasks.items()}), continuation_without_trigger


def totals(rows):
    out = {name: sum(int(row[name]) for row in rows)
           for name in ("native", "regularizer", "combined", "proprio", "contact", "burst4")}
    out["tasks"] = len(rows)
    for baseline in ("native", "contact", "proprio"):
        out[f"rescues_vs_{baseline}"] = sum(not int(r[baseline]) and bool(r["burst4"]) for r in rows)
        out[f"regressions_vs_{baseline}"] = sum(bool(int(r[baseline])) and not r["burst4"] for r in rows)
    return out


def main():
    with (PREVIOUS / "paired_tasks.csv").open() as file:
        rows = list(csv.DictReader(file))
    phases, phase_tasks, by_shape = Counter(), Counter(), {}
    continued_without_trigger = 0
    for shape in SHAPES:
        subset = [r for r in rows if r["shape"] == shape]
        events, tasks, continued = audit(shape, subset)
        phases.update(events)
        phase_tasks.update(tasks)
        continued_without_trigger += continued
        by_shape[shape] = totals(subset)
    report = {"total": totals(rows), "by_shape": by_shape, "bias_events_by_phase": phases,
              "biased_tasks_by_phase": phase_tasks,
              "continuation_events_without_new_trigger": continued_without_trigger,
              "biased_tasks": sum(bool(r["burst4_bias_count"]) for r in rows),
              "tasks_using_full_seven_bias_budget": sum(r["burst4_bias_count"] == 7 for r in rows),
              "audit": "passed: matching inputs, frozen weights, config, pre-trigger actions, burst timing and physical offsets"}
    with (ROOT / "paired_tasks.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (ROOT / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
