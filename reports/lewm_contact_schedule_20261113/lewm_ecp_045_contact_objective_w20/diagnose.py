"""Measure physical contact and verify the optimized pusher trajectory was executed."""

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.with_name("lewm_ecp_045_contact_objective")
METRICS_SOURCE = ROOT.with_name("lewm_ecp_045_contact") / "diagnose.py"
spec = importlib.util.spec_from_file_location("contact_metrics", METRICS_SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def main():
    rows = list(csv.DictReader((ROOT / "paired_tasks.csv").open()))
    report, flips = {}, {}
    for condition, source in (("objective", PREVIOUS), ("objective20", ROOT)):
        series = []
        events, approach, forward, touching = 0, 0, 0, 0
        distances = []
        loss_before, loss_after = [], []
        trajectory_error = 0.
        proprio_count = 0
        for shape in dict.fromkeys(r["shape"] for r in rows):
            with np.load(source / "replayed" / shape / f"{condition}.npz") as data:
                for row in rows:
                    if row["shape"] != shape or not row["first_proprio_replan"]:
                        continue
                    i = int(row["sample_idx"])
                    trigger = 5 * int(row["first_proprio_replan"])
                    last = int(min(5 * data["action_len"][i], data["states"].shape[1] - 1))
                    values = module.metrics(data["states"][i], data["contacts"][i], data["goals"][i], trigger, last)
                    if row["native"] == "0":
                        series.append(values)
                    if row["objective"] != row["objective20"]:
                        flips.setdefault(f"{shape}:{i}", {})[condition] = values
                name = "lewm_contact_objective.jsonl"
                for entry in records(source / "full" / shape / name):
                    step = 5 * entry["replan"]
                    for event in entry["actions"]:
                        events += 1
                        i = event["sample_idx"]
                        if condition in ("objective", "objective20"):
                            actual = data["states"][i, step + 1:step + 6, :2]
                            predicted = np.asarray(event["optimized_predicted_positions"])
                            err = np.abs(predicted - actual).max()
                            trajectory_error = max(trajectory_error, float(err))
                            np.testing.assert_allclose(predicted, actual, atol=.005, rtol=0)
                            loss_before.append(event["weighted_contact_cost_before"])
                            loss_after.append(event["weighted_contact_cost_after"])
                        if event["phase"] == "approach_contact":
                            approach += 1
                            direction = np.asarray(event["direction_px"])
                            displacement = data["states"][i, step + 5, :2] - data["states"][i, step, :2]
                            progress = float(displacement @ direction / np.linalg.norm(direction))
                            distances.append(progress)
                            forward += int(progress > 0)
                            touching += int(data["contacts"][i, step:step + 5].any())
                for event in records(source / "full" / shape / "lewm_proprio_weight.jsonl"):
                    ids = event["sample_idx"]
                    error = data["goals"][ids] - data["states"][ids, 5 * event["replan"]]
                    assert np.all(np.linalg.norm(error[:, 2:4], axis=1) < 20)
                    assert np.all(np.abs((error[:, 4] + np.pi) % (2 * np.pi) - np.pi) < np.pi / 9)
                    proprio_count += len(ids)
        out = {"n": len(series)}
        for key in series[0]:
            values = [v[key] for v in series if v[key] is not None]
            out[key] = sum(values) if isinstance(values[0], bool) else round(float(np.median(values)), 3)
        out.update(guidance_events=events, approach_events=approach,
                   approach_events_with_forward_movement=forward,
                   approach_events_with_actual_contact=touching,
                   approach_forward_displacement_median_px=float(np.median(distances)),
                   proprio_replans=proprio_count)
        if loss_before:
            before, after = np.asarray(loss_before), np.asarray(loss_after)
            out.update(contact_cost_decreased=int((after < before).sum()),
                       weighted_contact_cost_before_median=float(np.median(before)),
                       weighted_contact_cost_after_median=float(np.median(after)),
                       median_contact_cost_reduction_fraction=float(np.median(1 - after / np.maximum(before, 1e-12))),
                       max_predicted_vs_executed_position_error_px=trajectory_error)
        report[condition] = out
    report["outcome_flips"] = flips
    (ROOT / "diagnostics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("objective", "objective20")}, indent=2))


if __name__ == "__main__":
    main()
