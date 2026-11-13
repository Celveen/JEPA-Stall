"""Compare paired physical rollouts after isolated versus consecutive guidance."""

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.with_name("lewm_ecp_045_contact")
spec = importlib.util.spec_from_file_location("previous_diagnostics", PREVIOUS / "diagnose.py")
previous_diagnostics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous_diagnostics)
metrics = previous_diagnostics.metrics


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def main():
    with (ROOT / "paired_tasks.csv").open() as file:
        rows = list(csv.DictReader(file))
    report, flips = {}, {}
    for condition, source in (("contact", PREVIOUS), ("burst4", ROOT)):
        series, full_episode_contacts = [], []
        bias_amplitudes, approach_displacements = [], []
        weighted_count = 0
        for shape in dict.fromkeys(row["shape"] for row in rows):
            with np.load(source / "replayed" / shape / f"{condition}.npz") as data:
                for row in rows:
                    if row["shape"] != shape or not row["first_proprio_replan"]:
                        continue
                    i = int(row["sample_idx"])
                    trigger = 5 * int(row["first_proprio_replan"])
                    last = int(min(5 * data["action_len"][i], data["states"].shape[1] - 1))
                    values = metrics(data["states"][i], data["contacts"][i], data["goals"][i], trigger, last)
                    if row["native"] == "0":
                        series.append(values)
                        full_episode_contacts.append(int(data["contacts"][i, trigger:last].sum()))
                    if row["contact"] != row["burst4"]:
                        flips.setdefault(f"{shape}:{i}", {})[condition] = values
                for event in records(source / "full" / shape / "lewm_action_bias.jsonl"):
                    step = event["replan"] * 5
                    for action in event["actions"]:
                        i = action["sample_idx"]
                        offset = np.asarray(action["offset_px_per_control"])
                        bias_amplitudes.append(float(np.linalg.norm(offset)))
                        if action["phase"] == "approach_contact":
                            displacement = data["states"][i, step + 5, :2] - data["states"][i, step, :2]
                            approach_displacements.append(float(displacement @ offset / np.linalg.norm(offset)))
                for event in records(source / "full" / shape / "lewm_proprio_weight.jsonl"):
                    ids = event["sample_idx"]
                    error = data["goals"][ids] - data["states"][ids, 5 * event["replan"]]
                    position = np.linalg.norm(error[:, 2:4], axis=1)
                    angle = np.abs((error[:, 4] + np.pi) % (2 * np.pi) - np.pi)
                    assert np.all(position < 20) and np.all(angle < np.pi / 9)
                    weighted_count += len(ids)
        report[condition] = {"n": len(series)}
        for key in series[0]:
            observed = [value[key] for value in series if value[key] is not None]
            report[condition][key] = (sum(observed) if isinstance(observed[0], bool)
                                      else round(float(np.median(observed)), 3))
        report[condition]["proprio_replans"] = weighted_count
        report[condition]["post_trigger_contact_controls_on_native_failures"] = sum(full_episode_contacts)
        report[condition]["bias_amplitude_quantiles_px"] = np.quantile(bias_amplitudes, [0, .1, .25, .5, 1]).tolist()
        report[condition]["bias_events"] = len(bias_amplitudes)
        report[condition]["approach_events_without_positive_projected_motion"] = sum(x <= 0 for x in approach_displacements)
        report[condition]["approach_projected_motion_median_px"] = float(np.median(approach_displacements))
    report["outcome_flips"] = flips
    (ROOT / "diagnostics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("contact", "burst4")}, indent=2))


if __name__ == "__main__":
    main()
