"""Compare simulator geometry and contacts on the matched stalled tasks."""

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.with_name("lewm_ecp_045_proprio")
CONDITIONS = ("proprio", "contact")


def metrics(trace, contacts, goal, trigger, last):
    pusher = np.linalg.norm(trace[:, :2] - goal[:2], axis=1)
    obj = np.linalg.norm(trace[:, 2:4] - goal[2:4], axis=1)
    angle = np.abs((trace[:, 4] - goal[4] + np.pi) % (2 * np.pi) - np.pi)
    checkpoints = np.arange(trigger, last + 1, 5)
    outside = np.any((trace[:, :2] < 0) | (trace[:, :2] > 512), axis=1)
    outside_before = np.flatnonzero(outside[:trigger])
    window = min(trigger + 15, last)
    return {
        "trigger_pusher_error": float(pusher[trigger]),
        "trigger_object_error": float(obj[trigger]),
        "trigger_pusher_object_distance": float(np.linalg.norm(trace[trigger, :2] - trace[trigger, 2:4])),
        "trigger_outside": bool(outside[trigger]),
        "was_outside_before_trigger": bool(len(outside_before)),
        "first_outside_before_trigger_step": int(outside_before[0]) if len(outside_before) else None,
        "trigger_angle_deg": float(np.degrees(angle[trigger])),
        "pusher_error_reduction_after_15": float(pusher[trigger] - pusher[window]),
        "object_error_reduction_after_15": float(obj[trigger] - obj[window]),
        "object_displacement_after_15": float(np.linalg.norm(trace[window, 2:4] - trace[trigger, 2:4])),
        "object_moved_less_than_one_px_after_15": bool(np.linalg.norm(trace[window, 2:4] - trace[trigger, 2:4]) < 1),
        "object_position_error_dropped_five_px_after_15": bool(obj[trigger] - obj[window] > 5),
        "contact_in_next_15": bool(contacts[trigger:window].any()),
        "contact_before_trigger": bool(contacts[:trigger].any()),
        "contact_after_trigger": bool(contacts[trigger:last].any()),
        "object_translation_and_angle_pass": bool(((obj[checkpoints] < 20) & (angle[checkpoints] < np.pi / 9)).any()),
        "object_translation_pass": bool((obj[checkpoints] < 20).any()),
        "pusher_position_pass": bool((pusher[checkpoints] < 20).any()),
        "final_pusher_error": float(pusher[last]),
        "final_object_error": float(obj[last]),
        "final_angle_deg": float(np.degrees(angle[last])),
    }


def main():
    with (ROOT / "paired_tasks.csv").open() as file:
        rows = list(csv.DictReader(file))
    report = {}
    flips = {}
    failures = [row for row in rows if row["native"] == "0"]
    report["failure_coverage"] = {
        "native_failures": len(failures),
        "native_failures_with_trigger": sum(bool(row["first_proprio_replan"]) for row in failures),
    }
    for condition in CONDITIONS:
        series = []
        for shape in dict.fromkeys(row["shape"] for row in rows):
            source = ROOT if condition == "contact" else PREVIOUS
            with np.load(source / "replayed" / shape / f"{condition}.npz") as data:
                for row in rows:
                    if row["shape"] != shape or not row["first_proprio_replan"]:
                        continue
                    i = int(row["sample_idx"])
                    trigger = 5 * int(row["first_proprio_replan"])
                    last = min(5 * data["action_len"][i], data["states"].shape[1] - 1)
                    values = metrics(data["states"][i], data["contacts"][i], data["goals"][i], trigger, int(last))
                    if row["native"] == "0":
                        series.append(values)
                    if row["proprio"] != row["contact"]:
                        key = f"{shape}:{i}"
                        flips.setdefault(key, {})[condition] = values
        report[condition] = {}
        for key in series[0]:
            observed = [value[key] for value in series if value[key] is not None]
            if isinstance(observed[0], bool):
                report[condition][key] = sum(observed)
            else:
                report[condition][key] = round(float(np.median(observed)), 3)
        report[condition]["n"] = len(series)
    report["outcome_flips"] = flips
    for shape in dict.fromkeys(row["shape"] for row in rows):
        output = ROOT / "full" / shape / "lewm_proprio_weight.jsonl"
        if not output.exists():
            continue
        with np.load(ROOT / "replayed" / shape / "contact.npz") as data:
            for line in output.read_text().splitlines():
                record = json.loads(line)
                ids = record["sample_idx"]
                error = data["goals"][ids] - data["states"][ids, 5 * record["replan"]]
                position = np.linalg.norm(error[:, 2:4], axis=1)
                angle = np.abs((error[:, 4] + np.pi) % (2 * np.pi) - np.pi)
                assert np.all(position < 20) and np.all(angle < np.pi / 9)
    (ROOT / "diagnostics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({name: report[name] for name in CONDITIONS}, indent=2))


if __name__ == "__main__":
    main()
