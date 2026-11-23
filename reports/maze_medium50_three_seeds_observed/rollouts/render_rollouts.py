"""Rebuild the step-based figures from the recorded MuJoCo source arrays."""

import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator
import numpy as np

out = Path(__file__).resolve().parent
base = out.parent
out.mkdir(exist_ok=True)
summary = json.loads((base / "summary.json").read_text())
examples = [
    (100, "adajepa", "AdaJEPA", 28, 65),
    (101, "adajepa", "AdaJEPA", 24, 80),
    (102, "adajepa", "AdaJEPA", 12, 70),
    (101, "frozen", "Frozen", 30, 40),
]
CONTROLS_PER_STEP = 5


def step_at(control):
    return control / CONTROLS_PER_STEP

neutral, green = "#526279", "#087868"
ink, muted = "#142338", "#4C5B70"
plt.rcParams.update({"font.family":"DejaVu Sans", "pdf.fonttype":42, "ps.fonttype":42, "savefig.facecolor":"white"})
for seed, arm, label, index, middle in examples:
    filename = f"{arm}_seed{seed}_sample{index+1:02d}"
    folder = f"maze_medium50_s{seed}_g030_observed"
    source = base.parent / folder
    raw = out / "raw" / filename
    files = {arm: raw / "native.npz", arm + "_s": raw / "plus_s.npz"}
    episodes, distances, metrics = {}, {}, {}
    for condition, path in files.items():
        with np.load(path, allow_pickle=False) as packet:
            episodes[condition] = dict(packet)
        run = episodes[condition]
        offset = run["states"][:, :2] - run["state_g"][:2]
        distances[condition] = np.array([np.linalg.norm(point) for point in offset])
        case = next(x for x in summary["episodes"][condition]
                    if x["seed"] == seed and x["sample_idx"] == index)
        assert len(run["rgb"]) == len(run["states"]) == case["controls"] + 1
        np.testing.assert_allclose(float(distances[condition][-1]), case["distance"], rtol=0, atol=1e-7)
        metrics[condition] = case

    log = source / "per_control.jsonl.gz"
    selected_logs = {condition: [] for condition in files}
    with gzip.open(log, "rt") as stream:
        for line in stream:
            row = json.loads(line)
            if row["sample_idx"] == index and row["arm"] in selected_logs:
                selected_logs[row["arm"]].append(row)
    for condition, rows in selected_logs.items():
        assert [r["control"] for r in rows] == list(range(1, metrics[condition]["controls"] + 1))
        np.testing.assert_allclose([r["distance"] for r in rows], distances[condition][1:], rtol=0, atol=1e-7)
        np.testing.assert_array_equal([r["physical_action"] for r in rows], episodes[condition]["physical_actions"])
    native, with_s = episodes[arm], episodes[arm + "_s"]
    np.testing.assert_array_equal(native["target_rgb"], with_s["target_rgb"])
    np.testing.assert_array_equal(native["state_g"], with_s["state_g"])
    assert not metrics[arm]["success"] and metrics[arm + "_s"]["success"]
    assert np.flatnonzero(distances[arm] < .5).size == 0
    assert np.flatnonzero(distances[arm + "_s"] < .5).tolist() == [metrics[arm + "_s"]["controls"]]
    split = metrics[arm + "_s"]["paired_first_physical_change"]
    end = metrics[arm + "_s"]["controls"]
    assert split and split < middle < end
    np.testing.assert_array_equal(native["rgb"][:split], with_s["rgb"][:split])
    with np.load(source / "goals.npz", allow_pickle=False) as goals:
        layout, bfs = int(goals["map_indices"][index]), int(goals["block_dists"][index])

    times = [0, split - 1, middle, end, 150]
    titles = ["Start", "Before actions differ", "During execution", "+S reaches goal", "Step limit"]
    fig = plt.figure(figsize=(14, 7.25), facecolor="white")
    fig.text(.055, .944, f"{label}  /  Matched Maze rollout", fontsize=23, fontweight="bold", color=ink)
    fig.text(.055, .9, f"Seed {seed}  ·  Sample {index+1}  ·  Layout {layout}  ·  Shortest path {bfs} cells  ·  Recorded MuJoCo frames",
             fontsize=11, color=muted)
    grid = fig.add_gridspec(2, 6, left=.122, right=.965, bottom=.338, top=.840, wspace=.12, hspace=.36)
    for row_id, (condition, color) in enumerate(((arm, neutral), (arm + "_s", green))):
        for col, (t, title) in enumerate(zip(times, titles)):
            ax = fig.add_subplot(grid[row_id, col])
            ax.set_axis_off()
            if row_id == 0:
                ax.set_title(title, fontsize=10, color=ink, fontweight="semibold", pad=8)
            if t <= metrics[condition]["controls"]:
                ax.imshow(episodes[condition]["rgb"][t], interpolation="nearest")
                if col in (3, 4) and t == metrics[condition]["controls"]:
                    ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, linewidth=2.7,
                                           edgecolor=color, facecolor="none", clip_on=False))
                ax.text(.5, -.09, f"Step {step_at(t):g}  ·  d={distances[condition][t]:.2f}", transform=ax.transAxes,
                        ha="center", va="top", color=color, fontsize=9)
            else:
                ax.set_box_aspect(1)
                ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor="#EAF5F1", edgecolor="none"))
                ax.text(.5, .55, "Goal reached", transform=ax.transAxes, ha="center", color=green,
                        fontweight="bold", fontsize=11)
                ax.text(.5, .41, f"Stopped at step {step_at(end):g}", transform=ax.transAxes, ha="center", color=green, fontsize=10)
            if col == 0:
                bbox = ax.get_position()
                fig.text(.055, (bbox.y0+bbox.y1)/2, "Native" if row_id == 0 else "+S", fontsize=13,
                         fontweight="bold", color=color, va="center")

    goal_ax = fig.add_subplot(grid[:, 5])
    goal_ax.imshow(native["target_rgb"], interpolation="nearest")
    goal_ax.set_axis_off()
    goal_ax.set_title("Goal reference", fontsize=10, color=ink, fontweight="semibold", pad=8)
    goal_ax.text(.5, -.09, "Shared target observation", transform=goal_ax.transAxes, ha="center",
                 va="top", color=muted, fontsize=9)

    chart = fig.add_axes([.122, .128, .650, .158])
    chart.set_axisbelow(True)
    chart.grid(axis="y", color="#E4E9F0", linewidth=.75)
    for side in ("top", "right"):
        chart.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        chart.spines[side].set_color("#C2CCD8")
    for condition, c, w in ((arm, neutral, 1.8), (arm + "_s", green, 2.4)):
        d = distances[condition]
        chart.plot(step_at(np.arange(len(d))), d, c=c, lw=w)
        chart.scatter([step_at(len(d) - 1)], [d[-1]], c=c, s=24, zorder=4)
    chart.axhline(.5, c="#737F90", ls="--", lw=.9)
    chart.axvline(step_at(split - 1), c="#B7C0CD", ls=":", lw=1)
    chart.set(xlim=(0, 30), ylim=(0, max(max(v) for v in distances.values())*1.06),
              xlabel="Execution steps", ylabel="Distance to goal")
    chart.xaxis.set_major_locator(MultipleLocator(5))
    chart.tick_params(axis="both", labelsize=9, colors=muted)
    for row_id, (tag, c) in enumerate((("Native", neutral), ("+S", green))):
        condition = arm if row_id == 0 else arm + "_s"
        metric = metrics[condition]
        y = .251 - .051*row_id
        fig.text(.805, y, tag, fontsize=11, fontweight="bold", color=c)
        fig.text(.875, y, ("Success" if metric["success"] else "Failure") + f"  ·  d={metric['distance']:.2f}",
                 fontsize=10, color=c)
    fig.text(.805, .151, "Success threshold  d < 0.5", fontsize=9, color=muted)
    fig.text(.055, .043, "One full step executes five controls. Fractional steps show the exact point of early termination.",
             fontsize=9, color=muted)
    fig.savefig(out / f"{filename}.png", dpi=180)
    fig.savefig(out / f"{filename}.pdf")
    plt.close(fig)

    print(filename, times, metrics[arm]["distance"], metrics[arm + "_s"]["distance"])
