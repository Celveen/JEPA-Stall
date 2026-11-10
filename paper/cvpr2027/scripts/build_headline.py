#!/usr/bin/env python3
"""Rebuild the single-column, vector PushObj headline chart from exact counts."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/jepa-headline-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).with_name("headline_data.json"))
    parser.add_argument("--out", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    data = json.loads(args.data.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8.5,
        "axes.labelsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 8.5,
        "axes.linewidth": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
    })
    fig = plt.figure(figsize=(3.25, 2.25), facecolor="white")
    ax = fig.add_axes([0.155, 0.155, 0.825, 0.675])
    centers = np.array([0.0, 1.25])
    bar_width, spacing = 0.245, 0.285
    series = data["series"]
    groups = data["groups"]
    rates = np.array([[100 * n / group["denominator"] for n in group["success_counts"]]
                      for group in groups])
    for idx, item in enumerate(series):
        x = centers + (idx - 1) * spacing
        ax.bar(x, rates[:, idx], width=bar_width, color=item["color"],
               edgecolor="white", linewidth=0.5, zorder=3)
        for bx, by in zip(x, rates[:, idx]):
            ax.text(bx, by + 1.4, f"{by:.1f}", ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold" if idx == 2 else "normal",
                    color="#252B31", zorder=5)
    gain_values = []
    for center, group in zip(centers, groups):
        gain = 100 * (group["success_counts"][-1] - group["success_counts"][0]) / group["denominator"]
        gain_values.append(gain)
        # Identical height and span make the total-gain annotations comparable.
        x0, x1 = center - spacing, center + spacing
        ax.plot([x0, x0, x1, x1], [86, 88, 88, 86], color="#5F6870", linewidth=0.7, zorder=4)
        ax.text(center, 90.1, f"+{gain:.1f} pp", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color="#30373D")
    ax.set_xlim(-0.57, 1.82)
    ax.set_ylim(0, 100)
    ax.set_xticks(centers, [group["label"] for group in groups])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Success (%)", labelpad=3.0)
    ax.tick_params(axis="x", length=0, pad=6)
    ax.tick_params(axis="y", length=2.5, width=0.6, pad=3)
    ax.yaxis.grid(True, color="#E7EBEE", linewidth=0.5, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#6A737B")
    handles = [Patch(facecolor=item["color"], edgecolor="none", label=item["label"]) for item in series]
    fig.legend(handles=handles[:2], loc="upper center", bbox_to_anchor=(0.54, 0.995),
               ncol=2, frameon=False, fontsize=8.2, handlelength=1.0,
               handleheight=0.8, handletextpad=0.35, columnspacing=1.1, borderaxespad=0.0)
    fig.legend(handles=handles[2:], loc="upper center", bbox_to_anchor=(0.54, 0.925),
               ncol=1, frameon=False, fontsize=8.2, handlelength=1.0,
               handleheight=0.8, handletextpad=0.35, borderaxespad=0.0)
    outputs = []
    for ext in ("pdf", "svg", "png"):
        output = args.out / f"fig0_headline.{ext}"
        fig.savefig(output, dpi=240)
        outputs.append({"file": output.name, "sha256": hashlib.sha256(output.read_bytes()).hexdigest()})
    plt.close(fig)
    manifest = {
        "canvas_inches": [3.25, 2.25],
        "minimum_font_size_points": 8.2,
        "data_file": args.data.name,
        "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "success_rates_percent": rates.tolist(),
        "gain_from_exact_counts_percentage_points": gain_values,
        "y_axis": [0, 100],
        "outputs": outputs,
    }
    (args.out / "headline_figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
