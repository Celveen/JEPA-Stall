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
from pushobj_results import followup_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--data", type=Path, help="Optional archived figure data instead of completed paired results.")
    parser.add_argument("--out", type=Path, default=root / "figures")
    args = parser.parse_args()
    if args.data:
        data = json.loads(args.data.read_text())
    else:
        results = followup_rows(root.parents[1])
        original = json.loads((root / 'provenance/headline_data_v6.json').read_text())
        released_counts = {group['label']: group['success_counts'][0] for group in original['groups']}
        matched = json.loads((root / 'provenance/lewm_matched_results.json').read_text())
        released_counts['LeWM'] = matched['total']['random_native']
        original['series'][0]['label'] = 'Base data'
        groups = []
        for label, key in [('Frozen', 'ours_frozen'), ('LeWM', 'ours_lewm'), ('AdaJEPA', 'ours_adajepa')]:
            row = results[key, 'GD']
            groups.append({'label': label, 'denominator': 350,
                           'success_counts': [released_counts.get(label), row['native_successes'],
                                              sum(row['successes_by_shape'].values())],
                           'evidence': row})
        data = {'series': original['series'], 'groups': groups,
                'released_count_source': 'Frozen/AdaJEPA: headline_data_v6.json; LeWM: lewm_matched_results.json',
                'protocol': 'PushObj at 150 controls and threshold 0.45 for augmented ECP. '
                            'LeWM uses an epoch-10 frozen checkpoint and geometry-assisted contact objective, '
                            'weight 10. AdaJEPA uses an epoch-3 checkpoint with online updates and S+P. '
                            'Frozen and AdaJEPA use seed 100; LeWM uses seed 2000. '
                            'LeWM base data is matched random/source-action replay, not a released checkpoint. '
                            'Brackets compare base-data native planning with augmented ECP. '
                            'Settings were explored on these task lists.'}
        args.data = root / 'provenance/headline_data_latest.json'
        args.data.write_text(json.dumps(data, indent=2) + '\n')
    data['groups'].sort(key=lambda g: ['Frozen','LeWM','AdaJEPA'].index(g['label']))
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
    centers = np.arange(len(data['groups'])) * 1.25
    bar_width, spacing = 0.28, 0.32
    series = data["series"]
    groups = data["groups"]
    rates = np.array([[np.nan if n is None else 100 * n / group["denominator"] for n in group["success_counts"]]
                      for group in groups])
    for idx, item in enumerate(series):
        x = centers + (idx - (len(series) - 1) / 2) * spacing
        ax.bar(x, rates[:, idx], width=bar_width, color=item["color"],
               edgecolor="white", linewidth=0.5, zorder=3)
        for bx, by in zip(x, rates[:, idx]):
            if np.isnan(by):
                ax.text(bx, 2.0, "—", ha="center", va="bottom", fontsize=8.2, color="#5F6870")
                continue
            ax.text(bx, by + 1.4, f"{by:.1f}", ha="center", va="bottom",
                    fontsize=8.2, fontweight="bold" if idx == len(series) - 1 else "normal",
                    color="#252B31", zorder=5)
    gain_values = []
    for center, group in zip(centers, groups):
        first = next(i for i, count in enumerate(group['success_counts']) if count is not None)
        gain = 100 * (group["success_counts"][-1] - group["success_counts"][first]) / group["denominator"]
        gain_values.append(gain)
        # Span only the measured stages used to compute each gain.
        x0 = center + spacing * (first - (len(series) - 1) / 2)
        x1 = center + spacing * (len(series) - 1) / 2
        ax.plot([x0, x0, x1, x1], [86, 88, 88, 86], color="#5F6870", linewidth=0.7, zorder=4)
        ax.text(center, 90.1, f"+{gain:.1f} pp", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color="#30373D")
    ax.set_xlim(-0.57, centers[-1] + 0.57)
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
    if len(handles) > 2:
        fig.legend(handles=handles[2:], loc="upper center", bbox_to_anchor=(0.54, 0.925),
                   ncol=1, frameon=False, fontsize=8.2, handlelength=1.0,
                   handleheight=0.8, handletextpad=0.35, borderaxespad=0.0)
    outputs = []
    for ext in ("pdf", "svg", "png"):
        directory = args.out / {"pdf": "", "svg": "svg", "png": "previews"}[ext]
        directory.mkdir(parents=True, exist_ok=True)
        output = directory / f"fig0_headline.{ext}"
        fig.savefig(output, dpi=240)
        if ext == "svg":
            output.write_text("\n".join(line.rstrip() for line in output.read_text().splitlines()) + "\n")
        outputs.append({"file": str(output.relative_to(args.out)), "sha256": hashlib.sha256(output.read_bytes()).hexdigest()})
    plt.close(fig)
    manifest = {
        "canvas_inches": [3.25, 2.25],
        "minimum_font_size_points": 8.2,
        "data_file": args.data.name,
        "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "success_rates_percent": [[None if np.isnan(value) else value for value in row] for row in rates],
        "gain_from_exact_counts_percentage_points": gain_values,
        "y_axis": [0, 100],
        "outputs": outputs,
    }
    (root / "provenance/headline_figure_manifest_latest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
