#!/usr/bin/env python3
"""Add the matched Maze example from the released MuJoCo recordings to Figure 11."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import pymupdf


PAPER = Path(__file__).resolve().parents[1]
FIGURES = PAPER / 'figures'
ROLLOUTS = PAPER.parents[1] / 'reports/maze_medium50_three_seeds_observed/rollouts'
CASE = 'adajepa_seed102_sample13'
ARMS = ('adajepa', 'adajepa_s')
NATIVE, MEMORY = '#9e4d32', '#008275'
INK = '#163047'
STEPS = (14, 18)


def read_pair():
    provenance = json.loads((ROLLOUTS / 'provenance.json').read_text())[CASE]
    runs = []
    for arm in ARMS:
        record = provenance['rgb_source'][arm]
        path = ROLLOUTS / record['published_recording']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256']
        with np.load(path, allow_pickle=False) as data:
            runs.append({key: data[key] for key in ('states', 'state_g', 'rgb', 'target_rgb')})
    np.testing.assert_array_equal(runs[0]['state_g'], runs[1]['state_g'])
    np.testing.assert_array_equal(runs[0]['target_rgb'], runs[1]['target_rgb'])
    np.testing.assert_array_equal(runs[0]['rgb'][0], runs[1]['rgb'][0])
    distances = [np.linalg.norm(run['states'][:, :2] - run['state_g'][:2], axis=1) for run in runs]
    for arm, trace in zip(ARMS, distances):
        result = provenance['result'][arm]
        assert len(trace) - 1 == result['controls']
        np.testing.assert_allclose(trace[-1], result['distance'], atol=1e-6)
    return runs, distances, provenance['controls_per_step']


def gallery_panel(path, runs, controls_per_step):
    fig = plt.figure(figsize=(247.5 / 72, 84 / 72))
    fig.text(.024, .981, '(l) Maze · AdaJEPA / ECP (recorded frames)',
             fontsize=8.6, weight='bold', color=INK, va='top', family='DejaVu Serif')
    red, green, blue = np.moveaxis(runs[0]['target_rgb'].astype(int), -1, 0)
    goal_y, goal_x = np.where((green > red + 12) & (green > blue + 12))
    assert 0 < len(goal_x) and np.ptp(goal_x) < 12 and np.ptp(goal_y) < 12
    goal = (goal_x.mean(), goal_y.mean())
    for step_index, step in enumerate(STEPS):
        for arm, color in enumerate((NATIVE, MEMORY)):
            column = 2 * step_index + arm
            left = .037 + .242 * column
            fig.text(left + .104, .82, f'{("Native", "+S")[arm]} · {step}',
                     ha='center', va='top', fontsize=7.8, color=color, family='DejaVu Serif')
            ax = fig.add_axes([left, .057, .208, .613])
            ax.imshow(runs[arm]['rgb'][step * controls_per_step], interpolation='nearest')
            ax.add_patch(Circle(goal, radius=8, linewidth=.65, fill=False, edgecolor='#fff2be'))
            ax.set(xlim=(32, 212), ylim=(191, 11))
            ax.set_axis_off()
    fig.savefig(path, dpi=330, transparent=True)
    plt.close(fig)


def distance_plot(distances, controls_per_step):
    fig, ax = plt.subplots(figsize=(2.68, 1.24))
    fig.subplots_adjust(left=.17, right=.968, bottom=.26, top=.84)
    for label, trace, color in zip(('Native', '+S'), distances, (NATIVE, MEMORY)):
        steps = np.arange(len(trace)) / controls_per_step
        ax.plot(steps, trace, label=label, color=color, lw=1.15)
        ax.scatter(steps[-1], trace[-1], s=9, color=color, zorder=3)
    for step in STEPS:
        ax.scatter(step, distances[1][step * controls_per_step], s=9, color=MEMORY, zorder=3)
    ax.axhline(.5, color='#75808a', linestyle='--', lw=.65)
    ax.set(xlim=(0, 30), ylim=(0, 7), xticks=(0, 5, 10, 15, 20, 25, 30),
           yticks=(0, 2, 4, 6))
    ax.set_xlabel('Execution step', labelpad=1.5)
    ax.set_ylabel('Goal distance', labelpad=2)
    ax.tick_params(length=2, pad=1.5)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(loc='lower center', bbox_to_anchor=(.5, 1.02), ncols=2,
              frameon=False, borderaxespad=0, handlelength=1.4, columnspacing=1.4)
    fig.savefig(FIGURES / 'figS_maze_detour_distance.pdf')
    plt.close(fig)


def main():
    plt.rcParams.update({'font.size': 8, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    runs, distances, controls_per_step = read_pair()
    with TemporaryDirectory() as folder:
        panel_path = Path(folder) / 'panel.pdf'
        gallery_panel(panel_path, runs, controls_per_step)
        with pymupdf.open(FIGURES / 'figS_diverse_execution_v12.pdf') as gallery, pymupdf.open(panel_path) as panel:
            page = gallery[0]
            box = pymupdf.Rect(page.rect.width / 2, page.rect.height - 84, page.rect.width, page.rect.height)
            page.show_pdf_page(box, panel, 0)
            gallery.save(FIGURES / 'figS_diverse_execution_maze.pdf', garbage=4, deflate=True)
    distance_plot(distances, controls_per_step)


if __name__ == '__main__':
    main()
