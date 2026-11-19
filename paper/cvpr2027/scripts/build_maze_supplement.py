#!/usr/bin/env python3
"""Build the appendix Maze comparisons from the released MuJoCo recordings."""
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
GALLERY_CASE = 'adajepa_seed101_sample25'
ROLLOUT_CASE = 'adajepa_seed102_sample13'
ARMS = ('adajepa', 'adajepa_s')
NATIVE, MEMORY = '#9e4d32', '#008275'
INK = '#163047'
STEPS = (13, 14, 15, 18)


def read_pair(case):
    provenance = json.loads((ROLLOUTS / 'provenance.json').read_text())[case]
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


def show_frame(ax, rgb, goal=None):
    ax.imshow(rgb, interpolation='nearest')
    if goal is not None:
        ax.add_patch(Circle(goal, radius=8, linewidth=.7, fill=False, edgecolor='#fff2be'))
    ax.set(xlim=(32, 212), ylim=(191, 11))
    ax.set_axis_off()


def gallery_panel(path, runs):
    fig = plt.figure(figsize=(247.5 / 72, 84 / 72))
    fig.text(.024, .981, '(l) Maze · AdaJEPA / ECP',
             fontsize=8.6, weight='bold', color=INK, va='top', family='DejaVu Serif')
    frames = (runs[0]['rgb'][0], runs[0]['target_rgb'], runs[0]['rgb'][-1], runs[1]['rgb'][-1])
    for col, (rgb, label, color) in enumerate(zip(frames, ('Start', 'Goal', 'Failure', 'ECP success'),
                                                  (INK, INK, NATIVE, MEMORY))):
        left = .037 + .242 * col
        fig.text(left + .104, .83, label, ha='center', va='top', fontsize=8,
                 color=color, family='DejaVu Serif')
        show_frame(fig.add_axes([left, .098, .208, .613]), rgb)
    fig.savefig(path, dpi=330, transparent=True)
    plt.close(fig)


def rollout_plot(runs, distances, controls_per_step):
    fig = plt.figure(figsize=(6.6, 2.95))
    red, green, blue = np.moveaxis(runs[0]['target_rgb'].astype(int), -1, 0)
    goal_y, goal_x = np.where((green > red + 12) & (green > blue + 12))
    assert 0 < len(goal_x) and np.ptp(goal_x) < 12 and np.ptp(goal_y) < 12
    goal = (goal_x.mean(), goal_y.mean())
    grid = fig.add_gridspec(2, 5, left=.013, right=.99, bottom=.335, top=.897,
                           wspace=.11, hspace=.39)
    for row, (name, color) in enumerate((('Start', INK), ('Goal', INK))):
        ax = fig.add_subplot(grid[row, 0])
        rgb = runs[0]['rgb'][0] if row == 0 else runs[0]['target_rgb']
        show_frame(ax, rgb)
        ax.set_title(name, fontsize=8.5, color=color, pad=3)
    for col, step in enumerate(STEPS, 1):
        for arm, color in enumerate((NATIVE, MEMORY)):
            ax = fig.add_subplot(grid[arm, col])
            index = step * controls_per_step
            show_frame(ax, runs[arm]['rgb'][index], goal)
            ax.set_title(f'Step {step}' if arm == 0 else f'{distances[arm][index]:.2f} units',
                         fontsize=8.5, color=INK if arm == 0 else color, pad=3)

    ax = fig.add_axes([.115, .13, .66, .172])
    for label, trace, color in zip(('Native', '+S'), distances, (NATIVE, MEMORY)):
        steps = np.arange(len(trace)) / controls_per_step
        ax.plot(steps, trace, label=label, color=color, lw=1.15)
        ax.scatter(steps[-1], trace[-1], s=9, color=color, zorder=3)
    ax.scatter(STEPS, [distances[1][s * controls_per_step] for s in STEPS],
               s=9, color=MEMORY, zorder=3)
    ax.axhline(.5, color='#75808a', linestyle='--', lw=.65, label='Success threshold')
    ax.set(xlim=(0, 30), ylim=(0, 7), xticks=(0, 5, 10, 15, 20, 25, 30),
           yticks=(0, 3, 6))
    ax.set_xlabel('Execution step', labelpad=1.5)
    ax.set_ylabel('Goal\ndistance', labelpad=3)
    ax.tick_params(length=2, pad=1.5)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(loc='center left', bbox_to_anchor=(1.035, .54), frameon=False,
              borderaxespad=0, handlelength=1.4)
    fig.text(.228, .76, 'Native', fontsize=8.5, color=NATIVE, ha='right', va='center')
    fig.text(.228, .45, '+S', fontsize=8.5, color=MEMORY, ha='right', va='center')
    fig.savefig(FIGURES / 'figS_maze_detour_rollout.pdf', dpi=330)
    plt.close(fig)


def main():
    plt.rcParams.update({'font.size': 8, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    gallery_runs, _, _ = read_pair(GALLERY_CASE)
    with TemporaryDirectory() as folder:
        panel_path = Path(folder) / 'panel.pdf'
        gallery_panel(panel_path, gallery_runs)
        with pymupdf.open(FIGURES / 'figS_diverse_execution_v12.pdf') as gallery, pymupdf.open(panel_path) as panel:
            page = gallery[0]
            box = pymupdf.Rect(page.rect.width / 2, page.rect.height - 84, page.rect.width, page.rect.height)
            page.show_pdf_page(box, panel, 0)
            gallery.save(FIGURES / 'figS_diverse_execution_maze.pdf', garbage=4, deflate=True)
    rollout_plot(*read_pair(ROLLOUT_CASE))


if __name__ == '__main__':
    main()
