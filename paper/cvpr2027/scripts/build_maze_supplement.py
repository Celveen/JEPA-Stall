#!/usr/bin/env python3
"""Build the appendix Maze comparisons from the released MuJoCo recordings."""
import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
from matplotlib.patches import Circle
import numpy as np
from PIL import Image
import pymupdf


PAPER = Path(__file__).resolve().parents[1]
FIGURES = PAPER / 'figures'
ROLLOUTS = PAPER.parents[1] / 'reports/maze_medium50_three_seeds_observed/rollouts'
GALLERY_CASE = 'adajepa_seed101_sample25'
ROLLOUT_CASE = 'adajepa_seed102_sample13'
ARMS = ('adajepa', 'adajepa_s')
NATIVE, MEMORY = '#9e4d32', '#008275'
INK = '#163047'
CONTROLS = (65, 70, 72, 75, 81, 90)
LIGHT_PATHS = ('#F1AC8D', '#81DBCE')
SUBSTEPS = 20


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
    with np.load(PAPER / 'provenance/maze_recorded_figure_paths.npz') as archive:
        for run, name in zip(runs, ('native', 'plus_s')):
            path = archive[f'{case}_{name}']
            np.testing.assert_allclose(path[::SUBSTEPS], run['states'][:, :2], atol=5e-7, rtol=0)
            run['path'] = path
    # MuJoCo renders the site pose from the final forward pass before integration.
    frame_world, frame_pixels = [], []
    for run in runs:
        indices = np.maximum(np.arange(len(run['states'])) * SUBSTEPS - 1, 0)
        frame_world.extend(run['path'][indices])
        frame_pixels.extend(agent_center(frame) for frame in run['rgb'])
    frame_world.append(runs[0]['state_g'][:2])
    goal = agent_center(runs[0]['target_rgb'])
    frame_pixels.append(goal)
    coordinates = np.column_stack((frame_world, np.ones(len(frame_world))))
    transform = np.linalg.lstsq(coordinates, frame_pixels, rcond=None)[0]
    assert np.max(np.linalg.norm(coordinates @ transform - frame_pixels, axis=1)) < 2
    for run in runs:
        run['pixels'] = run['path'] @ transform[:2] + transform[2]
    return runs, distances, goal


def agent_center(rgb):
    red, green, blue = np.moveaxis(rgb.astype(int), -1, 0)
    y, x = np.where((green > red + 12) & (green > blue + 12))
    assert 0 < len(x) and np.ptp(x) < 12 and np.ptp(y) < 12
    return np.array([x.mean(), y.mean()])


def trace(ax, points, arm, width):
    color = LIGHT_PATHS[arm]
    line, = ax.plot(*points.T, lw=width, color=color, solid_capstyle='round', zorder=3)
    line.set_path_effects([pe.Stroke(linewidth=width + .58, foreground=INK), pe.Normal()])
    return color


def show_frame(ax, rgb, goal=None):
    ax.imshow(rgb, interpolation='nearest')
    if goal is not None:
        ax.add_patch(Circle(goal, radius=8, linewidth=.7, fill=False, edgecolor='#fff2be'))
    ax.set(xlim=(32, 212), ylim=(191, 11))
    ax.set_axis_off()


def gallery_panel(path, runs, goal):
    width, height = 247.5, 92
    fig = plt.figure(figsize=(width / 72, height / 72))
    fig.text(5.94/width, 1-11.17/height, '(l) Maze · AdaJEPA / ECP',
             fontsize=8.6, weight='bold', color='#243D51', va='baseline', family='STIXGeneral')
    frames = (runs[0]['rgb'][0], runs[0]['target_rgb'], runs[0]['rgb'][-1], runs[1]['rgb'][-1])
    for col, (rgb, label, color) in enumerate(zip(frames, ('Start', 'Goal', 'Failure', 'ECP success'),
                                                  ('#243D51', '#243D51', '#B95B38', '#087D83'))):
        left = 8.595 + 59.895 * col
        fig.text((left+25.61)/width, 1-22.32/height, label, ha='center', va='baseline',
                 fontsize=8, color=color, family='STIXGeneral')
        ax = fig.add_axes([left/width, 8.69/height, 51.22/width, 51.22/height])
        show_frame(ax, rgb)
        if col >= 2:
            arm = col - 2
            points = runs[arm]['pixels']
            path_color = trace(ax, points, arm, .70)
            ax.scatter(*points[5 * SUBSTEPS::5 * SUBSTEPS].T, s=3.8, c=path_color,
                       edgecolor=INK, linewidth=.22, zorder=4)
            # Place direction arrows on actual, moving integration segments.
            distances = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
            for fraction in (.25, .60):
                end = np.searchsorted(distances, fraction * distances[-1])
                start = np.searchsorted(distances, distances[end] - 3)
                ax.annotate('', xy=points[end], xytext=points[start],
                            arrowprops=dict(arrowstyle='-|>', color=path_color, lw=.55,
                                            mutation_scale=4, shrinkA=0, shrinkB=0), zorder=5)
            ax.scatter(*points[0], s=11, c='white', edgecolor=INK, linewidth=.65, zorder=6)
            ax.scatter(*points[-1], s=18, marker=('X', 'D')[arm], c=path_color,
                       edgecolor='white', linewidth=.55, zorder=7)
        ax.scatter(*goal, s=26, marker='*', c='#E7B33C', edgecolor=INK, linewidth=.5, zorder=8)
    fig.savefig(path, dpi=330, transparent=True)
    plt.close(fig)


def rollout_plot(runs, distances, goal):
    width, height = 6.875, 8.05
    fig = plt.figure(figsize=(width, height))

    def board(x, top, size):
        return fig.add_axes([x/width, 1 - (top+size)/height, size/width, size/height])

    for title, image, top in (('Start', runs[0]['rgb'][0], 1.62),
                              ('Goal', runs[0]['target_rgb'], 3.71)):
        ax = board(.045, top, 1.16)
        show_frame(ax, image, goal if title == 'Goal' else None)
        ax.set_title(title, fontsize=11, color=INK, pad=5)
    fig.add_artist(Line2D([1.31/width]*2, [1-6.49/height, 1-.17/height],
                          transform=fig.transFigure, color='#D5DBDD', lw=.85))

    for index, control in enumerate(CONTROLS):
        band, col = divmod(index, 3)
        for arm, run in enumerate(runs):
            top = (.31, 3.63)[band] + 1.48*arm
            size = 1.39
            x = 1.91 + col*1.70
            ax = board(x, top, size)
            frame = run['rgb'][control]
            show_frame(ax, frame)
            # Use the same four-control interval and scale for both planners.
            pixels = run['pixels'][(control-4)*SUBSTEPS:control*SUBSTEPS]
            trace(ax, pixels, arm, 1.25)
            ax.add_patch(Circle(goal, 8, fill=False, ec='#fff2be', lw=1.1, zorder=5))
            ax.add_patch(Circle(agent_center(frame), 5.5, fill=False,
                                ec='white', lw=.8, zorder=6))
            if arm == 0:
                ax.set_title(f'Control {control}', fontsize=10.5, color=INK, pad=4)
            if col == 0:
                fig.text(1.84/width, 1-(top+size/2)/height, ('Native', 'Ours')[arm],
                         ha='right', va='center', fontsize=10, color=(NATIVE, MEMORY)[arm])

    ax = fig.add_axes([.11, .059, .855, .093])
    for label, values, color in zip(('Native', 'Ours'), distances, (NATIVE, MEMORY)):
        ax.plot(np.arange(len(values)), values, label=label, color=color, lw=1.45)
        ax.scatter(len(values)-1, values[-1], s=18, color=color, zorder=3)
    ax.scatter(CONTROLS, distances[1][list(CONTROLS)], s=18, color=MEMORY, zorder=3)
    ax.axhline(.5, color='#75808a', ls='--', lw=.75, label='Success threshold')
    ax.set(xlim=(0, 150), ylim=(0, 7), xticks=np.arange(0, 151, 25), yticks=(0, 2, 4, 6))
    ax.set_xlabel('Executed controls', fontsize=10)
    ax.set_ylabel('Goal distance', fontsize=10)
    ax.tick_params(length=3, labelsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(loc='lower center', bbox_to_anchor=(.5, 1.01), ncols=3, frameon=False,
              borderaxespad=0, fontsize=9)
    fig.savefig(FIGURES / 'figS_maze_detour_rollout.pdf', dpi=330)
    plt.close(fig)


def separate_gallery_inputs(gallery):
    page = gallery[0]
    scenes = page.get_image_info(xrefs=True)
    assert len(scenes) == 44  # Eleven original cases with four views each.
    # PushObj's renderer includes a neutral-gray viewport border in some crops.
    # Remove just its edge pixels inside the raster, beneath all vector paths.
    for scene in scenes[:7 * 4]:
        pix = pymupdf.Pixmap(gallery, scene['xref'])
        rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3).copy()
        neutral = np.ptp(rgb, axis=2) == 0
        border = neutral & (rgb[:, :, 0] < 255) & (rgb[:, :, 0] >= 100)
        y, x = np.ogrid[:pix.h, :pix.w]
        columns = (border.mean(axis=0) > .6) & ((x < 10) | (x >= pix.w - 10))
        rows = (border.mean(axis=1)[:, None] > .6) & ((y < 10) | (y >= pix.h - 10))
        mask = border & (columns | rows)
        if mask.any():
            rgb[mask] = 255
            buffer = io.BytesIO()
            Image.fromarray(rgb).save(buffer, format='PNG')
            page.replace_image(scene['xref'], stream=buffer.getvalue())

    for row in range(6):
        goal, native = [pymupdf.Rect(scenes[row * 8 + j]['bbox']) for j in (1, 2)]
        for offset in (0, page.rect.width / 2):
            x = (goal.x1 + native.x0) / 2 + offset
            page.draw_line((x, goal.y0 - 12), (x, goal.y1), color=(.80, .83, .84), width=.65)


def main():
    plt.rcParams.update({'font.size': 8, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    gallery_runs, _, gallery_goal = read_pair(GALLERY_CASE)
    with TemporaryDirectory() as folder:
        panel_path = Path(folder) / 'panel.pdf'
        gallery_panel(panel_path, gallery_runs, gallery_goal)
        with pymupdf.open(FIGURES / 'figS_diverse_execution_v12.pdf') as gallery, pymupdf.open(panel_path) as panel:
            page = gallery[0]
            box = pymupdf.Rect(page.rect.width / 2, page.rect.height - panel[0].rect.height,
                               page.rect.width, page.rect.height)
            separate_gallery_inputs(gallery)
            page.show_pdf_page(box, panel, 0)
            gallery.save(FIGURES / 'figS_diverse_execution_maze.pdf', garbage=4, deflate=True)
    rollout_plot(*read_pair(ROLLOUT_CASE))


if __name__ == '__main__':
    main()
