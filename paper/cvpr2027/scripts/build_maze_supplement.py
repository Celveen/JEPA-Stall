#!/usr/bin/env python3
"""Build the appendix Maze comparisons from the released MuJoCo recordings."""
import argparse
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
ROLLOUTS_TO_PLOT = (
    ('adajepa_seed102_sample13', '(a) Maze · AdaJEPA', (65, 70, 72, 75, 90)),
    ('frozen_seed101_sample31', '(b) Maze · Frozen ResNet', (35, 37, 40, 42, 43)),
)
NATIVE, MEMORY = '#9e4d32', '#008275'
INK = '#163047'
LIGHT_PATHS = ('#F1AC8D', '#81DBCE')
SUBSTEPS = 20


def read_pair(case):
    provenance = json.loads((ROLLOUTS / 'provenance.json').read_text())[case]
    planner = case.split('_', 1)[0]
    arms = (planner, planner + '_s')
    runs = []
    for arm in arms:
        record = provenance['rgb_source'][arm]
        path = ROLLOUTS / record['published_recording']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256']
        with np.load(path, allow_pickle=False) as data:
            runs.append({key: data[key] for key in ('states', 'state_g', 'rgb', 'target_rgb')})
    np.testing.assert_array_equal(runs[0]['state_g'], runs[1]['state_g'])
    np.testing.assert_array_equal(runs[0]['target_rgb'], runs[1]['target_rgb'])
    np.testing.assert_array_equal(runs[0]['rgb'][0], runs[1]['rgb'][0])
    distances = [np.linalg.norm(run['states'][:, :2] - run['state_g'][:2], axis=1) for run in runs]
    for arm, trace in zip(arms, distances):
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


def rollout_plot():
    width, height = 6.875, 8.0
    fig = plt.figure(figsize=(width, height))

    def board(x, top, size):
        return fig.add_axes([x/width, 1 - (top+size)/height, size/width, size/height])

    for case_index, (case, title, controls) in enumerate(ROLLOUTS_TO_PLOT):
        runs, distances, goal = read_pair(case)
        red, green, blue = np.moveaxis(runs[0]['target_rgb'].astype(int), -1, 0)
        wall_y, wall_x = np.where((red > green + 18) & (green > blue + 18))
        view = dict(xlim=(wall_x.min()-5, wall_x.max()+5), ylim=(wall_y.max()+5, wall_y.min()-5))
        base = .06 + 4.00 * case_index
        fig.text(.015, 1-base/height, title, va='top', fontsize=11,
                 fontweight='bold', color=INK, family='STIXGeneral')
        for arm, run in enumerate(runs):
            top = base + .51 + 1.12*arm
            size = .95
            reference = board(.06, top, size)
            source = runs[0]['rgb'][0] if arm == 0 else runs[0]['target_rgb']
            show_frame(reference, source, goal if arm == 1 else None)
            reference.set(**view)
            reference.set_title(('Start', 'Goal')[arm], fontsize=8.5, color=INK, pad=2)
            fig.text(1.57/width, 1-(top+size/2)/height, ('Native', 'Ours')[arm],
                     ha='right', va='center', fontsize=9, color=(NATIVE, MEMORY)[arm])
            for col, control in enumerate(controls):
                ax = board(1.66 + 1.055*col, top, size)
                frame = run['rgb'][control]
                show_frame(ax, frame)
                ax.set(**view)
                trail = run['pixels'][(control-4)*SUBSTEPS:control*SUBSTEPS]
                trace(ax, trail, arm, .95)
                ax.add_patch(Circle(goal, 8, fill=False, ec='#fff2be', lw=.85, zorder=5))
                ax.add_patch(Circle(agent_center(frame), 5.5, fill=False,
                                    ec='white', lw=.7, zorder=6))
                if arm == 0:
                    ax.set_title(f'Control {control}', fontsize=8.5, color=INK, pad=2)

        fig.add_artist(Line2D([1.06/width]*2, [1-(base+2.58)/height, 1-(base+.35)/height],
                              transform=fig.transFigure, color='#D5DBDD', lw=.8))
        ax = fig.add_axes([.14, 1-(base+3.58)/height, .65, .70/height])
        for label, values, color in zip(('Native', 'Ours'), distances, (NATIVE, MEMORY)):
            ax.plot(np.arange(len(values)), values, label=label, color=color, lw=1.15)
            ax.scatter(len(values)-1, values[-1], s=10, color=color, zorder=3)
        ax.scatter(controls, distances[1][list(controls)], s=10, color=MEMORY, zorder=3)
        ax.axhline(.5, color='#75808a', ls='--', lw=.75, label='Success threshold')
        ax.set(xlim=(0,150), ylim=(0,7), xticks=np.arange(0,151,25), yticks=(0,3,6))
        ax.set_xlabel('Executed controls', fontsize=8, labelpad=1)
        ax.set_ylabel('Goal\ndistance', fontsize=8, labelpad=2)
        ax.tick_params(length=2, labelsize=7.5, pad=1.5)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(loc='center left', bbox_to_anchor=(1.02, .50), borderaxespad=0,
                  frameon=False, handlelength=1.3, fontsize=7.2)
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


def retain_ten_gallery_panels(gallery):
    """Remove the last gallery row (k, l), retaining a--j at their native size."""
    page = gallery[0]
    labels = {
        block[4].split(')', 1)[0] + ')': block
        for block in page.get_text('blocks')
        if block[4].startswith(('(a)', '(k)'))
    }
    cut_y = labels['(k)'][1] - labels['(a)'][1]
    page.add_redact_annot(
        pymupdf.Rect(0, cut_y, page.rect.width, page.rect.height),
        fill=None, cross_out=False,
    )
    page.apply_redactions(images=2, graphics=2, text=0)
    page.set_cropbox(pymupdf.Rect(0, 0, page.rect.width, cut_y))
    assert '(k)' not in page.get_text() and '(l)' not in page.get_text()
    assert len(page.get_image_info()) == 40


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only-rollout', action='store_true',
                        help='Rebuild the detour figure while keeping the gallery unchanged.')
    args = parser.parse_args()
    plt.rcParams.update({'font.size': 8, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    if args.only_rollout:
        rollout_plot()
        return
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
            retain_ten_gallery_panels(gallery)
            gallery.save(FIGURES / 'figS_diverse_execution_maze.pdf', garbage=4, deflate=True)
    rollout_plot()


if __name__ == '__main__':
    main()
