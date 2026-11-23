"""Build compact state comparisons and recorded simulator sequences.

Optional --import-repo imports and verifies archived observations once.
Subsequent builds use only the bundled images and their provenance manifest.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tarfile

os.environ.setdefault('MPLCONFIGDIR', '/tmp/jepa-appendix-figures')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--paper', type=Path, default=Path(__file__).resolve().parents[1])
parser.add_argument('--import-repo', type=Path)
parser.add_argument('--only-supplementary', action='store_true',
                    help='Rebuild the supplementary sequences without touching main-paper figures.')
args = parser.parse_args()
p = args.paper
dest = p / 'figures/recorded_rollouts'
dest.mkdir(exist_ok=True)
manifest_path = p / 'provenance/appendix_rollout_sources.json'
sha = lambda data: hashlib.sha256(data).hexdigest()

if args.import_repo:
    repo = args.import_repo
    manifest = {'source_kind': 'Recorded simulator observations, not model predictions',
                'framing': 'Full original viewport; only PushT static gray perimeter is hidden',
                'cases': [], 'source_archives': {}, 'files': {}}

    def copy_checked(source, relative, expected, pixels=None):
        data = source.read_bytes()
        assert sha(data) == expected, source
        if pixels is not None:
            np.testing.assert_array_equal(np.asarray(Image.open(io.BytesIO(data))), pixels)
        path = dest / relative
        shutil.copyfile(source, path)
        rel = path.relative_to(p).as_posix()
        manifest['files'][rel] = {'sha256': expected, 'source': source.relative_to(repo).as_posix(),
                                  'verified_against_recorded_rgb': pixels is not None}
        return rel

    src = repo / 'reports/lewm_shift_visuals_20261116'
    checks = json.loads((src / 'sha256.json').read_text())
    summary = json.loads((src / 'summary.json').read_text())
    arc = src / 'all_recorded_rollouts.tar.gz'
    assert sha(arc.read_bytes()) == checks[arc.name]
    manifest['source_archives'][arc.relative_to(repo).as_posix()] = checks[arc.name]
    with tarfile.open(arc) as tar:
        for cond, mid in [('red_agent', [50, 75, 100]), ('red_goal', [40, 60, 62])]:
            idx = summary['selected_targets'][cond]
            row = next(x for x in summary['all_episode_rows'] if x['condition'] == cond and x['target_index'] == idx)
            assert not row['native_success'] and row['ours_success']
            case = {'dataset': 'PushT', 'backbone': 'LeWM', 'condition': cond,
                    'target': idx, 'unit': 'Control', 'rows': [], 'hide_border': 6,
                    'shared_start': True}
            shared = None
            for arm in ['native', 'ours']:
                member = f'{cond}_{arm}/recorded_execution.npz'
                with np.load(io.BytesIO(tar.extractfile(member).read()), allow_pickle=False) as data:
                    stop = int(row[arm + '_stop_controls'])
                    recorded_stop = data['action_len'][idx]
                    assert (5 * int(recorded_stop) if np.isfinite(recorded_stop) else 150) == stop
                    assert bool(data['successes'][idx]) == (arm == 'ours')
                    goal = data['goal_rgb'][idx, 0]
                    first = data['rgb'][idx, 0]
                    if shared is not None:
                        np.testing.assert_array_equal(shared[0], first)
                        np.testing.assert_array_equal(shared[1], goal)
                    shared = (first.copy(), goal.copy())
                    frames = []
                    for step in [0, *mid, stop]:
                        source = src / f'figures/{cond}/{arm}_frames/{step:03d}.png'
                        relative = f'pusht_{cond}_{arm}_{step:03d}.png'
                        if source.exists():
                            rel = copy_checked(source, relative, checks[source.relative_to(src).as_posix()], data['rgb'][idx, step])
                        else:
                            # Per-control records include moments between the
                            # original five-control screenshot exports.
                            pixels = data['rgb'][idx, step]
                            assert pixels.dtype == np.uint8
                            path = dest / relative
                            if path.exists():
                                np.testing.assert_array_equal(np.asarray(Image.open(path)), pixels)
                            else:
                                Image.fromarray(pixels).save(path)
                            np.testing.assert_array_equal(np.asarray(Image.open(path)), pixels)
                            rel = path.relative_to(p).as_posix()
                            manifest['files'][rel] = {
                                'sha256': sha(path.read_bytes()),
                                'archive': arc.relative_to(repo).as_posix(),
                                'member': member, 'target_index': idx, 'step': step,
                                'verified_against_recorded_rgb': True}
                        frames.append({'path': rel, 'step': step})
                    source = src / f'figures/{cond}/target.png'
                    case['goal'] = copy_checked(source, f'pusht_{cond}_goal.png', checks[source.relative_to(src).as_posix()], goal)
                    case['rows'].append({'arm': arm, 'frames': frames, 'success': arm == 'ours'})
            manifest['cases'].append(case)

    src = repo / 'reports/rope_formal20/visuals_extra'
    prov = json.loads((src / 'provenance.json').read_text())
    receipt = json.loads((src / 'evidence_manifest.json').read_text())
    arc = src / 'evidence.tar.gz'
    assert sha(arc.read_bytes()) == receipt['archive_sha256']
    manifest['source_archives'][arc.relative_to(repo).as_posix()] = receipt['archive_sha256']
    case = {'dataset': 'Rope', 'backbone': 'DINO-WM', 'target': 15,
            'unit': 'Push', 'rows': [], 'hide_border': 0,
            'configuration': 'Native five-push plans; fixed-strength ECP three-push plans, five warm-up pushes',
            'excluded_from_aggregate': True, 'ecp_activations': 3,
            'shared_start': False}
    shared = None
    with tarfile.open(arc) as tar:
        for arm in ['native5', 'fixed3']:
            entry = next(x for x in prov['sources'] if x['arm'] == arm and x['sample'] == 5)
            member = f'qualitative_extra/{arm}/sample_05/recorded_execution.npz'
            cache = tar.extractfile(member).read()
            assert sha(cache) == entry['cache_sha256']
            with np.load(io.BytesIO(cache), allow_pickle=False) as data:
                rgb = data['obs_visual'][0]
                assert np.array_equal(rgb, rgb.astype(np.uint8))
                if shared is not None:
                    case['shared_start'] = bool(np.array_equal(shared, rgb[0]))
                shared = rgb[0].copy()
                stop = len(rgb) - 1
                frames = []
                for step in [0, 6, 12, stop]:
                    f = next(x for x in entry['frames'] if x['push'] == step)
                    rel = copy_checked(src / f['path'], f'rope_{arm}_{step:02d}.png', f['sha256'], rgb[step])
                    frames.append({'path': rel, 'step': step, 'cd': f['cd']})
                source = src / 'raw_frames/sample_05/goal.png'
                case['goal'] = copy_checked(source, 'rope_goal.png', entry['goal_png_sha256'])
                case['rows'].append({'arm': 'native' if arm == 'native5' else 'ours',
                                     'frames': frames, 'success': frames[-1]['cd'] < .3})
    manifest['cases'].append(case)
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')

manifest = json.loads(manifest_path.read_text())
for filename, info in manifest['files'].items():
    assert sha((p / filename).read_bytes()) == info['sha256'], filename

INK, BROWN, TEAL = '#243D51', '#B95B38', '#087D83'
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral'], 'font.size': 9,
                     'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
                     'savefig.facecolor': 'white'})
outputs = {}

def text(fig, x, y, value, **kwargs):
    return fig.text(x / fig.get_figwidth(), y / fig.get_figheight(), value, **kwargs)

def scene(fig, path, x, y, side, border=0):
    arr = np.asarray(Image.open(p / path))
    h, w = arr.shape[:2]
    ax = fig.add_axes([x / fig.get_figwidth(), y / fig.get_figheight(),
                       side / fig.get_figwidth(), side / fig.get_figheight()])
    ax.imshow(arr, interpolation='nearest', extent=(-.5, w - .5, h - .5, -.5))
    if border:
        for xy, ww, hh in [((-.5, -.5), w, border), ((-.5, h-border-.5), w, border),
                           ((-.5, -.5), border, h), ((w-border-.5, -.5), border, h)]:
            ax.add_patch(Rectangle(xy, ww, hh, fc='white', ec='none', antialiased=False))
    ax.set_xlim(-.5, w-.5); ax.set_ylim(h-.5, -.5); ax.axis('off')

def save(fig, name):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = [(t.get_text(), t.get_window_extent(renderer)) for t in fig.texts]
    collisions = [(a, b) for i, (a, x) in enumerate(bounds)
                  for b, y in bounds[i+1:] if x.overlaps(y)]
    assert not collisions, collisions
    for folder, extension in [('', 'pdf'), ('previews', 'png'), ('svg', 'svg')]:
        path = p / 'figures' / folder / f'{name}.{extension}'
        path.parent.mkdir(exist_ok=True)
        metadata = {'CreationDate': None} if extension == 'pdf' else {'Date': None} if extension == 'svg' else None
        fig.savefig(path, dpi=240, metadata=metadata)
        if extension == 'svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
        outputs[path.relative_to(p).as_posix()] = {'sha256': sha(path.read_bytes()),
                                                  'size_inches': list(fig.get_size_inches())}
    plt.close(fig)

# Shared start, pre-correction state and goal appear only once.
main_comparisons = [] if args.only_supplementary else [
    ('PushObj', 'fig1_pushobj_motivation', 'push', 15, 30, 17),
    ('Maze', 'fig3_maze_case', 'maze', 6, 30, 27)]
for task, name, prefix, before, native_end, ours_end in main_comparisons:
    fig = plt.figure(figsize=(6.875, 1.48))
    filenames = ['start', 'before', 'baseline_end', 'corrected_end', 'goal']
    headings = ['Start', 'Before correction', 'Without ECP', 'With ECP', 'Goal']
    notes = ['Replan 0', f'Replan {before}', f'Replan {native_end} · failure', f'Replan {ours_end} · success', 'Target observation']
    for col, (filename, heading, note) in enumerate(zip(filenames, headings, notes)):
        x = .14 + 1.34 * col
        color = BROWN if col == 2 else TEAL if col == 3 else INK
        scene(fig, f'figures/frames/{prefix}_{filename}.png', x, .18, 1.12)
        text(fig, x+.56, 1.36, heading, fontsize=9, ha='center', color=color)
        text(fig, x+.56, .055, note, fontsize=8.2, ha='center', color=color)
    save(fig, name)

def sequence(cases, name):
    block = 3.13
    fig = plt.figure(figsize=(6.875, block * len(cases)))
    side = 1.10
    for idx, case in enumerate(cases):
        top = fig.get_figheight() - idx * block
        suffix = case.get('condition', f"target {case['target']}").replace('_', ' ')
        title = f"({chr(97+idx)}) {case['dataset']} · {case['backbone']} · {suffix.title()}"
        text(fig, .14, top-.14, title, fontsize=10, weight='bold', color=INK)
        fig.add_artist(Line2D(
            [1.36 / fig.get_figwidth()] * 2,
            [(top - 3.00) / fig.get_figheight(), (top - .24) / fig.get_figheight()],
            transform=fig.transFigure, color='#D5DBDD', linewidth=.8))
        assert case['shared_start']
        np.testing.assert_array_equal(
            np.asarray(Image.open(p / case['rows'][0]['frames'][0]['path'])),
            np.asarray(Image.open(p / case['rows'][1]['frames'][0]['path'])))
        # Stack common references to preserve the original 1.10-inch frame size
        # while adding a fourth execution column.
        for row_index, (path, label) in enumerate([
                (case['rows'][0]['frames'][0]['path'], 'Shared start'),
                (case['goal'], 'Shared goal')]):
            y = top - 1.43 - 1.42 * row_index
            scene(fig, path, .14, y, side, case['hide_border'])
            text(fig, .69, y-.125, label, fontsize=8.15, ha='center', color=INK)
        for row_index, row in enumerate(case['rows']):
            assert len(row['frames'][1:]) == 4
            color = BROWN if row['arm'] == 'native' else TEAL
            y = top - 1.43 - 1.42 * row_index
            label = 'Without ECP' if row['arm'] == 'native' else 'With ECP'
            text(fig, 1.48, y+side+.035, label, fontsize=8.8, color=color, weight='bold')
            for col, frame in enumerate(row['frames'][1:], 1):
                x = .14 + 1.34 * col
                scene(fig, frame['path'], x, y, side, case['hide_border'])
                label = f"{case['unit']} {frame['step']}"
                if col == len(row['frames']) - 1:
                    label += f" · CD {frame['cd']:.3f}" if 'cd' in frame else (' · success' if row['success'] else ' · failure')
                text(fig, x+side/2, y-.125, label, fontsize=8.15, color=color, ha='center')
    save(fig, name)

sequence([c for c in manifest['cases'] if c['dataset']=='PushT'], 'figS_pusht_rollouts')
# Equal-push observations from the audited seed-101 executions.
rope = json.loads((p/'provenance/rope_fixed_visuals_seed101.json').read_text())
for name, record in rope['files'].items():
    assert sha((p/name).read_bytes()) == record['sha256']
for case in rope['cases']:
    case['shared_start'] = True
    for row in case['rows']:
        row['frames'][-1]['cd'] = row['final_cd']
sequence(rope['cases'], 'figS_rope_rollouts')
layout_path = p/'provenance/appendix_figure_layout.json'
if args.only_supplementary and layout_path.exists():
    prior = json.loads(layout_path.read_text())
    outputs = {**prior.get('outputs', {}), **outputs}
layout_path.write_text(json.dumps({
    'shared_states_displayed_once': True, 'local_crops': False, 'gray_dividers': True,
    'reference_separator': {'color': '#D5DBDD', 'width_points': .8},
    'shared_reference_column': 'Start above goal, both shared by the two execution rows',
    'execution_columns': 4, 'frame_side_inches': 1.10,
    'rope_display': 'Matched native/fixed-ECP executions on targets 15 and 21, seed 101; all four runs execute 20 pushes',
    'outputs': outputs}, indent=2)+'\n')
print('Built supplementary execution figures, including equal-push Rope observations.')
