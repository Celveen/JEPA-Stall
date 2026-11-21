"""Export real recorded observations and validated saved-action simulator replays."""
import csv
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import pickle
import tarfile

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/community-mpl')
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
MANIFEST = {'cases': [], 'sources': {}, 'selection': 'Purposefully selected qualitative paired examples with visible outcome differences.'}
FONT = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 20)
SMALL = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 16)
COLORS = ['#b34b47', '#178776']


def source(path):
    path = REPO / path
    rel = str(path.relative_to(REPO))
    if rel not in MANIFEST['sources']:
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        MANIFEST['sources'][rel] = digest.hexdigest()
    return path


def archive_npz(archive, member):
    with tarfile.open(source(archive)) as tar:
        return np.load(io.BytesIO(tar.extractfile(member).read()), allow_pickle=False)


def rgb(array):
    pixels = np.asarray(array).astype(np.uint8)
    np.testing.assert_array_equal(array, pixels)
    return pixels


def recorded(path, index, name, rope=False, row=None):
    z = np.load(source(path))
    count = len(z['physical_actions'][0]) if rope else z['rgb'].shape[1] - 1
    if not rope and np.isfinite(z['action_len'][index]):
        count = min(count, 5 * int(z['action_len'][index]))
    frames = rgb(z['obs_visual'][0] if rope else z['rgb'][index, :count + 1])
    states = z['states'][0] if rope else z['states'][index, :count + 1]
    return {'name': name, 'rgb': frames, 'states': states, 'source': str(path),
            'success': row['success'] == 'True' if row else bool(z['successes'][index]),
            'metrics': row or {}, 'capture': 'original recorded simulator observations'}


def pose_render(shape, state, goal_color=None):
    env = PUSHT.PushTEnv(shape=shape, with_velocity=True, render_size=224, goal_color=goal_color)
    env._setup()
    env.block.angle = float(state[4])
    env.block.position = tuple(state[2:4])
    env.agent.position = tuple(state[:2])
    env.agent.velocity = tuple(state[5:7])
    image = env.render('rgb_array')
    env.close()
    return image


def replay(archive, folder, index, shape, name, goal_color=None):
    z = archive_npz(archive, folder + '/final_eval_results.npz')
    with tarfile.open(REPO / archive) as tar:
        rows = [json.loads(line) for line in tar.extractfile(folder + '/per_iter_logs.json').read().splitlines()]
    rows = sorted([r for r in rows if r['sample_idx'] == index and 0 < r['step'] <= 30], key=lambda r: r['step'])
    actions = z['actions'][index, :len(rows)].reshape(-1, 2).astype(np.float32)
    physical = actions * np.array([.2019, .2002], np.float32) + np.array([-.0087, .0068], np.float32)
    env = PUSHT.PushTEnv(shape=shape, with_velocity=True, reset_to_state=z['state_0'][index], goal_color=goal_color)
    env.seed(100 * index + 1)
    obs, state = env.reset()
    frames, states = [obs['visual']], [state]
    for action in physical:
        obs, _, _, info = env.step(action)
        frames.append(obs['visual'])
        states.append(info['state'])
    error = float(np.max(np.abs(np.asarray(states)[5::5] - np.array([r['state'] for r in rows]))))
    assert error <= .00051, (name, error)
    env.close()
    return {'name': name, 'rgb': rgb(frames), 'states': np.asarray(states), 'success': bool(z['successes'][index]),
            'source': archive + '::' + folder, 'metrics': {'max_logged_endpoint_error': error},
            'capture': 'real simulator saved-action replay validated against logged five-control endpoints'}, z['state_g'][index]


def package(key, domain, purpose, arms, goal, notes, unit='Control', overlay=False):
    directory = OUT / key
    directory.mkdir(exist_ok=True)
    assert len(arms) == 2
    np.testing.assert_allclose(arms[0]['states'][0], arms[1]['states'][0], atol=1e-8)
    Image.fromarray(rgb(goal)).save(directory / 'goal.png')
    trajectories = [a['states'][:, :2] if a['states'].ndim == 2 else a['states'][:, :, [0, 2]].mean(1) for a in arms]
    all_xy = np.concatenate(trajectories)
    low, high = all_xy.min(0), all_xy.max(0)
    span = max(float((high - low).max()), .01)
    low = (low + high) / 2 - span * .6
    span *= 1.2
    receipt = {'key': key, 'domain': domain, 'purpose': purpose, 'notes': notes, 'unit': unit, 'arms': [], 'frames': []}
    for i, arm in enumerate(arms):
        folder = directory / ('baseline' if i == 0 else 'ours')
        folder.mkdir(exist_ok=True)
        for old_frame in folder.glob('frame_*.png'):
            old_frame.unlink()
        np.savez_compressed(folder / 'execution.npz', states=arm['states'])
        for step, array in enumerate(arm['rgb']):
            path = folder / f'frame_{step:04d}.png'
            Image.fromarray(array).save(path)
            receipt['frames'].append({'path': str(path.relative_to(OUT)), 'step': step,
                                      'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        receipt['arms'].append({k: v for k, v in arm.items() if k not in ('rgb', 'states')})
        receipt['arms'][-1]['last_step'] = len(arm['rgb']) - 1
    preview, story = [], []
    last = max(len(a['rgb']) for a in arms) - 1
    story_steps = sorted(set([0, last // 3, 2 * last // 3, last]))
    for step in range(last + 1):
        canvas = Image.new('RGB', (1280, 670), '#f5f6f8')
        draw = ImageDraw.Draw(canvas)
        draw.text((24, 12), domain + ' | ' + purpose, font=FONT, fill='#202833')
        draw.text((24, 42), f'{unit} {step}  |  trajectory from actual executed states', font=SMALL, fill='#52606b')
        for i, arm in enumerate(arms):
            actual = min(step, len(arm['rgb']) - 1)
            image = Image.fromarray(arm['rgb'][actual]).resize((448, 448), Image.Resampling.NEAREST)
            if overlay:
                trace = (trajectories[i][:actual + 1] * 448 / 512).tolist()
                if len(trace) > 1:
                    ImageDraw.Draw(image).line([tuple(v) for v in trace], fill=COLORS[i], width=3)
            x = 24 + 476 * i
            draw.text((x, 78), arm['name'], font=FONT, fill=COLORS[i])
            canvas.paste(image, (x, 112))
            ended = actual < step
            draw.text((x, 574), f'{unit} {actual}' + (' | terminal frame held' if ended else ''), font=SMALL, fill='#253342')
            if step == last:
                draw.text((x, 604), 'Success' if arm['success'] else 'Budget reached', font=FONT, fill=COLORS[i])
        canvas.paste(Image.fromarray(rgb(goal)).resize((224, 224), Image.Resampling.NEAREST), (1000, 112))
        draw.text((1000, 78), 'Goal', font=FONT, fill='#253342')
        draw.text((992, 360), 'Executed path (x, y)' if unit == 'Control' else 'Rope centroid (x, z)', font=SMALL, fill='#253342')
        draw.rectangle((1000, 394, 1224, 554), outline='#b9c4cc')
        for i, trace in enumerate(trajectories):
            xy = (trace[:min(step + 1, len(trace))] - low) / span
            xy = xy * [210, 146] + [1007, 401]
            if len(xy) > 1:
                draw.line([tuple(v) for v in xy], fill=COLORS[i], width=2)
        if step in story_steps:
            story.append(canvas.copy())
        preview.append(canvas)
        shown = directory / 'comparison_frames'
        shown.mkdir(exist_ok=True)
        canvas.save(shown / f'frame_{step:04d}.png')
    preview[0].save(directory / 'preview.gif', save_all=True, append_images=preview[1:], duration=150 if unit == 'Control' else 450, loop=0)
    sheet = Image.new('RGB', (1280, 670 * len(story)), 'white')
    for j, frame in enumerate(story):
        sheet.paste(frame, (0, j * 670))
    sheet.save(directory / 'storyboard.jpg', quality=92)
    (directory / 'manifest.json').write_text(json.dumps(receipt, indent=2) + '\n')
    MANIFEST['cases'].append({k: v for k, v in receipt.items() if k != 'frames'})
    print(key, [len(a['rgb']) for a in arms], flush=True)


def main():
    global PUSHT
    spec = importlib.util.spec_from_file_location('native_pusht', REPO / 'env/pusht/pusht_env.py')
    PUSHT = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(PUSHT)
    source('env/pusht/pusht_env.py')
    cases = json.loads(source('paper/cvpr2027/provenance/replayed_paths_v12.json').read_text())['cases']
    frozen = 'reports/publication_v14/archives/main_pushobj_frozen.tar.gz'
    adaptive = 'reports/publication_v14/archives/main_pushobj_adajepa.tar.gz'
    for key, shape, kind in [('small_tee', 'small_tee', 'methods')]:
        case = next(c for c in cases if c['key'] == key)
        base_folder = case['baseline_dir']
        archive = frozen
        if kind == 'methods':
            archive = adaptive
            base_folder = base_folder.replace('stag_frozen', 'stag_adajepa')
        base, goal_state = replay(archive, base_folder, case['sample_idx'], shape,
                                  'Frozen MPC | ECP off' if kind == 'ablation' else 'AdaJEPA')
        ours, ours_goal = replay(frozen, case['ecp_dir'], case['sample_idx'], shape, 'Ours | Frozen + ECP')
        np.testing.assert_array_equal(goal_state, ours_goal)
        assert not base['success'] and ours['success']
        package('pushobj_' + kind, 'PushObj', 'ECP escape' if kind == 'ablation' else 'Method comparison', [base, ours],
                pose_render(shape, goal_state), 'Same target and initial state; replay every saved physical control. Frozen+ECP versus AdaJEPA compares full methods.', overlay=True)
    cost_archive = 'reports/frozen_ecp_cost_20261117/raw_results.tar.gz'
    base, goal_state = replay(cost_archive, 'runs/chunk_16_Z/off', 4, 'Z', 'Frozen MPC | ECP off')
    ours, ours_goal = replay(cost_archive, 'runs/chunk_16_Z/on', 4, 'Z', 'Ours | Frozen + ECP')
    np.testing.assert_array_equal(goal_state, ours_goal)
    package('pushobj_ablation', 'PushObj', 'ECP escape', [base, ours], pose_render('Z', goal_state),
            'Matched Frozen checkpoint and planner, gamma_hi=.45. Shape Z, chunk16 sample4; selected from all 350 paired tasks by object-distance improvement.', overlay=True)
    visual = 'eval_outputs/lewm_shift_visuals_20261116'
    base = recorded(f'{visual}/red_goal_native/recorded_execution.npz', 6, 'LeWM | intervention off')
    ours = recorded(f'{visual}/red_goal_ours/recorded_execution.npz', 6, 'Ours | LeWM + intervention')
    z = np.load(REPO / visual / 'red_goal_ours/recorded_execution.npz')
    goal = z['goal_rgb'][6, 0]
    notes = 'Selected joint pusher/object goal rescue. LeWM intervention includes latent memory and contact/proprio objectives using object pose and geometry. Goal is the separate observation, not the fixed red anchor.'
    package('pusht_ablation', 'PushT', 'Intervention escape', [base, ours], goal, notes, overlay=True)
    base, state_g = replay('reports/publication_v14/archives/pusht_visual_complete.tar.gz',
                          'eval_outputs/pusht_visual_v8/augmented/redAnchor/seed100/adajepa/attempts/001', 6, 'T', 'AdaJEPA', 'Red')
    np.testing.assert_array_equal(state_g, z['state_g'][6])
    package('pusht_methods', 'PushT', 'Method comparison', [base, ours], goal, notes + ' Compared with augmented AdaJEPA, same external target 6.', overlay=True)
    rows = list(csv.DictReader(source('reports/rope_seed101_20/per_sample.csv').open()))
    by = {(r['arm'], int(r['sample'])): r for r in rows}
    arms = [recorded(f'eval_outputs/rope_seed101_20/{arm}/sample_05/recorded_execution.npz', 0, label, True, by[arm, 5])
            for arm, label in [('native3', 'DINO-WM 3/3 | ECP off'), ('fixed3', 'Ours | DINO-WM 3/3 + ECP')]]
    target = pickle.load(source('eval_outputs/rope_seed101_20/native3/sample_05/target.pkl').open('rb'))
    package('rope_ablation', 'Rope', 'ECP escape', arms, target['obs_g']['visual'][0, 0], 'Seed101 target15, matched 3/3 cadence. First ECP application at push7. One frame per executed push.', 'Push')
    visual_rope = 'eval_outputs/rope_qualitative20'
    selected = json.loads(source('reports/rope_formal20/visuals/provenance.json').read_text())['selection'][0]
    arms = []
    for arm, label, cd in [('native5', 'DINO-WM native 5/5', selected['native_cd']),
                            ('fixed3', 'Ours | DINO-WM 3/3 + ECP', selected['ours_cd'])]:
        row = {'success': str(cd < .3), 'final_cd': cd}
        arms.append(recorded(f'{visual_rope}/{arm}/sample_03/recorded_execution.npz', 0, label, True, row))
    rope_target = pickle.load(source(f'{visual_rope}/native5/sample_03/target.pkl').open('rb'))
    package('rope_methods', 'Rope', 'Method comparison', arms, rope_target['obs_g']['visual'][0, 0],
            'Separate recorded visualization, target13. Full planner comparison with native 5/5 versus Ours 3/3 cadence. ECP has zero triggers in this example; the matched-cadence ablation is provided separately.', 'Push')
    root = 'eval_outputs/maze_medium20_s100_g045'
    for kind, base_arm, sample in [('ablation', 'adajepa', 17), ('methods', 'frozen', 13)]:
        base = recorded(f'{root}/{base_arm}/recorded_execution.npz', sample, 'AdaJEPA | ECP off' if kind == 'ablation' else 'Frozen MPC')
        ours = recorded(f'{root}/adajepa_s/recorded_execution.npz', sample, 'Ours | AdaJEPA + gated ECP')
        z = np.load(source(f'{root}/adajepa_s/recorded_execution.npz'))
        b = np.load(source(f'{root}/{base_arm}/recorded_execution.npz'))
        np.testing.assert_array_equal(z['state_g'][sample], b['state_g'][sample])
        assert not base['success'] and ours['success']
        package('maze_' + kind, 'Maze', 'ECP escape' if kind == 'ablation' else 'Method comparison', [base, ours], z['target_rgb'][sample, 0],
                f'Seed100 medium maze target{sample}; gamma_hi=.45, gamma_lo=.05; real replay observations at every control. Path inset uses simulator world coordinates.')
    (OUT / 'manifest.json').write_text(json.dumps(MANIFEST, indent=2) + '\n')


if __name__ == '__main__':
    main()
