"""Audit every paired episode and export figures from recorded simulator RGB."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / 'eval_outputs/lewm_shift_visuals_20261116'
OUT = ROOT / 'reports/lewm_shift_visuals_20261116'
COLORS = {'native': '#ab343b', 'ours': '#167467'}
LABELS = {'native': 'Native (method off)', 'ours': 'Ours (method on)'}
CONDITIONS = {'red_goal': 'Red goal', 'red_agent': 'Red agent'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def errors(states, goal):
    pos = np.linalg.norm(states[..., :4] - goal[..., :4], axis=-1)
    obj = np.linalg.norm(states[..., 2:4] - goal[..., 2:4], axis=-1)
    angle = np.degrees(np.abs((states[..., 4] - goal[..., 4] + np.pi) % (2 * np.pi) - np.pi))
    return pos, obj, angle


def load(condition, arm):
    folder = RAW / f'{condition}_{arm}'
    receipt = json.loads((folder / 'complete.json').read_text())
    assert receipt['status'] == 'complete' and not receipt['diagnostic_only']
    with np.load(folder / 'recorded_execution.npz') as f:
        data = {k: f[k] for k in f.files}
    with np.load(folder / 'final_eval_results.npz') as f:
        for key in ['successes', 'action_len', 'state_0', 'state_g']:
            np.testing.assert_array_equal(data[key], f[key])
        data['normalized_actions'] = f['actions']
    frozen = json.loads((folder / 'frozen_model_check.json').read_text())
    assert frozen['unchanged']
    data['frozen_sha'] = frozen['weights_sha256']
    data['receipt'] = receipt
    data['stop'] = np.where(np.isfinite(data['action_len']), data['action_len'] * 5,
                            data['rgb'].shape[1] - 1).astype(int)
    assert data['rgb'].dtype == np.uint8 and data['rgb'].shape[2:] == (224, 224, 3)
    # Compare the first successful MPC checkpoint with the recorded final outcome.
    for i, stop in enumerate(data['stop']):
        pos, _, angle = errors(data['states'][i], data['state_g'][i])
        reached = (pos < 20) & (angle < 20)
        checkpoints = np.flatnonzero(reached[5::5]) * 5 + 5
        assert bool(reached[stop]) == bool(data['successes'][i])
        if data['successes'][i]:
            assert len(checkpoints) and checkpoints[0] == stop
        else:
            assert not len(checkpoints)
    return data


def panel(ax, rgb, title, color='#202b38'):
    ax.imshow(rgb, interpolation='nearest')
    ax.set_title(title, fontsize=10, color=color, pad=9)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color('#cbd2d8')


def export_example(condition, i, data, row):
    dest = OUT / 'figures' / condition
    dest.mkdir(parents=True, exist_ok=True)
    title = f"LeWM | {CONDITIONS[condition]} | target {i:02d}"
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.8))
    panel(axes[0], data['native']['rgb'][i, 0], 'Shared initial state')
    panel(axes[1], data['native']['goal_rgb'][i, 0], 'Actual target observation')
    for ax, arm in zip(axes[2:], ['native', 'ours']):
        stop = data[arm]['stop'][i]
        panel(ax, data[arm]['rgb'][i, stop],
              f"{LABELS[arm]} — {'SUCCESS' if row[arm + '_success'] else 'FAIL'}\n"
              f"{stop} controls | xy {row[arm + '_joint_xy_px']:.1f} px | angle {row[arm + '_angle_deg']:.1f}°",
              COLORS[arm])
        pusher = data[arm]['states'][i, stop, :2]
        if np.any((pusher < 0) | (pusher > 512)):
            ax.text(.5, -.07, 'Pusher is outside the simulator viewport',
                    transform=ax.transAxes, ha='center', fontsize=8, color=COLORS[arm])
    fig.suptitle(title + ' — selected paired rescue', fontsize=15, y=.99)
    fig.text(.5, .035, 'Same frozen checkpoint and 150-control limit. Success: joint pusher/object xy <20 px AND angle <20°.\n'
             'Unmodified simulator frames. The colored background T is a fixed visual anchor, not the task target.',
             ha='center', fontsize=9)
    fig.subplots_adjust(left=.02, right=.98, top=.77, bottom=.19, wspace=.10)
    for ext in ['png', 'pdf']:
        fig.savefig(dest / f'comparison.{ext}', dpi=200, facecolor='white')
    plt.close(fig)

    fig, axes = plt.subplots(2, 7, figsize=(16, 5.5))
    for r, arm in enumerate(['native', 'ours']):
        stop = int(data[arm]['stop'][i])
        for ax, t in zip(axes[r], range(0, 151, 25)):
            actual = min(t, stop)
            label = f't = {actual}' if t <= stop else f'Stopped at t = {stop}'
            panel(ax, data[arm]['rgb'][i, actual], label, COLORS[arm])
        axes[r, 0].set_ylabel(LABELS[arm], fontsize=11, color=COLORS[arm])
    fig.suptitle(title + ' — recorded execution', fontsize=16)
    fig.text(.5, .02, 't counts physical controls (5 controls per replan). Frames after success are explicitly held at the stopping frame.\n'
             'Ours: latent stall memory + analytic pusher/contact objectives; contact guidance uses observed object pose and known T geometry.',
             ha='center', fontsize=9)
    fig.subplots_adjust(left=.035, right=.99, top=.86, bottom=.13, wspace=.08, hspace=.3)
    fig.savefig(dest / 'timeline.png', dpi=180, facecolor='white')
    fig.savefig(dest / 'timeline.pdf', facecolor='white')
    plt.close(fig)

    # A lossless, every-three-control contact sheet plus all original PNG frames.
    for arm in ['native', 'ours']:
        stop = int(data[arm]['stop'][i])
        frame_dir = dest / f'{arm}_frames'
        frame_dir.mkdir(exist_ok=True)
        for t in range(stop + 1):
            Image.fromarray(data[arm]['rgb'][i, t]).save(frame_dir / f'{t:03d}.png')
        times = sorted(set(range(0, stop + 1, 3)) | {stop})
        nrows = (len(times) + 7) // 8
        fig, axes = plt.subplots(nrows, 8, figsize=(16, 2.25 * nrows + .6), squeeze=False)
        for ax in axes.flat:
            ax.axis('off')
        for ax, t in zip(axes.flat, times):
            panel(ax, data[arm]['rgb'][i, t], f't = {t}', COLORS[arm])
        fig.suptitle(f'{title} | {LABELS[arm]} | every 3 controls + terminal frame', fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, .97))
        fig.savefig(dest / f'{arm}_every3.png', dpi=140, facecolor='white')
        plt.close(fig)
    Image.fromarray(data['native']['goal_rgb'][i, 0]).save(dest / 'target.png')

    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font = ImageFont.truetype(font_path, 15)
    small = ImageFont.truetype(font_path, 12)
    animation = []
    max_stop = max(int(data[arm]['stop'][i]) for arm in ['native', 'ours'])
    for t in range(max_stop + 1):
        canvas = Image.new('RGB', (3 * 244 + 20, 315), 'white')
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 8), title, fill='#202b38', font=font)
        canvas.paste(Image.fromarray(data['native']['goal_rgb'][i, 0]), (10, 59))
        draw.text((10, 37), 'Actual target', fill='#202b38', font=font)
        for col, arm in enumerate(['native', 'ours'], 1):
            stop = int(data[arm]['stop'][i])
            actual = min(t, stop)
            x = col * 244 + 10
            canvas.paste(Image.fromarray(data[arm]['rgb'][i, actual]), (x, 59))
            draw.text((x, 37), LABELS[arm], fill=COLORS[arm], font=font)
            status = 'SUCCESS; stopped' if data[arm]['successes'][i] and t >= stop else ('FAIL; budget reached' if t == stop else 'Running')
            draw.text((x, 287), f't={actual}: {status}', fill=COLORS[arm], font=small)
        animation.append(canvas)
    animation[0].save(dest / 'paired_execution.gif', save_all=True, append_images=animation[1:],
                      duration=[100] * (len(animation) - 1) + [2000], loop=0, optimize=False)

    fig, axes = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True)
    for arm in ['native', 'ours']:
        stop = int(data[arm]['stop'][i])
        pos, _, angle = errors(data[arm]['states'][i, :stop + 1], data[arm]['state_g'][i])
        for ax, values in zip(axes, [pos, angle]):
            ax.plot(range(stop + 1), values, color=COLORS[arm], label=LABELS[arm])
            ax.scatter([stop], [values[-1]], color=COLORS[arm], s=22)
    for ax in axes:
        ax.axhline(20, ls='--', color='#777777', lw=1, label='Success threshold')
        ax.grid(alpha=.18)
    axes[0].set_ylabel('Joint xy error (px)')
    axes[1].set_ylabel('Object angle error (deg)')
    axes[1].set_xlabel('Physical controls')
    axes[0].legend(fontsize=9)
    fig.suptitle(title + ' — measured simulator state')
    fig.tight_layout()
    fig.savefig(dest / 'errors.png', dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows, summaries, selections = [], {}, {}
    for condition in CONDITIONS:
        data = {arm: load(condition, arm) for arm in ['native', 'ours']}
        n, o = data['native'], data['ours']
        assert n['frozen_sha'] == o['frozen_sha']
        assert n['receipt']['checkpoint_sha256'] == o['receipt']['checkpoint_sha256']
        for key in ['state_0', 'state_g', 'goal_rgb']:
            np.testing.assert_array_equal(n[key], o[key])
        np.testing.assert_array_equal(n['rgb'][:, 0], o['rgb'][:, 0])
        episodes = {v['sample_idx']: v for v in jsonl(RAW / f'{condition}_ours/lewm_ecp_episodes.jsonl')}
        contact = jsonl(RAW / f'{condition}_ours/lewm_contact_objective.jsonl')
        proprio = jsonl(RAW / f'{condition}_ours/lewm_proprio_weight.jsonl')
        current_rows = []
        for i in range(10):
            episode = episodes[i]
            common = min(n['stop'][i], o['stop'][i])
            changes = np.linalg.norm((o['controls'][i, :common] - n['controls'][i, :common]) * 100, axis=-1)
            first = np.flatnonzero(changes > 1e-4)
            # Before a target's first detector activation, both controls must match.
            prefix = min(common, 5 * episode['first_trigger']) if episode['first_trigger'] is not None else common
            pre_difference = float(changes[:prefix].max()) if prefix else 0.
            assert pre_difference < 1e-3, (condition, i, pre_difference)
            row = dict(condition=condition, target_index=i, native_success=bool(n['successes'][i]),
                       ours_success=bool(o['successes'][i]), first_trigger_replan=episode['first_trigger'],
                       trigger_count=episode['trigger_count'],
                       contact_guided_replans=sum(any(a['sample_idx'] == i for a in r['actions']) for r in contact),
                       proprio_guided_replans=sum(i in r['sample_idx'] for r in proprio),
                       first_changed_control=int(first[0] + 1) if len(first) else None,
                       changed_controls_in_common_prefix=int(len(first)),
                       max_command_difference_px=float(changes.max()) if len(changes) else 0.,
                       pretrigger_max_command_difference_px=pre_difference)
            for arm in ['native', 'ours']:
                stop = int(data[arm]['stop'][i])
                pos, obj, angle = errors(data[arm]['states'][i, stop], data[arm]['state_g'][i])
                row.update({arm + '_stop_controls': stop, arm + '_joint_xy_px': float(pos),
                            arm + '_object_xy_px': float(obj), arm + '_angle_deg': float(angle),
                            arm + '_pusher_xy_px': float(np.linalg.norm(
                                data[arm]['states'][i, stop, :2] - data[arm]['state_g'][i, :2]))})
            current_rows.append(row)
        rows.extend(current_rows)
        rescues = [r['target_index'] for r in current_rows if r['ours_success'] and not r['native_success']]
        regressions = [r['target_index'] for r in current_rows if r['native_success'] and not r['ours_success']]
        summaries[condition] = dict(native_successes=int(n['successes'].sum()), ours_successes=int(o['successes'].sum()),
                                    episodes=10, rescues=rescues, regressions=regressions)
        selections[condition] = rescues[0] if rescues else None
        if rescues and not args.audit_only:
            export_example(condition, rescues[0], data, current_rows[rescues[0]])
    with (OUT / 'paired_results.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(conditions=summaries, selected_targets=selections,
                   selection_rule='first target in file order with Ours success and Native failure; conditional qualitative examples',
                   all_weights_frozen=True, paired_targets_and_initial_rgb_exact=True,
                   pretrigger_action_agreement_verified=True, terminal_success_recomputed_from_recorded_states=True,
                   all_episode_rows=rows)
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'all_episode_rows'}, indent=2))
    if args.audit_only:
        return
    if any(i is None for i in selections.values()):
        raise RuntimeError('No rescue in a requested condition: retain results; do not fabricate a figure')
    evidence = OUT / 'evidence'
    evidence.mkdir(exist_ok=True)
    for condition in CONDITIONS:
        for arm in ['native', 'ours']:
            src = RAW / f'{condition}_{arm}'
            dst = evidence / src.name
            dst.mkdir(exist_ok=True)
            for path in src.iterdir():
                if path.name not in ['recorded_execution.npz', 'plan_targets.pkl'] and path.is_file():
                    shutil.copy2(path, dst / path.name)
    archive = OUT / 'all_recorded_rollouts.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        for condition in CONDITIONS:
            for arm in ['native', 'ours']:
                folder = RAW / f'{condition}_{arm}'
                tar.add(folder / 'recorded_execution.npz', arcname=folder.name + '/recorded_execution.npz')
                tar.add(RAW / (folder.name + '.log'), arcname=folder.name + '/console.log')
    lines = ['# LeWM: real red-goal and red-agent rescue examples', '',
             'Same frozen LeWM checkpoint, same initial/target states, same seed and planning budget. '
             'Native switches the intervention off; Ours enables the archived LeWM intervention. '
             'These are selected qualitative examples, not an unbiased estimate of the method’s overall benefit.', '',
             '| Condition | Native successes | Ours successes | Rescued targets | Regressed targets |',
             '|---|---:|---:|---|---|']
    for condition, s in summaries.items():
        lines.append(f"| {CONDITIONS[condition]} | {s['native_successes']}/10 | {s['ours_successes']}/10 | {s['rescues']} | {s['regressions']} |")
    for condition, i in selections.items():
        row = next(r for r in rows if r['condition'] == condition and r['target_index'] == i)
        lines += ['', f'## {CONDITIONS[condition]} — target {i:02d}', '',
                  f"Native fails after {row['native_stop_controls']} controls: joint xy error "
                  f"{row['native_joint_xy_px']:.2f} px, angle error {row['native_angle_deg']:.2f}°. "
                  f"Ours succeeds after {row['ours_stop_controls']} controls: "
                  f"{row['ours_joint_xy_px']:.2f} px and {row['ours_angle_deg']:.2f}°. "
                  f"Detector activations: {row['trigger_count']}; contact-guided replans: "
                  f"{row['contact_guided_replans']}. The first changed physical control is "
                  f"{row['first_changed_control']} (one-based).", '',
                  f'![{CONDITIONS[condition]} paired comparison](figures/{condition}/comparison.png)', '',
                  f'[Recorded execution GIF](figures/{condition}/paired_execution.gif) · '
                  f'[Timeline](figures/{condition}/timeline.png) · '
                  f'[State errors](figures/{condition}/errors.png) · '
                  f'[Native every 3 controls](figures/{condition}/native_every3.png) · '
                  f'[Ours every 3 controls](figures/{condition}/ours_every3.png)']
        if condition == 'red_goal':
            lines += ['', 'In this example Native already places the object within tolerance but leaves the '
                      'pusher too far from its target. Ours completes the joint pusher/object criterion. '
                      f"Object-only position error is {row['native_object_xy_px']:.2f} px for Native and "
                      f"{row['ours_object_xy_px']:.2f} px for Ours; this case should not be described as "
                      'an improvement in object-only position accuracy.']
        else:
            lines += ['', 'Native’s red pusher exits the 512×512 simulator viewport. Its absence in the '
                      'terminal screenshot is the recorded simulator output, not a crop or removed element. '
                      'The error plot and raw states retain its out-of-view position.']
    lines += ['', '## Protocol and evidence', '',
              '- First 10 fixed targets in file order, independently repeated under each color condition. No parameter tuning for displayed outcomes.',
              '- GD100, horizon 25 physical controls, execute 5 per replan, at most 30 replans / 150 controls. Reuse the optimized tail. This is separate from the formal clean CEM50 score.',
              '- PushT success: joint pusher/object xy error <20 px **and** wrapped object angle error <20°. Success is checked at each 5-control MPC checkpoint; stop at its first success.',
              '- Ours: latent stagnation memory, conditional analytic pusher objective and contact objective (weight 10; four-replan bursts, at most seven guided replans). The contact component uses observed simulator object pose and known T geometry. No weight adaptation and no post-optimization additive offset.',
              '- RGB comes directly from the final evaluated simulator rollout in the archived resim-from-start protocol. Screenshots preserve the original 224×224 pixels; labels are outside the image. GIF color quantization is for playback only. Original lossless PNGs and NPZ arrays are included.',
              '- The colored T-shaped background anchor is fixed by the environment. The actual planning goal is the separately shown target observation, generated by replaying the stored target actions; it is not the colored anchor.',
              '- Success trajectories end at first success. Timelines/GIFs explicitly hold that terminal frame while the Native trajectory continues. All physical steps are available; contact sheets sample every three controls and append the terminal frame.',
              '- Audit checks: exact paired states and initial/goal RGB; frozen weight digest; pre-trigger command agreement; final success and first-success time independently recomputed from recorded simulator states.', '',
              '[All 20 paired outcomes](paired_results.csv) · [Summary and checks](summary.json) · '
              '[Configs, per-episode intervention logs and actions](evidence/) · '
              '[All 40 recorded RGB/state/action trajectories](all_recorded_rollouts.tar.gz) · '
              '[Reproduction code](../../experiments/lewm_shift_visuals_20261116/)', '',
              f'Rollout archive SHA256: `{sha(archive)}`.', '',
              'Checkpoint SHA256: `9e2b8afee10e46f3b7ae2bd5a9b67ebc28e86af0ccc0d0d293e2fd77471d2576`. '
              'Target-file SHA256: `1950b39e5d3ba13b30c14a435a400d84cd3fbe30138cdfed55e98b202d76277b`.', '']
    (OUT / 'README.md').write_text('\n'.join(lines))
    manifest = {str(p.relative_to(OUT)): sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name != 'sha256.json'}
    (OUT / 'sha256.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
