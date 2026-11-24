"""Capture saved Rope actions at four-physics-step intervals without adding physics steps."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

REPO = Path(__file__).resolve().parents[3]
LOCAL = REPO / 'eval_outputs/rope_local'
sys.path[:0] = [str(LOCAL / 'runtime'), str(LOCAL / 'pyflex-build')]
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from env.deformable_env.FlexEnvWrapper import FlexEnvWrapper
import pyflex


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    src = args.source.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / 'recording.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX)
    if (out / 'complete.json').exists():
        return
    receipt = json.loads((src / 'run_manifest.json').read_text())
    planner = receipt['config']['planner']
    cache = np.load(src / 'recorded_execution.npz')
    actions = cache['physical_actions'][0]
    states = cache['states'][0]
    import pickle
    target_file = src / 'target.pkl'
    if not target_file.exists():
        target_file = src / 'plan_targets.pkl'
    target = pickle.load(target_file.open('rb'))
    goal = np.asarray(target['state_g']).reshape(-1, 4)[:, :3]
    Image.fromarray(np.asarray(target['obs_g']['visual'][0, 0], dtype=np.uint8)).save(out / 'goal.png')
    np.savez_compressed(out / 'source_execution.npz', actions=actions, states=states, goal=goal)
    (out / 'source_manifest.json').write_text(json.dumps(receipt, indent=2))
    env = FlexEnvWrapper('rope')
    original_step = pyflex.step
    original_pos, original_angle = pyflex.set_camPos, pyflex.set_camAngle
    camera = {}
    def set_pos(value):
        camera['pos'] = np.asarray(value).copy()
        return original_pos(value)
    def set_angle(value):
        camera['angle'] = np.asarray(value).copy()
        return original_angle(value)
    pyflex.set_camPos, pyflex.set_camAngle = set_pos, set_angle
    active = False
    physics_step = 0
    frame_count = 0
    frame_records = []
    folder = None
    def capture():
        nonlocal frame_count
        old = {k: v.copy() for k, v in camera.items()}
        original_pos(env.camPos_list[env.camera_view])
        original_angle(env.camAngle_list[env.camera_view])
        image = pyflex.render(render_depth=True).reshape(env.screenHeight, env.screenWidth, 5)[..., :3][..., ::-1].astype(np.uint8)
        original_pos(old['pos'])
        original_angle(old['angle'])
        path = folder / f'frame_{frame_count:05d}.png'
        Image.fromarray(image).save(path)
        frame_records.append({'frame': frame_count, 'physics_step': physics_step, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        frame_count += 1
    def traced_step(*values, **kwargs):
        nonlocal physics_step
        result = original_step(*values, **kwargs)
        if active:
            physics_step += 1
            if physics_step % 4 == 0:
                capture()
        return result
    pyflex.step = traced_step
    pushes = []
    limit = min(len(actions), args.limit or len(actions))
    for index, action in enumerate(actions[:limit]):
        # The original planner restores positions at each chunk boundary.
        if index % planner['chunk_size'] == 0:
            env.prepare(planner['episode_seed'], states[index])
        folder = out / f'push_{index+1:02d}'
        folder.mkdir(exist_ok=True)
        frame_count, physics_step, frame_records = 0, 0, []
        capture()
        active = True
        obs, _, _, info = env.step_multiple(action[None])
        active = False
        if frame_records[-1]['physics_step'] != physics_step:
            capture()
        final = info['state'][-1]
        cd = float(cKDTree(goal).query(final[:, :3])[0].mean() + cKDTree(final[:, :3]).query(goal)[0].mean())
        error = float(np.max(np.abs(final - states[index+1])))
        Image.fromarray(obs['visual'][-1].astype(np.uint8)).save(out / f'after_push_{index+1:02d}.png')
        np.save(out / f'after_push_{index+1:02d}.npy', final)
        metadata = {'push': index+1, 'physics_steps': physics_step, 'frames': frame_records, 'replay_cd': cd,
                    'max_particle_coordinate_difference_from_source': error, 'restored_chunk_start': index % planner['chunk_size'] == 0}
        (out / f'push_{index+1:02d}.json').write_text(json.dumps(metadata, indent=2))
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-framerate', '30', '-i', str(folder/'frame_%05d.png'),
                        '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out/f'push_{index+1:02d}.mp4')], check=True)
        with tarfile.open(out/f'push_{index+1:02d}_frames.tar.gz', 'w:gz') as tar:
            for path in sorted(folder.glob('*.png')):
                tar.add(path, arcname=path.name)
        for path in folder.glob('*.png'):
            path.unlink()
        folder.rmdir()
        pushes.append({k:v for k,v in metadata.items() if k != 'frames'} | {'frame_count':frame_count})
        print(json.dumps(pushes[-1]), flush=True)
    env.close()
    listing = out / 'concat.txt'
    listing.write_text(''.join(f"file 'push_{p['push']:02d}.mp4'\n" for p in pushes))
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', str(listing), '-c', 'copy', '-movflags', '+faststart', str(out/'full_execution.mp4')], check=True)
    summary = {'source':str(src.relative_to(REPO)), 'source_sha256':hashlib.sha256((src/'recorded_execution.npz').read_bytes()).hexdigest(),
               'capture':'Saved-action PyFleX replay, source positions restored at original chunk boundaries; direct pyflex.render adds no physics steps.',
               'sample_interval_physics_steps':4, 'playback_fps':30, 'pushes':pushes, 'final_replay_cd':pushes[-1]['replay_cd'],
               'original_final_cd':float(cKDTree(goal).query(states[-1,:,:3])[0].mean()+cKDTree(states[-1,:,:3]).query(goal)[0].mean())}
    (out/'complete.json').write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
