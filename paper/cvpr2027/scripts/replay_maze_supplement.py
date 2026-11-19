"""Export the two Maze figure paths with MuJoCo 2.2.2 and the repository environment."""
from pathlib import Path
import sys

import mujoco
import numpy as np

PAPER = Path(__file__).resolve().parents[1]
REPO = PAPER.parents[1]
RAW = REPO / 'reports/maze_medium50_three_seeds_observed/rollouts/raw'


def export_paths(point_maze):
    output = {}
    for seed, sample in ((101, 25), (102, 13)):
        case = f'adajepa_seed{seed}_sample{sample}'
        with np.load(REPO / f'reports/maze_medium50_s{seed}_g030_observed/goals.npz') as goals:
            with point_maze(goals['map_specs'][sample - 1]).asfile() as xml:
                model = mujoco.MjModel.from_xml_path(xml.name)
        for arm in ('native', 'plus_s'):
            with np.load(RAW / case / f'{arm}.npz') as packet:
                states = packet['states']
                controls = packet['physical_actions']
            data = mujoco.MjData(model)
            data.qpos[:], data.qvel[:] = states[0, :2], states[0, 2:]
            path = [data.qpos.copy()]
            for index, action in enumerate(controls, 1):
                data.ctrl[:] = np.clip(action, -1, 1)
                for _ in range(4):
                    data.qvel[:] = np.clip(data.qvel, -5, 5)
                    mujoco.mj_forward(model, data)
                    for _ in range(5):
                        mujoco.mj_step(model, data)
                        path.append(data.qpos.copy())
                np.testing.assert_allclose(np.r_[data.qpos, data.qvel], states[index], rtol=0, atol=5e-7)
            output[f'{case}_{arm}'] = np.asarray(path)
    np.savez_compressed(PAPER / 'provenance/maze_recorded_figure_paths.npz', **output)


if __name__ == '__main__':
    sys.path.insert(0, str(REPO))
    from env.pointmaze.maze_model import point_maze
    export_paths(point_maze)
