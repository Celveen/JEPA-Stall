"""Replay saved controls without rendering and verify outcomes against the run."""

import csv
import json
import multiprocessing as mp
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "source"))
from datasets.pusht_dset import ACTION_MEAN, ACTION_STD
from env.pusht.pusht_wrapper import PushTWrapper

SHAPES = ("I", "L", "square", "Z", "small_tee", "plus", "T")


def replay_shape(shape):
    env = PushTWrapper()
    env.update_env({"shape": "+" if shape == "plus" else shape})
    env._render_frame = lambda *args, **kwargs: np.zeros((1, 1, 3), dtype=np.uint8)
    touched_object = False

    def pusher_contact(arbiter, space, data):
        nonlocal touched_object
        first, second = (shape.body for shape in arbiter.shapes)
        if (first == env.agent and second == env.block) or (first == env.block and second == env.agent):
            touched_object = True

    target = ROOT / "replayed" / shape
    target.mkdir(parents=True, exist_ok=True)
    for condition, path in (("burst4", ROOT / "full" / shape),):
        with np.load(path / "final_eval_results.npz") as data:
            state_0, state_g = data["state_0"], data["state_g"]
            actions = data["actions"].reshape(len(state_0), -1, 2)
            controls = actions * ACTION_STD.numpy() + ACTION_MEAN.numpy()
            horizon = controls.shape[1] // 5
            state_trace = []
            contact_trace = []
            outcomes = []
            for i, (start, goal, commands) in enumerate(zip(state_0, state_g, controls)):
                _, state = env.prepare(2000 * i + 1, start)
                env.collision_handeler.post_solve = pusher_contact
                states = [state]
                contacts = []
                for control in commands:
                    touched_object = False
                    _, _, _, info = env.step(control)
                    states.append(info["state"])
                    contacts.append(touched_object)
                state_trace.append(states)
                contact_trace.append(contacts)
                last = int(min(data["action_len"][i], horizon)) * 5
                outcomes.append(env.eval_state(goal, states[last])["success"])
            np.testing.assert_array_equal(outcomes, data["successes"], err_msg=f"{shape} {condition}")
            np.savez_compressed(
                target / f"{condition}.npz", states=np.asarray(state_trace), contacts=np.asarray(contact_trace),
                goals=state_g, success=data["successes"], action_len=data["action_len"],
            )
    return shape


if __name__ == "__main__":
    with mp.get_context("fork").Pool(len(SHAPES)) as pool:
        for shape in pool.imap_unordered(replay_shape, SHAPES):
            print(f"{shape}: replay outcomes verified", flush=True)
