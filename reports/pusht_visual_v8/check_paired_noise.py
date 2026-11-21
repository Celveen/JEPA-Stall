#!/usr/bin/env python3
"""Check the evaluation noise contract without a model, GPU, or simulator."""
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np
from paired_noise import install_pairing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    path = args.repo / "planning/image_corruption.py"
    spec = importlib.util.spec_from_file_location("tested_corruption", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    class Evaluator:
        def __init__(self, seed):
            self.seed, self.ood_corruption, self.ood_level = [seed], "snp1", 0.01
        def _corrupt(self, obs, seed_offset=0, random_per_call=False):
            return module.corrupt_obs_dict(obs, self.ood_corruption, self.ood_level, seed=self.seed[0]+seed_offset)
    install_pairing(Evaluator, module.corrupt_obs_dict)
    obs = {"visual": np.full((1, 4, 64, 64, 3), 127, dtype=np.uint8),
           "proprio": np.array([[[1., 2.]]])}
    off, on = Evaluator(101), Evaluator(101)
    off0, on0 = off._corrupt(obs, random_per_call=True), on._corrupt(obs, random_per_call=True)
    off1, on1 = off._corrupt(obs, random_per_call=True), on._corrupt(obs, random_per_call=True)
    assert np.array_equal(off0["visual"], on0["visual"])
    assert np.array_equal(off1["visual"], on1["visual"])
    assert not np.array_equal(off0["visual"], off1["visual"])
    other = Evaluator(201)._corrupt(obs, random_per_call=True)
    assert not np.array_equal(off0["visual"], other["visual"])
    # A different-length preceding episode cannot shift the next episode's stream.
    interleaved = Evaluator(1)
    for _ in range(7):
        interleaved._corrupt(obs, random_per_call=True)
    interleaved.seed = [101]
    assert np.array_equal(interleaved._corrupt(obs, random_per_call=True)["visual"], off0["visual"])
    assert off0["proprio"] is obs["proprio"]
    assert np.all(obs["visual"] == 127)
    initial = Evaluator(101)._corrupt(obs, random_per_call=False)
    expected = module.corrupt_obs_dict(obs, "snp1", .01, seed=101)
    assert np.array_equal(initial["visual"], expected["visual"])
    changed = np.mean(off0["visual"][..., 0] != 127)
    assert .015 < changed < .025  # Independent 1% salt + 1% pepper, with repeated draws.
    checks = {"status": "passed", "no_model_or_simulator_executed": True,
              "same_episode_same_call_identical": True, "fresh_noise_each_call": True,
              "different_episode_different_noise": True, "independent_of_previous_episode_length": True,
              "initial_target_corruption_unchanged": True, "proprio_unchanged": True,
              "observed_changed_pixel_fraction": float(changed),
              "scope": "Noise-stream pairing only. Full pre-trigger planner equivalence requires the GPU smoke."}
    args.output.write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()

