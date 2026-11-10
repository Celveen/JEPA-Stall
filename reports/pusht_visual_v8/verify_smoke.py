#!/usr/bin/env python3
"""Verify completed smoke outputs before launching the full evaluation matrix."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.output / "manifest.json").read_text())
    assert manifest["n_evals"] == 2, "Use only the separate two-episode smoke output"
    checks = []
    for job in manifest["jobs"]:
        if not job["ecp"]:
            continue
        ecproot = args.output / job["id"]
        nativeroot = args.output / job["id"].removesuffix("_ecp")
        def completed(root):
            data = json.loads((root / "completion.json").read_text())
            return root / data["attempt"]
        ecpdir, nativedir = completed(ecproot), completed(nativeroot)
        ecp, native = np.load(ecpdir / "final_eval_results.npz"), np.load(nativedir / "final_eval_results.npz")
        np.testing.assert_array_equal(ecp["state_0"], native["state_0"])
        np.testing.assert_array_equal(ecp["state_g"], native["state_g"])
        episodes = [json.loads(line) for line in (ecpdir / "stag_episodes.jsonl").read_text().splitlines() if line.strip()]
        for episode in episodes:
            sample = episode["sample_idx"]
            first = episode["first_trigger"]
            shared = first if first is not None else min(ecp["actions"].shape[1], native["actions"].shape[1])
            left, right = ecp["actions"][sample, :shared], native["actions"][sample, :shared]
            np.testing.assert_allclose(left, right, atol=1e-5, rtol=1e-5)
            checks.append({"job": job["id"], "sample": sample, "shared_pretrigger_actions": int(shared),
                           "max_abs_action_difference": float(np.max(np.abs(left-right))) if shared else 0.,
                           "first_trigger": first})
    result = {"status": "passed", "pairs": len(checks), "checks": checks}
    (args.output / "smoke_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
