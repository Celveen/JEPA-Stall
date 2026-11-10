#!/usr/bin/env python3
"""Run plan.py with reproducible, freshly sampled salt/pepper noise only."""
import argparse
from pathlib import Path
import runpy
import sys

from paired_noise import install_pairing


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--repo", type=Path, required=True)
    args, overrides = parser.parse_known_args()
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo))
    from planning.evaluator import PlanEvaluator
    from planning.image_corruption import corrupt_obs_dict
    install_pairing(PlanEvaluator, corrupt_obs_dict)
    sys.argv = [str(repo / "plan.py")] + overrides
    runpy.run_path(str(repo / "plan.py"), run_name="__main__")


if __name__ == "__main__":
    main()
