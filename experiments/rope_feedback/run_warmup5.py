"""Run the five-push warmup ablation against the unchanged Native baseline."""

import copy
from pathlib import Path
import sys
import time

import run as base

VARIANT = Path(__file__).resolve()
original_build_config = base.build_config
original_validate_reference = base.validate_reference


def build_config(arm, chunk, output, reference=None):
    if arm != 'S':
        raise ValueError('This ablation reuses Native; run it with --arm S.')
    config = original_build_config(arm, chunk, output, reference)
    config['planner']['ecp'].update(warmup_pushes=5, variant_name='warmup5',
                                     variant_sha256=base.sha256(VARIANT))
    return config


def validate_reference(previous, receipt):
    # Only the observation-only detector warmup differs. Native has kappa=0,
    # so its actions do not depend on this detector or its accumulated memories.
    adjusted = copy.deepcopy(receipt)
    ecp = adjusted['config']['planner']['ecp']
    if ecp['warmup_pushes'] != 5 or previous['config']['planner']['ecp']['warmup_pushes'] != 3:
        raise ValueError('Expected the explicit 3-push versus 5-push warmup ablation.')
    if previous['config']['planner']['ecp']['kappa'] != 0:
        raise ValueError('Shared baseline must have no S intervention.')
    ecp['warmup_pushes'] = 3
    ecp.pop('variant_name')
    ecp.pop('variant_sha256')
    original_validate_reference(previous, adjusted)


if __name__ == '__main__':
    if '--wait-for-reference' in sys.argv:
        sys.argv.remove('--wait-for-reference')
        if '--reference' not in sys.argv:
            raise SystemExit('--wait-for-reference requires --reference.')
        reference = Path(sys.argv[sys.argv.index('--reference') + 1]).resolve()
        print(f'Waiting for completed Native baseline: {reference}', flush=True)
        while not (reference / 'complete.json').is_file():
            time.sleep(30)
    base.build_config = build_config
    base.validate_reference = validate_reference
    base.main()
