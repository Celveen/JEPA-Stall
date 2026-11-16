"""Read compact progress without loading the simulator or model."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root', type=Path, default=Path('eval_outputs/rope_formal20'))
args = parser.parse_args()
for arm in ('native5', 'native3', 'fixed3', 'dynamic3', 'fixed3_nowarm'):
    if not (args.root / arm).exists():
        continue
    paths = list((args.root / arm).glob('sample_*/complete.json'))
    active = []
    for path in (args.root / arm).glob('sample_*/ecp_progress.json'):
        if (path.parent / 'complete.json').exists():
            continue
        try:
            d = json.loads(path.read_text())
            active.append(dict(sample=path.parent.name, pushes=d['completed_pushes'], cd=d['cd']))
        except json.JSONDecodeError:
            pass
    print(arm, f'{len(paths)}/20', json.dumps(active))
failure = args.root / 'failure.json'
if failure.exists():
    print('FAILURE', failure.read_text())
