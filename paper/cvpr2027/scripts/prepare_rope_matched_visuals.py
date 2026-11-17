"""Extract full-frame observations from audited, equal-push Rope executions."""
import argparse
import collections
import csv
import hashlib
import io
import json
from pathlib import Path
import pickle
import tarfile
import numpy as np
from PIL import Image

class ArrayOnlyUnpickler(pickle.Unpickler):
    """Read NumPy observations without reconstructing archived torch objects."""
    def find_class(self, module, name):
        allowed = {
            ('numpy.core.multiarray', '_reconstruct'): np._core.multiarray._reconstruct,
            ('numpy', 'ndarray'): np.ndarray,
            ('numpy', 'dtype'): np.dtype,
            ('collections', 'OrderedDict'): collections.OrderedDict,
            ('torch.storage', '_load_from_bytes'): lambda _: None,
            ('torch._utils', '_rebuild_tensor_v2'): lambda *args: None,
        }
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f'Unsupported archived class: {module}.{name}')
        return allowed[module, name]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repo', type=Path, required=True)
parser.add_argument('--paper', type=Path, default=Path(__file__).resolve().parents[1])
args = parser.parse_args()
src = args.repo/'reports/rope_seed101_20'
p = args.paper
receipts = json.loads((src/'evidence_manifest.json').read_text())
checks = {v['archive']:v['sha256'] for v in receipts.values()}
sha = lambda b: hashlib.sha256(b).hexdigest()
rows = {(r['arm'], int(r['sample'])): r for r in csv.DictReader((src/'per_sample.csv').open())}
manifest = {'source': 'reports/rope_seed101_20', 'base_seed': 101,
    'protocol': 'Both arms predict and execute three pushes, observe after each push, stop at CD < 0.3 or 20 pushes; fixed-strength ECP starts after five warm-up pushes.',
    'selection': 'Illustrative lower-CD cases at equal final push counts with memory actually applied; not a substitute for aggregate results.',
    'aggregate_caveat': 'The main result pools the same 20 targets across three seeds; these two cases illustrate fixed-strength ECP.',
    'archives': {}, 'files': {}, 'cases': []}
archives = {}
for arm in ['native3', 'fixed3']:
    path = src/f'{arm}_evidence.tar.gz'
    assert sha(path.read_bytes()) == checks[path.name]
    manifest['archives'][path.relative_to(args.repo).as_posix()] = checks[path.name]
    archives[arm] = tarfile.open(path)

def save_frame(array, name, archive, member, step):
    assert array.shape == (224, 224, 3)
    assert np.isfinite(array).all() and array.min() >= 0 and array.max() <= 255
    pixels = np.rint(array).astype(np.uint8)
    np.testing.assert_array_equal(pixels, array)
    rel = 'figures/recorded_rollouts/'+name+'.png'
    target = p/rel
    Image.fromarray(pixels).save(target)
    manifest['files'][rel] = {'sha256': sha(target.read_bytes()),
        'archive': archive, 'member': member, 'step': step,
        'full_original_viewport': True, 'pixel_exact_to_recorded_observation': True}
    return rel

for sample, steps in [(5, [0, 6, 12, 20]), (11, [0, 6, 12, 20])]:
    native, ours = rows['native3', sample], rows['fixed3', sample]
    assert native['actual_pushes'] == ours['actual_pushes'] == str(steps[-1])
    assert int(ours['used_trigger_memories']) > 0
    assert float(ours['final_cd']) < float(native['final_cd'])
    case = {'dataset': 'Rope', 'backbone': 'DINO-WM', 'target': int(native['global_sample']),
        'sample': sample, 'steps': steps, 'rows': [], 'unit': 'Push', 'hide_border': 0,
        'first_S_application_push': int(ours['first_S_application_push']),
        'used_trigger_memories': int(ours['used_trigger_memories'])}
    targets = []
    for arm, label in [('native3', 'native'), ('fixed3', 'ours')]:
        tar = archives[arm]
        receipt = json.loads(tar.extractfile(f'{arm}/sample_{sample:02d}/complete.json').read())
        cfg = receipt['config']['planner']
        assert cfg['chunk_size'] == cfg['sub_planner']['horizon'] == 3
        assert cfg['ecp']['warmup_pushes'] == 5
        if arm == 'fixed3':
            assert cfg['strength_mode'] == 'fixed' and cfg['ecp']['kappa'] == 2
        member = f'{arm}/sample_{sample:02d}/recorded_execution.npz'
        archived = tar.extractfile(member).read()
        with np.load(io.BytesIO(archived), allow_pickle=False) as data:
            assert data['physical_actions'].shape[1] == steps[-1]
            frames = []
            for step in steps:
                rel = save_frame(data['obs_visual'][0, step], f'rope_seed101_fixed_t{case["target"]}_{label}_{step:03d}',
                    f'{arm}_evidence.tar.gz', member, step)
                frames.append({'path': rel, 'step': step})
            target_member = f'{arm}/sample_{sample:02d}/'+('target.pkl' if arm == 'native3' else 'plan_targets.pkl')
            target = ArrayOnlyUnpickler(io.BytesIO(tar.extractfile(target_member).read())).load()
            targets.append(target)
            case['rows'].append({'arm': label, 'frames': frames,
                'final_cd': float(rows[arm, sample]['final_cd']),
                'actual_pushes': int(rows[arm, sample]['actual_pushes']),
                'success': rows[arm, sample]['success'] == 'True',
                'recorded_execution_sha256': sha(archived)})
    for key in ['obs_0', 'obs_g']:
        np.testing.assert_array_equal(targets[0][key]['visual'], targets[1][key]['visual'])
    case['goal'] = save_frame(targets[0]['obs_g']['visual'][0, 0], f'rope_seed101_fixed_t{case["target"]}_goal',
        'native3_evidence.tar.gz', f'native3/sample_{sample:02d}/target.pkl', None)
    manifest['cases'].append(case)

for tar in archives.values():
    tar.close()
(p/'provenance/rope_fixed_visuals_seed101.json').write_text(json.dumps(manifest, indent=2)+'\n')
print(f'Exported {len(manifest["files"])} original observations for two equal-push Rope pairs.')
