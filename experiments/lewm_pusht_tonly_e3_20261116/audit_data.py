"""Check the supplied T-only archive, extract data safely, and validate training inputs."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import pickle
import zipfile

import numpy as np
import torch
from decord import VideoReader

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]


class DataUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in {('numpy.core.multiarray', 'scalar'), ('numpy', 'dtype')}:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f'Unexpected data type: {module}.{name}')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def audit(data, archive):
    expected = json.loads((REPO / 'reports/pushobj_T16k_bias/dataset_audit.json').read_text())
    result = {'archive_sha256': digest(archive), 'archive_bytes': archive.stat().st_size,
              'drive_file_id': '1VY__1KnP1Hbh1dWZws-ELxEHLyZ4In20', 'data_path': str(data), 'splits': {}}
    assert archive.stat().st_size == 538836291
    for split in ('train', 'val'):
        folder = data / split
        with (folder / 'seq_lengths.pkl').open('rb') as stream:
            lengths = np.asarray(DataUnpickler(stream).load(), dtype=int)
        with (folder / 'shapes.pkl').open('rb') as stream:
            shapes = DataUnpickler(stream).load()
        reference = expected[split]
        assert len(lengths) == len(shapes) == reference['episodes']
        assert set(shapes) == {'T'}
        assert int(lengths.sum()) == reference['frames']
        assert int((lengths - 19).sum()) == reference['slices']
        tensors = {}
        for name, width in [('states.pth', 5), ('rel_actions.pth', 2), ('velocities.pth', 2)]:
            tensor = torch.load(folder / name, map_location='cpu', weights_only=True)
            assert tensor.shape == (len(lengths), max(lengths), width), (name, tensor.shape)
            assert torch.isfinite(tensor).all(), name
            tensors[name] = {'shape': list(tensor.shape), 'sha256': digest(folder / name)}
        videos = sorted((folder / 'obses').glob('episode_*.mp4'))
        assert len(videos) == len(lengths)
        for i in range(len(lengths)):
            assert (folder / 'obses' / f'episode_{i:03d}.mp4').stat().st_size > 0
        # ZIP CRC verifies every video; decode a spread of episodes including both endpoints.
        decoded = []
        for i in np.unique(np.linspace(0, len(lengths) - 1, 32, dtype=int)):
            video = VideoReader(str(folder / 'obses' / f'episode_{i:03d}.mp4'), num_threads=1)
            assert len(video) == lengths[i]
            frames = video.get_batch([0, int(lengths[i]) - 1]).asnumpy()
            assert frames.shape == (2, 224, 224, 3)
            decoded.append(int(i))
        result['splits'][split] = {'episodes': len(lengths), 'frames': int(lengths.sum()),
            'shapes': dict(Counter(shapes)), 'windows': int((lengths - 19).sum()),
            'all_video_files_present': True, 'decoded_episode_indices': decoded, 'tensors': tensors}
    result['passed'] = True
    (ROOT / 'data_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=REPO / 'data/pushobj_T16k_bias')
    args = parser.parse_args()
    if not args.data.exists():
        with zipfile.ZipFile(args.archive) as package:
            names = package.namelist()
            prefixes = [n[:-len('train/seq_lengths.pkl')] for n in names if n.endswith('train/seq_lengths.pkl')]
            assert len(prefixes) == 1, prefixes
            prefix = prefixes[0]
            assert package.testzip() is None, 'Archive CRC failure'
            args.data.mkdir(parents=True)
            for entry in package.infolist():
                assert entry.filename.startswith(prefix), entry.filename
                relative = Path(entry.filename[len(prefix):])
                assert not relative.is_absolute() and '..' not in relative.parts
                assert ((entry.external_attr >> 16) & 0o170000) != 0o120000, 'Archive symlink'
                dest = args.data / relative
                if entry.is_dir():
                    dest.mkdir(parents=True, exist_ok=True)
                else:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with package.open(entry) as source, dest.open('xb') as target:
                        import shutil
                        shutil.copyfileobj(source, target)
    audit(args.data, args.archive)


if __name__ == '__main__':
    main()
