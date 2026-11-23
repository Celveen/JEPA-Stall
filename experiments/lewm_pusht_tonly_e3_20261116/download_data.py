"""Resumable, range-verified download of the user-supplied public T-only archive."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import time
import requests

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / 'eval_outputs/lewm_pusht_prepare/downloads'
SIZE = 538836291
CHUNK = 4 * 1024 * 1024
URL = 'https://drive.usercontent.google.com/download?id=1VY__1KnP1Hbh1dWZws-ELxEHLyZ4In20&export=download&confirm=t'


def download(index):
    start, end = index * CHUNK, min((index + 1) * CHUNK, SIZE) - 1
    path = ROOT / 'parts' / f'{index:04d}'
    if path.exists() and path.stat().st_size == end - start + 1:
        return index
    for attempt in range(4):
        try:
            with requests.get(URL, headers={'Range': f'bytes={start}-{end}'}, stream=True, timeout=(15, 90)) as response:
                assert response.status_code == 206, response.status_code
                assert response.headers['Content-Range'] == f'bytes {start}-{end}/{SIZE}', response.headers.get('Content-Range')
                with path.with_suffix('.part').open('wb') as stream:
                    for chunk in response.iter_content(256 * 1024):
                        stream.write(chunk)
            assert path.with_suffix('.part').stat().st_size == end - start + 1
            path.with_suffix('.part').replace(path)
            return index
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))


def main():
    (ROOT / 'parts').mkdir(parents=True, exist_ok=True)
    count = (SIZE + CHUNK - 1) // CHUNK
    with ThreadPoolExecutor(max_workers=8) as pool:
        jobs = [pool.submit(download, i) for i in range(count)]
        for finished, job in enumerate(as_completed(jobs), 1):
            job.result()
            if finished % 8 == 0 or finished == count:
                print(f'{finished}/{count} verified parts', flush=True)
    output = ROOT / 'pushobj_T16k_bias_data.zip'
    assert not output.exists(), 'Preserve existing archive'
    with output.with_suffix('.assembling').open('wb') as target:
        for i in range(count):
            with (ROOT / 'parts' / f'{i:04d}').open('rb') as source:
                import shutil
                shutil.copyfileobj(source, target)
    assert output.with_suffix('.assembling').stat().st_size == SIZE
    output.with_suffix('.assembling').replace(output)
    print('Complete:', output, flush=True)


if __name__ == '__main__':
    main()
