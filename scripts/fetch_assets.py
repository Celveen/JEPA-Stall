"""Download the external assets the Wall experiments need, in parallel byte ranges, resumably, with a sha256 check.

OSF throttles each connection (about 0.1-0.8 MB/s from our server), so a file is split into byte ranges fetched over
several connections at once. Progress is kept in <out>.parts.json; rerunning the command resumes. A file that an
earlier sequential download (curl -C -) left partly written is resumed from its current length.

    dinowm_wall    DINO-WM's Wall dataset, wall_single.zip (OSF)          -> data/_dl_dinowm/
    dinowm_ckpt    DINO-WM's released world models, outputs.zip (OSF)     -> data/_dl_dinowm_ckpt/
    dinov2_vits14  DINOv2 ViT-S/14 weights (torch.hub cache)              -> ~/.cache/torch/hub/checkpoints/

    python scripts/fetch_assets.py dinowm_wall dinowm_ckpt dinov2_vits14 --conn 8
"""
import argparse
import hashlib
import json
import os
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OSF = "https://osf.io/download/{}/?view_only=a56a296ce3b24cceaf408383a175ce28"  # DINO-WM's public OSF project bmw48
ASSETS = {
    "dinowm_wall": {"url": OSF.format("678ac8fff57516fd9edd6046"), "out": "data/_dl_dinowm/wall_single.zip",
                    "size": 1668205895, "sha256": "2b4ae4ed0ad03b337efac637f17752e7e7e27f864fec39dc25b51fef490c980d"},
    "dinowm_ckpt": {"url": OSF.format("67e17e214a42e7c96144a340"), "out": "data/_dl_dinowm_ckpt/outputs.zip",
                    "size": 953204628, "sha256": "425b0d8c2a4194d3ec996b553575f69409aa5714176b063d8401789db9179482"},
    "dinov2_vits14": {"url": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth",
                      "out": "~/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth", "size": 88283115, "sha256": None},
}
BLOCK = 1 << 20


def out_path(spec):
    p = Path(os.path.expanduser(spec["out"]))
    return p if p.is_absolute() else ROOT / p


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(16 * BLOCK), b""):
            h.update(b)
    return h.hexdigest()


def fetch(name, conn, pieces):
    spec = ASSETS[name]
    out, size = out_path(spec), spec["size"]
    out.parent.mkdir(parents=True, exist_ok=True)
    state = Path(str(out) + ".parts.json")
    if state.exists():
        parts = json.loads(state.read_text())
    else:  # fresh, or a sequential partial download: its first `have` bytes are done
        have = min(out.stat().st_size, size) if out.exists() else 0
        step = max(-(-(size - have) // pieces), BLOCK)
        parts = [[s, min(s + step, size), s] for s in range(have, size, step)]  # [start, end, next byte to fetch]
        state.write_text(json.dumps(parts))  # before the file is extended, so an interrupted run resumes correctly
    if not out.exists() or out.stat().st_size != size:
        with out.open("ab") as f:
            f.truncate(size)
    lock = threading.Lock()
    todo = [i for i, (s, e, nxt) in enumerate(parts) if nxt < e]

    def save():
        state.write_text(json.dumps(parts))

    def worker():
        while True:
            with lock:
                if not todo:
                    return
                i = todo.pop(0)
            for attempt in range(1000):
                s, e, nxt = parts[i]
                if nxt >= e:
                    break
                try:
                    req = urllib.request.Request(spec["url"], headers={"Range": f"bytes={nxt}-{e - 1}"})
                    with urllib.request.urlopen(req, timeout=60) as r, out.open("r+b") as f:
                        if r.status != 206:
                            raise RuntimeError(f"server ignored the byte range (HTTP {r.status})")
                        f.seek(nxt)
                        while nxt < e:
                            b = r.read(min(BLOCK, e - nxt))
                            if not b:
                                break
                            f.write(b)
                            nxt += len(b)
                            with lock:
                                parts[i][2] = nxt
                                save()
                except Exception as ex:  # dropped connection, timeout, expired redirect: retry from parts[i][2]
                    time.sleep(min(30, 2 + attempt))
                    if attempt % 10 == 9:
                        print(f"{name} piece {i}: retry {attempt + 1} ({type(ex).__name__}: {ex})", flush=True)

    t0, done0 = time.time(), sum(nxt - s for s, e, nxt in parts)
    threads = [threading.Thread(target=worker, daemon=True) for _ in range(conn)]
    for t in threads:
        t.start()
    while any(t.is_alive() for t in threads):
        time.sleep(30)
        done = sum(nxt - s for s, e, nxt in parts) + (parts[0][0] if parts else size)
        rate = (done - done0 - (parts[0][0] if parts else 0)) / max(time.time() - t0, 1) / 1e6
        print(f"{name}: {done / 1e9:.2f} / {size / 1e9:.2f} GB, {rate:.2f} MB/s", flush=True)
    if any(nxt < e for s, e, nxt in parts):
        raise SystemExit(f"{name}: incomplete, rerun to resume")
    if spec["sha256"]:
        got = sha256(out)
        if got != spec["sha256"]:
            raise SystemExit(f"{name}: sha256 mismatch ({got}); delete {out} and {state} and rerun")
    state.unlink(missing_ok=True)
    print(f"{name}: ok {out} ({size} bytes{', sha256 verified' if spec['sha256'] else ''})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+", choices=sorted(ASSETS))
    ap.add_argument("--conn", type=int, default=8, help="parallel connections per file")
    ap.add_argument("--pieces", type=int, default=64, help="byte ranges per file (load balancing)")
    a = ap.parse_args()
    for n in a.names:
        fetch(n, a.conn, a.pieces)


if __name__ == "__main__":
    main()
