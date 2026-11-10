#!/usr/bin/env python3
"""Restyle the original method diagram without moving its panels or labels."""

import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
import subprocess

import pypdfium2 as pdfium
from pypdf import PdfWriter
from pypdf.generic import FloatObject


ORIGINAL = '2de77b48daa6e3d5de30a180222a4d3406da6897'
PALETTE = {
    '#25495e': '#293640',
    '#117e84': '#4f6657',
    '#ba633e': '#88664e',
    '#627784': '#566d7c',
    '#ebf2f6': '#eef2f4',
    '#edf7f5': '#eef2ee',
    '#bccdd6': '#c8ced0',
    '#f5f7f9': '#f6f5f2',
    '#d8e1e6': '#d7d5cf',
    '#5384b9': '#748b9b',
    '#90a8ba': '#a2b2ba',
    '#adbdc7': '#aeb9be',
    '#d5e9e5': '#dae4db',
}
ORIGINAL_ENCODER = [
    ([107, 149], b'm'), ([150, 143], b'l'), ([150, 118], b'l'), ([107, 112], b'l'), ([], b'h'),
]
# The rounded rectangle fits inside the original trapezoid's bounding box.
ENCODER = [
    ([112, 148], b'm'),
    ([145, 148], b'l'),
    ([147.76, 148, 150, 145.76, 150, 143], b'c'),
    ([150, 118], b'l'),
    ([150, 115.24, 147.76, 113, 145, 113], b'c'),
    ([112, 113], b'l'),
    ([109.24, 113, 107, 115.24, 107, 118], b'c'),
    ([107, 143], b'l'),
    ([107, 145.76, 109.24, 148, 112, 148], b'c'),
    ([], b'h'),
]


def rgb(color):
    return tuple(bytes.fromhex(color[1:]))


def restyle_pdf(original, target):
    palette = {rgb(old): rgb(new) for old, new in PALETTE.items()}
    writer = PdfWriter(clone_from=BytesIO(original))
    page = writer.pages[0]
    contents = page.get_contents()
    operations = contents.operations
    size = len(ORIGINAL_ENCODER)
    matches = [i for i in range(len(operations)) if operations[i:i+size] == ORIGINAL_ENCODER]
    if len(matches) != 1:
        raise ValueError('Expected one encoder outline in the original PDF')
    start = matches[0]
    operations[start:start+size] = [([FloatObject(v) for v in xy], op) for xy, op in ENCODER]
    for i, (values, op) in enumerate(operations):
        if op in (b'rg', b'RG'):
            old = tuple(round(float(v) * 255) for v in values)
            if old in palette:
                operations[i] = ([FloatObject(v / 255) for v in palette[old]], op)
    contents.operations = operations
    page.replace_contents(contents)
    page.compress_content_streams(level=9)
    writer.write(target)


def restyle_svg(original, target):
    commands = {b'm': 'M', b'l': 'L', b'c': 'C', b'h': 'z'}
    path = '\n'.join(
        ' '.join([commands[op], *(f'{v if i % 2 == 0 else 234 - v:g}' for i, v in enumerate(xy))])
        for xy, op in ENCODER
    )
    old_encoder = r'M 107 85\s+L 150 91\s+L 150 116\s+L 107 122\s+z'
    svg, count = re.subn(old_encoder, lambda _: path, original.decode())
    if count != 1:
        raise ValueError('Expected one encoder outline in the original SVG')
    svg = re.sub(r'#[0-9a-fA-F]{6}\b', lambda m: PALETTE.get(m[0].lower(), m[0]), svg)
    target.write_text(svg)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    paper = args.paper.resolve()
    repo = Path(__file__).resolve().parents[3]

    def original(path):
        return subprocess.check_output(['git', '-C', str(repo), 'show', f'{ORIGINAL}:paper/cvpr2027/{path}'])

    pdf = paper / 'figures/fig2_method.pdf'
    svg = paper / 'figures/svg/fig2_method.svg'
    png = paper / 'figures/previews/fig2_method.png'
    original_pdf = original('figures/fig2_method.pdf')
    original_svg = original('figures/svg/fig2_method.svg')
    restyle_pdf(original_pdf, pdf)
    restyle_svg(original_svg, svg)
    document = pdfium.PdfDocument(pdf)
    document[0].render(scale=4).to_pil().save(png)
    document.close()
    frames = json.loads(original('provenance/method_v10_manifest.json'))['native_frame_hashes']
    manifest = {
        'canvas_pt': [468, 234],
        'original_commit': ORIGINAL,
        'original_pdf_sha256': hashlib.sha256(original_pdf).hexdigest(),
        'original_svg_sha256': hashlib.sha256(original_svg).hexdigest(),
        'encoder_style': 'rounded rectangle',
        'palette': PALETTE,
        'native_frame_hashes': frames,
        'outputs': {p.relative_to(paper).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (pdf, svg, png)},
        'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (paper / 'provenance/method_figure.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
