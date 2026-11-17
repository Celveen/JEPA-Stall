#!/usr/bin/env python3
"""Export the author's editable Figure 3 without redrawing its content.

Requires draw.io Desktop and PyMuPDF. Uses its local CLI, never a remote exporter.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
import pymupdf as fitz

PAPER = Path(__file__).resolve().parents[1]
REPO = PAPER.parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=PAPER/'figures/source/Figure3_user.drawio.svg')
    parser.add_argument('--drawio', default=os.environ.get('DRAWIO'))
    args = parser.parse_args()
    executable = args.drawio or shutil.which('drawio') or '/Applications/draw.io.app/Contents/MacOS/draw.io'
    if not Path(executable).is_file():
        raise FileNotFoundError('Supply --drawio /path/to/draw.io Desktop executable.')
    source = args.source.resolve()
    root = ET.fromstring(source.read_text())
    editable = root.attrib['content']
    assert ET.fromstring(editable).tag == 'mxfile', 'Expected embedded editable draw.io diagram.'
    target = PAPER/'figures/fig2_method.pdf'
    with tempfile.TemporaryDirectory(prefix='ecp-figure3-') as temporary:
        exported = Path(temporary)/'figure3.pdf'
        subprocess.run([str(executable), '--export', '--format', 'pdf', '--crop', '--theme', 'light',
                        '--disable-update', '--output', str(exported), str(source)], check=True)
        with fitz.open(exported) as document:
            assert len(document) == 1
            page = document[0]
            text = page.get_text()
            for label in ['OFFLINE TRAINING', 'Avoid revisits', 'Activate correction', 'Rope']:
                assert label in text, ('Missing extractable label', label)
            for xref in range(1, document.xref_length()):
                obj = document.xref_object(xref)
                if '/Type /Font\n' not in obj or '/Subtype /Type0' in obj:
                    continue
                descriptor = document.xref_get_key(xref, 'FontDescriptor')
                assert descriptor[0] == 'xref', 'Unembedded font'
                dx = int(descriptor[1].split()[0])
                assert any(document.xref_get_key(dx, key)[0] == 'xref'
                           for key in ['FontFile', 'FontFile2', 'FontFile3'])
                assert document.xref_get_key(xref, 'Subtype')[1] != '/Type3'
            canvas = list(page.rect)
            page.get_pixmap(matrix=fitz.Matrix(2, 2)).save(PAPER/'figures/previews/fig2_method.png')
        shutil.copyfile(exported, target)
    svg = PAPER/'figures/svg/fig2_method.svg'
    shutil.copyfile(source, svg)
    (PAPER/'Figure3.drawio').write_text(editable)
    delivery = REPO/'delivery'
    delivery.mkdir(exist_ok=True)
    (delivery/'Figure3.drawio').write_text(editable)
    shutil.copyfile(source, delivery/'Figure3.drawio.svg')
    outputs = [target, svg, PAPER/'Figure3.drawio', PAPER/'figures/previews/fig2_method.png',
               delivery/'Figure3.drawio', delivery/'Figure3.drawio.svg']
    report = {'source_type': 'author_supplied_editable_svg', 'source': str(source.relative_to(PAPER)),
              'source_sha256': sha(source), 'canvas_pt': canvas,
              'content_redrawn': False, 'all_annotations_extractable': True,
              'all_fonts_embedded': True, 'outputs': {str(p.relative_to(REPO)): sha(p) for p in outputs}}
    (PAPER/'provenance/method_figure_import.json').write_text(json.dumps(report, indent=2)+'\n')
    print('Exported author Figure 3 with editable source, embedded fonts and selectable text.')


if __name__ == '__main__':
    main()
