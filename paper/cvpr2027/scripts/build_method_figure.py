#!/usr/bin/env python3
"""Restore the advisor-reviewed Figure 3 from the archived Overleaf assets."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    paper = args.paper.resolve()
    repo = paper.parents[1]
    sources = paper / 'figures/source'
    pdf = paper / 'figures/fig2_method.pdf'
    drawio = repo / 'delivery/Figure3.drawio'
    drawio.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(sources / 'fig2_method_overleaf_20261115.pdf', pdf)
    shutil.copyfile(sources / 'Figure3_overleaf_20261115.drawio', drawio)
    import pypdfium2 as pdfium
    document = pdfium.PdfDocument(pdf)
    preview = paper / 'figures/previews/fig2_method.png'
    document[0].render(scale=4).to_pil().save(preview)
    document.close()
    manifest_path = paper / 'provenance/method_figure.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['outputs'] = {str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (pdf, preview, drawio)}
    manifest['generator_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print('Restored advisor Figure 3 and editable draw.io source.')


if __name__ == '__main__':
    main()
