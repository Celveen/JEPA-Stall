#!/usr/bin/env python3
"""Compile the manuscript and split its main/supplementary PDFs; no experiments."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def run(args):
    result = subprocess.run(args, cwd=ROOT, check=False, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stdout + '\n' + result.stderr)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', choices=['pdflatex', 'tectonic'],
                        help='Default: pdfLaTeX, matching a new Overleaf project.')
    parser.add_argument('--pdflatex', default=os.environ.get('PDFLATEX', 'pdflatex'))
    parser.add_argument('--tectonic', default=os.environ.get('TECTONIC'))
    parser.add_argument('--web-bundle', help='Optional Tectonic resource mirror URL.')
    parser.add_argument('--skip-compile', action='store_true')
    args = parser.parse_args()
    engine = args.engine or ('tectonic' if args.tectonic else 'pdflatex')
    if not args.skip_compile:
        if engine == 'pdflatex':
            executable = shutil.which(args.pdflatex)
            if executable is None:
                raise RuntimeError('pdfLaTeX not found; supply --pdflatex /path/to/pdflatex.')
            bibtex = str(Path(executable).parent / 'bibtex')
            command = [executable, '-no-shell-escape', '-interaction=nonstopmode',
                       '-halt-on-error', '-file-line-error', 'main.tex']
            run(command)
            run([bibtex, 'main'])
            # Resolve citations, back-references, line numbers and floats.
            for _ in range(3):
                run(command)
            print('pdfLaTeX and BibTeX compilation completed.')
        else:
            command = [args.tectonic or 'tectonic', '-X', 'compile', 'main.tex',
                       '--keep-logs', '--keep-intermediates']
            if args.web_bundle:
                command += ['--web-bundle', args.web_bundle]
            result = run(command)
            print(result.stdout)
            print(result.stderr)
    log = (ROOT / 'main.log').read_text(errors='replace')
    defects = re.findall(r'^.*(?:Overfull|undefined references|Citation .* undefined|Reference .* undefined).*$',
                         log, flags=re.MULTILINE)
    if defects:
        raise RuntimeError('Resolve layout/reference warnings before export:\n' + '\n'.join(defects))
    poppler = all(shutil.which(tool) for tool in ('pdftotext', 'pdfseparate', 'pdfunite'))
    if poppler:
        pages = run(['pdftotext', '-layout', 'main.pdf', '-']).stdout.split('\f')
        pages = [p for p in pages if p.strip()]
    else:
        from pypdf import PdfReader
        reader = PdfReader(ROOT / 'main.pdf')
        pages = [page.extract_text(extraction_mode='layout') for page in reader.pages]
    abstract = (ROOT / 'sections' / 'abstract.tex').read_text()
    abstract = re.sub(r'\\(?:begin|end)\{abstract\}', '', abstract).strip()
    abstract_words = len(abstract.split())
    if not abstract.strip():
        raise RuntimeError('Abstract is empty.')
    figure_one_page = next(i + 1 for i, p in enumerate(pages) if re.search(r'Figure\s+1\.', p))
    trajectory_figure_page = next(i + 1 for i, p in enumerate(pages) if re.search(r'Figure\s+2\.', p))
    if 'Abstract' not in pages[0] or figure_one_page != 1 or trajectory_figure_page != 2:
        raise RuntimeError('Expected abstract and headline chart on page one, trajectory Figure 2 on page two.')
    if poppler:
        bbox = ET.fromstring(run(['pdftotext', '-f', '1', '-l', '1', '-bbox', 'main.pdf', '-']).stdout)
        ns = {'pdf': 'http://www.w3.org/1999/xhtml'}
        page = bbox.find('.//pdf:page', ns)
        words = page.findall('pdf:word', ns)
        figure_caption = next(w for i, w in enumerate(words[:-1])
                              if w.text == 'Figure' and words[i + 1].text == '1.')
        caption_x, page_width = float(figure_caption.attrib['xMin']), float(page.attrib['width'])
    else:
        import pypdfium2 as pdfium
        document = pdfium.PdfDocument(ROOT / 'main.pdf')
        page = document[0]
        textpage = page.get_textpage()
        search = textpage.search('Figure 1.')
        found = search.get_next()
        if found is None:
            raise RuntimeError('First-page Figure 1 caption was not found.')
        caption_x, page_width = textpage.get_charbox(found[0])[0], page.get_width()
        search.close(); textpage.close(); page.close(); document.close()
    if caption_x <= page_width / 2:
        raise RuntimeError('Headline figure must be in the right column of page one.')
    statement_page = next(i for i, p in enumerate(pages) if 'LLM Usage Statement' in p)
    appendix_page = next(i for i, p in enumerate(pages) if 'Supplementary Material' in p)
    if statement_page > 8:
        raise RuntimeError(f'Main body is {statement_page} pages, exceeding eight.')
    if len(pages) > 22:
        raise RuntimeError(f'Combined manuscript has {len(pages)} pages, exceeding 22.')
    shutil.copyfile(ROOT / 'main.pdf', ROOT / 'paper.pdf')
    shutil.copyfile(ROOT / 'main.pdf', ROOT / 'paper_v14.pdf')
    if poppler:
        with tempfile.TemporaryDirectory(prefix='jepa-paper-pages-') as directory:
            pattern = str(Path(directory) / 'page-%d.pdf')
            run(['pdfseparate', 'main.pdf', pattern])
            numbered = [str(Path(directory) / f'page-{i+1}.pdf') for i in range(len(pages))]
            for name, selected in [('paper_main.pdf', numbered[:appendix_page]),
                                   ('paper_appendix.pdf', numbered[appendix_page:])]:
                destination = ROOT / name
                if destination.exists():
                    destination.unlink()
                run(['pdfunite', *selected, str(destination)])
    else:
        from pypdf import PdfWriter
        for name, selected in [('paper_main.pdf', reader.pages[:appendix_page]),
                               ('paper_appendix.pdf', reader.pages[appendix_page:])]:
            writer = PdfWriter()
            for page in selected:
                writer.add_page(page)
            writer.write(ROOT / name)
    report = {'engine': engine,
              'abstract_words_whitespace': abstract_words,
              'abstract_page': 1, 'figure_one_page': figure_one_page,
              'headline_on_first_page_right_column': True,
              'trajectory_figure_page': trajectory_figure_page,
              'body_pages': statement_page,
              'main_with_statement_and_references_pages': appendix_page,
              'appendix_pages': len(pages) - appendix_page,
              'combined_pages': len(pages),
              'undefined_references_or_citations': False,
              'overfull_boxes': False,
              'build_runs_experiments': False,
              'known_upstream_warning': ('lineno.sty has an invalid UTF-8 byte in a source comment; rendered output is unaffected.'
                                         if engine == 'tectonic' else None)}
    (ROOT / 'provenance' / 'build_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
