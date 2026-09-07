"""Read-only structural and export checks for the integrated manuscript.

Run from the project root after docs/ICSPS2026_LaTeX/build.ps1.
Dependencies: pypdf, pdfplumber. Page previews are rendered separately with Poppler.
"""
from pathlib import Path
import hashlib
import json
import re

import pdfplumber
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PAPER = ROOT / 'docs/ICSPS2026_LaTeX'
BUILD = ROOT / 'tmp/pdfs/paper_fig123/build'
PDF = PAPER / 'ICSPS2026_paper.pdf'
BASELINE = ROOT / 'tmp/pdfs/paper_fig123/before/ICSPS2026_paper.tex'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preserved_content(source):
    source = re.sub(r'^\s*%[^\n]*', '', source, flags=re.M)
    source = re.sub(r'\\begin\{figure\*?\}.*?\\end\{figure\*?\}',
                    lambda m: '' if any(f'fig:{n}' in m.group() for n in
                                         ('overview', 'fdr', 'representation_ams')) else m.group(),
                    source, flags=re.S)
    source = re.sub(r'\\newcommand\{\\figplaceholder\}.*?(?=\\begin\{document\})', '', source, flags=re.S)
    source = re.sub(r'\\(?:RequirePackage|usepackage)\[T1\]\{fontenc\}', '', source)
    source = re.sub(r'\\IEEEtriggeratref\{\d+\}', '', source)
    source = source.replace('\\-', '')
    return re.sub(r'\s+', ' ', source).strip()


def collect_fonts(resources, found, visited):
    if not resources:
        return
    resources = resources.get_object()
    for font in resources.get('/Font', {}).get_object().values() if '/Font' in resources else []:
        key = (getattr(font, 'idnum', None), getattr(font, 'generation', None))
        if key in visited:
            continue
        visited.add(key)
        font = font.get_object()
        descendants = font.get('/DescendantFonts')
        descriptor = (descendants[0].get_object() if descendants else font).get('/FontDescriptor')
        descriptor = descriptor.get_object() if descriptor else {}
        found.append({'name': str(font.get('/BaseFont')), 'subtype': str(font.get('/Subtype')),
                      'embedded': any(k in descriptor for k in ('/FontFile', '/FontFile2', '/FontFile3'))})
    if '/XObject' in resources:
        for obj in resources['/XObject'].get_object().values():
            obj = obj.get_object()
            if obj.get('/Subtype') == '/Form':
                collect_fonts(obj.get('/Resources'), found, visited)


def main():
    log = (BUILD / 'ICSPS2026_paper.log').read_text(encoding='utf-8', errors='replace')
    aux = (BUILD / 'ICSPS2026_paper.aux').read_text(encoding='utf-8')
    source = (PAPER / 'ICSPS2026_paper.tex').read_text(encoding='utf-8')
    names = ['fig1_overall_architecture.pdf', 'fig2_fdr_block.pdf', 'fig3_representation_ams.pdf',
             'fig4_buildings.pdf', 'fig5_real_sar.pdf']
    figures = []
    for number, name in enumerate(names, 1):
        original = ROOT / f'output/pdf/ICSPS2026_Fig{number}' / name
        installed = PAPER / 'figures' / name
        page = PdfReader(installed).pages[0]
        figures.append({'number': number, 'file': name, 'sha256': digest(installed),
                        'matches_delivered_original': digest(original) == digest(installed),
                        'width_mm': float(page.mediabox.width) * 25.4 / 72,
                        'height_mm': float(page.mediabox.height) * 25.4 / 72})
    labels = dict((label, {'number': number, 'page': int(page)}) for label, number, page in
                  re.findall(r'\\newlabel\{([^}]+)\}\{\{([^}]+)\}\{(\d+)\}', aux))
    pages, outside, visible_font_names = [], [], set()
    with pdfplumber.open(PDF) as pdf:
        for i, page in enumerate(pdf.pages, 1):
            chars = [c for c in page.chars if c.get('text', '').strip()]
            visible_font_names.update(c['fontname'] for c in chars)
            out = [c['text'] for c in chars if c['x0'] < -0.5 or c['top'] < -0.5 or
                   c['x1'] > page.width + 0.5 or c['bottom'] > page.height + 0.5]
            outside.extend({'page': i, 'text': c} for c in out)
            pages.append({'page': i, 'width_pt': page.width, 'height_pt': page.height,
                          'character_count': len(chars),
                          'bounds_pt': [round(min(c['x0'] for c in chars), 2),
                                        round(min(c['top'] for c in chars), 2),
                                        round(max(c['x1'] for c in chars), 2),
                                        round(max(c['bottom'] for c in chars), 2)]})
        last = pdf.pages[-1]
        column_bottoms = [max(c['bottom'] for c in last.chars if c['text'].strip() and lo <= c['x0'] < hi)
                          for lo, hi in [(0, last.width / 2), (last.width / 2, last.width)]]
    fonts = []
    visited = set()
    for page in PdfReader(PDF).pages:
        collect_fonts(page.get('/Resources'), fonts, visited)
    for font in fonts:
        font['used_for_visible_text'] = font['name'].lstrip('/') in visible_font_names
    report = {
        'paper_pdf_sha256': digest(PDF), 'tex_sha256': digest(PAPER / 'ICSPS2026_paper.tex'),
        'page_count': len(pages), 'figures': figures, 'labels': labels, 'pages': pages,
        'fonts': fonts, 'all_fonts_embedded': all(f['embedded'] for f in fonts),
        'all_rendered_fonts_embedded': all(f['embedded'] for f in fonts if f['used_for_visible_text']),
        'unused_unembedded_font_resources': [f for f in fonts if not f['embedded'] and not f['used_for_visible_text']],
        'text_outside_page': outside,
        'underfull_warnings': re.findall(r'Underfull \\[hv]box[^\n]*', log),
        'serious_warnings': re.findall(r'^.*(?:Overfull|undefined|multiply defined|Missing character|Font Warning|Float too large|Too many unprocessed floats|^!).*$', log, re.M),
        'placeholder_count': source.count('figplaceholder'),
        'bibliography_column_bottom_difference_pt': round(abs(column_bottoms[0] - column_bottoms[1]), 2),
        'unchanged_content_outside_fig123_and_formatting':
            preserved_content(BASELINE.read_text(encoding='utf-8')) == preserved_content(source) if BASELINE.exists() else None,
        'manual_review': 'See LAYOUT_CHECK.md for full-page visual review and warning interpretation.',
    }
    assert report['placeholder_count'] == 0
    assert not report['serious_warnings'], report['serious_warnings']
    assert not outside
    assert report['all_rendered_fonts_embedded']
    assert all(f['matches_delivered_original'] for f in figures)
    if BASELINE.exists():
        assert report['unchanged_content_outside_fig123_and_formatting']
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / 'paper_qa.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('page_count', 'all_rendered_fonts_embedded', 'serious_warnings',
                      'underfull_warnings', 'bibliography_column_bottom_difference_pt')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
