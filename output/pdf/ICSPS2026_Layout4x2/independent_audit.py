"""Read-only source and PDF audit for the finalized comparison layout."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[3]
QA = ROOT / 'tmp/pdfs/layout_4x2'
sys.path.insert(0, str(QA / 'runtime'))
import fitz
from PIL import Image
from pypdf import PdfReader

BEFORE = (QA / 'before/ICSPS2026_paper.tex').read_text(encoding='utf-8')
AFTER = (ROOT / 'docs/ICSPS2026_LaTeX/ICSPS2026_paper.tex').read_text(encoding='utf-8')
AUX = (QA / 'after/ICSPS2026_paper.aux').read_text(encoding='utf-8')

def clean(s):
    s = re.sub(r'(?<!\\)%[^\n]*', '', s)
    return re.sub(r'\s+', ' ', s).strip()

def envs(s, typ):
    return re.findall(r'\\begin\{(' + typ + r'\*?)\}.*?\\end\{\1\}', s, re.S)

def blocks(s, typ):
    return [m.group(0) for m in re.finditer(r'\\begin\{(' + typ + r'\*?)\}.*?\\end\{\1\}', s, re.S)]

def nofloats(s):
    s = s.split(r'\begin{document}', 1)[1]
    return clean(re.sub(r'\\begin\{(figure\*?|table\*?)\}.*?\\end\{\1\}', '', s, flags=re.S))

def arg(s, command):
    start = s.index('\\' + command + '{') + len(command) + 2
    level = 1
    for i in range(start, len(s)):
        if s[i] == '{' and (i == 0 or s[i-1] != '\\'): level += 1
        if s[i] == '}' and (i == 0 or s[i-1] != '\\'): level -= 1
        if level == 0: return s[start:i]
    raise ValueError(command)

def by_label(s, typ):
    return {arg(b, 'label'): b for b in blocks(s, typ)}

def commands(s, names):
    return re.findall(r'\\(' + names + r')(?:\[[^\]]*\])?\{([^}]+)\}', s)

checks = {}
checks['body_after_removing_floats_and_comments_identical'] = nofloats(BEFORE) == nofloats(AFTER)
checks['section_subsection_sequence_identical'] = commands(BEFORE, 'section|subsection') == commands(AFTER, 'section|subsection')
checks['all_ref_eqref_cite_relationships_identical'] = commands(BEFORE, 'ref|eqref|cite') == commands(AFTER, 'ref|eqref|cite')
checks['all_labels_preserved_once'] = Counter(commands(BEFORE, 'label')) == Counter(commands(AFTER, 'label'))
checks['all_equation_source_and_order_identical'] = [clean(b) for b in blocks(BEFORE, 'equation')] == [clean(b) for b in blocks(AFTER, 'equation')]
checks['entire_bibliography_source_identical'] = clean(blocks(BEFORE, 'thebibliography')[0]) == clean(blocks(AFTER, 'thebibliography')[0])
checks['packages_identical'] = commands(BEFORE, 'usepackage|RequirePackage') == commands(AFTER, 'usepackage|RequirePackage')
checks['documentclass_identical'] = commands(BEFORE, 'documentclass') == commands(AFTER, 'documentclass')
before_preamble = BEFORE.split(r'\begin{document}', 1)[0]
after_preamble = AFTER.split(r'\begin{document}', 1)[0]
after_preamble_without_panel_macro = after_preamble.split('% Original comparison images, placed without resampling; labels remain 8 pt.', 1)[0]
checks['preamble_unchanged_except_local_panel_macro'] = clean(before_preamble) == clean(after_preamble_without_panel_macro)
figure_old = by_label(BEFORE, 'figure')
figure_new = by_label(AFTER, 'figure')
checks['fig1_entire_definition_unchanged_and_double_column'] = clean(figure_old['fig:overview']) == clean(figure_new['fig:overview']) and r'\begin{figure*}[!t]' in figure_new['fig:overview']
checks['fig2_fig3_captions_unchanged'] = all(clean(arg(figure_old[k], 'caption')) == clean(arg(figure_new[k], 'caption')) for k in ['fig:fdr', 'fig:representation_ams'])

tables = {}
for lab, old in by_label(BEFORE, 'table').items():
    new = by_label(AFTER, 'table')[lab]
    grid_old = old.split(r'\toprule', 1)[1].split(r'\bottomrule', 1)[0]
    grid_new = new.split(r'\toprule', 1)[1].split(r'\bottomrule', 1)[0]
    note_old = re.search(r'\\scriptsize (.*)\}\s*\\end\{table\*?\}', old, re.S)
    note_new = re.search(r'\\scriptsize (.*)\}\s*\\end\{table\*?\}', new, re.S)
    tables[lab] = {
        'all_table_cells_including_formatting_identical': clean(grid_old) == clean(grid_new),
        'caption_identical': clean(arg(old, 'caption')) == clean(arg(new, 'caption')),
        'note_identical': clean(note_old.group(1) if note_old else '') == clean(note_new.group(1) if note_new else ''),
        'font_commands_identical': re.findall(r'\\(?:footnotesize|scriptsize|small|fontsize)', old) == re.findall(r'\\(?:footnotesize|scriptsize|small|fontsize)', new),
    }
checks['all_table_content_and_fonts_preserved'] = all(all(r.values()) for r in tables.values())

equation_labels = [arg(b, 'label') for b in blocks(BEFORE, 'equation')]
aux_labels = {k: (n, int(p)) for k, n, p in re.findall(r'\\newlabel\{([^}]+)\}\{\{([^}]+)\}\{(\d+)\}', AUX)}
eq_numbers = {lab: aux_labels[lab][0] for lab in equation_labels}
checks['equations_still_numbered_1_to_8'] = list(eq_numbers.values()) == [str(i) for i in range(1, len(equation_labels)+1)]
bibkeys = [k for cmd, k in commands(BEFORE, 'bibitem')]
aux_bibs = dict(re.findall(r'\\bibcite\{([^}]+)\}\{([^}]+)\}', AUX))
checks['citation_numbers_preserved'] = aux_bibs == {k: str(i+1) for i,k in enumerate(bibkeys)}
expected_figures = ['fig:overview', 'fig:fdr', 'fig:representation_ams', 'fig:synthetic_qual', 'fig:real_qual']
checks['figure_numbers_1_to_5'] = [aux_labels[k][0] for k in expected_figures] == [str(i) for i in range(1,6)]

readers = {side: PdfReader(QA / side / 'ICSPS2026_paper.pdf') for side in ['before', 'after']}
docs = {side: fitz.open(QA / side / 'ICSPS2026_paper.pdf') for side in ['before', 'after']}
checks['page_geometry_unchanged'] = all(tuple(p.mediabox) == tuple(readers['before'].pages[0].mediabox) for p in readers['after'].pages)
text_after = [p.get_text() for p in docs['after']]
references_page = next(i+1 for i,t in enumerate(text_after) if re.search(r'R\s*EFERENCES|REFERENCES', t))

panels = []
for n in [4,5]:
    originals = ROOT / f'output/pdf/ICSPS2026_Fig{n}'
    manifest = json.loads((originals/'figure_manifest.json').read_text(encoding='utf-8'))
    final_page = aux_labels[expected_figures[n-1]][1]
    embedded = [im.image for im in readers['after'].pages[final_page-1].images]
    rects = []
    for entry in manifest['panels']:
        src = originals/'source_images'/entry['file']
        dst = ROOT/'docs/ICSPS2026_LaTeX/figures'/f'fig{n}'/entry['file']
        with Image.open(src) as im:
            matches = [j for j,other in enumerate(embedded) if other.mode == im.mode and other.size == im.size and other.tobytes() == im.tobytes()]
            record = {
                'figure': n, 'label': entry['label'], 'filename': entry['file'],
                'size': list(im.size), 'mode': im.mode,
                'copied_file_byte_identical': src.read_bytes() == dst.read_bytes(),
                'sha256': hashlib.sha256(src.read_bytes()).hexdigest(),
                'unique_native_pixel_match_in_final_pdf': len(matches) == 1,
                'final_page': final_page,
            }
            page = docs['after'][final_page-1]
            for info in page.get_images(full=True):
                pix = fitz.Pixmap(docs['after'], info[0])
                if pix.width == im.width and pix.height == im.height and pix.samples == im.tobytes():
                    found = page.get_image_rects(info[0])
                    record['rects_pdf_pt'] = [list(r) for r in found]
                    rects.append(list(found[0]))
            if n == 5 and entry['file'] == 'noisy.png':
                red = [(x,y) for y in range(im.height) for x in range(im.width) if (lambda p:p[0]>200 and p[1]<100 and p[2]<100)(im.getpixel((x,y)))]
                record['red_pixel_count'] = len(red)
                record['red_box_bounds'] = [min(x for x,y in red),min(y for x,y in red),max(x for x,y in red),max(y for x,y in red)]
            panels.append(record)
    # All 8 bounding boxes must form 4 rows and 2 aligned columns in supplied order.
    checks[f'fig{n}_4_rows_2_columns_ordered'] = len(rects) == 8 and all(abs(rects[j][1]-rects[j+1][1])<0.01 and rects[j][0]<rects[j+1][0] for j in [0,2,4,6]) and all(rects[j][1]<rects[j+2][1] for j in range(6)) and len({round(r[0],2) for r in rects}) == 2
    checks[f'fig{n}_within_right_column'] = len(rects) == 8 and all(r[0]>=310 and r[2]<=563.6 for r in rects)
checks['all_16_source_png_bytes_and_embedded_pixels_preserved'] = len(panels)==16 and all(r['copied_file_byte_identical'] and r['unique_native_pixel_match_in_final_pdf'] for r in panels)
checks['fig5_before_references'] = aux_labels['fig:real_qual'][1] < references_page

report = {
    'checks': checks,
    'all_checks_pass': all(checks.values()),
    'before_pages': len(readers['before'].pages), 'after_pages': len(readers['after'].pages),
    'figure_number_page_mapping': {k: aux_labels[k] for k in expected_figures},
    'equation_numbers': eq_numbers,
    'reference_count': len(bibkeys), 'references_start_page': references_page,
    'table_checks': tables,
    'panels': panels,
    'visual_qa': {
        'viewed_renderings': ['after/page-4.png','after/page-5.png','after/page-6.png','after/page-7.png'],
        'finding': 'Four-by-two figures clear and aligned; all labels, ratios and red ROI retained; tables within their columns; Figure 3 edges complete; no large abnormal blank areas; no final spacing adjustment advised.',
    },
}
(QA/'independent_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k not in ['panels']}, ensure_ascii=True, indent=2))
if not report['all_checks_pass']:
    raise SystemExit(1)
