"""Export the existing Fig. 1 drawing commands as editable Excalidraw elements."""
import ast
import hashlib
import json
import re
import time
from pathlib import Path

from PIL import ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUTPUT = ROOT / 'output' / 'excalidraw' / 'Fig1_SAR_architecture.excalidraw'
SCALE = 4
PAD = 48
NOW = int(time.time() * 1000)
elements = []
rects = []


def identifier(prefix):
    return f'fig1-{prefix}-{len(elements):03d}'


def base(kind, x, y, w, h, color, fill='transparent', layer=3):
    eid = identifier(kind)
    element = {
        'id': eid, 'type': kind, 'x': PAD + x * SCALE, 'y': PAD + y * SCALE,
        'width': w * SCALE, 'height': h * SCALE, 'angle': 0,
        'strokeColor': color, 'backgroundColor': fill, 'fillStyle': 'solid',
        'strokeWidth': 2.8, 'strokeStyle': 'solid', 'roughness': 0, 'opacity': 100,
        'groupIds': [], 'frameId': None, 'roundness': None,
        'seed': int(hashlib.sha256(eid.encode()).hexdigest()[:7], 16),
        'version': 1, 'versionNonce': 1, 'isDeleted': False,
        'boundElements': [], 'updated': NOW, 'created': NOW, 'link': None,
        'locked': False, 'customData': {'source': 'GRSL Figure 1'},
        '_layer': layer,
    }
    elements.append(element)
    return element


def plain_math(text):
    replacements = {
        r'\widehat{Z}': 'Z\u0302', r'\widehat{X}': 'X\u0302',
        r'\widetilde{X}': 'X\u0303', r'T_\alpha^{-1}': 'Tα⁻¹',
        r'T_\alpha': 'Tα', r'\alpha': 'α', r'\times': ' × ',
        r'\!': '', r'\ ': ' ', '^2': '²',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r'_\{([0-9]+)\}', lambda m: m[1].translate(str.maketrans('0123456789', '₀₁₂₃₄₅₆₇₈₉')), text)
    text = re.sub(r'_([0-9]+)', lambda m: m[1].translate(str.maketrans('0123456789', '₀₁₂₃₄₅₆₇₈₉')), text)
    text = text.replace('$', '')
    assert '\\' not in text, text
    return text


def containing(x, y, w=0, h=0):
    matches = [r for r in rects if r['_box'][0] <= x and r['_box'][1] <= y
               and x + w <= r['_box'][0] + r['_box'][2]
               and y + h <= r['_box'][1] + r['_box'][3]]
    return min(matches, key=lambda r: r['width'] * r['height']) if matches else None


def label(x, y, s, size=8.5, bold=False, ha='center', color='#22313B', bounds=None):
    text = plain_math(s)
    font_size = size * SCALE
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', round(font_size))
    lines = text.split('\n')
    width = max(font.getlength(line) for line in lines)
    height = font_size * 1.15 * len(lines)
    left = x * SCALE - (width / 2 if ha == 'center' else width if ha == 'right' else 0)
    e = base('text', left / SCALE, y - height / SCALE / 2,
             width / SCALE, height / SCALE, color, layer=8)
    e.update({'text': text, 'originalText': text, 'fontSize': font_size,
              'fontFamily': 2, 'textAlign': ha if ha in ('left', 'right') else 'center',
              'verticalAlign': 'middle', 'containerId': None, 'autoResize': True,
              'lineHeight': 1.15, 'baseline': font_size * 0.92})
    e['customData']['emphasis'] = bold
    parent = containing(x, y)
    if parent:
        e['groupIds'] = parent['groupIds'][:]
    return e


def card(x, y, w, h, s='', style=('#FFFFFF', '#81919C'), size=8.5, bold=False, radius=3):
    parent = containing(x, y, w, h)
    e = base('rectangle', x, y, w, h, style[1], style[0])
    e['_box'] = (x, y, w, h)
    e['groupIds'] = [e['id'] + '-group'] + (parent['groupIds'] if parent else [])
    e['roundness'] = {'type': 3, 'value': radius * SCALE}
    rects.append(e)
    if s:
        label(x + w / 2, y + h / 2, s, size, bold)
    return e


def arrow(points, color='#22313B', dashed=False, head=True, lw=.85):
    x, y = points[0]
    xs, ys = zip(*points)
    e = base('arrow' if head else 'line', x, y, max(xs) - min(xs), max(ys) - min(ys), color, layer=2)
    e.update({'points': [[(px - x) * SCALE, (py - y) * SCALE] for px, py in points],
              'lastCommittedPoint': None, 'startBinding': None, 'endBinding': None,
              'startArrowhead': None, 'endArrowhead': 'triangle' if head else None,
              'elbowed': False, 'strokeWidth': lw * SCALE,
              'strokeStyle': 'dashed' if dashed else 'solid'})
    e['_endpoints'] = [points[0], points[-1]]
    return e


def point(x, y, color):
    base('ellipse', x - .8, y - .8, 1.6, 1.6, color, color, layer=6)


def panel(x, y, letter, title):
    label(x, y, letter, 10, True, ha='left')
    label(x + 13, y, title, 10, True, ha='left')


class Ax:
    def plot(self, xs, ys, **kwargs):
        e = arrow(list(zip(xs, ys)), kwargs['color'], head=False, lw=kwargs.get('lw', .65))
        e['_layer'] = 1


def binding(px, py):
    candidates = []
    for r in rects:
        x, y, w, h = r['_box']
        if x - .01 <= px <= x + w + .01 and y - .01 <= py <= y + h + .01:
            edge = min(abs(px - x), abs(px - x - w), abs(py - y), abs(py - y - h))
            if edge <= .01:
                candidates.append(r)
    if not candidates:
        return None
    r = min(candidates, key=lambda e: e['width'] * e['height'])
    x, y, w, h = r['_box']
    if abs(px - x) < .01 or abs(px - x - w) < .01:
        focus = 2 * (py - y) / h - 1
    else:
        focus = 2 * (px - x) / w - 1
    return {'elementId': r['id'], 'focus': focus, 'gap': 0,
            'fixedPoint': [(px - x) / w, (py - y) / h]}


def main():
    tree = ast.parse((HERE / 'draw_fig1.py').read_text(encoding='utf-8'))
    env = globals() | {'ax': Ax()}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            exec(compile(ast.Module(body=[node], type_ignores=[]), '<constants>', 'exec'), env)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'make_figure')
    start = next(i for i, n in enumerate(function.body) if isinstance(n, ast.Expr)
                 and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name)
                 and n.value.func.id == 'panel')
    stop = next(i for i, n in enumerate(function.body) if isinstance(n, ast.Expr)
                and ast.unparse(n).startswith('fig.canvas.draw()'))
    exec(compile(ast.Module(body=function.body[start:stop], type_ignores=[]), '<fig1-drawing>', 'exec'), env)
    for e in elements:
        if e['type'] == 'arrow':
            for key, (x, y) in zip(('startBinding', 'endBinding'), e['_endpoints']):
                b = binding(x, y)
                e[key] = b
                if b:
                    r = next(r for r in rects if r['id'] == b['elementId'])
                    if not any(v['id'] == e['id'] for v in r['boundElements']):
                        r['boundElements'].append({'id': e['id'], 'type': 'arrow'})
    elements.sort(key=lambda e: e['_layer'])
    for e in elements:
        for key in list(e):
            if key.startswith('_'):
                del e[key]
    scene = {'type': 'excalidraw', 'version': 2, 'source': 'https://excalidraw.com',
             'elements': elements, 'appState': {'viewBackgroundColor': '#ffffff',
             'gridSize': None, 'gridModeEnabled': False, 'theme': 'light'}, 'files': {}}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(scene, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    counts = {t: sum(e['type'] == t for e in elements) for t in sorted({e['type'] for e in elements})}
    print(json.dumps({'output': str(OUTPUT), 'counts': counts, 'elements': len(elements)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
