"""Read-only checks for the limited Figure 1 / Figure 3 layout refinement.

Run from the project root after docs/ICSPS2026_LaTeX/build.ps1. This script
writes only qa.json and textual_diff.patch beside itself. It never edits the
paper, figures, model, or baseline. PDF rendering and visual judgment remain
separate tasks; a successful structural check does not claim visual approval.

Dependencies: pypdf and pdfplumber (bundled Codex Python).
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import difflib
import hashlib
import json
import math
import re

import pdfplumber
from pypdf import PdfReader
from pypdf.generic import ContentStream


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BEFORE = HERE / "before"
PAPER = ROOT / "docs/ICSPS2026_LaTeX"
BUILD = ROOT / "tmp/pdfs/paper_fig123/build"
STEM = "ICSPS2026_paper"
FIGURES = {
    1: ("fig1_overall_architecture.pdf", "fig:overview"),
    2: ("fig2_fdr_block.pdf", "fig:fdr"),
    3: ("fig3_representation_ams.pdf", "fig:representation_ams"),
    4: ("fig4_buildings.pdf", "fig:synthetic_qual"),
    5: ("fig5_real_sar.pdf", "fig:real_qual"),
}
ALLOWED_FIGURE_LABELS = {"fig:overview", "fig:representation_ams"}
IDENTITY = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest(path: Path) -> str:
    return sha(path.read_bytes())


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig", errors="replace")


def relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def figure_environments(source: str) -> dict[str, str]:
    result = {}
    for match in re.finditer(r"\\begin\{(figure\*?)\}.*?\\end\{\1\}", source, re.S):
        for label in re.findall(r"\\label\{([^}]+)\}", match.group()):
            result[label] = match.group()
    return result


def environments(source: str, names: tuple[str, ...]) -> list[str]:
    names_pattern = "|".join(re.escape(name) for name in names)
    return [match.group() for match in re.finditer(
        rf"\\begin\{{({names_pattern})\}}.*?\\end\{{\1\}}", source, re.S
    )]


def comparable_prose(source: str) -> str:
    """Allow only the two figure environments and requested reference spelling.

    Deliberately keep all other source text, comments, formulas, spaces, and
    layout commands. Method additions are reported, not silently discarded.
    """
    def replace_figure(match):
        labels = set(re.findall(r"\\label\{([^}]+)\}", match.group()))
        if labels & ALLOWED_FIGURE_LABELS:
            return "% QA: permitted figure environment " + ",".join(sorted(labels))
        return match.group()

    source = re.sub(r"\\begin\{(figure\*?)\}.*?\\end\{\1\}", replace_figure, source, flags=re.S)
    return source.replace(r"Figure~\ref{", r"Fig.~\ref{")


def source_checks(before: str, after: str) -> dict:
    previous_figures = figure_environments(before)
    current_figures = figure_environments(after)
    protected_figures = {
        label: {
            "unchanged": previous_figures.get(label) == current_figures.get(label),
            "present_before_and_after": label in previous_figures and label in current_figures,
        }
        for _, label in (FIGURES[n] for n in (2, 4, 5))
    }
    equations_before = environments(before, ("equation", "equation*", "align", "align*", "gather", "gather*"))
    equations_after = environments(after, ("equation", "equation*", "align", "align*", "gather", "gather*"))
    tables_before = environments(before, ("table", "table*"))
    tables_after = environments(after, ("table", "table*"))
    bibliography_before = environments(before, ("thebibliography",))
    bibliography_after = environments(after, ("thebibliography",))
    # Include any standalone display-math delimiters, not only equation envs.
    displays_before = re.findall(r"\\\[.*?\\\]|\$\$.*?\$\$", before, flags=re.S)
    displays_after = re.findall(r"\\\[.*?\\\]|\$\$.*?\$\$", after, flags=re.S)
    before_prose, after_prose = comparable_prose(before), comparable_prose(after)
    diff = "".join(difflib.unified_diff(
        before_prose.splitlines(keepends=True), after_prose.splitlines(keepends=True),
        fromfile="before/ICSPS2026_paper.tex (Fig1/Fig3 hidden; reference spelling normalized)",
        tofile="after/ICSPS2026_paper.tex (same normalization)", n=3,
    ))
    (HERE / "textual_diff.patch").write_text(diff, encoding="utf-8")
    changes = []
    old_lines, new_lines = before_prose.splitlines(), after_prose.splitlines()
    for operation, a0, a1, b0, b1 in difflib.SequenceMatcher(a=old_lines, b=new_lines).get_opcodes():
        if operation == "equal":
            continue
        old, new = "\n".join(old_lines[a0:a1]), "\n".join(new_lines[b0:b1])
        changes.append({
            "operation": operation, "before_line": a0 + 1, "after_line": b0 + 1,
            "before": old, "after": new,
            "contains_direct_figure_reference": bool(re.search(
                r"\\ref\{fig:(?:overview|representation_ams)\}", old + new)),
            "review": "Review exact wording: only directly related method references/input clarification are authorized.",
        })
    source_labels = re.findall(r"\\label\{([^}]+)\}", after)
    duplicates = {label: count for label, count in Counter(source_labels).items() if count > 1}
    source_references = re.findall(r"\\(?:ref|eqref|pageref|autoref|subref)\{([^}]+)\}", after)
    includes = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", after)
    missing_graphics = [name for name in includes if not (PAPER / name).is_file()]
    invalid_old_subpanel_refs = re.findall(r"(?:Fig\.|Figure)~\\ref\{fig:overview\}\s*\([ab]\)", after)
    # Math outside the two changed figure environments should retain original
    # inline formulas. Additional input-symbol explanations may introduce math.
    inline_pattern = r"\\\(.*?\\\)|(?<!\\)\$(?!\$).*?(?<!\\)\$(?!\$)"
    inline_before = re.findall(inline_pattern, before_prose, flags=re.S)
    inline_after = re.findall(inline_pattern, after_prose, flags=re.S)
    inline_changes = []
    original_inline_preserved = True
    for op, a0, a1, b0, b1 in difflib.SequenceMatcher(a=inline_before, b=inline_after, autojunk=False).get_opcodes():
        if op != "equal":
            inline_changes.append({"operation": op, "before": inline_before[a0:a1], "after": inline_after[b0:b1]})
            if op in ("replace", "delete"):
                original_inline_preserved = False
    return {
        "preamble_exactly_unchanged": before.split(r"\begin{document}", 1)[0] == after.split(r"\begin{document}", 1)[0],
        "displayed_equation_environments_exactly_unchanged": equations_before == equations_after,
        "displayed_equation_count": len(equations_after),
        "standalone_display_math_exactly_unchanged": displays_before == displays_after,
        "tables_exactly_unchanged": tables_before == tables_after,
        "table_count": len(tables_after),
        "bibliography_exactly_unchanged": bibliography_before == bibliography_after,
        "protected_figure_environments": protected_figures,
        "all_original_inline_math_preserved": original_inline_preserved,
        "inline_math_changes": inline_changes,
        "duplicate_source_labels": duplicates,
        "unresolved_source_references": sorted(set(source_references) - set(source_labels)),
        "missing_graphics": missing_graphics,
        "invalid_old_fig1_subpanel_references": invalid_old_subpanel_refs,
        "remaining_long_form_figure_references": re.findall(r"Figure~\\ref\{[^}]+\}", after),
        "residual_text_changes": changes,
        "residual_diff_file": relative(HERE / "textual_diff.patch"),
        "residual_changes_require_manual_review": bool(changes),
        "permitted_automatically": ["Only fig:overview and fig:representation_ams figure environments", "Figure~\\ref to Fig.~\\ref globally"],
    }


def protected_checks() -> list[dict]:
    result = []
    for entry in json.loads(read(BEFORE / "protected_hashes.json")):
        path = Path(entry["Path"])
        allowed = path.name in {FIGURES[n][0] for n in (1, 3)}
        actual = digest(path) if path.is_file() else None
        result.append({
            "file": relative(path), "expected_sha256": entry["Hash"].lower(),
            "actual_sha256": actual, "unchanged": actual == entry["Hash"].lower(),
            "authorized_figure_change": allowed,
        })
    return result


def concatenate(outer, inner):
    """Composition outer(inner(point)); PDF cm updates current*new matrix."""
    a, b, c, d, e, f = outer
    g, h, i, j, k, l = inner
    return (a*g+c*h, b*g+d*h, a*i+c*j, b*i+d*j, a*k+c*l+e, b*k+d*l+f)


def point(matrix, x, y):
    a, b, c, d, e, f = matrix
    return (a*x+c*y+e, b*x+d*y+f)


def transformed_box(matrix, bbox):
    x0, y0, x1, y1 = map(float, bbox)
    corners = [point(matrix, x, y) for x, y in ((x0, y0), (x0, y1), (x1, y0), (x1, y1))]
    return [min(p[0] for p in corners), min(p[1] for p in corners),
            max(p[0] for p in corners), max(p[1] for p in corners)]


def inspect_drawn_objects(reader, stream, resources, ctm=IDENTITY, stack_path=(), depth=0):
    """Walk only invoked XObjects, carrying q/Q/cm state and Form matrices."""
    if depth > 24:
        raise RuntimeError("Unexpectedly deep PDF form nesting")
    if stream is None:
        return
    resources = resources.get_object() if resources else {}
    stack = []
    for args, operator in ContentStream(stream, reader).operations:
        if operator == b"q":
            stack.append(ctm)
        elif operator == b"Q":
            ctm = stack.pop() if stack else ctm
        elif operator == b"cm":
            ctm = concatenate(ctm, tuple(map(float, args)))
        elif operator == b"Do":
            name = args[0]
            xobjects = resources.get("/XObject", {})
            xobjects = xobjects.get_object() if hasattr(xobjects, "get_object") else xobjects
            if name not in xobjects:
                raise RuntimeError(f"Missing drawn XObject {name}")
            obj = xobjects[name].get_object()
            path = stack_path + (str(name),)
            if obj.get("/Subtype") == "/Form":
                matrix = concatenate(ctm, tuple(map(float, obj.get("/Matrix", IDENTITY))))
                bbox = list(map(float, obj.get("/BBox", [0, 0, 0, 0])))
                yield {"type": "form", "path": list(path), "stream_sha256": sha(obj.get_data()),
                       "matrix": list(matrix), "native_bbox_pt": bbox,
                       "placed_bbox_pt": transformed_box(matrix, bbox)}
                yield from inspect_drawn_objects(reader, obj, obj.get("/Resources", resources), matrix, path, depth+1)
            elif obj.get("/Subtype") == "/Image":
                yield {"type": "image", "path": list(path), "matrix": list(ctm),
                       "pixels": [obj.get("/Width"), obj.get("/Height")]}


def font_key(font):
    return str(font.get("/BaseFont", "")).lstrip("/")


def font_resources(resources, found, visited):
    if not resources:
        return
    resources = resources.get_object()
    fonts = resources.get("/Font", {})
    fonts = fonts.get_object() if hasattr(fonts, "get_object") else fonts
    for reference in fonts.values():
        obj = reference.get_object()
        identity = (getattr(reference, "idnum", None), getattr(reference, "generation", None), font_key(obj))
        if identity in visited:
            continue
        visited.add(identity)
        descendants = obj.get("/DescendantFonts")
        descendant = descendants[0].get_object() if descendants else obj
        descriptor = descendant.get("/FontDescriptor")
        descriptor = descriptor.get_object() if descriptor else {}
        embedded = any(key in descriptor for key in ("/FontFile", "/FontFile2", "/FontFile3"))
        # A Type 3 font stores its glyph programs directly in CharProcs.
        if obj.get("/Subtype") == "/Type3" and obj.get("/CharProcs"):
            embedded = True
        found.append({"name": font_key(obj), "subtype": str(obj.get("/Subtype")), "embedded": embedded})
    xobjects = resources.get("/XObject", {})
    xobjects = xobjects.get_object() if hasattr(xobjects, "get_object") else xobjects
    for reference in xobjects.values():
        obj = reference.get_object()
        if obj.get("/Subtype") == "/Form":
            font_resources(obj.get("/Resources"), found, visited)


def native_figure(path: Path) -> dict:
    reader = PdfReader(path)
    page = reader.pages[0]
    contents = page.get_contents()
    objects = list(inspect_drawn_objects(reader, contents, page.get("/Resources")))
    with pdfplumber.open(path) as pdf:
        visible = [char for char in pdf.pages[0].chars if char.get("text", "").strip()]
        size_counts = Counter(round(char["size"], 2) for char in visible)
        size_list = [{"size_pt": size, "visible_character_count": count} for size, count in sorted(size_counts.items())]
    return {
        "file": relative(path), "sha256": digest(path), "page_count": len(reader.pages),
        "width_pt": float(page.mediabox.width), "height_pt": float(page.mediabox.height),
        "width_mm": float(page.mediabox.width)*25.4/72,
        "height_mm": float(page.mediabox.height)*25.4/72,
        "content_stream_sha256": sha(contents.get_data()),
        "drawn_raster_image_count": sum(obj["type"] == "image" for obj in objects),
        "visible_character_sizes": size_list,
        "font_size_note": "Character heights include mathematical subscripts/superscripts; review ordinary labels visually.",
    }


def pdf_checks(pdf_path: Path, native_figures: dict[int, dict]) -> dict:
    reader = PdfReader(pdf_path)
    placements = []
    for page_number, page in enumerate(reader.pages, 1):
        for obj in inspect_drawn_objects(reader, page.get_contents(), page.get("/Resources")):
            if obj["type"] != "form":
                continue
            matches = [number for number, figure in native_figures.items()
                       if figure["content_stream_sha256"] == obj["stream_sha256"]]
            for number in matches:
                box = obj["placed_bbox_pt"]
                matrix = obj["matrix"]
                sx, sy = math.hypot(matrix[0], matrix[1]), math.hypot(matrix[2], matrix[3])
                width, height = box[2]-box[0], box[3]-box[1]
                placements.append({
                    "figure": number, "page": page_number, "form_path": obj["path"],
                    "identification": "Exact decoded source-page stream SHA-256 equals placed Form stream SHA-256",
                    "placement_matrix": matrix, "scale_x": sx, "scale_y": sy,
                    "uniform_scale": abs(sx-sy) <= 1e-5,
                    "bbox_pdf_pt": box,
                    "bbox_top_origin_pt": [box[0], float(page.mediabox.height)-box[3], box[2], float(page.mediabox.height)-box[1]],
                    "width_pt": width, "height_pt": height,
                    "width_mm": width*25.4/72, "height_mm": height*25.4/72,
                    "within_page": box[0] >= -0.5 and box[1] >= -0.5 and box[2] <= float(page.mediabox.width)+0.5 and box[3] <= float(page.mediabox.height)+0.5,
                    "ordinary_label_size_scaling": "Multiply native figure text sizes by scale_x (uniform drawing).",
                })
    fonts, visited = [], set()
    for page in reader.pages:
        font_resources(page.get("/Resources"), fonts, visited)
    visible_fonts, outside, pages, double_question_pages = set(), [], [], []
    with pdfplumber.open(pdf_path) as pdf:
        for number, page in enumerate(pdf.pages, 1):
            chars = [char for char in page.chars if char.get("text", "").strip()]
            visible_fonts.update(char["fontname"] for char in chars)
            for char in chars:
                if char["x0"] < -0.5 or char["top"] < -0.5 or char["x1"] > page.width+0.5 or char["bottom"] > page.height+0.5:
                    outside.append({"page": number, "text": char["text"], "box": [char["x0"], char["top"], char["x1"], char["bottom"]]})
            text = page.extract_text() or ""
            if "??" in text:
                double_question_pages.append(number)
            bounds = ([min(char["x0"] for char in chars), min(char["top"] for char in chars),
                       max(char["x1"] for char in chars), max(char["bottom"] for char in chars)] if chars else None)
            pages.append({"page": number, "width_pt": page.width, "height_pt": page.height,
                          "visible_character_count": len(chars), "text_bounds_pt": bounds})
    for font in fonts:
        font["used_for_visible_text"] = font["name"] in visible_fonts
    missing_resource_names = sorted(visible_fonts - {font["name"] for font in fonts})
    return {
        "file": relative(pdf_path), "sha256": digest(pdf_path), "page_count": len(reader.pages),
        "placements": placements, "pages": pages, "fonts": fonts,
        "unmatched_visible_font_names": missing_resource_names,
        "all_visible_fonts_embedded": not missing_resource_names and all(font["embedded"] for font in fonts if font["used_for_visible_text"]),
        "unused_unembedded_font_resources": [font for font in fonts if not font["embedded"] and not font["used_for_visible_text"]],
        "text_outside_page": outside, "double_question_mark_pages": double_question_pages,
        "all_five_figures_identified_once": Counter(item["figure"] for item in placements) == Counter(FIGURES.keys()),
    }


def warning_lines(log: str, pattern: str) -> list[str]:
    return re.findall(r"^.*(?:" + pattern + r").*$", log, re.M)


def normalize_warning(text: str) -> str:
    return re.sub(r"(?:at |in paragraph at )lines? \d+(?:--\d+)?", "at lines <changed>", text).strip()


def log_checks(before_log: str, after_log: str, aux: str) -> dict:
    categories = {
        "overfull": r"Overfull \\[hv]box",
        "underfull": r"Underfull \\[hv]box",
        "font": r"Font Warning|Missing character|font.*not found|font.*not load|font.*substitut",
        "reference": r"undefined|multiply defined|Label\(s\) may have changed|Rerun to get cross-references",
        "errors": r"^!|LaTeX Error|Emergency stop|Fatal error|Float too large|Too many unprocessed floats|File .* not found",
    }
    warnings = {}
    for category, pattern in categories.items():
        previous, current = warning_lines(before_log, pattern), warning_lines(after_log, pattern)
        new_counter = Counter(map(normalize_warning, current)) - Counter(map(normalize_warning, previous))
        warnings[category] = {"before": previous, "after": current, "new_normalized": list(new_counter.elements())}
    labels = re.findall(r"\\newlabel\{([^}]+)\}\{\{([^}]+)\}\{([^}]+)\}", aux)
    label_counts = Counter(item[0] for item in labels)
    return {
        "warnings": warnings,
        "aux_labels": {label: {"number": number, "page": page} for label, number, page in labels},
        "duplicate_aux_labels": {label: count for label, count in label_counts.items() if count > 1},
        "output_pdf_line": warning_lines(after_log, r"Output written on"),
    }


def main() -> int:
    required = [BEFORE / f"{STEM}.tex", BEFORE / f"{STEM}.pdf", BEFORE / f"{STEM}.log",
                BEFORE / "protected_hashes.json", PAPER / f"{STEM}.tex", PAPER / f"{STEM}.pdf",
                BUILD / f"{STEM}.log", BUILD / f"{STEM}.aux"]
    missing = [relative(path) for path in required if not path.is_file()]
    if missing:
        report = {"status": "missing_required_artifacts", "missing": missing}
        (HERE / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1

    sources = source_checks(read(BEFORE / f"{STEM}.tex"), read(PAPER / f"{STEM}.tex"))
    protected = protected_checks()
    figures_before = {n: native_figure(BEFORE / "figures" / name) for n, (name, _) in FIGURES.items()}
    figures_after = {n: native_figure(PAPER / "figures" / name) for n, (name, _) in FIGURES.items()}
    before_pdf = pdf_checks(BEFORE / f"{STEM}.pdf", figures_before)
    after_pdf = pdf_checks(PAPER / f"{STEM}.pdf", figures_after)
    compilation = log_checks(read(BEFORE / f"{STEM}.log"), read(BUILD / f"{STEM}.log"), read(BUILD / f"{STEM}.aux"))
    failures = []
    for key in ("preamble_exactly_unchanged", "displayed_equation_environments_exactly_unchanged",
                "standalone_display_math_exactly_unchanged", "tables_exactly_unchanged",
                "bibliography_exactly_unchanged", "all_original_inline_math_preserved"):
        if not sources[key]:
            failures.append(key)
    if any(not value["unchanged"] or not value["present_before_and_after"] for value in sources["protected_figure_environments"].values()):
        failures.append("Protected Fig2/Fig4/Fig5 figure environment changed")
    if any(not entry["unchanged"] and not entry["authorized_figure_change"] for entry in protected):
        failures.append("Protected figure/template SHA-256 changed or missing")
    for key in ("duplicate_source_labels", "unresolved_source_references", "missing_graphics", "invalid_old_fig1_subpanel_references", "remaining_long_form_figure_references"):
        if sources[key]:
            failures.append(key)
    for key in ("all_visible_fonts_embedded", "all_five_figures_identified_once"):
        if not after_pdf[key]:
            failures.append(key)
    for key in ("text_outside_page", "double_question_mark_pages"):
        if after_pdf[key]:
            failures.append(key)
    if not all(placement["uniform_scale"] and placement["within_page"] for placement in after_pdf["placements"]):
        failures.append("Figure placement deforms or exceeds page")
    if any(figures_after[number]["drawn_raster_image_count"] for number in (1, 3)):
        failures.append("Figure1/Figure3 invokes raster image objects")
    if compilation["duplicate_aux_labels"]:
        failures.append("duplicate_aux_labels")
    for category in ("overfull", "font"):
        if compilation["warnings"][category]["new_normalized"]:
            failures.append("New " + category + " warnings")
    for category in ("reference", "errors"):
        if compilation["warnings"][category]["after"]:
            failures.append("Final build " + category + " warnings")
    report = {
        "status": "structural_checks_failed" if failures else "structural_checks_passed_pending_manual_diff_and_visual_review",
        "failures": failures,
        "scope": "Read-only comparison of saved baseline and final artifacts; no rendering or visual approval is implied.",
        "source": sources, "recorded_protected_hashes": protected,
        "figures_before": figures_before, "figures_after": figures_after,
        "before_pdf": before_pdf, "after_pdf": after_pdf,
        "page_count_before": before_pdf["page_count"], "page_count_after": after_pdf["page_count"],
        "compilation": compilation,
        "known_unchanged_source_discrepancy": "AMS paper denominator is ||M||_1+epsilon; train_icsps2026_ams.py:111 uses mask.sum().clamp_min(1.0). This script does not alter either.",
        "manual_review_required": ["Exact residual prose diff", "Rendered figures at final publication size", "Arrow meaning, overlaps, clipping, spacing and pagination whitespace"],
    }
    (HERE / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    summary = {key: report[key] for key in ("status", "failures", "page_count_before", "page_count_after")}
    summary["figure_1_3_placed_dimensions"] = [{key: value for key, value in placement.items()
        if key in ("figure", "page", "width_mm", "height_mm", "uniform_scale")}
        for placement in after_pdf["placements"] if placement["figure"] in (1, 3)]
    summary["all_visible_fonts_embedded"] = after_pdf["all_visible_fonts_embedded"]
    summary["residual_prose_change_count"] = len(sources["residual_text_changes"])
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
