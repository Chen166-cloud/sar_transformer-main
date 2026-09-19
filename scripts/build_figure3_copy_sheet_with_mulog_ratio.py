"""Build a Word copy sheet containing the eight Figure 3 panels plus a ratio.

The MuLoG-DRUNet ratio is rendered from the preserved float32 dB-ratio array
using the same symmetric display range as the audited Figure 3 ratio products.
All nine pictures are inserted as independent inline Word objects at the exact
small-panel size used by Figure 3 in V3.docx.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import shutil
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from PIL import Image
from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Emu, Inches, Pt, Twips


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURE_ROOT = (
    PROJECT_ROOT / "output" / "umbra_buenos_aires_multimethod_1024" / "figure3"
)
SOURCE_PANEL_ROOT = FIGURE_ROOT / "word_ready_no_text" / "panels"
OUTPUT_ROOT = FIGURE_ROOT / "word_copy_with_mulog_ratio"
OUTPUT_PANEL_ROOT = OUTPUT_ROOT / "panels"
OUTPUT_DOCX = OUTPUT_ROOT / "Figure3_9_individual_images_copy_sheet.docx"
OUTPUT_ZIP = OUTPUT_ROOT / "Figure3_9_individual_images_copy_sheet.zip"

RATIO_NPY = FIGURE_ROOT / "ratios" / "ratio_mulog-drunet_db_noisy-over-output.npy"
REFERENCE_RATIO_PNG = (
    FIGURE_ROOT / "ratios" / "ratio_mulog-drunet_db_noisy-over-output.png"
)
FIGURE_MANIFEST = FIGURE_ROOT / "manifest.json"

V3_PANEL_EMU = 758_190
V3_PANEL_INCHES = V3_PANEL_EMU / 914_400
PANEL_PIXELS = 1024
WORD_READY_DPI = PANEL_PIXELS / V3_PANEL_INCHES

PANELS = (
    ("a_noisy.png", "Noisy", SOURCE_PANEL_ROOT / "a_noisy.png"),
    ("b_sar-bm3d.png", "SAR-BM3D", SOURCE_PANEL_ROOT / "b_sar-bm3d.png"),
    ("c_sar2sar.png", "SAR2SAR", SOURCE_PANEL_ROOT / "c_sar2sar.png"),
    ("d_sdudnet.png", "SDUDNet", SOURCE_PANEL_ROOT / "d_sdudnet.png"),
    ("e_trans-sar.png", "Trans-SAR", SOURCE_PANEL_ROOT / "e_trans-sar.png"),
    ("f_cl-sar.png", "CL-SAR", SOURCE_PANEL_ROOT / "f_cl-sar.png"),
    ("g_merlin.png", "MERLIN", SOURCE_PANEL_ROOT / "g_merlin.png"),
    ("h_mulog-drunet.png", "MuLoG-DRUNet", SOURCE_PANEL_ROOT / "h_mulog-drunet.png"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def map_range(array: np.ndarray, low: float, high: float) -> np.ndarray:
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        raise ValueError("Invalid ratio display range")
    return np.rint(
        np.clip((array.astype(np.float64) - low) / (high - low), 0.0, 1.0)
        * 255.0
    ).astype(np.uint8)


def add_image_row(document: Document, paths: list[Path], descriptions: list[str],
                  before_pt: float) -> None:
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_before = Pt(before_pt)
    paragraph.paragraph_format.space_after = Pt(0)
    paragraph.paragraph_format.line_spacing = 1.0
    for index, (path, description) in enumerate(zip(paths, descriptions, strict=True)):
        if index:
            paragraph.add_run(" ")
        shape = paragraph.add_run().add_picture(
            str(path), width=Emu(V3_PANEL_EMU), height=Emu(V3_PANEL_EMU)
        )
        shape._inline.docPr.set("name", path.name)
        shape._inline.docPr.set("descr", description)


def build_docx(paths: list[Path], descriptions: list[str]) -> None:
    document = Document()
    document.core_properties.title = "Figure 3 Image Copy Sheet"
    section = document.sections[0]
    section.orientation = WD_ORIENT.PORTRAIT
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.left_margin = Twips(979)
    section.right_margin = Twips(979)
    section.top_margin = Twips(1138)
    section.bottom_margin = Twips(979)

    normal = document.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(8)
    normal._element.rPr.rFonts.set(qn("w:ascii"), "Times New Roman")
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), "Times New Roman")

    settings = document.settings._element
    if settings.find(qn("w:doNotCompressPictures")) is None:
        settings.append(OxmlElement("w:doNotCompressPictures"))

    add_image_row(document, paths[0:4], descriptions[0:4], before_pt=4)
    add_image_row(document, paths[4:8], descriptions[4:8], before_pt=8)
    add_image_row(document, paths[8:9], descriptions[8:9], before_pt=8)

    document.save(OUTPUT_DOCX)


def build_preview(paths: list[Path]) -> Path:
    tile = 320
    gap = 18
    width = 4 * tile + 3 * gap
    height = 3 * tile + 2 * gap
    canvas = Image.new("RGB", (width, height), "white")
    for index, path in enumerate(paths):
        if index < 8:
            row, col = divmod(index, 4)
        else:
            row, col = 2, 0
        with Image.open(path) as image:
            pixels = image.convert("RGB").resize((tile, tile), Image.Resampling.LANCZOS)
        if index == 8:
            x = (width - tile) // 2
        else:
            x = col * (tile + gap)
        y = row * (tile + gap)
        canvas.paste(pixels, (x, y))
    preview = OUTPUT_ROOT / "Figure3_9_individual_images_preview.png"
    canvas.save(preview, optimize=True)
    return preview


def main() -> None:
    OUTPUT_PANEL_ROOT.mkdir(parents=True, exist_ok=True)

    exported_paths: list[Path] = []
    descriptions: list[str] = []
    records: list[dict[str, object]] = []
    for filename, description, source in PANELS:
        if not source.is_file():
            raise FileNotFoundError(source)
        target = OUTPUT_PANEL_ROOT / filename
        shutil.copy2(source, target)
        with Image.open(target) as image:
            if image.size != (PANEL_PIXELS, PANEL_PIXELS):
                raise ValueError(f"{target}: expected 1024 x 1024 pixels")
        exported_paths.append(target)
        descriptions.append(description)
        records.append(
            {
                "filename": filename,
                "description": description,
                "source": str(source.resolve()),
                "sha256": sha256(target),
            }
        )

    manifest = json.loads(FIGURE_MANIFEST.read_text(encoding="utf-8"))
    ratio_info = manifest["ratios"]["MuLoG-DRUNet"]
    low, high = (float(value) for value in ratio_info["display_range_db"])
    ratio = np.load(RATIO_NPY, allow_pickle=False)
    if ratio.shape != (PANEL_PIXELS, PANEL_PIXELS):
        raise ValueError(f"Unexpected ratio shape: {ratio.shape}")
    if not np.isfinite(ratio).all():
        raise ValueError("MuLoG-DRUNet ratio contains non-finite values")
    regenerated_pixels = map_range(ratio, low, high)
    with Image.open(REFERENCE_RATIO_PNG) as reference:
        reference_pixels = np.asarray(reference.convert("L"))
    difference = np.abs(
        regenerated_pixels.astype(np.int16) - reference_pixels.astype(np.int16)
    )
    # The stored NPY is float32, whereas the audited PNG was mapped from the
    # pre-cast float64 ratio.  Only threshold-tie pixels may differ by one gray
    # level.  Reuse the audited PNG pixels for exact continuity with Figure 3.
    if int(np.max(difference)) > 1 or int(np.count_nonzero(difference)) > 4:
        raise RuntimeError("Stored ratio array is inconsistent with the audited ratio PNG")
    ratio_pixels = reference_pixels

    ratio_target = OUTPUT_PANEL_ROOT / "i_mulog-drunet_ratio.png"
    Image.fromarray(ratio_pixels, mode="L").save(
        ratio_target, dpi=(WORD_READY_DPI, WORD_READY_DPI), optimize=True
    )
    exported_paths.append(ratio_target)
    descriptions.append("MuLoG-DRUNet ratio 10 log10 noisy over output")
    records.append(
        {
            "filename": ratio_target.name,
            "description": descriptions[-1],
            "source_array": str(RATIO_NPY.resolve()),
            "definition_db": ratio_info["definition_db"],
            "display_range_db": [low, high],
            "sha256": sha256(ratio_target),
            "audited_png_threshold_tie_pixels": int(np.count_nonzero(difference)),
            "minimum_db": float(np.min(ratio)),
            "median_db": float(np.median(ratio)),
            "maximum_db": float(np.max(ratio)),
        }
    )

    build_docx(exported_paths, descriptions)
    preview = build_preview(exported_paths)

    output_manifest = {
        "complete": True,
        "docx": OUTPUT_DOCX.name,
        "layout": "nine independent inline images in rows of 4, 4, and 1",
        "captions_embedded": False,
        "panel_display_emu": [V3_PANEL_EMU, V3_PANEL_EMU],
        "panel_display_inches": [V3_PANEL_INCHES, V3_PANEL_INCHES],
        "panels": records,
        "preview": preview.name,
    }
    output_manifest_path = OUTPUT_ROOT / "manifest.json"
    output_manifest_path.write_text(
        json.dumps(output_manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    package_members = [OUTPUT_DOCX, output_manifest_path, *exported_paths]
    with ZipFile(OUTPUT_ZIP, "w", compression=ZIP_DEFLATED) as package:
        for path in package_members:
            package.write(path, path.relative_to(OUTPUT_ROOT).as_posix())

    print(
        json.dumps(
            {
                "docx": str(OUTPUT_DOCX),
                "ratio_png": str(ratio_target),
                "preview": str(preview),
                "zip": str(OUTPUT_ZIP),
                "inline_image_count": len(exported_paths),
                "ratio_definition": ratio_info["definition_db"],
                "ratio_display_range_db": [low, high],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
