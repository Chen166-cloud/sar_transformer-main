"""Compose Figure 5 using the author-confirmed filenames and original pixels."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from PIL import Image
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


PANELS = [
    ("noisy.png", "(a) Noisy"),
    ("SAR-BM3D.png", "(b) SAR-BM3D"),
    ("SAR2SAR.png", "(c) SAR2SAR"),
    ("SDUDNet.png", "(d) SDUDNet"),
    ("sar-trans.png", "(e) Trans-SAR"),
    ("Ours-base.png", "(f) Ours-base"),
    ("Ours-ams.png", "(g) Ours+AMS"),
    ("Ours-ams-ratio.png", "(h) Ours+AMS ratio"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--font", type=Path, default=Path("C:/Windows/Fonts/arial.ttf"))
    parser.add_argument("--pdftoppm", default="pdftoppm")
    parser.add_argument("--install-dir", type=Path)
    args = parser.parse_args()
    out = Path(__file__).resolve().parent
    sources = out / "source_images"
    sources.mkdir(exist_ok=True)
    if args.source_dir:
        for filename, _ in PANELS:
            source = args.source_dir / filename
            destination = sources / filename
            if source.resolve() != destination.resolve():
                shutil.copy2(source, destination)

    # Match Figure 4 and IEEEtran's 43 pc text width (convert TeX pt to PDF pt).
    width = 516 * 72 / 72.27
    margin, gap, label_height, row_gap = 0.8, 6.0, 16.0, 8.0
    side = (width - 2 * margin - 3 * gap) / 4
    height = 2 * (side + label_height) + row_gap + 2 * margin
    pdfmetrics.registerFont(TTFont("FigureArial", str(args.font)))
    pdf = out / "fig5_real_sar.pdf"
    drawing = canvas.Canvas(str(pdf), pagesize=(width, height), pageCompression=1,
                            invariant=1)
    drawing.setTitle("Figure 5: Real-SAR comparison before and after AMS")
    drawing.setAuthor("")
    drawing.setSubject("One scene, eight author-named panels, original display values")
    drawing.setFillColorRGB(1, 1, 1)
    drawing.rect(0, 0, width, height, fill=1, stroke=0)
    records = []
    for index, (filename, label) in enumerate(PANELS):
        path = sources / filename
        with Image.open(path) as source_image:
            size, mode = source_image.size, source_image.mode
        row, col = divmod(index, 4)
        x = margin + col * (side + gap)
        y = height - margin - side - row * (side + label_height + row_gap)
        # Embed the original raster. Keep the red rectangle already in Noisy.
        # No grayscale conversion, crop, normalization, or synthetic image content.
        drawing.drawImage(ImageReader(str(path)), x, y, side, side,
                          preserveAspectRatio=True, anchor="c")
        drawing.setStrokeColorRGB(0.55, 0.55, 0.55)
        drawing.setLineWidth(0.3)
        drawing.rect(x - 0.15, y - 0.15, side + 0.3, side + 0.3, fill=0, stroke=1)
        drawing.setFillColorRGB(0, 0, 0)
        drawing.setFont("FigureArial", 8.5)
        if pdfmetrics.stringWidth(label, "FigureArial", 8.5) >= side:
            raise ValueError(f"Label exceeds panel width: {label}")
        drawing.drawCentredString(x + side / 2, y - 11.5, label)
        records.append({"file": filename, "label": label,
                        "native_pixels": list(size), "mode": mode,
                        "image_rect_pdf_pt": [x, y, side, side]})
    drawing.showPage()
    drawing.save()

    for dpi, suffix in [(600, ""), (150, "_preview")]:
        subprocess.run([args.pdftoppm, "-singlefile", "-r", str(dpi), "-png",
                        str(pdf), str(out / ("fig5_real_sar" + suffix))], check=True,
                       capture_output=True)
    manifest = {
        "layout": "2 rows x 4 columns, row-major reading order, matched to Figure 4",
        "panel_label_basis": "Author-confirmed renamed filenames",
        "scope": "Composition and layout only; no historical-output or model-identity audit",
        "width_pdf_pt": width, "height_pdf_pt": height,
        "width_mm": width / 72 * 25.4, "height_mm": height / 72 * 25.4,
        "label_font": "Arial", "label_font_pt": 8.5,
        "native_effective_dpi_at_textwidth": 256 / (side / 72),
        "assembly_processing": "Original images embedded without raster edits",
        "existing_annotation": "Red rectangle in Noisy retained from the input PNG",
        "ratio_panel": "Supplied Ours+AMS ratio visualization; no new display mapping applied",
        "panels": records,
    }
    (out / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                             encoding="utf-8")
    if args.install_dir:
        args.install_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pdf, args.install_dir / pdf.name)
    print(json.dumps({"pdf": str(pdf), "size_mm": [manifest["width_mm"], manifest["height_mm"]],
                      "panels": len(records)}, indent=2))


if __name__ == "__main__":
    main()
