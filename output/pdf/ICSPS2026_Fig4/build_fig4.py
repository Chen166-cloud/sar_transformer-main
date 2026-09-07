"""Compose Figure 4 from unchanged grayscale PNGs; no model inference or retouching."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from PIL import Image
from pypdf import PdfReader
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


PANELS = [
    ("00_Clean.png", "(a) Clean"),
    ("01_Noisy.png", "(b) Noisy"),
    ("LEE-MMSE.png", "(c) LEE"),
    ("SAR-BM3D.png", "(d) SAR-BM3D"),
    ("SAR-CAM.png", "(e) SAR-CAM"),
    ("SAR-Trans.png", "(f) Trans-SAR"),
    ("Ours-base.png", "(g) Ours"),
    ("Ours-base_ratio.png", "(h) Ours ratio"),
]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
            assert digest(source) == digest(destination)

    # IEEEtran conference: textwidth = 43 pc = 516 TeX pt.
    # PDF uses 72 pt/in, whereas TeX uses 72.27 pt/in.
    width = 516 * 72 / 72.27
    margin, gap, label_height, row_gap = 0.8, 6.0, 16.0, 8.0
    side = (width - 2 * margin - 3 * gap) / 4
    height = 2 * (side + label_height) + row_gap + 2 * margin
    pdfmetrics.registerFont(TTFont("FigureArial", str(args.font)))
    pdf = out / "fig4_buildings.pdf"
    drawing = canvas.Canvas(str(pdf), pagesize=(width, height), pageCompression=1,
                            invariant=1)
    drawing.setTitle("Figure 4: Buildings despeckling comparison")
    drawing.setAuthor("")
    drawing.setSubject("Eight original grayscale exports in a two-row comparison")
    drawing.setFillColorRGB(1, 1, 1)
    drawing.rect(0, 0, width, height, fill=1, stroke=0)
    records = []
    for index, (filename, label) in enumerate(PANELS):
        path = sources / filename
        with Image.open(path) as source_image:
            assert source_image.mode == "L" and source_image.size == (256, 256)
        row, col = divmod(index, 4)
        x = margin + col * (side + gap)
        y = height - margin - side - row * (side + label_height + row_gap)
        # Embed native 256 x 256 bytes losslessly. No crop, stretch, denoising,
        # sharpening, per-panel normalization, interpolation, or generated pixels.
        drawing.drawImage(ImageReader(str(path)), x, y, side, side)
        drawing.setStrokeColorRGB(0.55, 0.55, 0.55)
        drawing.setLineWidth(0.3)
        drawing.rect(x - 0.15, y - 0.15, side + 0.3, side + 0.3, fill=0, stroke=1)
        drawing.setFillColorRGB(0, 0, 0)
        drawing.setFont("FigureArial", 8.5)
        assert pdfmetrics.stringWidth(label, "FigureArial", 8.5) < side
        drawing.drawCentredString(x + side / 2, y - 11.5, label)
        records.append({"file": filename, "label": label, "sha256": digest(path),
                        "native_pixels": [256, 256], "mode": "L",
                        "image_rect_pdf_pt": [x, y, side, side]})
    drawing.showPage()
    drawing.save()

    reader = PdfReader(pdf)
    page = reader.pages[0]
    text = page.extract_text()
    assert len(reader.pages) == 1
    assert all(label in text for _, label in PANELS)
    # Verify the embedded image data, independently of the preview renderer.
    embedded = [im.image for im in page.images]
    assert len(embedded) == 8
    for filename, _ in PANELS:
        with Image.open(sources / filename) as original:
            assert sum(im.mode == original.mode and im.size == original.size
                       and im.tobytes() == original.tobytes() for im in embedded) == 1

    for dpi, suffix in [(600, ""), (150, "_preview")]:
        subprocess.run([args.pdftoppm, "-singlefile", "-r", str(dpi), "-png",
                        str(pdf), str(out / ("fig4_buildings" + suffix))], check=True,
                       capture_output=True)
    manifest = {
        "layout": "2 rows x 4 columns, row-major reading order",
        "panel_label_basis": "Author-confirmed renamed filenames; composition does not validate method identity",
        "width_pdf_pt": width, "height_pdf_pt": height,
        "width_mm": width / 72 * 25.4, "height_mm": height / 72 * 25.4,
        "label_font": "Arial", "label_font_pt": 8.5,
        "native_effective_dpi_at_textwidth": 256 / (side / 72),
        "assembly_processing": "Original PNGs embedded losslessly without pixel changes",
        "ratio_panel": "Supplied visualization; its mapping is separate from restoration panels",
        "panels": records,
        "checks": {"all_eight_embedded_images_pixel_identical": True,
                   "all_eight_labels_present": True, "one_page": True},
        "outputs": {name: digest(out / name) for name in
                    ["fig4_buildings.pdf", "fig4_buildings.png", "fig4_buildings_preview.png"]},
    }
    (out / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                             encoding="utf-8")
    if args.install_dir:
        args.install_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pdf, args.install_dir / pdf.name)
    print(json.dumps({"pdf": str(pdf), "size_mm": [manifest["width_mm"], manifest["height_mm"]],
                      "verified_original_panels": len(records)}, indent=2))


if __name__ == "__main__":
    main()
