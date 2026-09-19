"""Export Figure 3 as eight standalone, caption-free Word-ready PNG panels.

The geometry follows Figure 3 in V3.docx: two centered rows of four inline,
square images.  V3 stores each panel at 758190 EMU (0.8291667 inches), with
one ASCII-space gap between adjacent panels.  This export keeps the new
1024-by-1024 pixels and writes matching physical-size metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURE_ROOT = (
    PROJECT_ROOT / "output" / "umbra_buenos_aires_multimethod_1024" / "figure3"
)
DEFAULT_OUTPUT = FIGURE_ROOT / "word_ready_no_text"

PANEL_PIXELS = 1024
V3_PANEL_EMU = 758_190
EMU_PER_INCH = 914_400
V3_PANEL_INCHES = V3_PANEL_EMU / EMU_PER_INCH
PNG_DPI = PANEL_PIXELS / V3_PANEL_INCHES

# V3 label centers are spaced by about 1254.67 twips while each image is 1194
# twips wide.  The inferred 60.67-twip gap is 52 pixels at this export scale.
GRID_GAP_PIXELS = 52

PANELS = (
    ("a_noisy.png", "Noisy", FIGURE_ROOT / "panels" / "a_noisy_boxed.png"),
    ("b_sar-bm3d.png", "SAR-BM3D", FIGURE_ROOT / "panels" / "b_sar-bm3d.png"),
    ("c_sar2sar.png", "SAR2SAR", FIGURE_ROOT / "panels" / "c_sar2sar.png"),
    ("d_sdudnet.png", "SDUDNet", FIGURE_ROOT / "panels" / "d_sdudnet.png"),
    ("e_trans-sar.png", "Trans-SAR", FIGURE_ROOT / "panels" / "e_trans-sar.png"),
    ("f_cl-sar.png", "CL-SAR", FIGURE_ROOT / "panels" / "f_cl-sar.png"),
    ("g_merlin.png", "MERLIN", FIGURE_ROOT / "panels" / "g_merlin.png"),
    ("h_mulog-drunet.png", "MuLoG-DRUNet", FIGURE_ROOT / "panels" / "h_mulog-drunet.png"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_word_ready(source: Path, target: Path) -> dict[str, object]:
    if not source.is_file():
        raise FileNotFoundError(source)
    with Image.open(source) as image:
        pixels = image.copy()
    if pixels.size != (PANEL_PIXELS, PANEL_PIXELS):
        raise ValueError(f"{source}: expected 1024x1024, got {pixels.size}")
    if pixels.mode not in {"L", "RGB"}:
        pixels = pixels.convert("RGB")
    pixels.save(target, dpi=(PNG_DPI, PNG_DPI), optimize=True)
    with Image.open(target) as check:
        if check.size != pixels.size or check.mode != pixels.mode:
            raise RuntimeError(f"Round-trip mismatch for {target}")
        dpi = check.info.get("dpi")
    return {
        "source": str(source.resolve()),
        "source_sha256": sha256(source),
        "output": str(target.name),
        "output_sha256": sha256(target),
        "pixels": [PANEL_PIXELS, PANEL_PIXELS],
        "mode": pixels.mode,
        "png_dpi": list(dpi) if dpi else None,
        "word_display_inches": [V3_PANEL_INCHES, V3_PANEL_INCHES],
        "contains_text": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    output = args.output.resolve()
    panels_dir = output / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)

    panel_records: list[dict[str, object]] = []
    exported_paths: list[Path] = []
    for filename, method, source in PANELS:
        target = panels_dir / filename
        record = save_word_ready(source, target)
        record["method"] = method
        record["first_panel_red_inspection_box"] = method == "Noisy"
        panel_records.append(record)
        exported_paths.append(target)

    composite_width = 4 * PANEL_PIXELS + 3 * GRID_GAP_PIXELS
    composite_height = 2 * PANEL_PIXELS + GRID_GAP_PIXELS
    composite = Image.new("RGB", (composite_width, composite_height), "white")
    for index, path in enumerate(exported_paths):
        row, col = divmod(index, 4)
        with Image.open(path) as image:
            tile = image.convert("RGB")
        x = col * (PANEL_PIXELS + GRID_GAP_PIXELS)
        y = row * (PANEL_PIXELS + GRID_GAP_PIXELS)
        composite.paste(tile, (x, y))

    composite_path = output / "figure3_2x4_no_text.png"
    composite.save(composite_path, dpi=(PNG_DPI, PNG_DPI), optimize=True)

    readme = f"""# Figure 3 Word-ready panels

The `panels` directory contains eight standalone PNG files with no embedded
letters, method names, or caption text.  Panel (a) retains the red inspection
box, matching Figure 3 in V3.docx.

Each panel is {PANEL_PIXELS} x {PANEL_PIXELS} pixels and carries approximately
{PNG_DPI:.3f} dpi metadata, so its natural Word size is
{V3_PANEL_INCHES:.6f} x {V3_PANEL_INCHES:.6f} inches (the exact V3 value:
{V3_PANEL_EMU} EMU per side).  Insert four panels per centered row, separated by
one ordinary space.  Add panel letters and method names as editable Word text.

`figure3_2x4_no_text.png` is an optional caption-free composite using the
panel spacing inferred from V3.  The eight standalone files are recommended
when the labels will be edited in Word.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")

    manifest = {
        "complete": True,
        "source_reference_docx": r"D:\Users\Chen\Desktop\V3.docx",
        "reference_figure3": {
            "layout": "two centered paragraphs, four inline square images per row",
            "panel_source_pixels": [256, 256],
            "panel_display_emu": [V3_PANEL_EMU, V3_PANEL_EMU],
            "panel_display_inches": [V3_PANEL_INCHES, V3_PANEL_INCHES],
            "cropping": False,
            "text_embedded_in_images": False,
            "first_panel_has_red_inspection_box": True,
        },
        "export": {
            "panel_pixels": [PANEL_PIXELS, PANEL_PIXELS],
            "png_dpi_requested": PNG_DPI,
            "layout_order": [record["method"] for record in panel_records],
            "grid_gap_pixels": GRID_GAP_PIXELS,
            "composite_pixels": [composite_width, composite_height],
            "all_caption_text_omitted": True,
        },
        "panels": panel_records,
        "composite": {
            "path": composite_path.name,
            "sha256": sha256(composite_path),
            "contains_text": False,
        },
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    zip_path = output / "figure3_word_ready_no_text.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in exported_paths:
            archive.write(path, path.relative_to(output).as_posix())
        archive.write(composite_path, composite_path.name)
        archive.write(output / "README.md", "README.md")
        archive.write(manifest_path, "manifest.json")

    print(
        json.dumps(
            {
                "output": str(output),
                "panel_count": len(exported_paths),
                "panel_pixels": [PANEL_PIXELS, PANEL_PIXELS],
                "word_display_inches": [V3_PANEL_INCHES, V3_PANEL_INCHES],
                "png_dpi": PNG_DPI,
                "composite": str(composite_path),
                "zip": str(zip_path),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
