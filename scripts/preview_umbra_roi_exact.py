"""Preview exact SICD ROIs at native size and at the final Figure-3 panel size."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sarpy.io.complex.converter import open_complex


def display_font(size: int) -> ImageFont.ImageFont:
    for path in (Path(r"C:\Windows\Fonts\arial.ttf"), Path(r"C:\Windows\Fonts\tahoma.ttf")):
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size=size)


def db_preview(complex_roi: np.ndarray, percentiles: tuple[float, float]) -> tuple[Image.Image, list[float]]:
    amplitude = np.abs(np.asarray(complex_roi, dtype=np.complex64))
    if not np.isfinite(amplitude).all() or not np.any(amplitude > 0):
        raise ValueError("invalid SICD amplitude in ROI")
    db = 20.0 * np.log10(np.maximum(amplitude, np.finfo(np.float32).tiny))
    low, high = (float(value) for value in np.percentile(db, percentiles))
    if not low < high:
        raise ValueError("ROI display range is empty")
    pixels = np.rint(np.clip((db - low) / (high - low), 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(pixels, mode="L"), [low, high]


def contact_sheet(panels: list[tuple[Image.Image, str, str]], panel_size: int) -> Image.Image:
    columns = min(3, len(panels))
    rows = math.ceil(len(panels) / columns)
    gap, margin, label_height = 12, 12, 48
    width = 2 * margin + columns * panel_size + (columns - 1) * gap
    height = 2 * margin + rows * (panel_size + label_height) + (rows - 1) * gap
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    selected_font = display_font(17)
    for index, (panel, line1, line2) in enumerate(panels):
        row, col = divmod(index, columns)
        x = margin + col * (panel_size + gap)
        y = margin + row * (panel_size + label_height + gap)
        sheet.paste(panel.convert("RGB"), (x, y))
        draw.text((x + 2, y + panel_size + 4), line1, fill="black", font=selected_font)
        draw.text((x + 2, y + panel_size + 25), line2, fill="black", font=selected_font)
    return sheet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Umbra SICD complex NITF")
    parser.add_argument("--output", type=Path, required=True, help="new or empty preview directory")
    parser.add_argument(
        "--roi", type=int, nargs=2, action="append", required=True, metavar=("ROW", "COL"),
        help="zero-based top-left pixel; repeat to compare several exact ROIs",
    )
    parser.add_argument("--size", type=int, default=1024, help="square ROI side in pixels")
    parser.add_argument("--panel-size", type=int, default=502, help="Figure-3 preview side in pixels")
    parser.add_argument(
        "--display-percentiles", type=float, nargs=2, default=(2.0, 99.0),
        metavar=("LOW", "HIGH"), help="noisy dB percentile window per ROI (default: 2 99)",
    )
    args = parser.parse_args()
    if not args.source.is_file():
        parser.error(f"SICD source does not exist: {args.source}")
    if args.size <= 0 or args.panel_size <= 0:
        parser.error("--size and --panel-size must be positive")
    lower_percentile, upper_percentile = args.display_percentiles
    if not (0 <= lower_percentile < upper_percentile <= 100):
        parser.error("--display-percentiles requires 0 <= LOW < HIGH <= 100")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f"output must be new or empty: {args.output}")

    reader = open_complex(str(args.source.resolve()))
    try:
        sicd = reader.get_sicds_as_tuple()[0]
        height, width = int(sicd.ImageData.NumRows), int(sicd.ImageData.NumCols)
        for row, col in args.roi:
            if row < 0 or col < 0 or row + args.size > height or col + args.size > width:
                parser.error(
                    f"ROI ({row}, {col}, size={args.size}) exceeds SICD bounds {height}x{width}"
                )

        args.output.mkdir(parents=True, exist_ok=True)
        panel_records: list[tuple[Image.Image, str, str]] = []
        roi_records: list[dict[str, object]] = []
        percentiles = (float(lower_percentile), float(upper_percentile))
        for index, (row, col) in enumerate(args.roi, start=1):
            complex_roi = reader[row : row + args.size, col : col + args.size]
            full, limits = db_preview(complex_roi, percentiles)
            panel = full.resize((args.panel_size, args.panel_size), Image.Resampling.LANCZOS)
            stem = f"roi_{index:02d}_r{row}_c{col}"
            full_name, panel_name = f"{stem}_full.png", f"{stem}_panel.png"
            full.save(args.output / full_name, dpi=(600, 600), optimize=True)
            panel.save(args.output / panel_name, dpi=(600, 600), optimize=True)
            panel_records.append(
                (panel, f"#{index:02d} row={row}, col={col}", f"dB [{limits[0]:.1f}, {limits[1]:.1f}]")
            )
            roi_records.append(
                {
                    "index": index,
                    "row_start": row,
                    "col_start": col,
                    "size": args.size,
                    "display_db_limits": limits,
                    "full_png": full_name,
                    "panel_png": panel_name,
                }
            )
    finally:
        reader.close()

    sheet_name = "contact_sheet.png"
    contact_sheet(panel_records, args.panel_size).save(args.output / sheet_name, dpi=(600, 600), optimize=True)
    manifest = {
        "source": str(args.source.resolve()),
        "sicd_shape": [height, width],
        "input_domain": "SICD complex; display = 20*log10(abs(S))",
        "display_percentiles": list(percentiles),
        "panel_size_pixels": args.panel_size,
        "contact_sheet": sheet_name,
        "rois": roi_records,
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(manifest_path.resolve())


if __name__ == "__main__":
    main()
