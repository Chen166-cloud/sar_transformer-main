"""Generate a 3x3 contact sheet for selecting a fixed Umbra SICD test ROI.

Only display PNGs are generated.  The experiment input is later read directly
from the original complex SICD at the selected row/column coordinates.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sarpy.io.complex.converter import open_complex


def display(data: np.ndarray) -> tuple[np.ndarray, list[float]]:
    db = 20.0 * np.log10(np.maximum(np.abs(data), np.finfo(np.float32).tiny))
    low, high = (float(v) for v in np.percentile(db, (1.0, 99.7)))
    mapped = np.rint(np.clip((db - low) / (high - low), 0, 1) * 255).astype(np.uint8)
    return mapped, [low, high]


def font(size: int) -> ImageFont.ImageFont:
    for candidate in (Path(r"C:\Windows\Fonts\arial.ttf"), Path(r"C:\Windows\Fonts\calibri.ttf")):
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default(size=size)


def starts(length: int, size: int, count: int) -> list[int]:
    maximum = length - size
    if maximum < 0:
        raise ValueError(f"crop size {size} exceeds image dimension {length}")
    if count < 1:
        raise ValueError("grid count must be positive")
    fractions = (0.12, 0.50, 0.88) if count == 3 else np.linspace(0.03, 0.97, count)
    return [int(round(maximum * fraction)) for fraction in fractions]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--crop-size", type=int, default=1536)
    parser.add_argument("--thumbnail", type=int, default=384)
    parser.add_argument("--grid-rows", type=int, default=3)
    parser.add_argument("--grid-cols", type=int, default=3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    reader = open_complex(str(args.source))
    records = []
    tiles = []
    try:
        sicd = reader.get_sicds_as_tuple()[0]
        height, width = int(sicd.ImageData.NumRows), int(sicd.ImageData.NumCols)
        rows = starts(height, args.crop_size, args.grid_rows)
        cols = starts(width, args.crop_size, args.grid_cols)
        for row_index, row in enumerate(rows):
            for col_index, col in enumerate(cols):
                name = f"R{row_index + 1}C{col_index + 1}"
                complex_crop = np.asarray(
                    reader[row : row + args.crop_size, col : col + args.crop_size],
                    dtype=np.complex64,
                )
                pixels, limits = display(complex_crop)
                full_path = args.output / f"{name}_r{row}_c{col}.png"
                Image.fromarray(pixels).save(full_path, optimize=True)
                thumb = Image.fromarray(pixels).resize(
                    (args.thumbnail, args.thumbnail), Image.Resampling.LANCZOS
                )
                tile = Image.new("RGB", (args.thumbnail, args.thumbnail + 42), "white")
                tile.paste(thumb.convert("RGB"), (0, 0))
                draw = ImageDraw.Draw(tile)
                draw.text(
                    (7, args.thumbnail + 5),
                    f"{name}  row={row}, col={col}",
                    fill="black",
                    font=font(18),
                )
                tiles.append(tile)
                records.append(
                    {
                        "id": name,
                        "row_start": row,
                        "col_start": col,
                        "crop_size": args.crop_size,
                        "display_amplitude_db_percentiles_1_99p7": limits,
                        "png": str(full_path.resolve()),
                    }
                )
    finally:
        reader.close()

    gutter = 8
    tile_width, tile_height = tiles[0].size
    sheet = Image.new(
        "RGB",
        (
            args.grid_cols * tile_width + (args.grid_cols - 1) * gutter,
            args.grid_rows * tile_height + (args.grid_rows - 1) * gutter,
        ),
        "white",
    )
    for index, tile in enumerate(tiles):
        row, col = divmod(index, args.grid_cols)
        sheet.paste(tile, (col * (tile_width + gutter), row * (tile_height + gutter)))
    sheet_path = args.output / "contact_sheet.png"
    sheet.save(sheet_path, optimize=True)
    manifest = {
        "source": str(args.source.resolve()),
        "source_shape": [height, width],
        "grid_shape": [args.grid_rows, args.grid_cols],
        "selection_rule": "choose from noisy complex-input amplitude previews only",
        "candidates": records,
        "contact_sheet": str(sheet_path.resolve()),
    }
    (args.output / "candidates.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(sheet_path.resolve())


if __name__ == "__main__":
    main()
