"""Create display-only previews for candidate Umbra SICD city crops.

The source pixels remain complex64.  Preview PNGs use a logarithmic amplitude
mapping only for visual ROI selection; no preview is used as CL-SAR input.
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from sarpy.io.complex.converter import open_complex


ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(
    r"E:\SAR_Data\Umbra\Buenos_Aires_20250131"
    r"\2025-01-31-14-10-46_UMBRA-08_SICD.nitf"
)
OUTPUT = ROOT / "output" / "cl_sar_umbra_buenos_aires" / "roi_candidates"

# (name, row_start, col_start, size).  Locations were selected from a
# stride-32 full-scene SICD overview and intentionally cover different urban
# structures rather than a homogeneous speckle field.
CANDIDATES = (
    ("A_central_roundabout", 9000, 10600, 1536),
    ("B_eastern_industrial", 7200, 17000, 1536),
    ("C_southern_corridor", 12800, 10200, 1536),
    ("D_northwest_grid", 4800, 6200, 1536),
)


def display_amplitude(data: np.ndarray) -> tuple[np.ndarray, float, float]:
    db = 20.0 * np.log10(np.maximum(np.abs(data), np.finfo(np.float32).tiny))
    lo, hi = np.percentile(db, (1.0, 99.7))
    mapped = np.rint(np.clip((db - lo) / (hi - lo), 0.0, 1.0) * 255.0).astype(np.uint8)
    return mapped, float(lo), float(hi)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    reader = open_complex(str(SOURCE))
    tiles = []
    try:
        for name, row, col, size in CANDIDATES:
            complex_data = reader[row : row + size, col : col + size]
            image, lo, hi = display_amplitude(complex_data)
            Image.fromarray(image).save(OUTPUT / f"{name}.png")
            thumb = Image.fromarray(image).resize((384, 384), Image.Resampling.LANCZOS)
            tile = Image.new("L", (384, 416), "white")
            tile.paste(thumb, (0, 0))
            draw = ImageDraw.Draw(tile)
            draw.text((8, 394), f"{name}: r={row}, c={col}; {lo:.1f}..{hi:.1f} dB", fill="black")
            tiles.append(tile.convert("RGB"))
    finally:
        reader.close()
    contact = Image.new("RGB", (768, 832), "white")
    for index, tile in enumerate(tiles):
        contact.paste(tile, ((index % 2) * 384, (index // 2) * 416))
    contact.save(OUTPUT / "contact_sheet.png")
    print(OUTPUT / "contact_sheet.png")


if __name__ == "__main__":
    main()
