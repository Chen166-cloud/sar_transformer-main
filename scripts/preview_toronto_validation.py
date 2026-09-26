"""Create paired contact sheets and input-only statistics for Toronto validation."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(r"E:\SAR_Data\Toronto_Paired_SAR\validation_full")
OUTPUT = Path(r"D:\research\sar_transformer-main\output\toronto_showcase_20260926\selection")
SHORTLIST = (
    "5120_3072.tiff",
    "5120_8704.tiff",
    "5120_22528.tiff",
    "5632_5120.tiff",
    "5632_6144.tiff",
    "5632_20992.tiff",
    "5632_25088.tiff",
)


def read(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        value = np.asarray(image)
    if value.ndim == 3:
        if value.shape[2] not in (3, 4) or not np.array_equal(value[..., 0], value[..., 1]):
            raise ValueError(f"Unexpected channels: {path} {value.shape}")
        value = value[..., 0]
    if value.shape != (512, 512) or value.dtype != np.uint8:
        raise ValueError(f"Unexpected image: {path} {value.shape} {value.dtype}")
    return value


def natural_key(path: Path) -> tuple[int, int]:
    row, col = path.stem.split("_")
    return int(row), int(col)


def main() -> None:
    noisy_files = sorted((ROOT / "Noisy_val").glob("*.tiff"), key=natural_key)
    if len(noisy_files) != 100:
        raise ValueError(f"Expected 100 noisy files, got {len(noisy_files)}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 18)
    rows = []
    for sheet_index in range(4):
        canvas = Image.new("RGB", (1530, 935), "white")
        draw = ImageDraw.Draw(canvas)
        for local_index, noisy_path in enumerate(noisy_files[sheet_index * 25 : (sheet_index + 1) * 25]):
            reference_path = ROOT / "GTruth_val" / noisy_path.name
            noisy, reference = read(noisy_path), read(reference_path)
            if noisy.shape != reference.shape:
                raise ValueError(f"Pair mismatch: {noisy_path.name}")
            row, col = divmod(local_index, 5)
            left, top = col * 306, row * 187
            draw.text((left + 8, top + 2), noisy_path.stem, fill="black", font=font)
            canvas.paste(Image.fromarray(noisy).resize((140, 140)), (left + 8, top + 32))
            canvas.paste(Image.fromarray(reference).resize((140, 140)), (left + 156, top + 32))
            gy, gx = np.gradient(noisy.astype(np.float32))
            rows.append({
                "filename": noisy_path.name,
                "mean": float(noisy.mean()),
                "std": float(noisy.std()),
                "fraction_zero": float(np.mean(noisy == 0)),
                "fraction_255": float(np.mean(noisy == 255)),
                "fraction_dark_0_20": float(np.mean(noisy <= 20)),
                "gradient_mean": float(np.mean(np.hypot(gx, gy))),
                "reference_mean": float(reference.mean()),
            })
        canvas.save(OUTPUT / f"contact_sheet_{sheet_index + 1}.png")
    with (OUTPUT / "input_statistics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for name in SHORTLIST:
        noisy = read(ROOT / "Noisy_val" / name)
        reference = read(ROOT / "GTruth_val" / name)
        comparison = Image.new("L", (1024, 512))
        comparison.paste(Image.fromarray(noisy), (0, 0))
        comparison.paste(Image.fromarray(reference), (512, 0))
        comparison.save(OUTPUT / f"shortlist_{Path(name).stem}.png")
    print(f"Created four contact sheets and input statistics in {OUTPUT}")


if __name__ == "__main__":
    main()
