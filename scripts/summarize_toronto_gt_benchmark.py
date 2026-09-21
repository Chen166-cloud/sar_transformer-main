"""Compute full-reference metrics and render the Toronto paired benchmark."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.io import loadmat
from skimage.metrics import structural_similarity


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(r"E:\SAR_Data\Toronto_Paired_SAR\selected")
OUTPUT_ROOT = ROOT / "output" / "toronto_gt_benchmark"
SCENES = (
    ("01_urban", "城区"),
    ("02_coast_urban_rural", "湖岸—城区—农田"),
    ("03_water_islands_farmland", "水域—岛屿—农田"),
)
METHODS = (
    "Noisy",
    "SAR-BM3D",
    "SAR2SAR",
    "SDUDNet",
    "Trans-SAR",
    "CL-SAR",
    "MuLoG-DRUNet",
)


def load_outputs(scene: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    data = DATA_ROOT / scene
    runs = OUTPUT_ROOT / scene / "runs"
    reference = np.load(data / "ground_truth.npy", allow_pickle=False).astype(np.float64)
    arrays = {
        "Noisy": np.load(data / "noisy_intensity.npy", allow_pickle=False).astype(np.float64),
        "SAR-BM3D": loadmat(runs / "sarbm3d" / "result.mat")["denoised_intensity"].astype(np.float64),
        "SAR2SAR": loadmat(runs / "sar2sar" / "result.mat")["denoised_intensity"].astype(np.float64),
        # SDUDNet wrapper preserves the official [0,1] output; restore published uint8 units.
        "SDUDNet": np.load(runs / "sdudnet" / "denoised.npy", allow_pickle=False).astype(np.float64) * 255.0,
        "Trans-SAR": np.load(runs / "transsar" / "denoised.npy", allow_pickle=False).astype(np.float64),
        "CL-SAR": np.load(runs / "cl_sar" / "denoised.npy", allow_pickle=False).astype(np.float64),
        "MuLoG-DRUNet": np.load(runs / "mulog_drunet" / "denoised.npy", allow_pickle=False).astype(np.float64),
    }
    for method, value in arrays.items():
        if value.shape != reference.shape or not np.isfinite(value).all():
            raise ValueError(f"Invalid {scene} {method} output: {value.shape}")
    return reference, arrays


def metrics(reference: np.ndarray, estimate: np.ndarray) -> dict[str, float]:
    error = estimate - reference
    mse = float(np.mean(error * error))
    return {
        "mse": mse,
        "mae": float(np.mean(np.abs(error))),
        "psnr_db": float("inf") if mse == 0 else 10.0 * math.log10(255.0**2 / mse),
        "ssim": float(structural_similarity(reference, estimate, data_range=255.0)),
        "fraction_below_0": float(np.mean(estimate < 0)),
        "fraction_above_255": float(np.mean(estimate > 255)),
        "min": float(estimate.min()),
        "max": float(estimate.max()),
    }


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (Path(r"C:\Windows\Fonts\arial.ttf"), Path(r"C:\Windows\Fonts\calibri.ttf")):
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def render_scene(scene: str, label: str, reference: np.ndarray,
                 arrays: dict[str, np.ndarray]) -> None:
    destination = OUTPUT_ROOT / scene / "figures"
    panels = destination / "panels"
    panels.mkdir(parents=True, exist_ok=True)
    ordered = [("Noisy", arrays["Noisy"]), ("Ground Truth", reference)] + [
        (name, arrays[name]) for name in METHODS if name != "Noisy"
    ]
    for index, (name, value) in enumerate(ordered):
        image = Image.fromarray(np.rint(np.clip(value, 0, 255)).astype(np.uint8))
        image.save(panels / f"{index + 1:02d}_{name.lower().replace(' ', '_')}.png")

    cell, title_h = 512, 42
    canvas = Image.new("L", (4 * cell, 2 * (cell + title_h)), 255)
    draw = ImageDraw.Draw(canvas)
    title_font = font(26)
    for index, (name, value) in enumerate(ordered):
        row, col = divmod(index, 4)
        x, y = col * cell, row * (cell + title_h)
        image = Image.fromarray(np.rint(np.clip(value, 0, 255)).astype(np.uint8))
        canvas.paste(image, (x, y))
        text = f"({chr(97 + index)}) {name}"
        bbox = draw.textbbox((0, 0), text, font=title_font)
        draw.text((x + (cell - (bbox[2] - bbox[0])) // 2, y + cell + 5), text, fill=0, font=title_font)
    canvas.save(destination / "comparison_8panel.png", dpi=(600, 600))


def render_average_chart(rows: list[dict[str, object]]) -> None:
    denoised = [method for method in METHODS if method != "Noisy"]
    psnr = [np.mean([float(r["psnr_db"]) for r in rows if r["method"] == method]) for method in denoised]
    ssim = [np.mean([float(r["ssim"]) for r in rows if r["method"] == method]) for method in denoised]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    colors = plt.cm.tab10(np.arange(len(denoised)))
    axes[0].bar(denoised, psnr, color=colors)
    axes[0].set_ylabel("PSNR (dB)")
    axes[0].set_title("Mean over three paired scenes")
    axes[1].bar(denoised, ssim, color=colors)
    axes[1].set_ylabel("SSIM")
    axes[1].set_title("Mean over three paired scenes")
    for axis in axes:
        axis.tick_params(axis="x", rotation=35)
        axis.grid(axis="y", alpha=0.25)
    figure.savefig(OUTPUT_ROOT / "average_metrics.png", dpi=300)
    plt.close(figure)


def main() -> None:
    rows: list[dict[str, object]] = []
    for scene, label in SCENES:
        reference, arrays = load_outputs(scene)
        render_scene(scene, label, reference, arrays)
        for method in METHODS:
            rows.append({"scene": scene, "scene_label": label, "method": method,
                         **metrics(reference, arrays[method])})

    with (OUTPUT_ROOT / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUTPUT_ROOT / "metrics.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    render_average_chart(rows)
    averages = {}
    for method in METHODS:
        selected = [r for r in rows if r["method"] == method]
        averages[method] = {
            "psnr_db": float(np.mean([float(r["psnr_db"]) for r in selected])),
            "ssim": float(np.mean([float(r["ssim"]) for r in selected])),
            "mse": float(np.mean([float(r["mse"]) for r in selected])),
        }
    (OUTPUT_ROOT / "averages.json").write_text(
        json.dumps(averages, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(averages, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
