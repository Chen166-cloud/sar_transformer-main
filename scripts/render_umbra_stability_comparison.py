"""Render one completed Umbra stability scene in the frozen Figure-3 layout."""

from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import json
import math
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.io import loadmat, savemat


NAMES = ["Noisy", "SAR-BM3D", "SAR2SAR", "SDUDNet", "Trans-SAR", "CL-SAR", "MERLIN", "MuLoG-DRUNet"]
STEMS = ["a_noisy", "b_sar-bm3d", "c_sar2sar", "d_sdudnet", "e_trans-sar", "f_cl-sar", "g_merlin", "h_mulog-drunet"]
DPI = 600
WORD_COLUMN_INCHES = 3.485


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def intensity_db(value: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(value.astype(np.float64), np.finfo(np.float32).tiny))


def map_range(value: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.rint(np.clip((value - low) / (high - low), 0, 1) * 255).astype(np.uint8)


def array(value: np.ndarray, name: str) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=np.float32)
    if result.shape != (1024, 1024) or not np.isfinite(result).all() or np.any(result < 0):
        raise ValueError(f"invalid {name} array: shape={result.shape}")
    return result


def mat_field(path: Path, field: str) -> np.ndarray:
    payload = loadmat(path, variable_names=[field])
    if field not in payload:
        raise KeyError(f"{path} has no field {field}")
    return payload[field]


def font(size: int) -> ImageFont.ImageFont:
    for path in (Path(r"C:\Windows\Fonts\times.ttf"), Path(r"C:\Windows\Fonts\timesnewroman.ttf")):
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size=size)


def centered(draw: ImageDraw.ImageDraw, x: int, y: int, label: str, selected_font: ImageFont.ImageFont) -> None:
    box = draw.textbbox((0, 0), label, font=selected_font)
    draw.text((x - (box[2] - box[0]) / 2, y), label, fill="black", font=selected_font)


def figure(paths: list[Path], labels: list[str], target: Path) -> list[int]:
    width = round(WORD_COLUMN_INCHES * DPI)
    panel = round(0.24 * width)
    gap = round(0.01 * width)
    side = max(0, (width - 4 * panel - 3 * gap) // 2)
    selected_font = font(round(8 * DPI / 72))
    label_step = round(9 * DPI / 72)
    label_block = 2 * label_step + round(DPI / 72)
    row_gap = round(3 * DPI / 72)
    height = 2 * (panel + label_block) + row_gap
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (path, label) in enumerate(zip(paths, labels, strict=True)):
        row, col = divmod(index, 4)
        x = side + col * (panel + gap)
        y = row * (panel + label_block + row_gap)
        with Image.open(path) as image:
            canvas.paste(image.convert("RGB").resize((panel, panel), Image.Resampling.LANCZOS), (x, y))
        centered(draw, x + panel // 2, y + panel + 2, f"({'abcdefgh'[index]})", selected_font)
        centered(draw, x + panel // 2, y + panel + label_step, label, selected_font)
    canvas.save(target, dpi=(DPI, DPI), optimize=True)
    return list(canvas.size)


def stats(value: np.ndarray) -> dict[str, object]:
    x = value.astype(np.float64)
    return {
        "min": float(x.min()), "median": float(np.median(x)), "mean": float(x.mean()),
        "p01": float(np.percentile(x, 1)), "p997": float(np.percentile(x, 99.7)),
        "max": float(x.max()), "finite": bool(np.isfinite(x).all()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roi", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--scene-name", required=True)
    parser.add_argument(
        "--display-percentiles",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        help="optional noisy-intensity dB percentiles for the shared display window",
    )
    args = parser.parse_args()
    if args.display_percentiles is not None:
        lower_percentile, upper_percentile = args.display_percentiles
        if not (0 <= lower_percentile < upper_percentile <= 100):
            parser.error("--display-percentiles requires 0 <= LOW < HIGH <= 100")
    roi, experiment = args.roi.resolve(), args.experiment.resolve()
    runs, output = experiment / "runs", experiment / "figure3_style"
    panels, ratios = output / "panels", output / "ratios"
    panels.mkdir(parents=True, exist_ok=True)
    ratios.mkdir(parents=True, exist_ok=True)
    roi_report = json.loads((roi / "run.json").read_text(encoding="utf-8"))
    adapter_low, adapter_high = (float(v) for v in roi_report["display"]["shared_db_limits"])

    sources = {
        "Noisy": roi / "noisy_intensity.npy",
        "SAR-BM3D": runs / "sarbm3d" / "result.mat",
        "SAR2SAR": runs / "sar2sar" / "result.mat",
        "SDUDNet": runs / "sdudnet_db_display" / "denoised.npy",
        "Trans-SAR": runs / "transsar" / "denoised.npy",
        "CL-SAR": runs / "cl_sar" / "denoised.npy",
        "MERLIN": runs / "merlin_stride64_weighted" / "denoised.npy",
        "MuLoG-DRUNet": runs / "mulog_drunet" / "denoised.npy",
    }
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing method outputs:\n" + "\n".join(missing))
    sdud_coordinate = array(np.load(sources["SDUDNet"], allow_pickle=False), "SDUDNet coordinate")
    sdud_db = adapter_low + np.clip(sdud_coordinate.astype(np.float64), 0, 1) * (adapter_high - adapter_low)
    values = {
        "Noisy": array(np.load(sources["Noisy"], allow_pickle=False), "Noisy"),
        "SAR-BM3D": array(mat_field(sources["SAR-BM3D"], "denoised_intensity"), "SAR-BM3D"),
        "SAR2SAR": array(mat_field(sources["SAR2SAR"], "denoised_intensity"), "SAR2SAR"),
        "SDUDNet": array(np.power(10.0, sdud_db / 10.0), "SDUDNet"),
        "Trans-SAR": array(np.load(sources["Trans-SAR"], allow_pickle=False), "Trans-SAR"),
        "CL-SAR": array(np.load(sources["CL-SAR"], allow_pickle=False), "CL-SAR"),
        "MERLIN": array(np.load(sources["MERLIN"], allow_pickle=False), "MERLIN"),
        "MuLoG-DRUNet": array(np.load(sources["MuLoG-DRUNet"], allow_pickle=False), "MuLoG-DRUNet"),
    }
    if args.display_percentiles is None:
        low, high = adapter_low, adapter_high
        display_percentiles = [1.0, 99.7]
    else:
        display_percentiles = [float(v) for v in args.display_percentiles]
        low, high = (
            float(v) for v in np.percentile(intensity_db(values["Noisy"]), display_percentiles)
        )
    merlin_shift = float(np.median(intensity_db(values["MERLIN"])) - np.median(intensity_db(values["Noisy"])))
    paths, strict_paths = [], []
    for name, stem in zip(NAMES, STEMS, strict=True):
        db = intensity_db(values[name])
        strict_pixels = map_range(db, low, high)
        strict_path = panels / f"{stem}_shared-db.png" if name == "MERLIN" else panels / f"{stem}.png"
        Image.fromarray(strict_pixels).save(strict_path, dpi=(DPI, DPI), optimize=True)
        path = panels / f"{stem}.png"
        pixels = map_range(db, low + merlin_shift, high + merlin_shift) if name == "MERLIN" else strict_pixels
        Image.fromarray(pixels).save(path, dpi=(DPI, DPI), optimize=True)
        paths.append(path)
        strict_paths.append(strict_path)

    labels = NAMES.copy()
    labels[6] = "MERLIN†"
    figure_name = "figure3_style_7methods_600dpi.png"
    figure_size = figure(paths, labels, output / figure_name)
    figure(strict_paths, NAMES, output / "figure3_style_7methods_strict_shared_db_600dpi.png")

    noisy = values["Noisy"]
    ratio_records = {}
    all_ratio_db = []
    for name in NAMES[1:]:
        ratio_db = intensity_db(noisy) - intensity_db(values[name])
        all_ratio_db.append(np.abs(ratio_db).ravel())
    ratio_limit = float(np.percentile(np.concatenate(all_ratio_db), 99.0))
    for name, stem in zip(NAMES[1:], STEMS[1:], strict=True):
        ratio_db = (intensity_db(noisy) - intensity_db(values[name])).astype(np.float32)
        npy = ratios / f"ratio_{stem[2:]}_db_noisy-over-output.npy"
        png = ratios / f"ratio_{stem[2:]}_db_noisy-over-output.png"
        np.save(npy, ratio_db, allow_pickle=False)
        Image.fromarray(map_range(ratio_db, -ratio_limit, ratio_limit)).save(png, dpi=(DPI, DPI), optimize=True)
        ratio_records[name] = {"npy": str(npy.relative_to(output)), "png": str(png.relative_to(output))}

    savemat(output / "scientific_arrays.mat", {name.replace("-", "_"): value for name, value in values.items()}, do_compression=True)
    manifest = {
        "complete": True,
        "scene": args.scene_name,
        "roi": roi_report["roi"],
        "methods": {name: {"source": str(sources[name]), "statistics": stats(values[name])} for name in NAMES},
        "display": {
            "shared_db_limits": [low, high],
            "shared_db_percentiles_from_noisy": display_percentiles,
            "sdudnet_adapter_db_limits": [adapter_low, adapter_high],
            "merlin_display_only_median_shift_db": merlin_shift,
            "figure_pixels": figure_size,
            "figure_dpi": DPI,
        },
        "ratios": {"definition": "10*log10(noisy/output)", "shared_symmetric_limit_db": ratio_limit, "files": ratio_records},
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    files = [path for path in output.rglob("*") if path.is_file() and path.name != "manifest.json"]
    manifest["files"] = {str(path.relative_to(output)): {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in files}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(output / figure_name)


if __name__ == "__main__":
    main()
