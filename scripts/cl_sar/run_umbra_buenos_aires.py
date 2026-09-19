"""Run the pinned official CL-SAR model on a native Umbra SICD city crop.

Scientific outputs stay in linear intensity.  Figure panels use one fixed dB
mapping derived from the noisy input, so the noisy and restored renderings are
visually comparable.  No clean reference or full-reference metric is implied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import sarpy
import torch
from PIL import Image, ImageDraw, ImageFont
from sarpy.io.complex.converter import open_complex
from scipy.io import savemat


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
import run_local as clsar  # noqa: E402


SOURCE = Path(
    r"E:\SAR_Data\Umbra\Buenos_Aires_20250131"
    r"\2025-01-31-14-10-46_UMBRA-08_SICD.nitf"
)
SOURCE_MANIFEST = SOURCE.parent / "download_manifest.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "cl_sar_umbra_buenos_aires" / "figure3_clsar"
ROW_START = 7456
COL_START = 17256
SIZE = 1024
DISPLAY_PERCENTILES = (1.0, 99.7)
DETAIL_URBAN = (500, 160, 512)  # x, y, square size; input-selected
DETAIL_RAIL = (20, 480, 512)
# Input-selected 128 x 128 low-texture window.  It is used only as the red
# inspection box and an ENL diagnostic; model outputs never influence its
# location.
INSPECTION_BOX = (192, 160, 128, 128)
FIGURE_DPI = 600
WORD_COLUMN_INCHES = 3.485


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def intensity_db(array: np.ndarray) -> np.ndarray:
    floor = np.finfo(np.float32).tiny
    return 10.0 * np.log10(np.maximum(array, floor))


def map_db(array_db: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.rint(np.clip((array_db - low) / (high - low), 0.0, 1.0) * 255.0).astype(np.uint8)


def save_png(path: Path, pixels: np.ndarray) -> None:
    Image.fromarray(pixels).save(path, dpi=(FIGURE_DPI, FIGURE_DPI), optimize=True)


def crop(array: np.ndarray, spec: tuple[int, int, int]) -> np.ndarray:
    x, y, size = spec
    return array[y : y + size, x : x + size]


def font(size: int) -> ImageFont.FreeTypeFont:
    candidates = (Path(r"C:\Windows\Fonts\times.ttf"), Path(r"C:\Windows\Fonts\timesnewroman.ttf"))
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default(size=size)


def centered(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str,
             selected_font: ImageFont.ImageFont, fill: str = "black") -> None:
    x, y = xy
    bounds = draw.textbbox((0, 0), text, font=selected_font)
    draw.text((x - (bounds[2] - bounds[0]) / 2, y), text, font=selected_font, fill=fill)


def make_diagnostic_figure(output: Path, panels: list[tuple[str, str, str]]) -> tuple[int, int]:
    """Render a Figure-3-style 2x4 layout at the manuscript column width."""
    width = round(WORD_COLUMN_INCHES * FIGURE_DPI)
    panel = round(0.24 * width)
    gap = round(0.01 * width)
    side = max(0, (width - (4 * panel + 3 * gap)) // 2)
    label_font = font(round(8 * FIGURE_DPI / 72))
    label_step = round(9 * FIGURE_DPI / 72)
    label_block = label_step * 2 + round(FIGURE_DPI / 72)
    row_gap = round(3 * FIGURE_DPI / 72)
    height = 2 * (panel + label_block) + row_gap
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (name, letter, label) in enumerate(panels):
        row, col = divmod(index, 4)
        x = side + col * (panel + gap)
        y = row * (panel + label_block + row_gap)
        with Image.open(output / name) as image:
            image = image.convert("RGB").resize((panel, panel), Image.Resampling.LANCZOS)
            canvas.paste(image, (x, y))
        centered(draw, (x + panel // 2, y + panel + 2), letter, label_font)
        centered(draw, (x + panel // 2, y + panel + label_step), label, label_font)
    path = output / "figure3_clsar_diagnostic_600dpi.png"
    canvas.save(path, dpi=(FIGURE_DPI, FIGURE_DPI), optimize=True)
    canvas.save(output / "figure3_clsar_diagnostic_600dpi.pdf", "PDF", resolution=FIGURE_DPI)
    return canvas.size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError(f"Output must be new or empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    reader = open_complex(str(args.source))
    try:
        sicd = reader.get_sicds_as_tuple()[0]
        complex_roi = np.ascontiguousarray(
            reader[ROW_START : ROW_START + SIZE, COL_START : COL_START + SIZE],
            dtype=np.complex64,
        )
        center_geo = sicd.project_image_to_ground_geo(
            [[ROW_START + SIZE / 2, COL_START + SIZE / 2]]
        )[0]
        sicd_info = {
            "namespace": sicd._xml_ns["default"],
            "pixel_type": sicd.ImageData.PixelType,
            "full_shape": [int(sicd.ImageData.NumRows), int(sicd.ImageData.NumCols)],
            "row_sample_spacing_m": float(sicd.Grid.Row.SS),
            "col_sample_spacing_m": float(sicd.Grid.Col.SS),
            "row_impulse_response_width_m": float(sicd.Grid.Row.ImpRespWid),
            "col_impulse_response_width_m": float(sicd.Grid.Col.ImpRespWid),
            "polarization": str(sicd.ImageFormation.TxRcvPolarizationProc),
            "radar_mode": str(sicd.CollectionInfo.RadarMode.ModeType),
        }
    finally:
        reader.close()

    noisy = np.ascontiguousarray(
        complex_roi.real * complex_roi.real + complex_roi.imag * complex_roi.imag,
        dtype=np.float32,
    )
    normalized, normalization = clsar.prepare_input(noisy, "intensity", scale=1.0)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    clsar.configure_runtime()
    model, model_provenance = clsar.build_model(device)
    raw, seconds = clsar.predict(model, normalized, device)
    denoised = clsar.restore_output(raw, normalization)
    ratio = np.divide(noisy, np.maximum(denoised, np.finfo(np.float32).tiny), dtype=np.float32)

    np.save(args.output / "sicd_complex_roi.npy", complex_roi, allow_pickle=False)
    np.save(args.output / "noisy_intensity.npy", noisy, allow_pickle=False)
    np.save(args.output / "CL-SAR_intensity.npy", denoised, allow_pickle=False)
    np.save(args.output / "CL-SAR_network_raw.npy", raw, allow_pickle=False)
    np.save(args.output / "CL-SAR_ratio.npy", ratio, allow_pickle=False)
    savemat(
        args.output / "CL-SAR_result.mat",
        {
            "noisy": noisy,
            "denoised": denoised,
            "ratio": ratio,
            "complex_roi": complex_roi,
            "input_domain": "intensity",
            "output_domain": "intensity",
        },
        do_compression=True,
    )

    noisy_db = intensity_db(noisy)
    denoised_db = intensity_db(denoised)
    low, high = (float(value) for value in np.percentile(noisy_db, DISPLAY_PERCENTILES))
    noisy_panel = map_db(noisy_db, low, high)
    denoised_panel = map_db(denoised_db, low, high)
    ratio_db = noisy_db - denoised_db
    ratio_limit = float(np.percentile(np.abs(ratio_db), 99.0))
    ratio_panel = map_db(ratio_db, -ratio_limit, ratio_limit)

    save_png(args.output / "noisy_no_box.png", noisy_panel)
    boxed = Image.fromarray(noisy_panel).convert("RGB")
    box_draw = ImageDraw.Draw(boxed)
    x, y, w, h = INSPECTION_BOX
    box_draw.rectangle((x, y, x + w - 1, y + h - 1), outline=(230, 25, 35), width=5)
    boxed.save(args.output / "noisy.png", dpi=(FIGURE_DPI, FIGURE_DPI), optimize=True)
    save_png(args.output / "CL-SAR.png", denoised_panel)
    save_png(args.output / "CL-SAR_ratio.png", ratio_panel)
    for stem, panel in (("noisy", noisy_panel), ("CL-SAR", denoised_panel), ("ratio", ratio_panel)):
        save_png(args.output / f"{stem}_urban_detail.png", crop(panel, DETAIL_URBAN))
        save_png(args.output / f"{stem}_rail_detail.png", crop(panel, DETAIL_RAIL))

    diagnostic_size = make_diagnostic_figure(
        args.output,
        [
            ("noisy.png", "(a)", "Noisy"),
            ("CL-SAR.png", "(b)", "CL-SAR"),
            ("noisy_urban_detail.png", "(c)", "Noisy detail"),
            ("CL-SAR_urban_detail.png", "(d)", "CL-SAR detail"),
            ("noisy_rail_detail.png", "(e)", "Noisy rail"),
            ("CL-SAR_rail_detail.png", "(f)", "CL-SAR rail"),
            ("CL-SAR_ratio.png", "(g)", "CL-SAR ratio"),
            ("ratio_urban_detail.png", "(h)", "Ratio detail"),
        ],
    )

    manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8")) if SOURCE_MANIFEST.exists() else None
    box_slice = np.s_[y : y + h, x : x + w]
    def enl(array: np.ndarray) -> float:
        values = array[box_slice].astype(np.float64)
        return float(values.mean() ** 2 / values.var(ddof=1))

    output_files = [
        "sicd_complex_roi.npy", "noisy_intensity.npy", "CL-SAR_intensity.npy",
        "CL-SAR_network_raw.npy", "CL-SAR_ratio.npy", "CL-SAR_result.mat",
        "noisy.png", "noisy_no_box.png", "CL-SAR.png", "CL-SAR_ratio.png",
        "figure3_clsar_diagnostic_600dpi.png", "figure3_clsar_diagnostic_600dpi.pdf",
    ]
    report = {
        "method": "CL-SAR",
        "scope": "official pretrained inference on a native Umbra SICD intensity crop",
        "source": {
            "path": str(args.source.resolve()),
            "bytes": args.source.stat().st_size,
            "download_manifest": manifest,
            "sicd": sicd_info,
        },
        "roi": {
            "row_start": ROW_START,
            "col_start": COL_START,
            "height": SIZE,
            "width": SIZE,
            "center_latitude_deg": float(center_geo[0]),
            "center_longitude_deg": float(center_geo[1]),
            "center_hae_m": float(center_geo[2]),
            "selection": "chosen from the noisy input only; urban buildings, rail lines, and industrial structures",
        },
        "scientific_domain": {
            "complex_to_intensity": "I = real(S)^2 + imag(S)^2",
            "model_input_domain": "linear intensity",
            "model_output_domain": "linear intensity",
            "normalization": normalization,
            "no_radiometric_calibration_added": True,
            "no_spatial_resampling_before_inference": True,
        },
        "model": model_provenance,
        "runtime": {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "torch": torch.__version__,
            "sarpy": sarpy.__version__,
            "forward_seconds": seconds,
            "direct_full_roi_inference": True,
            "tiling": False,
        },
        "author_postprocessing": {
            "normalized_amplitude_clamps": [[-20.0, 20.0], [0.0, 1.0]],
            "fraction_outside_0_1_before_author_clamp": float(np.mean((raw < 0) | (raw > 1))),
        },
        "display": {
            "panel_shape": [SIZE, SIZE],
            "panel_png": "8-bit grayscale, except noisy.png is RGB solely for the red box",
            "mapping": "10*log10(linear intensity), then one shared affine mapping for noisy and CL-SAR",
            "source_percentiles": list(DISPLAY_PERCENTILES),
            "shared_db_limits": [low, high],
            "ratio_definition_db": "10*log10(noisy / CL-SAR)",
            "ratio_symmetric_limit_db": ratio_limit,
            "figure_dpi": FIGURE_DPI,
            "diagnostic_figure_pixels": list(diagnostic_size),
            "word_target_panel_size_mm": 21.06,
            "inspection_box_xywh": list(INSPECTION_BOX),
        },
        "diagnostics": {
            "inspection_enl_noisy": enl(noisy),
            "inspection_enl_clsar": enl(denoised),
            "mean_intensity_noisy": float(noisy.mean()),
            "mean_intensity_clsar": float(denoised.mean()),
            "no_clean_reference": True,
            "psnr_ssim_not_reported": True,
        },
        "outputs": {
            name: {"bytes": (args.output / name).stat().st_size, "sha256": sha256(args.output / name)}
            for name in output_files
        },
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (args.output / "run.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    caption = (
        "CL-SAR despeckling of a real single-look Umbra X-band VV SICD urban scene in Buenos Aires. "
        "All intensity panels use the same fixed dB display range. Ratio panels show "
        "10 log10(noisy/CL-SAR) for diagnosis; no clean reference is available.\n"
    )
    (args.output / "caption.txt").write_text(caption, encoding="utf-8")
    readme = f"""# Umbra Buenos Aires CL-SAR result

This directory contains a direct, non-tiled CL-SAR inference on a native
{SIZE} x {SIZE} crop from the downloaded Umbra SICD complex image. The model
input and scientific output are linear intensity. PNGs are display products.

Figure 3 compatible panels:

- `noisy.png`: source panel with the red input-selected inspection box.
- `noisy_no_box.png`: source panel without annotation.
- `CL-SAR.png`: CL-SAR restoration; 1024 x 1024, grayscale, no embedded label.
- `CL-SAR_ratio.png`: ratio diagnostic, not a restoration.

The noisy and CL-SAR panels share exactly the same dB limits ({low:.6f},
{high:.6f}). Do not apply a separate auto-contrast to either panel. At the
current V3 Figure 3 physical panel size of 21.06 mm, each 1024 px source panel
provides about 1235 effective ppi, compared with about 309 ppi for the old
256 px panels.

`figure3_clsar_diagnostic_600dpi.png` follows V3's 2 x 4, Times New Roman 8 pt
panel style, but it is a CL-SAR diagnostic proof rather than a fabricated
eight-method comparison. Scientific arrays and complete provenance are in
`CL-SAR_result.mat`, the NPY files, and `run.json`.
"""
    (args.output / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "device": str(device),
        "forward_seconds": seconds,
        "panel": str((args.output / "CL-SAR.png").resolve()),
        "diagnostic": str((args.output / "figure3_clsar_diagnostic_600dpi.png").resolve()),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
