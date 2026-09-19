"""Render the fixed Umbra Buenos Aires ROI as a Figure-3-style comparison.

All method panels are derived from preserved floating-point outputs.  The
default display uses one absolute dB mapping fixed from the noisy input.  At
the user's request, the primary figure applies a disclosed display-only,
median-aligned dB-window shift to MERLIN and retains a strict shared-dB figure
for audit.

The last two panels use independent official external baselines: MERLIN on
the complex Spotlight data and MuLoG-DRUNet on single-look intensity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.io import loadmat, savemat


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CL_SAR_ROOT = PROJECT_ROOT / "output" / "cl_sar_umbra_buenos_aires" / "figure3_clsar_final"
EXPERIMENT_ROOT = PROJECT_ROOT / "output" / "umbra_buenos_aires_multimethod_1024"
RUNS_ROOT = EXPERIMENT_ROOT / "runs"
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "figure3"
FIGURE_DPI = 600
WORD_COLUMN_INCHES = 3.485
PANEL_SHAPE = (1024, 1024)
INSPECTION_BOX = (192, 160, 128, 128)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_mat_field(path: Path, field: str) -> np.ndarray:
    payload = loadmat(path, variable_names=[field])
    if field not in payload:
        raise ValueError(f"{path} has no MAT field {field!r}")
    return np.asarray(payload[field])


def as_array(value: np.ndarray, name: str) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=np.float32)
    if array.shape != PANEL_SHAPE:
        raise ValueError(f"{name}: expected {PANEL_SHAPE}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name}: non-finite values")
    if np.any(array < 0):
        raise ValueError(f"{name}: negative values are invalid for an intensity panel")
    return array


def intensity_db(array: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(array.astype(np.float64), np.finfo(np.float32).tiny))


def map_range(array: np.ndarray, low: float, high: float) -> np.ndarray:
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        raise ValueError("Invalid display range")
    return np.rint(np.clip((array - low) / (high - low), 0.0, 1.0) * 255.0).astype(np.uint8)


def save_png(path: Path, pixels: np.ndarray) -> None:
    Image.fromarray(pixels).save(path, dpi=(FIGURE_DPI, FIGURE_DPI), optimize=True)


def font(size: int) -> ImageFont.ImageFont:
    for path in (Path(r"C:\Windows\Fonts\times.ttf"), Path(r"C:\Windows\Fonts\timesnewroman.ttf")):
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default(size=size)


def centered(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str,
             selected_font: ImageFont.ImageFont) -> None:
    bounds = draw.textbbox((0, 0), text, font=selected_font)
    draw.text((xy[0] - (bounds[2] - bounds[0]) / 2, xy[1]), text,
              font=selected_font, fill="black")


def make_figure(output: Path, panel_files: list[Path], labels: list[str],
                filename: str, boxed_noisy: bool = False) -> tuple[int, int]:
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
    letters = "abcdefgh"
    for index, (path, label) in enumerate(zip(panel_files, labels, strict=True)):
        row, col = divmod(index, 4)
        x = side + col * (panel + gap)
        y = row * (panel + label_block + row_gap)
        actual = output / "panels" / "a_noisy_boxed.png" if index == 0 and boxed_noisy else path
        with Image.open(actual) as source:
            tile = source.convert("RGB").resize((panel, panel), Image.Resampling.LANCZOS)
        canvas.paste(tile, (x, y))
        centered(draw, (x + panel // 2, y + panel + 2), f"({letters[index]})", label_font)
        centered(draw, (x + panel // 2, y + panel + label_step), label, label_font)
    target = output / filename
    canvas.save(target, dpi=(FIGURE_DPI, FIGURE_DPI), optimize=True)
    return canvas.size


def array_stats(array: np.ndarray, low: float, high: float) -> dict:
    values = array.astype(np.float64)
    db = intensity_db(values)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "finite": bool(np.isfinite(values).all()),
        "min": float(values.min()),
        "p01": float(np.percentile(values, 1.0)),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "p997": float(np.percentile(values, 99.7)),
        "max": float(values.max()),
        "negative_fraction": float(np.mean(values < 0)),
        "zero_fraction": float(np.mean(values == 0)),
        "below_shared_db_fraction": float(np.mean(db < low)),
        "above_shared_db_fraction": float(np.mean(db > high)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    panels_dir = output / "panels"
    ratios_dir = output / "ratios"
    panels_dir.mkdir(parents=True, exist_ok=True)
    ratios_dir.mkdir(parents=True, exist_ok=True)
    # These are generated products from the superseded historical-Ours layout,
    # not source results.  Remove only the exact stale names so a rerendered
    # directory cannot be mistaken for containing ten current panels or mixed
    # linear-ratio/dB-ratio conventions.
    for stale in (
        panels_dir / "g_ours-base_historical.png",
        panels_dir / "h_ours-ams_historical.png",
        ratios_dir / "ratio_cl-sar_noisy-over-output.npy",
        ratios_dir / "ratio_cl-sar_noisy-over-output.png",
        ratios_dir / "ratio_ours-ams_noisy-over-output.npy",
        ratios_dir / "ratio_ours-ams_noisy-over-output.png",
    ):
        stale.unlink(missing_ok=True)

    clsar_report_path = CL_SAR_ROOT / "run.json"
    clsar_report = read_json(clsar_report_path)
    low, high = map(float, clsar_report["display"]["shared_db_limits"])
    noisy_path = CL_SAR_ROOT / "noisy_intensity.npy"
    noisy = as_array(np.load(noisy_path, allow_pickle=False), "Noisy")

    source_files: dict[str, list[Path]] = {
        "Noisy": [noisy_path, clsar_report_path],
        "SAR-BM3D": [RUNS_ROOT / "sarbm3d" / "result.mat", RUNS_ROOT / "sarbm3d" / "summary.json"],
        "SAR2SAR": [RUNS_ROOT / "sar2sar" / "result.mat", RUNS_ROOT / "sar2sar" / "summary.json"],
        "SDUDNet": [RUNS_ROOT / "sdudnet_db_display" / "denoised.npy", RUNS_ROOT / "sdudnet_db_display" / "run.json"],
        "Trans-SAR": [RUNS_ROOT / "transsar" / "denoised.npy", RUNS_ROOT / "transsar" / "run.json"],
        "CL-SAR": [CL_SAR_ROOT / "CL-SAR_intensity.npy", clsar_report_path],
        "MERLIN": [
            RUNS_ROOT / "merlin_stride64_weighted" / "denoised.npy",
            RUNS_ROOT / "merlin_stride64_weighted" / "run.json",
        ],
        "MuLoG-DRUNet": [RUNS_ROOT / "mulog_drunet" / "denoised.npy", RUNS_ROOT / "mulog_drunet" / "run.json"],
    }
    missing = [str(path) for paths in source_files.values() for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required method artifacts:\n" + "\n".join(missing))

    # SDUDNet's released real-image model consumes an 8-bit grayscale raster.
    # It was run on the fixed noisy dB rendering; invert that known affine dB
    # coordinate here so its panel can use the same display mapper as the rest.
    sdud_coordinate = as_array(
        np.load(source_files["SDUDNet"][0], allow_pickle=False), "SDUDNet display coordinate"
    )
    sdud_coordinate_for_plot = np.clip(sdud_coordinate.astype(np.float64), 0.0, 1.0)
    sdud_db = low + sdud_coordinate_for_plot * (high - low)
    sdud_intensity = np.ascontiguousarray(np.power(10.0, sdud_db / 10.0), dtype=np.float32)
    np.save(output / "sdudnet_db_display_plot_intensity.npy", sdud_intensity, allow_pickle=False)

    arrays: dict[str, np.ndarray] = {
        "Noisy": noisy,
        "SAR-BM3D": as_array(load_mat_field(source_files["SAR-BM3D"][0], "denoised_intensity"), "SAR-BM3D"),
        "SAR2SAR": as_array(load_mat_field(source_files["SAR2SAR"][0], "denoised_intensity"), "SAR2SAR"),
        "SDUDNet": sdud_intensity,
        "Trans-SAR": as_array(np.load(source_files["Trans-SAR"][0], allow_pickle=False), "Trans-SAR"),
        "CL-SAR": as_array(np.load(source_files["CL-SAR"][0], allow_pickle=False), "CL-SAR"),
        "MERLIN": as_array(np.load(source_files["MERLIN"][0], allow_pickle=False), "MERLIN"),
        "MuLoG-DRUNet": as_array(np.load(source_files["MuLoG-DRUNet"][0], allow_pickle=False), "MuLoG-DRUNet"),
    }
    noisy_db = intensity_db(noisy)
    merlin_db = intensity_db(arrays["MERLIN"])
    noisy_median_db = float(np.median(noisy_db))
    merlin_median_db = float(np.median(merlin_db))
    merlin_display_shift_db = merlin_median_db - noisy_median_db
    merlin_display_low = low + merlin_display_shift_db
    merlin_display_high = high + merlin_display_shift_db

    names = list(arrays)
    main_figure_labels = names.copy()
    main_figure_labels[names.index("MERLIN")] = "MERLIN†"
    strict_figure_labels = names.copy()
    strict_figure_labels[names.index("MERLIN")] = "MERLIN"
    stems = ["a_noisy", "b_sar-bm3d", "c_sar2sar", "d_sdudnet", "e_trans-sar",
             "f_cl-sar", "g_merlin", "h_mulog-drunet"]
    panel_files: list[Path] = []
    strict_shared_db_panel_files: list[Path] = []
    for name, stem in zip(names, stems, strict=True):
        db = intensity_db(arrays[name])
        shared_panel = map_range(db, low, high)
        path = panels_dir / f"{stem}.png"
        if name == "MERLIN":
            strict_path = panels_dir / "g_merlin_shared-db.png"
            save_png(strict_path, shared_panel)
            panel = map_range(db, merlin_display_low, merlin_display_high)
        else:
            strict_path = path
            panel = shared_panel
        save_png(path, panel)
        panel_files.append(path)
        strict_shared_db_panel_files.append(strict_path)

    boxed = Image.open(panel_files[0]).convert("RGB")
    box_draw = ImageDraw.Draw(boxed)
    x, y, w, h = INSPECTION_BOX
    box_draw.rectangle((x, y, x + w - 1, y + h - 1), outline=(230, 25, 35), width=5)
    boxed.save(panels_dir / "a_noisy_boxed.png", dpi=(FIGURE_DPI, FIGURE_DPI), optimize=True)

    figure_size = make_figure(
        output, panel_files, main_figure_labels,
        "figure3_umbra_buenos_aires_methods_600dpi.png", boxed_noisy=False,
    )
    make_figure(
        output, panel_files, main_figure_labels,
        "figure3_umbra_buenos_aires_methods_boxed_600dpi.png", boxed_noisy=True,
    )
    make_figure(
        output, strict_shared_db_panel_files, strict_figure_labels,
        "figure3_umbra_buenos_aires_methods_strict_shared_db_600dpi.png",
        boxed_noisy=False,
    )

    ratio_outputs: dict[str, dict] = {}
    ratio_limit_db = float(clsar_report["display"]["ratio_symmetric_limit_db"])
    for method, stem in (("CL-SAR", "ratio_cl-sar_db_noisy-over-output"),
                         ("MERLIN", "ratio_merlin_db_noisy-over-output"),
                         ("MuLoG-DRUNet", "ratio_mulog-drunet_db_noisy-over-output")):
        ratio64 = intensity_db(noisy) - intensity_db(arrays[method])
        ratio = np.ascontiguousarray(ratio64, dtype=np.float32)
        npy_path = ratios_dir / f"{stem}.npy"
        png_path = ratios_dir / f"{stem}.png"
        np.save(npy_path, ratio, allow_pickle=False)
        save_png(png_path, map_range(ratio, -ratio_limit_db, ratio_limit_db))
        ratio_outputs[method] = {
            "definition_db": "10*log10(noisy/output)",
            "display_range_db": [-ratio_limit_db, ratio_limit_db],
            "npy": str(npy_path.relative_to(output)),
            "png": str(png_path.relative_to(output)),
            "saturated_high_fraction": float(np.mean(ratio64 > ratio_limit_db)),
            "saturated_low_fraction": float(np.mean(ratio64 < -ratio_limit_db)),
        }

    savemat(
        output / "figure3_arrays.mat",
        {
            "noisy_intensity": arrays["Noisy"],
            "sarbm3d_intensity": arrays["SAR-BM3D"],
            "sar2sar_intensity": arrays["SAR2SAR"],
            "sdudnet_display_adapter_plot_intensity": arrays["SDUDNet"],
            "transsar_intensity": arrays["Trans-SAR"],
            "clsar_intensity": arrays["CL-SAR"],
            "merlin_intensity": arrays["MERLIN"],
            "mulog_drunet_intensity": arrays["MuLoG-DRUNet"],
        },
        do_compression=True,
    )

    methods: dict[str, dict] = {}
    for name, path in zip(names, panel_files, strict=True):
        methods[name] = {
            "statistics": array_stats(arrays[name], low, high),
            "array_domain": (
                "display-domain coordinate encoded as intensity only for common plotting"
                if name == "SDUDNet" else "linear intensity"
            ),
            "panel": str(path.relative_to(output)),
            "panel_sha256": sha256(path),
            "source_files": [
                {"path": str(item.resolve()), "sha256": sha256(item)} for item in source_files[name]
            ],
        }
    methods["SDUDNet"]["adapter"] = {
        "network_input": "fixed noisy 8-bit dB panel, matching the released grayscale-raster ToTensor contract",
        "forward_display_mapping": f"u=round(255*clip((10log10(I)-{low})/({high}-{low}),0,1))/255",
        "inverse_for_common_panel": f"I_plot=10^(({low}+clip(u_hat,0,1)*({high}-{low}))/10)",
        "not_raw_linear_intensity_inference": True,
        "raw_linear_intensity_sensitivity_run": str((RUNS_ROOT / "sdudnet").resolve()),
    }
    methods["MERLIN"]["adapter"] = {
        "figure_label": "MERLIN†",
        "network_input": "native SICD real and imaginary parts",
        "published_model": "TerraSAR-X HS Spotlight",
        "symetrise_real_imaginary_parts": True,
        "patch_size": 256,
        "stride_size": 64,
        "overlap_aggregation": (
            "separable positive-Hann weighted overlap-add in the network-output domain; "
            "the official patch predictions are unchanged"
        ),
        "overlap_window_1d": "numpy.hanning(patch_size + 2)[1:-1]",
        "package_default_aggregator": False,
        "official_output_domain": "amplitude",
        "figure_array": "official amplitude squared exactly once by the method runner to linear intensity",
        "cross_sensor_application": "published Spotlight checkpoint applied to Umbra X-band Spotlight data",
        "radiometric_gain_matching": False,
    }
    methods["MERLIN"]["display"] = {
        "primary_panel_mapping": "affine 8-bit map of 10*log10(scientific linear intensity)",
        "window_rule": (
            "preserve the noisy-derived shared dB span and translate the window so the "
            "MERLIN median maps to the same grayscale as the noisy median"
        ),
        "noisy_median_db": noisy_median_db,
        "merlin_median_db": merlin_median_db,
        "window_shift_db": merlin_display_shift_db,
        "primary_panel_db_limits": [merlin_display_low, merlin_display_high],
        "primary_panel_low_clip_fraction": float(np.mean(merlin_db < merlin_display_low)),
        "primary_panel_high_clip_fraction": float(np.mean(merlin_db > merlin_display_high)),
        "display_only": True,
        "scientific_array_modified": False,
        "ratio_products_modified": False,
        "strict_shared_db_panel": str((panels_dir / "g_merlin_shared-db.png").relative_to(output)),
        "strict_shared_db_panel_sha256": sha256(panels_dir / "g_merlin_shared-db.png"),
    }
    methods["MuLoG-DRUNet"]["adapter"] = {
        "d1_input_output": "scalar covariance / linear intensity",
        "looks": 1,
        "admm_iterations": 10,
    }

    report = {
        "complete": True,
        "scope": "Figure-3-style qualitative comparison on one fixed real Umbra SICD ROI",
        "source_sicd_sha256": "dfe8c9fb6adc1e0dd6efb93b29f076f192976c328037ca519d323221b0da1ac9",
        "roi": {"row_start": 7456, "col_start": 17256, "height": 1024, "width": 1024,
                "center_latitude_deg": -34.58670400173253,
                "center_longitude_deg": -58.37800039641809},
        "display": {
            "primary_mapping": (
                "10*log10(linear intensity); noisy-derived shared affine map except for "
                "the disclosed display-only MERLIN median-aligned window shift"
            ),
            "shared_db_limits": [low, high],
            "shared_db_methods": [name for name in names if name != "MERLIN"],
            "shared_limits_source": "noisy ROI percentiles 1.0 and 99.7",
            "per_method_autocontrast": True,
            "per_method_autocontrast_methods": ["MERLIN"],
            "merlin_primary_db_limits": [merlin_display_low, merlin_display_high],
            "merlin_limits_source": (
                "same span as the noisy-derived window, translated by the MERLIN-minus-noisy "
                "median dB offset"
            ),
            "merlin_display_shift_db": merlin_display_shift_db,
            "strict_shared_db_figure": "figure3_umbra_buenos_aires_methods_strict_shared_db_600dpi.png",
            "panel_pixels": [1024, 1024],
            "panel_png_mode": "8-bit grayscale",
            "figure_dpi": FIGURE_DPI,
            "figure_pixels": list(figure_size),
            "word_panel_size_mm": 21.06,
        },
        "methods": methods,
        "ratios": ratio_outputs,
        "interpretation_limits": {
            "clean_reference_available": False,
            "psnr_ssim_reported": False,
            "sar2sar_cross_sensor": "official Single-Look Sentinel-1 checkpoint applied to Umbra X-band",
            "merlin_cross_sensor": "official Spotlight model applied to Umbra rather than its TerraSAR-X training sensor",
            "mulog_looks": 1,
            "sdudnet_display_domain_adapter": True,
            "radiometric_brightness_comparable_across_all_main_panels": False,
            "publication_ready_as_qualitative_structural_comparison_with_disclosed_display_exception": True,
        },
        "files": {},
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    caption = (
        "Fig. 3. Visual comparison on an Umbra X-band Spotlight VV SICD intensity image over "
        "Buenos Aires: (a) noisy input, (b) SAR-BM3D, (c) SAR2SAR, (d) SDUDNet, "
        "(e) Trans-SAR, (f) CL-SAR, (g) MERLIN†, and (h) MuLoG-DRUNet. All panels show "
        "the same native ROI. Panels (a)-(f) and (h) use the noisy-derived shared dB window; "
        f"MERLIN† alone uses the same {high - low:.2f}-dB span shifted by "
        f"{merlin_display_shift_db:.2f} dB to align its median display level with the noisy input. "
        "The dagger adjustment changes PNG display only; the stored MERLIN intensity and ratio "
        "products are unchanged, and a strict shared-dB rendering is retained separately. "
        "MERLIN uses the published "
        "TerraSAR-X HS Spotlight checkpoint on the complex Umbra SICD (cross-sensor); its "
        "unchanged patch predictions are combined with positive-Hann weighted overlap, with "
        "no radiometric gain matching. "
        "MuLoG-DRUNet uses L=1. No clean reference "
        "is available; therefore the comparison is qualitative and PSNR/SSIM are not reported.\n"
    )
    (output / "caption_diagnostic.txt").write_text(caption, encoding="utf-8")
    readme = f"""# Umbra Buenos Aires multi-method run

The main comparison is `figure3_umbra_buenos_aires_methods_600dpi.png`.
It is 2 x 4 at {FIGURE_DPI} dpi and follows the V3 panel order and typography.
The boxed variant is optional; the unboxed version is recommended because the
ratio is no longer a main panel.

Every panel is 1024 x 1024. Panels (a)-(f) and (h) use the noisy-input dB
limits `[{low:.9f}, {high:.9f}]`. MERLIN† alone uses the same
{high - low:.9f}-dB span shifted by {merlin_display_shift_db:.9f} dB, giving
`[{merlin_display_low:.9f}, {merlin_display_high:.9f}]`, so its median display
level matches the noisy panel. This is a display-only adjustment: the stored
scientific array, MAT field, and ratio products are unchanged. The fully shared
mapping is retained in
`figure3_umbra_buenos_aires_methods_strict_shared_db_600dpi.png`.

Scientific arrays are collected in `figure3_arrays.mat`; original per-method
outputs and provenance remain under `../runs/`.

Panels (g) and (h) are MERLIN† and MuLoG-DRUNet; the historical Ours outputs
are not included. MERLIN uses the official Spotlight model with complex
real/imaginary data. Its unchanged 256 x 256 patch predictions (stride 64) are
combined by a separable positive-Hann weighted overlap-add adapter to suppress
patch-boundary seams. No gain matching, resize, or scientific-output clipping is
applied; the dagger denotes only the disclosed display-window shift.
MuLoG-DRUNet uses its D=1 intensity path with L=1.
MERLIN remains a cross-sensor transfer because its public Spotlight model was
developed for another sensor. SAR2SAR is likewise a Sentinel-1 checkpoint used
cross-sensor. SDUDNet uses its released grayscale-raster contract on the fixed
noisy dB rendering; its raw-linear sensitivity run is preserved separately
under `../runs/sdudnet/`.

No clean reference exists, so PSNR and SSIM are intentionally not reported.
See `manifest.json` for hashes, transforms, saturation fractions, and exact
input/output statistics.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    for path in sorted(item for item in output.rglob("*") if item.is_file() and item.name != "manifest.json"):
        report["files"][str(path.relative_to(output))] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    (output / "manifest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(output),
        "figure": str((output / "figure3_umbra_buenos_aires_methods_600dpi.png").resolve()),
        "shape": list(figure_size),
        "shared_db_limits": [low, high],
        "merlin_primary_db_limits": [merlin_display_low, merlin_display_high],
        "merlin_display_shift_db": merlin_display_shift_db,
        "strict_shared_db_figure": str(
            (output / "figure3_umbra_buenos_aires_methods_strict_shared_db_600dpi.png").resolve()
        ),
        "publication_ready_as_qualitative_structural_comparison_with_disclosed_display_exception": True,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
