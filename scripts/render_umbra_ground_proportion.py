"""Render completed Umbra SICD ROI results on an equal-metre display grid.

This is a display-only projection to a local constant-height ellipsoid tangent plane.
The original complex ROI, method inputs, outputs, and ratio statistics are untouched.
It is not terrain orthorectification and cannot undo radar layover or shadow.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from sarpy.geometry.geocoords import geodetic_to_ecf
from sarpy.io.complex.converter import open_complex
from scipy.io import loadmat
from scipy.ndimage import map_coordinates

from render_umbra_stability_comparison import (
    DPI,
    NAMES,
    STEMS,
    WORD_COLUMN_INCHES,
    centered,
    font,
    intensity_db,
    map_range,
)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def local_east_north(geo: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """Convert geodetic points to metre-scale EN coordinates about origin."""
    lat, lon = np.deg2rad(origin[:2])
    east = np.array([-np.sin(lon), np.cos(lon), 0.0])
    north = np.array([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)])
    offset = geodetic_to_ecf(geo) - geodetic_to_ecf(origin)
    return np.column_stack((offset @ east, offset @ north))


def ground_grid(
    source: Path, roi: dict[str, object], orientation: str = "row-horizontal"
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Fit a local image-to-ground affine and invert it on an equal-metre grid."""
    row0, col0 = int(roi["row_start"]), int(roi["col_start"])
    height, width = int(roi["height"]), int(roi["width"])
    if (height, width) != (1024, 1024):
        raise ValueError("the fixed experiment protocol requires a 1024x1024 ROI")
    reader = open_complex(str(source))
    try:
        sicd = reader.get_sicds_as_tuple()[0]
        center_image = np.array([[row0 + 511.5, col0 + 511.5]], dtype=np.float64)
        center_geo = sicd.project_image_to_ground_geo(center_image)[0]
        sample = np.linspace(0, 1023, 5, dtype=np.float64)
        rr, cc = np.meshgrid(sample, sample, indexing="ij")
        relative = np.column_stack((rr.ravel(), cc.ravel()))
        global_image = relative + np.array([row0, col0], dtype=np.float64)
        # A constant ellipsoid height makes this an honest ground-plane display
        # approximation. Tall building layover remains in the original SAR data.
        geo = sicd.project_image_to_ground_geo(
            global_image, projection_type="HAE", hae0=float(center_geo[2])
        )
    finally:
        reader.close()

    en = local_east_north(geo, center_geo)
    design = np.column_stack((relative, np.ones(relative.shape[0])))
    en_coefficients, _, _, _ = np.linalg.lstsq(design, en, rcond=None)
    if orientation == "row-horizontal":
        x_axis = en_coefficients[0, :] / np.linalg.norm(en_coefficients[0, :])
        y_axis = np.array([-x_axis[1], x_axis[0]])
        if en_coefficients[1, :] @ y_axis < 0:
            y_axis = -y_axis
        basis = np.column_stack((x_axis, y_axis))
        axis_description = "source row runs right; source column generally runs down"
    elif orientation == "north-up":
        basis = np.array([[1.0, 0.0], [0.0, -1.0]])
        axis_description = "east right; north up"
    else:
        raise ValueError(f"unrecognized orientation: {orientation}")
    display_xy = en @ basis
    coefficients, _, _, _ = np.linalg.lstsq(design, display_xy, rcond=None)
    residual = np.linalg.norm(design @ coefficients - display_xy, axis=1)
    linear = coefficients[:2, :].T  # display [x,y] = linear @ [row,col] + offset
    if abs(np.linalg.det(linear)) < 1e-9:
        raise ValueError("degenerate local image-to-ground projection")
    offset = coefficients[2, :]
    corners = np.array(
        [[-0.5, -0.5], [-0.5, 1023.5], [1023.5, -0.5], [1023.5, 1023.5]],
        dtype=np.float64,
    )
    corner_en = corners @ coefficients[:2, :] + offset
    x_min, y_min = corner_en.min(axis=0)
    x_max, y_max = corner_en.max(axis=0)
    span_x, span_y = float(x_max - x_min), float(y_max - y_min)
    metres_per_pixel = max(span_x, span_y) / 1024.0
    out_width = max(1, int(math.ceil(span_x / metres_per_pixel)))
    out_height = max(1, int(math.ceil(span_y / metres_per_pixel)))
    x = x_min + (np.arange(out_width, dtype=np.float64) + 0.5) * metres_per_pixel
    y = y_min + (np.arange(out_height, dtype=np.float64) + 0.5) * metres_per_pixel
    xx, yy = np.meshgrid(x, y)
    row_col = np.linalg.inv(linear) @ np.stack((xx - offset[0], yy - offset[1]), axis=0).reshape(2, -1)
    source_rows = row_col[0].reshape(out_height, out_width)
    source_cols = row_col[1].reshape(out_height, out_width)
    valid = (
        (source_rows >= 0) & (source_rows <= 1023)
        & (source_cols >= 0) & (source_cols <= 1023)
    )
    if not valid.any():
        raise ValueError("projected ROI has no valid output pixels")
    info = {
        "method": "constant-HAE local EN tangent-plane affine fit; display only",
        "not_terrain_orthorectified": True,
        "orientation": orientation,
        "orientation_description": axis_description,
        "basis_east_north_to_display_xy": basis.tolist(),
        "hae_m": float(center_geo[2]),
        "origin_lat_lon": [float(center_geo[0]), float(center_geo[1])],
        "image_row_col_to_display_xy_m": {
            "linear": linear.tolist(), "offset": offset.tolist(),
        },
        "ground_span_xy_m": [span_x, span_y],
        "source_ground_step_row_m": float(np.linalg.norm(linear[:, 0])),
        "source_ground_step_col_m": float(np.linalg.norm(linear[:, 1])),
        "source_ground_axes_angle_deg": float(
            np.rad2deg(np.arccos(np.clip(
                linear[:, 0] @ linear[:, 1]
                / (np.linalg.norm(linear[:, 0]) * np.linalg.norm(linear[:, 1])), -1, 1
            )))
        ),
        "affine_residual_max_m": float(residual.max()),
        "affine_residual_rms_m": float(np.sqrt(np.mean(residual ** 2))),
        "output_pixel_metres": float(metres_per_pixel),
        "output_width_height_pixels": [out_width, out_height],
        "valid_pixel_fraction": float(valid.mean()),
        "bounds_xy_m": [float(x_min), float(y_min), float(x_max), float(y_max)],
        "interpolation": "bilinear in dB, performed only for display PNGs",
    }
    if info["affine_residual_max_m"] > metres_per_pixel:
        raise ValueError(
            f"affine approximation exceeds one output pixel: {info['affine_residual_max_m']:.3f} m"
        )
    return source_rows, source_cols, valid, info


def warp_db(
    db: np.ndarray, source_rows: np.ndarray, source_cols: np.ndarray, valid: np.ndarray,
    low: float, high: float,
) -> np.ndarray:
    interpolated = map_coordinates(
        np.asarray(db, dtype=np.float32), (source_rows, source_cols),
        order=1, mode="nearest", prefilter=False,
    )
    pixels = map_range(interpolated, low, high)
    pixels[~valid] = 255  # exterior of the non-rectangular source footprint
    return pixels


def word_panel(path: Path, target: Path, width: int) -> None:
    with Image.open(path) as image:
        original = image.convert("L")
        height = max(1, round(original.height * width / original.width))
        resized = original.resize((width, height), Image.Resampling.LANCZOS)
    resized.save(target, dpi=(DPI, DPI), optimize=True)


def figure(paths: list[Path], target: Path) -> list[int]:
    width = round(WORD_COLUMN_INCHES * DPI)
    panel = round(0.24 * width)
    with Image.open(paths[0]) as first:
        panel_height = first.height
    gap = round(0.01 * width)
    side = max(0, (width - 4 * panel - 3 * gap) // 2)
    selected_font = font(round(8 * DPI / 72))
    label_step = round(9 * DPI / 72)
    label_block = 2 * label_step + round(DPI / 72)
    row_gap = round(3 * DPI / 72)
    height = 2 * (panel_height + label_block) + row_gap
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    for index, path in enumerate(paths):
        row, col = divmod(index, 4)
        x = side + col * (panel + gap)
        y = row * (panel_height + label_block + row_gap)
        with Image.open(path) as image:
            if image.size != (panel, panel_height):
                raise ValueError(f"word panel has incorrect size: {path}")
            canvas.paste(image.convert("RGB"), (x, y))
        centered(draw, x + panel // 2, y + panel_height + 2, f"({'abcdefgh'[index]})", selected_font)
        label = "MERLIN†" if index == 6 else NAMES[index]
        centered(draw, x + panel // 2, y + panel_height + label_step, label, selected_font)
    canvas.save(target, dpi=(DPI, DPI), optimize=True)
    return list(canvas.size)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roi", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--orientation", choices=("row-horizontal", "north-up"), default="row-horizontal")
    args = parser.parse_args()
    roi = args.roi.resolve()
    experiment = args.experiment.resolve()
    source_report = json.loads((roi / "run.json").read_text(encoding="utf-8"))
    source = Path(source_report["source"]["path"])
    slant_root = experiment / "figure3_style"
    slant_manifest = json.loads((slant_root / "manifest.json").read_text(encoding="utf-8"))
    if not slant_manifest.get("complete"):
        raise ValueError("the SICD-grid comparison must finish before ground display rendering")
    roi_keys = ("row_start", "col_start", "height", "width")
    if any(slant_manifest["roi"][key] != source_report["roi"][key] for key in roi_keys):
        raise ValueError("SICD-grid results belong to a different ROI")
    arrays = loadmat(slant_root / "scientific_arrays.mat", variable_names=[name.replace("-", "_") for name in NAMES])
    values = {name: arrays[name.replace("-", "_")] for name in NAMES}
    for name, value in values.items():
        if value.shape != (1024, 1024) or not np.isfinite(value).all() or np.any(value < 0):
            raise ValueError(f"invalid scientific input for {name}")
    if not np.array_equal(values["Noisy"], np.load(roi / "noisy_intensity.npy", allow_pickle=False)):
        raise ValueError("SICD-grid scientific arrays do not match this ROI's noisy intensity")
    source_rows, source_cols, valid, geo = ground_grid(source, source_report["roi"], args.orientation)
    folder_name = "figure3_ground_equal_scale" if args.orientation == "row-horizontal" else "figure3_ground_north_up"
    output = experiment / folder_name
    panels = output / "panels"
    word = output / "panels_word"
    ratios = output / "ratios"
    square = output / "panels_center_square"
    square_word = output / "panels_word_center_square"
    for folder in (panels, word, ratios, square, square_word):
        folder.mkdir(parents=True, exist_ok=True)
    low, high = (float(v) for v in slant_manifest["display"]["shared_db_limits"])
    merlin_shift = float(slant_manifest["display"]["merlin_display_only_median_shift_db"])
    word_width = round(0.24 * round(WORD_COLUMN_INCHES * DPI))
    word_paths = []
    square_word_paths = []
    map_height, map_width = source_rows.shape
    square_side = min(map_width, map_height)
    square_left = (map_width - square_side) // 2
    square_top = (map_height - square_side) // 2
    square_box = (square_left, square_top, square_left + square_side, square_top + square_side)
    square_valid_fraction = float(valid[square_top:square_top + square_side, square_left:square_left + square_side].mean())
    for name, stem in zip(NAMES, STEMS, strict=True):
        db = intensity_db(values[name])
        if name == "MERLIN":
            db = db - merlin_shift  # same display-only convention as the SICD-grid figure
        pixels = warp_db(db, source_rows, source_cols, valid, low, high)
        native = panels / f"{stem}.png"
        Image.fromarray(pixels).save(native, dpi=(DPI, DPI), optimize=True)
        fitted = word / f"{stem}.png"
        word_panel(native, fitted, word_width)
        word_paths.append(fitted)
        if args.orientation == "row-horizontal":
            square_native = square / f"{stem}.png"
            Image.fromarray(pixels[square_top:square_top + square_side, square_left:square_left + square_side]).save(
                square_native, dpi=(DPI, DPI), optimize=True
            )
            square_fitted = square_word / f"{stem}.png"
            word_panel(square_native, square_fitted, word_width)
            square_word_paths.append(square_fitted)
    ratio_limit = float(slant_manifest["ratios"]["shared_symmetric_limit_db"])
    for name, stem in zip(NAMES[1:], STEMS[1:], strict=True):
        db = intensity_db(values["Noisy"]) - intensity_db(values[name])
        pixels = warp_db(db, source_rows, source_cols, valid, -ratio_limit, ratio_limit)
        Image.fromarray(pixels).save(
            ratios / f"ratio_{stem[2:]}_db_noisy-over-output.png", dpi=(DPI, DPI), optimize=True
        )
    figure_path = output / f"{folder_name}_7methods_600dpi.png"
    figure_size = figure(word_paths, figure_path)
    square_figure = None
    square_figure_size = None
    if args.orientation == "row-horizontal":
        square_figure = output / "figure3_ground_center_square_7methods_600dpi.png"
        square_figure_size = figure(square_word_paths, square_figure)
    manifest = {
        "complete": True,
        "scene": slant_manifest["scene"],
        "roi": source_report["roi"],
        "science_input_unchanged": True,
        "scientific_arrays_source": str((slant_root / "scientific_arrays.mat").resolve()),
        "scientific_arrays_sha256": sha256(slant_root / "scientific_arrays.mat"),
        "ground_display": geo,
        "display": {
            "source_noisy_db_limits": [low, high],
            "source_noisy_db_percentiles": slant_manifest["display"]["shared_db_percentiles_from_noisy"],
            "merlin_display_only_db_subtraction": merlin_shift,
            "word_panel_pixels": [word_width, int(round(word_width * geo["output_width_height_pixels"][1] / geo["output_width_height_pixels"][0]))],
            "figure_pixels": figure_size,
            "figure_dpi": DPI,
            "ratio_shared_symmetric_limit_db": ratio_limit,
            "center_square_display_only": {
                "created": args.orientation == "row-horizontal",
                "map_pixel_box_left_top_right_bottom": list(square_box),
                "ground_side_m": float(square_side * geo["output_pixel_metres"]),
                "valid_pixel_fraction": square_valid_fraction,
                "figure": str(square_figure.name) if square_figure else None,
                "figure_pixels": square_figure_size,
            },
        },
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    manifest["files"] = {
        str(path.relative_to(output)): {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in output.rglob("*") if path.is_file() and path.name != "manifest.json"
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(figure_path)


if __name__ == "__main__":
    main()
