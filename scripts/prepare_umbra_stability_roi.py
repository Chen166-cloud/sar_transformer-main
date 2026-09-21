"""Extract one fixed 1024x1024 complex Umbra SICD ROI for all seven methods."""

from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import json
import time

import numpy as np
from PIL import Image
from sarpy.io.complex.converter import open_complex
from scipy.io import savemat


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--row-start", type=int, required=True)
    parser.add_argument("--col-start", type=int, required=True)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--selection-note", default="selected from noisy-input preview only")
    args = parser.parse_args()
    if args.size != 1024:
        parser.error("the frozen Figure 3 protocol requires --size 1024")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError(f"output must be new or empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    reader = open_complex(str(args.source))
    try:
        sicd = reader.get_sicds_as_tuple()[0]
        height, width = int(sicd.ImageData.NumRows), int(sicd.ImageData.NumCols)
        if args.row_start < 0 or args.col_start < 0:
            raise ValueError("ROI starts must be nonnegative")
        if args.row_start + args.size > height or args.col_start + args.size > width:
            raise ValueError("ROI extends beyond the SICD image")
        complex_roi = np.ascontiguousarray(
            reader[
                args.row_start : args.row_start + args.size,
                args.col_start : args.col_start + args.size,
            ],
            dtype=np.complex64,
        )
        center_geo = sicd.project_image_to_ground_geo(
            [[args.row_start + args.size / 2, args.col_start + args.size / 2]]
        )[0]
        metadata = {
            "namespace": sicd._xml_ns["default"],
            "pixel_type": str(sicd.ImageData.PixelType),
            "full_shape": [height, width],
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
    if not np.isfinite(noisy).all() or np.any(noisy < 0) or not np.any(noisy > 0):
        raise ValueError("invalid complex-to-intensity conversion")
    intensity_scale = float(np.max(noisy))
    noisy_db = 10.0 * np.log10(np.maximum(noisy, np.finfo(np.float32).tiny))
    low, high = (float(v) for v in np.percentile(noisy_db, (1.0, 99.7)))
    preview = np.rint(np.clip((noisy_db - low) / (high - low), 0, 1) * 255).astype(np.uint8)

    complex_path = args.output / "sicd_complex_roi.npy"
    intensity_path = args.output / "noisy_intensity.npy"
    mat_path = args.output / "input.mat"
    np.save(complex_path, complex_roi, allow_pickle=False)
    np.save(intensity_path, noisy, allow_pickle=False)
    savemat(
        mat_path,
        {"noisy": noisy, "complex_roi": complex_roi, "input_domain": "linear_intensity"},
        do_compression=True,
    )
    Image.fromarray(preview).save(args.output / "noisy.png", dpi=(600, 600), optimize=True)

    report = {
        "status": "prepared",
        "source": {
            "path": str(args.source.resolve()),
            "bytes": args.source.stat().st_size,
            "sha256": sha256(args.source),
            "sicd": metadata,
        },
        "roi": {
            "row_start": args.row_start,
            "col_start": args.col_start,
            "height": args.size,
            "width": args.size,
            "center_latitude_deg": float(center_geo[0]),
            "center_longitude_deg": float(center_geo[1]),
            "center_hae_m": float(center_geo[2]),
            "selection": args.selection_note,
        },
        "scientific_domain": {
            "complex_to_intensity": "I = real(S)^2 + imag(S)^2",
            "dtype": "float32",
            "radiometric_calibration_added": False,
            "spatial_resampling": False,
        },
        "frozen_method_adapter": {
            "intensity_scale": intensity_scale,
            "scale_origin": "exact observed maximum of this fixed 1024x1024 ROI, matching the Figure 3 adapter rule",
            "minimum_subtraction": False,
            "per_tile_normalization": False,
        },
        "display": {
            "mapping": "10*log10(linear intensity), noisy percentiles 1.0 and 99.7",
            "shared_db_limits": [low, high],
            "preview_only": True,
        },
        "outputs": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in (complex_path, intensity_path, mat_path, args.output / "noisy.png")
        },
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (args.output / "run.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output.resolve()), "intensity_scale": intensity_scale}, indent=2))


if __name__ == "__main__":
    main()
