#!/usr/bin/env python3
"""Run the official MERLIN Spotlight model on the fixed Umbra complex ROI.

The official network returns an amplitude estimate.  This adapter preserves that
array and squares it exactly once to obtain the linear-intensity result used by
the common Figure 3 renderer.  No resizing or scene-wise normalization is
performed by this adapter.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import types
from datetime import datetime, timezone

import numpy as np
from scipy import io as sio
from scipy import signal
import torch


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    ROOT
    / "output"
    / "cl_sar_umbra_buenos_aires"
    / "figure3_clsar_final"
    / "sicd_complex_roi.npy"
)
DEFAULT_REFERENCE_INTENSITY = DEFAULT_INPUT.with_name("noisy_intensity.npy")
DEFAULT_OUTPUT = (
    ROOT
    / "output"
    / "umbra_buenos_aires_multimethod_1024"
    / "runs"
    / "merlin"
)
REPOSITORY = ROOT / "external" / "deepdespeckling"
CHECKPOINT = (
    REPOSITORY
    / "deepdespeckling"
    / "merlin"
    / "saved_models"
    / "spotlight.pth"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPOSITORY), *args],
        text=True,
        encoding="utf-8",
    ).strip()


def array_stats(array: np.ndarray) -> dict[str, object]:
    finite = np.isfinite(array)
    values = array[finite]
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "finite_fraction": float(finite.mean()),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean(dtype=np.float64)),
        "median": float(np.median(values)),
        "percentiles_1_50_99_99p7": [
            float(v) for v in np.percentile(values, [1.0, 50.0, 99.0, 99.7])
        ],
        "zero_count": int(np.count_nonzero(values == 0)),
        "negative_count": int(np.count_nonzero(values < 0)),
    }


def calculate_official_symmetry_rolls(
    real_part: np.ndarray, imag_part: np.ndarray
) -> tuple[int, int]:
    """Reproduce only the shift-selection arithmetic for provenance logging.

    The actual transformed patch returned to MERLIN is always produced by the
    unmodified official function.  This helper records its selected circular
    row/range rolls without replacing the numerical transform.
    """

    spectrum = np.fft.fftshift(
        np.fft.fft2(real_part[0, :, :, 0] + 1j * imag_part[0, :, :, 0])
    )

    row_profile = np.mean(np.abs(spectrum), axis=1)
    row_corr = np.real(
        np.fft.ifft(
            np.fft.fft(row_profile)
            * np.conjugate(np.fft.fft(row_profile[::-1]))
        )
    )
    row_peak = int(np.unravel_index(row_corr.argmax(), row_profile.shape[0])[0])
    row_roll_1 = (
        int(round(-(row_peak - 1) / 2)) % row_profile.shape[0]
        + int(row_profile.shape[0] / 2)
    )
    row_roll_2 = (
        int(round(-(row_peak - 1 - row_profile.shape[0]) / 2))
        % row_profile.shape[0]
        + int(row_profile.shape[0] / 2)
    )
    row_window = signal.windows.gaussian(
        row_profile.shape[0], std=0.2 * row_profile.shape[0]
    )
    row_score_1 = np.sum(row_window * np.roll(row_profile, row_roll_1))
    row_score_2 = np.sum(row_window * np.roll(row_profile, row_roll_2))
    row_roll = row_roll_1 if row_score_1 >= row_score_2 else row_roll_2

    # Match the official code exactly: the column profile is computed from the
    # original centered spectrum, before the selected row roll is applied.
    col_profile = np.mean(np.abs(spectrum), axis=0)
    col_corr = np.real(
        np.fft.ifft(
            np.fft.fft(col_profile)
            * np.conjugate(np.fft.fft(col_profile[::-1]))
        )
    )
    col_peak = int(np.unravel_index(col_corr.argmax(), col_profile.shape[0])[0])
    col_roll_1 = (
        int(round(-(col_peak - 1) / 2)) % col_profile.shape[0]
        + int(col_profile.shape[0] / 2)
    )
    col_roll_2 = (
        int(round(-(col_peak - 1 - col_profile.shape[0]) / 2))
        % col_profile.shape[0]
        + int(col_profile.shape[0] / 2)
    )
    col_window = signal.windows.gaussian(
        col_profile.shape[0], std=0.2 * col_profile.shape[0]
    )
    col_score_1 = np.sum(col_window * np.roll(col_profile, col_roll_1))
    col_score_2 = np.sum(col_window * np.roll(col_profile, col_roll_2))
    col_roll = col_roll_1 if col_score_1 >= col_score_2 else col_roll_2
    return int(row_roll), int(col_roll)


def install_unused_gdal_import_stub() -> None:
    """Satisfy an unused top-level GDAL import without installing GDAL.

    MERLIN inference on an already loaded [H,W,2] numpy array never invokes the
    package's TIFF loader.  The official utils module nevertheless imports
    ``osgeo.gdal`` at module import time, so a deliberately empty import-only
    stub keeps this adapter independent of an irrelevant GDAL installation.
    """

    if "osgeo" in sys.modules:
        return
    osgeo_module = types.ModuleType("osgeo")
    gdal_module = types.ModuleType("gdal")
    osgeo_module.gdal = gdal_module
    sys.modules["osgeo"] = osgeo_module
    sys.modules["osgeo.gdal"] = gdal_module


def run_once(
    image_real_imag: np.ndarray,
    patch_size: int,
    stride_size: int,
) -> tuple[dict[str, dict[str, np.ndarray]], list[dict[str, int]], float]:
    install_unused_gdal_import_stub()
    from deepdespeckling.merlin import merlin_denoiser as official_merlin

    official_symetrise = official_merlin.symetrise_real_and_imaginary_parts
    selected_rolls: list[tuple[int, int]] = []

    def audited_symetrise(
        real_part: np.ndarray, imag_part: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        selected_rolls.append(
            calculate_official_symmetry_rolls(real_part, imag_part)
        )
        return official_symetrise(real_part, imag_part)

    official_merlin.symetrise_real_and_imaginary_parts = audited_symetrise
    try:
        denoiser = official_merlin.MerlinDenoiser(
            model_name="spotlight", symetrise=True
        )
        started = time.perf_counter()
        with torch.inference_mode():
            result = denoiser.denoise_image(
                image_real_imag,
                patch_size=patch_size,
                stride_size=stride_size,
            )
        elapsed = time.perf_counter() - started
    finally:
        official_merlin.symetrise_real_and_imaginary_parts = official_symetrise

    x_positions = denoiser.initialize_axis_range(
        image_real_imag.shape[0], patch_size, stride_size
    )
    y_positions = denoiser.initialize_axis_range(
        image_real_imag.shape[1], patch_size, stride_size
    )
    expected_count = len(x_positions) * len(y_positions)
    if len(selected_rolls) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} symmetrisation calls, got {len(selected_rolls)}"
        )
    records: list[dict[str, int]] = []
    for index, ((row, col), (row_roll, col_roll)) in enumerate(
        zip(
            ((x, y) for x in x_positions for y in y_positions),
            selected_rolls,
            strict=True,
        )
    ):
        records.append(
            {
                "index": index,
                "row_start": int(row),
                "col_start": int(col),
                "row_circular_roll": row_roll,
                "col_circular_roll": col_roll,
            }
        )
    return result, records, elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--reference-intensity", type=Path, default=DEFAULT_REFERENCE_INTENSITY
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--stride-size", type=int, default=254)
    parser.add_argument(
        "--skip-repeat",
        action="store_true",
        help="Skip the second full inference used for deterministic verification.",
    )
    args = parser.parse_args()

    input_path = args.input.resolve()
    reference_intensity_path = args.reference_intensity.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.patch_size != 256 or args.stride_size not in {64, 254}:
        raise ValueError(
            "This provenance adapter is locked to patch_size=256 and the "
            "requested MERLIN strides 254 (official default) or 64 (high overlap)."
        )
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)

    complex_roi = np.load(input_path, allow_pickle=False)
    if complex_roi.shape != (1024, 1024):
        raise ValueError(f"Expected fixed 1024x1024 ROI, got {complex_roi.shape}")
    if complex_roi.dtype != np.complex64:
        raise TypeError(f"Expected complex64 SICD pixels, got {complex_roi.dtype}")
    if not np.isfinite(complex_roi).all():
        raise ValueError("Complex ROI contains non-finite values")

    image_real_imag = np.stack(
        (complex_roi.real, complex_roi.imag), axis=-1
    ).astype(np.float32, copy=False)
    if image_real_imag.shape != (1024, 1024, 2):
        raise AssertionError(image_real_imag.shape)
    if not np.array_equal(image_real_imag[..., 0], complex_roi.real):
        raise AssertionError("Real-channel adapter changed source values")
    if not np.array_equal(image_real_imag[..., 1], complex_roi.imag):
        raise AssertionError("Imaginary-channel adapter changed source values")

    noisy_intensity = (
        np.square(image_real_imag[..., 0], dtype=np.float32)
        + np.square(image_real_imag[..., 1], dtype=np.float32)
    ).astype(np.float32, copy=False)
    reference_intensity = np.load(reference_intensity_path, allow_pickle=False)
    reference_max_abs_error = float(
        np.max(
            np.abs(
                noisy_intensity.astype(np.float64)
                - reference_intensity.astype(np.float64)
            )
        )
    )
    reference_exact = bool(np.array_equal(noisy_intensity, reference_intensity))
    if not reference_exact:
        raise AssertionError(
            "Adapter intensity differs from the fixed CL-SAR noisy_intensity.npy; "
            f"max abs error={reference_max_abs_error}"
        )

    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    primary, symmetry_records, primary_seconds = run_once(
        image_real_imag, args.patch_size, args.stride_size
    )
    amplitude = np.asarray(primary["denoised"]["full"], dtype=np.float32)
    denoised_intensity = np.square(amplitude, dtype=np.float32)

    if amplitude.shape != (1024, 1024):
        raise AssertionError(f"Unexpected amplitude shape: {amplitude.shape}")
    if denoised_intensity.shape != (1024, 1024):
        raise AssertionError(denoised_intensity.shape)
    if not np.isfinite(amplitude).all() or not np.isfinite(denoised_intensity).all():
        raise ValueError("MERLIN produced non-finite values")
    if np.any(amplitude < 0) or np.any(denoised_intensity < 0):
        raise ValueError("MERLIN amplitude/intensity violates non-negativity")

    repeat_validation: dict[str, object]
    if args.skip_repeat:
        repeat_validation = {"performed": False}
    else:
        repeat, repeat_symmetry_records, repeat_seconds = run_once(
            image_real_imag, args.patch_size, args.stride_size
        )
        repeat_amplitude = np.asarray(
            repeat["denoised"]["full"], dtype=np.float32
        )
        repeat_intensity = np.square(repeat_amplitude, dtype=np.float32)
        max_abs = float(
            np.max(
                np.abs(
                    repeat_intensity.astype(np.float64)
                    - denoised_intensity.astype(np.float64)
                )
            )
        )
        repeat_validation = {
            "performed": True,
            "elapsed_seconds": repeat_seconds,
            "array_equal": bool(np.array_equal(repeat_intensity, denoised_intensity)),
            "max_abs_error": max_abs,
            "symmetry_records_equal": repeat_symmetry_records == symmetry_records,
            "in_memory_sha256_primary": hashlib.sha256(
                denoised_intensity.tobytes(order="C")
            ).hexdigest(),
            "in_memory_sha256_repeat": hashlib.sha256(
                repeat_intensity.tobytes(order="C")
            ).hexdigest(),
        }
        if not repeat_validation["array_equal"]:
            raise AssertionError(
                f"Repeated MERLIN inference was not bitwise equal; max abs={max_abs}"
            )

    paths = {
        "denoised.npy": output_dir / "denoised.npy",
        "merlin_amplitude.npy": output_dir / "merlin_amplitude.npy",
        "merlin_input_real_imag.npy": output_dir / "merlin_input_real_imag.npy",
        "noisy_intensity.npy": output_dir / "noisy_intensity.npy",
        "result.mat": output_dir / "result.mat",
    }
    np.save(paths["denoised.npy"], denoised_intensity, allow_pickle=False)
    np.save(paths["merlin_amplitude.npy"], amplitude, allow_pickle=False)
    np.save(paths["merlin_input_real_imag.npy"], image_real_imag, allow_pickle=False)
    np.save(paths["noisy_intensity.npy"], noisy_intensity, allow_pickle=False)
    sio.savemat(
        paths["result.mat"],
        {
            "denoised": denoised_intensity,
            "denoised_intensity": denoised_intensity,
            "denoised_amplitude": amplitude,
            "noisy_intensity": noisy_intensity,
            "input_real": image_real_imag[..., 0],
            "input_imag": image_real_imag[..., 1],
        },
        do_compression=True,
    )

    npy_roundtrip = np.load(paths["denoised.npy"], allow_pickle=False)
    mat_roundtrip = sio.loadmat(paths["result.mat"])["denoised"]
    if not np.array_equal(npy_roundtrip, denoised_intensity):
        raise AssertionError("denoised.npy round-trip mismatch")
    if not np.array_equal(mat_roundtrip, denoised_intensity):
        raise AssertionError("result.mat and denoised.npy differ")

    repository_commit = git_output("rev-parse", "HEAD")
    tracked_clean = (
        subprocess.run(
            ["git", "-C", str(REPOSITORY), "diff", "--quiet"],
            check=False,
        ).returncode
        == 0
        and subprocess.run(
            ["git", "-C", str(REPOSITORY), "diff", "--cached", "--quiet"],
            check=False,
        ).returncode
        == 0
    )
    checkpoint_state = torch.load(
        CHECKPOINT, map_location="cpu", weights_only=True
    )
    parameter_count = int(sum(value.numel() for value in checkpoint_state.values()))
    patch_starts = list(range(0, 1024 - args.patch_size, args.stride_size))
    if patch_starts[-1] + args.patch_size < 1024:
        patch_starts.append(1024 - args.patch_size)
    coverage_count = np.zeros((1024, 1024), dtype=np.uint8)
    for row_start in patch_starts:
        for col_start in patch_starts:
            coverage_count[
                row_start : row_start + args.patch_size,
                col_start : col_start + args.patch_size,
            ] += 1
    if np.any(coverage_count == 0):
        raise AssertionError("Official overlap grid left uncovered output pixels")
    source_files = [
        REPOSITORY / "deepdespeckling" / "merlin" / "merlin_denoiser.py",
        REPOSITORY / "deepdespeckling" / "model.py",
        REPOSITORY / "deepdespeckling" / "denoiser.py",
        REPOSITORY / "deepdespeckling" / "utils" / "utils.py",
        REPOSITORY / "deepdespeckling" / "utils" / "constants.py",
    ]

    gpu_name = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
    output_metadata = {
        name: {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for name, path in paths.items()
    }
    run_record = {
        "method": "MERLIN",
        "scope": "official pretrained Spotlight inference on fixed native Umbra SICD complex ROI",
        "source": {
            "complex_roi_path": str(input_path),
            "complex_roi_sha256": sha256_file(input_path),
            "complex_roi_shape": list(complex_roi.shape),
            "complex_roi_dtype": str(complex_roi.dtype),
            "reference_noisy_intensity_path": str(reference_intensity_path),
            "reference_noisy_intensity_sha256": sha256_file(
                reference_intensity_path
            ),
            "adapter_intensity_exactly_matches_reference": reference_exact,
            "adapter_intensity_reference_max_abs_error": reference_max_abs_error,
            "parent_sicd": {
                "path": "E:\\SAR_Data\\Umbra\\Buenos_Aires_20250131\\2025-01-31-14-10-46_UMBRA-08_SICD.nitf",
                "sha256": "dfe8c9fb6adc1e0dd6efb93b29f076f192976c328037ca519d323221b0da1ac9",
                "mode": "SPOTLIGHT",
                "polarization": "V:V",
            },
            "roi": {
                "row_start": 7456,
                "col_start": 17256,
                "height": 1024,
                "width": 1024,
            },
        },
        "adapter": {
            "complex_to_network_input": "stack(real(S), imag(S), axis=-1) as float32 [H,W,2]",
            "complex_to_noisy_intensity": "real(S)^2 + imag(S)^2 in float32",
            "network_input_shape": list(image_real_imag.shape),
            "network_input_dtype": str(image_real_imag.dtype),
            "input_resized": False,
            "input_cropped_after_fixed_roi": False,
            "scene_wise_minmax_normalization": False,
            "extra_input_scale": None,
            "input_clipped": False,
            "radiometric_calibration_added": False,
            "gdal_import_stub": {
                "used": True,
                "reason": "official numpy-array inference imports an unused TIFF loader at module import; no GDAL function was called",
            },
        },
        "model": {
            "repository": "https://github.com/hi-paris/deepdespeckling",
            "repository_commit": repository_commit,
            "repository_commit_date": git_output("show", "-s", "--format=%cI", "HEAD"),
            "repository_tracked_sources_unchanged": tracked_clean,
            "package_version": importlib.metadata.version("deepdespeckling"),
            "model_name": "spotlight",
            "checkpoint_path": str(CHECKPOINT),
            "checkpoint_bytes": CHECKPOINT.stat().st_size,
            "checkpoint_sha256": sha256_file(CHECKPOINT),
            "checkpoint_tensor_parameter_count": parameter_count,
            "official_pretrained": True,
            "strict_state_dict_loaded": True,
            "source_sha256": {
                str(path.relative_to(REPOSITORY)).replace("\\", "/"): sha256_file(path)
                for path in source_files
            },
        },
        "inference": {
            "patch_size": args.patch_size,
            "stride_size": args.stride_size,
            "patch_grid": {
                "row_starts": patch_starts,
                "col_starts": patch_starts,
                "patch_count": len(patch_starts) ** 2,
                "overlap_averaging": True,
                "coverage_count_min": int(coverage_count.min()),
                "coverage_count_max": int(coverage_count.max()),
                "uncovered_pixel_count": int(np.count_nonzero(coverage_count == 0)),
            },
            "symetrise": True,
            "symetrise_spelling": "official API spelling",
            "symetrise_transform": "for each complex patch: centered 2-D FFT, estimate row and column circular rolls from autocorrelation of reversed mean-magnitude spectral profiles with Gaussian-window selection, roll the spectrum, then inverse FFT",
            "selected_patch_rolls": symmetry_records,
            "internal_network_transform": {
                "per_component_formula": "(log(component^2 + 1e-3) - 2*m) / (2*(M-m))",
                "M": 10.089038980848645,
                "m": -1.429329123112601,
                "epsilon": 0.001,
                "normalization_is_official_fixed_constants_not_scene_minmax": True,
            },
            "official_network_output_domain": "amplitude",
            "network_output_domain": "amplitude",
            "official_amplitude_formula": "sqrt(0.5*(denormalized_real_estimate^2 + denormalized_imag_estimate^2))",
            "figure_scientific_output_domain": "linear intensity",
            "figure_intensity_formula": "official_amplitude^2 exactly once",
            "figure_intensity": "amplitude^2",
            "denoised_npy_domain": "linear intensity",
            "output_clipped": False,
            "official_loader_calls_model_eval": False,
            "architecture_has_dropout_or_batchnorm": False,
            "primary_elapsed_seconds": primary_seconds,
            "repeat_validation": repeat_validation,
        },
        "runtime": {
            "python": platform.python_version(),
            "executable": sys.executable,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device": "cuda:0" if torch.cuda.is_available() else "cpu",
            "gpu": gpu_name,
            "numpy": np.__version__,
            "scipy": importlib.metadata.version("scipy"),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "torch_inference_mode": True,
        },
        "validation": {
            "output_shape_is_1024x1024": denoised_intensity.shape == (1024, 1024),
            "output_all_finite": bool(np.isfinite(denoised_intensity).all()),
            "output_nonnegative": bool(np.all(denoised_intensity >= 0)),
            "denoised_exactly_equals_amplitude_squared": bool(
                np.array_equal(
                    denoised_intensity,
                    np.square(amplitude, dtype=np.float32),
                )
            ),
            "npy_roundtrip_array_equal": bool(
                np.array_equal(npy_roundtrip, denoised_intensity)
            ),
            "mat_npy_array_equal": bool(
                np.array_equal(mat_roundtrip, denoised_intensity)
            ),
            "no_clean_reference": True,
            "psnr_ssim_not_reported": True,
        },
        "statistics": {
            "noisy_linear_intensity": array_stats(noisy_intensity),
            "merlin_official_amplitude": array_stats(amplitude),
            "merlin_linear_intensity": array_stats(denoised_intensity),
            "mean_intensity_ratio_merlin_over_noisy": float(
                denoised_intensity.mean(dtype=np.float64)
                / noisy_intensity.mean(dtype=np.float64)
            ),
        },
        "outputs": output_metadata,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    run_json_path = output_dir / "run.json"
    run_json_path.write_text(
        json.dumps(run_record, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps({
        "output_dir": str(output_dir),
        "denoised_shape": list(denoised_intensity.shape),
        "denoised_dtype": str(denoised_intensity.dtype),
        "denoised_min": float(denoised_intensity.min()),
        "denoised_max": float(denoised_intensity.max()),
        "denoised_mean": float(denoised_intensity.mean(dtype=np.float64)),
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "repeat_validation": repeat_validation,
        "run_json": str(run_json_path),
    }, indent=2))


if __name__ == "__main__":
    main()
