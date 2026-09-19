#!/usr/bin/env python3
"""MERLIN Spotlight inference with positive-Hann weighted overlap-add.

The official per-patch preprocessing and network forward passes are unchanged.
Only the assembly of overlapping patch predictions differs from the official
uniform count average.  During the same forward pass this script also rebuilds
the official uniform result and requires bitwise equality with the previously
saved stride-64 run.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import subprocess
import sys
import time
import types
from datetime import datetime, timezone

import numpy as np
from scipy import io as sio
import torch


ROOT = Path(__file__).resolve().parents[2]
REPOSITORY = ROOT / "external" / "deepdespeckling"
CHECKPOINT = (
    REPOSITORY
    / "deepdespeckling"
    / "merlin"
    / "saved_models"
    / "spotlight.pth"
)
INPUT = (
    ROOT
    / "output"
    / "cl_sar_umbra_buenos_aires"
    / "figure3_clsar_final"
    / "sicd_complex_roi.npy"
)
REFERENCE_NOISY = INPUT.with_name("noisy_intensity.npy")
REFERENCE_UNIFORM = (
    ROOT
    / "output"
    / "umbra_buenos_aires_multimethod_1024"
    / "runs"
    / "merlin_stride64"
    / "denoised.npy"
)
OUTPUT = (
    ROOT
    / "output"
    / "umbra_buenos_aires_multimethod_1024"
    / "runs"
    / "merlin_stride64_weighted"
)
PATCH_SIZE = 256
STRIDE_SIZE = 64


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
    values = array[np.isfinite(array)]
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "finite_fraction": float(np.isfinite(array).mean()),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean(dtype=np.float64)),
        "median": float(np.median(values)),
        "percentiles_1_50_99_99p7": [
            float(value)
            for value in np.percentile(values, [1.0, 50.0, 99.0, 99.7])
        ],
        "zero_count": int(np.count_nonzero(values == 0)),
        "negative_count": int(np.count_nonzero(values < 0)),
    }


def install_unused_gdal_import_stub() -> None:
    if "osgeo" in sys.modules:
        return
    osgeo_module = types.ModuleType("osgeo")
    gdal_module = types.ModuleType("gdal")
    osgeo_module.gdal = gdal_module
    sys.modules["osgeo"] = osgeo_module
    sys.modules["osgeo.gdal"] = gdal_module


def positive_hann_window(size: int) -> tuple[np.ndarray, np.ndarray]:
    """Return a separable Hann window whose endpoints are strictly positive."""

    one_dimensional = np.hanning(size + 2)[1:-1].astype(np.float64)
    two_dimensional = np.outer(one_dimensional, one_dimensional)
    two_dimensional /= two_dimensional.max()
    return one_dimensional, two_dimensional


def run_once(image_real_imag: np.ndarray) -> dict[str, np.ndarray | float | list[int]]:
    install_unused_gdal_import_stub()
    from deepdespeckling.merlin import merlin_denoiser as official_merlin

    denoiser = official_merlin.MerlinDenoiser(
        model_name="spotlight", symetrise=True
    )
    model = denoiser.load_model(patch_size=PATCH_SIZE)
    reshaped = np.array(image_real_imag).reshape(
        1, image_real_imag.shape[0], image_real_imag.shape[1], 2
    )
    noisy_amplitude, real_part, imag_part = denoiser.preprocess_noisy_image(
        reshaped
    )
    row_starts = denoiser.initialize_axis_range(
        image_real_imag.shape[0], PATCH_SIZE, STRIDE_SIZE
    )
    col_starts = denoiser.initialize_axis_range(
        image_real_imag.shape[1], PATCH_SIZE, STRIDE_SIZE
    )
    one_dimensional_window, window = positive_hann_window(PATCH_SIZE)
    window_4d = window.reshape(1, PATCH_SIZE, PATCH_SIZE, 1)

    uniform_real = np.zeros(real_part.shape, dtype=np.float64)
    uniform_imag = np.zeros(real_part.shape, dtype=np.float64)
    uniform_count = np.zeros(real_part.shape, dtype=np.float64)
    weighted_real = np.zeros(real_part.shape, dtype=np.float64)
    weighted_imag = np.zeros(real_part.shape, dtype=np.float64)
    weighted_count = np.zeros(real_part.shape, dtype=np.float64)

    started = time.perf_counter()
    with torch.inference_mode():
        for row_start in row_starts:
            for col_start in col_starts:
                patch_real = real_part[
                    :,
                    row_start : row_start + PATCH_SIZE,
                    col_start : col_start + PATCH_SIZE,
                    :,
                ]
                patch_imag = imag_part[
                    :,
                    row_start : row_start + PATCH_SIZE,
                    col_start : col_start + PATCH_SIZE,
                    :,
                ]
                patch_real, patch_imag = (
                    official_merlin.symetrise_real_and_imaginary_parts(
                        patch_real, patch_imag
                    )
                )
                tensor_real = torch.tensor(
                    patch_real, device=denoiser.device, dtype=torch.float32
                )
                tensor_imag = torch.tensor(
                    patch_imag, device=denoiser.device, dtype=torch.float32
                )
                tensor_real = (
                    torch.log(torch.square(tensor_real) + 1e-3) - 2 * denoiser.m
                ) / (2 * (denoiser.M - denoiser.m))
                tensor_imag = (
                    torch.log(torch.square(tensor_imag) + 1e-3) - 2 * denoiser.m
                ) / (2 * (denoiser.M - denoiser.m))

                predicted_real = np.moveaxis(
                    model.forward(tensor_real).cpu().detach().numpy(), 1, -1
                )
                predicted_imag = np.moveaxis(
                    model.forward(tensor_imag).cpu().detach().numpy(), 1, -1
                )
                target = (
                    slice(None),
                    slice(row_start, row_start + PATCH_SIZE),
                    slice(col_start, col_start + PATCH_SIZE),
                    slice(None),
                )
                uniform_real[target] += predicted_real
                uniform_imag[target] += predicted_imag
                uniform_count[target] += 1.0
                weighted_real[target] += predicted_real * window_4d
                weighted_imag[target] += predicted_imag * window_4d
                weighted_count[target] += window_4d
    elapsed = time.perf_counter() - started

    uniform_amplitude, _, _ = denoiser.preprocess_denoised_image(
        uniform_real, uniform_imag, uniform_count
    )
    weighted_amplitude, _, _ = denoiser.preprocess_denoised_image(
        weighted_real, weighted_imag, weighted_count
    )
    return {
        "noisy_amplitude": np.asarray(noisy_amplitude, dtype=np.float32),
        "uniform_amplitude": np.asarray(uniform_amplitude, dtype=np.float32),
        "weighted_amplitude": np.asarray(weighted_amplitude, dtype=np.float32),
        "uniform_count": np.squeeze(uniform_count),
        "weighted_count": np.squeeze(weighted_count),
        "one_dimensional_window": one_dimensional_window,
        "two_dimensional_window": window,
        "row_starts": [int(value) for value in row_starts],
        "col_starts": [int(value) for value in col_starts],
        "elapsed_seconds": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--reference-noisy", type=Path, default=REFERENCE_NOISY)
    parser.add_argument(
        "--reference-uniform-stride64", type=Path, default=REFERENCE_UNIFORM
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()

    input_path = args.input.resolve()
    reference_noisy_path = args.reference_noisy.resolve()
    reference_uniform_path = args.reference_uniform_stride64.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    complex_roi = np.load(input_path, allow_pickle=False)
    if complex_roi.shape != (1024, 1024) or complex_roi.dtype != np.complex64:
        raise ValueError(f"Unexpected fixed ROI contract: {complex_roi.shape}, {complex_roi.dtype}")
    if not np.isfinite(complex_roi).all():
        raise ValueError("Non-finite complex input")
    image_real_imag = np.stack(
        (complex_roi.real, complex_roi.imag), axis=-1
    ).astype(np.float32, copy=False)
    noisy_intensity = (
        np.square(image_real_imag[..., 0], dtype=np.float32)
        + np.square(image_real_imag[..., 1], dtype=np.float32)
    ).astype(np.float32, copy=False)
    reference_noisy = np.load(reference_noisy_path, allow_pickle=False)
    if not np.array_equal(noisy_intensity, reference_noisy):
        raise AssertionError("Fixed noisy intensity mismatch")

    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    primary = run_once(image_real_imag)
    uniform_intensity = np.square(
        primary["uniform_amplitude"], dtype=np.float32
    )
    weighted_amplitude = np.asarray(primary["weighted_amplitude"], dtype=np.float32)
    weighted_intensity = np.square(weighted_amplitude, dtype=np.float32)
    reference_uniform = np.load(reference_uniform_path, allow_pickle=False)
    uniform_exact = bool(np.array_equal(uniform_intensity, reference_uniform))
    uniform_max_abs = float(
        np.max(
            np.abs(
                uniform_intensity.astype(np.float64)
                - reference_uniform.astype(np.float64)
            )
        )
    )
    if not uniform_exact:
        raise AssertionError(
            "The same patch predictions did not reproduce official stride64 uniform "
            f"aggregation bitwise; max abs={uniform_max_abs}"
        )

    repeat = run_once(image_real_imag)
    repeat_weighted_intensity = np.square(
        np.asarray(repeat["weighted_amplitude"], dtype=np.float32),
        dtype=np.float32,
    )
    repeat_exact = bool(np.array_equal(weighted_intensity, repeat_weighted_intensity))
    repeat_max_abs = float(
        np.max(
            np.abs(
                weighted_intensity.astype(np.float64)
                - repeat_weighted_intensity.astype(np.float64)
            )
        )
    )
    if not repeat_exact:
        raise AssertionError(f"Weighted repeat mismatch; max abs={repeat_max_abs}")
    if weighted_intensity.shape != (1024, 1024):
        raise AssertionError(weighted_intensity.shape)
    if not np.isfinite(weighted_intensity).all() or np.any(weighted_intensity < 0):
        raise ValueError("Invalid weighted MERLIN output")
    weighted_count = np.asarray(primary["weighted_count"], dtype=np.float64)
    if np.any(weighted_count <= 0):
        raise AssertionError("Weighted overlap has zero/nonpositive coverage")

    paths = {
        "denoised.npy": output_dir / "denoised.npy",
        "merlin_amplitude.npy": output_dir / "merlin_amplitude.npy",
        "merlin_input_real_imag.npy": output_dir / "merlin_input_real_imag.npy",
        "noisy_intensity.npy": output_dir / "noisy_intensity.npy",
        "overlap_weight_sum.npy": output_dir / "overlap_weight_sum.npy",
        "uniform_reconstruction_intensity.npy": output_dir
        / "uniform_reconstruction_intensity.npy",
        "result.mat": output_dir / "result.mat",
    }
    np.save(paths["denoised.npy"], weighted_intensity, allow_pickle=False)
    np.save(paths["merlin_amplitude.npy"], weighted_amplitude, allow_pickle=False)
    np.save(paths["merlin_input_real_imag.npy"], image_real_imag, allow_pickle=False)
    np.save(paths["noisy_intensity.npy"], noisy_intensity, allow_pickle=False)
    np.save(paths["overlap_weight_sum.npy"], weighted_count, allow_pickle=False)
    np.save(
        paths["uniform_reconstruction_intensity.npy"],
        uniform_intensity,
        allow_pickle=False,
    )
    sio.savemat(
        paths["result.mat"],
        {
            "denoised": weighted_intensity,
            "denoised_intensity": weighted_intensity,
            "denoised_amplitude": weighted_amplitude,
            "noisy_intensity": noisy_intensity,
            "input_real": image_real_imag[..., 0],
            "input_imag": image_real_imag[..., 1],
            "overlap_weight_sum": weighted_count,
            "uniform_reconstruction_intensity": uniform_intensity,
        },
        do_compression=True,
    )
    npy_roundtrip = np.load(paths["denoised.npy"], allow_pickle=False)
    mat_roundtrip = sio.loadmat(paths["result.mat"])["denoised"]
    if not np.array_equal(npy_roundtrip, weighted_intensity):
        raise AssertionError("Weighted NPY roundtrip mismatch")
    if not np.array_equal(mat_roundtrip, weighted_intensity):
        raise AssertionError("Weighted MAT/NPY mismatch")

    checkpoint_state = torch.load(
        CHECKPOINT, map_location="cpu", weights_only=True
    )
    parameter_count = int(sum(value.numel() for value in checkpoint_state.values()))
    source_files = [
        REPOSITORY / "deepdespeckling" / "merlin" / "merlin_denoiser.py",
        REPOSITORY / "deepdespeckling" / "model.py",
        REPOSITORY / "deepdespeckling" / "denoiser.py",
        REPOSITORY / "deepdespeckling" / "utils" / "utils.py",
        REPOSITORY / "deepdespeckling" / "utils" / "constants.py",
    ]
    output_metadata = {
        name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for name, path in paths.items()
    }
    window_1d = np.asarray(primary["one_dimensional_window"])
    window_2d = np.asarray(primary["two_dimensional_window"])
    uniform_count = np.asarray(primary["uniform_count"])
    run_record = {
        "method": "MERLIN",
        "variant": "spotlight_stride64_positive_hann_weighted_overlap",
        "scope": "same official MERLIN Spotlight patch predictions; adapter changes overlap aggregation only",
        "source": {
            "complex_roi_path": str(input_path),
            "complex_roi_sha256": sha256_file(input_path),
            "shape": list(complex_roi.shape),
            "dtype": str(complex_roi.dtype),
            "reference_noisy_intensity_path": str(reference_noisy_path),
            "noisy_intensity_exact": bool(np.array_equal(noisy_intensity, reference_noisy)),
            "parent_sicd_sha256": "dfe8c9fb6adc1e0dd6efb93b29f076f192976c328037ca519d323221b0da1ac9",
            "roi": {"row_start": 7456, "col_start": 17256, "height": 1024, "width": 1024},
        },
        "model": {
            "repository": "https://github.com/hi-paris/deepdespeckling",
            "repository_commit": git_output("rev-parse", "HEAD"),
            "package_version": importlib.metadata.version("deepdespeckling"),
            "model_name": "spotlight",
            "checkpoint_path": str(CHECKPOINT),
            "checkpoint_sha256": sha256_file(CHECKPOINT),
            "checkpoint_bytes": CHECKPOINT.stat().st_size,
            "checkpoint_tensor_parameter_count": parameter_count,
            "strict_state_dict_loaded": True,
            "official_pretrained": True,
            "source_sha256": {
                str(path.relative_to(REPOSITORY)).replace("\\", "/"): sha256_file(path)
                for path in source_files
            },
        },
        "inference": {
            "patch_size": PATCH_SIZE,
            "stride_size": STRIDE_SIZE,
            "row_starts": primary["row_starts"],
            "col_starts": primary["col_starts"],
            "patch_count": len(primary["row_starts"]) * len(primary["col_starts"]),
            "symetrise": True,
            "network_patch_predictions_changed": False,
            "official_per_patch_preprocessing_changed": False,
            "network_output_domain": "amplitude",
            "figure_intensity": "amplitude^2",
            "denoised_npy_domain": "linear intensity",
            "scene_wise_minmax_normalization": False,
            "radiometric_gain_matching": False,
            "input_resized": False,
            "input_scaled_or_clipped": False,
            "output_scaled_or_clipped": False,
            "primary_elapsed_seconds": primary["elapsed_seconds"],
            "repeat_elapsed_seconds": repeat["elapsed_seconds"],
        },
        "overlap_adapter": {
            "official_baseline": "uniform sum divided by integer coverage count",
            "weighted_variant": "sum(window * patch_prediction) divided by sum(window)",
            "window_name": "separable positive Hann",
            "one_dimensional_formula": "numpy.hanning(patch_size + 2)[1:-1]",
            "two_dimensional_formula": "outer(window_1d, window_1d), normalized to max 1",
            "rationale_for_interior_samples": "pure Hann endpoints are zero; dropping those endpoints keeps every image-border pixel strictly positively covered without an arbitrary epsilon floor",
            "window_1d": {
                "length": int(window_1d.size),
                "min": float(window_1d.min()),
                "max": float(window_1d.max()),
                "sum": float(window_1d.sum()),
            },
            "window_2d": {
                "shape": list(window_2d.shape),
                "min": float(window_2d.min()),
                "max": float(window_2d.max()),
                "sum": float(window_2d.sum()),
            },
            "uniform_coverage": {
                "min": float(uniform_count.min()),
                "max": float(uniform_count.max()),
                "zero_count": int(np.count_nonzero(uniform_count == 0)),
            },
            "weighted_coverage": {
                "min": float(weighted_count.min()),
                "max": float(weighted_count.max()),
                "mean": float(weighted_count.mean()),
                "percentiles_1_50_99": [
                    float(value)
                    for value in np.percentile(weighted_count, [1, 50, 99])
                ],
                "nonpositive_count": int(np.count_nonzero(weighted_count <= 0)),
            },
        },
        "validation": {
            "same_pass_uniform_reconstruction_reference_path": str(reference_uniform_path),
            "same_pass_uniform_reconstruction_bitwise_equal": uniform_exact,
            "same_pass_uniform_reconstruction_max_abs_error": uniform_max_abs,
            "weighted_repeat_bitwise_equal": repeat_exact,
            "weighted_repeat_max_abs_error": repeat_max_abs,
            "weighted_repeat_in_memory_sha256_primary": hashlib.sha256(
                weighted_intensity.tobytes(order="C")
            ).hexdigest(),
            "weighted_repeat_in_memory_sha256_repeat": hashlib.sha256(
                repeat_weighted_intensity.tobytes(order="C")
            ).hexdigest(),
            "denoised_exactly_equals_amplitude_squared": bool(
                np.array_equal(
                    weighted_intensity,
                    np.square(weighted_amplitude, dtype=np.float32),
                )
            ),
            "shape_is_1024x1024": weighted_intensity.shape == (1024, 1024),
            "all_finite": bool(np.isfinite(weighted_intensity).all()),
            "nonnegative": bool(np.all(weighted_intensity >= 0)),
            "mat_npy_array_equal": bool(np.array_equal(mat_roundtrip, weighted_intensity)),
            "npy_roundtrip_array_equal": bool(np.array_equal(npy_roundtrip, weighted_intensity)),
            "no_clean_reference": True,
            "psnr_ssim_not_reported": True,
        },
        "statistics": {
            "noisy_linear_intensity": array_stats(noisy_intensity),
            "weighted_merlin_amplitude": array_stats(weighted_amplitude),
            "weighted_merlin_linear_intensity": array_stats(weighted_intensity),
            "uniform_stride64_linear_intensity": array_stats(uniform_intensity),
        },
        "runtime": {
            "python": platform.python_version(),
            "executable": sys.executable,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device": "cuda:0" if torch.cuda.is_available() else "cpu",
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "numpy": np.__version__,
            "scipy": importlib.metadata.version("scipy"),
            "torch_inference_mode": True,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
        },
        "outputs": output_metadata,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    run_json = output_dir / "run.json"
    run_json.write_text(
        json.dumps(run_record, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "denoised_shape": list(weighted_intensity.shape),
                "denoised_min": float(weighted_intensity.min()),
                "denoised_max": float(weighted_intensity.max()),
                "denoised_mean": float(weighted_intensity.mean(dtype=np.float64)),
                "uniform_reference_bitwise_equal": uniform_exact,
                "weighted_repeat_bitwise_equal": repeat_exact,
                "weighted_repeat_max_abs_error": repeat_max_abs,
                "weighted_coverage_min": float(weighted_count.min()),
                "weighted_coverage_max": float(weighted_count.max()),
                "run_json": str(run_json),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
