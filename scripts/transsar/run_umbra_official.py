"""Run the official TransSARV2 checkpoint on the fixed Umbra SICD ROI.

The released model is an amplitude-domain model.  Its synthetic-data path
constructs intensity, applies single-look Gamma speckle, then takes ``sqrt``
before the network.  Its real-image demo maps an 8-bit amplitude display ``d``
to ``(d + 1) / 256`` and maps the prediction back with ``256*y - 1``.

For the fixed Umbra ROI there is no 8-bit display DN.  This adapter uses the
same predeclared linear-intensity scale as the other historical-model adapters,
``S=120347.515625`` (the observed ROI maximum), without a min subtraction or
8-bit quantization:

    a       = sqrt(raw_intensity / S)
    x       = (255 * clip(a, 0, 1) + 1) / 256
    dn_hat  = 256 * y - 1
    a_hat   = clip(dn_hat / 255, 0, 1)
    raw_hat = S * a_hat**2

The clipping corresponds to the finite [0, 255] domain of the released real
image entry point.  It is measured and recorded.  No spatial resampling and no
8-bit conversion are performed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import types
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy.io import savemat
import torch


OFFICIAL_COMMIT = "b3ac845f96f2332aa4f1af94b455f71630978b17"
EXPECTED_CHECKPOINT_SHA256 = (
    "cdab8293c7b6ea72ccd192c119b1368b6d661b739d52cc43611cc75755263a70"
)
LEGACY_FIXED_INTENSITY_SCALE = 120347.515625


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=root
        / "output"
        / "cl_sar_umbra_buenos_aires"
        / "figure3_clsar_final"
        / "noisy_intensity.npy",
    )
    parser.add_argument(
        "--parent-run-json",
        type=Path,
        default=root
        / "output"
        / "cl_sar_umbra_buenos_aires"
        / "figure3_clsar_final"
        / "run.json",
    )
    parser.add_argument(
        "--official-repo",
        type=Path,
        default=root / "external" / "TransSAR_official",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=root / "pretrained_models" / "model.pth",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root
        / "output"
        / "umbra_buenos_aires_multimethod_1024"
        / "runs"
        / "transsar",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--intensity-scale",
        type=float,
        default=None,
        help=(
            "Explicit raw-linear-intensity divisor. When omitted, use the exact "
            "observed maximum of the supplied fixed ROI (the legacy Figure 3 rule)."
        ),
    )
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument(
        "--reflect-pad",
        type=int,
        default=96,
        help="Symmetric reflect context. 96 makes 1024+2p align to stride 192.",
    )
    return parser.parse_args()


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def stats(array: np.ndarray) -> Dict[str, Any]:
    x = np.asarray(array)
    percentiles = np.percentile(x, [0, 0.1, 1, 50, 90, 95, 99, 99.7, 99.9, 100])
    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "finite": bool(np.isfinite(x).all()),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "mean": float(np.mean(x, dtype=np.float64)),
        "std": float(np.std(x, dtype=np.float64)),
        "percentiles": {
            key: float(value)
            for key, value in zip(
                ["0", "0.1", "1", "50", "90", "95", "99", "99.7", "99.9", "100"],
                percentiles,
            )
        },
    }


def git_output(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, encoding="utf-8"
    ).strip()


def import_official_model(repo: Path):
    """Import upstream source unchanged; provide only its unused mmcv symbol."""
    # transform_main.py imports ConvModule but never references it.  Avoid a
    # heavyweight binary mmcv install while leaving upstream source untouched.
    if "mmcv" not in sys.modules:
        mmcv = types.ModuleType("mmcv")
        mmcv_cnn = types.ModuleType("mmcv.cnn")
        mmcv_cnn.ConvModule = object
        mmcv.cnn = mmcv_cnn
        sys.modules["mmcv"] = mmcv
        sys.modules["mmcv.cnn"] = mmcv_cnn

    repo_text = str(repo.resolve())
    sys.path.insert(0, repo_text)
    try:
        source = repo / "transform_main.py"
        spec = importlib.util.spec_from_file_location(
            "official_transsar_transform_main", source
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot import official source: {source}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.TransSARV2
    finally:
        if sys.path and sys.path[0] == repo_text:
            sys.path.pop(0)


def load_strict_model(model_class, checkpoint: Path, device: torch.device):
    model = model_class()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state:
        raise TypeError("Official checkpoint is not a non-empty state_dict")
    keys = list(state.keys())
    if not all(isinstance(key, str) for key in keys):
        raise TypeError("Official checkpoint contains non-string keys")

    # The published file was saved from nn.DataParallel.  Removing one uniform
    # wrapper prefix changes no tensor and still uses strict=True below.
    if all(key.startswith("module.") for key in keys):
        state = {key[len("module.") :]: value for key, value in state.items()}
        key_transform = "stripped one uniform leading 'module.' DataParallel prefix"
    else:
        key_transform = "none"
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Strict state load failed: {incompatible}")
    model.eval().to(device)
    return model, key_transform, len(state)


def tile_starts(length: int, tile: int, stride: int) -> List[int]:
    if length < tile:
        raise ValueError(f"Padded length {length} is smaller than tile {tile}")
    if (length - tile) % stride:
        raise ValueError(
            f"Padded length {length} is not aligned: (length-tile) % stride != 0"
        )
    return list(range(0, length - tile + 1, stride))


def cosine_tile_weight(
    tile: int,
    overlap: int,
    row_start: int,
    col_start: int,
    padded_height: int,
    padded_width: int,
) -> np.ndarray:
    """Separable sin^2/cos^2 partition-of-unity overlap weight."""
    if not 0 < overlap < tile:
        raise ValueError("Expected 0 < overlap < tile")
    theta = (np.arange(overlap, dtype=np.float64) + 0.5) * math.pi / (2 * overlap)
    rise = np.sin(theta) ** 2
    fall = np.cos(theta) ** 2
    wy = np.ones(tile, dtype=np.float64)
    wx = np.ones(tile, dtype=np.float64)
    if row_start > 0:
        wy[:overlap] = rise
    if row_start + tile < padded_height:
        wy[-overlap:] = fall
    if col_start > 0:
        wx[:overlap] = rise
    if col_start + tile < padded_width:
        wx[-overlap:] = fall
    return wy[:, None] * wx[None, :]


def infer_tiled(
    model: torch.nn.Module,
    network_input: np.ndarray,
    device: torch.device,
    tile: int,
    overlap: int,
    reflect_pad: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if network_input.ndim != 2:
        raise ValueError(f"Expected HxW input, got {network_input.shape}")
    if reflect_pad < 0 or reflect_pad >= min(network_input.shape):
        raise ValueError("Invalid reflect padding")
    padded = np.pad(
        network_input,
        ((reflect_pad, reflect_pad), (reflect_pad, reflect_pad)),
        mode="reflect",
    ).astype(np.float32, copy=False)
    stride = tile - overlap
    rows = tile_starts(padded.shape[0], tile, stride)
    cols = tile_starts(padded.shape[1], tile, stride)
    accum = np.zeros(padded.shape, dtype=np.float64)
    weight_sum = np.zeros(padded.shape, dtype=np.float64)

    started = time.perf_counter()
    with torch.inference_mode():
        for row in rows:
            for col in cols:
                patch = np.ascontiguousarray(padded[row : row + tile, col : col + tile])
                tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(device)
                pred = model(tensor)
                if tuple(pred.shape) != (1, 1, tile, tile):
                    raise RuntimeError(f"Unexpected model output shape: {tuple(pred.shape)}")
                pred_np = pred[0, 0].detach().cpu().numpy().astype(np.float32, copy=False)
                weight = cosine_tile_weight(
                    tile, overlap, row, col, padded.shape[0], padded.shape[1]
                )
                accum[row : row + tile, col : col + tile] += pred_np * weight
                weight_sum[row : row + tile, col : col + tile] += weight
    if np.any(weight_sum <= 0):
        raise RuntimeError("Overlap blend left uncovered pixels")
    blended = accum / weight_sum
    if reflect_pad:
        cropped = blended[
            reflect_pad : reflect_pad + network_input.shape[0],
            reflect_pad : reflect_pad + network_input.shape[1],
        ]
        cropped_weights = weight_sum[
            reflect_pad : reflect_pad + network_input.shape[0],
            reflect_pad : reflect_pad + network_input.shape[1],
        ]
    else:
        cropped = blended
        cropped_weights = weight_sum
    elapsed = time.perf_counter() - started
    return cropped.astype(np.float32), {
        "tile_size": tile,
        "overlap": overlap,
        "stride": stride,
        "reflect_pad_each_side": reflect_pad,
        "padding_mode": "reflect",
        "padded_shape": list(padded.shape),
        "row_starts": rows,
        "col_starts": cols,
        "tile_count": len(rows) * len(cols),
        "blend": "separable complementary sin^2/cos^2 ramps over each overlap",
        "accumulator_dtype": "float64",
        "cropped_weight_sum_min": float(np.min(cropped_weights)),
        "cropped_weight_sum_max": float(np.max(cropped_weights)),
        "forward_seconds": elapsed,
        "output_shape_after_crop": list(cropped.shape),
    }


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[2]
    input_path = args.input.resolve()
    parent_run_path = args.parent_run_json.resolve()
    official_repo = args.official_repo.resolve()
    checkpoint = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()

    for required in [input_path, parent_run_path, official_repo, checkpoint]:
        if not required.exists():
            raise FileNotFoundError(required)
    commit = git_output(official_repo, "rev-parse", "HEAD")
    if commit != OFFICIAL_COMMIT:
        raise RuntimeError(f"Unexpected official source commit: {commit}")
    source_status = git_output(
        official_repo, "status", "--porcelain", "--untracked-files=no"
    )
    if source_status:
        raise RuntimeError(f"Official source tree is modified:\n{source_status}")
    checkpoint_hash = sha256(checkpoint)
    if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(f"Unexpected checkpoint SHA256: {checkpoint_hash}")

    noisy = np.load(input_path, allow_pickle=False)
    if noisy.dtype != np.float32:
        raise TypeError(f"Expected float32 input, got {noisy.dtype}")
    if noisy.ndim != 2 or min(noisy.shape) < args.tile_size:
        raise ValueError(
            f"Expected a 2-D ROI at least {args.tile_size}x{args.tile_size}, got {noisy.shape}"
        )
    if not np.isfinite(noisy).all() or np.any(noisy < 0):
        raise ValueError("Input must be finite nonnegative linear intensity")

    with parent_run_path.open("r", encoding="utf-8") as handle:
        parent = json.load(handle)
    roi = parent["roi"]
    if [int(roi["height"]), int(roi["width"])] != list(noisy.shape):
        raise RuntimeError("Parent ROI metadata does not match intensity array")
    sicd_path = Path(parent["source"]["path"])
    observed_max = float(np.max(noisy))
    intensity_scale = observed_max if args.intensity_scale is None else float(args.intensity_scale)
    if not math.isfinite(intensity_scale) or intensity_scale <= 0:
        raise ValueError("--intensity-scale must be finite and positive")
    if intensity_scale != observed_max:
        raise RuntimeError(
            f"Fixed scale {intensity_scale} does not match ROI maximum {observed_max}"
        )
    normalized_intensity = (noisy / np.float32(intensity_scale)).astype(
        np.float32
    )
    normalized_amplitude = np.sqrt(
        np.maximum(normalized_intensity, 0.0)
    ).astype(np.float32)
    amplitude_clipped = np.clip(normalized_amplitude, 0.0, 1.0).astype(np.float32)
    # Continuous lift of upstream test.py's exact (uint8_DN + 1)/256 mapping.
    network_input = ((255.0 * amplitude_clipped + 1.0) / 256.0).astype(np.float32)
    if float(network_input.min()) < 1.0 / 256.0 or float(network_input.max()) > 1.0:
        raise RuntimeError("Network input escaped the official [1/256, 1] test domain")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    model_class = import_official_model(official_repo)
    model, key_transform, state_tensor_count = load_strict_model(
        model_class, checkpoint, device
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    network_output, tile_record = infer_tiled(
        model,
        network_input,
        device,
        args.tile_size,
        args.overlap,
        args.reflect_pad,
    )
    if not np.isfinite(network_output).all():
        raise RuntimeError("Model output contains NaN or Inf")

    # Exact continuous inverse of x=(255*a+1)/256, with explicit clipping for
    # values outside the representable amplitude range of the released demo.
    recovered_amplitude_unclipped = (256.0 * network_output - 1.0) / 255.0
    recovered_amplitude = np.clip(recovered_amplitude_unclipped, 0.0, 1.0)
    denoised_normalized_intensity = np.square(
        recovered_amplitude, dtype=np.float32
    )
    denoised = (
        np.float32(intensity_scale) * denoised_normalized_intensity
    ).astype(np.float32)
    if denoised.shape != noisy.shape or not np.isfinite(denoised).all():
        raise RuntimeError("Invalid recovered raw linear-intensity output")

    output_dir.mkdir(parents=True, exist_ok=True)
    denoised_path = output_dir / "denoised.npy"
    input_copy_path = output_dir / "noisy_intensity.npy"
    network_input_path = output_dir / "network_input_amplitude.npy"
    network_output_path = output_dir / "network_output_amplitude.npy"
    network_output_dn_path = output_dir / "network_output_dn_unclipped.npy"
    mat_path = output_dir / "result.mat"
    np.save(denoised_path, denoised, allow_pickle=False)
    np.save(input_copy_path, noisy, allow_pickle=False)
    np.save(network_input_path, network_input, allow_pickle=False)
    np.save(network_output_path, network_output, allow_pickle=False)
    np.save(network_output_dn_path, 256.0 * network_output - 1.0, allow_pickle=False)
    savemat(
        mat_path,
        {
            "noisy": noisy,
            "denoised": denoised,
            "network_input_amplitude": network_input,
            "network_output_amplitude": network_output,
            "network_output_dn_unclipped": 256.0 * network_output - 1.0,
        },
        do_compression=True,
    )

    source_files = [
        "transform_main.py",
        "arch/trans_basenetworks.py",
        "create_synthetic_data.py",
        "utils.py",
        "test.py",
    ]
    artifacts = [
        denoised_path,
        input_copy_path,
        network_input_path,
        network_output_path,
        network_output_dn_path,
        mat_path,
    ]
    report: Dict[str, Any] = {
        "method": "Trans-SAR (official TransSARV2 pretrained checkpoint)",
        "status": "completed",
        "scientific_output": {
            "file": "denoised.npy",
            "domain": "original SICD raw linear intensity, float32",
            "not_normalized_or_display_domain": True,
            "shape": list(denoised.shape),
        },
        "input": {
            "path": str(input_path),
            "sha256": sha256(input_path),
            "domain": "I = real(S)^2 + imag(S)^2, original SICD raw linear intensity",
            "stats": stats(noisy),
        },
        "source_sicd": {
            "path": str(sicd_path),
            "sha256_from_verified_parent_manifest": parent["source"].get(
                "sha256",
                parent["source"].get("download_manifest", {}).get("sicd_sha256"),
            ),
            "roi": roi,
        },
        "official_model": {
            "repository": "https://github.com/malshaV/sar_transformer",
            "local_repository": str(official_repo),
            "commit": commit,
            "tracked_sources_unchanged": True,
            "source_sha256": {
                relative: sha256(official_repo / Path(relative)) for relative in source_files
            },
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_hash,
            "checkpoint_matches_official_clone": (
                checkpoint_hash
                == sha256(official_repo / "pretrained_models" / "model.pth")
            ),
            "checkpoint_state_tensor_count": state_tensor_count,
            "state_dict_key_transform": key_transform,
            "strict_state_dict_loaded": True,
            "architecture": "TransSARV2",
            "registered_parameters": parameter_count,
            "eval_mode": not model.training,
            "upstream_unused_import_compatibility": (
                "runtime-only mmcv.cnn.ConvModule placeholder; symbol is imported but never "
                "referenced; official source file remained unchanged"
            ),
        },
        "official_numeric_domain_evidence": {
            "synthetic_generation": (
                "create_synthetic_data.py stores clean=((gray+1)/256)^2 and "
                "noisy=clean*Gamma(shape=1,scale=1), both as intensity"
            ),
            "training_loader": (
                "utils.py applies sqrt(noisy+1e-10) and sqrt(clean+1e-10) before "
                "TransSARV2, so the released checkpoint is amplitude-domain"
            ),
            "training_loader_quantization": (
                "The upstream loader then calls torchvision F.to_pil_image without "
                "mode='F' and later F.to_tensor; under the pinned torchvision path "
                "this is an 8-bit amplitude quantization. The present adapter does not "
                "repeat that quantization, as explicitly required for this comparison."
            ),
            "real_test_input": "test.py uses x=(float32(uint8_gray)+1)/256",
            "real_test_output": "test.py writes 256*y-1 as grayscale DN",
            "network_output_activation": "tanh",
            "adapter_interpretation": (
                "continuous-valued lift of the official real-test DN transform; expected "
                "network range is [1/256,1] normalized amplitude"
            ),
        },
        "domain_conversion": {
            "purpose": (
                "Fixed reversible bridge from this ROI's raw linear intensity to the "
                "released 8-bit amplitude convention, shared with the historical-model "
                "adapters"
            ),
            "forward": [
                f"normalized_intensity = raw_intensity/{intensity_scale}",
                "normalized_amplitude = sqrt(max(normalized_intensity,0))",
                "amplitude_clipped = clip(normalized_amplitude,0,1)",
                "network_input = (255*amplitude_clipped+1)/256",
            ],
            "inverse": [
                "DN_hat_unclipped = 256*network_output-1",
                "amplitude_hat_unclipped = DN_hat_unclipped/255",
                "amplitude_hat = clip(amplitude_hat_unclipped,0,1)",
                f"denoised_raw_intensity = {intensity_scale}*amplitude_hat^2",
            ],
            "fixed_intensity_scale": intensity_scale,
            "scale_origin": "predeclared observed maximum of the fixed 1024x1024 ROI",
            "minimum_subtraction": False,
            "per_image_minmax": False,
            "percentile_normalization": False,
            "eight_bit_quantization": False,
            "spatial_resampling": False,
            "input_fraction_normalized_amplitude_above_1_clipped": float(
                np.mean(normalized_amplitude > 1.0)
            ),
            "input_fraction_normalized_amplitude_below_0_clipped": 0.0,
            "output_fraction_below_representable_amplitude_clipped": float(
                np.mean(recovered_amplitude_unclipped < 0.0)
            ),
            "output_fraction_above_representable_amplitude_clipped": float(
                np.mean(recovered_amplitude_unclipped > 1.0)
            ),
            "network_input_stats": stats(network_input),
            "network_output_stats": stats(network_output),
            "network_output_dn_unclipped_stats": stats(256.0 * network_output - 1.0),
            "denoised_normalized_intensity_stats": stats(
                denoised_normalized_intensity
            ),
            "cross_sensor_caveat": (
                "The official checkpoint was trained on single-look synthetic speckle "
                "over BSDS grayscale reflectivity, not Umbra X-band SICD. The fixed-scale "
                "mapping is reproducible and invertible inside the official amplitude "
                "range, but it does not remove that domain shift."
            ),
        },
        "tiling": tile_record,
        "runtime": {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "inference_dtype": "float32",
            "autocast_or_amp": False,
            "deterministic_seed": 0,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "tf32": False,
        },
        "output_stats": stats(denoised),
        "outputs": {
            path.name: {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in artifacts
        },
        "adapter": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
        },
    }
    run_json_path = output_dir / "run.json"
    with run_json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(json.dumps({
        "output": str(denoised_path),
        "output_stats": report["output_stats"],
        "input_saturation_fraction": report["domain_conversion"]
        ["input_fraction_normalized_amplitude_above_1_clipped"],
        "output_low_clip_fraction": report["domain_conversion"]
        ["output_fraction_below_representable_amplitude_clipped"],
        "tiles": tile_record["tile_count"],
        "seconds": tile_record["forward_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
