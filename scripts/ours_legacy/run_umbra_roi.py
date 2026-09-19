"""Run historical Ours checkpoints on one native Umbra linear-intensity ROI.

This is NOT the current intensity_v1 model or encoder-frozen AMS protocol.
The explicit fixed scale converts raw intensity to the historical normalized
amplitude domain. No percentile clipping, affine offset, resizing, or image
quantization is used. Outputs return to the source linear-intensity scale.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.io import savemat
import torch
import timm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from numeric_domain import LEGACY_AMPLITUDE_DOMAIN, prepare_synthetic_pair
from transform_main import TransSARV2_DualFreqNG_Bottle


CHECKPOINTS = {
    "base": (
        "TransSARV2_DualFreqNG_Bottle_best_e47.pth",
        "cafe9c69f045defa3b199ae017efd8ca44829934ceffcebe868494360484430d",
        "Historical Ours-base (legacy amplitude, e47)",
    ),
    "ams": (
        "TransSARV2_DualFreqNG_Bottle_AMS_best_e8.pth",
        "c82fa45dc9ce747b995a474b75ca868a04c76ba2257ee5f340fa56989e383595",
        "Historical Ours+AMS (legacy amplitude, all-parameter adaptation, e8)",
    ),
}
TILE, OVERLAP, STRIDE, HALO = 256, 64, 192, 32


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def stats(array: np.ndarray) -> dict:
    return {
        "shape": list(array.shape), "dtype": str(array.dtype),
        "finite": bool(np.isfinite(array).all()),
        "minimum": float(array.min()), "maximum": float(array.max()),
        "mean": float(array.mean(dtype=np.float64)),
        "negative_count": int(np.count_nonzero(array < 0)),
        "negative_fraction": float(np.mean(array < 0)),
        "zero_count": int(np.count_nonzero(array == 0)),
        "above_one_count": int(np.count_nonzero(array > 1)),
    }


def layout(shape: tuple[int, int]):
    counts = [max(1, math.ceil((n + 2 * HALO - TILE) / STRIDE) + 1) for n in shape]
    padded_shape = tuple((n - 1) * STRIDE + TILE for n in counts)
    padding = tuple((HALO, padded_shape[i] - shape[i] - HALO) for i in range(2))
    coordinates = [(y * STRIDE, x * STRIDE) for y in range(counts[0]) for x in range(counts[1])]
    ramp = (0.5 - 0.5 * np.cos(np.pi * (np.arange(OVERLAP) + 0.5) / OVERLAP)).astype(np.float32)
    axis = np.ones(TILE, dtype=np.float32)
    axis[:OVERLAP], axis[-OVERLAP:] = ramp, ramp[::-1]
    weight = np.maximum(axis[:, None] * axis[None, :], 1e-6)
    denominator = np.zeros(padded_shape, dtype=np.float32)
    for y, x in coordinates:
        denominator[y:y + TILE, x:x + TILE] += weight
    if not np.all(denominator > 0):
        raise RuntimeError("Incomplete tile coverage")
    return padding, coordinates, weight, denominator


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=CHECKPOINTS, required=True)
    parser.add_argument("--intensity-scale", type=float, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    start = time.perf_counter()
    source, output = args.input.resolve(), args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing run: {output}")
    intensity = np.load(source, allow_pickle=False)
    if intensity.shape != (1024, 1024) or intensity.dtype != np.float32:
        raise ValueError("Expected a native 1024x1024 float32 linear-intensity NPY")
    if not np.isfinite(intensity).all() or np.any(intensity < 0):
        raise ValueError("Intensity must be finite and nonnegative")
    scale = float(args.intensity_scale)
    if not math.isfinite(scale) or scale <= 0 or float(intensity.max()) > scale:
        raise ValueError("Explicit intensity scale must be finite, positive, and cover every input value")
    normalized_intensity = (intensity / np.float32(scale)).astype(np.float32)
    # Use the canonical legacy-domain conversion after explicit no-clipping
    # range validation. The second return is identical and is not a clean GT.
    amplitude, _ = prepare_synthetic_pair(
        normalized_intensity, normalized_intensity, LEGACY_AMPLITUDE_DOMAIN
    )
    if not np.array_equal(amplitude, np.sqrt(normalized_intensity)):
        raise AssertionError("Unexpected historical amplitude transform")
    reconstructed_input = amplitude.astype(np.float64) ** 2 * scale
    roundtrip_error = float(np.max(np.abs(reconstructed_input - intensity)))
    np.testing.assert_allclose(reconstructed_input, intensity, rtol=2e-7, atol=1e-9)
    checkpoint_name, expected_hash, label = CHECKPOINTS[args.variant]
    checkpoint = ROOT / "trained_models" / checkpoint_name
    checkpoint_hash = sha256(checkpoint)
    if checkpoint_hash != expected_hash:
        raise ValueError(f"Historical checkpoint checksum mismatch: {checkpoint}")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not all(torch.is_tensor(v) for v in state.values()):
        raise ValueError("Expected the audited historical tensor-only state_dict")
    model = TransSARV2_DualFreqNG_Bottle(numeric_domain=LEGACY_AMPLITUDE_DOMAIN)
    incompatible = model.load_state_dict(state, strict=True)
    if not isinstance(model.log_branch.active, torch.nn.Tanh) or not isinstance(model.active, torch.nn.Tanh):
        raise AssertionError("Historical activation path was changed")
    parameter_count = sum(p.numel() for p in model.parameters())
    if parameter_count != 39_352_778:
        raise AssertionError("Historical architecture parameter count mismatch")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.manual_seed(42)
    torch.set_num_threads(4)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats(device)
    model.to(device).eval()
    padding, coordinates, weight, denominator = layout(amplitude.shape)
    padded = np.pad(amplitude, padding, mode="reflect")
    accumulation = np.zeros_like(padded)
    identity = np.zeros_like(padded)
    tile_negatives = 0
    tile_samples = 0
    with torch.inference_mode():
        for index, (y, x) in enumerate(coordinates, 1):
            tile_input = padded[y:y + TILE, x:x + TILE].copy()
            tensor = torch.from_numpy(tile_input)[None, None].to(device)
            prediction = model(tensor).detach().float().cpu().numpy()[0, 0]
            if prediction.shape != (TILE, TILE) or not np.isfinite(prediction).all():
                raise RuntimeError(f"Invalid network output on tile {index}")
            tile_negatives += int(np.count_nonzero(prediction < 0))
            tile_samples += prediction.size
            accumulation[y:y + TILE, x:x + TILE] += prediction * weight
            identity[y:y + TILE, x:x + TILE] += tile_input * weight
            if index == 1 or index % 6 == 0:
                print(f"{label}: tile {index}/{len(coordinates)}", flush=True)
    h, w = amplitude.shape
    raw_amplitude = (accumulation / denominator)[HALO:HALO + h, HALO:HALO + w].copy()
    identity_result = (identity / denominator)[HALO:HALO + h, HALO:HALO + w]
    identity_error = float(np.max(np.abs(identity_result - amplitude)))
    if identity_error > 1e-6:
        raise AssertionError(f"Identity tile reconstruction error: {identity_error}")
    if not np.isfinite(raw_amplitude).all() or float(raw_amplitude.max()) > 1.0 + 1e-6:
        raise RuntimeError("Blended legacy tanh output is invalid")
    # Tanh can be negative; retain the complete raw prediction before applying
    # the physical nonnegative-amplitude projection. Do not square negatives.
    nonnegative_amplitude = np.maximum(raw_amplitude, 0).astype(np.float32)
    denoised = (nonnegative_amplitude.astype(np.float64) ** 2 * scale).astype(np.float32)
    if denoised.shape != intensity.shape or not np.isfinite(denoised).all():
        raise RuntimeError("Invalid restored linear intensity")
    output.mkdir(parents=True, exist_ok=False)
    arrays = {
        "noisy_intensity": intensity,
        "normalized_intensity": normalized_intensity,
        "normalized_amplitude_input": amplitude,
        "network_raw_amplitude": raw_amplitude,
        "nonnegative_amplitude": nonnegative_amplitude,
        "denoised": denoised,
    }
    for name, array in arrays.items():
        np.save(output / f"{name}.npy", array, allow_pickle=False)
    savemat(output / "result.mat", {**arrays, "intensity_scale": scale}, do_compression=True)
    report = {
        "complete": True, "method": label, "variant": args.variant,
        "historical": True, "formal_current_paper_model": False,
        "numeric_domain": LEGACY_AMPLITUDE_DOMAIN,
        "scope": "historical cross-sensor exploratory inference; no retraining",
        "limitations": [
            "Not the current intensity_v1 architecture semantics",
            "Historical AMS is all-parameter adaptation, not encoder-frozen AMS",
            "Historical real-data scripts used inconsistent stored-pixel normalization; this run explicitly uses the documented legacy amplitude domain",
            "The fixed ROI-observed peak scale is an explicit transfer adapter, not training-time physical radiometric calibration",
            "Overlap tiles limit context and Fourier support to 256x256, so this is not a whole-image 1024x1024 forward",
        ],
        "input": {"path": str(source), "sha256": sha256(source), **stats(intensity)},
        "normalization": {
            "intensity_scale": scale, "offset": 0.0,
            "scale_provenance": "explicit fixed ROI observed peak agreed for this multimethod experiment",
            "forward": "A=sqrt(I/S)",
            "inverse": "I_hat=S*max(A_hat,0)^2 after amplitude-domain tile blending",
            "input_clipped": False, "percentile_normalization": False,
            "per_tile_normalization": False, "min_max_normalization": False,
            "resizing": False, "quantization": False,
            "input_roundtrip_max_absolute_error": roundtrip_error,
        },
        "checkpoint": {
            "path": str(checkpoint), "sha256": checkpoint_hash,
            "strict_load": True, "missing_keys": list(incompatible.missing_keys),
            "unexpected_keys": list(incompatible.unexpected_keys),
            "state_dict_tensor_count": len(state), "parameter_count": parameter_count,
            "constructor": {"class": "TransSARV2_DualFreqNG_Bottle", "numeric_domain": LEGACY_AMPLITUDE_DOMAIN},
        },
        "tiling": {
            "tile": TILE, "overlap": OVERLAP, "stride": STRIDE, "halo": HALO,
            "padding": padding, "padding_mode": "reflect", "tiles": len(coordinates),
            "blend": "positive separable cosine, sum weighted normalized amplitudes / sum weights",
            "identity_stitch_max_absolute_error": identity_error, "batch_size": 1,
        },
        "statistics": {name: stats(array) for name, array in arrays.items()},
        "tile_output_negative_count_including_overlaps": tile_negatives,
        "tile_output_negative_fraction_including_overlaps": tile_negatives / tile_samples,
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in (
            Path(__file__).resolve(), ROOT / "transform_main.py", ROOT / "numeric_domain.py",
            ROOT / "ablation_config.py", ROOT / "arch/trans_basenetworks.py",
        )},
        "runtime": {
            "python": platform.python_version(), "torch": torch.__version__,
            "timm": timm.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
            "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
            "seconds": time.perf_counter() - start,
        },
        "files": {path.name: {"sha256": sha256(path), "bytes": path.stat().st_size} for path in sorted(output.iterdir()) if path.is_file()},
        "command": [sys.executable, *sys.argv],
    }
    (output / "run.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "statistics": report["statistics"]["denoised"], "runtime": report["runtime"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
