"""Local SAR-CAM inference using the untouched, pinned authors' model.

Requires a trained checkpoint. Random initialization is never an inference
fallback. Run train_local.py for an explicitly labelled short training demo.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import platform
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.io import loadmat, savemat

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_ROOT = PROJECT_ROOT / "external" / "SAR-CAM"
OFFICIAL_URL = "https://github.com/JK-the-Ko/SAR-CAM.git"
OFFICIAL_COMMIT = "ea5ee3bed00ab22735a7c87518fe5388c2d6c49a"
CONSTRUCTOR = dict(scale=2, in_channels=1, channels=128, kernel_size=3,
                   stride=1, dilation=1, bias=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def verify_source(root: Path) -> dict:
    root = root.resolve()
    if git(root, "rev-parse", "HEAD") != OFFICIAL_COMMIT:
        raise ValueError(f"Expected official SAR-CAM commit {OFFICIAL_COMMIT}")
    origin = git(root, "remote", "get-url", "origin")
    if origin.rstrip("/").removesuffix(".git").lower() != OFFICIAL_URL.removesuffix(".git").lower():
        raise ValueError(f"Unexpected source repository: {origin}")
    if git(root, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("Official SAR-CAM checkout has modified tracked files")
    return {"repository": OFFICIAL_URL, "commit": OFFICIAL_COMMIT,
            "root": str(root), "tracked_sources_unchanged": True,
            "source_sha256": {name: sha256(root / name) for name in
                              ("model.py", "model_parts.py", "loss.py")}}


def import_official(root: Path, filename: str):
    """Keep the author's top-level model_parts import isolated from this repo."""
    root = root.resolve()
    previous_parts = sys.modules.pop("model_parts", None)
    old_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root))
    try:
        module_name = "_local_official_sar_cam_" + Path(filename).stem
        spec = importlib.util.spec_from_file_location(module_name, root / filename)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import {root / filename}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(root))
        sys.modules.pop("model_parts", None)
        if previous_parts is not None:
            sys.modules["model_parts"] = previous_parts
        sys.dont_write_bytecode = old_write_bytecode


def build_model(root: Path) -> torch.nn.Module:
    verify_source(root)
    return import_official(root, "model.py").Model(**CONSTRUCTOR)


def model_metadata() -> dict:
    return {"method": "sar_cam", "class_name": "SAR_CAM", "variant": None,
            "external_repository": OFFICIAL_URL, "external_commit": OFFICIAL_COMMIT,
            "constructor_kwargs": dict(CONSTRUCTOR)}


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path,
                    metadata_path: Path | None = None) -> dict:
    # The local checkpoints contain tensors and primitive metadata only.
    # Never silently fall back to unrestricted pickle execution.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Checkpoint must contain a state dict and model metadata")
    sidecar = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path else None
    if "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
        metadata = checkpoint.get("model_metadata")
        config = checkpoint.get("run_config", {})
        provenance = checkpoint.get("provenance")
        if sidecar is not None:
            raise ValueError("A structured checkpoint already has metadata; do not override it with a sidecar")
    else:
        if not sidecar:
            raise ValueError("Raw weights need --checkpoint-metadata with model_metadata, numeric_domain, "
                             "provenance, and checkpoint_sha256; no identity will be guessed")
        if sidecar.get("checkpoint_sha256") != sha256(checkpoint_path):
            raise ValueError("Checkpoint SHA-256 does not match its metadata sidecar")
        state = checkpoint
        metadata = sidecar.get("model_metadata")
        config = sidecar
        provenance = sidecar.get("provenance")
    if not isinstance(metadata, Mapping):
        raise ValueError("Checkpoint is missing model_metadata")
    for key in ("method", "external_commit", "constructor_kwargs"):
        if metadata.get(key) != model_metadata()[key]:
            raise ValueError(f"Checkpoint model metadata mismatch: {key}")
    if config.get("numeric_domain") != "intensity_v1":
        raise ValueError("Checkpoint must explicitly declare numeric_domain='intensity_v1'")
    if provenance is None and config.get("protocol_id"):
        provenance = {"kind": "project_retrained", "protocol_id": config["protocol_id"],
                      "official_pretrained": False}
    if not isinstance(provenance, Mapping) or not provenance.get("kind"):
        raise ValueError("Checkpoint must record provenance.kind; random weights are not a denoiser")
    if not isinstance(state, Mapping) or not state or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError("Invalid tensor state_dict")
    if all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
        prefix_removed = True
    else:
        prefix_removed = False
    for name, value in state.items():
        if not torch.isfinite(value).all():
            raise ValueError(f"Checkpoint contains nonfinite weights: {name}")
    model.load_state_dict(state, strict=True)
    return {"path": str(checkpoint_path.resolve()), "sha256": sha256(checkpoint_path),
            "model_metadata": dict(metadata), "provenance": dict(provenance),
            "strict_state_dict_loaded": True, "data_parallel_prefix_removed": prefix_removed,
            "global_step": checkpoint.get("global_step"),
            "numeric_domain": "intensity_v1"}


def load_intensity(path: Path, mat_field: str = "noisy") -> tuple[np.ndarray, dict]:
    if path.suffix.lower() == ".mat":
        payload = loadmat(path, variable_names=[mat_field])
        if mat_field not in payload:
            raise ValueError(f"MAT file {path} has no exact field {mat_field!r}")
        raw = payload[mat_field]
        conversion = "MAT_values_preserved"
    elif path.suffix.lower() == ".npy":
        raw = np.load(path, allow_pickle=False)
        conversion = "NPY_values_preserved"
    else:
        with Image.open(path) as image:
            if image.mode in ("RGB", "RGBA", "P", "CMYK"):
                image = image.convert("L")
            raw = np.asarray(image)
        conversion = "raster_values_preserved"
        if np.issubdtype(raw.dtype, np.integer):
            raw = raw.astype(np.float32) / np.iinfo(raw.dtype).max
            conversion = "integer_raster_divided_by_dtype_max"
    if np.iscomplexobj(raw) or np.asarray(raw).ndim != 2:
        raise ValueError(f"Expected a real 2-D intensity array, got {np.shape(raw)}")
    array = np.ascontiguousarray(raw, dtype=np.float32)
    if min(array.shape) < 2 or not np.isfinite(array).all() or (array < 0).any():
        raise ValueError("Intensity input must be at least 2x2, finite, and nonnegative")
    return array, {"path": str(path.resolve()), "sha256": sha256(path),
                   "mat_field": mat_field if path.suffix.lower() == ".mat" else None,
                   "conversion": conversion, "min": float(array.min()), "max": float(array.max())}


def normalize_intensity(array: np.ndarray, mode: str) -> tuple[np.ndarray, dict]:
    if mode == "none":
        if float(array.max()) > 1.000001:
            raise ValueError("SAR-CAM expects normalized intensity [0,1]. Use --normalization clip01 "
                             "for the project's synthetic protocol or --normalization percentile for real data.")
        return array, {"mode": "none", "clipped": False}
    if mode == "clip01":
        return np.clip(array, 0, 1), {"mode": mode, "clipped": bool((array > 1).any()), "range": [0, 1]}
    low, high = np.percentile(array, [1, 99])
    if high <= low:
        raise ValueError("Cannot percentile-normalize constant or degenerate intensity")
    return np.clip((array - low) / (high - low), 0, 1).astype(np.float32), {
        "mode": mode, "clipped": True, "percentiles": [1, 99],
        "low": float(low), "high": float(high), "output_domain": "normalized_intensity"}


def select_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable in this Python environment")
    return device


def predict(model: torch.nn.Module, intensity: np.ndarray, device: torch.device) -> tuple[np.ndarray, dict]:
    height, width = intensity.shape
    padding = (height % 2, width % 2)
    array = np.pad(intensity, ((0, padding[0]), (0, padding[1])), mode="reflect")
    tensor = torch.from_numpy(np.ascontiguousarray(array))[None, None].to(device)
    model.eval()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        output = model(tensor)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    if output.shape != tensor.shape or not torch.isfinite(output).all():
        raise ValueError("SAR-CAM returned invalid dimensions or nonfinite prediction")
    prediction = output[0, 0, :height, :width].cpu().numpy().copy()
    return prediction, {"seconds": elapsed, "padding_bottom_right": list(padding),
                        "padding_mode": "reflect", "full_frame_inference": True}


def psnr(reference: np.ndarray, prediction: np.ndarray) -> float | None:
    mse = float(np.mean((reference.astype(np.float64) - prediction) ** 2))
    return 10 * math.log10(1 / mse) if mse > 0 else None


def preview(array: np.ndarray) -> Image.Image:
    return Image.fromarray(np.round(255 * np.clip(array, 0, 1)).astype(np.uint8))


def save_result(output: Path, intensity: np.ndarray, prediction: np.ndarray,
                summary: dict, clean: np.ndarray | None = None) -> None:
    # The directory is an output boundary: never replace a previous run.
    output.mkdir(parents=True, exist_ok=False)
    arrays = {"noisy_intensity": intensity, "prediction": prediction,
              "denoised_intensity": prediction}
    if clean is not None:
        arrays["clean_intensity"] = clean
    np.save(output / "prediction.npy", prediction, allow_pickle=False)
    savemat(output / "result.mat", arrays, do_compression=True)
    preview(intensity).save(output / "input.png")
    preview(prediction).save(output / "denoised.png")
    panels = [intensity, prediction] if clean is None else [clean, intensity, prediction]
    separator = np.ones((intensity.shape[0], 6), dtype=np.float32)
    joined = panels[0]
    for panel in panels[1:]:
        joined = np.concatenate((joined, separator, panel), axis=1)
    preview(joined).save(output / "comparison.png")
    summary.update({"scientific_prediction_clipped": False, "display_range": [0, 1],
                    "prediction_min": float(prediction.min()), "prediction_max": float(prediction.max()),
                    "comparison_panel_order": (["noisy", "prediction"] if clean is None else
                                               ["clean", "noisy", "prediction"])})
    if clean is not None:
        summary["metrics"] = {"domain": "normalized_intensity", "data_range": 1,
                              "prediction_clipped": False, "noisy_psnr_db": psnr(clean, intensity),
                              "denoised_psnr_db": psnr(clean, prediction)}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-metadata", type=Path)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--mat-field", default="noisy")
    parser.add_argument("--clean", type=Path)
    parser.add_argument("--clean-mat-field", default="clean")
    parser.add_argument("--normalization", choices=("none", "clip01", "percentile"), default="none")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=OFFICIAL_ROOT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    if args.threads < 1:
        raise ValueError("threads must be positive")
    torch.set_num_threads(args.threads)
    device = select_device(args.device)
    model = build_model(args.source_root)
    checkpoint = load_checkpoint(model, args.checkpoint, args.checkpoint_metadata)
    model.to(device)
    intensity, input_info = load_intensity(args.input, args.mat_field)
    intensity, normalization = normalize_intensity(intensity, args.normalization)
    clean = None
    if args.clean:
        if args.normalization == "percentile":
            raise ValueError("Reference metrics require aligned normalized data, not per-image percentile normalization")
        clean, _ = load_intensity(args.clean, args.clean_mat_field)
        clean, _ = normalize_intensity(clean, args.normalization)
        if clean.shape != intensity.shape:
            raise ValueError("Clean reference shape differs from input")
    prediction, timing = predict(model, intensity, device)
    summary = {"schema_version": 1, "algorithm": "SAR-CAM", "source": verify_source(args.source_root),
               "checkpoint": checkpoint, "input": input_info, "normalization": normalization,
               "numeric_domain": "intensity_v1", "image_shape": list(intensity.shape),
               "inference": timing, "python_version": platform.python_version(),
               "torch_version": str(torch.__version__), "device": str(device),
               "cuda_version": torch.version.cuda,
               "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
               "parameter_count": sum(p.numel() for p in model.parameters())}
    save_result(args.output, intensity, prediction, summary, clean)
    print(json.dumps({"output": str(args.output.resolve()), "provenance": checkpoint["provenance"],
                      "inference_seconds": timing["seconds"], "metrics": summary.get("metrics")}, indent=2))


if __name__ == "__main__":
    main()
