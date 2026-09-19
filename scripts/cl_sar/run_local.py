"""Pinned official CL-SAR inference, with an explicit intensity/amplitude contract.

The published predict.py clips normalized amplitude predictions to [0, 1]
through tensor2img. We reproduce that behavior and also save raw predictions.
No file values are silently divided by 255 or independently display-stretched.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import math
import platform
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.io import loadmat, savemat

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_ROOT = PROJECT_ROOT / "external" / "CL-SAR"
OFFICIAL_URL = "https://github.com/YangtianFang2002/CL-SAR-Despeckling"
OFFICIAL_COMMIT = "b12129d1d3448750b9098b239397587eeb359857"
WEIGHT_RELATIVE = "experiments/MDN1-default/models/net_g_latest.pth"
WEIGHT_SHA256 = "94765b1e6dfb7584842dbad4b3683a566f77c5b3d85595dbb8913f90fef8c80b"
MODEL_CONFIG = dict(img_channel=1, width=32, sc_width=32, eap_pooling="max",
                    enc_blk_nums=[1, 1, 2, 2], middle_blk_num=2, dec_blk_nums=[2, 2, 1, 1])


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(OFFICIAL_ROOT), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def verify_source() -> dict:
    if git("rev-parse", "HEAD") != OFFICIAL_COMMIT:
        raise ValueError(f"Expected official CL-SAR commit {OFFICIAL_COMMIT}")
    if git("remote", "get-url", "origin").rstrip("/").removesuffix(".git") != OFFICIAL_URL:
        raise ValueError("Unexpected CL-SAR source repository")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("Official checkout has modified tracked files")
    files = ["basicsr/models/archs/MDN1_arch.py", "basicsr/models/archs/arch_util.py",
             "basicsr/data/data_util.py", "basicsr/utils/img_util.py", "basicsr/predict.py",
             "options/real/MDN1-default.yml"]
    return {"repository": OFFICIAL_URL, "commit": OFFICIAL_COMMIT,
            "tracked_sources_unchanged": True,
            "source_sha256": {name: sha256(OFFICIAL_ROOT / name) for name in files}}


def load_official_architecture():
    """Import unchanged architecture without BasicSR's unrelated training imports.

    Temporary namespace packages avoid auto-importing all models/losses. The only
    compatibility shim is an unused logging utility referenced by arch_util;
    LayerNorm2d and the entire MDN1 are executed from their original files.
    All namespace changes and the original file's sys.path edits are restored.
    """
    names = ["basicsr", "basicsr.models", "basicsr.models.archs", "basicsr.utils",
             "basicsr.models.archs.arch_util", "basicsr.models.archs.MDN1_arch"]
    previous = {name: sys.modules.get(name) for name in names}
    original_path = sys.path.copy()
    try:
        for name in names[:4]:
            module = types.ModuleType(name)
            module.__path__ = [str(OFFICIAL_ROOT.joinpath(*name.split(".")))]
            sys.modules[name] = module
        sys.modules["basicsr.utils"].get_root_logger = logging.getLogger
        for name in names[4:]:
            path = OFFICIAL_ROOT.joinpath(*name.split(".")).with_suffix(".py")
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        return module.MDN1
    finally:
        sys.path[:] = original_path
        for name, old in previous.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


def build_model(device: torch.device) -> tuple[torch.nn.Module, dict]:
    provenance = verify_source()
    checkpoint = OFFICIAL_ROOT / WEIGHT_RELATIVE
    digest = sha256(checkpoint)
    if digest != WEIGHT_SHA256:
        raise ValueError("Official CL-SAR weight checksum mismatch")
    model = load_official_architecture()(**MODEL_CONFIG)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)["params"]
    if not all(torch.isfinite(value).all() for value in state.values()):
        raise ValueError("Checkpoint contains nonfinite values")
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    provenance.update({"checkpoint": str(checkpoint), "checkpoint_sha256": digest,
                       "checkpoint_bytes": checkpoint.stat().st_size,
                       "checkpoint_parameter_key": "params", "official_pretrained": True,
                       "strict_state_dict_loaded": True, "model_config": MODEL_CONFIG,
                       "registered_parameters": sum(p.numel() for p in model.parameters()),
                       "eval_mode": True,
                       "import_compatibility": "temporary namespace packages; logging-only shim; original model unchanged"})
    return model, provenance


def validate_array(array: np.ndarray) -> np.ndarray:
    raw = np.asarray(array)
    if np.iscomplexobj(raw) or raw.ndim != 2 or raw.dtype.kind not in "buif":
        raise ValueError("Expected a real numeric 2-D grayscale array")
    if min(raw.shape) < 2:
        raise ValueError("Input must be at least 2x2")
    if not np.isfinite(raw).all() or (raw < 0).any():
        raise ValueError("Input must be finite and nonnegative")
    array = np.ascontiguousarray(raw, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError("Input overflows float32")
    return array


def load_array(path: Path, field: str = "noisy") -> tuple[np.ndarray, dict]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".mat":
        payload = loadmat(path, variable_names=[field])
        if field not in payload:
            raise ValueError(f"{path} has no MAT field {field!r}")
        raw = payload[field]
    elif suffix == ".npy":
        raw = np.load(path, allow_pickle=False)
    else:
        with Image.open(path) as image:
            if image.mode in ("RGB", "RGBA", "P", "CMYK", "LA"):
                raise ValueError("Use a single-channel grayscale raster; color is not silently converted")
            raw = np.asarray(image)
    array = validate_array(raw)
    return array, {"path": str(path.resolve()), "sha256": sha256(path),
                   "field": field if suffix == ".mat" else None,
                   "original_dtype": str(raw.dtype),
                   "conversion": "cast to float32; raw file units preserved; no automatic /255",
                   "min": float(array.min()), "max": float(array.max())}


def prepare_input(array: np.ndarray, input_domain: str, scale: float = 1.0) -> tuple[np.ndarray, dict]:
    array = validate_array(array)
    if input_domain not in ("intensity", "amplitude"):
        raise ValueError("input_domain must explicitly be intensity or amplitude")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        scaled = array / scale
    if not np.isfinite(scaled).all() or np.any((array > 0) & (scaled == 0)):
        raise ValueError("Scale causes overflow or underflow in float32")
    amplitude = np.sqrt(scaled) if input_domain == "intensity" else scaled.copy()
    positive = amplitude[amplitude != 0]
    if not positive.size:
        raise ValueError("Official normalization is undefined for an all-zero image")
    amp_min, amp_max = positive.min(), amplitude.max()
    if amp_max <= amp_min:
        raise ValueError("Official normalization needs at least two distinct positive amplitudes")
    # Exactly data_util.max_normalize for valid finite nonnegative amplitudes:
    # min ignores zero, and abs is applied AFTER subtracting this positive min.
    normalized = np.abs((amplitude - amp_min) / (amp_max - amp_min))
    if not np.isfinite(normalized).all():
        raise ValueError("Official normalization produced nonfinite values")
    return np.ascontiguousarray(normalized, dtype=np.float32), {
        "input_domain": input_domain, "output_domain": input_domain, "scale": scale,
        "sqrt_input": input_domain == "intensity", "amplitude_min_nonzero": float(amp_min),
        "amplitude_max": float(amp_max),
        "formula": "abs((amplitude - min_nonzero_amplitude) / (max_amplitude - min_nonzero_amplitude))",
        "input_clipped": False, "zero_pixel_count": int(np.count_nonzero(array == 0)),
        "normalized_input_min": float(normalized.min()), "normalized_input_max": float(normalized.max()),
        "zero_and_constant_policy": "reject undefined official normalization; no invented epsilon fallback"}


def restore_output(raw: np.ndarray, normalization: dict) -> np.ndarray:
    # predict.py explicitly clamps to [-20,20], then tensor2img defaults to [0,1].
    normalized = np.clip(np.clip(raw, -20.0, 20.0), 0.0, 1.0)
    amplitude = normalized * (normalization["amplitude_max"] - normalization["amplitude_min_nonzero"])
    amplitude = amplitude + normalization["amplitude_min_nonzero"]
    output = np.square(amplitude) if normalization["input_domain"] == "intensity" else amplitude
    output = np.ascontiguousarray(output * normalization["scale"], dtype=np.float32)
    if not np.isfinite(output).all():
        raise ValueError("Output overflow after inverse transform")
    return output


def configure_runtime() -> None:
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def predict(model: torch.nn.Module, normalized: np.ndarray, device: torch.device) -> tuple[np.ndarray, float]:
    tensor = torch.from_numpy(np.ascontiguousarray(normalized))[None, None].to(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        prediction = model(tensor)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    raw = prediction[0, 0].cpu().numpy().copy()
    if raw.shape != normalized.shape or not np.isfinite(raw).all():
        raise ValueError("Model returned invalid shape or nonfinite values")
    return raw, seconds


def save_previews(output: Path, noisy: np.ndarray, denoised: np.ndarray, scale: float) -> None:
    height, width = noisy.shape
    canvas = Image.new("RGB", (width * 2, height + 32), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (label, array) in enumerate((("Input", noisy), ("CL-SAR", denoised))):
        preview = Image.fromarray(np.rint(np.clip(array / scale, 0, 1) * 255).astype(np.uint8))
        canvas.paste(preview, (i * width, 32))
        draw.text((i * width + 8, 9), label, fill="black")
        if i:
            preview.save(output / "denoised.png")
    canvas.save(output / "comparison.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--input-domain", choices=("intensity", "amplitude"), required=True)
    parser.add_argument("--mat-field", default="noisy")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Known divisor in input-domain units; restored on output. Also fixes PNG range to [0,scale]")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "output" / "cl_sar_local" /
                        time.strftime("run_%Y%m%d_%H%M%S"))
    args = parser.parse_args()
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        raise ValueError(f"Output must be a new or empty directory: {args.output}")
    noisy, input_info = load_array(args.input, args.mat_field)
    normalized, normalization = prepare_input(noisy, args.input_domain, args.scale)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    configure_runtime()
    model, provenance = build_model(device)
    raw, seconds = predict(model, normalized, device)
    denoised = restore_output(raw, normalization)
    report = {"method": "CL-SAR", "scope": "official pretrained inference; no training or paper benchmark reproduction",
              "source": provenance, "input": input_info, "normalization": normalization,
              "shape": list(noisy.shape), "adapter_sha256": sha256(Path(__file__)),
              "postprocessing": {"official_normalized_amplitude_clamps": [[-20, 20], [0, 1]],
                  "fraction_clipped_at_20": float(np.mean((raw < -20) | (raw > 20))),
                  "fraction_clipped_at_0_1": float(np.mean((raw < 0) | (raw > 1))),
                  "network_raw_file": "network_raw.npy", "network_raw_domain": "normalized amplitude, BEFORE author clipping",
                  "square_output": args.input_domain == "intensity", "inverse_scale_on_output": True},
              "output": {"domain": args.input_domain, "min": float(denoised.min()), "max": float(denoised.max()),
                  "all_finite": True, "extra_adapter_clipping": False, "author_clipping_applied": True,
                  "display_range_input_units": [0, args.scale], "display_per_image_stretch": False,
                  "input_fraction_outside_display": float(np.mean(noisy > args.scale)),
                  "output_fraction_outside_display": float(np.mean(denoised > args.scale))},
              "forward_seconds_including_first_call": seconds,
              "runtime": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
                  "cuda": torch.version.cuda, "device": str(device),
                  "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                  "fp32": True, "tf32": False, "cudnn_deterministic": True},
              "metrics": None}
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "denoised.npy", denoised, allow_pickle=False)
    np.save(args.output / "network_raw.npy", raw, allow_pickle=False)
    savemat(args.output / "result.mat", {"noisy": noisy, "denoised": denoised,
            "residual": noisy - denoised, "input_domain": args.input_domain,
            "output_domain": args.input_domain, "network_output_normalized_amplitude_raw": raw}, do_compression=True)
    save_previews(args.output, noisy, denoised, args.scale)
    (args.output / "run.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "device": str(device),
                      "domain": args.input_domain, "forward_seconds": seconds,
                      "author_clipped_fraction": report["postprocessing"]["fraction_clipped_at_0_1"]}, indent=2))


if __name__ == "__main__":
    main()
