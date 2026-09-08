"""Run the pinned author's SDUDNet denoiser and distributed pretrained weights.

Raster uint8 inputs follow test.py's ToTensor (/255) convention. MAT/NPY
values are preserved; --scale explicitly maps other numeric scales to [0, 1].
Scientific arrays and metrics use unclipped predictions. PNGs are previews.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.io import loadmat, savemat

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_ROOT = PROJECT_ROOT / "external" / "SDUDNet"
OFFICIAL_URL = "https://github.com/BFY-official/SDUDNet"
OFFICIAL_COMMIT = "0c799911e84ef79c5fbe6f58f44cb8ee033048a1"
WEIGHT_HASHES = {
    "real": "3e1b9bc4eaaa01a81544e9b96c5e238415f7fd8af089881f3ae19e3469b7e69d",
    "synthetic": "3d9c28c80fad49510240b9ed7fd589049870f0e789edd5ac5382147903a9486c",
}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(OFFICIAL_ROOT), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def verify_source() -> dict:
    if git("rev-parse", "HEAD") != OFFICIAL_COMMIT:
        raise ValueError(f"Expected official SDUDNet commit {OFFICIAL_COMMIT}")
    if git("remote", "get-url", "origin").rstrip("/").removesuffix(".git") != OFFICIAL_URL:
        raise ValueError("Unexpected SDUDNet source repository")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("Official checkout has modified tracked files")
    return {"repository": OFFICIAL_URL, "commit": OFFICIAL_COMMIT,
            "tracked_sources_unchanged": True,
            "dnn_sha256": sha256(OFFICIAL_ROOT / "net" / "DNN.py")}


def build_model(variant: str, device: torch.device) -> tuple[torch.nn.Module, dict]:
    provenance = verify_source()
    checkpoint = OFFICIAL_ROOT / "models" / f"{variant}.pth"
    digest = sha256(checkpoint)
    if digest != WEIGHT_HASHES[variant]:
        raise ValueError(f"Official {variant} weight checksum mismatch")
    spec = importlib.util.spec_from_file_location("_official_sdudnet_dnn", OFFICIAL_ROOT / "net" / "DNN.py")
    if spec is None or spec.loader is None:
        raise ImportError("Could not import the official DNN")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.DNN()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    if not all(torch.isfinite(value).all() for value in state.values()):
        raise ValueError("Checkpoint contains nonfinite values")
    model.to(device).eval()
    provenance.update({"variant": variant, "checkpoint": str(checkpoint),
                       "checkpoint_sha256": digest, "official_pretrained": True,
                       "strict_state_dict_loaded": True,
                       "registered_parameters": sum(p.numel() for p in model.parameters()),
                       "forward_output": "first output: out (second is input - out)",
                       "eval_mode": True})
    return model, provenance


def load_array(path: Path, field: str) -> tuple[np.ndarray, dict]:
    suffix = path.suffix.lower()
    if suffix == ".mat":
        payload = loadmat(path, variable_names=[field])
        if field not in payload:
            raise ValueError(f"{path} has no MAT field {field!r}")
        raw = payload[field]
        conversion = "MAT values preserved"
    elif suffix == ".npy":
        raw = np.load(path, allow_pickle=False)
        conversion = "NPY values preserved"
    else:
        with Image.open(path) as raster:
            if raster.mode in ("RGB", "RGBA", "P", "CMYK"):
                raise ValueError("Use a single-channel grayscale image; RGB is not silently converted")
            raw = np.asarray(raster)
        conversion = "raster values preserved"
        if raw.dtype == np.uint8:
            raw = raw.astype(np.float32) / 255.0
            conversion = "uint8 / 255, matching official ToTensor"
        # Other raster types need an explicit --scale, as do unnormalized MATs.
    if np.iscomplexobj(raw) or np.asarray(raw).ndim != 2:
        raise ValueError(f"Expected a real 2-D grayscale array, got {np.shape(raw)}")
    array = np.ascontiguousarray(raw, dtype=np.float32)
    if min(array.shape) < 2 or not np.isfinite(array).all() or (array < 0).any():
        raise ValueError("Input must be at least 2x2, finite, and nonnegative")
    return array, {"path": str(path.resolve()), "sha256": sha256(path),
                   "field": field if suffix == ".mat" else None,
                   "conversion": conversion, "min": float(array.min()), "max": float(array.max())}


def predict(model: torch.nn.Module, array: np.ndarray, device: torch.device) -> tuple[np.ndarray, float]:
    tensor = torch.from_numpy(np.ascontiguousarray(array))[None, None].to(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        output, _ = model(tensor)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - start
    result = output[0, 0].cpu().numpy().copy()
    if result.shape != array.shape or not np.isfinite(result).all():
        raise ValueError("Model returned invalid shape or nonfinite predictions")
    return result, seconds


def psnr(reference: np.ndarray, estimate: np.ndarray, data_range: float) -> float | str:
    mse = float(np.mean((reference.astype(np.float64) - estimate.astype(np.float64)) ** 2))
    return 10.0 * math.log10(data_range ** 2 / mse) if mse else "Infinity"


def save_comparison(output: Path, noisy: np.ndarray, prediction: np.ndarray,
                    clean: np.ndarray | None, variant: str) -> None:
    panels = [("Input", noisy), (f"SDUDNet ({variant})", prediction)]
    if clean is not None:
        panels.append(("Reference", clean))
    height, width = noisy.shape
    canvas = Image.new("RGB", (len(panels) * width, height + 32), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (label, array) in enumerate(panels):
        # Same fixed [0, 1] display range for every panel; never per-image stretching.
        preview = Image.fromarray(np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8))
        canvas.paste(preview, (i * width, 32))
        draw.text((i * width + 8, 9), label, fill="black")
    canvas.save(output / "comparison.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=OFFICIAL_ROOT / "my_datasets" / "01233.jpg")
    parser.add_argument("--model", choices=WEIGHT_HASHES, default="real")
    parser.add_argument("--mat-field", default="noisy")
    parser.add_argument("--clean", type=Path)
    parser.add_argument("--clean-mat-field", default="clean")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Divide input AND reference by this known scale before inference")
    parser.add_argument("--data-range", type=float, default=1.0,
                        help="PSNR peak range in the original (before --scale) units")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "output" / "sdudnet_local" /
                        time.strftime("run_%Y%m%d_%H%M%S"))
    args = parser.parse_args()
    for name in ("scale", "data_range"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError(f"Output directory is not empty; choose a new path: {args.output}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    noisy, input_info = load_array(args.input, args.mat_field)
    clean, clean_info = load_array(args.clean, args.clean_mat_field) if args.clean else (None, None)
    if clean is not None and clean.shape != noisy.shape:
        raise ValueError("Input and reference shapes differ")
    network_input = np.ascontiguousarray(noisy / args.scale, dtype=np.float32)
    if network_input.max() > 1.000001:
        raise ValueError("Network input exceeds [0, 1]; specify its known --scale. No automatic clipping is applied")
    if clean is not None and clean.max() / args.scale > 1.000001:
        raise ValueError("Reference exceeds the same [0, 1] scale as the network input")
    model, provenance = build_model(args.model, device)
    prediction_normalized, seconds = predict(model, network_input, device)
    prediction = prediction_normalized * args.scale
    if not np.isfinite(prediction).all():
        raise ValueError("Output overflow after inverse scaling")
    metrics = None if clean is None else {
        "psnr_input_db": psnr(clean, noisy, args.data_range),
        "psnr_output_db": psnr(clean, prediction, args.data_range),
        "data_range": args.data_range, "clipped_for_metrics": False,
        "scope": "local supplied sample, not the paper benchmark",
    }
    report = {"method": "SDUDNet", "scope": "official pretrained inference",
              "source": provenance, "input": input_info, "reference": clean_info,
              "adapter_sha256": sha256(Path(__file__)), "shape": list(noisy.shape),
              "normalization": {"divide_by": args.scale, "inverse_scale_on_output": True,
                                "sqrt_or_log_transform": False, "input_clipped": False},
              "output": {"min": float(prediction.min()), "max": float(prediction.max()),
                         "all_finite": True, "saved_arrays_clipped": False,
                         "display_range_original_units": [0, args.scale],
                         "fraction_outside_display_range": float(np.mean((prediction_normalized < 0) | (prediction_normalized > 1)))},
              "metrics": metrics, "forward_seconds_including_first_call": seconds,
              "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                          "numpy": np.__version__, "cuda": torch.version.cuda,
                          "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                          "cudnn_deterministic": True, "tf32": False}}
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "denoised.npy", prediction, allow_pickle=False)
    arrays = {"noisy": noisy, "denoised": prediction, "residual": noisy - prediction}
    if clean is not None:
        arrays["clean"] = clean
    savemat(args.output / "result.mat", arrays, do_compression=True)
    Image.fromarray(np.rint(np.clip(prediction_normalized, 0, 1) * 255).astype(np.uint8)).save(args.output / "denoised.png")
    save_comparison(args.output, network_input, prediction_normalized,
                    clean / args.scale if clean is not None else None, args.model)
    (args.output / "run.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "device": str(device),
                      "forward_seconds": seconds, "metrics": metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
