"""Pinned authors' MuLoG with correlated-speckle DRUNet, single-channel inference.

Explicit physical domain and effective looks are mandatory. Numeric input values
are preserved, including uint8 rasters; --scale controls known scaling and PNGs.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import json
import math
import platform
import sys
import time
import types
import zipfile
import zlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.io import loadmat, savemat
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_ROOT = PROJECT_ROOT / "external" / "MuLoG-DRUNet"
OFFICIAL_URL = "https://gitlab.telecom-paris.fr/ring/mulog-drunet"
OFFICIAL_COMMIT = "f468573f5bd4d7d30065380b8580269bcefbec19"
SOURCE_SHA256 = "74204053abd6c28b3b56bbf1205e5a60f8463f081499e9b5d70679d334534594"
MODEL_BLOB_SHA1 = "eff7e4c921bdd05caf875b126e7a8432c6c4d3e8"
MODEL_URL = f"https://gitlab.telecom-paris.fr/api/v4/projects/6784/repository/blobs/{MODEL_BLOB_SHA1}/raw"
MODEL_SIZE = 130581503
MODEL_CRC32 = 1667635814
# SHA-256 of the extracted first member after verification of the fixed
# official HTTPS blob endpoint, ZIP local header, DEFLATE stream, size and CRC32.
GENERIC_MODEL_SHA256 = "20bc285c8710214003f72bdf5e48004c1accb63547d4a304c89714c385dff124"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_checkpoint(path: Path, record: dict) -> tuple[str, str]:
    """Verify only generic weights, with explicit bootstrap versus pinned trust."""
    if path.stat().st_size != MODEL_SIZE:
        raise ValueError("Generic checkpoint size mismatch")
    if (record.get("download_url") != MODEL_URL or record.get("member") != "models/generic_model.pth"
            or record.get("archive_git_blob_identifier_for_url") != MODEL_BLOB_SHA1
            or record.get("member_size_bytes") != MODEL_SIZE or record.get("member_crc32") != MODEL_CRC32
            or record.get("complete_archive_git_blob_verified") is not False):
        raise ValueError("Unexpected generic checkpoint provenance record")
    checksum, crc = hashlib.sha256(), 0
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            checksum.update(block)
            crc = zlib.crc32(block, crc)
    actual_sha = checksum.hexdigest()
    if crc != MODEL_CRC32 or actual_sha != record.get("checkpoint_sha256"):
        raise ValueError("Generic checkpoint CRC32 or recorded SHA256 mismatch")
    if GENERIC_MODEL_SHA256 is not None:
        if actual_sha != GENERIC_MODEL_SHA256:
            raise ValueError("Generic checkpoint differs from fixed SHA256 pin")
        level = "fixed_checkpoint_sha256"
    else:
        # The recorded hash detects local changes, not a coordinated replacement
        # of both manifest and checkpoint. CRC is not a cryptographic signature.
        level = "bootstrap_official_https_fixed_zip_metadata_crc32_and_local_manifest_sha256"
    return actual_sha, level


def positive(value: float, name: str) -> float:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def validate_array(array: np.ndarray) -> np.ndarray:
    raw = np.asarray(array)
    if np.iscomplexobj(raw) or raw.ndim != 2 or not np.issubdtype(raw.dtype, np.number):
        raise ValueError("Expected a real numeric two-dimensional single-channel array")
    array = np.ascontiguousarray(raw, dtype=np.float64)
    if min(array.shape) < 8 or not np.isfinite(array).all() or (array < 0).any():
        raise ValueError("Input must be at least 8x8, finite, and nonnegative")
    if not np.any(array > 0):
        raise ValueError("All-zero inputs are outside the author's log-domain implementation")
    return array


def load_array(path: Path, field: str = "noisy") -> tuple[np.ndarray, dict]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".mat":
        payload = loadmat(path, variable_names=[field])
        if field not in payload:
            raise ValueError(f"MAT field {field!r} not found in {path}")
        raw = payload[field]
    elif suffix == ".npy":
        raw = np.load(path, allow_pickle=False)
    else:
        with Image.open(path) as raster:
            if raster.mode in ("RGB", "RGBA", "P", "CMYK", "LA"):
                raise ValueError("Expected single-channel grayscale; no RGB conversion is performed")
            raw = np.asarray(raster)
    array = validate_array(raw)
    return array, {"path": str(path.resolve()), "sha256": sha256(path),
                   "mat_field": field if suffix == ".mat" else None,
                   "original_dtype": str(raw.dtype), "conversion": "numeric values preserved; no automatic /255",
                   "min": float(array.min()), "max": float(array.max())}


def to_intensity(array: np.ndarray, input_domain: str, scale: float) -> np.ndarray:
    if input_domain not in ("intensity", "amplitude"):
        raise ValueError("input_domain must be intensity or amplitude")
    positive(scale, "scale")
    array = validate_array(array)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        result = array / scale
        if input_domain == "amplitude":
            result = result ** 2
    if not np.isfinite(result).all() or np.any((array > 0) & (result == 0)):
        raise ValueError("Input scaling/domain transform overflows or underflows")
    return np.ascontiguousarray(result)


def from_intensity(array: np.ndarray, input_domain: str, scale: float) -> np.ndarray:
    if input_domain not in ("intensity", "amplitude"):
        raise ValueError("input_domain must be intensity or amplitude")
    positive(scale, "scale")
    array = validate_array(array)
    with np.errstate(over="ignore", invalid="ignore"):
        result = (np.sqrt(array) if input_domain == "amplitude" else array) * scale
    if not np.isfinite(result).all():
        raise ValueError("Output overflows after restoring scale/domain")
    return np.ascontiguousarray(result)


def verify_source() -> dict:
    required = ("py_functions.zip", "source_manifest.json", "models/generic_model.pth")
    missing = [name for name in required if not (OFFICIAL_ROOT / name).is_file()]
    if missing:
        raise RuntimeError("Official MuLoG assets incomplete; finish scripts/mulog_drunet/setup_windows.ps1 first. "
                           f"Missing: {', '.join(missing)}")
    source_zip = OFFICIAL_ROOT / "py_functions.zip"
    if sha256(source_zip) != SOURCE_SHA256:
        raise ValueError("Official source archive checksum mismatch")
    manifest = json.loads((OFFICIAL_ROOT / "source_manifest.json").read_text(encoding="utf-8"))
    if manifest["commit"] != OFFICIAL_COMMIT or manifest["repository"] != OFFICIAL_URL:
        raise ValueError("Unexpected source manifest")
    with zipfile.ZipFile(source_zip) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            path = OFFICIAL_ROOT / member.filename
            if path.read_bytes() != archive.read(member):
                raise ValueError(f"Modified official source: {path}")
    checkpoint = OFFICIAL_ROOT / "models/generic_model.pth"
    checkpoint_sha, verification_level = verify_checkpoint(checkpoint, manifest["generic_model"])
    return {"repository": OFFICIAL_URL, "commit": OFFICIAL_COMMIT,
            "source_archive_sha256": SOURCE_SHA256,
            "model_download_url": MODEL_URL,
            "model_archive_git_blob_identifier_for_url": MODEL_BLOB_SHA1,
            "complete_model_archive_verified": False,
            "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_sha,
            "checkpoint_verification_level": verification_level,
            "checkpoint_independent_sha256_pin_available": GENERIC_MODEL_SHA256 is not None,
            "official_sources_unchanged": True}


class NumpyCompat:
    """Only the official module sees the removed NumPy complex alias."""
    complex = complex

    def __getattr__(self, name):
        return getattr(np, name)


def unavailable(*args, **kwargs):
    raise RuntimeError("This adapter supports DRUNet with scalar D=1 only; optional branch was unexpectedly called")


def import_official():
    if "py_functions" in sys.modules:
        found = Path(sys.modules["py_functions"].__file__).resolve()
        if not found.is_relative_to(OFFICIAL_ROOT.resolve()):
            raise ImportError("Another py_functions package is already loaded")
    temporary = []
    for name, function in (("sparsesvd", "sparsesvd"), ("bm3d", "bm3d")):
        if name not in sys.modules and importlib.util.find_spec(name) is None:
            stub = types.ModuleType(name)
            setattr(stub, function, unavailable)
            sys.modules[name] = stub
            temporary.append(name)
    sys.path.insert(0, str(OFFICIAL_ROOT))
    try:
        core = importlib.import_module("py_functions.mulog")
        arch = importlib.import_module("py_functions.DRUNet_architecture")
        gaussian = importlib.import_module("py_functions.gaussian_denoisers")
    finally:
        sys.path.remove(str(OFFICIAL_ROOT))
        for name in temporary:
            sys.modules.pop(name, None)
    return core, arch, gaussian


class Denoiser:
    """Author's quantile/float32 forward and inverse, with cached explicit-device model."""
    def __init__(self, model, tools, device):
        self.model, self.tools, self.device = model, tools, device
        self.calls = []

    def __call__(self, y, sig):
        y_prime = np.atleast_3d(y.copy())
        qm = self.tools.quantile(y_prime.flatten(), 0.003)
        qM = self.tools.quantile(y_prime.flatten(), 0.997)
        if not np.isfinite(qm + qM) or qM <= qm:
            raise ValueError("Degenerate official 0.003/0.997 log-channel quantile range")
        normalized = (y_prime - qm) / (qM - qm)
        sigma = torch.tensor([sig / (qM - qm)], dtype=torch.float32, device=self.device)
        output = np.zeros_like(y_prime)
        height, width, channels = y_prime.shape
        bottom, right = (-height) % 8, (-width) % 8
        with torch.inference_mode():
            for channel in range(channels):
                tensor = torch.tensor(normalized[:, :, channel][None, None], dtype=torch.float32, device=self.device)
                if bottom or right:
                    tensor = F.pad(tensor, (0, right, 0, bottom), mode="replicate")
                output[:, :, channel] = self.model(tensor, sigma)[0, 0, :height, :width].cpu().numpy()
        output = output * (qM - qm) + qm
        if not np.isfinite(output).all():
            raise ValueError("Nonfinite DRUNet prediction (not silently zero-filled)")
        self.calls.append({"sig": float(sig), "quantile_003": float(qm), "quantile_997": float(qM),
                           "padding_bottom_right": [bottom, right]})
        return output[:, :, 0] if np.ndim(y) == 2 else output


def build_denoiser(device):
    provenance = verify_source()
    if not provenance["checkpoint_independent_sha256_pin_available"]:
        print("MuLoG checkpoint: official HTTPS/ZIP CRC32 bootstrap and local SHA256 integrity; "
              "independent SHA256 pin pending, complete model archive not verified.", file=sys.stderr)
    core, arch, gaussian = import_official()
    model = arch.DRUNet()
    state = torch.load(provenance["checkpoint"], map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    if not all(torch.isfinite(v).all() for v in state.values()):
        raise ValueError("Checkpoint contains nonfinite weights")
    model.to(device).eval()
    provenance.update({"strict_state_dict_loaded": True, "registered_parameters": sum(p.numel() for p in model.parameters())})
    return Denoiser(model, core.to, device), core, gaussian, provenance


@contextlib.contextmanager
def compatibility(core):
    original_np, original_os = core.mf.np, core.ad.os
    core.mf.np = NumpyCompat()
    # Only scheduling changes: keep each row's scalar Newton iterations unchanged.
    core.ad.os = types.SimpleNamespace(cpu_count=lambda: 4)
    try:
        yield
    finally:
        core.mf.np, core.ad.os = original_np, original_os


def predict(intensity, looks, iterations, denoiser, core):
    positive(looks, "looks")
    if iterations < 1:
        raise ValueError("iterations must be at least 1")
    if denoiser.device.type == "cuda":
        torch.cuda.synchronize(denoiser.device)
    start = time.perf_counter()
    with compatibility(core):
        result = core.mulog(intensity, looks, denoiser, "T", iterations)
    if denoiser.device.type == "cuda":
        torch.cuda.synchronize(denoiser.device)
    result = np.asarray(result)
    imaginary_max = float(np.max(np.abs(np.imag(result))))
    if imaginary_max > 1e-10 * max(1.0, float(np.max(np.abs(result)))):
        raise ValueError("Scalar MuLoG unexpectedly returned a nonreal estimate")
    result = validate_array(np.real(result))
    if result.shape != intensity.shape:
        raise ValueError("Output shape mismatch")
    return result, time.perf_counter() - start, imaginary_max


def configure_runtime():
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--input-domain", choices=("intensity", "amplitude"), required=True)
    parser.add_argument("--looks", type=float, required=True)
    parser.add_argument("--mat-field", default="noisy")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--iterations", type=int, default=10, help="ADMM T; official example uses 10, function default is 6")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "output" / "mulog_drunet_local" / time.strftime("run_%Y%m%d_%H%M%S"))
    args = parser.parse_args()
    positive(args.looks, "--looks")
    positive(args.scale, "--scale")
    if args.iterations < 1:
        parser.error("--iterations must be at least 1")
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        raise ValueError(f"Output must be a new or empty directory: {args.output}")
    noisy, input_info = load_array(args.input, args.mat_field)
    intensity = to_intensity(noisy, args.input_domain, args.scale)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    configure_runtime()
    denoiser, core, _, provenance = build_denoiser(device)
    result, elapsed, imaginary_max = predict(intensity, args.looks, args.iterations, denoiser, core)
    prediction = from_intensity(result, args.input_domain, args.scale)
    report = {"method": "MuLoG-DRUNet", "scope": "official generic checkpoint; scalar intensity inference",
              "source": provenance, "input": input_info, "shape": list(noisy.shape),
              "input_domain": args.input_domain, "output_domain": args.input_domain,
              "parameters": {"looks": args.looks, "looks_source": "explicit user argument; never estimated",
                             "admm_iterations": args.iterations, "newton_iterations": 10,
                             "initial_beta": 1 + 2 / args.looks, "lambda": 1,
                             "scalar_log_channel_sigma_rule": "sqrt(polygamma(1, max(1, L))), unchanged author implementation"},
              "normalization": {"divide_by_in_input_domain": args.scale,
                                "amplitude_squared_before_mulog": args.input_domain == "amplitude",
                                "sqrt_after_mulog": args.input_domain == "amplitude",
                                "output_multiplied_by_scale": True,
                                "zero_count": int(np.sum(intensity == 0)),
                                "official_zero_handling": "zeros replaced by minimum strictly positive intensity",
                                "official_stabilization": "relative perturbation 1e-6, author seed 4242",
                                "denoiser_internal_quantile_transforms": denoiser.calls,
                                "input_clipped": False},
              "compatibility": {"numpy_complex_alias": "module-local proxy, restored after inference",
                                "unused_dependencies": "fail-fast import stubs for absent sparsesvd/BM3D; never invoked at D=1",
                                "model_cached": True, "cpu_tensor_conversion_fixed": True,
                                "newton_worker_count": 4,
                                "denoiser_padding": "bottom/right replicate to multiple of 8, cropped each call"},
              "output": {"dtype": str(prediction.dtype), "min": float(prediction.min()), "max": float(prediction.max()),
                         "saved_arrays_clipped": False, "all_finite": True, "imaginary_residue_max": imaginary_max,
                         "display_range_original_units": [0, args.scale]},
              "runtime": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
                          "device": str(device), "cuda": torch.version.cuda,
                          "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
              "forward_seconds_including_first_call": elapsed, "adapter_sha256": sha256(Path(__file__))}
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "denoised.npy", prediction, allow_pickle=False)
    savemat(args.output / "result.mat", {"noisy": noisy, "denoised": prediction, "residual": noisy - prediction}, do_compression=True)
    previews = [Image.fromarray(np.rint(np.clip(a / args.scale, 0, 1) * 255).astype(np.uint8)) for a in (noisy, prediction)]
    previews[1].save(args.output / "denoised.png")
    height, width = noisy.shape
    canvas = Image.new("RGB", (2 * width, height + 28), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (panel, title) in enumerate(zip(previews, ("Input", "MuLoG-DRUNet"))):
        canvas.paste(panel, (index * width, 28))
        draw.text((index * width + 3, 5), title, fill="black")
    canvas.save(args.output / "comparison.png")
    (args.output / "run.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "device": str(device), "seconds": elapsed}, indent=2))


if __name__ == "__main__":
    main()
