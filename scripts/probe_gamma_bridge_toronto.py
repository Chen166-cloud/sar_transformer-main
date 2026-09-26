"""Isolated official GammaBridge inference check; does not modify benchmark results.

Fixed official real-SAR defaults: five schedule points, deterministic reverse,
input-only P90 patch-ENL estimation, and output/input mean matching. Only the
loader is adapted to the already prepared Toronto arrays. GT is read only after
inference, and metrics use the existing benchmark's un-clipped 0..255 convention.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "external" / "GammaBridge"
sys.path.insert(0, str(SOURCE))
from eval.run_external import denoise_one, find_target_step, make_smartstart_step_list
from eval.metrics import enl_homogeneous
from gbridge_core.diffusion import GammaBridge
from gbridge_core.network import build_default_unet

EXPECTED_SHA256 = "3f9aae3e638e814a706852a8cd5896df279b2a74b7347ecd609d75e7baff9b2d"
EXPECTED_COMMIT = "35d49ed2e7c3fbb3c7bb66fe0e3d859a1411f5ef"
SCENES = ("01_city_roads", "02_waterfront", "03_farmland")
LEGACY_SCENES = ("01_urban", "02_coast_urban_rural", "03_water_islands_farmland")


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def metrics(reference: np.ndarray, estimate: np.ndarray) -> dict:
    ref = reference.astype(np.float64)
    out = estimate.astype(np.float64)
    mse = float(np.mean((out - ref) ** 2))
    return {"psnr_db": 10 * math.log10(255.0**2 / mse),
            "ssim": float(structural_similarity(ref, out, data_range=255.0)),
            "mse": mse, "mae": float(np.mean(np.abs(out-ref))),
            "min": float(out.min()), "max": float(out.max()),
            "fraction_below_0": float(np.mean(out < 0)),
            "fraction_above_255": float(np.mean(out > 255))}


def display_png(array: np.ndarray, path: Path) -> None:
    Image.fromarray(np.rint(np.clip(array, 0, 255)).astype(np.uint8)).save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(r"E:\SAR_Data\Toronto_Paired_SAR\showcase_selected_20260926"))
    parser.add_argument("--checkpoint", type=Path, default=Path(r"E:\SAR_Models\GammaBridge\gbridge_L1.pt"))
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "gamma_bridge_probe_20260926")
    parser.add_argument("--scenes", nargs="+", choices=SCENES + LEGACY_SCENES, default=[SCENES[0]])
    args = parser.parse_args()
    for scene in args.scenes:
        if (args.output / scene).exists():
            raise FileExistsError(f"Choose a new output directory; preserving {args.output / scene}")
    source_commit = subprocess.check_output(["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True).strip()
    if source_commit != EXPECTED_COMMIT or digest(args.checkpoint) != EXPECTED_SHA256:
        raise RuntimeError("Source/checkpoint changed; audit before running")
    if not torch.cuda.is_available():
        raise RuntimeError("This timing/VRAM probe requires CUDA")
    torch.manual_seed(20260926)
    np.random.seed(20260926)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint["args"]
    if not config.get("cond_x1") or config.get("direct_pred", False) or config.get("schedule", "log") != "log":
        raise RuntimeError("Checkpoint does not match the official flagship inference path")
    network = build_default_unet(image_size=config["image_size"], in_channels=2,
                                 out_channels=1, model_channels=config["model_channels"],
                                 channel_mult=(1, 2, 4))
    state = checkpoint.get("G_ema") or checkpoint["G"]
    match = network.load_state_dict(state, strict=True)
    network = network.eval().to(device)
    bridge = GammaBridge(num_timesteps=config["T"], L_obs=config["L_obs"],
                         L_max=config["L_max"], device=device)
    metadata = {"source": "https://github.com/Teriri1999/GammaBridge",
                "source_commit": source_commit,
                "checkpoint": str(args.checkpoint), "checkpoint_sha256": EXPECTED_SHA256,
                "checkpoint_step": checkpoint["step"], "checkpoint_args": config,
                "state": "G_ema", "strict_load": True,
                "missing_keys": match.missing_keys, "unexpected_keys": match.unexpected_keys,
                "parameter_count": sum(p.numel() for p in network.parameters()),
                "python": platform.python_version(), "torch": torch.__version__,
                "numpy": np.__version__, "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0), "seed": 20260926,
                "settings": {"num_steps": 5, "ot_ode": True, "tile_size": 512,
                             "looks": "input-only 32x32 patch ENL P90, stride 16",
                             "mean_matching": True, "dtype": "float32"},
                "input": "published 0..255 array divided by 255; no extra clipping/sqrt/resize",
                "evaluation": "unclipped float outputs in published 0..255 units, fixed data_range=255",
                "reference": "Toronto ten-acquisition average pseudo-reference, not noise-free physical truth",
                "scope": "inference diagnostic, not training reproduction or formal seven-method benchmark",
                "scenes": []}
    del state, checkpoint
    args.output.mkdir(parents=True, exist_ok=True)
    print("Checkpoint strict-load succeeded", flush=True)
    for scene in args.scenes:
        destination = args.output / scene
        destination.mkdir()
        noisy_path = args.data / scene / "noisy_intensity.npy"
        noisy = np.load(noisy_path, allow_pickle=False).astype(np.float32)
        if noisy.ndim != 2 or not np.isfinite(noisy).all() or noisy.min() < 0 or noisy.max() > 255:
            raise ValueError("Unexpected prepared input")
        normalized = noisy / np.float32(255)
        L_est = float(enl_homogeneous(normalized))
        if not np.isfinite(L_est):
            raise ValueError("Cannot estimate looks")
        L_in = float(np.clip(L_est, bridge.L_obs, bridge.L_max))
        step_list = make_smartstart_step_list(find_target_step(bridge, L_in), 5)
        calls = [0]
        def record_call(_module, _inputs, _output):
            calls[0] += 1
        hook = network.register_forward_hook(record_call)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            raw = denoise_one(network, bridge, normalized, L_in, 5, device, ot_ode=True, tile_size=512)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        hook.remove()
        factor = float(normalized.mean()) / float(raw.mean()) if float(raw.mean()) > 1e-6 else 1.0
        estimate = (raw * factor * np.float32(255)).astype(np.float32)
        if estimate.shape != noisy.shape or not np.isfinite(estimate).all():
            raise ValueError("Invalid prediction")
        np.save(destination / "denoised.npy", estimate, allow_pickle=False)
        np.save(destination / "before_mean_matching.npy", raw * np.float32(255), allow_pickle=False)
        display_png(estimate, destination / "denoised.png")
        # No reference is loaded until predictions and all inference settings are fixed.
        reference_path = args.data / scene / "ground_truth.npy"
        reference = np.load(reference_path, allow_pickle=False)
        result = {"scene": scene, "input_path": str(noisy_path),
                  "input_sha256": digest(noisy_path), "reference_sha256": digest(reference_path),
                  "shape": list(noisy.shape), "estimated_looks": L_est, "used_looks": L_in,
                  "schedule_points": step_list, "actual_network_calls": calls[0],
                  "inference_seconds": elapsed,
                  "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                  "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
                  "mean_matching_factor": factor, "input_mean": float(noisy.mean()),
                  "output_mean": float(estimate.mean()),
                  "metrics": metrics(reference, estimate), "noisy_metrics": metrics(reference, noisy)}
        (destination / "run.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        cell_w, cell_h = noisy.shape[1], noisy.shape[0]
        preview = Image.new("L", (3 * cell_w, cell_h + 35), 255)
        draw = ImageDraw.Draw(preview)
        font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 22)
        for i, (label, array) in enumerate((("Noisy", noisy), ("Pseudo-reference", reference), ("Gamma-Bridge (default)", estimate))):
            tile = Image.fromarray(np.rint(np.clip(array, 0, 255)).astype(np.uint8))
            preview.paste(tile, (i * cell_w, 0))
            draw.text((i * cell_w + 10, cell_h + 5), label, fill=0, font=font)
        preview.save(destination / "preview.png")
        metadata["scenes"].append(result)
        (args.output / "probe.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
