"""Small, explicit-D=1 parity and domain tests; does not train or run a benchmark."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import run_local as adapter


def reference_wrapper(model, tools, device):
    """Literal official funcDRUnet arithmetic, CPU tensor creation corrected.

    Kept separate from Denoiser: no cache-related math, no padding, no hidden reuse
    of adapter normalization. Used on dimensions divisible by 8.
    """
    def denoise(y, sig):
        y_prime = y.copy()
        reshaped = False
        if y_prime.ndim == 2:
            y_prime = np.atleast_3d(y_prime)
            reshaped = True
        x = np.zeros_like(y_prime)
        count = y_prime.shape[2]
        qm = tools.quantile(y_prime.flatten(), 0.003)
        qM = tools.quantile(y_prime.flatten(), 0.997)
        y_prime = (y_prime - qm) / (qM - qm)
        noise_sigma = torch.FloatTensor([sig / (qM - qm)]).to(device)
        y_prime = torch.FloatTensor(y_prime[None, None, :, :, :]).to(device)
        model.eval()
        for channel in range(count):
            with torch.no_grad():
                x[:, :, channel] = model(y_prime[:, :, :, :, channel], noise_sigma).cpu().numpy().squeeze()
        x = x * (qM - qm) + qm
        x[np.isnan(x)] = 0
        return np.squeeze(x) if reshaped else x
    return denoise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, default=adapter.PROJECT_ROOT / "output" / "mulog_drunet_local" / "verification.json")
    args = parser.parse_args()
    device = torch.device(args.device)
    adapter.configure_runtime()
    path = adapter.PROJECT_ROOT / "datasets/NWPU_RESISC45_SAR_global_L_v2/val/L1/airplane/airplane_00003.mat"
    sample, _ = adapter.load_array(path)
    crop = sample[:32, :40]
    intensity = adapter.to_intensity(crop, "intensity", 1.0)
    denoiser, core, gaussian, provenance = adapter.build_denoiser(device)
    reference = reference_wrapper(denoiser.model, core.to, device)
    # Compare independently written wrappers before comparing the whole ADMM path.
    y = np.log(np.maximum(crop, np.min(crop[crop > 0])))[:, :, None]
    expected_forward = reference(y, 1.0)
    actual_forward = denoiser(y, 1.0)
    np.testing.assert_array_equal(actual_forward, expected_forward)
    with adapter.compatibility(core):
        expected = core.mulog(intensity, 1.0, reference, "T", 2)
    actual, seconds, _ = adapter.predict(intensity, 1.0, 2, denoiser, core)
    np.testing.assert_array_equal(actual, np.real(expected))
    # Prove the scheduling shim does not alter the operator on this crop.
    original_np = core.mf.np
    try:
        core.mf.np = adapter.NumpyCompat()
        original_scheduling = core.mulog(intensity, 1.0, reference, "T", 2)
    finally:
        core.mf.np = original_np
    np.testing.assert_array_equal(actual, np.real(original_scheduling))
    np.testing.assert_array_equal(denoiser(y, 1.0), actual_forward)
    amplitude = np.sqrt(crop) * 7.0
    transformed = adapter.to_intensity(amplitude, "amplitude", 7.0)
    np.testing.assert_allclose(transformed, crop, rtol=1e-14, atol=1e-15)
    restored = adapter.from_intensity(actual, "amplitude", 7.0)
    np.testing.assert_allclose((restored / 7.0) ** 2, actual, rtol=1e-14, atol=1e-15)
    # Exercise boundary padding and float-preserving shape restoration.
    odd_input = intensity[:29, :37]
    odd_output, odd_seconds, _ = adapter.predict(odd_input, 1.0, 2, denoiser, core)
    assert odd_output.shape == odd_input.shape and np.isfinite(odd_output).all()
    assert core.mf.np is np  # compatibility context restored the actual module
    assert "complex" not in np.__dict__  # no process-wide NumPy alias injection
    report = {"passed": True, "device": str(device), "source": provenance,
              "sample": str(path), "crop_shape": list(crop.shape), "looks": 1.0,
              "comparison": "official funcDRUnet arithmetic with CPU tensor fix, official MuLoG/ADMM; no padding on parity crop",
              "forward_max_abs": float(np.max(np.abs(actual_forward - expected_forward))),
              "whole_mulog_max_abs": float(np.max(np.abs(actual - np.real(expected)))),
              "domain_roundtrip_passed": True, "admm_iterations_for_parity": 2,
              "cached_model_repeated_call_max_abs": 0.0,
              "original_worker_count_max_abs": float(np.max(np.abs(actual - np.real(original_scheduling)))),
              "runtime_seconds": seconds,
              "non_multiple_of_8": {"shape": list(odd_output.shape), "seconds": odd_seconds, "all_finite": True},
              "numpy_module_restored": True, "verification_script_sha256": adapter.sha256(Path(__file__))}
    # On CUDA, independently execute the author's unmodified wrapper too.
    if device.type == "cuda":
        import os
        previous = Path.cwd()
        try:
            os.chdir(adapter.OFFICIAL_ROOT)
            official_forward = gaussian.funcDRUnet(y, 1.0)
            with adapter.compatibility(core):
                official_uncached = core.mulog(intensity, 1.0, gaussian.funcDRUnet, "T", 2)
        finally:
            os.chdir(previous)
        np.testing.assert_array_equal(actual_forward, official_forward)
        np.testing.assert_array_equal(actual, np.real(official_uncached))
        report["unmodified_author_gpu_wrapper_max_abs"] = float(np.max(np.abs(actual_forward - official_forward)))
        report["unmodified_author_uncached_mulog_max_abs"] = float(np.max(np.abs(actual - np.real(official_uncached))))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
