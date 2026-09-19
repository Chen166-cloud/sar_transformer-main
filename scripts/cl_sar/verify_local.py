"""Verify adapter parity with pinned author preprocessing and postprocessing.

This executes original helper definitions via AST (without unrelated training
imports), and calls the untouched MDN1 directly for reference inference.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import torch

import run_local as adapter


def official_helpers() -> dict:
    namespace = {"np": np, "torch": torch}
    for relative, names in [
        ("basicsr/data/data_util.py", {"max_normalize", "max_denormalize", "normalizedAmp2intensity"}),
        ("basicsr/utils/img_util.py", {"tensor2img"}),
    ]:
        path = adapter.OFFICIAL_ROOT / relative
        tree = ast.parse(path.read_text(encoding="utf-8"))
        definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        if {node.name for node in definitions} != names:
            raise AssertionError("Pinned official helper definitions are missing")
        exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def compare_official(model, image, device, helper):
    official_amplitude = np.sqrt(image)
    expected_input, amp_min, amp_max = helper["max_normalize"](official_amplitude)
    local_input, info = adapter.prepare_input(image, "intensity")
    np.testing.assert_array_equal(local_input, expected_input)
    with torch.inference_mode():
        direct = model(torch.from_numpy(expected_input)[None, None].to(device))
    official_normalized = helper["tensor2img"]([torch.clamp(direct, -20, 20)], out_type=np.float32)
    expected_output = helper["normalizedAmp2intensity"](official_normalized, amp_min, amp_max)
    raw, seconds = adapter.predict(model, local_input, device)
    output = adapter.restore_output(raw, info)
    np.testing.assert_array_equal(raw, direct[0, 0].cpu().numpy())
    np.testing.assert_allclose(output, expected_output, rtol=2e-7, atol=2e-7)
    amplitude_input, amplitude_info = adapter.prepare_input(np.sqrt(image), "amplitude")
    np.testing.assert_array_equal(amplitude_input, expected_input)
    amplitude_output = adapter.restore_output(raw, amplitude_info)
    np.testing.assert_allclose(amplitude_output ** 2, output, rtol=2e-7, atol=2e-7)
    return {"shape": list(image.shape), "preprocessing_exact": True, "raw_forward_exact": True,
            "official_output_max_abs": float(np.max(np.abs(output - expected_output))),
            "amplitude_has_no_second_sqrt": True,
            "amplitude_squared_vs_intensity_max_abs": float(np.max(np.abs(amplitude_output ** 2 - output))),
            "forward_seconds_including_first_call": seconds}, output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--input", type=Path, default=adapter.PROJECT_ROOT /
                        "datasets/NWPU_RESISC45_SAR_global_L_v2/val/L1/airplane/airplane_00003.mat")
    parser.add_argument("--output", type=Path, default=adapter.PROJECT_ROOT / "output/cl_sar_local/verification.json")
    args = parser.parse_args()
    adapter.configure_runtime()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    model, provenance = adapter.build_model(device)
    helpers = official_helpers()
    image, input_info = adapter.load_array(args.input)
    parity, _ = compare_official(model, image, device, helpers)
    # Non-square input exercises official padding/cropping; zeros test min_nonzero.
    rng = np.random.default_rng(20260918)
    nonsquare = rng.uniform(0.01, 1, size=(47, 65)).astype(np.float32)
    nonsquare[0, 0] = 0
    nonsquare_parity, device_output = compare_official(model, nonsquare, device, helpers)
    cpu_model, _ = adapter.build_model(torch.device("cpu"))
    cpu_parity, cpu_output = compare_official(cpu_model, nonsquare, torch.device("cpu"), helpers)
    np.testing.assert_allclose(cpu_output, device_output, rtol=3e-5, atol=3e-5)
    # Explicitly exercise the author's [0,1] clipping rather than trusting a
    # sample prediction to happen to have out-of-range elements.
    raw = np.array([[-21, -0.1, 0, 0.5, 1, 1.1, 21]], dtype=np.float32)
    _, info = adapter.prepare_input(nonsquare, "intensity")
    expected = helpers["normalizedAmp2intensity"](
        helpers["tensor2img"](torch.from_numpy(raw)[None, None].clamp(-20, 20), out_type=np.float32),
        np.float32(info["amplitude_min_nonzero"]), np.float32(info["amplitude_max"]))
    np.testing.assert_allclose(adapter.restore_output(raw, info), expected, rtol=2e-7, atol=2e-7)
    report = {"passed": True, "scope": "author helper/forward parity and domain adapter checks; not paper quality reproduction",
              "source": provenance, "input": input_info, "device": str(device),
              "known_synthetic_intensity_sample": parity, "non_square_with_zero": nonsquare_parity,
              "cpu_non_square": cpu_parity, "cpu_device_max_abs": float(np.max(np.abs(cpu_output - device_output))),
              "author_clipping_edge_cases_passed": True,
              "source_after_validation": adapter.verify_source(),
              "verification_script_sha256": adapter.sha256(Path(__file__)),
              "adapter_sha256": adapter.sha256(Path(adapter.__file__))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
