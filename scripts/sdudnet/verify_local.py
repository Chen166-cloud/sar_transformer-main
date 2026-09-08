"""Numerical parity with the computational steps of the official test.py.

Uses an actually supplied JPEG because the author's S4_L1.png is absent.
No GUI or OpenCV dependency is needed for the numerical comparison.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToTensor

import run_local as adapter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=adapter.PROJECT_ROOT / "output" / "sdudnet_local" / "verification.json")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    path = adapter.OFFICIAL_ROOT / "my_datasets" / "01233.jpg"
    with Image.open(path) as image:
        official_input = ToTensor()(image).reshape(1, 1, 256, 256)
    local_input, _ = adapter.load_array(path, "noisy")
    np.testing.assert_array_equal(official_input[0, 0].numpy(), local_input)
    spec = importlib.util.spec_from_file_location("reference_dnn", adapter.OFFICIAL_ROOT / "net" / "DNN.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    variants = {}
    for variant in ("real", "synthetic"):
        reference = module.DNN()
        reference.load_state_dict(torch.load(adapter.OFFICIAL_ROOT / "models" / f"{variant}.pth",
                                            map_location="cpu", weights_only=True), strict=True)
        # test.py leaves the freshly constructed network in training mode.
        reference.to(device)
        assert reference.training
        with torch.no_grad():
            expected, _ = reference(official_input.to(device))
        local_model, _ = adapter.build_model(variant, device)
        actual, _ = adapter.predict(local_model, local_input, device)
        expected = expected[0, 0].cpu().numpy()
        np.testing.assert_array_equal(actual, expected)
        variants[variant] = {"train_mode_author_vs_eval_mode_adapter_max_abs": float(np.max(np.abs(actual - expected))),
                             "input_preprocessing_exact": True, "forward_exact": True}
        del reference, expected, local_model
    # An unequal-sided image exercises the adapter's removal of the hard-coded reshape.
    crop = local_input[:173, :211]
    cpu_model, _ = adapter.build_model("real", torch.device("cpu"))
    cpu_result, cpu_seconds = adapter.predict(cpu_model, crop, torch.device("cpu"))
    assert cpu_result.shape == (173, 211) and np.isfinite(cpu_result).all()
    cpu_gpu_max_abs = None
    if device.type == "cuda":
        gpu_model, _ = adapter.build_model("real", device)
        gpu_result, _ = adapter.predict(gpu_model, crop, device)
        np.testing.assert_allclose(cpu_result, gpu_result, atol=2e-5, rtol=2e-5)
        cpu_gpu_max_abs = float(np.max(np.abs(cpu_result - gpu_result)))
    report = {"passed": True, "source": adapter.verify_source(), "device": str(device),
              "comparison": "Computational steps from test.py; available JPEG substituted for missing S4_L1.png",
              "variants": variants, "cpu_non_square": {"shape": list(cpu_result.shape),
                  "forward_seconds": cpu_seconds, "cpu_gpu_max_abs": cpu_gpu_max_abs},
              "verification_script_sha256": adapter.sha256(Path(__file__))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
