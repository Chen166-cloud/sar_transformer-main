"""Verify the Apple Silicon image with one real optimizer step."""

from __future__ import annotations

import importlib
import json
import platform
import sys
from pathlib import Path

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def main() -> None:
    dependency_names = [
        "cv2",
        "mmcv",
        "numpy",
        "pandas",
        "scipy",
        "timm",
        "torchvision",
    ]
    dependencies = {}
    for name in dependency_names:
        module = importlib.import_module(name)
        dependencies[name] = getattr(module, "__version__", "unknown")

    from transform_main import TransSARV2_DualFreqNG_Bottle

    device = torch.device("cpu")
    torch.manual_seed(42)
    model = TransSARV2_DualFreqNG_Bottle().to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    sample = torch.rand(1, 1, 64, 64, device=device)
    target = torch.rand_like(sample)

    optimizer.zero_grad(set_to_none=True)
    output = model(sample)
    loss = torch.nn.functional.mse_loss(output, target)
    loss.backward()
    optimizer.step()

    gradients_finite = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item())
        for parameter in model.parameters()
    )
    result = {
        "architecture": platform.machine(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "dependencies": dependencies,
        "input_shape": list(sample.shape),
        "output_shape": list(output.shape),
        "loss": float(loss.detach().item()),
        "output_finite": bool(torch.isfinite(output).all().item()),
        "gradients_finite": gradients_finite,
        "optimizer_step_completed": True,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if result["architecture"] not in {"aarch64", "arm64"}:
        raise SystemExit("The image is not running as Linux/arm64")
    if result["output_shape"] != [1, 1, 64, 64]:
        raise SystemExit("Unexpected model output shape")
    if not result["output_finite"] or not result["gradients_finite"]:
        raise SystemExit("The training step produced NaN/Inf")


if __name__ == "__main__":
    main()
