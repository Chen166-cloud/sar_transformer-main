import importlib
import json
from pathlib import Path
import sys

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def main():
    modules = [
        "cv2",
        "mmcv",
        "numpy",
        "pandas",
        "scipy",
        "timm",
        "torchvision",
    ]
    versions = {}
    for name in modules:
        module = importlib.import_module(name)
        versions[name] = getattr(module, "__version__", "unknown")

    from transform_main import TransSARV2_DualFreqNG_Bottle

    checkpoint_path = (
        REPOSITORY_ROOT
        / "experiments"
        / "TransSARV2_DualFreqNG_Bottle"
        / "best_model.pth"
    )

    model = TransSARV2_DualFreqNG_Bottle()
    checkpoint_loaded = False
    missing_keys = []
    unexpected_keys = []
    if checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
        state_dict = {
            (key[7:] if key.startswith("module.") else key): value
            for key, value in state_dict.items()
        }
        incompatible = model.load_state_dict(state_dict, strict=True)
        missing_keys = list(incompatible.missing_keys)
        unexpected_keys = list(incompatible.unexpected_keys)
        checkpoint_loaded = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    sample = torch.rand(1, 1, 256, 256, device=device)
    with torch.no_grad():
        output = model(sample)

    result = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "dependencies": versions,
        "checkpoint": str(checkpoint_path),
        "checkpoint_loaded": checkpoint_loaded,
        "checkpoint_missing_keys": missing_keys,
        "checkpoint_unexpected_keys": unexpected_keys,
        "input_shape": list(sample.shape),
        "output_shape": list(output.shape),
        "output_finite": bool(torch.isfinite(output).all().item()),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if not result["cuda_available"]:
        raise SystemExit("CUDA is not available inside the container")
    if not checkpoint_loaded or missing_keys or unexpected_keys:
        raise SystemExit("Checkpoint compatibility verification failed")
    if result["output_shape"] != [1, 1, 256, 256] or not result["output_finite"]:
        raise SystemExit("Model forward verification failed")


if __name__ == "__main__":
    main()
