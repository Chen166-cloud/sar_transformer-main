"""Small GPU/CPU smoke test for the grouped split, model presets, and domains."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import torch

from experiment_protocol import load_checkpoint_strict, seed_everything
from numeric_domain import INTENSITY_DOMAIN
from resplit_real_sar_dataset import verify_manifest
from build_bsds500_split_manifest import verify_manifest as verify_synthetic_manifest
from ablation_config import ABLATION_PRESETS
from transform_main import TransSARV2_DualFreqNG_Bottle
from utils import BSD_SAR, RealSARDataset


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="datasets/real_sar_dataset")
    parser.add_argument("--manifest", default="datasets/real_sar_dataset/real_split_grouped_seed42.json")
    parser.add_argument("--synthetic-root", default="datasets/bsds500_synthetic_dataset")
    parser.add_argument(
        "--synthetic-manifest",
        default="datasets/bsds500_synthetic_dataset/official_split_manifest.json",
    )
    parser.add_argument(
        "--checkpoint",
        default="experiments/TransSARV2_DualFreqNG_Bottle/best_model.pth",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--input-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seed_everything(args.seed)
    root = Path(args.dataset_root).resolve()
    with open(args.manifest, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    split_check = verify_manifest(root, manifest)
    synthetic_root = Path(args.synthetic_root).resolve()
    with open(args.synthetic_manifest, "r", encoding="utf-8") as handle:
        synthetic_manifest = json.load(handle)
    synthetic_split_check = verify_synthetic_manifest(
        synthetic_root, synthetic_manifest, verify_hashes=True
    )
    real_sample = RealSARDataset(root, args.manifest, "train")[0]
    synthetic_noisy, synthetic_clean, _ = BSD_SAR(
        synthetic_root,
        (min(256, args.input_size), min(256, args.input_size)),
        False,
        INTENSITY_DOMAIN,
        args.synthetic_manifest,
        "val",
    )[0]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    results = {
        "device": str(device),
        "split_check": split_check,
        "synthetic_split_check": synthetic_split_check,
        "real_range": [float(real_sample.min()), float(real_sample.max())],
        "synthetic_noisy_range": [float(synthetic_noisy.min()), float(synthetic_noisy.max())],
        "synthetic_clean_range": [float(synthetic_clean.min()), float(synthetic_clean.max())],
        "presets": {},
    }
    full_initialization = None
    for preset in ABLATION_PRESETS:
        seed_everything(args.seed)
        model = TransSARV2_DualFreqNG_Bottle(
            ablation=preset, numeric_domain=INTENSITY_DOMAIN
        ).to(device)
        initialization = {
            key: hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()
            for key, value in model.state_dict().items()
        }
        if full_initialization is None:
            full_initialization = initialization
            initialization_mismatches = []
        else:
            shared = sorted(set(full_initialization).intersection(initialization))
            initialization_mismatches = [
                key for key in shared if full_initialization[key] != initialization[key]
            ]
        if preset == "full":
            load_checkpoint_strict(model, args.checkpoint)
        model.eval()
        sample = torch.rand(1, 1, args.input_size, args.input_size, device=device)
        with torch.no_grad():
            output = model(sample)
        results["presets"][preset] = {
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "shape": list(output.shape),
            "finite": bool(torch.isfinite(output).all().item()),
            "minimum": float(output.min().item()),
            "maximum": float(output.max().item()),
            "shared_initialization_mismatches": initialization_mismatches,
        }
        del model, sample, output
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(json.dumps(results, indent=2, sort_keys=True))
    if not all(item["finite"] for item in results["presets"].values()):
        raise RuntimeError("At least one ablation produced NaN/Inf")
    if not all(
        not item["shared_initialization_mismatches"]
        for item in results["presets"].values()
    ):
        raise RuntimeError("Shared ablation parameters do not have identical initialization")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
