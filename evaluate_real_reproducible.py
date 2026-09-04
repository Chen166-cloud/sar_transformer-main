"""Canonical real-SAR evaluation using grouped manifests and shared ROIs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.io import loadmat

from experiment_protocol import load_checkpoint_strict, prepare_run_directory, seed_everything
from numeric_domain import INTENSITY_DOMAIN, normalize_real_intensity
from resplit_real_sar_dataset import verify_manifest
from sar_metrics import real_sar_metrics
from transform_main import TransSARV2_DualFreqNG_Bottle


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--roi-size", type=int, default=32)
    parser.add_argument("--num-rois", type=int, default=5)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--save-images", action="store_true")
    return parser.parse_args()


def _load_image(path: Path) -> np.ndarray:
    data = loadmat(path)
    for key in ("noisy", "sar", "image", "img", "data", "input"):
        if key in data:
            return normalize_real_intensity(data[key])
    raise KeyError(f"No supported image field in {path}")


def _write_gray(path: Path, image: np.ndarray) -> None:
    cv2.imwrite(str(path), np.round(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8))


def _display_stretch(image: np.ndarray) -> np.ndarray:
    low, high = np.percentile(image, (1, 99))
    if high <= low:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - low) / (high - low), 0.0, 1.0).astype(np.float32)


def _summary(values, rng, bootstrap_samples):
    array = np.asarray(values, dtype=np.float64)
    means = np.empty(bootstrap_samples, dtype=np.float64)
    for index in range(bootstrap_samples):
        means[index] = np.mean(rng.choice(array, size=len(array), replace=True))
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "ci95_low": float(np.percentile(means, 2.5)),
        "ci95_high": float(np.percentile(means, 97.5)),
    }


def main() -> int:
    args = parse_args()
    seed_everything(args.seed)
    root = Path(args.dataset_root).resolve()
    with open(args.split_manifest, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    verification = verify_manifest(root, manifest)
    relative_files = manifest["files"][args.split]
    if args.max_images:
        relative_files = relative_files[: args.max_images]
    if not relative_files:
        raise RuntimeError(f"No files in manifest split {args.split}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    run_config = vars(args).copy()
    run_config.update(
        {
            "numeric_domain": INTENSITY_DOMAIN,
            "normalization": "per-image percentile [1,99] to [0,1]",
            "roi_policy": "selected once on noisy and reused for noisy/prediction",
            "manifest_verification": verification,
        }
    )
    output = prepare_run_directory(
        args.output_dir,
        run_config,
        [__file__, "transform_main.py", "numeric_domain.py", "sar_metrics.py", args.split_manifest],
        checkpoint_path=args.checkpoint,
    )
    image_dir = output / "images"
    if args.save_images:
        image_dir.mkdir()

    model = TransSARV2_DualFreqNG_Bottle(
        ablation="full", numeric_domain=INTENSITY_DOMAIN
    )
    load_checkpoint_strict(model, args.checkpoint)
    model.to(device).eval()

    rows = []
    with torch.no_grad():
        for relative_path in relative_files:
            path = root / relative_path
            noisy = _load_image(path)
            prediction = model(torch.from_numpy(noisy)[None, None].to(device)).cpu().numpy()[0, 0]
            prediction = np.clip(prediction, 0.0, 1.0)
            metrics = real_sar_metrics(
                noisy, prediction, roi_size=args.roi_size, num_rois=args.num_rois, seed=args.seed
            )
            rows.append(
                {
                    "file": relative_path,
                    "roi_coordinates": json.dumps(metrics.pop("rois"), separators=(",", ":")),
                    **metrics,
                }
            )
            if args.save_images:
                stem = Path(relative_path).stem
                ratio = noisy / np.maximum(prediction, 1e-6)
                gx = cv2.Sobel(prediction, cv2.CV_32F, 1, 0, ksize=3)
                gy = cv2.Sobel(prediction, cv2.CV_32F, 0, 1, ksize=3)
                _write_gray(image_dir / f"{stem}_noisy.png", noisy)
                _write_gray(image_dir / f"{stem}_prediction.png", prediction)
                _write_gray(image_dir / f"{stem}_ratio.png", _display_stretch(ratio))
                _write_gray(image_dir / f"{stem}_edge.png", _display_stretch(np.sqrt(gx * gx + gy * gy)))

    with open(output / "metrics_per_image.csv", "x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    rng = np.random.default_rng(args.seed)
    summary = {"num_images": len(rows), "split": args.split}
    for key in ("noisy_enl", "prediction_enl", "m_index", "epi"):
        summary[key] = _summary([row[key] for row in rows], rng, args.bootstrap_samples)
    with open(output / "summary.json", "x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
