"""Canonical intensity-domain PSNR/SSIM evaluation for paired synthetic SAR."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat

from experiment_protocol import load_checkpoint_strict, prepare_run_directory, seed_everything
from synthetic_manifest import verify_paired_sar_manifest
from numeric_domain import INTENSITY_DOMAIN, prepare_synthetic_pair
from sar_metrics import psnr, ssim
from ablation_config import ABLATION_PRESETS
from transform_main import TransSARV2_DualFreqNG_Bottle


def summarize_rows_by_l(rows):
    grouped = {}
    for row in rows:
        looks = row.get("global_L")
        if looks in (None, ""):
            continue
        grouped.setdefault(int(looks), []).append(row)
    if not grouped:
        return {}, {}
    metric_keys = [
        key for key in rows[0]
        if key not in ("file", "global_L")
    ]
    by_l = {}
    for looks, group_rows in sorted(grouped.items()):
        summary = {"global_L": looks, "num_images": len(group_rows)}
        for key in metric_keys:
            values = [float(row[key]) for row in group_rows]
            summary[key + "_mean"] = float(np.mean(values))
            summary[key + "_median"] = float(np.median(values))
        by_l[f"L{looks}"] = summary
    macro = {"num_L_values": len(by_l)}
    for key in metric_keys:
        macro[key + "_mean"] = float(np.mean([
            summary[key + "_mean"] for summary in by_l.values()
        ]))
    return by_l, macro


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset")
    parser.add_argument("--dataset-root")
    parser.add_argument("--split-manifest")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ablation", choices=sorted(ABLATION_PRESETS), default="full")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--crop", type=int, default=256)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--verify-data-hashes", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_mode = bool(args.dataset_root or args.split_manifest)
    if manifest_mode and args.dataset:
        raise ValueError("Choose either the manifest pair or --dataset, not both")
    if manifest_mode and (not args.dataset_root or not args.split_manifest):
        raise ValueError("--dataset-root and --split-manifest must be supplied together")
    if not manifest_mode and not args.dataset:
        raise ValueError("Supply --dataset-root/--split-manifest or the legacy --dataset")
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    run_config = vars(args).copy()
    run_config.update(
        {
            "numeric_domain": INTENSITY_DOMAIN,
            "data_range": 1.0,
            "border_crop": 0,
            "data_protocol": "paired_sar_manifest" if manifest_mode else "legacy_directory",
        }
    )
    source_files = [__file__, "transform_main.py", "numeric_domain.py", "sar_metrics.py"]
    if manifest_mode:
        source_files.extend(["synthetic_manifest.py", args.split_manifest])
    output = prepare_run_directory(
        args.output_dir,
        run_config,
        source_files,
        checkpoint_path=args.checkpoint,
    )
    model = TransSARV2_DualFreqNG_Bottle(
        ablation=args.ablation, numeric_domain=INTENSITY_DOMAIN
    )
    load_checkpoint_strict(model, args.checkpoint)
    model.to(device).eval()
    if manifest_mode:
        dataset_root = Path(args.dataset_root).resolve()
        with open(args.split_manifest, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        verify_paired_sar_manifest(
            dataset_root,
            manifest,
            required_splits=(args.split,),
            verify_hashes=args.verify_data_hashes,
            require_exact_partition=args.verify_data_hashes,
            verify_files=args.verify_data_hashes,
        )
        files = [dataset_root / relative for relative in manifest["files"][args.split]]
        relative_by_path = {
            str(dataset_root / relative): relative
            for relative in manifest["files"][args.split]
        }
        looks_by_file = manifest.get("global_L_by_file", {})
    else:
        files = sorted(Path(args.dataset).resolve().rglob("*.mat"))
        relative_by_path = {}
        looks_by_file = {}
    if args.max_images:
        files = files[: args.max_images]
    if not files:
        raise RuntimeError("No synthetic .mat files found")

    rows = []
    with torch.no_grad():
        for path in files:
            data = loadmat(path)
            noisy, clean = prepare_synthetic_pair(data["noisy"], data["clean"], INTENSITY_DOMAIN)
            height, width = noisy.shape
            if height < args.crop or width < args.crop:
                raise ValueError(f"{path} is smaller than crop={args.crop}: {noisy.shape}")
            top = (height - args.crop) // 2
            left = (width - args.crop) // 2
            noisy = noisy[top : top + args.crop, left : left + args.crop]
            clean = clean[top : top + args.crop, left : left + args.crop]
            tensor = torch.from_numpy(noisy)[None, None].to(device)
            prediction = model(tensor).cpu().numpy()[0, 0]
            row = {
                    "file": str(path),
                    "noisy_psnr": psnr(noisy, clean),
                    "noisy_ssim": ssim(noisy, clean),
                    "prediction_psnr": psnr(prediction, clean),
                    "prediction_ssim": ssim(prediction, clean),
                }
            if manifest_mode and looks_by_file:
                relative = relative_by_path[str(path)]
                row["global_L"] = int(looks_by_file[relative])
            rows.append(row)
    with open(output / "metrics_per_image.csv", "x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"num_images": len(rows)}
    for key in rows[0]:
        if key not in ("file", "global_L"):
            values = [row[key] for row in rows]
            summary[key + "_mean"] = float(np.mean(values))
            summary[key + "_median"] = float(np.median(values))
    by_l, macro = summarize_rows_by_l(rows)
    if by_l:
        summary["by_L"] = by_l
        summary["macro_average"] = macro
        per_l_fields = list(next(iter(by_l.values())))
        with open(output / "summary_by_L.csv", "x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=per_l_fields)
            writer.writeheader()
            writer.writerows(by_l.values())
            macro_row = {
                "global_L": "macro",
                "num_images": sum(item["num_images"] for item in by_l.values()),
            }
            for key, value in macro.items():
                if key != "num_L_values":
                    macro_row[key] = value
            writer.writerow(macro_row)
    with open(output / "summary.json", "x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
