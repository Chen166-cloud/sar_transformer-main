"""Canonical supervised trainer for the intensity-v1 ablation protocol."""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from experiment_protocol import (
    checkpoint_payload,
    dataloader_generator,
    load_checkpoint_strict,
    prepare_run_directory,
    seed_everything,
    seed_worker,
    sha256_file,
)
from numeric_domain import INTENSITY_DOMAIN, NUMERIC_DOMAIN_CHOICES
from sar_metrics import batch_psnr_ssim
from ablation_config import ABLATION_PRESETS
from synthetic_manifest import verify_paired_sar_manifest
from transform_main import TransSARV2_DualFreqNG_Bottle
from utils import BSD_SAR


def total_variation(batch: torch.Tensor) -> torch.Tensor:
    vertical = torch.abs(batch[:, :, 1:, :] - batch[:, :, :-1, :]).mean()
    horizontal = torch.abs(batch[:, :, :, 1:] - batch[:, :, :, :-1]).mean()
    return vertical + horizontal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dataset")
    parser.add_argument("--val-dataset")
    parser.add_argument("--dataset-root")
    parser.add_argument("--split-manifest")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ablation", choices=sorted(ABLATION_PRESETS), default="full")
    parser.add_argument("--numeric-domain", choices=NUMERIC_DOMAIN_CHOICES, default=INTENSITY_DOMAIN)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--crop", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--lambda-tv", type=float, default=5e-7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--load-checkpoint")
    parser.add_argument("--verify-data-hashes", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=100,
        help="Append batch-level JSONL progress every N training batches; 0 disables it.",
    )
    return parser.parse_args()


def _limited(loader, maximum):
    for index, batch in enumerate(loader):
        if maximum and index >= maximum:
            break
        yield batch


def _inventory(path: str) -> dict:
    root = Path(path).resolve()
    files = sorted(item.name for item in root.glob("*.mat"))
    return {"root": str(root), "count": len(files), "files": files}


def _resolve_data_args(args: argparse.Namespace) -> bool:
    manifest_mode = bool(args.dataset_root or args.split_manifest)
    direct_mode = bool(args.train_dataset or args.val_dataset)
    if manifest_mode and direct_mode:
        raise ValueError(
            "Choose either --dataset-root/--split-manifest or "
            "--train-dataset/--val-dataset, not both"
        )
    if manifest_mode:
        if not args.dataset_root or not args.split_manifest:
            raise ValueError("--dataset-root and --split-manifest must be supplied together")
        return True
    if not args.train_dataset or not args.val_dataset:
        raise ValueError(
            "Supply the official --dataset-root/--split-manifest pair. The direct "
            "directory arguments remain available only for historical compatibility."
        )
    return False


def main() -> int:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    if args.progress_interval < 0:
        raise ValueError("progress-interval must be non-negative")
    manifest_mode = _resolve_data_args(args)
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    run_config = vars(args).copy()
    run_config.update(
        {
            "model": "TransSARV2_DualFreqNG_Bottle",
            "metric_border": 0,
            "data_protocol": "paired_sar_manifest" if manifest_mode else "legacy_directories",
        }
    )
    source_files = [
        __file__, "transform_main.py", "numeric_domain.py", "sar_metrics.py", "utils.py",
        "ablation_config.py", "experiment_protocol.py",
    ]
    if manifest_mode:
        with open(args.split_manifest, "r", encoding="utf-8") as manifest_handle:
            manifest = json.load(manifest_handle)
        verification = verify_paired_sar_manifest(
            Path(args.dataset_root),
            manifest,
            verify_hashes=args.verify_data_hashes,
            require_exact_partition=args.verify_data_hashes,
            verify_files=args.verify_data_hashes,
        )
        run_config["dataset"] = manifest.get("dataset")
        run_config["manifest_verification"] = verification
        run_config["L_assignment"] = manifest.get("L_assignment")
        run_config["validation_sets"] = {
            label: {
                "global_L": definition.get("global_L"),
                "count": definition.get("count"),
            }
            for label, definition in manifest.get("validation_sets", {}).items()
        }
        source_files.extend(["synthetic_manifest.py", args.split_manifest])
    output_dir = prepare_run_directory(
        args.output_dir,
        run_config,
        source_files=source_files,
        checkpoint_path=args.load_checkpoint,
    )
    inventory_path = output_dir / "dataset_inventory.json"
    with open(inventory_path, "x", encoding="utf-8") as handle:
        if manifest_mode:
            inventory = {
                "dataset_root": str(Path(args.dataset_root).resolve()),
                "split_manifest": str(Path(args.split_manifest).resolve()),
                "split_manifest_sha256": sha256_file(args.split_manifest),
                "counts": manifest["counts"],
                "train_files": manifest["files"]["train"],
                "val_files": manifest["files"]["val"],
            }
        else:
            inventory = {
                "warning": "legacy directory mode does not guarantee an independent BSDS500 test split",
                "train": _inventory(args.train_dataset),
                "val": _inventory(args.val_dataset),
            }
        json.dump(
            inventory,
            handle,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        handle.write("\n")

    crop = (args.crop, args.crop)
    if manifest_mode:
        train_set = BSD_SAR(
            args.dataset_root, crop, True, args.numeric_domain, args.split_manifest, "train"
        )
        val_set = BSD_SAR(
            args.dataset_root, crop, False, args.numeric_domain, args.split_manifest, "val"
        )
    else:
        train_set = BSD_SAR(args.train_dataset, crop, True, args.numeric_domain)
        val_set = BSD_SAR(args.val_dataset, crop, False, args.numeric_domain)
    generator = dataloader_generator(args.seed)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        worker_init_fn=seed_worker,
        generator=dataloader_generator(args.seed + 1),
    )

    model = TransSARV2_DualFreqNG_Bottle(
        ablation=args.ablation, numeric_domain=args.numeric_domain
    ).to(device)
    if args.load_checkpoint:
        load_checkpoint_strict(model, args.load_checkpoint, map_location=device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-6
    )

    metrics_path = output_dir / "epoch_metrics.csv"
    progress_path = output_dir / "batch_progress.jsonl"
    progress_path.touch(exist_ok=False)
    base_metric_fields = [
        "epoch", "learning_rate", "train_loss", "val_loss", "val_psnr", "val_ssim"
    ]
    looks_by_file = manifest.get("global_L_by_file") if manifest_mode else None
    validation_looks = sorted(
        {int(looks_by_file[path]) for path in manifest["files"]["val"]}
    ) if looks_by_file else []
    per_look_fields = [
        f"val_L{looks}_{metric}"
        for looks in validation_looks
        for metric in ("loss", "psnr", "ssim")
    ]
    macro_fields = [
        "val_macro_loss", "val_macro_psnr", "val_macro_ssim"
    ] if validation_looks else []
    metric_fields = base_metric_fields + per_look_fields + macro_fields
    with open(metrics_path, "x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=metric_fields)
        writer.writeheader()
    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        epoch_started = time.monotonic()
        total_train_batches = (
            min(len(train_loader), args.max_train_batches)
            if args.max_train_batches else len(train_loader)
        )
        for batch_index, (noisy, clean, _) in enumerate(
            _limited(train_loader, args.max_train_batches), start=1
        ):
            noisy, clean = noisy.to(device), clean.to(device)
            prediction = model(noisy)
            loss = F.mse_loss(prediction, clean) + args.lambda_tv * total_variation(prediction)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))
            if args.progress_interval and (
                batch_index % args.progress_interval == 0
                or batch_index == total_train_batches
            ):
                progress = {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "epoch": epoch,
                    "batch": batch_index,
                    "batches_in_epoch": total_train_batches,
                    "mean_train_loss_so_far": sum(train_losses) / len(train_losses),
                    "elapsed_seconds": time.monotonic() - epoch_started,
                }
                with progress_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(progress, sort_keys=True) + "\n")
                print(json.dumps({"progress": progress}, sort_keys=True), flush=True)
        if not train_losses:
            raise RuntimeError("No training batches were produced")

        model.eval()
        val_losses, val_psnr, val_ssim = [], [], []
        grouped = {
            looks: {"loss": [], "psnr": [], "ssim": []}
            for looks in validation_looks
        }
        with torch.no_grad():
            for noisy, clean, filenames in _limited(val_loader, args.max_val_batches):
                noisy, clean = noisy.to(device), clean.to(device)
                prediction = model(noisy)
                sample_losses = F.mse_loss(
                    prediction, clean, reduction="none"
                ).mean(dim=(1, 2, 3))
                for sample_index, filename in enumerate(filenames):
                    sample_prediction = prediction[sample_index : sample_index + 1]
                    sample_clean = clean[sample_index : sample_index + 1]
                    image_psnr, image_ssim = batch_psnr_ssim(
                        sample_prediction, sample_clean
                    )
                    image_loss = float(sample_losses[sample_index].item())
                    val_losses.append(image_loss)
                    val_psnr.append(image_psnr)
                    val_ssim.append(image_ssim)
                    if looks_by_file:
                        looks = int(looks_by_file[Path(filename).as_posix()])
                        grouped[looks]["loss"].append(image_loss)
                        grouped[looks]["psnr"].append(image_psnr)
                        grouped[looks]["ssim"].append(image_ssim)
        if not val_losses:
            raise RuntimeError("No validation batches were produced")
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": sum(train_losses) / len(train_losses),
            "val_loss": sum(val_losses) / len(val_losses),
            "val_psnr": sum(val_psnr) / len(val_psnr),
            "val_ssim": sum(val_ssim) / len(val_ssim),
        }
        if validation_looks:
            populated_looks = []
            for looks in validation_looks:
                if not grouped[looks]["loss"]:
                    if not args.max_val_batches:
                        raise RuntimeError(f"No validation samples produced for L={looks}")
                    for metric in ("loss", "psnr", "ssim"):
                        row[f"val_L{looks}_{metric}"] = None
                    continue
                populated_looks.append(looks)
                for metric in ("loss", "psnr", "ssim"):
                    values = grouped[looks][metric]
                    row[f"val_L{looks}_{metric}"] = sum(values) / len(values)
            if not populated_looks:
                raise RuntimeError("No global-L validation groups were populated")
            for metric in ("loss", "psnr", "ssim"):
                row[f"val_macro_{metric}"] = sum(
                    row[f"val_L{looks}_{metric}"] for looks in populated_looks
                ) / len(populated_looks)
        selection_loss = row.get("val_macro_loss", row["val_loss"])
        scheduler.step(selection_loss)
        torch.save(checkpoint_payload(model, epoch, run_config, row), output_dir / "last_model.pth")
        if selection_loss < best_loss:
            best_loss = selection_loss
            torch.save(checkpoint_payload(model, epoch, run_config, row), output_dir / "best_model.pth")
        with open(metrics_path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=metric_fields)
            writer.writerow(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
