"""Canonical masked real-SAR adaptation with leakage and forgetting guards."""

from __future__ import annotations

import argparse
import csv
import json
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
)
from numeric_domain import INTENSITY_DOMAIN
from synthetic_manifest import verify_paired_sar_manifest
from sar_metrics import batch_psnr_ssim
from transform_main import TransSARV2_DualFreqNG_Bottle
from utils import BSD_SAR, RealSARDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-dataset-root", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--synthetic-val-dataset")
    parser.add_argument("--synthetic-dataset-root")
    parser.add_argument("--synthetic-split-manifest")
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--crop", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--mask-ratio", type=float, default=0.2)
    parser.add_argument("--lambda-tv", type=float, default=0.001)
    parser.add_argument("--max-synthetic-psnr-drop", type=float, default=0.5)
    parser.add_argument("--finetune-scope", choices=("all", "decoder"), default="decoder")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--verify-data-hashes", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--max-synthetic-batches", type=int, default=0)
    return parser.parse_args()


def _use_synthetic_manifest(args: argparse.Namespace) -> bool:
    manifest_mode = bool(args.synthetic_dataset_root or args.synthetic_split_manifest)
    if manifest_mode and args.synthetic_val_dataset:
        raise ValueError(
            "Choose either the synthetic manifest pair or --synthetic-val-dataset, not both"
        )
    if manifest_mode:
        if not args.synthetic_dataset_root or not args.synthetic_split_manifest:
            raise ValueError(
                "--synthetic-dataset-root and --synthetic-split-manifest must be supplied together"
            )
        return True
    if not args.synthetic_val_dataset:
        raise ValueError(
            "Supply the official synthetic manifest pair; --synthetic-val-dataset is retained "
            "only for historical compatibility"
        )
    return False


def _limited(loader, maximum):
    for index, batch in enumerate(loader):
        if maximum and index >= maximum:
            break
        yield batch


def _mask(x: torch.Tensor, ratio: float, generator: torch.Generator):
    mask = (torch.rand(x.shape, generator=generator) < ratio).to(x.device, x.dtype)
    return x * (1.0 - mask), mask


def _masked_l1(prediction, target, mask):
    return (torch.abs(prediction - target) * mask).sum() / mask.sum().clamp_min(1.0)


def _tv(batch):
    return (
        torch.abs(batch[:, :, 1:, :] - batch[:, :, :-1, :]).mean()
        + torch.abs(batch[:, :, :, 1:] - batch[:, :, :, :-1]).mean()
    )


def _synthetic_eval(model, loader, device, maximum):
    model.eval()
    losses, psnr_values, ssim_values = [], [], []
    with torch.no_grad():
        for noisy, clean, _ in _limited(loader, maximum):
            noisy, clean = noisy.to(device), clean.to(device)
            prediction = model(noisy)
            losses.append(float(F.mse_loss(prediction, clean).item()))
            image_psnr, image_ssim = batch_psnr_ssim(prediction, clean)
            psnr_values.append(image_psnr)
            ssim_values.append(image_ssim)
    if not losses:
        raise RuntimeError("No synthetic validation batches were produced")
    return {
        "synthetic_val_loss": sum(losses) / len(losses),
        "synthetic_val_psnr": sum(psnr_values) / len(psnr_values),
        "synthetic_val_ssim": sum(ssim_values) / len(ssim_values),
    }


def main() -> int:
    args = parse_args()
    if not 0.0 < args.mask_ratio < 1.0:
        raise ValueError("mask-ratio must be between 0 and 1")
    synthetic_manifest_mode = _use_synthetic_manifest(args)
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    run_config = vars(args).copy()
    run_config.update(
        {
            "model": "TransSARV2_DualFreqNG_Bottle",
            "numeric_domain": INTENSITY_DOMAIN,
            "synthetic_data_protocol": (
                "paired_sar_manifest" if synthetic_manifest_mode else "legacy_directory"
            ),
        }
    )
    source_files = [
        __file__, "transform_main.py", "numeric_domain.py", "sar_metrics.py",
        "utils.py", args.split_manifest,
    ]
    if synthetic_manifest_mode:
        with open(args.synthetic_split_manifest, "r", encoding="utf-8") as handle:
            synthetic_manifest = json.load(handle)
        verify_paired_sar_manifest(
            Path(args.synthetic_dataset_root),
            synthetic_manifest,
            verify_hashes=args.verify_data_hashes,
            require_exact_partition=args.verify_data_hashes,
            verify_files=args.verify_data_hashes,
        )
        source_files.extend(
            ["synthetic_manifest.py", args.synthetic_split_manifest]
        )
    output_dir = prepare_run_directory(
        args.output_dir,
        run_config,
        source_files=source_files,
        checkpoint_path=args.base_checkpoint,
    )

    train_set = RealSARDataset(args.real_dataset_root, args.split_manifest, "train")
    val_set = RealSARDataset(args.real_dataset_root, args.split_manifest, "val")
    if synthetic_manifest_mode:
        synthetic_set = BSD_SAR(
            args.synthetic_dataset_root,
            (args.crop, args.crop),
            False,
            INTENSITY_DOMAIN,
            args.synthetic_split_manifest,
            "val",
        )
    else:
        synthetic_set = BSD_SAR(
            args.synthetic_val_dataset, (args.crop, args.crop), False, INTENSITY_DOMAIN
        )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        worker_init_fn=seed_worker,
        generator=dataloader_generator(args.seed),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        worker_init_fn=seed_worker,
        generator=dataloader_generator(args.seed + 1),
    )
    synthetic_loader = DataLoader(
        synthetic_set,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        worker_init_fn=seed_worker,
        generator=dataloader_generator(args.seed + 2),
    )

    model = TransSARV2_DualFreqNG_Bottle(
        ablation="full", numeric_domain=INTENSITY_DOMAIN
    ).to(device)
    load_checkpoint_strict(model, args.base_checkpoint, map_location=device)
    if args.finetune_scope == "decoder":
        for parameter in model.log_branch.Tenc.parameters():
            parameter.requires_grad = False
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    baseline = _synthetic_eval(model, synthetic_loader, device, args.max_synthetic_batches)
    print(json.dumps({"epoch": 0, **baseline}, sort_keys=True))
    metrics_path = output_dir / "epoch_metrics.csv"
    metric_fields = [
        "epoch",
        "train_selfsup_loss",
        "val_selfsup_loss",
        "synthetic_val_loss",
        "synthetic_val_psnr",
        "synthetic_val_ssim",
        "synthetic_psnr_drop",
        "eligible_for_best",
    ]
    with open(metrics_path, "x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=metric_fields)
        writer.writeheader()
    best_real_loss = float("inf")
    train_mask_generator = dataloader_generator(args.seed + 10)
    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.finetune_scope == "decoder":
            model.log_branch.Tenc.eval()
        train_losses = []
        for noisy in _limited(train_loader, args.max_train_batches):
            noisy = noisy.to(device)
            masked, mask = _mask(noisy, args.mask_ratio, train_mask_generator)
            prediction = model(masked)
            loss = _masked_l1(prediction, noisy, mask) + args.lambda_tv * _tv(prediction)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))
        if not train_losses:
            raise RuntimeError("No real-SAR training batches were produced")

        model.eval()
        val_losses = []
        val_mask_generator = dataloader_generator(args.seed + 1000 + epoch)
        with torch.no_grad():
            for noisy in _limited(val_loader, args.max_val_batches):
                noisy = noisy.to(device)
                masked, mask = _mask(noisy, args.mask_ratio, val_mask_generator)
                prediction = model(masked)
                val_losses.append(
                    float((_masked_l1(prediction, noisy, mask) + args.lambda_tv * _tv(prediction)).item())
                )
        if not val_losses:
            raise RuntimeError("No real-SAR validation batches were produced")
        synthetic = _synthetic_eval(model, synthetic_loader, device, args.max_synthetic_batches)
        row = {
            "epoch": epoch,
            "train_selfsup_loss": sum(train_losses) / len(train_losses),
            "val_selfsup_loss": sum(val_losses) / len(val_losses),
            **synthetic,
        }
        row["synthetic_psnr_drop"] = baseline["synthetic_val_psnr"] - row["synthetic_val_psnr"]
        row["eligible_for_best"] = row["synthetic_psnr_drop"] <= args.max_synthetic_psnr_drop
        torch.save(checkpoint_payload(model, epoch, run_config, row), output_dir / "last_model.pth")
        if row["eligible_for_best"] and row["val_selfsup_loss"] < best_real_loss:
            best_real_loss = row["val_selfsup_loss"]
            torch.save(checkpoint_payload(model, epoch, run_config, row), output_dir / "best_model.pth")
        with open(metrics_path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=metric_fields)
            writer.writerow(row)
        print(json.dumps(row, sort_keys=True))

    if best_real_loss == float("inf"):
        raise RuntimeError(
            "No checkpoint passed the synthetic-domain retention threshold; "
            "the run is intentionally rejected."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
