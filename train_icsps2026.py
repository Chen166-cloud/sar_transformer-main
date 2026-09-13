"""Unified update-based trainer for the frozen ICSPS 2026 SAR protocol."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler

from experiment_protocol import (
    atomic_torch_save,
    atomic_json_dump,
    dataloader_generator,
    load_training_checkpoint_strict,
    prepare_run_directory,
    require_finite_number,
    require_finite_tensor,
    seed_everything,
    seed_worker,
    sha256_file,
    training_checkpoint_payload,
)
from icsps2026_protocol import (
    ICSPSFixedPairDataset,
    ICSPSOnlineGammaDataset,
    SCHEDULE_FIELDS,
    build_train_schedule,
    read_csv_records,
    validate_pair_manifest,
    validate_source_manifest,
    validate_train_schedule,
)
from model_registry import build_model, canonical_model_metadata
from sar_metrics import batch_psnr_ssim


PROTOCOL_ID = "ICSPS26-FROZEN-v2"


class IndexRangeSampler(Sampler[int]):
    """Yield a deterministic contiguous slice without materializing its indices."""

    def __init__(self, start: int, stop: int) -> None:
        self.start = int(start)
        self.stop = int(stop)
        if self.start < 0 or self.stop < self.start:
            raise ValueError(f"Invalid sampler range [{self.start}, {self.stop})")

    def __iter__(self) -> Iterable[int]:
        return iter(range(self.start, self.stop))

    def __len__(self) -> int:
        return self.stop - self.start


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/icsps2026_frozen_v2.json")
    parser.add_argument("--source-root", required=True, help="Original NWPU class-folder root")
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--train-schedule", required=True)
    parser.add_argument("--fixed-pair-root", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--method", required=True, choices=("ours", "transsar_v2", "sar_cam"))
    parser.add_argument("--variant", default=None)
    parser.add_argument("--sar-cam-root")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", help="Resume from checkpoint_last.pth")
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument(
        "--nonformal-smoke",
        action="store_true",
        help="Permit a short diagnostic run; outputs are permanently marked non-formal",
    )
    parser.add_argument("--max-updates", type=int, default=0, help="Only with --nonformal-smoke")
    parser.add_argument("--max-val-pairs", type=int, default=0, help="Only with --nonformal-smoke")
    return parser.parse_args()


def total_variation(batch: torch.Tensor) -> torch.Tensor:
    return (
        torch.abs(batch[:, :, 1:, :] - batch[:, :, :-1, :]).mean()
        + torch.abs(batch[:, :, :, 1:] - batch[:, :, :, :-1]).mean()
    )


def first(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().reshape(-1)[0].item()
    elif isinstance(value, (list, tuple)):
        value = value[0]
    return value


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    config = payload.get("effective_protocol", payload)
    if config.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(
            f"Expected protocol_id={PROTOCOL_ID}, found {config.get('protocol_id')!r}"
        )
    return config


def training_values(config: dict[str, Any]) -> dict[str, Any]:
    section = config.get("training", {})
    schedule = config.get("schedule", {})
    optimizer = section.get("optimizer", {})
    loss = section.get("loss", {})
    scheduler = section.get("scheduler", {})
    if str(loss.get("tv_definition")) != "mean_total_variation":
        raise ValueError("Formal supervised training requires mean_total_variation")
    lambda_tv = float(loss.get("lambda_tv", 0.03))
    if lambda_tv != 0.03:
        raise ValueError(
            "Formal supervised training requires lambda_tv=0.03; "
            "regenerate PREP_ROOT so config_snapshot.json is current"
        )
    return {
        "updates": int(section.get("optimizer_updates", schedule.get("updates", 100_000))),
        "batch_size": int(section.get("batch_size", 1)),
        "val_interval": int(section.get("validation_interval_updates", 5_000)),
        "learning_rate": float(optimizer.get("learning_rate", 1e-3)),
        "weight_decay": float(optimizer.get("weight_decay", 1e-5)),
        "lambda_tv": lambda_tv,
        "scheduler_factor": float(scheduler.get("factor", 0.5)),
        "scheduler_patience": int(scheduler.get("patience_validation_events", 4)),
        "scheduler_min_lr": float(scheduler.get("min_lr", 1e-6)),
    }


def validate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    maximum: int = 0,
) -> dict[str, float | int]:
    grouped: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(
        lambda: {"mse": [], "psnr": [], "ssim": []}
    )
    model.eval()
    pair_count = 0
    with torch.inference_mode():
        for batch in loader:
            if maximum and pair_count >= maximum:
                break
            noisy = batch["noisy"].to(device, non_blocking=True)
            clean = batch["clean"].to(device, non_blocking=True)
            raw_prediction = model(noisy)
            require_finite_tensor(raw_prediction, "validation prediction")
            prediction = torch.clamp(raw_prediction, 0.0, 1.0)
            image_mse = float(F.mse_loss(prediction, clean).item())
            image_psnr, image_ssim = batch_psnr_ssim(prediction, clean, border=0)
            class_name = str(first(batch["class_name"]))
            looks = int(first(batch["L"]))
            grouped[(class_name, looks)]["mse"].append(image_mse)
            grouped[(class_name, looks)]["psnr"].append(image_psnr)
            grouped[(class_name, looks)]["ssim"].append(image_ssim)
            pair_count += 1
    if not grouped:
        raise RuntimeError("Validation loader produced no fixed pairs")
    macro = {
        metric: sum(sum(values[metric]) / len(values[metric]) for values in grouped.values())
        / len(grouped)
        for metric in ("mse", "psnr", "ssim")
    }
    for metric, value in macro.items():
        require_finite_number(value, f"validation macro {metric}")
    return {
        "val_pairs": pair_count,
        "val_class_L_cells": len(grouped),
        "val_macro_mse": float(macro["mse"]),
        "val_macro_psnr": float(macro["psnr"]),
        "val_macro_ssim": float(macro["ssim"]),
    }


def append_csv(path: Path, fieldnames: list[str], row: dict[str, Any]) -> None:
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()


def reconcile_logs_with_checkpoint(
    metrics_path: Path, progress_path: Path, checkpoint_step: int
) -> dict[str, int]:
    """Atomically discard log records written after the resumable checkpoint."""

    with open(metrics_path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if not fieldnames or "global_step" not in fieldnames:
            raise ValueError(f"Invalid resume metrics CSV: {metrics_path}")
        metric_rows = list(reader)
    kept_metrics = [
        row for row in metric_rows if int(row["global_step"]) <= checkpoint_step
    ]
    if not kept_metrics:
        raise ValueError("Resume metrics contain no row at or before checkpoint step")

    kept_progress: list[str] = []
    progress_count = 0
    if progress_path.exists():
        with open(progress_path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                progress_count += 1
                if int(payload["global_step"]) <= checkpoint_step:
                    kept_progress.append(
                        json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
                    )

    def replace_text(path: Path, writer_fn: Any) -> None:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            writer_fn(temporary)
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()

    if len(kept_metrics) != len(metric_rows):
        def write_metrics(temporary: Path) -> None:
            with open(temporary, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(kept_metrics)

        replace_text(metrics_path, write_metrics)
    if progress_path.exists() and len(kept_progress) != progress_count:
        replace_text(
            progress_path,
            lambda temporary: temporary.write_text("".join(kept_progress), encoding="utf-8"),
        )
    return {
        "metrics_rows_removed": len(metric_rows) - len(kept_metrics),
        "progress_rows_removed": progress_count - len(kept_progress),
    }


def main() -> int:
    args = parse_args()
    if args.workers < 0 or args.checkpoint_interval <= 0 or args.progress_interval < 0:
        raise ValueError("Invalid worker/checkpoint/progress interval")
    if not args.nonformal_smoke and (args.max_updates or args.max_val_pairs):
        raise ValueError("Truncation flags require --nonformal-smoke")
    config = load_config(args.config)
    artifact_protocol_id = str(config.get("artifact_protocol_id", PROTOCOL_ID))
    frozen = training_values(config)
    if frozen["batch_size"] != 1:
        raise ValueError("The frozen protocol requires batch_size=1")
    if args.seed not in [int(value) for value in config["schedule"]["run_seeds"]]:
        raise ValueError(f"Seed {args.seed} is not declared in the frozen config")

    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    source_manifest = Path(args.source_manifest).resolve()
    schedule_path = Path(args.train_schedule).resolve()
    val_manifest = Path(args.val_manifest).resolve()
    protocol_config = Path(args.config).resolve()
    protocol_hashes = {
        "protocol_config_sha256": sha256_file(protocol_config),
        "source_manifest_sha256": sha256_file(source_manifest),
        "train_schedule_sha256": sha256_file(schedule_path),
        "validation_manifest_sha256": sha256_file(val_manifest),
    }

    source_rows = read_csv_records(source_manifest)
    schedule_rows = read_csv_records(schedule_path)
    validation_rows = read_csv_records(val_manifest)
    validate_source_manifest(
        source_rows, config, source_root=args.source_root, verify_files=True
    )
    train_source_rows = [row for row in source_rows if row["split"] == "train"]
    validate_train_schedule(
        schedule_rows,
        train_source_rows,
        updates=int(config["schedule"]["updates"]),
        looks=config["looks"],
        run_seed=args.seed,
        synthesis_seed=int(config["seeds"]["synthesis"]),
        protocol_id=artifact_protocol_id,
        d4_count=int(config["schedule"]["d4_count"]),
    )
    expected_schedule = build_train_schedule(
        train_source_rows,
        updates=int(config["schedule"]["updates"]),
        looks=config["looks"],
        run_seed=args.seed,
        synthesis_seed=int(config["seeds"]["synthesis"]),
        protocol_id=artifact_protocol_id,
        d4_count=int(config["schedule"]["d4_count"]),
    )
    expected_schedule_strings = [
        {field: str(row[field]) for field in SCHEDULE_FIELDS}
        for row in expected_schedule
    ]
    if schedule_rows != expected_schedule_strings:
        raise ValueError("Training schedule is valid-looking but not the canonical seeded schedule")
    del expected_schedule, expected_schedule_strings
    validate_pair_manifest(
        validation_rows,
        expected_dataset="NWPU-RESISC45",
        expected_classes=int(config["nwpu"]["expected_classes"]),
        expected_per_class_per_look=int(config["nwpu"]["validation_per_class"]),
        looks=config["looks"],
        output_root=args.fixed_pair_root,
        verify_file_hashes=False,
        synthesis_seed=int(config["seeds"]["validation"]),
    )
    expected_validation_ids = {
        row["source_id"] for row in source_rows if row["split"] == "validation"
    }
    if {row["source_id"] for row in validation_rows} != expected_validation_ids:
        raise ValueError(
            "NWPU fixed validation sources do not match source_manifest validation"
        )
    train_dataset = ICSPSOnlineGammaDataset(
        args.source_root,
        source_rows,
        schedule_rows,
        image_size=tuple(config["image_size"]),
        protocol_id=artifact_protocol_id,
    )
    val_dataset = ICSPSFixedPairDataset(
        args.fixed_pair_root,
        validation_rows,
        protocol_id=artifact_protocol_id,
        verify_hashes=True,
    )
    formal_updates = frozen["updates"]
    target_updates = args.max_updates if args.max_updates else formal_updates
    if args.nonformal_smoke and target_updates > len(train_dataset):
        target_updates = len(train_dataset)
    if not args.nonformal_smoke and len(train_dataset) != formal_updates:
        raise ValueError(
            f"Formal schedule must contain {formal_updates} rows, found {len(train_dataset)}"
        )
    if target_updates <= 0 or target_updates > len(train_dataset):
        raise ValueError(
            f"Requested {target_updates} updates but schedule has {len(train_dataset)} rows"
        )
    expected_val_pairs = 225 * len(config["looks"])
    if not args.nonformal_smoke and len(val_dataset) != expected_val_pairs:
        raise ValueError(
            f"Formal NWPU validation must contain {expected_val_pairs} pairs, found {len(val_dataset)}"
        )

    model_metadata = canonical_model_metadata(args.method, args.variant)
    resume_metadata = (
        torch.load(args.resume, map_location="cpu", weights_only=False)
        if args.resume else None
    )
    model = build_model(
        args.method,
        variant=args.variant,
        external_root=args.sar_cam_root,
        checkpoint_metadata=resume_metadata,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=frozen["learning_rate"],
        weight_decay=frozen["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=frozen["scheduler_factor"],
        patience=frozen["scheduler_patience"],
        min_lr=frozen["scheduler_min_lr"],
    )
    run_config = {
        "protocol_id": PROTOCOL_ID,
        "artifact_protocol_id": artifact_protocol_id,
        "numeric_domain": "intensity_v1",
        "formal_run": not args.nonformal_smoke,
        "method": args.method,
        "variant": model_metadata["variant"],
        "model_metadata": model_metadata,
        "seed": args.seed,
        "target_updates": target_updates,
        "batch_size": 1,
        "validation_interval": frozen["val_interval"],
        "learning_rate": frozen["learning_rate"],
        "weight_decay": frozen["weight_decay"],
        "lambda_tv": frozen["lambda_tv"],
        "scheduler_factor": frozen["scheduler_factor"],
        "scheduler_patience": frozen["scheduler_patience"],
        "scheduler_min_lr": frozen["scheduler_min_lr"],
        "protocol_hashes": protocol_hashes,
        "source_root": str(Path(args.source_root).resolve()),
        "fixed_pair_root": str(Path(args.fixed_pair_root).resolve()),
        "workers": args.workers,
        "tensorboard": args.tensorboard,
    }
    invariant_keys = (
        "protocol_id",
        "artifact_protocol_id",
        "formal_run",
        "method",
        "variant",
        "seed",
        "target_updates",
        "batch_size",
        "validation_interval",
        "learning_rate",
        "weight_decay",
        "lambda_tv",
        "scheduler_factor",
        "scheduler_patience",
        "scheduler_min_lr",
    )
    expected_invariants = {key: run_config[key] for key in invariant_keys}
    output = prepare_run_directory(
        args.output_dir,
        run_config,
        source_files=(
            __file__,
            "experiment_protocol.py",
            "icsps2026_protocol.py",
            "model_registry.py",
            "transform_main.py",
            "ablation_config.py",
            protocol_config,
            source_manifest,
            schedule_path,
            val_manifest,
        ),
        checkpoint_path=args.resume,
        resume=bool(args.resume),
    )
    metrics_path = output / "step_metrics.csv"
    progress_path = output / "train_progress.jsonl"
    metrics_fields = [
        "global_step",
        "train_loss_since_validation",
        "val_pairs",
        "val_class_L_cells",
        "val_macro_mse",
        "val_macro_psnr",
        "val_macro_ssim",
        "learning_rate_before_scheduler",
        "learning_rate_after_scheduler",
        "is_best",
        "formal_run",
        "elapsed_seconds",
    ]

    start_step = 0
    best: dict[str, Any] = {"val_macro_mse": math.inf, "global_step": None}
    last_metrics: dict[str, Any] = {}
    loss_sum_since_validation = 0.0
    loss_count_since_validation = 0
    previous_wall_seconds = 0.0
    synthesis_accumulator: dict[str, dict[str, float | int | None]] = {
        str(int(looks)): {
            "updates": 0,
            "preclip_max_min": None,
            "preclip_max_max": None,
            "postclip_max_min": None,
            "postclip_max_max": None,
            "saturation_rate_sum": 0.0,
            "saturation_rate_max": None,
        }
        for looks in config["looks"]
    }
    if args.resume:
        checkpoint = load_training_checkpoint_strict(
            model,
            optimizer,
            scheduler,
            args.resume,
            expected_protocol_hashes=protocol_hashes,
            expected_run_invariants=expected_invariants,
            map_location=device,
        )
        start_step = int(checkpoint["global_step"])
        best = dict(checkpoint["best"])
        last_metrics = dict(checkpoint.get("metrics", {}))
        accumulator = checkpoint.get("loss_accumulator")
        if not isinstance(accumulator, dict):
            raise ValueError(
                "Resume checkpoint lacks loss_accumulator and cannot reproduce "
                "the interval-level training-loss record exactly"
            )
        loss_sum_since_validation = float(accumulator["sum"])
        loss_count_since_validation = int(accumulator["count"])
        restored_synthesis = checkpoint.get("synthesis_accumulator")
        if not isinstance(restored_synthesis, dict) or set(restored_synthesis) != set(
            synthesis_accumulator
        ):
            raise ValueError(
                "Resume checkpoint lacks the complete synthesis_accumulator and "
                "cannot reproduce the all-update clipping audit"
            )
        synthesis_accumulator = {
            key: dict(value) for key, value in restored_synthesis.items()
        }
        previous_wall_seconds = float(checkpoint.get("wall_seconds_accumulated", 0.0))
        if start_step >= target_updates:
            raise ValueError(
                f"Checkpoint already reached step {start_step}, target is {target_updates}"
            )
        if not metrics_path.exists():
            raise FileNotFoundError(f"Resume metrics file is missing: {metrics_path}")
        reconciliation = reconcile_logs_with_checkpoint(
            metrics_path, progress_path, start_step
        )
        if any(reconciliation.values()):
            print(json.dumps({"resume_log_reconciliation": reconciliation}, sort_keys=True))
    elif metrics_path.exists() or progress_path.exists():
        raise FileExistsError("Fresh run found pre-existing metric files")

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=dataloader_generator(args.seed + 1),
    )

    def save_checkpoint(path: Path, step: int, metrics: dict[str, Any]) -> None:
        payload = training_checkpoint_payload(
            model,
            optimizer,
            scheduler,
            step,
            run_config,
            protocol_hashes,
            best,
            metrics,
        )
        payload["model_metadata"] = model_metadata
        payload["loss_accumulator"] = {
            "sum": loss_sum_since_validation,
            "count": loss_count_since_validation,
        }
        payload["synthesis_accumulator"] = synthesis_accumulator
        payload["wall_seconds_accumulated"] = (
            previous_wall_seconds + time.monotonic() - run_started
        )
        atomic_torch_save(payload, path)

    run_started = time.monotonic()
    tensorboard_writer = None
    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter

        tensorboard_writer = SummaryWriter(
            log_dir=str(output / "tensorboard"),
            purge_step=start_step if args.resume else None,
        )
    if start_step == 0:
        validation = validate_model(
            model, val_loader, device, maximum=args.max_val_pairs
        )
        lr_before = float(optimizer.param_groups[0]["lr"])
        scheduler.step(float(validation["val_macro_mse"]))
        lr_after = float(optimizer.param_groups[0]["lr"])
        best = {"val_macro_mse": validation["val_macro_mse"], "global_step": 0}
        last_metrics = {
            "global_step": 0,
            "train_loss_since_validation": "",
            **validation,
            "learning_rate_before_scheduler": lr_before,
            "learning_rate_after_scheduler": lr_after,
            "is_best": True,
            "formal_run": not args.nonformal_smoke,
            "elapsed_seconds": time.monotonic() - run_started,
        }
        append_csv(metrics_path, metrics_fields, last_metrics)
        if tensorboard_writer is not None:
            for metric in ("val_macro_mse", "val_macro_psnr", "val_macro_ssim"):
                tensorboard_writer.add_scalar(f"validation/{metric[4:]}", validation[metric], 0)
            tensorboard_writer.add_scalar("train/learning_rate", lr_after, 0)
        save_checkpoint(output / "checkpoint_best.pth", 0, last_metrics)
        save_checkpoint(output / "checkpoint_last.pth", 0, last_metrics)

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        sampler=IndexRangeSampler(start_step, target_updates),
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=dataloader_generator(args.seed),
    )
    for batch in train_loader:
        global_step = int(first(batch["global_step"]))
        expected_step = start_step + 1
        if global_step != expected_step:
            raise RuntimeError(
                f"Schedule replay failure: expected global_step={expected_step}, got {global_step}"
            )
        model.train()
        noisy = batch["noisy"].to(device, non_blocking=True)
        clean = batch["clean"].to(device, non_blocking=True)
        prediction = model(noisy)
        loss = F.mse_loss(prediction, clean) + frozen["lambda_tv"] * total_variation(prediction)
        update_context = (
            f"global_step={global_step}, source_id={first(batch['source_id'])}"
        )
        require_finite_tensor(prediction, "training prediction", update_context)
        require_finite_tensor(loss, "training loss", update_context)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        start_step = global_step
        loss_sum_since_validation += float(loss.item())
        loss_count_since_validation += 1

        looks_key = str(int(first(batch["L"])))
        preclip_max = float(first(batch["preclip_max"]))
        postclip_max = float(first(batch["postclip_max"]))
        saturation_rate = float(first(batch["saturation_rate"]))
        for value, label in (
            (preclip_max, "training preclip_max"),
            (postclip_max, "training postclip_max"),
            (saturation_rate, "training saturation_rate"),
        ):
            require_finite_number(value, label)
        audit = synthesis_accumulator[looks_key]
        audit["updates"] = int(audit["updates"]) + 1
        for prefix, value in (("preclip_max", preclip_max), ("postclip_max", postclip_max)):
            minimum_key, maximum_key = prefix + "_min", prefix + "_max"
            current_minimum = audit[minimum_key]
            current_maximum = audit[maximum_key]
            audit[minimum_key] = value if current_minimum is None else min(float(current_minimum), value)
            audit[maximum_key] = value if current_maximum is None else max(float(current_maximum), value)
        audit["saturation_rate_sum"] = float(audit["saturation_rate_sum"]) + saturation_rate
        current_saturation_max = audit["saturation_rate_max"]
        audit["saturation_rate_max"] = (
            saturation_rate
            if current_saturation_max is None
            else max(float(current_saturation_max), saturation_rate)
        )

        if args.progress_interval and global_step % args.progress_interval == 0:
            progress = {
                "global_step": global_step,
                "loss": float(loss.item()),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "source_id": str(first(batch["source_id"])),
                "class_name": str(first(batch["class_name"])),
                "L": int(first(batch["L"])),
                "speckle_seed": int(first(batch["speckle_seed"])),
                "d4_id": int(first(batch["d4_id"])),
                "preclip_max": float(first(batch["preclip_max"])),
                "postclip_max": float(first(batch["postclip_max"])),
                "saturation_rate": float(first(batch["saturation_rate"])),
                "formal_run": not args.nonformal_smoke,
            }
            with open(progress_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(progress, ensure_ascii=False, sort_keys=True) + "\n")
            if tensorboard_writer is not None:
                tensorboard_writer.add_scalar("train/loss", progress["loss"], global_step)
                tensorboard_writer.add_scalar(
                    "train/learning_rate", progress["learning_rate"], global_step
                )
                tensorboard_writer.add_scalar(
                    "data/saturation_rate", progress["saturation_rate"], global_step
                )

        due_validation = global_step % frozen["val_interval"] == 0 or global_step == target_updates
        if due_validation:
            validation = validate_model(
                model, val_loader, device, maximum=args.max_val_pairs
            )
            lr_before = float(optimizer.param_groups[0]["lr"])
            scheduler.step(float(validation["val_macro_mse"]))
            lr_after = float(optimizer.param_groups[0]["lr"])
            candidate = float(validation["val_macro_mse"])
            is_best = candidate < float(best["val_macro_mse"])
            if is_best:
                best = {"val_macro_mse": candidate, "global_step": global_step}
            last_metrics = {
                "global_step": global_step,
                "train_loss_since_validation": (
                    loss_sum_since_validation / loss_count_since_validation
                ),
                **validation,
                "learning_rate_before_scheduler": lr_before,
                "learning_rate_after_scheduler": lr_after,
                "is_best": is_best,
                "formal_run": not args.nonformal_smoke,
                "elapsed_seconds": time.monotonic() - run_started,
            }
            append_csv(metrics_path, metrics_fields, last_metrics)
            if tensorboard_writer is not None:
                for metric in ("val_macro_mse", "val_macro_psnr", "val_macro_ssim"):
                    tensorboard_writer.add_scalar(
                        f"validation/{metric[4:]}", validation[metric], global_step
                    )
                tensorboard_writer.flush()
            loss_sum_since_validation = 0.0
            loss_count_since_validation = 0
            if is_best:
                save_checkpoint(output / "checkpoint_best.pth", global_step, last_metrics)

        if global_step % args.checkpoint_interval == 0 or global_step == target_updates:
            save_checkpoint(output / "checkpoint_last.pth", global_step, last_metrics)

    if start_step != target_updates:
        raise RuntimeError(
            f"Refusing completion: reached {start_step}/{target_updates} updates"
        )
    if int(last_metrics.get("global_step", -1)) != target_updates:
        raise RuntimeError("Refusing completion without final-step validation metrics")
    for metric in ("val_macro_mse", "val_macro_psnr", "val_macro_ssim"):
        require_finite_number(last_metrics.get(metric), f"final {metric}")
    require_finite_number(best.get("val_macro_mse"), "best val_macro_mse")
    if best.get("global_step") is None:
        raise RuntimeError("Refusing completion without a selected finite checkpoint")

    synthesis_summary = {}
    for looks_key, audit in synthesis_accumulator.items():
        count = int(audit["updates"])
        synthesis_summary[looks_key] = {
            "updates": count,
            "preclip_max_min": audit["preclip_max_min"],
            "preclip_max_max": audit["preclip_max_max"],
            "postclip_max_min": audit["postclip_max_min"],
            "postclip_max_max": audit["postclip_max_max"],
            "saturation_rate_mean": (
                float(audit["saturation_rate_sum"]) / count if count else None
            ),
            "saturation_rate_max": audit["saturation_rate_max"],
        }
    if sum(int(value["updates"]) for value in synthesis_summary.values()) != target_updates:
        raise RuntimeError("All-update synthesis audit count does not match completed updates")
    if not args.nonformal_smoke:
        expected_per_look = target_updates // len(synthesis_summary)
        if {int(value["updates"]) for value in synthesis_summary.values()} != {expected_per_look}:
            raise RuntimeError("Formal all-update synthesis audit is not balanced by L")

    wall_seconds_this_invocation = time.monotonic() - run_started
    completion = {
        "protocol_id": PROTOCOL_ID,
        "formal_run": not args.nonformal_smoke,
        "completed_updates": start_step,
        "target_updates": target_updates,
        "best": best,
        "training_synthesis_by_L": synthesis_summary,
        "checkpoint_best_sha256": sha256_file(output / "checkpoint_best.pth"),
        "checkpoint_last_sha256": sha256_file(output / "checkpoint_last.pth"),
        "wall_seconds_this_invocation": wall_seconds_this_invocation,
        "wall_seconds_accumulated": previous_wall_seconds + wall_seconds_this_invocation,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json_dump(completion, output / "completion.json")
    if tensorboard_writer is not None:
        tensorboard_writer.close()
    print(json.dumps(completion, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
