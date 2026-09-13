"""Encoder-frozen masked post-adaptation for ICSPS26-FROZEN-v2."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from experiment_protocol import (
    atomic_torch_save,
    atomic_json_dump,
    canonical_json_sha256,
    capture_rng_state,
    dataloader_generator,
    load_checkpoint_strict,
    prepare_run_directory,
    require_finite_number,
    require_finite_tensor,
    restore_rng_state,
    seed_everything,
    seed_worker,
    sha256_file,
    validate_completed_checkpoint,
)
from model_registry import build_model, canonical_model_metadata
from prepare_icsps2026_real import apply_fixed_mapping, load_real_array


PROTOCOL_ID = "ICSPS26-FROZEN-v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/icsps2026_frozen_v2.json")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--real-qc-manifest", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume")
    parser.add_argument("--nonformal-smoke", action="store_true")
    parser.add_argument("--max-train-images", type=int, default=0)
    parser.add_argument("--max-val-images", type=int, default=0)
    return parser.parse_args()


def stable_seed(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def bool_value(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


class RealQCNoisyDataset(Dataset):
    def __init__(self, dataset_root: Path, rows: list[dict[str, str]], mapping: dict[str, Any]):
        self.dataset_root = dataset_root.resolve()
        self.rows = list(rows)
        self.mapping = dict(mapping)
        sample_ids = [row["sample_id"] for row in self.rows]
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("Real QC rows contain duplicate sample_id values")
        for row in self.rows:
            path = (self.dataset_root / row["noisy_path"]).resolve()
            try:
                path.relative_to(self.dataset_root)
            except ValueError as error:
                raise ValueError(
                    f"Real QC path escapes dataset root: {row['noisy_path']}"
                ) from error
            if sha256_file(path) != row["noisy_sha256"]:
                raise AssertionError(f"Real noisy source SHA-256 mismatch: {path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        path = (self.dataset_root / row["noisy_path"]).resolve()
        raw, _ = load_real_array(path, "noisy")
        image = apply_fixed_mapping(raw, self.mapping)
        return {
            "noisy": torch.from_numpy(image[None]).float(),
            "sample_id": row["sample_id"],
            "parent_id": row["parent_id"],
        }


def make_mask(shape: tuple[int, ...], ratio: float, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return (torch.rand(shape, generator=generator) < ratio).to(torch.float32)


def masked_l1(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (torch.abs(prediction - target) * mask).sum() / mask.sum().clamp_min(1.0)


def total_variation(batch: torch.Tensor) -> torch.Tensor:
    return (
        torch.abs(batch[:, :, 1:, :] - batch[:, :, :-1, :]).mean()
        + torch.abs(batch[:, :, :, 1:] - batch[:, :, :, :-1]).mean()
    )


def first(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().reshape(-1)[0].item()
    if isinstance(value, (list, tuple)):
        return value[0]
    return value


def reconcile_epoch_metrics(path: Path, checkpoint_epoch: int) -> int:
    """Remove epoch rows that were flushed before the matching checkpoint."""

    with open(path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if not fieldnames or "epoch" not in fieldnames:
            raise ValueError(f"Invalid AMS resume metrics CSV: {path}")
        rows = list(reader)
    kept = [row for row in rows if int(row["epoch"]) <= checkpoint_epoch]
    removed = len(rows) - len(kept)
    if removed:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            with open(temporary, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(kept)
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
    return removed


def require_selected_ams_best(best: dict[str, Any]) -> None:
    if best.get("epoch") is None:
        raise RuntimeError(
            "No adapted checkpoint was selected by fixed real-validation masked loss; "
            "completion.json was not created"
        )
    require_finite_number(best.get("fixed_val_masked_loss"), "AMS best validation loss")


def fixed_mask_validation(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    mask_ratio: float,
    lambda_tv: float,
    mask_seeds: dict[str, int],
    maximum: int,
) -> tuple[float, int]:
    model.eval()
    losses = []
    with torch.inference_mode():
        for batch in loader:
            if maximum and len(losses) >= maximum:
                break
            sample_id = str(first(batch["sample_id"]))
            noisy = batch["noisy"].to(device, non_blocking=True)
            mask = make_mask(tuple(noisy.shape), mask_ratio, mask_seeds[sample_id]).to(device)
            prediction = model(noisy * (1.0 - mask))
            loss = masked_l1(prediction, noisy, mask) + lambda_tv * total_variation(prediction)
            require_finite_tensor(prediction, "AMS fixed-mask validation prediction")
            require_finite_tensor(loss, "AMS fixed-mask validation loss")
            losses.append(float(loss.item()))
    if not losses:
        raise RuntimeError("Real validation produced no samples")
    return float(np.mean(losses)), len(losses)


def main() -> int:
    args = parse_args()
    if (
        args.workers < 0
        or args.max_train_images < 0
        or args.max_val_images < 0
    ):
        raise ValueError("workers and max-* values must be non-negative")
    truncation = args.max_train_images or args.max_val_images
    if truncation and not args.nonformal_smoke:
        raise ValueError("All max-* flags require --nonformal-smoke")
    with open(args.config, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    config = payload.get("effective_protocol", payload)
    if config.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(f"Expected {PROTOCOL_ID}, found {config.get('protocol_id')!r}")
    supervised_loss = config.get("training", {}).get("loss", {})
    if (
        str(supervised_loss.get("tv_definition")) != "mean_total_variation"
        or float(supervised_loss.get("lambda_tv", float("nan"))) != 0.03
    ):
        raise ValueError(
            "AMS requires the revised supervised lambda_tv=0.03; "
            "regenerate PREP_ROOT so config_snapshot.json is current"
        )
    ams = config.get("ams", {})
    ams_optimizer = ams.get("optimizer", {})
    ams_mask = ams.get("mask", {})
    ams_loss = ams.get("loss", {})
    artifact_protocol_id = str(config.get("artifact_protocol_id", PROTOCOL_ID))
    epochs = int(ams.get("epochs", 8))
    learning_rate = float(ams_optimizer.get("learning_rate", 1e-6))
    weight_decay = float(ams_optimizer.get("weight_decay", 1e-5))
    mask_ratio = float(ams_mask.get("ratio", 0.2))
    lambda_tv = float(ams_loss.get("lambda_tv", 1e-3))

    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    dataset_root = Path(args.dataset_root).resolve()
    qc_manifest_path = Path(args.real_qc_manifest).resolve()
    with open(qc_manifest_path, "r", encoding="utf-8") as handle:
        qc_manifest = json.load(handle)
    qc_csv_path = (qc_manifest_path.parent / qc_manifest["artifacts"]["qc_csv"]).resolve()
    with open(qc_csv_path, "r", newline="", encoding="utf-8") as handle:
        all_rows = list(csv.DictReader(handle))
    train_rows = [row for row in all_rows if row["split"] == "train" and bool_value(row["valid_no_reference"])]
    val_rows = [row for row in all_rows if row["split"] == "val" and bool_value(row["valid_no_reference"])]
    if not args.nonformal_smoke and (len(train_rows), len(val_rows)) != (4700, 584):
        raise ValueError(
            "Formal AMS requires 4700 valid train and 584 valid validation patches; "
            f"found {len(train_rows)}/{len(val_rows)}"
        )
    if args.max_train_images:
        train_rows = train_rows[: args.max_train_images]
    if args.max_val_images:
        val_rows = val_rows[: args.max_val_images]
    mask_shape = (
        1,
        1,
        int(ams.get("crop_height", 256)),
        int(ams.get("crop_width", 256)),
    )
    mask_rows = []
    mask_seeds: dict[str, int] = {}
    for row in val_rows:
        sample_id = row["sample_id"]
        mask_seed = stable_seed(PROTOCOL_ID, "ams-validation-mask", args.seed, sample_id)
        mask = make_mask(mask_shape, mask_ratio, mask_seed)
        mask_hash = hashlib.sha256(mask.numpy().tobytes()).hexdigest()
        mask_seeds[sample_id] = mask_seed
        mask_rows.append(
            {
                "sample_id": sample_id,
                "parent_id": row["parent_id"],
                "mask_seed": mask_seed,
                "mask_sha256": mask_hash,
            }
        )
    real_train = RealQCNoisyDataset(dataset_root, train_rows, qc_manifest["numeric_mapping"])
    real_val = RealQCNoisyDataset(dataset_root, val_rows, qc_manifest["numeric_mapping"])

    base_payload = torch.load(
        args.base_checkpoint, map_location="cpu", weights_only=False
    )
    if not isinstance(base_payload, dict):
        raise ValueError("Base checkpoint must be a metadata-bearing mapping")
    if not args.nonformal_smoke:
        validate_completed_checkpoint(args.base_checkpoint, PROTOCOL_ID)
        base_run = base_payload.get("run_config")
        if not isinstance(base_run, dict):
            raise ValueError("Formal AMS requires base checkpoint run_config metadata")
        expected_base = {
            "protocol_id": PROTOCOL_ID,
            "artifact_protocol_id": artifact_protocol_id,
            "formal_run": True,
            "numeric_domain": "intensity_v1",
            "method": "ours",
            "variant": "full",
            "seed": args.seed,
            "target_updates": int(config["training"]["optimizer_updates"]),
        }
        mismatches = {
            key: {"expected": value, "found": base_run.get(key)}
            for key, value in expected_base.items()
            if base_run.get(key) != value
        }
        if base_payload.get("checkpoint_format") != "icsps26-resumable-v1":
            mismatches["checkpoint_format"] = {
                "expected": "icsps26-resumable-v1",
                "found": base_payload.get("checkpoint_format"),
            }
        if mismatches:
            raise ValueError(f"Base checkpoint is not a compatible formal run: {mismatches}")
    model = build_model(
        "ours", variant="full", checkpoint_metadata=base_payload
    )
    load_checkpoint_strict(model, args.base_checkpoint, map_location="cpu")
    if not hasattr(model, "log_branch") or not hasattr(model.log_branch, "Tenc"):
        raise TypeError("AMS requires the Ours Full model with log_branch.Tenc")
    for parameter in model.log_branch.Tenc.parameters():
        parameter.requires_grad = False
    model.to(device)
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    hashes = {
        "protocol_config_sha256": sha256_file(args.config),
        "real_qc_manifest_sha256": sha256_file(qc_manifest_path),
        "real_qc_csv_sha256": sha256_file(qc_csv_path),
        "base_checkpoint_sha256": sha256_file(args.base_checkpoint),
        "validation_masks_sha256": canonical_json_sha256(mask_rows),
    }
    run_config = {
        "protocol_id": PROTOCOL_ID,
        "artifact_protocol_id": artifact_protocol_id,
        "numeric_domain": "intensity_v1",
        "formal_run": not args.nonformal_smoke,
        "method": "ours",
        "variant": "full",
        "adaptation": "ams",
        "model_metadata": canonical_model_metadata("ours", "full"),
        "seed": args.seed,
        "epochs": epochs,
        "batch_size": 1,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "mask_ratio": mask_ratio,
        "lambda_tv": lambda_tv,
        "frozen_module": "log_branch.Tenc",
        "selection": "lowest fixed-mask real-validation loss over adapted epochs 1-8",
        "protocol_hashes": hashes,
    }
    output = prepare_run_directory(
        args.output_dir,
        run_config,
        source_files=(
            __file__,
            "experiment_protocol.py",
            "icsps2026_protocol.py",
            "prepare_icsps2026_real.py",
            "model_registry.py",
            args.config,
            qc_manifest_path,
            qc_csv_path,
        ),
        checkpoint_path=args.resume or args.base_checkpoint,
        resume=bool(args.resume),
    )

    masks_path = output / "ams_validation_masks.csv"
    if not args.resume:
        with open(masks_path, "x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(mask_rows[0]))
            writer.writeheader()
            writer.writerows(mask_rows)
    elif not masks_path.exists():
        raise FileNotFoundError(f"Resume mask manifest is missing: {masks_path}")
    else:
        with open(masks_path, "r", newline="", encoding="utf-8") as handle:
            actual_mask_rows = list(csv.DictReader(handle))
        expected_mask_rows = [
            {key: str(value) for key, value in row.items()} for row in mask_rows
        ]
        if actual_mask_rows != expected_mask_rows:
            raise ValueError("Resume mask manifest differs from the frozen validation masks")

    real_val_loader = DataLoader(
        real_val, batch_size=1, shuffle=False, num_workers=args.workers,
        pin_memory=device.type == "cuda", worker_init_fn=seed_worker,
        generator=dataloader_generator(stable_seed(PROTOCOL_ID, "ams-real-val", args.seed)),
    )
    metrics_path = output / "epoch_metrics.csv"
    fields = [
        "epoch", "train_masked_loss", "fixed_val_masked_loss", "val_images",
        "is_best", "learning_rate", "formal_run", "elapsed_seconds",
    ]
    start_epoch = 0
    best = {"epoch": None, "fixed_val_masked_loss": math.inf}
    previous_wall_seconds = 0.0
    if args.resume:
        payload = torch.load(args.resume, map_location=device, weights_only=False)
        if payload.get("checkpoint_format") != "icsps26-ams-v1":
            raise ValueError("--resume requires an icsps26-ams-v1 checkpoint")
        if payload.get("protocol_hashes") != hashes or payload.get("run_config") != run_config:
            raise ValueError("AMS resume protocol/config mismatch")
        model.load_state_dict(payload["state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        restore_rng_state(payload["rng_state"])
        start_epoch = int(payload["epoch"])
        best = dict(payload["best"])
        previous_wall_seconds = float(payload.get("wall_seconds_accumulated", 0.0))
        if not metrics_path.exists():
            raise FileNotFoundError(f"Resume metrics file is missing: {metrics_path}")
        removed_rows = reconcile_epoch_metrics(metrics_path, start_epoch)
        if removed_rows:
            print(json.dumps({"resume_epoch_rows_removed": removed_rows}, sort_keys=True))
    else:
        baseline_val_loss, val_count = fixed_mask_validation(
            model, real_val_loader, device, mask_ratio, lambda_tv, mask_seeds, args.max_val_images
        )
        baseline_row = {
            "epoch": 0,
            "train_masked_loss": "",
            "fixed_val_masked_loss": baseline_val_loss,
            "val_images": val_count,
            "is_best": False,
            "learning_rate": learning_rate,
            "formal_run": not args.nonformal_smoke,
            "elapsed_seconds": 0.0,
        }
        with open(metrics_path, "x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerow(baseline_row)

    def save(path: Path, epoch: int, metrics: dict[str, Any]) -> None:
        payload = {
            "checkpoint_format": "icsps26-ams-v1",
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "rng_state": capture_rng_state(),
            "best": best,
            "metrics": metrics,
            "run_config": run_config,
            "protocol_hashes": hashes,
            "model_metadata": canonical_model_metadata("ours", "full"),
            "wall_seconds_accumulated": previous_wall_seconds + time.monotonic() - started,
        }
        atomic_torch_save(payload, path)

    started = time.monotonic()
    for epoch in range(start_epoch + 1, epochs + 1):
        order = np.random.default_rng(stable_seed(PROTOCOL_ID, "ams-order", args.seed, epoch)).permutation(len(real_train))
        train_loader = DataLoader(
            Subset(real_train, order.tolist()), batch_size=1, shuffle=False,
            num_workers=args.workers, pin_memory=device.type == "cuda", worker_init_fn=seed_worker,
            generator=dataloader_generator(
                stable_seed(PROTOCOL_ID, "ams-train-loader", args.seed, epoch)
            ),
        )
        model.train()
        model.log_branch.Tenc.eval()
        losses = []
        for batch in train_loader:
            sample_id = str(first(batch["sample_id"]))
            noisy = batch["noisy"].to(device, non_blocking=True)
            mask_seed = stable_seed(PROTOCOL_ID, "ams-train-mask", args.seed, epoch, sample_id)
            mask = make_mask(tuple(noisy.shape), mask_ratio, mask_seed).to(device)
            prediction = model(noisy * (1.0 - mask))
            loss = masked_l1(prediction, noisy, mask) + lambda_tv * total_variation(prediction)
            update_context = f"epoch={epoch}, sample_id={sample_id}"
            require_finite_tensor(prediction, "AMS training prediction", update_context)
            require_finite_tensor(loss, "AMS training loss", update_context)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        fixed_val_loss, val_count = fixed_mask_validation(
            model, real_val_loader, device, mask_ratio, lambda_tv, mask_seeds, args.max_val_images
        )
        is_best = fixed_val_loss < float(best["fixed_val_masked_loss"])
        if is_best:
            best = {"epoch": epoch, "fixed_val_masked_loss": fixed_val_loss}
        row = {
            "epoch": epoch,
            "train_masked_loss": float(np.mean(losses)),
            "fixed_val_masked_loss": fixed_val_loss,
            "val_images": val_count,
            "is_best": is_best,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "formal_run": not args.nonformal_smoke,
            "elapsed_seconds": time.monotonic() - started,
        }
        with open(metrics_path, "a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=fields).writerow(row)
        if is_best:
            save(output / "checkpoint_best.pth", epoch, row)
        save(output / "checkpoint_last.pth", epoch, row)

    require_selected_ams_best(best)
    wall_seconds_this_invocation = time.monotonic() - started
    completion = {
        "protocol_id": PROTOCOL_ID,
        "formal_run": not args.nonformal_smoke,
        "completed_epochs": epochs,
        "best": best,
        "checkpoint_best_sha256": sha256_file(output / "checkpoint_best.pth"),
        "checkpoint_last_sha256": sha256_file(output / "checkpoint_last.pth"),
        "wall_seconds_this_invocation": wall_seconds_this_invocation,
        "wall_seconds_accumulated": previous_wall_seconds + wall_seconds_this_invocation,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json_dump(completion, output / "completion.json")
    print(json.dumps(completion, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
