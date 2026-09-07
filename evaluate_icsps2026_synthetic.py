"""Evaluate learned methods or the noisy input on a frozen synthetic manifest."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from experiment_protocol import (
    atomic_json_dump,
    dataloader_generator,
    load_checkpoint_strict,
    prepare_run_directory,
    seed_everything,
    seed_worker,
    sha256_file,
    validate_completed_checkpoint,
)
from icsps2026_protocol import (
    ICSPSFixedPairDataset,
    read_csv_records,
    validate_pair_manifest,
)
from model_registry import build_model
from numeric_domain import INTENSITY_DOMAIN
from sar_metrics import psnr, ssim
from verify_icsps2026_pretest_gate import verify as verify_pretest_gate


PROTOCOL_ID = "ICSPS26-FROZEN-v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/icsps2026_frozen_v2.json")
    parser.add_argument("--dataset", required=True, choices=("nwpu_validation", "ucm_test"))
    parser.add_argument("--fixed-pair-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--method", required=True, choices=("noisy", "ours", "transsar_v2", "sar_cam"))
    parser.add_argument("--variant", default=None)
    parser.add_argument("--checkpoint")
    parser.add_argument("--sar-cam-root")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--run-root",
        help="Required pretest-registration root for every UCM test evaluation",
    )
    parser.add_argument("--nonformal-smoke", action="store_true")
    parser.add_argument("--max-pairs", type=int, default=0, help="Only with --nonformal-smoke")
    return parser.parse_args()


def first(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().reshape(-1)[0].item()
    if isinstance(value, (list, tuple)):
        return value[0]
    return value


def main() -> int:
    args = parse_args()
    pretest_gate = None
    if args.dataset == "ucm_test":
        if not args.run_root:
            raise ValueError("Every UCM evaluation requires --run-root for the pretest gate")
        pretest_gate = verify_pretest_gate(args.run_root)
    if args.workers < 0 or args.max_pairs < 0:
        raise ValueError("workers and max-pairs must be non-negative")
    if args.method == "noisy" and args.checkpoint:
        raise ValueError("The noisy-input baseline does not accept --checkpoint")
    if args.method == "noisy" and (args.variant is not None or args.sar_cam_root is not None):
        raise ValueError("The noisy-input baseline accepts neither variant nor SAR-CAM options")
    if args.method != "noisy" and not args.checkpoint:
        raise ValueError(f"{args.method} requires --checkpoint")
    if not args.nonformal_smoke and args.max_pairs:
        raise ValueError("--max-pairs requires --nonformal-smoke")
    with open(args.config, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    config = payload.get("effective_protocol", payload)
    if config.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(f"Expected {PROTOCOL_ID}, found {config.get('protocol_id')!r}")
    seed_everything(args.seed)
    artifact_protocol_id = str(config.get("artifact_protocol_id", PROTOCOL_ID))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    manifest_rows = read_csv_records(args.manifest)
    if pretest_gate is not None and sha256_file(args.manifest) != pretest_gate[
        "ucm_test_manifest_sha256"
    ]:
        raise ValueError("UCM manifest does not match the pretest-seal binding")
    if args.dataset == "nwpu_validation":
        expected_dataset = "NWPU-RESISC45"
        expected_classes = int(config["nwpu"]["expected_classes"])
        expected_per_class = int(config["nwpu"]["validation_per_class"])
        synthesis_seed = int(config["seeds"]["validation"])
    else:
        expected_dataset = "UCMerced_LandUse"
        expected_classes = int(config["ucm"]["expected_classes"])
        expected_per_class = int(config["ucm"]["expected_per_class"])
        synthesis_seed = int(config["seeds"]["ucm_test"])
    validate_pair_manifest(
        manifest_rows,
        expected_dataset=expected_dataset,
        expected_classes=expected_classes,
        expected_per_class_per_look=expected_per_class,
        looks=config["looks"],
        output_root=args.fixed_pair_root,
        verify_file_hashes=False,
        synthesis_seed=synthesis_seed,
    )
    dataset = ICSPSFixedPairDataset(
        args.fixed_pair_root,
        manifest_rows,
        protocol_id=artifact_protocol_id,
        verify_hashes=True,
    )
    expected_pairs = 900 if args.dataset == "nwpu_validation" else 8400
    if not args.nonformal_smoke and len(dataset) != expected_pairs:
        raise ValueError(
            f"Formal {args.dataset} manifest must contain {expected_pairs} pairs, found {len(dataset)}"
        )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=dataloader_generator(args.seed),
    )

    model = None
    checkpoint_metadata = None
    if args.checkpoint:
        checkpoint_metadata = torch.load(
            args.checkpoint, map_location="cpu", weights_only=False
        )
        if not isinstance(checkpoint_metadata, dict):
            raise ValueError("Checkpoint must be a metadata-bearing mapping")
        checkpoint_run = checkpoint_metadata.get("run_config", {})
        if not isinstance(checkpoint_run, dict):
            raise ValueError("Checkpoint run_config must be a mapping")
        if not args.nonformal_smoke:
            validate_completed_checkpoint(args.checkpoint, PROTOCOL_ID)
            expected_checkpoint = {
                "protocol_id": PROTOCOL_ID,
                "artifact_protocol_id": artifact_protocol_id,
                "formal_run": True,
                "numeric_domain": INTENSITY_DOMAIN,
                "seed": args.seed,
                "target_updates": int(config["training"]["optimizer_updates"]),
            }
            mismatches = {
                key: {"expected": value, "found": checkpoint_run.get(key)}
                for key, value in expected_checkpoint.items()
                if checkpoint_run.get(key) != value
            }
            if checkpoint_metadata.get("checkpoint_format") != "icsps26-resumable-v1":
                mismatches["checkpoint_format"] = {
                    "expected": "icsps26-resumable-v1",
                    "found": checkpoint_metadata.get("checkpoint_format"),
                }
            if checkpoint_run.get("adaptation") is not None:
                mismatches["adaptation"] = {
                    "expected": None,
                    "found": checkpoint_run.get("adaptation"),
                }
            if mismatches:
                raise ValueError(f"Checkpoint is not a compatible formal supervised run: {mismatches}")
        model = build_model(
            args.method,
            variant=args.variant,
            external_root=args.sar_cam_root,
            checkpoint_metadata=checkpoint_metadata,
        )
        load_checkpoint_strict(model, args.checkpoint, map_location="cpu")
        model.to(device).eval()
    canonical_variant = (
        model.registry_metadata["variant"] if model is not None else None
    )

    run_config = {
        "protocol_id": PROTOCOL_ID,
        "artifact_protocol_id": artifact_protocol_id,
        "formal_run": not args.nonformal_smoke,
        "dataset": args.dataset,
        "method": args.method,
        "variant": canonical_variant,
        "seed": args.seed,
        "fixed_pair_root": str(Path(args.fixed_pair_root).resolve()),
        "manifest": str(Path(args.manifest).resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "metric_domain": "post-clip linear normalized intensity [0,1]",
        "data_range": 1.0,
        "border": 0,
        "pretest_gate": pretest_gate,
    }
    output = prepare_run_directory(
        args.output_dir,
        run_config,
        source_files=(
            __file__,
            "icsps2026_protocol.py",
            "sar_metrics.py",
            "model_registry.py",
            args.config,
            args.manifest,
        ),
        checkpoint_path=args.checkpoint,
    )

    rows: list[dict[str, Any]] = []
    for index, batch in enumerate(loader):
        if args.max_pairs and index >= args.max_pairs:
            break
        noisy_tensor = batch["noisy"].to(device, non_blocking=True)
        clean_tensor = batch["clean"].to(device, non_blocking=True)
        if model is None:
            prediction_tensor = noisy_tensor
        else:
            with torch.inference_mode():
                prediction_tensor = model(noisy_tensor)
        prediction = torch.clamp(prediction_tensor, 0.0, 1.0).detach().cpu().numpy()[0, 0]
        clean = clean_tensor.detach().cpu().numpy()[0, 0]
        pair_id = str(first(batch["pair_id"]))
        row = {
            "protocol_id": PROTOCOL_ID,
            "artifact_protocol_id": artifact_protocol_id,
            "formal_run": not args.nonformal_smoke,
            "dataset": args.dataset,
            "method": args.method,
            "variant": canonical_variant or "",
            "seed": args.seed,
            "pair_id": pair_id,
            "source_id": str(first(batch["source_id"])),
            "class_name": str(first(batch["class_name"])),
            "L": int(first(batch["L"])),
            "psnr": psnr(prediction, clean, data_range=1.0, border=0),
            "ssim": ssim(prediction, clean, data_range=1.0, border=0),
        }
        rows.append(row)
    if not rows:
        raise RuntimeError("No fixed pairs were evaluated")

    per_image_path = output / "per_image.csv"
    with open(per_image_path, "x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["class_name"]), int(row["L"]))].append(row)
    cells = []
    for (class_name, looks), values in sorted(grouped.items()):
        cells.append(
            {
                "class_name": class_name,
                "L": looks,
                "pairs": len(values),
                **{
                    metric: float(np.mean([float(value[metric]) for value in values]))
                    for metric in ("psnr", "ssim")
                },
            }
        )
    by_look = {}
    for looks in sorted({int(cell["L"]) for cell in cells}):
        look_cells = [cell for cell in cells if int(cell["L"]) == looks]
        by_look[str(looks)] = {
            metric: float(np.mean([float(cell[metric]) for cell in look_cells]))
            for metric in ("psnr", "ssim")
        }
    aggregate = {
        "protocol_id": PROTOCOL_ID,
        "artifact_protocol_id": artifact_protocol_id,
        "formal_run": not args.nonformal_smoke,
        "dataset": args.dataset,
        "method": args.method,
        "variant": canonical_variant,
        "seed": args.seed,
        "pairs": len(rows),
        "sources": len({str(row["source_id"]) for row in rows}),
        "class_L_cells": len(cells),
        "by_L_class_macro": by_look,
        "macro": {
            metric: float(np.mean([float(cell[metric]) for cell in cells]))
            for metric in ("psnr", "ssim")
        },
        "checkpoint_sha256": sha256_file(args.checkpoint) if args.checkpoint else None,
        "manifest_sha256": sha256_file(args.manifest),
        "per_image_sha256": sha256_file(per_image_path),
        "pretest_gate": pretest_gate,
    }
    with open(output / "class_L_cells.csv", "x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cells[0]))
        writer.writeheader()
        writer.writerows(cells)
    atomic_json_dump(aggregate, output / "aggregate.json")
    print(json.dumps(aggregate, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
