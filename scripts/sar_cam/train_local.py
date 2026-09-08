"""Local SAR-CAM training using the official model and DG+TV loss.

This is a local retraining demonstration, not the paper's trained weights or
full training protocol. Training and validation are separate paired MAT roots.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch

from run_local import (OFFICIAL_ROOT, build_model, import_official, load_intensity,
                       load_checkpoint, model_metadata, normalize_intensity, predict, psnr,
                       save_result, select_device, sha256, verify_source)


def collect_samples(root: Path, count: int, seed: int) -> list[Path]:
    paths = sorted(root.rglob("*.mat"))
    if not paths:
        raise FileNotFoundError(f"No paired MAT images found under {root}")
    rng = np.random.default_rng(seed)
    indices = sorted(rng.choice(len(paths), size=min(count, len(paths)), replace=False).tolist())
    return [paths[index] for index in indices]


def load_pair(path: Path, noisy_field: str, clean_field: str) -> tuple[np.ndarray, np.ndarray]:
    noisy, _ = load_intensity(path, noisy_field)
    clean, _ = load_intensity(path, clean_field)
    # Existing normalized intensity_v1 dataset and official raster ToTensor
    # inputs share [0,1]. Keep the chosen numeric domain explicit.
    noisy, _ = normalize_intensity(noisy, "none")
    clean, _ = normalize_intensity(clean, "none")
    if clean.shape != noisy.shape:
        raise ValueError(f"Mismatched noisy/clean pair: {path}")
    return noisy, clean


def crop_pair(pair: tuple[np.ndarray, np.ndarray], size: int, rng=None):
    noisy, clean = pair
    h, w = noisy.shape
    if min(h, w) < size:
        raise ValueError(f"Training image {noisy.shape} smaller than requested crop {size}")
    if rng is None:
        row, col = (h - size) // 2, (w - size) // 2
    else:
        row, col = int(rng.integers(h - size + 1)), int(rng.integers(w - size + 1))
    noisy = noisy[row:row + size, col:col + size]
    clean = clean[row:row + size, col:col + size]
    if rng is not None:
        if rng.random() < 0.5:
            noisy, clean = noisy[::-1], clean[::-1]
        if rng.random() < 0.5:
            noisy, clean = noisy[:, ::-1], clean[:, ::-1]
    return np.ascontiguousarray(noisy), np.ascontiguousarray(clean)


def clean_identity(clean: np.ndarray) -> str:
    array = np.ascontiguousarray(clean, dtype=np.float32)
    digest = hashlib.sha256(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def validate(model, pairs, device) -> dict:
    scores = []
    noisy_scores = []
    for noisy, clean in pairs:
        prediction, _ = predict(model, noisy, device)
        scores.append(psnr(clean, prediction))
        noisy_scores.append(psnr(clean, noisy))
    if any(value is None for value in scores + noisy_scores):
        raise ValueError("Validation has an exactly identical pair; PSNR is not finite")
    return {"denoised_psnr_db": float(np.mean(scores)),
            "noisy_psnr_db": float(np.mean(noisy_scores)),
            "data_range": 1, "domain": "normalized_intensity", "prediction_clipped": False,
            "validation_images": len(pairs)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--val-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=OFFICIAL_ROOT)
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--resume", type=Path, help="Resume checkpoint_last.pth; --updates is the total target step")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--crop-size", type=int, default=64)
    parser.add_argument("--train-samples", type=int, default=512)
    parser.add_argument("--val-samples", type=int, default=8)
    parser.add_argument("--validation-interval", type=int, default=50)
    parser.add_argument("--noisy-field", default="noisy")
    parser.add_argument("--clean-field", default="clean")
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    for key in ("updates", "batch_size", "train_samples", "val_samples", "validation_interval", "threads"):
        if getattr(args, key) < 1:
            raise ValueError(f"{key} must be positive")
    if args.crop_size < 16 or args.crop_size % 2:
        raise ValueError("crop-size must be even and at least 16")
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    train_root, val_root = args.train_root.resolve(), args.val_root.resolve()
    if train_root == val_root or train_root in val_root.parents or val_root in train_root.parents:
        raise ValueError("Train and validation roots must be disjoint")
    source = verify_source(args.source_root)
    train_paths = collect_samples(train_root, args.train_samples, args.seed)
    val_paths = collect_samples(val_root, args.val_samples, args.seed + 1)
    train_ids = {str(path.relative_to(train_root)).lower() for path in train_paths}
    val_ids = {str(path.relative_to(val_root)).lower() for path in val_paths}
    if train_ids & val_ids:
        raise ValueError("Train and validation sample identities overlap")
    # Hash the exact selected files and reject byte-identical pairs across splits.
    train_hashes = {str(path): sha256(path) for path in train_paths}
    val_hashes = {str(path): sha256(path) for path in val_paths}
    if set(train_hashes.values()) & set(val_hashes.values()):
        raise ValueError("Identical paired MAT files occur in training and validation")
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    device = select_device(args.device)
    model = build_model(args.source_root).to(device)
    loss_function = import_official(args.source_root, "loss.py").Loss(device, 2e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    train_pairs = [load_pair(path, args.noisy_field, args.clean_field) for path in train_paths]
    full_val_pairs = [load_pair(path, args.noisy_field, args.clean_field) for path in val_paths]
    train_clean_hashes = {str(path): clean_identity(pair[1]) for path, pair in zip(train_paths, train_pairs)}
    val_clean_hashes = {str(path): clean_identity(pair[1]) for path, pair in zip(val_paths, full_val_pairs)}
    if set(train_clean_hashes.values()) & set(val_clean_hashes.values()):
        raise ValueError("Clean source image content overlaps between train and validation, possibly at different L/noise")
    val_pairs = [crop_pair(pair, args.crop_size) for pair in full_val_pairs]
    # Catch invalid crop requests and zero-noise pairs before opening outputs.
    for pair in train_pairs:
        crop_pair(pair, args.crop_size)
    provenance = {"kind": "locally_retrained_smoke" if args.updates <= 1000 else "locally_retrained",
                  "official_pretrained": False,
                  "paper_reproduction": False, "initialized_from": "random_default_official_constructor",
                  "training_updates_requested": args.updates,
                  "description": "Local training of the untouched official SAR-CAM architecture and DG+TV loss"}
    run_config = {"numeric_domain": "intensity_v1", "method": "sar_cam", "formal_run": False,
                  "seed": args.seed, "updates": args.updates, "batch_size": args.batch_size,
                  "crop_size": args.crop_size, "learning_rate": 1e-4, "weight_decay": 1e-5,
                  "lambda_tv": 2e-4, "scheduler": "fixed_learning_rate",
                  "paired_random_flips": True, "validation_crop": "center",
                  "train_root": str(train_root), "val_root": str(val_root),
                  "train_sample_count": len(train_paths), "val_sample_count": len(val_paths),
                  "validation_interval": args.validation_interval,
                  "noisy_field": args.noisy_field, "clean_field": args.clean_field}
    rng = np.random.default_rng(args.seed)
    start_step = 0
    resume_payload = None
    if args.resume:
        checkpoint_info = load_checkpoint(model, args.resume)
        resume_payload = torch.load(args.resume, map_location="cpu", weights_only=True)
        if resume_payload.get("checkpoint_format") != "sar-cam-local-resumable-v2":
            raise ValueError("Exact resume requires a v2 checkpoint_last.pth with optimizer/RNG state")
        previous_config = resume_payload["run_config"]
        invariants = ("numeric_domain", "seed", "batch_size", "crop_size", "learning_rate",
                      "weight_decay", "lambda_tv", "scheduler", "train_root", "val_root",
                      "train_sample_count", "val_sample_count", "validation_interval", "noisy_field", "clean_field")
        for key in invariants:
            if previous_config.get(key) != run_config[key]:
                raise ValueError(f"Resume training setting changed: {key}")
        if resume_payload.get("train_files_sha256") != train_hashes or resume_payload.get("val_files_sha256") != val_hashes:
            raise ValueError("Resume input selection or file contents differ from checkpoint")
        start_step = int(resume_payload["global_step"])
        if start_step >= args.updates:
            raise ValueError("--updates is the total target and must exceed the resumed global step")
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        rng.bit_generator.state = resume_payload["crop_rng_state"]
        torch.set_rng_state(resume_payload["torch_rng_state"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(resume_payload["cuda_rng_state"])
        provenance["initialized_from"] = "exact_resume"
        provenance["resume_checkpoint_sha256"] = checkpoint_info["sha256"]
        run_config["resumed_global_step"] = start_step
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {"source": source, "provenance": provenance, "run_config": run_config,
                "train_files_sha256": train_hashes, "val_files_sha256": val_hashes,
                "train_clean_sha256": train_clean_hashes, "val_clean_sha256": val_clean_hashes,
                "training_validation_clean_content_disjoint": True,
                "training_validation_ids_disjoint": True, "python_version": platform.python_version(),
                "torch_version": str(torch.__version__), "device": str(device),
                "cuda_version": torch.version.cuda,
                "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None}
    (args.output / "training_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    started = time.perf_counter()
    initial_metrics = (resume_payload["initial_random_validation"] if resume_payload else
                       validate(model, val_pairs, device))
    if start_step == 0:
        print(json.dumps({"step": 0, "label": "random_initialization_validation_only", **initial_metrics}), flush=True)
    else:
        print(json.dumps({"exact_resume_step": start_step, "total_target_updates": args.updates}), flush=True)
    best_psnr = resume_payload["best_metrics"]["denoised_psnr_db"] if resume_payload else -float("inf")
    best_step = resume_payload["best_step"] if resume_payload else 0
    best_metrics = resume_payload["best_metrics"] if resume_payload else None
    best_state = resume_payload["best_state_dict"] if resume_payload else None
    metrics_rows = []
    losses = []

    def checkpoint(path, step, metrics, resumable=False):
        state = ({key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                 if resumable else best_state)
        payload = {"checkpoint_format": "sar-cam-local-v1", "state_dict": state,
                   "model_metadata": model_metadata(), "provenance": provenance,
                   "run_config": run_config, "global_step": step, "metrics": metrics}
        if resumable:
            payload.update({"checkpoint_format": "sar-cam-local-resumable-v2",
                            "optimizer_state_dict": optimizer.state_dict(),
                            "crop_rng_state": rng.bit_generator.state,
                            "torch_rng_state": torch.get_rng_state(),
                            "cuda_rng_state": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                            "train_files_sha256": train_hashes, "val_files_sha256": val_hashes,
                            "best_state_dict": best_state, "best_metrics": best_metrics,
                            "best_step": best_step, "initial_random_validation": initial_metrics})
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    with (args.output / "progress.jsonl").open("x", encoding="utf-8") as progress:
        for step in range(start_step + 1, args.updates + 1):
            model.train()
            crops = [crop_pair(train_pairs[int(rng.integers(len(train_pairs)))], args.crop_size, rng)
                     for _ in range(args.batch_size)]
            inputs = torch.from_numpy(np.stack([pair[0] for pair in crops]))[:, None].to(device)
            targets = torch.from_numpy(np.stack([pair[1] for pair in crops]))[:, None].to(device)
            if torch.mean((inputs - targets) ** 2) <= 0:
                raise ValueError("Zero-noise training batch makes the author's DG loss undefined")
            optimizer.zero_grad(set_to_none=True)
            predictions = model(inputs)
            loss = loss_function(inputs, predictions, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite official training loss at update {step}")
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError(f"Nonfinite training gradient at update {step}")
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            if step % 10 == 0 or step == 1:
                event = {"step": step, "official_dg_tv_loss": losses[-1],
                         "elapsed_seconds": time.perf_counter() - started}
                print(json.dumps(event), flush=True)
                progress.write(json.dumps(event) + "\n")
                progress.flush()
            if step % args.validation_interval == 0 or step == args.updates:
                metrics = validate(model, val_pairs, device)
                row = {"step": step, "train_loss_mean": float(np.mean(losses)), **metrics}
                metrics_rows.append(row)
                losses.clear()
                print(json.dumps(row), flush=True)
                progress.write(json.dumps(row) + "\n")
                progress.flush()
                if metrics["denoised_psnr_db"] > best_psnr:
                    best_psnr, best_step, best_metrics = metrics["denoised_psnr_db"], step, metrics
                    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                    checkpoint(args.output / "checkpoint_best.pth", step, metrics)
                checkpoint(args.output / "checkpoint_last.pth", step, metrics, resumable=True)
        if not (args.output / "checkpoint_best.pth").exists():
            checkpoint(args.output / "checkpoint_best.pth", best_step, best_metrics)
    with (args.output / "metrics.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics_rows[0]))
        writer.writeheader()
        writer.writerows(metrics_rows)
    payload = torch.load(args.output / "checkpoint_best.pth", map_location=device, weights_only=True)
    model.load_state_dict(payload["state_dict"], strict=True)
    noisy, clean = val_pairs[0]
    prediction, timing = predict(model, noisy, device)
    save_result(args.output / "validation_example", noisy, prediction,
                {"algorithm": "SAR-CAM", "provenance": provenance, "best_step": best_step,
                 "source": source, "input_path": str(val_paths[0]), "crop": "center",
                 "inference": timing}, clean)
    summary = {"schema_version": 1, "provenance": provenance, "source": source,
               "run_config": run_config, "initial_random_validation": initial_metrics,
               "best_step": best_step, "best_validation": best_metrics,
               "elapsed_seconds": time.perf_counter() - started,
               "checkpoint_best_sha256": sha256(args.output / "checkpoint_best.pth"),
               "parameter_count": sum(p.numel() for p in model.parameters()),
               "device": str(device), "torch_version": str(torch.__version__),
               "cuda_version": torch.version.cuda,
               "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
