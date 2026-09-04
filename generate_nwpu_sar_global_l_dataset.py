"""Generate the controlled global-L NWPU synthetic-SAR protocol.

The 45 NWPU-RESISC45 classes are split deterministically into 560 training and
140 validation source images per class.  Each training source receives exactly
one global number of looks (L in {1, 2, 4, 8}); assignments are balanced within
every class.  Every validation source is rendered four times, once for each L.

The previous 4x4 mixed-L dataset is intentionally not modified by this script.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.io import loadmat, savemat

from synthetic_manifest import verify_paired_sar_manifest


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
EXPECTED_CLASSES = 45
EXPECTED_PER_CLASS = 700
TRAIN_PER_CLASS = 560
VAL_PER_CLASS = 140
LOOKS = (1, 2, 4, 8)

SYNTHESIS_CONFIG: dict[str, Any] = {
    "schema_version": "nwpu_global_l_composite_sar_v1",
    "image_size": [256, 256],
    "grayscale": "0.299R+0.587G+0.114B, normalized to [0,1]",
    "clean_intensity": "square(normalized_grayscale)",
    "speckle_model": "Y=X*N, N~Gamma(shape=L, scale=1/L)",
    "speckle_mode": "one global L per image",
    "global_L_choices": list(LOOKS),
    "strong_scatter_changes_L": False,
    "bright_percentile": 90.0,
    "edge_percentile": 85.0,
    "strong_mask_dilation_iterations": 1,
    "target_gain": 0.20,
    "impulse_probability": 0.008,
    "impulse_gain_range": [1.1, 1.6],
    "additive_noise_sigma": 0.015,
    "misregistration_alpha": 0.08,
    "shift_xy": [1, 2],
    "stripe_amplitude": 0.02,
    "row_probability": 0.08,
    "column_probability": 0.08,
    "final_clip": [0.0, 1.0],
    "mat_fields": ["clean", "noisy", "global_L"],
    "numeric_domain": "intensity_v1",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(base_seed: int, namespace: str) -> int:
    payload = f"{base_seed}:{namespace}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def discover_sources(source_root: Path) -> dict[str, list[Path]]:
    classes = sorted(path for path in source_root.iterdir() if path.is_dir())
    if len(classes) != EXPECTED_CLASSES:
        raise AssertionError(f"Expected {EXPECTED_CLASSES} class folders, found {len(classes)}")
    discovered: dict[str, list[Path]] = {}
    for class_dir in classes:
        files = sorted(
            path for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if len(files) != EXPECTED_PER_CLASS:
            raise AssertionError(
                f"Class {class_dir.name!r} has {len(files)} images, expected {EXPECTED_PER_CLASS}"
            )
        discovered[class_dir.name] = files
    return discovered


def balanced_look_assignment(
    paths: list[Path], class_name: str, assignment_seed: int
) -> dict[Path, int]:
    if len(paths) % len(LOOKS):
        raise AssertionError("Training count must be divisible by the number of L values")
    rng = np.random.default_rng(stable_seed(assignment_seed, f"train-L:{class_name}"))
    shuffled = [paths[int(index)] for index in rng.permutation(len(paths))]
    per_look = len(paths) // len(LOOKS)
    assigned: dict[Path, int] = {}
    for look_index, looks in enumerate(LOOKS):
        start = look_index * per_look
        for path in shuffled[start : start + per_look]:
            assigned[path] = looks
    return assigned


def build_generation_plan(
    source_root: Path,
    split_seed: int,
    assignment_seed: int,
    synthesis_seed: int,
) -> dict[str, Any]:
    source_root = source_root.resolve()
    classes = discover_sources(source_root)
    records: list[dict[str, Any]] = []
    class_counts: dict[str, Any] = {}

    for class_name, paths in classes.items():
        # Keep the exact clean-image split used by the retained mixed-L dataset,
        # so robustness comparisons do not introduce a split confound.
        split_rng = np.random.default_rng(stable_seed(split_seed, class_name))
        permutation = split_rng.permutation(len(paths))
        val_indices = set(int(value) for value in permutation[:VAL_PER_CLASS])
        train_paths = [path for index, path in enumerate(paths) if index not in val_indices]
        val_paths = [path for index, path in enumerate(paths) if index in val_indices]
        if len(train_paths) != TRAIN_PER_CLASS or len(val_paths) != VAL_PER_CLASS:
            raise AssertionError(f"Unexpected class split for {class_name}")

        train_assignment = balanced_look_assignment(
            train_paths, class_name, assignment_seed
        )
        train_counts = Counter(train_assignment.values())
        expected_train_counts = {looks: TRAIN_PER_CLASS // len(LOOKS) for looks in LOOKS}
        if dict(sorted(train_counts.items())) != expected_train_counts:
            raise AssertionError(f"Unbalanced L assignment for {class_name}: {train_counts}")

        for path in train_paths:
            source_relative = path.relative_to(source_root).as_posix()
            looks = train_assignment[path]
            output_relative = f"train/L{looks}/{class_name}/{path.stem}.mat"
            records.append(
                {
                    "class": class_name,
                    "split": "train",
                    "global_L": looks,
                    "source_relative": source_relative,
                    "output_relative": output_relative,
                    "sample_seed": stable_seed(
                        synthesis_seed, f"train:{source_relative}:L{looks}"
                    ),
                    "source_sha256": sha256_file(path),
                }
            )

        for path in val_paths:
            source_relative = path.relative_to(source_root).as_posix()
            source_hash = sha256_file(path)
            for looks in LOOKS:
                output_relative = f"val/L{looks}/{class_name}/{path.stem}.mat"
                records.append(
                    {
                        "class": class_name,
                        "split": "val",
                        "global_L": looks,
                        "source_relative": source_relative,
                        "output_relative": output_relative,
                        "sample_seed": stable_seed(
                            synthesis_seed, f"val:{source_relative}:L{looks}"
                        ),
                        "source_sha256": source_hash,
                    }
                )

        class_counts[class_name] = {
            "train_sources": TRAIN_PER_CLASS,
            "val_sources": VAL_PER_CLASS,
            "train_by_L": {f"L{looks}": expected_train_counts[looks] for looks in LOOKS},
            "val_pairs_by_L": {f"L{looks}": VAL_PER_CLASS for looks in LOOKS},
        }

    records.sort(
        key=lambda item: (
            item["split"],
            item["class"],
            item["source_relative"],
            item["global_L"],
        )
    )
    return {
        "schema_version": 2,
        "dataset": "NWPU-RESISC45 controlled global-L paired synthetic SAR",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_root),
        "split": {
            "method": "per-class deterministic stratified 8:2",
            "seed": split_seed,
            "train_per_class": TRAIN_PER_CLASS,
            "val_per_class": VAL_PER_CLASS,
            "source_overlap": 0,
        },
        "L_assignment": {
            "method": "per-class deterministic balanced assignment",
            "seed": assignment_seed,
            "choices": list(LOOKS),
            "train_per_class_per_L": TRAIN_PER_CLASS // len(LOOKS),
            "validation": "each validation source rendered once at every L",
        },
        "synthesis_seed": synthesis_seed,
        "synthesis": SYNTHESIS_CONFIG,
        "robustness_dataset": {
            "path": "../NWPU_RESISC45_SAR_intensity_v1",
            "role": "4x4 mixed-L complex-noise robustness ablation only",
        },
        "class_counts": class_counts,
        "counts": {
            "train": EXPECTED_CLASSES * TRAIN_PER_CLASS,
            "val": EXPECTED_CLASSES * VAL_PER_CLASS * len(LOOKS),
            "unique_source_images": EXPECTED_CLASSES * EXPECTED_PER_CLASS,
            "total_pairs": EXPECTED_CLASSES * (TRAIN_PER_CLASS + VAL_PER_CLASS * len(LOOKS)),
        },
        "records": records,
    }


def _gradient_magnitude(image: np.ndarray) -> np.ndarray:
    gx = np.zeros_like(image, dtype=np.float32)
    gy = np.zeros_like(image, dtype=np.float32)
    gx[:, 1:-1] = (image[:, 2:] - image[:, :-2]) * 0.5
    gx[:, 0] = image[:, 1] - image[:, 0]
    gx[:, -1] = image[:, -1] - image[:, -2]
    gy[1:-1, :] = (image[2:, :] - image[:-2, :]) * 0.5
    gy[0, :] = image[1, :] - image[0, :]
    gy[-1, :] = image[-1, :] - image[-2, :]
    return np.sqrt(gx * gx + gy * gy)


def _dilate_mask(mask: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    padded = np.pad(mask, ((1, 1), (1, 1)), mode="edge")
    output = np.zeros_like(mask, dtype=bool)
    for dy in range(3):
        for dx in range(3):
            output |= padded[dy : dy + height, dx : dx + width]
    return output


def synthesize_noisy_global_l(
    clean: np.ndarray, rng: np.random.Generator, looks: int
) -> np.ndarray:
    if looks not in LOOKS:
        raise ValueError(f"Unsupported L={looks}")
    height, width = clean.shape
    speckle = rng.gamma(
        shape=float(looks), scale=1.0 / float(looks), size=(height, width)
    ).astype(np.float32)
    noisy = clean * speckle

    gradient = _gradient_magnitude(clean)
    bright_threshold = np.percentile(clean, SYNTHESIS_CONFIG["bright_percentile"])
    edge_threshold = np.percentile(gradient, SYNTHESIS_CONFIG["edge_percentile"])
    strong_mask = _dilate_mask(
        (clean >= bright_threshold) | (gradient >= edge_threshold)
    )
    # The mask changes only gain/impulse terms.  It never changes L, preserving
    # the single-global-L experimental control.
    noisy[strong_mask] *= 1.0 + SYNTHESIS_CONFIG["target_gain"]
    impulse_mask = strong_mask & (
        rng.random((height, width)) < SYNTHESIS_CONFIG["impulse_probability"]
    )
    if impulse_mask.any():
        low, high = SYNTHESIS_CONFIG["impulse_gain_range"]
        noisy[impulse_mask] *= rng.uniform(
            low, high, size=int(impulse_mask.sum())
        ).astype(np.float32)
    noisy = np.clip(noisy, 0.0, 1.0)

    noisy += rng.normal(
        0.0, SYNTHESIS_CONFIG["additive_noise_sigma"], size=(height, width)
    ).astype(np.float32)
    noisy = np.clip(noisy, 0.0, 1.0)

    dy, dx = SYNTHESIS_CONFIG["shift_xy"]
    shifted = np.roll(np.roll(noisy, shift=dy, axis=0), shift=dx, axis=1)
    alpha = SYNTHESIS_CONFIG["misregistration_alpha"]
    noisy = (1.0 - alpha) * noisy + alpha * shifted

    stripe_amplitude = SYNTHESIS_CONFIG["stripe_amplitude"]
    row_offsets = np.zeros((height, 1), dtype=np.float32)
    column_offsets = np.zeros((1, width), dtype=np.float32)
    row_mask = rng.random(height) < SYNTHESIS_CONFIG["row_probability"]
    column_mask = rng.random(width) < SYNTHESIS_CONFIG["column_probability"]
    if row_mask.any():
        row_offsets[row_mask, 0] = rng.normal(
            0.0, stripe_amplitude, size=int(row_mask.sum())
        ).astype(np.float32)
    if column_mask.any():
        column_offsets[0, column_mask] = rng.normal(
            0.0, stripe_amplitude, size=int(column_mask.sum())
        ).astype(np.float32)
    rows = np.arange(height, dtype=np.float32)
    columns = np.arange(width, dtype=np.float32)
    row_offsets += (
        0.5 * stripe_amplitude * np.sin(2 * np.pi * rows / max(16, height // 4))
    )[:, None]
    column_offsets += (
        0.5 * stripe_amplitude * np.sin(2 * np.pi * columns / max(16, width // 5))
    )[None, :]
    return np.clip(noisy + row_offsets + column_offsets, 0.0, 1.0).astype(np.float32)


def _load_clean(source: Path) -> np.ndarray:
    with Image.open(source) as image:
        rgb = image.convert("RGB").resize(
            tuple(SYNTHESIS_CONFIG["image_size"]), Image.Resampling.BILINEAR
        )
        array = np.asarray(rgb, dtype=np.float32)
    gray = (
        0.299 * array[..., 0] + 0.587 * array[..., 1] + 0.114 * array[..., 2]
    ) / 255.0
    return np.square(np.clip(gray, 0.0, 1.0)).astype(np.float32)


def _validate_mat(path: Path, expected_looks: int) -> None:
    data = loadmat(path)
    for field in ("clean", "noisy"):
        if field not in data:
            raise AssertionError(f"Missing {field}: {path}")
        value = np.asarray(data[field])
        if value.shape != (256, 256) or value.dtype != np.float32:
            raise AssertionError(f"Invalid {field} in {path}: {value.shape}, {value.dtype}")
        if not np.isfinite(value).all() or value.min() < 0.0 or value.max() > 1.0:
            raise AssertionError(f"Invalid {field} range in {path}")
    saved_looks = int(np.asarray(data.get("global_L", [])).squeeze())
    if saved_looks != expected_looks:
        raise AssertionError(f"L mismatch in {path}: {saved_looks} != {expected_looks}")


def _generate_one(arguments: tuple[str, str, dict[str, Any]]) -> dict[str, Any]:
    source_root_text, output_root_text, record = arguments
    source_root = Path(source_root_text)
    output_root = Path(output_root_text)
    source = source_root / record["source_relative"]
    output = output_root / record["output_relative"]
    output.parent.mkdir(parents=True, exist_ok=True)
    existed = output.exists()
    looks = int(record["global_L"])
    if existed:
        _validate_mat(output, looks)
    else:
        clean = _load_clean(source)
        noisy = synthesize_noisy_global_l(
            clean, np.random.default_rng(record["sample_seed"]), looks
        )
        temporary = output.with_suffix(".tmp.mat")
        savemat(
            temporary,
            {"clean": clean, "noisy": noisy, "global_L": np.int16(looks)},
            do_compression=False,
        )
        _validate_mat(temporary, looks)
        temporary.replace(output)
    return {
        "output_relative": record["output_relative"],
        "sha256": sha256_file(output),
        "reused": existed,
    }


def _write_assignment_tables(output_root: Path, plan: dict[str, Any]) -> None:
    fieldnames = [
        "split", "class", "source_relative", "output_relative", "global_L", "sample_seed"
    ]
    for split, filename in (("train", "train_l_assignment.csv"), ("val", "validation_l_sets.csv")):
        path = output_root / filename
        rows = [record for record in plan["records"] if record["split"] == split]
        if path.exists():
            continue
        temporary = path.with_suffix(".tmp")
        with temporary.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for record in rows:
                writer.writerow({key: record[key] for key in fieldnames})
        temporary.replace(path)


def _load_or_create_plan(
    source_root: Path,
    output_root: Path,
    split_seed: int,
    assignment_seed: int,
    synthesis_seed: int,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    plan_path = output_root / "generation_plan.json"
    if plan_path.exists():
        with plan_path.open("r", encoding="utf-8") as handle:
            plan = json.load(handle)
        if plan.get("split", {}).get("seed") != split_seed:
            raise AssertionError("Existing generation plan uses a different split seed")
        if plan.get("L_assignment", {}).get("seed") != assignment_seed:
            raise AssertionError("Existing generation plan uses a different L-assignment seed")
        if plan.get("synthesis_seed") != synthesis_seed:
            raise AssertionError("Existing generation plan uses a different synthesis seed")
        if plan.get("synthesis") != SYNTHESIS_CONFIG:
            raise AssertionError("Existing generation plan uses a different synthesis configuration")
        if len(plan.get("records", [])) != plan["counts"]["total_pairs"]:
            raise AssertionError("Existing generation plan is incomplete")
    else:
        plan = build_generation_plan(
            source_root, split_seed, assignment_seed, synthesis_seed
        )
        with plan_path.open("x", encoding="utf-8") as handle:
            json.dump(plan, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
    _write_assignment_tables(output_root, plan)
    return plan


def validate_protocol_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    mapping = manifest.get("global_L_by_file", {})
    files = manifest.get("files", {})
    assigned_paths = files.get("train", []) + files.get("val", [])
    if set(mapping) != set(assigned_paths):
        raise AssertionError("global_L_by_file must cover every generated pair exactly once")
    train_by_class_l: dict[tuple[str, int], int] = defaultdict(int)
    for relative in files["train"]:
        parts = Path(relative).parts
        train_by_class_l[(parts[2], int(mapping[relative]))] += 1
    bad = {
        f"{class_name}/L{looks}": count
        for (class_name, looks), count in train_by_class_l.items()
        if count != TRAIN_PER_CLASS // len(LOOKS)
    }
    if bad or len(train_by_class_l) != EXPECTED_CLASSES * len(LOOKS):
        raise AssertionError(f"Training L balance failed: {bad}")
    val_counts = Counter(int(mapping[path]) for path in files["val"])
    expected_val = EXPECTED_CLASSES * VAL_PER_CLASS
    if val_counts != Counter({looks: expected_val for looks in LOOKS}):
        raise AssertionError(f"Validation L counts failed: {val_counts}")
    train_sources = set(manifest["source_files"]["train"])
    val_sources = set(manifest["source_files"]["val"])
    overlap = train_sources & val_sources
    if overlap:
        raise AssertionError(f"Train/validation source overlap: {len(overlap)}")
    return {
        "valid": True,
        "train_per_class_per_L": TRAIN_PER_CLASS // len(LOOKS),
        "validation_pairs_per_L": expected_val,
        "source_overlap": 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default="NWPU-RESISC45")
    parser.add_argument("--output-root", default="NWPU_RESISC45_SAR_global_L_v1")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--assignment-seed", type=int, default=42)
    parser.add_argument("--synthesis-seed", type=int, default=20260904)
    parser.add_argument(
        "--workers", type=int, default=min(8, max(1, (os.cpu_count() or 2) - 1))
    )
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_root = Path(args.source_root).resolve()
    output_root = Path(args.output_root).resolve()
    manifest_path = output_root / "dataset_manifest.json"
    if args.verify_only:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        verification = verify_paired_sar_manifest(
            output_root, manifest, verify_hashes=True
        )
        verification["global_L_protocol"] = validate_protocol_manifest(manifest)
        print(json.dumps(verification, indent=2, ensure_ascii=False, sort_keys=True))
        return 0

    plan = _load_or_create_plan(
        source_root,
        output_root,
        args.split_seed,
        args.assignment_seed,
        args.synthesis_seed,
    )
    if args.plan_only:
        print(json.dumps(
            {
                "plan": str(output_root / "generation_plan.json"),
                "train_assignment": str(output_root / "train_l_assignment.csv"),
                "validation_sets": str(output_root / "validation_l_sets.csv"),
                "counts": plan["counts"],
            },
            indent=2,
        ))
        return 0
    if manifest_path.exists():
        raise FileExistsError(
            f"Completed dataset manifest already exists: {manifest_path}; use --verify-only"
        )
    if args.workers <= 0:
        raise ValueError("workers must be positive")

    records = plan["records"]
    jobs = ((str(source_root), str(output_root), record) for record in records)
    output_hashes: dict[str, str] = {}
    reused = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for index, result in enumerate(
            executor.map(_generate_one, jobs, chunksize=8), start=1
        ):
            output_hashes[result["output_relative"]] = result["sha256"]
            reused += int(result["reused"])
            if index % 250 == 0 or index == len(records):
                print(
                    f"generated_or_verified={index}/{len(records)} reused={reused}",
                    flush=True,
                )

    files = {
        split: [
            record["output_relative"] for record in records if record["split"] == split
        ]
        for split in ("train", "val")
    }
    unique_sources = {
        split: sorted({
            record["source_relative"] for record in records if record["split"] == split
        })
        for split in ("train", "val")
    }
    validation_sets = {
        f"L{looks}": {
            "global_L": looks,
            "count": sum(
                record["split"] == "val" and record["global_L"] == looks
                for record in records
            ),
            "files": sorted(
                record["output_relative"]
                for record in records
                if record["split"] == "val" and record["global_L"] == looks
            ),
        }
        for looks in LOOKS
    }
    source_hashes = {}
    for record in records:
        source_hashes[record["source_relative"]] = record["source_sha256"]
    manifest = {
        "schema_version": 2,
        "dataset": plan["dataset"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": Path(__file__).name,
        "generator_sha256": sha256_file(Path(__file__)),
        "source_root": str(source_root),
        "split": plan["split"],
        "L_assignment": plan["L_assignment"],
        "synthesis_seed": plan["synthesis_seed"],
        "synthesis": plan["synthesis"],
        "robustness_dataset": plan["robustness_dataset"],
        "counts": {"train": len(files["train"]), "val": len(files["val"])},
        "unique_source_counts": {
            split: len(paths) for split, paths in unique_sources.items()
        },
        "files": files,
        "validation_sets": validation_sets,
        "global_L_by_file": {
            record["output_relative"]: record["global_L"] for record in records
        },
        "file_sha256": dict(sorted(output_hashes.items())),
        "source_files": unique_sources,
        "source_sha256": dict(sorted(source_hashes.items())),
        "sample_seeds": {
            record["output_relative"]: record["sample_seed"] for record in records
        },
        "assignment_files": {
            name: sha256_file(output_root / name)
            for name in ("train_l_assignment.csv", "validation_l_sets.csv")
        },
    }
    temporary_manifest = manifest_path.with_suffix(".tmp")
    with temporary_manifest.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    temporary_manifest.replace(manifest_path)

    verification = verify_paired_sar_manifest(
        output_root, manifest, verify_hashes=False
    )
    verification["global_L_protocol"] = validate_protocol_manifest(manifest)
    print(json.dumps(verification, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
