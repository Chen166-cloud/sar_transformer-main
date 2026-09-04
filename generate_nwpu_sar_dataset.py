"""Generate the report-defined paired SAR dataset from NWPU-RESISC45.

The split is stratified per scene class (560 train / 140 validation images per
class). Generation is deterministic per source path, resumable, non-destructive,
and produces a manifest containing source/output SHA-256 hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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
SYNTHESIS_CONFIG: dict[str, Any] = {
    "schema_version": "nwpu_complex_sar_v1",
    "image_size": [256, 256],
    "grayscale": "0.299R+0.587G+0.114B, normalized to [0,1]",
    "clean_intensity": "square(normalized_grayscale)",
    "grid": [4, 4],
    "region_L_choices": [1, 2, 4, 8],
    "strong_L": 1,
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
    "mat_fields": ["clean", "noisy"],
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


def build_generation_plan(
    source_root: Path,
    split_seed: int,
    synthesis_seed: int,
) -> dict[str, Any]:
    source_root = source_root.resolve()
    classes = discover_sources(source_root)
    records: list[dict[str, Any]] = []
    class_counts: dict[str, dict[str, int]] = {}
    for class_name, paths in classes.items():
        rng = np.random.default_rng(stable_seed(split_seed, class_name))
        permutation = rng.permutation(len(paths))
        val_indices = set(int(value) for value in permutation[:140])
        counts = {"train": 0, "val": 0}
        for index, path in enumerate(paths):
            split = "val" if index in val_indices else "train"
            counts[split] += 1
            source_relative = path.relative_to(source_root).as_posix()
            output_relative = f"{split}/{class_name}/{path.stem}.mat"
            records.append(
                {
                    "class": class_name,
                    "split": split,
                    "source_relative": source_relative,
                    "output_relative": output_relative,
                    "sample_seed": stable_seed(synthesis_seed, source_relative),
                    "source_sha256": sha256_file(path),
                }
            )
        if counts != {"train": 560, "val": 140}:
            raise AssertionError(f"Unexpected class split for {class_name}: {counts}")
        class_counts[class_name] = counts
    records.sort(key=lambda item: item["source_relative"])
    return {
        "schema_version": 1,
        "dataset": "NWPU-RESISC45 paired synthetic SAR",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_root),
        "split": {
            "method": "per-class deterministic stratified 8:2",
            "seed": split_seed,
            "train_per_class": 560,
            "val_per_class": 140,
        },
        "synthesis_seed": synthesis_seed,
        "synthesis": SYNTHESIS_CONFIG,
        "class_counts": class_counts,
        "counts": {"train": 25200, "val": 6300, "total": 31500},
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


def _build_l_map(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    grid_y, grid_x = SYNTHESIS_CONFIG["grid"]
    choices = SYNTHESIS_CONFIG["region_L_choices"]
    y_edges = np.linspace(0, height, grid_y + 1, dtype=int)
    x_edges = np.linspace(0, width, grid_x + 1, dtype=int)
    l_map = np.zeros((height, width), dtype=np.float32)
    for row in range(grid_y):
        for column in range(grid_x):
            l_map[y_edges[row] : y_edges[row + 1], x_edges[column] : x_edges[column + 1]] = rng.choice(choices)
    return l_map


def synthesize_noisy(clean: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    height, width = clean.shape
    gradient = _gradient_magnitude(clean)
    bright_threshold = np.percentile(clean, SYNTHESIS_CONFIG["bright_percentile"])
    edge_threshold = np.percentile(gradient, SYNTHESIS_CONFIG["edge_percentile"])
    strong_mask = _dilate_mask(
        (clean >= bright_threshold) | (gradient >= edge_threshold)
    )

    l_map = _build_l_map(height, width, rng)
    l_map[strong_mask] = SYNTHESIS_CONFIG["strong_L"]
    noisy = np.zeros_like(clean, dtype=np.float32)
    for looks in np.unique(l_map).astype(np.int32):
        mask = l_map == looks
        speckle = rng.gamma(
            shape=float(looks), scale=1.0 / float(looks), size=int(mask.sum())
        ).astype(np.float32)
        noisy[mask] = clean[mask] * speckle

    noisy[strong_mask] *= 1.0 + SYNTHESIS_CONFIG["target_gain"]
    impulse_mask = strong_mask & (
        rng.random((height, width)) < SYNTHESIS_CONFIG["impulse_probability"]
    )
    if impulse_mask.any():
        low, high = SYNTHESIS_CONFIG["impulse_gain_range"]
        noisy[impulse_mask] *= rng.uniform(low, high, size=int(impulse_mask.sum())).astype(np.float32)
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


def _validate_mat(path: Path) -> None:
    data = loadmat(path)
    if "clean" not in data or "noisy" not in data:
        raise AssertionError(f"Missing clean/noisy fields: {path}")
    for field in ("clean", "noisy"):
        value = np.asarray(data[field])
        if value.shape != (256, 256) or value.dtype != np.float32:
            raise AssertionError(f"Invalid {field} array in {path}: {value.shape}, {value.dtype}")
        if not np.isfinite(value).all() or float(value.min()) < 0.0 or float(value.max()) > 1.0:
            raise AssertionError(f"Invalid {field} range in {path}")


def _generate_one(arguments: tuple[str, str, dict[str, Any]]) -> dict[str, Any]:
    source_root_text, output_root_text, record = arguments
    source_root = Path(source_root_text)
    output_root = Path(output_root_text)
    source = source_root / record["source_relative"]
    output = output_root / record["output_relative"]
    output.parent.mkdir(parents=True, exist_ok=True)
    existed = output.exists()
    if existed:
        _validate_mat(output)
    else:
        clean = _load_clean(source)
        noisy = synthesize_noisy(clean, np.random.default_rng(record["sample_seed"]))
        temporary = output.with_suffix(".tmp.mat")
        savemat(temporary, {"clean": clean, "noisy": noisy}, do_compression=False)
        _validate_mat(temporary)
        temporary.replace(output)
    return {
        "output_relative": record["output_relative"],
        "sha256": sha256_file(output),
        "reused": existed,
    }


def _load_or_create_plan(
    source_root: Path,
    output_root: Path,
    split_seed: int,
    synthesis_seed: int,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    plan_path = output_root / "generation_plan.json"
    if plan_path.exists():
        with plan_path.open("r", encoding="utf-8") as handle:
            plan = json.load(handle)
        if plan.get("split", {}).get("seed") != split_seed or plan.get("synthesis_seed") != synthesis_seed:
            raise AssertionError("Existing generation plan uses different seeds")
        if plan.get("synthesis") != SYNTHESIS_CONFIG:
            raise AssertionError("Existing generation plan uses a different synthesis configuration")
        if len(plan.get("records", [])) != 31500:
            raise AssertionError("Existing generation plan is incomplete")
        return plan
    plan = build_generation_plan(source_root, split_seed, synthesis_seed)
    with plan_path.open("x", encoding="utf-8") as handle:
        json.dump(plan, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    return plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default="NWPU-RESISC45")
    parser.add_argument("--output-root", default="NWPU_RESISC45_SAR_intensity_v1")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--synthesis-seed", type=int, default=20260904)
    parser.add_argument("--workers", type=int, default=min(8, max(1, (os.cpu_count() or 2) - 1)))
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
        print(json.dumps(
            verify_paired_sar_manifest(output_root, manifest, verify_hashes=True),
            indent=2, ensure_ascii=False, sort_keys=True,
        ))
        return 0

    plan = _load_or_create_plan(
        source_root, output_root, args.split_seed, args.synthesis_seed
    )
    if args.plan_only:
        print(json.dumps({"plan": str(output_root / "generation_plan.json"), "counts": plan["counts"]}, indent=2))
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
        for index, result in enumerate(executor.map(_generate_one, jobs, chunksize=8), start=1):
            output_hashes[result["output_relative"]] = result["sha256"]
            reused += int(result["reused"])
            if index % 250 == 0 or index == len(records):
                print(
                    f"generated_or_verified={index}/{len(records)} reused={reused}",
                    flush=True,
                )

    files = {
        split: sorted(record["output_relative"] for record in records if record["split"] == split)
        for split in ("train", "val")
    }
    source_files = {
        split: sorted(record["source_relative"] for record in records if record["split"] == split)
        for split in ("train", "val")
    }
    manifest = {
        "schema_version": 1,
        "dataset": plan["dataset"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": Path(__file__).name,
        "generator_sha256": sha256_file(Path(__file__)),
        "source_root": str(source_root),
        "source_counts": plan["counts"],
        "split": plan["split"],
        "synthesis_seed": plan["synthesis_seed"],
        "synthesis": plan["synthesis"],
        "counts": {split: len(paths) for split, paths in files.items()},
        "files": files,
        "file_sha256": dict(sorted(output_hashes.items())),
        "source_files": source_files,
        "source_sha256": {
            record["source_relative"]: record["source_sha256"] for record in records
        },
        "sample_seeds": {
            record["output_relative"]: record["sample_seed"] for record in records
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
    print(json.dumps(verification, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
