"""Data protocol utilities for the frozen ICSPS 2026 experiments.

This module deliberately keeps the synthetic training stream independent of
directory traversal order and ``DataLoader`` worker count.  A CSV schedule is
the source of truth: each row identifies the source image, look number, Gamma
seed and one of the eight interpolation-free dihedral transforms.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


SUPPORTED_IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}

D4_NAMES = (
    "identity",
    "rot90",
    "rot180",
    "rot270",
    "flip_lr",
    "flip_lr_rot90",
    "flip_lr_rot180",
    "flip_lr_rot270",
)

SOURCE_MANIFEST_FIELDS = (
    "protocol_id",
    "split",
    "class_name",
    "source_id",
    "source_relative_path",
    "selection_rank",
    "parent_split",
    "source_sha256",
    "source_pixel_sha256",
    "source_phash",
    "original_height",
    "original_width",
    "numeric_domain",
)

SCHEDULE_FIELDS = (
    "global_step",
    "source_id",
    "class_name",
    "source_occurrence",
    "L",
    "speckle_seed",
    "d4_id",
    "protocol_id",
    "run_seed",
)

PAIR_MANIFEST_FIELDS = (
    "protocol_id",
    "dataset",
    "split",
    "pair_id",
    "source_id",
    "source_relative_path",
    "class_name",
    "selection_rank",
    "parent_split",
    "global_L",
    "rng_seed",
    "mat_path",
    "clean_path",
    "clean_field",
    "noisy_path",
    "noisy_field",
    "source_sha256",
    "source_pixel_sha256",
    "source_phash",
    "clean_sha256",
    "noisy_sha256",
    "mat_sha256",
    "preclip_max",
    "postclip_max",
    "saturation_rate",
    "original_height",
    "original_width",
    "processed_height",
    "processed_width",
    "size_policy",
    "size_changed",
    "numeric_domain",
)


def load_protocol_config(path: str | Path, smoke: bool = False) -> dict[str, Any]:
    """Load and validate the protocol JSON, applying declared smoke overrides."""

    with Path(path).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    required = {
        "protocol_id",
        "numeric_domain",
        "image_size",
        "looks",
        "nwpu",
        "ucm",
        "schedule",
        "training",
        "validation",
        "real",
        "ams",
        "seeds",
        "synthesis",
    }
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Protocol config is missing keys: {sorted(missing)}")
    if sorted(int(value) for value in config["looks"]) != [1, 2, 4, 8]:
        raise ValueError("ICSPS26-FROZEN-v2 requires looks {1,2,4,8}")
    forbidden = {
        "additional_gaussian_sigma": 0.0,
        "impulse_probability": 0.0,
        "misregistration_alpha": 0.0,
        "stripe_amplitude": 0.0,
    }
    for key, expected in forbidden.items():
        if float(config["synthesis"].get(key, float("nan"))) != expected:
            raise ValueError(f"Gamma-only protocol requires synthesis.{key}={expected}")

    effective = copy.deepcopy(config)
    effective["mode"] = "smoke" if smoke else "formal"
    effective["artifact_protocol_id"] = (
        f'{config["protocol_id"]}-SMOKE' if smoke else config["protocol_id"]
    )
    if smoke:
        smoke_config = config.get("smoke")
        if not isinstance(smoke_config, Mapping):
            raise ValueError("--smoke requested but config.smoke is absent")
        effective["nwpu"].update(
            {
                "expected_classes": int(smoke_config["nwpu_classes"]),
                "expected_per_class": int(smoke_config["nwpu_expected_per_class"]),
                "master_train_per_class": int(smoke_config["nwpu_master_train_per_class"]),
                "master_validation_per_class": int(smoke_config["nwpu_master_validation_per_class"]),
                "train_per_class": int(smoke_config["nwpu_train_per_class"]),
                "validation_per_class": int(smoke_config["nwpu_validation_per_class"]),
            }
        )
        effective["schedule"]["updates"] = int(smoke_config["schedule_updates"])
        effective["training"]["optimizer_updates"] = int(smoke_config["schedule_updates"])
        effective["training"]["validation_interval_updates"] = int(
            smoke_config["validation_interval_updates"]
        )
        effective["ucm"].update(
            {
                "expected_classes": int(smoke_config["ucm_classes"]),
                "expected_per_class": int(smoke_config["ucm_expected_per_class"]),
            }
        )
    _validate_effective_config(effective)
    return effective


def _validate_effective_config(config: Mapping[str, Any]) -> None:
    if str(config["numeric_domain"]) != "linear_normalized_intensity_0_1":
        raise ValueError("The frozen protocol requires linear normalized intensity in [0,1]")
    if tuple(int(value) for value in config["image_size"]) != (256, 256):
        raise ValueError("The frozen protocol requires 256x256 inputs")
    nwpu = config["nwpu"]
    if int(nwpu["master_train_per_class"]) + int(nwpu["master_validation_per_class"]) != int(nwpu["expected_per_class"]):
        raise ValueError("NWPU master train/validation counts must cover every source in the selected pool")
    if int(nwpu["train_per_class"]) > int(nwpu["master_train_per_class"]):
        raise ValueError("NWPU train subset is larger than the master-train pool")
    if int(nwpu["validation_per_class"]) > int(nwpu["master_validation_per_class"]):
        raise ValueError("NWPU validation subset is larger than the master-validation pool")
    updates = int(config["schedule"]["updates"])
    if updates <= 0 or updates % len(config["looks"]) != 0:
        raise ValueError("Schedule updates must be positive and divisible by the number of looks")
    d4_count = int(config["schedule"]["d4_count"])
    if d4_count != len(D4_NAMES) or updates % d4_count != 0:
        raise ValueError("Schedule updates must be divisible equally across all eight D4 transforms")
    training = config["training"]
    if int(training["batch_size"]) != 1:
        raise ValueError("The frozen protocol requires training batch_size=1")
    if int(training["optimizer_updates"]) != updates:
        raise ValueError("training.optimizer_updates must equal schedule.updates")
    interval = int(training["validation_interval_updates"])
    if interval <= 0 or updates % interval:
        raise ValueError("Validation interval must divide optimizer updates")
    if str(config["schedule"]["global_step_indexing"]) != "one_based":
        raise ValueError("The frozen schedule uses one-based global steps")
    real = config["real"]
    expected_real_metrics = {
        "ratio_mean_bias",
        "ratio_acf_sidelobe_energy",
        "sobel_gc_noisy_output",
        "noisy_enl",
        "output_enl",
    }
    expected_forbidden_real_metrics = {"qpsnr_gt16", "qssim_gt16"}
    if (
        real.get("use_gt16") is not False
        or real.get("evaluation_scope") != "no_reference_only"
        or real.get("split_manifest_sha256")
        != "3d2161938861dddd66aed2950160b355bf067143564a35af51f6209dba659ec0"
        or set(real.get("reported_metrics", ())) != expected_real_metrics
        or set(real.get("forbidden_paper_metrics", ()))
        != expected_forbidden_real_metrics
    ):
        raise ValueError("The frozen real-SAR protocol must be the declared no-GT16 branch")
    if int(config["validation"]["pairs_per_source"]) != len(config["looks"]):
        raise ValueError("Validation must contain one fixed pair per source and look")
    if str(nwpu["native_size_policy"]) != "require_exact_256x256_no_resampling":
        raise ValueError("NWPU resampling is forbidden by the frozen protocol")
    canonical_master_hash = str(nwpu["canonical_master_manifest_sha256"]).lower()
    if len(canonical_master_hash) != 64 or any(
        character not in set("0123456789abcdef") for character in canonical_master_hash
    ):
        raise ValueError("NWPU canonical master-manifest SHA-256 is invalid")
    phash_audit = nwpu["cross_dataset_near_duplicate_audit"]
    if (
        str(phash_audit["algorithm"]) != "phash_dct_32x32_low8x8_excluding_dc"
        or int(phash_audit["maximum_hamming_distance"]) != 5
        or str(phash_audit["action"]) != "manual_review_only_no_automatic_exclusion"
    ):
        raise ValueError("Cross-dataset pHash audit differs from ICSPS26-FROZEN-v2")
    ucm_policy = config["ucm"]["size_policy"]
    if str(ucm_policy["name"]) != "identity_if_256_else_bilinear_resize":
        raise ValueError("Unexpected frozen UCM size policy")
    if (int(ucm_policy["target_height"]), int(ucm_policy["target_width"])) != (256, 256):
        raise ValueError("The frozen UCM target size is 256x256")

    optimizer = training["optimizer"]
    loss = training["loss"]
    scheduler = training["scheduler"]
    if (
        str(optimizer["name"]) != "Adam"
        or float(optimizer["learning_rate"]) != 1e-3
        or float(optimizer["weight_decay"]) != 1e-5
        or str(loss["tv_definition"]) != "mean_total_variation"
        or float(loss["lambda_tv"]) != 0.03
    ):
        raise ValueError("Supervised optimizer/loss differs from ICSPS26-FROZEN-v2")
    if (
        str(scheduler["name"]) != "ReduceLROnPlateau"
        or str(scheduler["mode"]) != "min"
        or float(scheduler["factor"]) != 0.5
        or int(scheduler["patience_validation_events"]) != 4
        or float(scheduler["min_lr"]) != 1e-6
        or str(scheduler["monitor"]) != "val_macro_mse"
    ):
        raise ValueError("Supervised scheduler differs from ICSPS26-FROZEN-v2")
    ams = config["ams"]
    ams_optimizer = ams["optimizer"]
    ams_selection = ams["checkpoint_selection"]
    if (
        int(ams["epochs"]) != 8
        or int(ams["batch_size"]) != 1
        or (int(ams["crop_height"]), int(ams["crop_width"])) != (256, 256)
        or str(ams_optimizer["name"]) != "Adam"
        or float(ams_optimizer["learning_rate"]) != 1e-6
        or float(ams_optimizer["weight_decay"]) != 1e-5
        or float(ams["mask"]["ratio"]) != 0.2
        or float(ams["loss"]["lambda_tv"]) != 1e-3
        or set(ams_selection) != {
            "selection_metric", "selection_mode", "candidate_epochs"
        }
        or str(ams_selection["selection_metric"])
        != "fixed_real_validation_masked_loss"
        or str(ams_selection["selection_mode"]) != "min"
        or str(ams_selection["candidate_epochs"]) != "adapted_epochs_1_to_8"
        or str(ams["inference_mask"]) != "none"
    ):
        raise ValueError("AMS settings differ from ICSPS26-FROZEN-v2")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def perceptual_hash_rgb(rgb: np.ndarray) -> str:
    """Return a deterministic 64-bit DCT perceptual hash for audit only."""

    try:
        from scipy.fftpack import dct
    except ImportError as error:  # pragma: no cover - environment issue
        raise RuntimeError("Perceptual-hash audit requires scipy") from error
    value = np.asarray(rgb, dtype=np.uint8)
    if value.ndim != 3 or value.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB for pHash, got {value.shape}")
    gray = Image.fromarray(value, mode="RGB").convert("L")
    resampling = getattr(Image, "Resampling", Image)
    small = np.asarray(gray.resize((32, 32), resampling.LANCZOS), dtype=np.float64)
    coefficients = dct(dct(small, axis=0, norm="ortho"), axis=1, norm="ortho")
    low = coefficients[:8, :8].reshape(-1)
    median = float(np.median(low[1:]))
    bits = low[1:] > median
    packed = 0
    for bit in bits:
        packed = (packed << 1) | int(bool(bit))
    return f"{packed:016x}"


def stable_u64(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little", signed=False)


def stable_id(namespace: str, relative_path: str) -> str:
    return hashlib.sha256(f"{namespace}\x1f{relative_path}".encode("utf-8")).hexdigest()[:24]


def portable_relative_path(value: str | Path) -> str:
    """Return a canonical POSIX relative path and reject path traversal."""

    text = str(value).replace("\\", "/")
    path = PurePosixPath(text)
    if not text or path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ValueError(f"Path is not a portable relative path: {value!r}")
    if path.parts and ":" in path.parts[0]:
        raise ValueError(f"Drive-qualified paths are forbidden in manifests: {value!r}")
    return path.as_posix()


def resolve_portable(root: str | Path, relative_path: str) -> Path:
    root_path = Path(root).resolve()
    relative = portable_relative_path(relative_path)
    resolved = (root_path / Path(*PurePosixPath(relative).parts)).resolve()
    try:
        resolved.relative_to(root_path)
    except ValueError as error:
        raise ValueError(f"Manifest path escapes dataset root: {relative_path!r}") from error
    return resolved


def read_csv_records(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_records(
    path: str | Path,
    fields: Sequence[str],
    records: Iterable[Mapping[str, object]],
) -> None:
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        for record in records:
            writer.writerow({field: record[field] for field in fields})


def _hash_rank(values: Iterable[Any], namespace: str, key) -> list[Any]:
    return sorted(
        values,
        key=lambda value: (stable_u64(namespace, key(value)), key(value)),
    )


def discover_class_images(
    root: str | Path,
    expected_classes: int,
    expected_per_class: int,
    *,
    smoke: bool = False,
    selection_seed: int = 0,
) -> dict[str, list[Path]]:
    """Discover class folders without depending on filesystem traversal order.

    Formal mode requires exact dataset cardinalities.  Smoke mode may point at
    the full datasets; it deterministically selects the requested number of
    classes and source files instead of requiring a separate miniature copy.
    """

    root_path = Path(root).resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root_path}")
    class_dirs = sorted(path for path in root_path.iterdir() if path.is_dir())
    if smoke:
        if len(class_dirs) < expected_classes:
            raise AssertionError(f"Need at least {expected_classes} class folders, found {len(class_dirs)}")
        class_dirs = _hash_rank(class_dirs, f"smoke-classes:{selection_seed}", lambda path: path.name)[:expected_classes]
        class_dirs.sort(key=lambda path: path.name)
    elif len(class_dirs) != expected_classes:
        raise AssertionError(f"Expected {expected_classes} class folders, found {len(class_dirs)}")

    discovered: dict[str, list[Path]] = {}
    for class_dir in class_dirs:
        files = sorted(
            path
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        )
        if smoke:
            if len(files) < expected_per_class:
                raise AssertionError(
                    f"Class {class_dir.name!r} needs at least {expected_per_class} images, found {len(files)}"
                )
            files = _hash_rank(
                files,
                f"smoke-sources:{selection_seed}:{class_dir.name}",
                lambda path: path.name,
            )[:expected_per_class]
            files.sort(key=lambda path: path.name)
        elif len(files) != expected_per_class:
            raise AssertionError(
                f"Class {class_dir.name!r} has {len(files)} images, expected {expected_per_class}"
            )
        discovered[class_dir.name] = files
    return discovered


def _load_rgb_with_metadata(path: Path) -> tuple[np.ndarray, int, int]:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        pixels = np.asarray(rgb, dtype=np.uint8)
    return pixels, height, width


def _fingerprint_nwpu_source(path: Path, image_size: Sequence[int]) -> dict[str, object]:
    pixels, height, width = _load_rgb_with_metadata(path)
    target_height, target_width = (int(image_size[0]), int(image_size[1]))
    if (height, width) != (target_height, target_width):
        raise ValueError(
            f"NWPU source {path} has shape {(height, width)}; the frozen protocol forbids resampling"
        )
    return {
        "source_sha256": sha256_file(path),
        "source_pixel_sha256": sha256_array(pixels),
        "source_phash": perceptual_hash_rgb(pixels),
        "original_height": height,
        "original_width": width,
    }


def _normalise_split(value: object) -> str:
    split = str(value).strip().lower()
    if split == "val":
        split = "validation"
    if split not in {"train", "validation"}:
        raise ValueError(f"Unsupported NWPU master split: {value!r}")
    return split


def load_nwpu_master_manifest(path: str | Path) -> list[dict[str, str]]:
    """Read CSV or legacy JSON master splits and deduplicate repeated L rows."""

    manifest_path = Path(path)
    if manifest_path.suffix.lower() == ".csv":
        raw_records: Iterable[Mapping[str, Any]] = read_csv_records(manifest_path)
    else:
        with manifest_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload.get("records"), list):
            raw_records = payload["records"]
        elif isinstance(payload.get("source_files"), Mapping):
            expanded = []
            source_hashes = payload.get("source_sha256", {})
            for split, values in payload["source_files"].items():
                for relative in values:
                    portable = portable_relative_path(relative)
                    record = {
                        "split": split,
                        "class_name": PurePosixPath(portable).parts[0],
                        "source_relative_path": portable,
                    }
                    if isinstance(source_hashes, Mapping) and portable in source_hashes:
                        record["source_sha256"] = str(source_hashes[portable])
                    expanded.append(record)
            raw_records = expanded
        else:
            raise ValueError("NWPU master manifest needs records[] or source_files{}")

    unique: dict[str, dict[str, str]] = {}
    for row in raw_records:
        relative_raw = row.get("source_relative_path", row.get("source_relative"))
        class_raw = row.get("class_name", row.get("class"))
        if relative_raw is None or class_raw is None or row.get("split") is None:
            raise ValueError("Master manifest rows need source path, class and split")
        relative = portable_relative_path(str(relative_raw))
        record = {
            "source_relative_path": relative,
            "class_name": str(class_raw),
            "split": _normalise_split(row["split"]),
        }
        if row.get("source_sha256"):
            record["source_sha256"] = str(row["source_sha256"]).lower()
        previous = unique.get(relative)
        if previous is not None and previous != record:
            raise ValueError(f"Conflicting master assignments for {relative}")
        unique[relative] = record
    return sorted(unique.values(), key=lambda row: row["source_relative_path"])


def _fallback_master_records(
    source_root: Path,
    discovered: Mapping[str, Sequence[Path]],
    master_train_per_class: int,
    master_validation_per_class: int,
    split_seed: int,
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for class_name, paths in sorted(discovered.items()):
        ranked = _hash_rank(
            paths,
            f"nwpu-master:{split_seed}:{class_name}",
            lambda path: path.relative_to(source_root).as_posix(),
        )
        validation = set(ranked[:master_validation_per_class])
        expected = master_train_per_class + master_validation_per_class
        if len(ranked) != expected:
            raise AssertionError(f"Unexpected source pool for {class_name}: {len(ranked)} != {expected}")
        for path in ranked:
            records.append(
                {
                    "source_relative_path": path.relative_to(source_root).as_posix(),
                    "class_name": class_name,
                    "split": "validation" if path in validation else "train",
                }
            )
    return records


def build_nwpu_source_manifest(
    source_root: str | Path,
    config: Mapping[str, Any],
    master_manifest_path: str | Path | None = None,
) -> list[dict[str, object]]:
    """Select the frozen 20/5-per-class development sources.

    If a repaired master manifest is supplied, its split is authoritative.  A
    deterministic fallback master split is available for fresh mirrors and is
    identified later in the preparation summary.  In both cases the selected
    train and validation content is checked by decoded-pixel and file hashes.
    """

    source_root_path = Path(source_root).resolve()
    nwpu = config["nwpu"]
    discovered = discover_class_images(
        source_root_path,
        int(nwpu["expected_classes"]),
        int(nwpu["expected_per_class"]),
        smoke=config["mode"] == "smoke",
        selection_seed=int(config["seeds"]["subset"]),
    )
    allowed_paths = {
        path.relative_to(source_root_path).as_posix(): class_name
        for class_name, paths in discovered.items()
        for path in paths
    }
    if master_manifest_path is None:
        master_records = _fallback_master_records(
            source_root_path,
            discovered,
            int(nwpu["master_train_per_class"]),
            int(nwpu["master_validation_per_class"]),
            int(config["seeds"]["split"]),
        )
    else:
        if config["mode"] == "formal":
            actual_master_hash = sha256_file(master_manifest_path)
            expected_master_hash = str(nwpu["canonical_master_manifest_sha256"]).lower()
            if actual_master_hash != expected_master_hash:
                raise AssertionError(
                    "NWPU repaired master-manifest hash mismatch: "
                    f"{actual_master_hash} != {expected_master_hash}"
                )
        master_records = load_nwpu_master_manifest(master_manifest_path)
        if config["mode"] == "formal":
            master_paths = {row["source_relative_path"] for row in master_records}
            allowed = set(allowed_paths)
            missing = allowed.difference(master_paths)
            unknown = master_paths.difference(allowed)
            if missing or unknown:
                raise AssertionError(
                    "Formal master manifest must assign every NWPU source exactly once: "
                    f"missing={len(missing)}, unknown={len(unknown)}"
                )
            unhashed = [
                row["source_relative_path"]
                for row in master_records
                if not row.get("source_sha256")
            ]
            if unhashed:
                raise AssertionError(
                    "Formal repaired master manifest must contain source_sha256 "
                    f"for every source; missing={len(unhashed)}"
                )
        master_records = [
            row for row in master_records if row["source_relative_path"] in allowed_paths
        ]

    by_class_split: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in master_records:
        relative = portable_relative_path(row["source_relative_path"])
        class_name = row["class_name"]
        if relative not in allowed_paths:
            continue
        if allowed_paths[relative] != class_name:
            raise ValueError(f"Class/path disagreement for {relative}: {class_name}")
        by_class_split[(class_name, row["split"])].append(row)

    for class_name in discovered:
        for split, expected_master, selected_minimum in (
            (
                "train",
                int(nwpu["master_train_per_class"]),
                int(nwpu["train_per_class"]),
            ),
            (
                "validation",
                int(nwpu["master_validation_per_class"]),
                int(nwpu["validation_per_class"]),
            ),
        ):
            actual = len(by_class_split[(class_name, split)])
            if actual != expected_master:
                raise AssertionError(
                    f"Master split has {actual} {split} sources for {class_name}, "
                    f"expected exactly {expected_master}"
                )
            if actual < selected_minimum:
                raise AssertionError(
                    f"Master split has too few {split} sources for {class_name}: "
                    f"{actual} < {selected_minimum}"
                )

    selected: list[dict[str, object]] = []
    used_file_hashes: dict[str, str] = {}
    used_pixel_hashes: dict[str, str] = {}
    # Train is selected first, then validation explicitly excludes train content.
    for split, per_class in (
        ("train", int(nwpu["train_per_class"])),
        ("validation", int(nwpu["validation_per_class"])),
    ):
        for class_name in sorted(discovered):
            candidates = _hash_rank(
                by_class_split[(class_name, split)],
                f'nwpu-subset:{config["seeds"]["subset"]}:{split}:{class_name}',
                lambda row: row["source_relative_path"],
            )
            class_selected = 0
            for selection_rank, candidate in enumerate(candidates, start=1):
                relative = candidate["source_relative_path"]
                source = resolve_portable(source_root_path, relative)
                if not source.is_file():
                    raise FileNotFoundError(source)
                fingerprint = _fingerprint_nwpu_source(source, config["image_size"])
                file_hash = str(fingerprint["source_sha256"])
                pixel_hash = str(fingerprint["source_pixel_sha256"])
                expected_source_hash = candidate.get("source_sha256")
                if expected_source_hash and file_hash != str(expected_source_hash).lower():
                    raise AssertionError(
                        f"NWPU source differs from repaired master content: {relative}"
                    )
                if file_hash in used_file_hashes or pixel_hash in used_pixel_hashes:
                    # Same-split duplicates are also skipped so the advertised
                    # source count represents unique decoded content.
                    continue
                source_id = stable_id("nwpu", relative)
                selected.append(
                    {
                        "protocol_id": config["artifact_protocol_id"],
                        "split": split,
                        "class_name": class_name,
                        "source_id": source_id,
                        "source_relative_path": relative,
                        "selection_rank": selection_rank,
                        "parent_split": split,
                        **fingerprint,
                        "numeric_domain": config["numeric_domain"],
                    }
                )
                used_file_hashes[file_hash] = split
                used_pixel_hashes[pixel_hash] = split
                class_selected += 1
                if class_selected == per_class:
                    break
            if class_selected != per_class:
                raise RuntimeError(
                    f"Could not select {per_class} unique {split} sources for {class_name}; got {class_selected}"
                )
    selected.sort(key=lambda row: (str(row["split"]), str(row["class_name"]), str(row["source_relative_path"])))
    validate_source_manifest(selected, config)
    return selected


def validate_source_manifest(
    records: Sequence[Mapping[str, object]],
    config: Mapping[str, Any],
    source_root: str | Path | None = None,
    *,
    verify_files: bool = False,
) -> dict[str, object]:
    nwpu = config["nwpu"]
    expected_classes = int(nwpu["expected_classes"])
    expected = {
        "train": int(nwpu["train_per_class"]),
        "validation": int(nwpu["validation_per_class"]),
    }
    counts = Counter((str(row["split"]), str(row["class_name"])) for row in records)
    classes = sorted({str(row["class_name"]) for row in records})
    if len(classes) != expected_classes:
        raise AssertionError(f"Source manifest has {len(classes)} classes, expected {expected_classes}")
    for class_name in classes:
        for split, per_class in expected.items():
            if counts[(split, class_name)] != per_class:
                raise AssertionError(
                    f"{class_name}/{split}: {counts[(split, class_name)]} != {per_class}"
                )
    source_ids = [str(row["source_id"]) for row in records]
    paths = [portable_relative_path(str(row["source_relative_path"])) for row in records]
    if len(source_ids) != len(set(source_ids)):
        raise AssertionError("Duplicate source_id in source manifest")
    if len(paths) != len(set(paths)):
        raise AssertionError("Duplicate source path in source manifest")
    expected_protocol = str(config["artifact_protocol_id"])
    expected_domain = str(config["numeric_domain"])
    expected_height, expected_width = (int(value) for value in config["image_size"])
    hexadecimal = set("0123456789abcdef")
    for row in records:
        if str(row["protocol_id"]) != expected_protocol:
            raise AssertionError(f"Unexpected source protocol_id: {row['protocol_id']}")
        if str(row["numeric_domain"]) != expected_domain:
            raise AssertionError(f"Unexpected source numeric domain: {row['numeric_domain']}")
        if str(row["parent_split"]) != str(row["split"]):
            raise AssertionError(f"Source parent_split mismatch: {row['source_id']}")
        if int(row["selection_rank"]) <= 0:
            raise AssertionError(f"Invalid selection rank: {row['source_id']}")
        if (int(row["original_height"]), int(row["original_width"])) != (
            expected_height,
            expected_width,
        ):
            raise AssertionError(f"Unexpected NWPU source shape: {row['source_id']}")
        for field in ("source_sha256", "source_pixel_sha256"):
            digest = str(row[field]).lower()
            if len(digest) != 64 or any(character not in hexadecimal for character in digest):
                raise AssertionError(f"Invalid {field} for {row['source_id']}")
        source_phash = str(row["source_phash"]).lower()
        if len(source_phash) != 16 or any(character not in hexadecimal for character in source_phash):
            raise AssertionError(f"Invalid source_phash for {row['source_id']}")
    for field in ("source_sha256", "source_pixel_sha256"):
        values = [str(row[field]) for row in records]
        if len(values) != len(set(values)):
            raise AssertionError(f"Duplicate selected source content by {field}")
    train = [row for row in records if str(row["split"]) == "train"]
    validation = [row for row in records if str(row["split"]) == "validation"]
    for field in ("source_sha256", "source_pixel_sha256"):
        overlap = {str(row[field]) for row in train}.intersection(
            str(row[field]) for row in validation
        )
        if overlap:
            raise AssertionError(f"Train/validation {field} overlap: {len(overlap)}")
    if verify_files:
        if source_root is None:
            raise ValueError("source_root is required when verify_files=True")
        for row in records:
            source = resolve_portable(source_root, str(row["source_relative_path"]))
            fingerprint = _fingerprint_nwpu_source(source, config["image_size"])
            for field in ("source_sha256", "source_pixel_sha256", "source_phash"):
                if str(row[field]) != str(fingerprint[field]):
                    raise AssertionError(f"Source hash mismatch for {source}: {field}")
    return {
        "classes": len(classes),
        "train_sources": len(train),
        "validation_sources": len(validation),
        "path_overlap": 0,
        "file_hash_overlap": 0,
        "pixel_hash_overlap": 0,
    }


def _balanced_quota(keys: Sequence[str], total: int, namespace: str) -> dict[str, int]:
    if not keys:
        raise ValueError("Cannot distribute a quota over an empty key set")
    base, remainder = divmod(total, len(keys))
    ranked = _hash_rank(keys, namespace, lambda value: value)
    extras = set(ranked[:remainder])
    return {key: base + int(key in extras) for key in keys}


def _balanced_values(values: Sequence[int], total: int, namespace: str) -> list[int]:
    quotas = _balanced_quota([str(value) for value in values], total, namespace)
    tagged: list[tuple[int, int]] = []
    for value in values:
        tagged.extend((int(value), occurrence) for occurrence in range(quotas[str(value)]))
    tagged.sort(key=lambda item: (stable_u64(namespace, item[0], item[1]), item))
    return [value for value, _ in tagged]


def build_train_schedule(
    train_records: Sequence[Mapping[str, object]],
    *,
    updates: int,
    looks: Sequence[int],
    run_seed: int,
    synthesis_seed: int,
    protocol_id: str,
    d4_count: int = 8,
) -> list[dict[str, object]]:
    """Build a deterministic schedule with simultaneous exact marginal quotas."""

    if updates % len(looks) or updates % d4_count:
        raise ValueError("updates must divide equally across looks and D4")
    by_class: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in train_records:
        if str(row["split"]) != "train":
            raise ValueError("build_train_schedule accepts train records only")
        by_class[str(row["class_name"])].append(row)
    classes = sorted(by_class)
    if not classes:
        raise ValueError("No training sources")
    class_quota = _balanced_quota(classes, updates, f"class-quota:{run_seed}")

    source_tokens: list[tuple[str, int]] = []
    source_lookup: dict[str, Mapping[str, object]] = {}
    for class_name in classes:
        rows = sorted(by_class[class_name], key=lambda row: str(row["source_id"]))
        ids = [str(row["source_id"]) for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate training source in {class_name}")
        source_lookup.update({str(row["source_id"]): row for row in rows})
        source_quota = _balanced_quota(
            ids,
            class_quota[class_name],
            f"source-quota:{run_seed}:{class_name}",
        )
        for source_id in ids:
            source_tokens.extend((source_id, token) for token in range(source_quota[source_id]))
    source_tokens.sort(
        key=lambda item: (stable_u64("schedule-source-order", run_seed, item[0], item[1]), item)
    )
    look_stream = _balanced_values(
        [int(value) for value in looks], updates, f"schedule-look-order:{run_seed}"
    )
    d4_stream = _balanced_values(
        list(range(d4_count)), updates, f"schedule-d4-order:{run_seed}"
    )
    if not (len(source_tokens) == len(look_stream) == len(d4_stream) == updates):
        raise AssertionError("Internal schedule cardinality error")

    occurrences: Counter[str] = Counter()
    schedule: list[dict[str, object]] = []
    for index, ((source_id, _), look, d4_id) in enumerate(
        zip(source_tokens, look_stream, d4_stream), start=1
    ):
        occurrences[source_id] += 1
        occurrence = occurrences[source_id]
        source = source_lookup[source_id]
        speckle_seed = stable_u64(
            synthesis_seed,
            run_seed,
            source_id,
            occurrence,
            look,
        )
        schedule.append(
            {
                "global_step": index,
                "source_id": source_id,
                "class_name": str(source["class_name"]),
                "source_occurrence": occurrence,
                "L": look,
                "speckle_seed": speckle_seed,
                "d4_id": d4_id,
                "protocol_id": protocol_id,
                "run_seed": run_seed,
            }
        )
    validate_train_schedule(
        schedule,
        train_records,
        updates=updates,
        looks=looks,
        run_seed=run_seed,
        synthesis_seed=synthesis_seed,
        protocol_id=protocol_id,
        d4_count=d4_count,
    )
    return schedule


def validate_train_schedule(
    schedule: Sequence[Mapping[str, object]],
    train_records: Sequence[Mapping[str, object]],
    *,
    updates: int,
    looks: Sequence[int],
    run_seed: int,
    synthesis_seed: int,
    protocol_id: str,
    d4_count: int = 8,
) -> dict[str, object]:
    if len(schedule) != updates:
        raise AssertionError(f"Schedule has {len(schedule)} rows, expected {updates}")
    source_lookup = {str(row["source_id"]): row for row in train_records}
    if len(source_lookup) != len(train_records):
        raise AssertionError("Duplicate source IDs in train manifest")
    look_counts: Counter[int] = Counter()
    d4_counts: Counter[int] = Counter()
    source_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    seen_occurrences: Counter[str] = Counter()
    for expected_step, row in enumerate(schedule, start=1):
        if int(row["global_step"]) != expected_step:
            raise AssertionError(f"Non-contiguous global_step at row {expected_step}")
        source_id = str(row["source_id"])
        if source_id not in source_lookup:
            raise AssertionError(f"Unknown schedule source_id: {source_id}")
        source = source_lookup[source_id]
        if str(row["class_name"]) != str(source["class_name"]):
            raise AssertionError(f"Schedule class mismatch for {source_id}")
        look = int(row["L"])
        d4_id = int(row["d4_id"])
        if look not in {int(value) for value in looks} or not 0 <= d4_id < d4_count:
            raise AssertionError(f"Invalid look/D4 at step {expected_step}")
        if int(row["run_seed"]) != run_seed or str(row["protocol_id"]) != protocol_id:
            raise AssertionError(f"Run/protocol mismatch at step {expected_step}")
        seen_occurrences[source_id] += 1
        if int(row["source_occurrence"]) != seen_occurrences[source_id]:
            raise AssertionError(f"Non-sequential source occurrence for {source_id}")
        expected_seed = stable_u64(
            synthesis_seed,
            run_seed,
            source_id,
            seen_occurrences[source_id],
            look,
        )
        if int(row["speckle_seed"]) != expected_seed:
            raise AssertionError(f"Speckle seed mismatch at step {expected_step}")
        look_counts[look] += 1
        d4_counts[d4_id] += 1
        source_counts[source_id] += 1
        class_counts[str(row["class_name"])] += 1
    expected_look = updates // len(looks)
    if set(look_counts) != {int(value) for value in looks} or set(look_counts.values()) != {expected_look}:
        raise AssertionError(f"Look counts are not balanced: {dict(look_counts)}")
    expected_d4 = updates // d4_count
    if set(d4_counts) != set(range(d4_count)) or set(d4_counts.values()) != {expected_d4}:
        raise AssertionError(f"D4 counts are not balanced: {dict(d4_counts)}")
    source_floor, source_remainder = divmod(updates, len(train_records))
    expected_source_values = {source_floor} | ({source_floor + 1} if source_remainder else set())
    if set(source_counts.values()) != expected_source_values:
        raise AssertionError(f"Source counts are not floor/ceil balanced: {set(source_counts.values())}")
    class_floor, class_remainder = divmod(updates, len(class_counts))
    expected_class_values = {class_floor} | ({class_floor + 1} if class_remainder else set())
    if set(class_counts.values()) != expected_class_values:
        raise AssertionError(f"Class counts are not floor/ceil balanced: {set(class_counts.values())}")
    return {
        "rows": updates,
        "look_counts": dict(sorted(look_counts.items())),
        "d4_counts": dict(sorted(d4_counts.items())),
        "source_count_min": min(source_counts.values()),
        "source_count_max": max(source_counts.values()),
        "class_count_min": min(class_counts.values()),
        "class_count_max": max(class_counts.values()),
    }


def clean_intensity_from_rgb(rgb: np.ndarray) -> np.ndarray:
    value = np.asarray(rgb, dtype=np.float32)
    if value.ndim != 3 or value.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB, got {value.shape}")
    gray = (
        0.299 * value[..., 0]
        + 0.587 * value[..., 1]
        + 0.114 * value[..., 2]
    ) / 255.0
    return np.square(np.clip(gray, 0.0, 1.0)).astype(np.float32)


def load_clean_intensity(
    path: str | Path,
    image_size: Sequence[int] = (256, 256),
    *,
    size_policy: str = "require_exact_256x256_no_resampling",
) -> tuple[np.ndarray, dict[str, object]]:
    """Load an optical source and apply the declared spatial-size policy."""

    source = Path(path)
    with Image.open(source) as image:
        rgb_image = image.convert("RGB")
        original_width, original_height = rgb_image.size
        target_height, target_width = (int(image_size[0]), int(image_size[1]))
        changed = (original_height, original_width) != (target_height, target_width)
        if changed:
            if size_policy == "require_exact_256x256_no_resampling":
                raise ValueError(
                    f"Source {source} has shape {(original_height, original_width)}; resizing is forbidden"
                )
            if size_policy != "identity_if_256_else_bilinear_resize":
                raise ValueError(f"Unknown size policy: {size_policy}")
            resampling = getattr(Image, "Resampling", Image)
            rgb_image = rgb_image.resize((target_width, target_height), resampling.BILINEAR)
        rgb = np.asarray(rgb_image, dtype=np.uint8)
    clean = clean_intensity_from_rgb(rgb)
    return clean, {
        "original_height": original_height,
        "original_width": original_width,
        "processed_height": int(clean.shape[0]),
        "processed_width": int(clean.shape[1]),
        "size_policy": size_policy,
        "size_changed": bool(changed),
    }


def synthesize_gamma_only(
    clean: np.ndarray,
    looks: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, float]]:
    """Apply only unit-mean Gamma speckle followed by the frozen [0,1] clip."""

    if int(looks) <= 0:
        raise ValueError("looks must be positive")
    clean_01 = np.asarray(clean, dtype=np.float32)
    if clean_01.ndim != 2 or not np.isfinite(clean_01).all():
        raise ValueError("clean must be a finite two-dimensional array")
    if float(clean_01.min()) < 0.0 or float(clean_01.max()) > 1.0:
        raise ValueError("clean must lie in [0,1]")
    rng = np.random.default_rng(int(seed))
    speckle = rng.gamma(
        shape=float(looks), scale=1.0 / float(looks), size=clean_01.shape
    ).astype(np.float32)
    preclip = clean_01 * speckle
    metadata = {
        "preclip_max": float(np.max(preclip)),
        "postclip_max": float(np.max(np.clip(preclip, 0.0, 1.0))),
        "saturation_rate": float(np.mean(preclip > 1.0)),
    }
    return np.clip(preclip, 0.0, 1.0).astype(np.float32), metadata


def apply_d4(array: np.ndarray, d4_id: int) -> np.ndarray:
    """Apply one of eight paired, interpolation-free square-image transforms."""

    if not 0 <= int(d4_id) < len(D4_NAMES):
        raise ValueError(f"d4_id must be in [0,7], got {d4_id}")
    value = np.asarray(array)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError(f"D4 requires a square 2-D array, got {value.shape}")
    transformed = np.fliplr(value) if int(d4_id) >= 4 else value
    transformed = np.rot90(transformed, k=int(d4_id) % 4)
    return np.ascontiguousarray(transformed)


@lru_cache(maxsize=256)
def _cached_nwpu_clean(path_text: str, height: int, width: int) -> np.ndarray:
    clean, _ = load_clean_intensity(
        path_text,
        (height, width),
        size_policy="require_exact_256x256_no_resampling",
    )
    clean.setflags(write=False)
    return clean


class ICSPSOnlineGammaDataset(Dataset):
    """Replay the frozen online Gamma/D4 training schedule exactly."""

    def __init__(
        self,
        source_root: str | Path,
        source_manifest: str | Path | Sequence[Mapping[str, object]],
        train_schedule: str | Path | Sequence[Mapping[str, object]],
        *,
        image_size: Sequence[int] = (256, 256),
        protocol_id: str | None = None,
        verify_source_hashes: bool = False,
    ) -> None:
        self.source_root = Path(source_root).resolve()
        self.image_size = (int(image_size[0]), int(image_size[1]))
        source_rows = (
            read_csv_records(source_manifest)
            if isinstance(source_manifest, (str, Path))
            else list(source_manifest)
        )
        schedule_rows = (
            read_csv_records(train_schedule)
            if isinstance(train_schedule, (str, Path))
            else list(train_schedule)
        )
        self.sources = {
            str(row["source_id"]): dict(row)
            for row in source_rows
            if str(row["split"]) == "train"
        }
        self.schedule = [dict(row) for row in schedule_rows]
        if not self.sources or not self.schedule:
            raise ValueError("Online dataset needs non-empty train sources and schedule")
        if verify_source_hashes:
            for source in self.sources.values():
                path = resolve_portable(
                    self.source_root, str(source["source_relative_path"])
                )
                expected = str(source.get("source_sha256", "")).lower()
                if len(expected) != 64 or sha256_file(path) != expected:
                    raise AssertionError(f"Training source SHA-256 mismatch: {path}")
        for row in self.schedule:
            if str(row["source_id"]) not in self.sources:
                raise ValueError(f"Schedule references unknown source {row['source_id']}")
            if protocol_id is not None and str(row["protocol_id"]) != protocol_id:
                raise ValueError("Schedule protocol_id does not match the requested protocol")

    def __len__(self) -> int:
        return len(self.schedule)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.schedule[index]
        source = self.sources[str(row["source_id"])]
        path = resolve_portable(self.source_root, str(source["source_relative_path"]))
        clean = np.array(
            _cached_nwpu_clean(str(path), self.image_size[0], self.image_size[1]),
            copy=True,
        )
        noisy, synthesis = synthesize_gamma_only(
            clean, int(row["L"]), int(row["speckle_seed"])
        )
        clean = apply_d4(clean, int(row["d4_id"]))
        noisy = apply_d4(noisy, int(row["d4_id"]))
        return {
            "noisy": torch.from_numpy(noisy[None, ...]),
            "clean": torch.from_numpy(clean[None, ...]),
            "global_step": int(row["global_step"]),
            "source_id": str(row["source_id"]),
            "class_name": str(row["class_name"]),
            "source_occurrence": int(row["source_occurrence"]),
            "L": int(row["L"]),
            # Keep the full uint64 seed losslessly without asking PyTorch's
            # default collate function to create an overflowing signed int64.
            # Trainers can recover it with ``int(batch["speckle_seed"][0])``.
            "speckle_seed": str(row["speckle_seed"]),
            "d4_id": int(row["d4_id"]),
            "preclip_max": synthesis["preclip_max"],
            "postclip_max": synthesis["postclip_max"],
            "saturation_rate": synthesis["saturation_rate"],
        }


class ICSPSFixedPairDataset(Dataset):
    """Read frozen NWPU-validation or UCM-test pairs from manifest MAT files.

    ``pair_root`` is the root against which every manifest path is resolved;
    absolute paths and traversal are rejected by :func:`resolve_portable`.
    Clean and noisy arrays may live in the same MAT file (the normal frozen
    layout) or in separate MAT files, as declared by their respective fields.
    """

    def __init__(
        self,
        pair_root: str | Path,
        pair_manifest: str | Path | Sequence[Mapping[str, object]],
        *,
        protocol_id: str | None = None,
        verify_hashes: bool = False,
    ) -> None:
        self.pair_root = Path(pair_root).resolve()
        rows = (
            read_csv_records(pair_manifest)
            if isinstance(pair_manifest, (str, Path))
            else list(pair_manifest)
        )
        if not rows:
            raise ValueError("Fixed-pair dataset needs a non-empty manifest")
        self.pairs = [dict(row) for row in rows]
        self.verify_hashes = bool(verify_hashes)
        for row in self.pairs:
            for field in ("mat_path", "clean_path", "noisy_path"):
                portable_relative_path(str(row[field]))
            if protocol_id is not None and str(row["protocol_id"]) != protocol_id:
                raise ValueError("Pair protocol_id does not match the requested protocol")
            if self.verify_hashes:
                if not (
                    str(row["mat_path"])
                    == str(row["clean_path"])
                    == str(row["noisy_path"])
                ):
                    raise ValueError(
                        "Hash-verified fixed pairs require mat_path, clean_path, and "
                        "noisy_path to identify the same self-contained MAT file"
                    )
                for field in ("mat_sha256", "clean_sha256", "noisy_sha256"):
                    value = str(row.get(field, "")).lower()
                    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                        raise ValueError(
                            f"Fixed pair {row.get('pair_id')} has invalid {field}"
                        )

    def __len__(self) -> int:
        return len(self.pairs)

    @staticmethod
    def _coerce_mat_field(
        payload: Mapping[str, object], path: Path, field: str
    ) -> np.ndarray:
        if field not in payload:
            raise KeyError(f"MAT field {field!r} is absent from {path}")
        array = np.asarray(payload[field], dtype=np.float32).squeeze()
        if array.ndim != 2 or not np.isfinite(array).all():
            raise ValueError(f"MAT field {field!r} in {path} is not a finite 2-D array")
        if float(array.min()) < 0.0 or float(array.max()) > 1.0:
            raise ValueError(f"MAT field {field!r} in {path} is outside [0,1]")
        return np.ascontiguousarray(array)

    @classmethod
    def _load_mat_field(cls, path: Path, field: str) -> np.ndarray:
        try:
            from scipy.io import loadmat
        except ImportError as error:  # pragma: no cover - environment issue
            raise RuntimeError("ICSPSFixedPairDataset requires scipy") from error
        payload = loadmat(path, variable_names=[field])
        return cls._coerce_mat_field(payload, path, field)

    @classmethod
    def _load_verified_pair(
        cls, path: Path, row: Mapping[str, object]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Hash and decode a self-contained pair with a single file read."""

        try:
            from scipy.io import loadmat
        except ImportError as error:  # pragma: no cover - environment issue
            raise RuntimeError("ICSPSFixedPairDataset requires scipy") from error
        encoded = path.read_bytes()
        actual_file_hash = hashlib.sha256(encoded).hexdigest()
        if actual_file_hash != str(row["mat_sha256"]).lower():
            raise AssertionError(f"Fixed-pair MAT SHA-256 mismatch: {path}")
        clean_field = str(row["clean_field"])
        noisy_field = str(row["noisy_field"])
        payload = loadmat(
            io.BytesIO(encoded), variable_names=sorted({clean_field, noisy_field})
        )
        clean = cls._coerce_mat_field(payload, path, clean_field)
        noisy = cls._coerce_mat_field(payload, path, noisy_field)
        if sha256_array(clean) != str(row["clean_sha256"]).lower():
            raise AssertionError(f"Fixed-pair clean-array SHA-256 mismatch: {path}")
        if sha256_array(noisy) != str(row["noisy_sha256"]).lower():
            raise AssertionError(f"Fixed-pair noisy-array SHA-256 mismatch: {path}")
        return clean, noisy

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.pairs[index]
        clean_path = resolve_portable(self.pair_root, str(row["clean_path"]))
        noisy_path = resolve_portable(self.pair_root, str(row["noisy_path"]))
        clean_field = str(row["clean_field"])
        noisy_field = str(row["noisy_field"])
        if self.verify_hashes:
            clean, noisy = self._load_verified_pair(clean_path, row)
        elif clean_path == noisy_path:
            try:
                from scipy.io import loadmat
            except ImportError as error:  # pragma: no cover - environment issue
                raise RuntimeError("ICSPSFixedPairDataset requires scipy") from error
            payload = loadmat(
                clean_path, variable_names=sorted({clean_field, noisy_field})
            )
            clean = self._coerce_mat_field(payload, clean_path, clean_field)
            noisy = self._coerce_mat_field(payload, noisy_path, noisy_field)
        else:
            clean = self._load_mat_field(clean_path, clean_field)
            noisy = self._load_mat_field(noisy_path, noisy_field)
        if clean.shape != noisy.shape:
            raise ValueError(
                f"Fixed pair {row['pair_id']} has mismatched shapes: {clean.shape} vs {noisy.shape}"
            )
        return {
            "noisy": torch.from_numpy(noisy[None, ...]),
            "clean": torch.from_numpy(clean[None, ...]),
            "pair_id": str(row["pair_id"]),
            "source_id": str(row["source_id"]),
            "class_name": str(row["class_name"]),
            "L": int(row["global_L"]),
        }


def validate_pair_manifest(
    records: Sequence[Mapping[str, object]],
    *,
    expected_dataset: str,
    expected_classes: int,
    expected_per_class_per_look: int,
    looks: Sequence[int],
    output_root: str | Path | None = None,
    verify_file_hashes: bool = False,
    synthesis_seed: int | None = None,
) -> dict[str, object]:
    expected_looks = {int(value) for value in looks}
    counts = Counter(
        (str(row["class_name"]), int(row["global_L"])) for row in records
    )
    classes = sorted({str(row["class_name"]) for row in records})
    if len(classes) != expected_classes:
        raise AssertionError(f"Pair manifest has {len(classes)} classes, expected {expected_classes}")
    for row in records:
        if str(row["dataset"]) != expected_dataset:
            raise AssertionError(f"Unexpected pair dataset: {row['dataset']}")
        portable_relative_path(str(row["source_relative_path"]))
        for field in ("mat_path", "clean_path", "noisy_path"):
            portable_relative_path(str(row[field]))
        if int(row["global_L"]) not in expected_looks:
            raise AssertionError(f"Unexpected look: {row['global_L']}")
        if int(row["selection_rank"]) <= 0:
            raise AssertionError(f"Invalid pair selection rank: {row['pair_id']}")
        if str(row["parent_split"]) != str(row["split"]):
            raise AssertionError(f"Pair parent_split mismatch: {row['pair_id']}")
    for class_name in classes:
        for look in expected_looks:
            if counts[(class_name, look)] != expected_per_class_per_look:
                raise AssertionError(
                    f"{expected_dataset}/{class_name}/L{look}: "
                    f"{counts[(class_name, look)]} != {expected_per_class_per_look}"
                )
    pair_ids = [str(row["pair_id"]) for row in records]
    mat_paths = [str(row["mat_path"]) for row in records]
    if len(pair_ids) != len(set(pair_ids)) or len(mat_paths) != len(set(mat_paths)):
        raise AssertionError("Pair IDs and MAT paths must be unique")
    source_look_keys = [
        (str(row["source_id"]), int(row["global_L"])) for row in records
    ]
    if len(source_look_keys) != len(set(source_look_keys)):
        raise AssertionError("Each source/look combination must occur exactly once")
    source_metadata: dict[str, tuple[str, str, str, str, str]] = {}
    source_looks: dict[str, set[int]] = defaultdict(set)
    source_clean_hashes: dict[str, set[str]] = defaultdict(set)
    hexadecimal = set("0123456789abcdef")
    for row in records:
        source_id = str(row["source_id"])
        metadata = (
            str(row["class_name"]),
            str(row["source_relative_path"]),
            str(row["source_sha256"]),
            str(row["source_pixel_sha256"]),
            str(row["source_phash"]),
        )
        if source_id in source_metadata and source_metadata[source_id] != metadata:
            raise AssertionError(f"Inconsistent metadata across looks for {source_id}")
        source_metadata[source_id] = metadata
        source_looks[source_id].add(int(row["global_L"]))
        source_clean_hashes[source_id].add(str(row["clean_sha256"]))
        if not (
            str(row["mat_path"]) == str(row["clean_path"]) == str(row["noisy_path"])
        ):
            raise AssertionError(
                f"Frozen pair must keep clean/noisy in one MAT file: {row['pair_id']}"
            )
        if not str(row["clean_field"]) or not str(row["noisy_field"]):
            raise AssertionError(f"Empty MAT field name for pair {row['pair_id']}")
        for field in (
            "source_sha256",
            "source_pixel_sha256",
            "clean_sha256",
            "noisy_sha256",
            "mat_sha256",
        ):
            digest = str(row[field]).lower()
            if len(digest) != 64 or any(character not in hexadecimal for character in digest):
                raise AssertionError(f"Invalid {field} for pair {row['pair_id']}")
        source_phash = str(row["source_phash"]).lower()
        if len(source_phash) != 16 or any(character not in hexadecimal for character in source_phash):
            raise AssertionError(f"Invalid source_phash for pair {row['pair_id']}")
        preclip_max = float(row["preclip_max"])
        postclip_max = float(row["postclip_max"])
        saturation_rate = float(row["saturation_rate"])
        if preclip_max < 0.0 or not 0.0 <= postclip_max <= 1.0:
            raise AssertionError(f"Invalid clip maxima for pair {row['pair_id']}")
        if abs(postclip_max - min(preclip_max, 1.0)) > 1e-6:
            raise AssertionError(f"Inconsistent postclip_max for pair {row['pair_id']}")
        if not 0.0 <= saturation_rate <= 1.0:
            raise AssertionError(f"Invalid saturation rate for pair {row['pair_id']}")
        if synthesis_seed is not None:
            expected_seed = stable_u64(
                int(synthesis_seed), source_id, int(row["global_L"])
            )
            if int(row["rng_seed"]) != expected_seed:
                raise AssertionError(f"Fixed-pair RNG seed mismatch: {row['pair_id']}")
    if any(actual != expected_looks for actual in source_looks.values()):
        raise AssertionError("Every fixed-pair source must have the complete look grid")
    expected_sources = expected_classes * expected_per_class_per_look
    if len(source_metadata) != expected_sources:
        raise AssertionError(
            f"Pair manifest has {len(source_metadata)} sources, expected {expected_sources}"
        )
    if any(len(hashes) != 1 for hashes in source_clean_hashes.values()):
        raise AssertionError("Clean content changed across the fixed look grid")
    if output_root is not None:
        for row in records:
            for field in ("mat_path", "clean_path", "noisy_path"):
                if not resolve_portable(output_root, str(row[field])).is_file():
                    raise FileNotFoundError(resolve_portable(output_root, str(row[field])))
    if verify_file_hashes:
        if output_root is None:
            raise ValueError("output_root is required when verify_file_hashes=True")
        try:
            from scipy.io import loadmat
        except ImportError as error:  # pragma: no cover - environment issue
            raise RuntimeError("Semantic MAT verification requires scipy") from error
        for row in records:
            mat_path = resolve_portable(output_root, str(row["mat_path"]))
            if sha256_file(mat_path) != str(row["mat_sha256"]):
                raise AssertionError(f"MAT hash mismatch: {mat_path}")
            clean_field = str(row["clean_field"])
            noisy_field = str(row["noisy_field"])
            payload = loadmat(
                mat_path,
                variable_names=[clean_field, noisy_field, "global_L", "rng_seed"],
            )
            required_fields = {clean_field, noisy_field, "global_L", "rng_seed"}
            missing_fields = required_fields.difference(payload)
            if missing_fields:
                raise AssertionError(
                    f"MAT fields missing from {mat_path}: {sorted(missing_fields)}"
                )
            clean = np.asarray(payload[clean_field], dtype=np.float32).squeeze()
            noisy = np.asarray(payload[noisy_field], dtype=np.float32).squeeze()
            if clean.ndim != 2 or noisy.ndim != 2 or clean.shape != noisy.shape:
                raise AssertionError(f"Invalid fixed-pair array shapes in {mat_path}")
            if not np.isfinite(clean).all() or not np.isfinite(noisy).all():
                raise AssertionError(f"Non-finite fixed-pair values in {mat_path}")
            if (
                float(clean.min()) < 0.0
                or float(clean.max()) > 1.0
                or float(noisy.min()) < 0.0
                or float(noisy.max()) > 1.0
            ):
                raise AssertionError(f"Fixed-pair arrays outside [0,1] in {mat_path}")
            if sha256_array(clean) != str(row["clean_sha256"]):
                raise AssertionError(f"Clean array hash mismatch: {mat_path}")
            if sha256_array(noisy) != str(row["noisy_sha256"]):
                raise AssertionError(f"Noisy array hash mismatch: {mat_path}")
            stored_look = int(np.asarray(payload["global_L"]).reshape(-1)[0])
            stored_seed = int(np.asarray(payload["rng_seed"]).reshape(-1)[0])
            if stored_look != int(row["global_L"]) or stored_seed != int(row["rng_seed"]):
                raise AssertionError(f"MAT metadata mismatch: {mat_path}")
            regenerated_noisy, regenerated_metadata = synthesize_gamma_only(
                clean, stored_look, stored_seed
            )
            if not np.array_equal(regenerated_noisy, noisy):
                raise AssertionError(f"Noisy array is not the frozen Gamma replay: {mat_path}")
            for field in ("preclip_max", "postclip_max", "saturation_rate"):
                if abs(float(row[field]) - float(regenerated_metadata[field])) > 1e-12:
                    raise AssertionError(f"Synthesis metadata mismatch ({field}): {mat_path}")
    synthesis_by_look: dict[str, dict[str, float | int]] = {}
    for look in sorted(expected_looks):
        look_rows = [row for row in records if int(row["global_L"]) == look]
        preclip = np.asarray(
            [float(row["preclip_max"]) for row in look_rows], dtype=np.float64
        )
        saturation = np.asarray(
            [float(row["saturation_rate"]) for row in look_rows], dtype=np.float64
        )
        synthesis_by_look[str(look)] = {
            "pairs": len(look_rows),
            "preclip_max_min": float(np.min(preclip)),
            "preclip_max_max": float(np.max(preclip)),
            "preclip_max_mean": float(np.mean(preclip)),
            "saturation_rate_min": float(np.min(saturation)),
            "saturation_rate_max": float(np.max(saturation)),
            "saturation_rate_mean": float(np.mean(saturation)),
        }
    return {
        "dataset": expected_dataset,
        "classes": len(classes),
        "pairs": len(records),
        "semantic_mat_files_verified": bool(verify_file_hashes),
        "per_look": {
            str(look): sum(count for (class_name, actual), count in counts.items() if actual == look)
            for look in sorted(expected_looks)
        },
        "synthesis_by_L": synthesis_by_look,
    }
