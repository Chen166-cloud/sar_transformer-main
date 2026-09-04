"""Generic verification for paired synthetic-SAR dataset manifests."""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_paired_sar_manifest(
    dataset_root: str | Path,
    manifest: dict[str, Any],
    required_splits: Iterable[str] = ("train", "val"),
    verify_hashes: bool = False,
    require_exact_partition: bool = True,
    verify_files: bool = True,
) -> dict[str, Any]:
    root = Path(dataset_root).resolve()
    files = manifest.get("files")
    required = tuple(required_splits)
    if not isinstance(files, dict) or any(split not in files for split in required):
        raise AssertionError(f"Manifest is missing required splits: {required}")

    assigned: list[str] = []
    counts: dict[str, int] = {}
    for split, relative_paths in files.items():
        if not isinstance(relative_paths, list):
            raise AssertionError(f"Manifest split {split!r} is not a list")
        counts[split] = len(relative_paths)
        declared_count = manifest.get("counts", {}).get(split)
        if declared_count is not None and int(declared_count) != len(relative_paths):
            raise AssertionError(
                f"Declared count mismatch for {split}: {declared_count} != {len(relative_paths)}"
            )
        for relative in relative_paths:
            if not isinstance(relative, str) or not relative.lower().endswith(".mat"):
                raise AssertionError(f"Invalid paired-SAR path: {relative!r}")
            relative_path = Path(relative)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise AssertionError(f"Manifest path escapes dataset root: {relative}")
            if verify_files or verify_hashes:
                path = (root / relative_path).resolve()
                try:
                    path.relative_to(root)
                except ValueError as error:
                    raise AssertionError(f"Manifest path escapes dataset root: {relative}") from error
                if not path.is_file():
                    raise FileNotFoundError(path)
            assigned.append(Path(relative).as_posix())

    duplicates = sorted(name for name, count in Counter(assigned).items() if count != 1)
    if duplicates:
        raise AssertionError(f"Duplicate manifest paths: {duplicates[:5]}")

    global_looks = manifest.get("global_L_by_file")
    if global_looks is not None:
        source_files = manifest.get("source_files", {})
        source_hashes = manifest.get("source_sha256", {})
        if source_hashes and "train" in source_files and "val" in source_files:
            train_hashes = {source_hashes[path] for path in source_files["train"]}
            val_hashes = {source_hashes[path] for path in source_files["val"]}
            if train_hashes & val_hashes:
                raise AssertionError("Exact source-image content leaks across train/validation")
        if not isinstance(global_looks, dict) or set(global_looks) != set(assigned):
            raise AssertionError(
                "global_L_by_file must map every manifest path exactly once"
            )
        invalid_looks = {
            relative: looks
            for relative, looks in global_looks.items()
            if int(looks) not in (1, 2, 4, 8)
        }
        if invalid_looks:
            raise AssertionError(f"Invalid global L values: {list(invalid_looks.items())[:5]}")

        validation_sets = manifest.get("validation_sets")
        if validation_sets is not None:
            declared_validation: list[str] = []
            for label, definition in validation_sets.items():
                if not isinstance(definition, dict) or "files" not in definition:
                    raise AssertionError(f"Invalid validation set definition: {label}")
                looks = int(definition.get("global_L"))
                paths = definition["files"]
                if int(definition.get("count", -1)) != len(paths):
                    raise AssertionError(f"Validation count mismatch for {label}")
                if any(int(global_looks[path]) != looks for path in paths):
                    raise AssertionError(f"Validation L mismatch for {label}")
                declared_validation.extend(paths)
            if sorted(declared_validation) != sorted(files.get("val", [])):
                raise AssertionError("validation_sets must exactly partition files['val']")

    missing: list[str] = []
    unexpected: list[str] = []
    if require_exact_partition:
        if not verify_files:
            raise ValueError("require_exact_partition=True requires verify_files=True")
        discovered = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*.mat")
            if path.is_file()
        }
        assigned_set = set(assigned)
        missing = sorted(discovered - assigned_set)
        unexpected = sorted(assigned_set - discovered)
        if missing or unexpected or len(discovered) != len(assigned):
            raise AssertionError(
                "Manifest is not an exact partition: "
                f"unlisted={len(missing)}, missing_on_disk={len(unexpected)}"
            )

    hashes_checked = 0
    if verify_hashes:
        expected_hashes = manifest.get("file_sha256")
        if not isinstance(expected_hashes, dict) or set(expected_hashes) != set(assigned):
            raise AssertionError("Complete file_sha256 mapping is required")
        for relative in assigned:
            if sha256_file(root / relative) != expected_hashes[relative]:
                raise AssertionError(f"SHA-256 mismatch: {relative}")
            hashes_checked += 1

    return {
        "valid": True,
        "dataset": manifest.get("dataset"),
        "counts": counts,
        "total": len(assigned),
        "duplicates": len(duplicates),
        "unlisted_files": len(missing),
        "missing_files": len(unexpected),
        "hashes_checked": hashes_checked,
    }
