"""Build and verify a non-destructive BSDS500 train/val/test manifest.

The historical synthetic-data folders in this repository mix official BSDS500
source splits: ``train/`` contains trn, val, and tst files, while ``val/``
contains the other half of val.  This script assigns samples from their filename
prefix instead of their current folder, so the official test images never enter
training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PREFIX_TO_SPLIT = {"trn": "train", "val": "val", "tst": "test"}
EXPECTED_COUNTS = {"train": 200, "val": 100, "test": 200}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_split(filename: str) -> str:
    prefix = Path(filename).stem.split("_", 1)[0].lower()
    if prefix not in PREFIX_TO_SPLIT:
        raise ValueError(f"Unknown BSDS500 source prefix in {filename!r}")
    return PREFIX_TO_SPLIT[prefix]


def discover_files(dataset_root: Path) -> list[Path]:
    files = sorted(path for path in dataset_root.rglob("*.mat") if path.is_file())
    if not files:
        raise RuntimeError(f"No .mat files found below {dataset_root}")
    return files


def build_manifest(dataset_root: Path, include_hashes: bool = True) -> dict[str, Any]:
    dataset_root = dataset_root.resolve()
    files_by_split: dict[str, list[str]] = {key: [] for key in EXPECTED_COUNTS}
    seen_names: set[str] = set()
    hashes: dict[str, str] = {}

    for path in discover_files(dataset_root):
        relative = path.relative_to(dataset_root).as_posix()
        if path.name in seen_names:
            raise AssertionError(f"Duplicate BSDS500 source filename: {path.name}")
        seen_names.add(path.name)
        files_by_split[source_split(path.name)].append(relative)
        if include_hashes:
            hashes[relative] = sha256_file(path)

    for paths in files_by_split.values():
        paths.sort()
    counts = {split: len(paths) for split, paths in files_by_split.items()}
    if counts != EXPECTED_COUNTS:
        raise AssertionError(f"Unexpected BSDS500 split counts: {counts}")

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "BSDS500 synthetic SAR",
        "split_rule": "official source prefix: trn->train, val->val, tst->test",
        "counts": counts,
        "files": files_by_split,
    }
    if include_hashes:
        manifest["file_sha256"] = dict(sorted(hashes.items()))
    return manifest


def verify_manifest(dataset_root: Path, manifest: dict[str, Any], verify_hashes: bool = False) -> dict[str, Any]:
    dataset_root = dataset_root.resolve()
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != set(EXPECTED_COUNTS):
        raise AssertionError("Manifest must contain exactly train/val/test file lists")

    assigned: list[str] = []
    for split, expected_count in EXPECTED_COUNTS.items():
        paths = files[split]
        if len(paths) != expected_count:
            raise AssertionError(f"{split} has {len(paths)} files, expected {expected_count}")
        for relative in paths:
            path = (dataset_root / relative).resolve()
            try:
                path.relative_to(dataset_root)
            except ValueError as error:
                raise AssertionError(f"Manifest path escapes dataset root: {relative}") from error
            if not path.is_file():
                raise FileNotFoundError(path)
            if source_split(path.name) != split:
                raise AssertionError(f"Source split mismatch: {relative} is listed under {split}")
            assigned.append(relative)

    duplicates = sorted(name for name, count in Counter(assigned).items() if count != 1)
    discovered = {path.relative_to(dataset_root).as_posix() for path in discover_files(dataset_root)}
    assigned_set = set(assigned)
    missing = sorted(discovered - assigned_set)
    unexpected = sorted(assigned_set - discovered)
    if duplicates or missing or unexpected or len(assigned) != len(discovered):
        raise AssertionError(
            f"Manifest is not a one-to-one partition: duplicates={len(duplicates)}, "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )

    checked_hashes = 0
    if verify_hashes:
        expected_hashes = manifest.get("file_sha256")
        if not isinstance(expected_hashes, dict) or set(expected_hashes) != assigned_set:
            raise AssertionError("Complete file_sha256 mapping is required for hash verification")
        for relative in assigned:
            if sha256_file(dataset_root / relative) != expected_hashes[relative]:
                raise AssertionError(f"SHA-256 mismatch: {relative}")
            checked_hashes += 1

    return {
        "valid": True,
        "counts": {split: len(files[split]) for split in EXPECTED_COUNTS},
        "total": len(assigned),
        "duplicates": len(duplicates),
        "missing": len(missing),
        "unexpected": len(unexpected),
        "hashes_checked": checked_hashes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="bsds500_synthetic_dataset")
    parser.add_argument(
        "--output", default="bsds500_synthetic_dataset/official_split_manifest.json"
    )
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--skip-hashes", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.dataset_root).resolve()
    output = Path(args.output).resolve()
    if args.verify_only:
        with output.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    else:
        manifest = build_manifest(root, include_hashes=not args.skip_hashes)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
    result = verify_manifest(root, manifest, verify_hashes=not args.skip_hashes)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
