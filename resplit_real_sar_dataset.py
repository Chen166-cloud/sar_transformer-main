"""Create and verify a non-destructive, parent-image grouped real-SAR split.

Historical train/val/test folders are treated as a source pool only. Files are
never moved by this script; canonical loaders consume the generated manifest.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


SPLITS = ("train", "val", "test")
PARENT_PATTERN = re.compile(
    r"^(?P<parent_row>-?\d+)_(?P<parent_col>-?\d+)_y-?\d+_x-?\d+\.mat$",
    re.IGNORECASE,
)


def parent_group_id(filename: str) -> str:
    match = PARENT_PATTERN.match(Path(filename).name)
    if not match:
        raise ValueError(
            f"Cannot derive 512x512 parent group from filename {filename!r}; "
            "expected '<row>_<col>_y<offset>_x<offset>.mat'."
        )
    return f"{match.group('parent_row')}_{match.group('parent_col')}"


def collect_source_files(dataset_dir: Path) -> List[Tuple[str, str]]:
    """Return (relative_path, historical_split) records from the source pool."""
    records: List[Tuple[str, str]] = []
    seen_names: Dict[str, str] = {}
    for split in SPLITS:
        directory = dataset_dir / split
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.mat")):
            if path.name in seen_names:
                raise ValueError(
                    f"Duplicate basename {path.name!r} in {seen_names[path.name]} and {path}"
                )
            seen_names[path.name] = str(path)
            records.append((path.relative_to(dataset_dir).as_posix(), split))
    if not records:
        raise ValueError(f"No .mat files found below {dataset_dir}")
    return records


def _split_counts(total: int, ratios: Sequence[int]) -> List[int]:
    if len(ratios) != 3 or any(value <= 0 for value in ratios):
        raise ValueError("ratios must contain three positive integers")
    denominator = sum(ratios)
    counts = [int(total * value / denominator) for value in ratios[:-1]]
    counts.append(total - sum(counts))
    return counts


def grouped_split(
    records: Sequence[Tuple[str, str]], seed: int, ratios: Sequence[int]
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    files_by_group: Dict[str, List[str]] = defaultdict(list)
    for relative_path, _ in records:
        files_by_group[parent_group_id(relative_path)].append(relative_path)

    groups = sorted(files_by_group)
    random.Random(seed).shuffle(groups)
    train_count, val_count, _ = _split_counts(len(groups), ratios)
    groups_by_split = {
        "train": groups[:train_count],
        "val": groups[train_count : train_count + val_count],
        "test": groups[train_count + val_count :],
    }
    files_by_split = {
        split: sorted(
            path
            for group in groups_by_split[split]
            for path in files_by_group[group]
        )
        for split in SPLITS
    }
    groups_by_split = {split: sorted(values) for split, values in groups_by_split.items()}
    return files_by_split, groups_by_split


def group_intersections(groups_by_split: Mapping[str, Iterable[str]]) -> Dict[str, List[str]]:
    sets = {split: set(groups_by_split[split]) for split in SPLITS}
    return {
        "train_val": sorted(sets["train"] & sets["val"]),
        "train_test": sorted(sets["train"] & sets["test"]),
        "val_test": sorted(sets["val"] & sets["test"]),
        "train_val_test": sorted(sets["train"] & sets["val"] & sets["test"]),
    }


def assert_group_disjoint(groups_by_split: Mapping[str, Iterable[str]]) -> None:
    intersections = group_intersections(groups_by_split)
    leaked = {name: values for name, values in intersections.items() if values}
    if leaked:
        counts = {name: len(values) for name, values in leaked.items()}
        raise AssertionError(f"Parent-group leakage detected: {counts}")


def historical_leakage(records: Sequence[Tuple[str, str]]) -> Dict[str, object]:
    groups = {split: set() for split in SPLITS}
    files = {split: [] for split in SPLITS}
    for relative_path, split in records:
        group = parent_group_id(relative_path)
        groups[split].add(group)
        files[split].append((relative_path, group))
    intersections = group_intersections(groups)
    train_groups = groups["train"]
    test_with_train_sibling = sum(group in train_groups for _, group in files["test"])
    return {
        "patch_counts": {split: len(files[split]) for split in SPLITS},
        "group_counts": {split: len(groups[split]) for split in SPLITS},
        "intersection_counts": {
            name: len(values) for name, values in intersections.items()
        },
        "test_patches_with_train_parent": test_with_train_sibling,
        "test_patch_count": len(files["test"]),
    }


def build_manifest(dataset_dir: Path, seed: int, ratios: Sequence[int]) -> Tuple[dict, dict]:
    records = collect_source_files(dataset_dir)
    files_by_split, groups_by_split = grouped_split(records, seed, ratios)
    assert_group_disjoint(groups_by_split)

    all_source_files = sorted(path for path, _ in records)
    all_assigned_files = sorted(path for values in files_by_split.values() for path in values)
    if all_source_files != all_assigned_files:
        raise AssertionError("Grouped assignment is not an exact partition of the source pool")

    created_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "schema_version": 2,
        "created_at_utc": created_at,
        "dataset_root": dataset_dir.name,
        "seed": seed,
        "ratios": dict(zip(SPLITS, ratios)),
        "group_definition": {
            "unit": "512x512_parent_image",
            "filename_regex": PARENT_PATTERN.pattern,
            "group_id": "<parent_row>_<parent_col>",
        },
        "counts": {
            "total_files": len(records),
            "total_groups": len(set(parent_group_id(path) for path, _ in records)),
            "files": {split: len(files_by_split[split]) for split in SPLITS},
            "groups": {split: len(groups_by_split[split]) for split in SPLITS},
        },
        "files": files_by_split,
        "groups": groups_by_split,
    }
    intersections = group_intersections(groups_by_split)
    leakage_report = {
        "schema_version": 1,
        "created_at_utc": created_at,
        "manifest": f"real_split_grouped_seed{seed}.json",
        "group_definition": manifest["group_definition"],
        "new_split": {
            "intersection_counts": {
                name: len(values) for name, values in intersections.items()
            },
            "intersections": intersections,
            "all_groups_disjoint": not any(intersections.values()),
            "all_source_files_assigned_once": all_source_files == all_assigned_files,
        },
        "historical_patch_random_split": historical_leakage(records),
    }
    return manifest, leakage_report


def verify_manifest(dataset_dir: Path, manifest: Mapping[str, object]) -> Dict[str, object]:
    files_by_split = manifest["files"]
    groups_by_split = manifest["groups"]
    if not isinstance(files_by_split, dict) or not isinstance(groups_by_split, dict):
        raise TypeError("Manifest files/groups must be objects")
    assert_group_disjoint(groups_by_split)
    listed = [path for split in SPLITS for path in files_by_split[split]]
    missing = sorted(path for path in listed if not (dataset_dir / path).is_file())
    duplicates = sorted(path for path in set(listed) if listed.count(path) > 1)
    group_mismatches = []
    for split in SPLITS:
        allowed = set(groups_by_split[split])
        for path in files_by_split[split]:
            if parent_group_id(path) not in allowed:
                group_mismatches.append(path)
    if missing or duplicates or group_mismatches:
        raise AssertionError(
            f"Manifest verification failed: missing={len(missing)}, "
            f"duplicates={len(duplicates)}, group_mismatches={len(group_mismatches)}"
        )
    return {
        "files": len(listed),
        "groups": sum(len(groups_by_split[split]) for split in SPLITS),
        "missing": 0,
        "duplicates": 0,
        "group_mismatches": 0,
        "all_groups_disjoint": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a non-destructive parent-image grouped real-SAR manifest."
    )
    parser.add_argument("--dataset_dir", default="datasets/real_sar_dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ratios", type=int, nargs=3, default=(8, 1, 1))
    parser.add_argument("--manifest", default="")
    parser.add_argument("--leakage_report", default="")
    parser.add_argument("--verify", default="", help="Verify an existing manifest and exit")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


def main() -> int:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).resolve()
    if args.verify:
        with open(args.verify, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        print(json.dumps(verify_manifest(dataset_dir, manifest), indent=2))
        return 0

    manifest, leakage_report = build_manifest(dataset_dir, args.seed, args.ratios)
    manifest_path = Path(args.manifest or dataset_dir / f"real_split_grouped_seed{args.seed}.json")
    report_path = Path(args.leakage_report or dataset_dir / "real_split_leakage_check.json")
    print(json.dumps({"counts": manifest["counts"], "leakage": leakage_report}, indent=2))
    if args.dry_run:
        print(f"[dry-run] Would write {manifest_path} and {report_path}")
        return 0
    _write_json(manifest_path, manifest)
    _write_json(report_path, leakage_report)
    print(f"Wrote {manifest_path}")
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
