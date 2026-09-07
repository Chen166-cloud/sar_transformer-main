#!/usr/bin/env python3
"""Prepare and verify the immutable ICSPS26-FROZEN-v2 data protocol.

The script writes portable, relative-path CSV manifests.  It never overwrites
an existing output directory.  Formal mode enforces NWPU-RESISC45 45x700 and
UCMerced_LandUse 21x100 cardinalities; ``--smoke`` applies only the miniature
cardinalities declared in the protocol JSON.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from icsps2026_protocol import (
    PAIR_MANIFEST_FIELDS,
    SCHEDULE_FIELDS,
    SOURCE_MANIFEST_FIELDS,
    build_nwpu_source_manifest,
    build_train_schedule,
    discover_class_images,
    load_clean_intensity,
    load_protocol_config,
    perceptual_hash_rgb,
    portable_relative_path,
    read_csv_records,
    resolve_portable,
    sha256_array,
    sha256_file,
    stable_id,
    stable_u64,
    synthesize_gamma_only,
    validate_pair_manifest,
    validate_source_manifest,
    validate_train_schedule,
    write_csv_records,
)


DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "icsps2026_frozen_v2.json"
DETERMINISTIC_MAT_HEADER = (
    b"MATLAB 5.0 MAT-file, Platform: portable, "
    b"Created by prepare_icsps2026_data.py"
)
PHASH_AUDIT_FIELDS = (
    "protocol_id",
    "nwpu_source_id",
    "nwpu_class_name",
    "nwpu_source_relative_path",
    "ucm_source_id",
    "ucm_class_name",
    "ucm_source_relative_path",
    "nwpu_phash",
    "ucm_phash",
    "hamming_distance",
    "review_status",
    "action",
)


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _write_pair_mat(
    path: Path,
    *,
    clean: np.ndarray,
    noisy: np.ndarray,
    look: int,
    rng_seed: int,
    compression: bool,
) -> None:
    """Write a MAT v5 pair with a deterministic descriptive header."""

    try:
        from scipy.io import savemat
    except ImportError as error:  # pragma: no cover - environment issue
        raise RuntimeError("Data preparation requires scipy") from error
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite fixed pair: {path}")
    savemat(
        path,
        {
            "clean": np.asarray(clean, dtype=np.float32),
            "noisy": np.asarray(noisy, dtype=np.float32),
            "global_L": np.asarray([[int(look)]], dtype=np.int32),
            "rng_seed": np.asarray([[int(rng_seed)]], dtype=np.uint64),
        },
        appendmat=False,
        do_compression=bool(compression),
        oned_as="row",
    )
    # scipy's MAT v5 writer embeds the wall-clock time in bytes 0:116.  The
    # descriptive field has no semantic role, so freeze it for reproducible
    # artifact hashes while retaining the remaining standard header bytes.
    with path.open("r+b") as handle:
        handle.write(DETERMINISTIC_MAT_HEADER[:116].ljust(116, b" "))


def _ucm_sources(
    ucm_root: Path,
    config: Mapping[str, Any],
    nwpu_records: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    ucm = config["ucm"]
    discovered = discover_class_images(
        ucm_root,
        int(ucm["expected_classes"]),
        int(ucm["expected_per_class"]),
        smoke=config["mode"] == "smoke",
        selection_seed=int(config["seeds"]["ucm_test"]),
    )
    blocked_files = {str(row["source_sha256"]) for row in nwpu_records}
    blocked_pixels = {str(row["source_pixel_sha256"]) for row in nwpu_records}
    seen_files: dict[str, str] = {}
    seen_pixels: dict[str, str] = {}
    records: list[dict[str, object]] = []
    for class_name, paths in sorted(discovered.items()):
        for selection_rank, path in enumerate(paths, start=1):
            relative = portable_relative_path(path.relative_to(ucm_root).as_posix())
            source_hash = sha256_file(path)
            # Pixel-content leakage must be checked before the declared UCM
            # resize, otherwise two differently sized originals could be hidden.
            from PIL import Image

            with Image.open(path) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            pixel_hash = sha256_array(rgb)
            perceptual_hash = perceptual_hash_rgb(rgb)
            if source_hash in blocked_files or pixel_hash in blocked_pixels:
                raise AssertionError(f"UCM source overlaps selected NWPU content: {relative}")
            if source_hash in seen_files:
                raise AssertionError(
                    f"Duplicate UCM file content: {relative} and {seen_files[source_hash]}"
                )
            if pixel_hash in seen_pixels:
                raise AssertionError(
                    f"Duplicate UCM decoded pixels: {relative} and {seen_pixels[pixel_hash]}"
                )
            seen_files[source_hash] = relative
            seen_pixels[pixel_hash] = relative
            records.append(
                {
                    "class_name": class_name,
                    "source_id": stable_id("ucm", relative),
                    "source_relative_path": relative,
                    "selection_rank": selection_rank,
                    "parent_split": "external_test",
                    "source_sha256": source_hash,
                    "source_pixel_sha256": pixel_hash,
                    "source_phash": perceptual_hash,
                    "original_height": int(rgb.shape[0]),
                    "original_width": int(rgb.shape[1]),
                }
            )
    return records


def _generate_fixed_pairs(
    *,
    sources: Sequence[Mapping[str, object]],
    source_root: Path,
    output_root: Path,
    output_subdir: str,
    dataset: str,
    split: str,
    config: Mapping[str, Any],
    synthesis_seed: int,
    size_policy: str,
    mat_compression: bool,
    quiet: bool,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    total = len(sources) * len(config["looks"])
    completed = 0
    for source in sorted(
        sources,
        key=lambda row: (str(row["class_name"]), str(row["source_relative_path"])),
    ):
        source_path = resolve_portable(source_root, str(source["source_relative_path"]))
        clean, spatial = load_clean_intensity(
            source_path,
            config["image_size"],
            size_policy=size_policy,
        )
        for look_raw in config["looks"]:
            look = int(look_raw)
            source_id = str(source["source_id"])
            rng_seed = stable_u64(synthesis_seed, source_id, look)
            noisy, synthesis = synthesize_gamma_only(clean, look, rng_seed)
            pair_id = stable_id(
                f"{config['artifact_protocol_id']}:{dataset}:L{look}", source_id
            )
            relative_mat = portable_relative_path(
                f"fixed_pairs/{output_subdir}/L{look}/{source['class_name']}/{pair_id}.mat"
            )
            mat_path = resolve_portable(output_root, relative_mat)
            _write_pair_mat(
                mat_path,
                clean=clean,
                noisy=noisy,
                look=look,
                rng_seed=rng_seed,
                compression=mat_compression,
            )
            records.append(
                {
                    "protocol_id": config["artifact_protocol_id"],
                    "dataset": dataset,
                    "split": split,
                    "pair_id": pair_id,
                    "source_id": source_id,
                    "source_relative_path": source["source_relative_path"],
                    "class_name": source["class_name"],
                    "selection_rank": source.get("selection_rank", ""),
                    "parent_split": source.get("parent_split", split),
                    "global_L": look,
                    "rng_seed": rng_seed,
                    "mat_path": relative_mat,
                    "clean_path": relative_mat,
                    "clean_field": "clean",
                    "noisy_path": relative_mat,
                    "noisy_field": "noisy",
                    "source_sha256": source["source_sha256"],
                    "source_pixel_sha256": source["source_pixel_sha256"],
                    "source_phash": source["source_phash"],
                    "clean_sha256": sha256_array(clean),
                    "noisy_sha256": sha256_array(noisy),
                    "mat_sha256": sha256_file(mat_path),
                    "preclip_max": synthesis["preclip_max"],
                    "postclip_max": synthesis["postclip_max"],
                    "saturation_rate": synthesis["saturation_rate"],
                    "original_height": spatial["original_height"],
                    "original_width": spatial["original_width"],
                    "processed_height": spatial["processed_height"],
                    "processed_width": spatial["processed_width"],
                    "size_policy": spatial["size_policy"],
                    "size_changed": str(bool(spatial["size_changed"])).lower(),
                    "numeric_domain": config["numeric_domain"],
                }
            )
            completed += 1
            if not quiet and (completed == total or completed % 250 == 0):
                print(f"[{dataset}] wrote {completed}/{total} fixed pairs", flush=True)
    return records


def _cross_dataset_phash_audit(
    nwpu_sources: Sequence[Mapping[str, object]],
    ucm_sources: Sequence[Mapping[str, object]],
    *,
    protocol_id: str,
    maximum_distance: int,
) -> list[dict[str, object]]:
    """List near-duplicate candidates without excluding any test source."""

    if not 0 <= int(maximum_distance) <= 64:
        raise ValueError("pHash Hamming threshold must be in [0,64]")
    records: list[dict[str, object]] = []
    nwpu_values = [
        (row, int(str(row["source_phash"]), 16)) for row in nwpu_sources
    ]
    for ucm in ucm_sources:
        ucm_value = int(str(ucm["source_phash"]), 16)
        for nwpu, nwpu_value in nwpu_values:
            distance = bin(nwpu_value ^ ucm_value).count("1")
            if distance <= int(maximum_distance):
                records.append(
                    {
                        "protocol_id": protocol_id,
                        "nwpu_source_id": nwpu["source_id"],
                        "nwpu_class_name": nwpu["class_name"],
                        "nwpu_source_relative_path": nwpu["source_relative_path"],
                        "ucm_source_id": ucm["source_id"],
                        "ucm_class_name": ucm["class_name"],
                        "ucm_source_relative_path": ucm["source_relative_path"],
                        "nwpu_phash": nwpu["source_phash"],
                        "ucm_phash": ucm["source_phash"],
                        "hamming_distance": distance,
                        "review_status": "pending_manual_review",
                        "action": "manual_review_only_no_automatic_exclusion",
                    }
                )
    return sorted(
        records,
        key=lambda row: (
            int(row["hamming_distance"]),
            str(row["nwpu_source_id"]),
            str(row["ucm_source_id"]),
        ),
    )


def _critical_hashes(output_root: Path, relative_paths: Sequence[str]) -> dict[str, str]:
    return {
        portable_relative_path(relative): sha256_file(resolve_portable(output_root, relative))
        for relative in relative_paths
    }


def _csv_string_records(
    records: Sequence[Mapping[str, object]], fields: Sequence[str]
) -> list[dict[str, str]]:
    return [
        {field: str(record[field]) for field in fields}
        for record in records
    ]


def _verify_fixed_clean_sources(
    pair_records: Sequence[Mapping[str, object]],
    sources: Sequence[Mapping[str, object]],
    *,
    source_root: str | Path,
    image_size: Sequence[int],
    size_policy: str,
) -> None:
    by_source: dict[str, list[Mapping[str, object]]] = {}
    for row in pair_records:
        by_source.setdefault(str(row["source_id"]), []).append(row)
    expected_ids = {str(source["source_id"]) for source in sources}
    if set(by_source) != expected_ids:
        raise AssertionError("Fixed-pair source IDs do not match their canonical source set")
    for source in sources:
        source_id = str(source["source_id"])
        clean, spatial = load_clean_intensity(
            resolve_portable(source_root, str(source["source_relative_path"])),
            image_size,
            size_policy=size_policy,
        )
        clean_hash = sha256_array(clean)
        for row in by_source[source_id]:
            for field in ("selection_rank", "parent_split"):
                if field in source and str(row[field]) != str(source[field]):
                    raise AssertionError(
                        f"Fixed-pair source metadata mismatch ({field}): {source_id}"
                    )
            if str(row["clean_sha256"]) != clean_hash:
                raise AssertionError(
                    f"Fixed clean does not match source preprocessing: {source_id}"
                )
            for field in (
                "original_height",
                "original_width",
                "processed_height",
                "processed_width",
                "size_policy",
            ):
                if str(row[field]) != str(spatial[field]):
                    raise AssertionError(
                        f"Fixed-pair spatial metadata mismatch ({field}): {source_id}"
                    )
            if str(row["size_changed"]).lower() != str(bool(spatial["size_changed"])).lower():
                raise AssertionError(f"Fixed-pair size_changed mismatch: {source_id}")


def prepare_data(
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    nwpu_root: str | Path,
    ucm_root: str | Path,
    output_root: str | Path,
    nwpu_master_manifest: str | Path | None = None,
    run_seed: int = 42,
    smoke: bool = False,
    mat_compression: bool = True,
    quiet: bool = False,
) -> dict[str, Any]:
    """Build source/schedule manifests and all frozen validation/test MAT pairs."""

    config = load_protocol_config(config_path, smoke=smoke)
    allowed_run_seeds = {int(value) for value in config["schedule"]["run_seeds"]}
    if not smoke and int(run_seed) not in allowed_run_seeds:
        raise ValueError(f"Formal run_seed must be one of {sorted(allowed_run_seeds)}")
    if not smoke and nwpu_master_manifest is None:
        raise ValueError(
            "Formal preparation requires --nwpu-master-manifest from the repaired "
            "exact-content-disjoint 560/140 split; fallback splitting is smoke-only."
        )
    nwpu_root_path = Path(nwpu_root).resolve()
    ucm_root_path = Path(ucm_root).resolve()
    final_output_root = Path(output_root).resolve()
    if final_output_root.exists() or final_output_root.is_symlink():
        raise FileExistsError(
            f"Refusing to overwrite protocol directory: {final_output_root}. "
            "Choose a new --output-root."
        )
    final_output_root.parent.mkdir(parents=True, exist_ok=True)
    output_root_path = final_output_root.with_name(
        f".{final_output_root.name}.INCOMPLETE"
    )
    if output_root_path.exists() or output_root_path.is_symlink():
        raise FileExistsError(
            f"An incomplete staging directory already exists: {output_root_path}. "
            "Inspect it and move it aside before retrying."
        )
    output_root_path.mkdir()

    source_records = build_nwpu_source_manifest(
        nwpu_root_path,
        config,
        master_manifest_path=nwpu_master_manifest,
    )
    source_manifest_name = "source_manifest.csv"
    write_csv_records(
        output_root_path / source_manifest_name,
        SOURCE_MANIFEST_FIELDS,
        source_records,
    )

    train_records = [row for row in source_records if row["split"] == "train"]
    schedule_records = build_train_schedule(
        train_records,
        updates=int(config["schedule"]["updates"]),
        looks=config["looks"],
        run_seed=int(run_seed),
        synthesis_seed=int(config["seeds"]["synthesis"]),
        protocol_id=str(config["artifact_protocol_id"]),
        d4_count=int(config["schedule"]["d4_count"]),
    )
    schedule_name = f"train_schedule_seed{int(run_seed)}.csv"
    write_csv_records(
        output_root_path / schedule_name,
        SCHEDULE_FIELDS,
        schedule_records,
    )
    source_audit = validate_source_manifest(source_records, config)
    schedule_audit = validate_train_schedule(
        schedule_records,
        train_records,
        updates=int(config["schedule"]["updates"]),
        looks=config["looks"],
        run_seed=int(run_seed),
        synthesis_seed=int(config["seeds"]["synthesis"]),
        protocol_id=str(config["artifact_protocol_id"]),
        d4_count=int(config["schedule"]["d4_count"]),
    )

    validation_sources = [
        row for row in source_records if row["split"] == "validation"
    ]
    nwpu_pairs = _generate_fixed_pairs(
        sources=validation_sources,
        source_root=nwpu_root_path,
        output_root=output_root_path,
        output_subdir="nwpu_validation",
        dataset="NWPU-RESISC45",
        split="validation",
        config=config,
        synthesis_seed=int(config["seeds"]["validation"]),
        size_policy=str(config["nwpu"]["native_size_policy"]),
        mat_compression=mat_compression,
        quiet=quiet,
    )
    nwpu_pair_manifest_name = "nwpu_validation_manifest.csv"
    write_csv_records(
        output_root_path / nwpu_pair_manifest_name,
        PAIR_MANIFEST_FIELDS,
        nwpu_pairs,
    )

    ucm_sources = _ucm_sources(ucm_root_path, config, source_records)
    phash_config = config["nwpu"]["cross_dataset_near_duplicate_audit"]
    phash_candidates = _cross_dataset_phash_audit(
        source_records,
        ucm_sources,
        protocol_id=str(config["artifact_protocol_id"]),
        maximum_distance=int(phash_config["maximum_hamming_distance"]),
    )
    phash_audit_name = "cross_dataset_phash_audit.csv"
    write_csv_records(
        output_root_path / phash_audit_name,
        PHASH_AUDIT_FIELDS,
        phash_candidates,
    )
    ucm_size_policy = str(config["ucm"]["size_policy"]["name"])
    ucm_pairs = _generate_fixed_pairs(
        sources=ucm_sources,
        source_root=ucm_root_path,
        output_root=output_root_path,
        output_subdir="ucm_test",
        dataset="UCMerced_LandUse",
        split="external_test",
        config=config,
        synthesis_seed=int(config["seeds"]["ucm_test"]),
        size_policy=ucm_size_policy,
        mat_compression=mat_compression,
        quiet=quiet,
    )
    ucm_pair_manifest_name = "ucm_test_manifest.csv"
    write_csv_records(
        output_root_path / ucm_pair_manifest_name,
        PAIR_MANIFEST_FIELDS,
        ucm_pairs,
    )

    nwpu_pair_audit = validate_pair_manifest(
        nwpu_pairs,
        expected_dataset="NWPU-RESISC45",
        expected_classes=int(config["nwpu"]["expected_classes"]),
        expected_per_class_per_look=int(config["nwpu"]["validation_per_class"]),
        looks=config["looks"],
        output_root=output_root_path,
        synthesis_seed=int(config["seeds"]["validation"]),
    )
    ucm_pair_audit = validate_pair_manifest(
        ucm_pairs,
        expected_dataset="UCMerced_LandUse",
        expected_classes=int(config["ucm"]["expected_classes"]),
        expected_per_class_per_look=int(config["ucm"]["expected_per_class"]),
        looks=config["looks"],
        output_root=output_root_path,
        synthesis_seed=int(config["seeds"]["ucm_test"]),
    )
    ucm_shape_counts = Counter(
        (
            int(row["original_height"]),
            int(row["original_width"]),
            str(row["size_changed"]),
        )
        for row in ucm_pairs
        if int(row["global_L"]) == int(config["looks"][0])
    )

    config_snapshot = {
        "effective_protocol": config,
        "preparation": {
            "run_seed": int(run_seed),
            "nwpu_root_name": nwpu_root_path.name,
            "ucm_root_name": ucm_root_path.name,
            "nwpu_master_split_source": (
                {
                    "file_name": Path(nwpu_master_manifest).name,
                    "sha256": sha256_file(nwpu_master_manifest),
                }
                if nwpu_master_manifest is not None
                else {"method": "deterministic_fallback", "split_seed": int(config["seeds"]["split"])}
            ),
            "mat_compression": bool(mat_compression),
            "absolute_paths_persisted": False,
        },
    }
    config_snapshot_name = "config_snapshot.json"
    _write_json_exclusive(output_root_path / config_snapshot_name, config_snapshot)

    summary: dict[str, Any] = {
        "state": "complete",
        "protocol_id": config["artifact_protocol_id"],
        "mode": config["mode"],
        "run_seed": int(run_seed),
        "source_manifest": {
            "path": source_manifest_name,
            **source_audit,
        },
        "train_schedule": {
            "path": schedule_name,
            **schedule_audit,
        },
        "fixed_pairs": {
            "nwpu_validation": {
                "manifest": nwpu_pair_manifest_name,
                **nwpu_pair_audit,
            },
            "ucm_test": {
                "manifest": ucm_pair_manifest_name,
                **ucm_pair_audit,
                "declared_size_policy": config["ucm"]["size_policy"],
                "source_original_shape_and_resize_counts": [
                    {
                        "original_height": height,
                        "original_width": width,
                        "size_changed": changed == "true",
                        "sources": count,
                    }
                    for (height, width, changed), count in sorted(ucm_shape_counts.items())
                ],
            },
            "cross_dataset_near_duplicate_audit": {
                "path": phash_audit_name,
                "algorithm": phash_config["algorithm"],
                "maximum_hamming_distance": int(phash_config["maximum_hamming_distance"]),
                "candidate_pairs_for_manual_review": len(phash_candidates),
                "automatic_exclusions": 0,
            },
        },
    }
    summary_name = "prepare_summary.json"
    _write_json_exclusive(output_root_path / summary_name, summary)
    critical_names = [
        config_snapshot_name,
        source_manifest_name,
        schedule_name,
        nwpu_pair_manifest_name,
        ucm_pair_manifest_name,
        phash_audit_name,
        summary_name,
    ]
    hashes = {
        "state": "complete",
        "protocol_id": config["artifact_protocol_id"],
        "critical_files": _critical_hashes(output_root_path, critical_names),
        "fixed_pair_mat_hashes_are_recorded_in": [
            nwpu_pair_manifest_name,
            ucm_pair_manifest_name,
        ],
    }
    _write_json_exclusive(output_root_path / "artifact_hashes.json", hashes)
    # The user-visible final directory appears only after every artifact and
    # hash has been written successfully.  Rename is atomic on one filesystem.
    output_root_path.rename(final_output_root)
    if not quiet:
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def verify_prepared_data(
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    nwpu_root: str | Path,
    ucm_root: str | Path,
    output_root: str | Path,
    nwpu_master_manifest: str | Path | None = None,
    run_seed: int = 42,
    smoke: bool = False,
    verify_source_files: bool = False,
    verify_pair_files: bool = False,
) -> dict[str, Any]:
    """Verify canonical selection, quotas, paths, hashes and optional MAT contents."""

    config = load_protocol_config(config_path, smoke=smoke)
    if not smoke and nwpu_master_manifest is None:
        raise ValueError("Formal verification requires --nwpu-master-manifest")
    output = Path(output_root).resolve()
    if not output.is_dir():
        raise FileNotFoundError(output)

    schedule_name = f"train_schedule_seed{int(run_seed)}.csv"
    required_critical = {
        "config_snapshot.json",
        "source_manifest.csv",
        schedule_name,
        "nwpu_validation_manifest.csv",
        "ucm_test_manifest.csv",
        "cross_dataset_phash_audit.csv",
        "prepare_summary.json",
    }
    with (output / "config_snapshot.json").open("r", encoding="utf-8") as handle:
        snapshot = json.load(handle)
    if snapshot.get("effective_protocol") != config:
        raise AssertionError("Prepared config snapshot differs from the requested protocol")
    preparation = snapshot.get("preparation", {})
    if int(preparation.get("run_seed", -1)) != int(run_seed):
        raise AssertionError("Prepared config snapshot has a different run_seed")
    with (output / "prepare_summary.json").open("r", encoding="utf-8") as handle:
        frozen_summary = json.load(handle)
    if (
        frozen_summary.get("state") != "complete"
        or str(frozen_summary.get("protocol_id")) != str(config["artifact_protocol_id"])
        or str(frozen_summary.get("mode")) != str(config["mode"])
        or int(frozen_summary.get("run_seed", -1)) != int(run_seed)
    ):
        raise AssertionError("Prepared summary identity/state mismatch")
    with (output / "artifact_hashes.json").open("r", encoding="utf-8") as handle:
        frozen_hashes = json.load(handle)
    if (
        frozen_hashes.get("state") != "complete"
        or str(frozen_hashes.get("protocol_id")) != str(config["artifact_protocol_id"])
        or set(frozen_hashes.get("critical_files", {})) != required_critical
    ):
        raise AssertionError("artifact_hashes.json has incomplete or unexpected coverage")
    for relative, expected_hash in frozen_hashes["critical_files"].items():
        actual = sha256_file(resolve_portable(output, relative))
        if actual != expected_hash:
            raise AssertionError(f"Critical artifact hash mismatch: {relative}")

    source_records = read_csv_records(output / "source_manifest.csv")
    expected_source_records = build_nwpu_source_manifest(
        nwpu_root,
        config,
        master_manifest_path=nwpu_master_manifest,
    )
    if source_records != _csv_string_records(expected_source_records, SOURCE_MANIFEST_FIELDS):
        raise AssertionError(
            "source_manifest.csv is not the canonical subset reconstructed from the repaired master"
        )
    source_audit = validate_source_manifest(
        source_records,
        config,
        source_root=nwpu_root,
        verify_files=verify_source_files,
    )
    train_records = [row for row in source_records if row["split"] == "train"]
    schedule_records = read_csv_records(output / schedule_name)
    schedule_audit = validate_train_schedule(
        schedule_records,
        train_records,
        updates=int(config["schedule"]["updates"]),
        looks=config["looks"],
        run_seed=int(run_seed),
        synthesis_seed=int(config["seeds"]["synthesis"]),
        protocol_id=str(config["artifact_protocol_id"]),
        d4_count=int(config["schedule"]["d4_count"]),
    )
    expected_schedule = build_train_schedule(
        train_records,
        updates=int(config["schedule"]["updates"]),
        looks=config["looks"],
        run_seed=int(run_seed),
        synthesis_seed=int(config["seeds"]["synthesis"]),
        protocol_id=str(config["artifact_protocol_id"]),
        d4_count=int(config["schedule"]["d4_count"]),
    )
    if schedule_records != _csv_string_records(expected_schedule, SCHEDULE_FIELDS):
        raise AssertionError("Training schedule is valid-looking but not the canonical schedule")
    nwpu_pairs = read_csv_records(output / "nwpu_validation_manifest.csv")
    ucm_pairs = read_csv_records(output / "ucm_test_manifest.csv")
    nwpu_pair_audit = validate_pair_manifest(
        nwpu_pairs,
        expected_dataset="NWPU-RESISC45",
        expected_classes=int(config["nwpu"]["expected_classes"]),
        expected_per_class_per_look=int(config["nwpu"]["validation_per_class"]),
        looks=config["looks"],
        output_root=output,
        verify_file_hashes=verify_pair_files,
        synthesis_seed=int(config["seeds"]["validation"]),
    )
    ucm_pair_audit = validate_pair_manifest(
        ucm_pairs,
        expected_dataset="UCMerced_LandUse",
        expected_classes=int(config["ucm"]["expected_classes"]),
        expected_per_class_per_look=int(config["ucm"]["expected_per_class"]),
        looks=config["looks"],
        output_root=output,
        verify_file_hashes=verify_pair_files,
        synthesis_seed=int(config["seeds"]["ucm_test"]),
    )
    if {str(row["protocol_id"]) for row in source_records + schedule_records + nwpu_pairs + ucm_pairs} != {
        str(config["artifact_protocol_id"])
    }:
        raise AssertionError("Mixed or unexpected protocol_id in prepared manifests")
    nwpu_files = {str(row["source_sha256"]) for row in source_records}
    nwpu_pixels = {str(row["source_pixel_sha256"]) for row in source_records}
    if nwpu_files.intersection(str(row["source_sha256"]) for row in ucm_pairs):
        raise AssertionError("NWPU/UCM file-hash overlap")
    if nwpu_pixels.intersection(str(row["source_pixel_sha256"]) for row in ucm_pairs):
        raise AssertionError("NWPU/UCM pixel-hash overlap")

    expected_validation_sources = [
        row for row in expected_source_records if str(row["split"]) == "validation"
    ]
    _verify_fixed_clean_sources(
        nwpu_pairs,
        expected_validation_sources,
        source_root=nwpu_root,
        image_size=config["image_size"],
        size_policy=str(config["nwpu"]["native_size_policy"]),
    )

    expected_ucm_sources = _ucm_sources(
        Path(ucm_root).resolve(), config, expected_source_records
    )
    _verify_fixed_clean_sources(
        ucm_pairs,
        expected_ucm_sources,
        source_root=ucm_root,
        image_size=config["image_size"],
        size_policy=str(config["ucm"]["size_policy"]["name"]),
    )
    actual_ucm_by_id: dict[str, Mapping[str, object]] = {}
    for row in ucm_pairs:
        actual_ucm_by_id.setdefault(str(row["source_id"]), row)
    ucm_source_fields = (
        "class_name",
        "source_id",
        "source_relative_path",
        "source_sha256",
        "source_pixel_sha256",
        "source_phash",
        "selection_rank",
        "parent_split",
        "original_height",
        "original_width",
    )
    actual_ucm_sources = [
        {field: str(actual_ucm_by_id[str(expected["source_id"])][field]) for field in ucm_source_fields}
        for expected in expected_ucm_sources
        if str(expected["source_id"]) in actual_ucm_by_id
    ]
    expected_ucm_string = _csv_string_records(expected_ucm_sources, ucm_source_fields)
    if actual_ucm_sources != expected_ucm_string:
        raise AssertionError("UCM test sources do not match the current external-test root")

    phash_config = config["nwpu"]["cross_dataset_near_duplicate_audit"]
    expected_phash_audit = _cross_dataset_phash_audit(
        expected_source_records,
        expected_ucm_sources,
        protocol_id=str(config["artifact_protocol_id"]),
        maximum_distance=int(phash_config["maximum_hamming_distance"]),
    )
    actual_phash_audit = read_csv_records(output / "cross_dataset_phash_audit.csv")
    if actual_phash_audit != _csv_string_records(expected_phash_audit, PHASH_AUDIT_FIELDS):
        raise AssertionError("Cross-dataset pHash audit is incomplete or inconsistent")
    return {
        "state": (
            "fully_verified"
            if verify_pair_files
            else "structure_and_selected_source_content_verified"
        ),
        "protocol_id": config["artifact_protocol_id"],
        "source_manifest": source_audit,
        "train_schedule": schedule_audit,
        "nwpu_validation": nwpu_pair_audit,
        "ucm_test": ucm_pair_audit,
        "selected_source_hashes_rebuilt": True,
        "verified_source_file_hashes_second_pass": bool(verify_source_files),
        "verified_pair_file_hashes": bool(verify_pair_files),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare or verify the immutable ICSPS26-FROZEN-v2 data artifacts."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--nwpu-root", type=Path, required=True)
    parser.add_argument("--ucm-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--nwpu-master-manifest", type=Path)
    parser.add_argument("--run-seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--verify-source-files", action="store_true")
    parser.add_argument("--verify-pair-files", action="store_true")
    parser.add_argument(
        "--no-mat-compression",
        action="store_true",
        help="Disable scipy MAT compression (compression is enabled by default).",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.verify_only:
        audit = verify_prepared_data(
            config_path=args.config,
            nwpu_root=args.nwpu_root,
            ucm_root=args.ucm_root,
            output_root=args.output_root,
            nwpu_master_manifest=args.nwpu_master_manifest,
            run_seed=args.run_seed,
            smoke=args.smoke,
            verify_source_files=args.verify_source_files or not args.smoke,
            verify_pair_files=args.verify_pair_files or not args.smoke,
        )
        if not args.quiet:
            print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)
        return
    if args.verify_source_files or args.verify_pair_files:
        raise SystemExit("--verify-source-files/--verify-pair-files require --verify-only")
    prepare_data(
        config_path=args.config,
        nwpu_root=args.nwpu_root,
        ucm_root=args.ucm_root,
        output_root=args.output_root,
        nwpu_master_manifest=args.nwpu_master_manifest,
        run_seed=args.run_seed,
        smoke=args.smoke,
        mat_compression=not args.no_mat_compression,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()
