"""Shared semantic validators for the ICSPS 2026 pretest seal."""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Mapping, Sequence

import torch

from experiment_protocol import sha256_file
from icsps2026_protocol import validate_pair_manifest
from verify_icsps2026_artifact import verify as verify_artifact


PROTOCOL_ID = "ICSPS26-FROZEN-v2"
REQUIRED_UCM_LOOKS = {1, 4}
REQUIRED_REAL_REGION_TYPES = {"homogeneous", "structured"}
REAL_SPLIT_MANIFEST_SHA256 = "3d2161938861dddd66aed2950160b355bf067143564a35af51f6209dba659ec0"


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Registration CSV has no rows: {path}")
    return rows


def canonical_int(value: object, field: str) -> int:
    text = str(value)
    if text != text.strip():
        raise ValueError(f"{field} must not contain surrounding whitespace")
    try:
        parsed = int(text)
    except ValueError as error:
        raise ValueError(f"{field} must be an integer, found {value!r}") from error
    if str(parsed) != text:
        raise ValueError(f"{field} must be a canonical integer, found {value!r}")
    return parsed


def bool_value(value: object, field: str) -> bool:
    text = str(value)
    if text != text.strip():
        raise ValueError(f"{field} must not contain surrounding whitespace")
    normalized = text.lower()
    if normalized not in {"true", "false"}:
        raise ValueError(f"{field} must be TRUE or FALSE, found {value!r}")
    return normalized == "true"


def _required_exact(row: Mapping[str, object], field: str) -> str:
    value = str(row.get(field, ""))
    if not value or value != value.strip():
        raise ValueError(f"Registered figure row requires canonical non-empty {field}")
    return value


def validate_crop_fields(
    row: Mapping[str, object], *, height: int, width: int
) -> tuple[int, int, int, int]:
    x, y, crop_width, crop_height = (
        canonical_int(row.get(field, ""), field)
        for field in ("crop_x", "crop_y", "crop_width", "crop_height")
    )
    if x < 0 or y < 0 or crop_width <= 0 or crop_height <= 0:
        raise ValueError("Registered crop origins must be non-negative and sizes positive")
    if x + crop_width > width or y + crop_height > height:
        raise ValueError(
            f"Registered crop {(x, y, crop_width, crop_height)} exceeds {width}x{height}"
        )
    return x, y, crop_width, crop_height


def select_registered_ucm_pairs(
    figure_rows: Sequence[Mapping[str, object]],
    manifest_rows: Sequence[Mapping[str, object]],
) -> list[tuple[dict[str, object], dict[str, object]]]:
    selected = [dict(row) for row in figure_rows if row.get("dataset") == "UCM"]
    if not selected:
        raise ValueError("figure_selection.csv contains no canonical dataset=UCM rows")
    seen: set[tuple[str, int]] = set()
    looks_seen: set[int] = set()
    matched: list[tuple[dict[str, object], dict[str, object]]] = []
    for registration in selected:
        if registration.get("protocol_id") != PROTOCOL_ID:
            raise ValueError("Registered UCM row has the wrong/non-canonical protocol_id")
        if not bool_value(
            registration.get("selected_before_test_unlock", ""),
            "selected_before_test_unlock",
        ):
            raise ValueError("Registered UCM row was not selected before test unlock")
        source_id = _required_exact(registration, "source_id")
        class_name = _required_exact(registration, "class_name")
        _required_exact(registration, "selection_rule")
        looks = canonical_int(registration.get("L", ""), "L")
        if looks not in {1, 2, 4, 8}:
            raise ValueError("Registered UCM figure L must be one of 1/2/4/8")
        key = (source_id, looks)
        if key in seen:
            raise ValueError(f"Duplicate registered UCM source/L row: {key}")
        seen.add(key)
        candidates = [
            dict(row)
            for row in manifest_rows
            if row.get("source_id") == source_id
            and canonical_int(row.get("global_L", ""), "global_L") == looks
        ]
        if len(candidates) != 1:
            raise ValueError(f"Registered UCM row {key} resolves to {len(candidates)} pairs")
        pair = candidates[0]
        if pair.get("protocol_id") != PROTOCOL_ID or pair.get("dataset") != "UCMerced_LandUse":
            raise ValueError(f"Registered UCM pair {key} has wrong manifest identity")
        if pair.get("class_name") != class_name:
            raise ValueError(f"Registered UCM class does not match manifest for {key}")
        height = canonical_int(pair.get("processed_height", ""), "processed_height")
        width = canonical_int(pair.get("processed_width", ""), "processed_width")
        if (height, width) != (256, 256):
            raise ValueError(f"Registered UCM pair {key} is not 256x256")
        validate_crop_fields(registration, height=height, width=width)
        looks_seen.add(looks)
        matched.append((registration, pair))
    if not REQUIRED_UCM_LOOKS.issubset(looks_seen):
        raise ValueError("Registered UCM figures must include canonical L=1 and L=4 rows")
    return matched


def _parse_shape(value: object) -> tuple[int, int]:
    text = str(value)
    if text != text.strip() or re.fullmatch(r"[1-9][0-9]*x[1-9][0-9]*", text) is None:
        raise ValueError(f"Invalid canonical real-SAR noisy_shape: {value!r}")
    height, width = (int(part) for part in text.split("x"))
    return height, width


def select_registered_real_samples(
    figure_rows: Sequence[Mapping[str, object]],
    qc_rows: Sequence[Mapping[str, object]],
) -> list[tuple[dict[str, object], dict[str, object]]]:
    selected = [dict(row) for row in figure_rows if row.get("dataset") == "RealSAR"]
    if not selected:
        raise ValueError("figure_selection.csv contains no canonical dataset=RealSAR rows")
    by_sample: dict[str, list[dict[str, object]]] = {}
    for row in qc_rows:
        by_sample.setdefault(str(row.get("sample_id", "")), []).append(dict(row))
    seen_samples: set[str] = set()
    region_types: set[str] = set()
    matched: list[tuple[dict[str, object], dict[str, object]]] = []
    for registration in selected:
        if registration.get("protocol_id") != PROTOCOL_ID:
            raise ValueError("Registered RealSAR row has the wrong/non-canonical protocol_id")
        if not bool_value(
            registration.get("selected_before_test_unlock", ""),
            "selected_before_test_unlock",
        ):
            raise ValueError("Registered RealSAR row was not selected before test unlock")
        sample_id = _required_exact(registration, "sample_id")
        parent_id = _required_exact(registration, "parent_id")
        _required_exact(registration, "selection_rule")
        region_type = _required_exact(registration, "region_type")
        if region_type != region_type.lower() or region_type not in REQUIRED_REAL_REGION_TYPES:
            raise ValueError("RealSAR region_type must be homogeneous or structured")
        if sample_id in seen_samples:
            raise ValueError(f"Duplicate registered RealSAR sample_id: {sample_id}")
        seen_samples.add(sample_id)
        candidates = by_sample.get(sample_id, [])
        if len(candidates) != 1:
            raise ValueError(
                f"Registered RealSAR sample_id={sample_id!r} resolves to {len(candidates)} QC rows"
            )
        qc = candidates[0]
        if qc.get("parent_id") != parent_id:
            raise ValueError(f"Registered RealSAR parent mismatch for sample_id={sample_id}")
        if qc.get("split") != "test" or not bool_value(
            qc.get("valid_no_reference", ""), "valid_no_reference"
        ):
            raise ValueError(f"Registered RealSAR sample is not a valid frozen test patch: {sample_id}")
        if not bool_value(qc.get("valid_enl_rois", ""), "valid_enl_rois"):
            raise ValueError(f"Registered RealSAR sample has no valid frozen ENL ROIs: {sample_id}")
        height, width = _parse_shape(qc.get("noisy_shape", ""))
        validate_crop_fields(registration, height=height, width=width)
        region_types.add(region_type)
        matched.append((registration, qc))
    if not REQUIRED_REAL_REGION_TYPES.issubset(region_types):
        raise ValueError(
            "Registered RealSAR figures must include homogeneous and structured rows"
        )
    return matched


def validate_figure_selection(
    figure_rows: Sequence[Mapping[str, object]],
    ucm_manifest_rows: Sequence[Mapping[str, object]],
    real_qc_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if any(row.get("dataset") not in {"UCM", "RealSAR"} for row in figure_rows):
        raise ValueError("Every registered figure row must use exactly UCM or RealSAR")
    figure_panels: set[tuple[str, str]] = set()
    for row in figure_rows:
        key = (_required_exact(row, "figure"), _required_exact(row, "panel"))
        if key in figure_panels:
            raise ValueError(f"Duplicate registered figure/panel: {key}")
        figure_panels.add(key)
    ucm = select_registered_ucm_pairs(figure_rows, ucm_manifest_rows)
    real = select_registered_real_samples(figure_rows, real_qc_rows)
    return {
        "ucm_matches": ucm,
        "real_matches": real,
        "ucm_rows": len(ucm),
        "real_rows": len(real),
    }


def validate_baseline_selection(
    run_root: Path, validation_rows: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    baseline_rows = [
        row for row in validation_rows
        if row.get("selection_role") == "strongest_non_ours_baseline"
    ]
    if len(baseline_rows) != 1:
        raise ValueError(
            "validation_selection.csv must contain exactly one canonical "
            "selection_role=strongest_non_ours_baseline row"
        )
    row = baseline_rows[0]
    if row.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Predeclared baseline has the wrong/non-canonical protocol_id")
    method = str(row.get("method", ""))
    if method not in {"transsar_v2", "sar_cam"}:
        raise ValueError("Strongest non-ours baseline must be transsar_v2 or sar_cam")
    if str(row.get("variant", "")) != "":
        raise ValueError("TransSARV2/SAR-CAM baseline variant must be empty")
    if not bool_value(row.get("formal_run", ""), "formal_run"):
        raise ValueError("Predeclared strongest baseline must come from a formal run")
    seed = canonical_int(row.get("seed", ""), "seed")
    step = canonical_int(row.get("global_step", ""), "global_step")
    if seed != 42 or step < 0 or step > 100_000 or step % 5_000 != 0:
        raise ValueError("Baseline must be seed 42 at a frozen 0..100000 validation step")
    try:
        validation_mse = float(str(row.get("val_macro_mse", "")))
    except ValueError as error:
        raise ValueError("Baseline val_macro_mse must be numeric") from error
    if not math.isfinite(validation_mse) or validation_mse < 0:
        raise ValueError("Baseline val_macro_mse must be finite and non-negative")
    checkpoint_hash = str(row.get("checkpoint_sha256", ""))
    if re.fullmatch(r"[0-9a-f]{64}", checkpoint_hash) is None:
        raise ValueError("Baseline checkpoint_sha256 must be canonical lowercase SHA-256")

    run_dir = run_root / "train" / method / f"seed{seed}"
    completion_path = run_dir / "completion.json"
    checkpoint_path = run_dir / "checkpoint_best.pth"
    metrics_path = run_dir / "step_metrics.csv"
    verify_artifact(
        "supervised", completion_path, expected_method=method,
        expected_variant="", expected_seed=seed, expected_adaptation="",
    )
    with completion_path.open("r", encoding="utf-8") as handle:
        completion = json.load(handle)
    best = completion.get("best", {})
    if best.get("global_step") != step or not math.isclose(
        float(best.get("val_macro_mse")), validation_mse, rel_tol=1e-12, abs_tol=1e-15
    ):
        raise ValueError("Predeclared baseline does not match completion.json best selection")
    actual_hash = sha256_file(checkpoint_path)
    if actual_hash != checkpoint_hash or completion.get("checkpoint_best_sha256") != actual_hash:
        raise ValueError("Predeclared baseline checkpoint SHA-256 does not match the actual best file")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise ValueError(f"Cannot read selected baseline checkpoint: {error}") from error
    if not isinstance(checkpoint, dict) or checkpoint.get("global_step") != step:
        raise ValueError("Predeclared baseline step does not match checkpoint_best.pth")
    checkpoint_metrics = checkpoint.get("metrics")
    if not isinstance(checkpoint_metrics, dict) or not math.isclose(
        float(checkpoint_metrics.get("val_macro_mse", "nan")), validation_mse,
        rel_tol=1e-12, abs_tol=1e-15,
    ):
        raise ValueError("Predeclared baseline MSE does not match checkpoint_best.pth")
    metric_rows = csv_rows(metrics_path)
    candidates = [
        metric for metric in metric_rows
        if canonical_int(metric.get("global_step", ""), "global_step") == step
    ]
    if len(candidates) != 1:
        raise ValueError("Selected baseline step does not resolve uniquely in step_metrics.csv")
    metric = candidates[0]
    if not bool_value(metric.get("is_best", ""), "is_best") or not math.isclose(
        float(metric.get("val_macro_mse", "nan")), validation_mse,
        rel_tol=1e-12, abs_tol=1e-15,
    ):
        raise ValueError("Predeclared baseline does not match its best step_metrics.csv row")
    return {
        "method": method,
        "variant": None,
        "seed": seed,
        "global_step": step,
        "val_macro_mse": validation_mse,
        "run_directory": str(run_dir.resolve()),
        "completion_sha256": sha256_file(completion_path),
        "step_metrics_sha256": sha256_file(metrics_path),
        "checkpoint_best_sha256": actual_hash,
    }


def build_binding(run_root: Path, prep_root: Path, real_protocol_root: Path) -> dict[str, object]:
    records = run_root / "records"
    validation_path = records / "validation_selection.csv"
    figures_path = records / "figure_selection.csv"
    config_path = prep_root / "config_snapshot.json"
    ucm_path = prep_root / "ucm_test_manifest.csv"
    prepared_hashes_path = prep_root / "artifact_hashes.json"
    real_manifest_path = real_protocol_root / "real_qc_manifest.json"
    for path in (
        validation_path, figures_path, config_path, ucm_path,
        prepared_hashes_path, real_manifest_path,
    ):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing/non-empty pretest input: {path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config_payload = json.load(handle)
    config = config_payload.get("effective_protocol", config_payload)
    if (
        config.get("protocol_id") != PROTOCOL_ID
        or config.get("artifact_protocol_id") != PROTOCOL_ID
        or config.get("mode") != "formal"
    ):
        raise ValueError("Prepared config is not a formal ICSPS26-FROZEN-v2 snapshot")
    real_config = config.get("real")
    if (
        not isinstance(real_config, dict)
        or real_config.get("use_gt16") is not False
        or real_config.get("evaluation_scope") != "no_reference_only"
        or set(real_config.get("forbidden_paper_metrics", ()))
        != {"qpsnr_gt16", "qssim_gt16"}
    ):
        raise ValueError("Prepared config does not bind the frozen no-GT16 real protocol")
    with prepared_hashes_path.open("r", encoding="utf-8") as handle:
        prepared_hashes = json.load(handle)
    critical_files = prepared_hashes.get("critical_files")
    required_critical_files = {
        "config_snapshot.json", "source_manifest.csv", "train_schedule_seed42.csv",
        "nwpu_validation_manifest.csv", "ucm_test_manifest.csv",
        "cross_dataset_phash_audit.csv", "prepare_summary.json",
    }
    if (
        prepared_hashes.get("state") != "complete"
        or prepared_hashes.get("protocol_id") != PROTOCOL_ID
        or not isinstance(critical_files, dict)
        or set(critical_files) != required_critical_files
    ):
        raise ValueError("Prepared-data hash index is not a completed formal artifact")
    for relative, digest in critical_files.items():
        artifact = (prep_root / str(relative)).resolve()
        try:
            artifact.relative_to(prep_root.resolve())
        except ValueError as error:
            raise ValueError("Prepared-data critical path escapes PREP_ROOT") from error
        if not artifact.is_file() or sha256_file(artifact) != digest:
            raise ValueError(f"Prepared-data critical artifact SHA-256 mismatch: {relative}")
    if critical_files.get(config_path.name) != sha256_file(config_path):
        raise ValueError("Prepared config is not covered by the prepared-data hash chain")
    if critical_files.get(ucm_path.name) != sha256_file(ucm_path):
        raise ValueError("UCM test manifest is not covered by the prepared-data hash chain")
    ucm_rows = csv_rows(ucm_path)
    if len(ucm_rows) != 8_400:
        raise ValueError(f"Frozen UCM manifest must contain 8400 rows, found {len(ucm_rows)}")
    validate_pair_manifest(
        ucm_rows,
        expected_dataset="UCMerced_LandUse",
        expected_classes=int(config["ucm"]["expected_classes"]),
        expected_per_class_per_look=int(config["ucm"]["expected_per_class"]),
        looks=config["looks"],
        output_root=None,
        verify_file_hashes=False,
        synthesis_seed=int(config["seeds"]["ucm_test"]),
    )
    if any(
        row.get("protocol_id") != PROTOCOL_ID
        or canonical_int(row.get("processed_height", ""), "processed_height") != 256
        or canonical_int(row.get("processed_width", ""), "processed_width") != 256
        for row in ucm_rows
    ):
        raise ValueError("Frozen UCM manifest has wrong protocol identity or processed shape")
    with real_manifest_path.open("r", encoding="utf-8") as handle:
        real_manifest = json.load(handle)
    if (
        real_manifest.get("schema_version") != 1
        or real_manifest.get("expected_shape") != [256, 256]
        or real_manifest.get("split_manifest_sha256") != REAL_SPLIT_MANIFEST_SHA256
    ):
        raise ValueError("Real-SAR QC manifest does not match the frozen split/shape protocol")
    counts = real_manifest.get("counts")
    if (
        real_manifest.get("gt16_root") is not None
        or real_manifest.get("gt16_disabled_by_protocol") is not True
        or not isinstance(counts, dict)
        or counts.get("gt16_files_discovered") != 0
        or counts.get("gt16_files_matched_one_to_one") != 0
        or counts.get("gt16_files_unmatched") != 0
    ):
        raise ValueError("Real-SAR QC manifest violates the frozen no-GT16 protocol")
    artifacts = real_manifest.get("artifacts", {})
    qc_name = artifacts.get("qc_csv")
    if not qc_name:
        raise ValueError("Real-SAR QC manifest does not declare qc_csv")
    real_qc_path = (real_protocol_root / str(qc_name)).resolve()
    try:
        real_qc_path.relative_to(real_protocol_root.resolve())
    except ValueError as error:
        raise ValueError("Real-SAR qc_csv escapes REAL_PROTOCOL_ROOT") from error
    if sha256_file(real_qc_path) != artifacts.get("qc_csv_sha256"):
        raise ValueError("Real-SAR QC CSV SHA-256 mismatch")
    real_rows = csv_rows(real_qc_path)
    required_no_gt16_fields = {
        "gt16_path", "gt16_sha256", "pair_exists", "valid_quasi_reference"
    }
    if required_no_gt16_fields.difference(real_rows[0]):
        raise ValueError("Real-SAR QC CSV lacks required no-GT16 audit fields")
    if any(
        str(row.get("gt16_path", "")) != ""
        or str(row.get("gt16_sha256", "")) != ""
        or bool_value(row.get("pair_exists", ""), "pair_exists")
        or bool_value(
            row.get("valid_quasi_reference", ""), "valid_quasi_reference"
        )
        for row in real_rows
    ):
        raise ValueError("Real-SAR QC rows contain GT16 data in the frozen no-GT16 branch")
    expected_real = {"train": (4_700, 1_175), "val": (584, 146), "test": (592, 148)}
    valid_by_split: dict[str, list[dict[str, str]]] = {}
    for split, (expected_patches, expected_parents) in expected_real.items():
        valid = [
            row for row in real_rows
            if row.get("split") == split
            and bool_value(row.get("valid_no_reference", ""), "valid_no_reference")
        ]
        if len(valid) != expected_patches or len({row.get("parent_id") for row in valid}) != expected_parents:
            raise ValueError(
                f"Frozen real {split} QC must contain {expected_patches} valid patches "
                f"from {expected_parents} parents"
            )
        valid_by_split[split] = valid
    valid_test = valid_by_split["test"]
    validation_rows = csv_rows(validation_path)
    figure_rows = csv_rows(figures_path)
    baseline = validate_baseline_selection(run_root, validation_rows)
    figure_validation = validate_figure_selection(figure_rows, ucm_rows, real_rows)
    return {
        "protocol_id": PROTOCOL_ID,
        "run_root": str(run_root.resolve()),
        "prep_root": str(prep_root.resolve()),
        "real_protocol_root": str(real_protocol_root.resolve()),
        "validation_selection_sha256": sha256_file(validation_path),
        "figure_selection_sha256": sha256_file(figures_path),
        "prepared_config_sha256": sha256_file(config_path),
        "prepared_artifact_hashes_sha256": sha256_file(prepared_hashes_path),
        "ucm_test_manifest_sha256": sha256_file(ucm_path),
        "ucm_test_pairs": len(ucm_rows),
        "real_qc_manifest_sha256": sha256_file(real_manifest_path),
        "real_qc_csv_sha256": sha256_file(real_qc_path),
        "real_test_patches": len(valid_test),
        "real_test_parents": len({row.get("parent_id") for row in valid_test}),
        "real_use_gt16": False,
        "real_evaluation_scope": "no_reference_only",
        "real_valid_quasi_reference": 0,
        "registered_ucm_rows": figure_validation["ucm_rows"],
        "registered_real_rows": figure_validation["real_rows"],
        "strongest_non_ours_baseline": baseline,
    }
