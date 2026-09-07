"""Read-only, fail-closed verifier for completed ICSPS 2026 artifacts.

Identity tokens are compared after lower-casing; ``None``, ``null`` and the
empty string all mean "no variant/adaptation/seed".  ``ours/full`` remains an
explicit variant and is never treated as equivalent to an empty variant.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from experiment_protocol import sha256_file
from finalize_icsps2026_sarbm3d_raw import (
    SARBM3D_DOMAIN_CONVERSION,
    SARBM3D_INPUT_DOMAIN,
    SARBM3D_OUTPUT_DOMAIN,
)


PROTOCOL_ID = "ICSPS26-FROZEN-v2"
_UNSET = object()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kind",
        required=True,
        choices=("supervised", "ams", "ucm", "real", "sarbm_raw"),
    )
    parser.add_argument("--path", required=True, help="completion.json or aggregate.json")
    parser.add_argument("--expected-method")
    parser.add_argument(
        "--expected-variant",
        help="Expected canonical variant; pass an empty string for no variant",
    )
    parser.add_argument(
        "--expected-seed",
        help="Expected integer seed; pass an empty string when seed is not applicable",
    )
    parser.add_argument(
        "--expected-adaptation",
        help="Expected adaptation; pass an empty string for no adaptation",
    )
    return parser.parse_args()


def _canonical_method(value: object) -> str:
    if value is None or not str(value).strip():
        raise ValueError("Artifact method identity is missing")
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "lee_mmse": "lee_mmse",
        "sar_bm3d": "sar_bm3d",
    }
    return aliases.get(normalized, normalized)


def _canonical_nullable(value: object) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    return None if normalized in {"", "none", "null"} else normalized


def _canonical_seed(value: object) -> int | None:
    nullable = _canonical_nullable(value)
    if nullable is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"Invalid seed identity: {value!r}")
    try:
        integer = int(nullable)
    except ValueError as error:
        raise ValueError(f"Invalid seed identity: {value!r}") from error
    if str(integer) != nullable:
        raise ValueError(f"Seed identity is not a canonical integer: {value!r}")
    return integer


def _identity(mapping: dict[str, Any], label: str) -> dict[str, Any]:
    try:
        return {
            "method": _canonical_method(mapping.get("method")),
            "variant": _canonical_nullable(mapping.get("variant")),
            "seed": _canonical_seed(mapping.get("seed")),
            "adaptation": _canonical_nullable(mapping.get("adaptation")),
        }
    except ValueError as error:
        raise ValueError(f"Invalid {label} identity: {error}") from error


def _expected_identity(
    *,
    method: object = _UNSET,
    variant: object = _UNSET,
    seed: object = _UNSET,
    adaptation: object = _UNSET,
) -> dict[str, Any]:
    expected: dict[str, Any] = {}
    if method is not _UNSET:
        expected["method"] = _canonical_method(method)
    if variant is not _UNSET:
        expected["variant"] = _canonical_nullable(variant)
    if seed is not _UNSET:
        expected["seed"] = _canonical_seed(seed)
    if adaptation is not _UNSET:
        expected["adaptation"] = _canonical_nullable(adaptation)
    return expected


def _assert_identity(
    actual: dict[str, Any], expected: dict[str, Any], label: str
) -> None:
    mismatches = {
        field: {"expected": value, "found": actual.get(field)}
        for field, value in expected.items()
        if actual.get(field) != value
    }
    if mismatches:
        raise ValueError(f"{label} identity mismatch: {mismatches}")


def _load_checkpoint_identity(
    path: Path, label: str, expected_format: str
) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise ValueError(f"Unreadable {label}: {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a metadata-bearing mapping")
    if payload.get("checkpoint_format") != expected_format:
        raise ValueError(
            f"{label} format mismatch: expected {expected_format!r}, "
            f"found {payload.get('checkpoint_format')!r}"
        )
    run_config = payload.get("run_config")
    model_metadata = payload.get("model_metadata")
    if not isinstance(run_config, dict) or not isinstance(model_metadata, dict):
        raise ValueError(f"{label} lacks run_config/model_metadata")
    run_identity = _identity(run_config, f"{label} run_config")
    model_identity = {
        "method": _canonical_method(model_metadata.get("method")),
        "variant": _canonical_nullable(model_metadata.get("variant")),
    }
    _assert_identity(
        model_identity,
        {key: run_identity[key] for key in ("method", "variant")},
        f"{label} model_metadata",
    )
    return run_identity


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Unreadable JSON artifact {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"Artifact must be a JSON object: {path}")
    return payload


def _expect(payload: dict[str, Any], expected: dict[str, Any], label: str) -> None:
    mismatches = {
        key: {"expected": value, "found": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value or type(payload.get(key)) is not type(value)
    }
    if mismatches:
        raise ValueError(f"Invalid {label}: {mismatches}")


def _safe_sibling(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"Artifact path escapes its run directory: {relative}") from error
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def _verify_hash(path: Path, expected: object, label: str) -> str:
    expected_text = str(expected).lower()
    actual = sha256_file(path)
    if len(expected_text) != 64 or actual != expected_text:
        raise ValueError(f"{label} SHA-256 mismatch: {path}")
    return actual


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV artifact is empty: {path}")
    return rows


def _verify_pretest_credential(payload: dict[str, Any], label: str) -> dict[str, Any]:
    credential = payload.get("pretest_gate")
    if not isinstance(credential, dict):
        raise ValueError(f"{label} lacks the pretest-gate credential")
    expected = {
        "ok": True,
        "protocol_id": PROTOCOL_ID,
        "real_use_gt16": False,
        "real_evaluation_scope": "no_reference_only",
        "real_valid_quasi_reference": 0,
    }
    for field, value in expected.items():
        if credential.get(field) != value or type(credential.get(field)) is not type(value):
            raise ValueError(f"{label} has an invalid pretest credential field: {field}")
    for field in ("registered_ucm_rows", "registered_real_rows"):
        value = credential.get(field)
        if type(value) is not int or value < 2:
            raise ValueError(f"{label} has an invalid pretest credential count: {field}")
    if credential.get("strongest_non_ours_baseline") not in {"transsar_v2", "sar_cam"}:
        raise ValueError(f"{label} has an invalid predeclared comparison baseline")
    timestamp = credential.get("test_unlock_utc")
    if not isinstance(timestamp, str) or not timestamp.endswith("Z"):
        raise ValueError(f"{label} has an invalid pretest unlock timestamp")
    for field in (
        "registration_sha256",
        "pretest_seal_sha256",
        "validation_selection_sha256",
        "figure_selection_sha256",
        "ucm_test_manifest_sha256",
        "real_qc_manifest_sha256",
        "real_qc_csv_sha256",
    ):
        digest = credential.get(field)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"{label} has an invalid {field}")
    seal_path = credential.get("pretest_seal")
    if not isinstance(seal_path, str) or not Path(seal_path).is_absolute():
        raise ValueError(f"{label} does not identify its absolute pretest seal")
    return credential


def verify_supervised(
    path: Path, payload: dict[str, Any], expected_identity: dict[str, Any]
) -> dict[str, Any]:
    _expect(
        payload,
        {
            "protocol_id": PROTOCOL_ID,
            "formal_run": True,
            "completed_updates": 100_000,
            "target_updates": 100_000,
        },
        "supervised completion",
    )
    best = _safe_sibling(path.parent, "checkpoint_best.pth")
    last = _safe_sibling(path.parent, "checkpoint_last.pth")
    _verify_hash(best, payload.get("checkpoint_best_sha256"), "best checkpoint")
    _verify_hash(last, payload.get("checkpoint_last_sha256"), "last checkpoint")
    best_identity = _load_checkpoint_identity(
        best, "best supervised checkpoint", "icsps26-resumable-v1"
    )
    last_identity = _load_checkpoint_identity(
        last, "last supervised checkpoint", "icsps26-resumable-v1"
    )
    _assert_identity(last_identity, best_identity, "supervised checkpoint pair")
    _assert_identity(best_identity, expected_identity, "supervised checkpoint")
    if best_identity["adaptation"] is not None:
        raise ValueError("Supervised checkpoint unexpectedly declares an adaptation")
    if not isinstance(payload.get("best"), dict) or payload["best"].get("global_step") is None:
        raise ValueError("Supervised completion has no selected best checkpoint")
    synthesis = payload.get("training_synthesis_by_L")
    if not isinstance(synthesis, dict) or set(synthesis) != {"1", "2", "4", "8"}:
        raise ValueError("Supervised completion lacks the four-L all-update synthesis audit")
    for looks, item in synthesis.items():
        if not isinstance(item, dict) or item.get("updates") != 25_000:
            raise ValueError(f"L={looks} synthesis audit must contain 25000 updates")
        for field in (
            "preclip_max_min", "preclip_max_max", "postclip_max_min",
            "postclip_max_max", "saturation_rate_mean", "saturation_rate_max",
        ):
            try:
                value = float(item[field])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"L={looks} synthesis audit lacks finite {field}") from error
            if not math.isfinite(value):
                raise ValueError(f"L={looks} synthesis audit has non-finite {field}")
    return {"kind": "supervised", "updates": 100_000, "identity": best_identity}


def verify_ams(
    path: Path, payload: dict[str, Any], expected_identity: dict[str, Any]
) -> dict[str, Any]:
    _expect(
        payload,
        {
            "protocol_id": PROTOCOL_ID,
            "formal_run": True,
            "completed_epochs": 8,
        },
        "AMS completion",
    )
    retired_fields = {
        "eligible_checkpoint_found",
        "base_synthetic_macro_psnr",
        "synthetic_macro_psnr",
        "synthetic_psnr_drop",
    }.intersection(payload)
    if retired_fields:
        raise ValueError(
            f"AMS completion contains retired synthetic-selection fields: {sorted(retired_fields)}"
        )
    best = _safe_sibling(path.parent, "checkpoint_best.pth")
    last = _safe_sibling(path.parent, "checkpoint_last.pth")
    _verify_hash(best, payload.get("checkpoint_best_sha256"), "AMS best checkpoint")
    _verify_hash(last, payload.get("checkpoint_last_sha256"), "AMS last checkpoint")
    best_identity = _load_checkpoint_identity(best, "best AMS checkpoint", "icsps26-ams-v1")
    last_identity = _load_checkpoint_identity(last, "last AMS checkpoint", "icsps26-ams-v1")
    _assert_identity(last_identity, best_identity, "AMS checkpoint pair")
    _assert_identity(best_identity, expected_identity, "AMS checkpoint")
    if best_identity["adaptation"] != "ams":
        raise ValueError("AMS checkpoint does not declare adaptation='ams'")
    selected = payload.get("best")
    if not isinstance(selected, dict):
        raise ValueError("AMS completion has no selected best epoch")
    selected_epoch = selected.get("epoch")
    if type(selected_epoch) is not int or not 1 <= selected_epoch <= 8:
        raise ValueError("AMS selected best epoch must be an integer in [1,8]")
    try:
        selected_loss = float(selected["fixed_val_masked_loss"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("AMS selected best loss is missing or invalid") from error
    if not math.isfinite(selected_loss):
        raise ValueError("AMS selected best loss must be finite")
    best_payload = torch.load(best, map_location="cpu", weights_only=False)
    last_payload = torch.load(last, map_location="cpu", weights_only=False)
    if best_payload.get("epoch") != selected_epoch:
        raise ValueError("AMS best checkpoint epoch does not match completion best epoch")
    if last_payload.get("epoch") != 8:
        raise ValueError("AMS last checkpoint must record completed epoch 8")
    for label, checkpoint in (("best", best_payload), ("last", last_payload)):
        if checkpoint.get("best") != selected:
            raise ValueError(f"AMS {label} checkpoint selected-best record differs from completion")
        run_config = checkpoint.get("run_config")
        if not isinstance(run_config, dict) or run_config.get("selection") != (
            "lowest fixed-mask real-validation loss over adapted epochs 1-8"
        ):
            raise ValueError(f"AMS {label} checkpoint does not declare real-only selection")
        retired_checkpoint_fields = {
            "base_synthetic_macro_psnr",
            "synthetic_validation_manifest_sha256",
            "max_synthetic_psnr_drop",
        }.intersection(checkpoint) | {
            field
            for field in ("max_synthetic_psnr_drop",)
            if field in run_config
        }
        if retired_checkpoint_fields:
            raise ValueError(
                f"AMS {label} checkpoint contains retired synthetic-selection fields: "
                f"{sorted(retired_checkpoint_fields)}"
            )
    return {
        "kind": "ams",
        "epochs": 8,
        "best_epoch": selected_epoch,
        "identity": best_identity,
    }


def verify_ucm(
    path: Path, payload: dict[str, Any], expected_identity: dict[str, Any]
) -> dict[str, Any]:
    _expect(
        payload,
        {
            "protocol_id": PROTOCOL_ID,
            "formal_run": True,
            "dataset": "ucm_test",
            "pairs": 8_400,
            "sources": 2_100,
        },
        "UCM aggregate",
    )
    credential = _verify_pretest_credential(payload, "UCM aggregate")
    manifest_hash = payload.get("manifest_sha256", payload.get("ucm_test_manifest_sha256"))
    if manifest_hash != credential["ucm_test_manifest_sha256"]:
        raise ValueError("UCM aggregate manifest does not match its pretest credential")
    per_image = _safe_sibling(path.parent, "per_image.csv")
    _verify_hash(per_image, payload.get("per_image_sha256"), "UCM per-image CSV")
    rows = _csv_rows(per_image)
    if "rpsd_error" in rows[0]:
        raise ValueError("UCM artifacts must not contain the retired custom RPSD metric")
    if payload.get("radial_profiles") is not None or payload.get("radial_profiles_sha256") is not None:
        raise ValueError("UCM artifacts must not contain retired radial-profile outputs")
    nested_summaries = [payload.get("macro"), *(payload.get("by_L_class_macro") or {}).values()]
    if any(isinstance(item, dict) and "rpsd_error" in item for item in nested_summaries):
        raise ValueError("UCM aggregate must not contain the retired custom RPSD metric")
    aggregate_identity = _identity(payload, "UCM aggregate")
    _assert_identity(aggregate_identity, expected_identity, "UCM aggregate")
    required = {
        "protocol_id", "formal_run", "dataset", "method", "variant", "seed",
        "pair_id", "source_id", "class_name", "L", "psnr", "ssim",
    }
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"UCM per-image CSV lacks fields: {sorted(missing)}")
    pair_ids = [row.get("pair_id", "") for row in rows]
    source_ids = [row.get("source_id", "") for row in rows]
    if len(rows) != 8_400 or len(set(pair_ids)) != 8_400 or "" in pair_ids:
        raise ValueError("UCM per-image CSV must have 8400 unique pair_id rows")
    source_counts = Counter(source_ids)
    if len(source_counts) != 2_100 or "" in source_counts or set(source_counts.values()) != {4}:
        raise ValueError("UCM per-image CSV must contain exactly 2100 source_id groups")
    if any(
        row.get("protocol_id") != PROTOCOL_ID
        or row.get("dataset") != "ucm_test"
        or row.get("formal_run", "").lower() != "true"
        for row in rows
    ):
        raise ValueError("UCM per-image CSV contains wrong-protocol/non-formal rows")
    grouped: dict[str, dict[str, list[int]]] = {}
    for index, row in enumerate(rows, start=2):
        row_identity = _identity(row, f"UCM CSV row {index}")
        _assert_identity(row_identity, aggregate_identity, f"UCM CSV row {index}")
        try:
            looks = int(row["L"])
            metrics = [float(row[name]) for name in ("psnr", "ssim")]
        except ValueError as error:
            raise ValueError(f"UCM CSV row {index} has invalid numeric values") from error
        if str(looks) != row["L"].strip() or looks not in {1, 2, 4, 8}:
            raise ValueError(f"UCM CSV row {index} has invalid frozen look count")
        if not all(math.isfinite(value) for value in metrics):
            raise ValueError(f"UCM CSV row {index} has non-finite paper metrics")
        grouped.setdefault(row["class_name"], {}).setdefault(row["source_id"], []).append(looks)
    if len(grouped) != 21:
        raise ValueError(f"UCM CSV must contain exactly 21 classes, found {len(grouped)}")
    for class_name, sources in grouped.items():
        if len(sources) != 100 or any(sorted(values) != [1, 2, 4, 8] for values in sources.values()):
            raise ValueError(
                f"UCM class {class_name!r} must contain 100 sources with exactly one row per L"
            )
    return {
        "kind": "ucm", "pairs": len(rows), "identity": aggregate_identity,
        "pretest_seal_sha256": credential["pretest_seal_sha256"],
    }


def verify_real(
    path: Path, payload: dict[str, Any], expected_identity: dict[str, Any]
) -> dict[str, Any]:
    _expect(
        payload,
        {
            "protocol_id": PROTOCOL_ID,
            "formal_run": True,
            "split": "test",
            "num_patches": 592,
            "num_parent_clusters": 148,
        },
        "real-SAR aggregate",
    )
    credential = _verify_pretest_credential(payload, "real-SAR aggregate")
    if (
        payload.get("qc_manifest_sha256") != credential["real_qc_manifest_sha256"]
        or payload.get("qc_csv_sha256") != credential["real_qc_csv_sha256"]
    ):
        raise ValueError("Real-SAR aggregate QC artifacts do not match its pretest credential")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("Real aggregate has no artifacts mapping")
    relative = artifacts.get("real_per_patch_csv")
    if not relative:
        raise ValueError("Real aggregate has no per-patch CSV path")
    per_patch = _safe_sibling(path.parent, str(relative))
    _verify_hash(per_patch, artifacts.get("real_per_patch_sha256"), "real per-patch CSV")
    rows = _csv_rows(per_patch)
    if "qrpsd_error_gt16" in rows[0]:
        raise ValueError("Real-SAR artifacts must not contain the retired custom qRPSD metric")
    aggregate_identity = _identity(payload, "real-SAR aggregate")
    _assert_identity(aggregate_identity, expected_identity, "real-SAR aggregate")
    core_metrics = (
        "ratio_mean_bias", "ratio_acf_sidelobe_energy", "sobel_gc_noisy_output"
    )
    q_metrics = ("qpsnr_gt16", "qssim_gt16")
    required = {
        "protocol_id", "formal_run", "split", "method", "variant", "seed",
        "adaptation", "sample_id", "parent_id", "has_valid_gt16", *core_metrics,
        *q_metrics,
    }
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"Real per-patch CSV lacks fields: {sorted(missing)}")
    if any(str(row.get("has_valid_gt16", "")).strip().lower() != "false" for row in rows):
        raise ValueError("Real per-patch CSV contains GT16-valid rows in the no-GT16 protocol")
    if payload.get("num_valid_gt16_pairs") != 0:
        raise ValueError("Real aggregate has nonzero GT16-valid pairs in the no-GT16 protocol")
    bootstrap = payload.get("parent_cluster_bootstrap")
    if not isinstance(bootstrap, dict):
        raise ValueError("Real aggregate has no parent-cluster bootstrap mapping")
    leaked_q = sorted(set(q_metrics).intersection(bootstrap))
    if leaked_q:
        raise ValueError(
            f"Real aggregate exposes q-reference summaries in the no-GT16 protocol: {leaked_q}"
        )
    sample_ids = [row.get("sample_id", "") for row in rows]
    parent_ids = [row.get("parent_id", "") for row in rows]
    parents = set(parent_ids)
    if len(rows) != 592 or len(set(sample_ids)) != 592 or len(parents) != 148:
        raise ValueError("Real per-patch CSV does not match 592 patches / 148 parents")
    if "" in sample_ids or "" in parents or any(
        row.get("protocol_id") != PROTOCOL_ID
        or row.get("split") != "test"
        or row.get("formal_run", "").lower() != "true"
        for row in rows
    ):
        raise ValueError("Real per-patch CSV contains wrong-protocol/non-formal rows")
    if set(Counter(parent_ids).values()) != {4}:
        raise ValueError("Every real-SAR parent cluster must contain exactly four patches")
    for index, row in enumerate(rows, start=2):
        row_identity = _identity(row, f"real-SAR CSV row {index}")
        _assert_identity(row_identity, aggregate_identity, f"real-SAR CSV row {index}")
        try:
            metrics = [float(row[name]) for name in core_metrics]
        except ValueError as error:
            raise ValueError(f"Real-SAR CSV row {index} has invalid core metrics") from error
        try:
            q_values = [float(row[name]) for name in q_metrics]
        except ValueError as error:
            raise ValueError(f"Real-SAR CSV row {index} has invalid q-metric placeholders") from error
        if not all(math.isnan(value) for value in q_values):
            raise ValueError(
                f"Real-SAR CSV row {index} contains q-metric values in the no-GT16 protocol"
            )
        if not all(math.isfinite(value) for value in metrics):
            raise ValueError(f"Real-SAR CSV row {index} has non-finite core metrics")
    return {
        "kind": "real", "patches": len(rows), "parents": len(parents),
        "identity": aggregate_identity,
        "pretest_seal_sha256": credential["pretest_seal_sha256"],
    }


def verify_sarbm_raw(
    path: Path, payload: dict[str, Any], expected_identity: dict[str, Any]
) -> dict[str, Any]:
    _expect(
        payload,
        {
            "protocol_id": PROTOCOL_ID,
            "formal_run": True,
            "completed_jobs": 8_400,
            "expected_jobs": 8_400,
            "sarbm3d_input_domain": SARBM3D_INPUT_DOMAIN,
            "scored_output_domain": SARBM3D_OUTPUT_DOMAIN,
            "domain_conversion": SARBM3D_DOMAIN_CONVERSION,
        },
        "SAR-BM3D raw completion",
    )
    credential = _verify_pretest_credential(payload, "SAR-BM3D raw completion")
    if payload.get("pair_manifest_sha256") != credential["ucm_test_manifest_sha256"]:
        raise ValueError("SAR-BM3D raw inputs do not match the pretest UCM manifest")
    completion_identity = _identity(payload, "SAR-BM3D raw completion")
    _assert_identity(completion_identity, expected_identity, "SAR-BM3D raw completion")
    relative = payload.get("prediction_manifest")
    if not relative:
        raise ValueError("SAR-BM3D raw completion has no prediction_manifest")
    manifest = _safe_sibling(path.parent, str(relative))
    _verify_hash(manifest, payload.get("prediction_manifest_sha256"), "prediction manifest")
    rows = _csv_rows(manifest)
    if len(rows) != 8_400:
        raise ValueError("SAR-BM3D prediction manifest must have 8400 rows")
    required = {
        "protocol_id", "formal_run", "method", "variant", "pair_id", "L",
        "input_mat_path", "input_mat_sha256", "input_noisy_field",
        "prediction_path", "prediction_sha256", "height", "width",
        "inference_seconds", "raw_min", "raw_max",
        "prediction_amplitude_raw_min", "prediction_amplitude_raw_max",
        "input_domain", "output_domain", "domain_conversion", "job_index",
    }
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"SAR-BM3D prediction manifest lacks fields: {sorted(missing)}")
    pair_ids = [row["pair_id"] for row in rows]
    prediction_paths = [row["prediction_path"] for row in rows]
    if "" in pair_ids or len(pair_ids) != len(set(pair_ids)):
        raise ValueError("SAR-BM3D prediction manifest has duplicate/empty pair_id values")
    if len(prediction_paths) != len(set(prediction_paths)):
        raise ValueError("SAR-BM3D prediction manifest has duplicate prediction paths")
    looks = Counter()
    for index, row in enumerate(rows, start=2):
        if row["protocol_id"] != PROTOCOL_ID or row["formal_run"].lower() != "true":
            raise ValueError(f"SAR-BM3D prediction row {index} is non-formal/wrong-protocol")
        if row["prediction_path"] != f"predictions/{row['pair_id']}.mat":
            raise ValueError(f"SAR-BM3D prediction row {index} has a non-canonical path")
        input_hash = row["input_mat_sha256"].lower()
        if len(input_hash) != 64 or any(character not in "0123456789abcdef" for character in input_hash):
            raise ValueError(f"SAR-BM3D prediction row {index} has an invalid input hash")
        row_identity = _identity(row, f"SAR-BM3D prediction row {index}")
        _assert_identity(row_identity, completion_identity, f"SAR-BM3D prediction row {index}")
        if (
            row["input_domain"] != SARBM3D_INPUT_DOMAIN
            or row["output_domain"] != SARBM3D_OUTPUT_DOMAIN
            or row["domain_conversion"] != SARBM3D_DOMAIN_CONVERSION
        ):
            raise ValueError(f"SAR-BM3D prediction row {index} has wrong domain metadata")
        try:
            look = int(row["L"])
            height = int(row["height"])
            width = int(row["width"])
            job_index = int(row["job_index"])
            numeric = [
                float(row["inference_seconds"]), float(row["raw_min"]), float(row["raw_max"]),
                float(row["prediction_amplitude_raw_min"]),
                float(row["prediction_amplitude_raw_max"]),
            ]
        except ValueError as error:
            raise ValueError(f"SAR-BM3D prediction row {index} has invalid numeric data") from error
        if look not in {1, 2, 4, 8} or (height, width) != (256, 256):
            raise ValueError(f"SAR-BM3D prediction row {index} has invalid L/shape")
        if (
            job_index != index - 1
            or numeric[0] < 0
            or numeric[3] > numeric[4]
            or not all(map(math.isfinite, numeric))
        ):
            raise ValueError(f"SAR-BM3D prediction row {index} has invalid timing/range data")
        looks[look] += 1
        prediction = _safe_sibling(path.parent, row.get("prediction_path", ""))
        _verify_hash(prediction, row.get("prediction_sha256"), "SAR-BM3D prediction")
    if looks != Counter({1: 2_100, 2: 2_100, 4: 2_100, 8: 2_100}):
        raise ValueError("SAR-BM3D prediction manifest must contain exactly 2100 rows per L")
    return {
        "kind": "sarbm_raw", "predictions": len(rows), "identity": completion_identity,
        "pretest_seal_sha256": credential["pretest_seal_sha256"],
    }


VERIFIERS = {
    "supervised": verify_supervised,
    "ams": verify_ams,
    "ucm": verify_ucm,
    "real": verify_real,
    "sarbm_raw": verify_sarbm_raw,
}


def verify(
    kind: str,
    path: str | Path,
    *,
    expected_method: object = _UNSET,
    expected_variant: object = _UNSET,
    expected_seed: object = _UNSET,
    expected_adaptation: object = _UNSET,
) -> dict[str, Any]:
    artifact = Path(path).resolve()
    payload = _load_json(artifact)
    expected = _expected_identity(
        method=expected_method,
        variant=expected_variant,
        seed=expected_seed,
        adaptation=expected_adaptation,
    )
    return {
        "ok": True,
        "path": str(artifact),
        "json_sha256": sha256_file(artifact),
        **VERIFIERS[kind](artifact, payload, expected),
    }


def main() -> int:
    args = parse_args()
    optional = {}
    for argument in ("method", "variant", "seed", "adaptation"):
        value = getattr(args, f"expected_{argument}")
        if value is not None:
            optional[f"expected_{argument}"] = value
    print(json.dumps(verify(args.kind, args.path, **optional), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
