"""Validate SAR-BM3D raw predictions and publish a restart-safe completion chain."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

from experiment_protocol import atomic_json_dump, sha256_file
from icsps2026_protocol import read_csv_records, resolve_portable


PROTOCOL_ID = "ICSPS26-FROZEN-v2"
SARBM3D_INPUT_DOMAIN = "sqrt_linear_normalized_intensity"
SARBM3D_OUTPUT_DOMAIN = "linear_normalized_intensity_after_squaring"
SARBM3D_DOMAIN_CONVERSION = "sqrt_before_SARBM3D_v10;square_after"
REQUIRED_JOB_FIELDS = {
    "pair_id", "mat_path", "noisy_field", "L", "input_mat_sha256",
    "output_relative_path",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", required=True)
    parser.add_argument("--fixed-pair-root", required=True)
    parser.add_argument("--prediction-root", required=True)
    parser.add_argument("--nonformal-smoke", action="store_true")
    return parser.parse_args()


def _mat_scalar(payload: dict[str, Any], field: str, path: Path) -> float:
    if field not in payload:
        raise KeyError(f"Missing MAT variable {field!r}: {path}")
    values = np.asarray(payload[field]).squeeze()
    if values.size != 1:
        raise ValueError(f"MAT variable {field!r} is not scalar: {path}")
    result = float(values.reshape(-1)[0])
    if not np.isfinite(result):
        raise ValueError(f"MAT variable {field!r} is non-finite: {path}")
    return result


def _mat_text(payload: dict[str, Any], field: str, path: Path) -> str:
    if field not in payload:
        raise KeyError(f"Missing MAT variable {field!r}: {path}")
    values = np.asarray(payload[field]).squeeze()
    if values.dtype.kind not in {"U", "S"}:
        raise ValueError(f"MAT variable {field!r} is not text: {path}")
    pieces = values.reshape(-1).tolist()
    result = "".join(
        piece.decode("utf-8") if isinstance(piece, bytes) else str(piece)
        for piece in pieces
    )
    if not result:
        raise ValueError(f"MAT variable {field!r} is empty: {path}")
    return result


def _finite_2d(payload: dict[str, Any], field: str, path: Path) -> np.ndarray:
    if field not in payload:
        raise KeyError(f"Missing MAT variable {field!r}: {path}")
    raw = np.asarray(payload[field])
    if np.iscomplexobj(raw):
        raise ValueError(f"MAT variable {field!r} is complex-valued: {path}")
    array = np.asarray(raw, dtype=np.float64).squeeze()
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError(f"MAT variable {field!r} is not a finite 2-D array: {path}")
    return array


def _atomic_csv(rows: list[dict[str, Any]], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def finalize_raw_predictions(
    jobs_path: str | Path,
    fixed_pair_root: str | Path,
    prediction_root: str | Path,
    *,
    formal: bool,
) -> dict[str, Any]:
    jobs_file = Path(jobs_path).resolve()
    pair_root = Path(fixed_pair_root).resolve()
    output_root = Path(prediction_root).resolve()
    completion_path = output_root / "completion.json"
    if completion_path.exists():
        raise FileExistsError(
            f"Completion marker already exists; verify it instead of replacing it: {completion_path}"
        )
    jobs = read_csv_records(jobs_file)
    if not jobs:
        raise ValueError("SAR-BM3D job manifest is empty")
    missing = REQUIRED_JOB_FIELDS.difference(jobs[0])
    if missing:
        raise ValueError(f"SAR-BM3D job manifest lacks fields: {sorted(missing)}")
    expected_jobs = 8_400 if formal else len(jobs)
    if formal and len(jobs) != expected_jobs:
        raise ValueError(f"Formal SAR-BM3D finalization requires 8400 jobs, found {len(jobs)}")

    pair_ids = [row["pair_id"] for row in jobs]
    if "" in pair_ids or len(pair_ids) != len(set(pair_ids)):
        raise ValueError("SAR-BM3D jobs must have unique non-empty pair_id values")
    if formal and any(re.fullmatch(r"[0-9a-f]{24}", pair_id) is None for pair_id in pair_ids):
        raise ValueError("Formal SAR-BM3D pair_id values must be canonical 24-hex IDs")
    looks = []
    for row in jobs:
        try:
            look = int(row["L"])
        except ValueError as error:
            raise ValueError(f"Invalid L for pair_id={row['pair_id']}") from error
        if str(look) != row["L"].strip() or look not in {1, 2, 4, 8}:
            raise ValueError(f"Invalid frozen L for pair_id={row['pair_id']}: {row['L']!r}")
        looks.append(look)
    if formal and Counter(looks) != Counter({1: 2_100, 2: 2_100, 4: 2_100, 8: 2_100}):
        raise ValueError("Formal SAR-BM3D jobs must contain exactly 2100 jobs per L")

    metadata_path = jobs_file.with_suffix(jobs_file.suffix + ".json")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing SAR-BM3D job metadata: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    expected_metadata = {
        "protocol_id": PROTOCOL_ID,
        "formal_run": formal,
        "jobs": len(jobs),
        "job_manifest_sha256": sha256_file(jobs_file),
        "sarbm3d_input_domain": SARBM3D_INPUT_DOMAIN,
        "scored_output_domain": SARBM3D_OUTPUT_DOMAIN,
        "domain_conversion": SARBM3D_DOMAIN_CONVERSION,
    }
    mismatches = {
        key: {"expected": value, "found": metadata.get(key)}
        for key, value in expected_metadata.items()
        if metadata.get(key) != value or type(metadata.get(key)) is not type(value)
    }
    if mismatches:
        raise ValueError(f"SAR-BM3D job metadata mismatch: {mismatches}")
    pretest_gate = metadata.get("pretest_gate")
    if formal and not isinstance(pretest_gate, dict):
        raise ValueError("Formal SAR-BM3D job metadata lacks the pretest-gate credential")

    manifest_rows: list[dict[str, Any]] = []
    for index, (job, look) in enumerate(zip(jobs, looks), start=1):
        pair_id = job["pair_id"]
        input_path = resolve_portable(pair_root, job["mat_path"])
        if sha256_file(input_path) != job["input_mat_sha256"].lower():
            raise ValueError(f"Input MAT SHA-256 mismatch for pair_id={pair_id}")
        input_payload = loadmat(
            input_path, variable_names=[job["noisy_field"], "global_L"]
        )
        noisy = _finite_2d(input_payload, job["noisy_field"], input_path)
        if float(np.min(noisy)) < 0.0 or float(np.max(noisy)) > 1.0:
            raise ValueError(f"Input noisy intensity is outside [0,1] for pair_id={pair_id}")
        if formal and noisy.shape != (256, 256):
            raise ValueError(
                f"Formal input shape must be 256x256 for pair_id={pair_id}, found {noisy.shape}"
            )
        input_look = _mat_scalar(input_payload, "global_L", input_path)
        if input_look != look:
            raise ValueError(f"Input MAT L mismatch for pair_id={pair_id}")

        expected_relative = f"predictions/{pair_id}.mat"
        if job["output_relative_path"] != expected_relative:
            raise ValueError(
                f"Unexpected output path for pair_id={pair_id}: {job['output_relative_path']!r}"
            )
        prediction_path = resolve_portable(output_root, job["output_relative_path"])
        prediction_payload = loadmat(
            prediction_path,
            variable_names=[
                "prediction", "pair_id", "looks", "inference_seconds",
                "prediction_amplitude_min", "prediction_amplitude_max",
                "input_domain", "output_domain", "domain_conversion",
            ],
        )
        prediction = _finite_2d(prediction_payload, "prediction", prediction_path)
        if prediction.shape != noisy.shape:
            raise ValueError(
                f"Prediction/input shape mismatch for pair_id={pair_id}: "
                f"{prediction.shape} != {noisy.shape}"
            )
        if _mat_text(prediction_payload, "pair_id", prediction_path) != pair_id:
            raise ValueError(f"Prediction MAT pair_id mismatch for {pair_id}")
        prediction_look = _mat_scalar(prediction_payload, "looks", prediction_path)
        if prediction_look != look:
            raise ValueError(f"Prediction MAT L mismatch for pair_id={pair_id}")
        input_domain = _mat_text(prediction_payload, "input_domain", prediction_path)
        output_domain = _mat_text(prediction_payload, "output_domain", prediction_path)
        conversion = _mat_text(prediction_payload, "domain_conversion", prediction_path)
        if (
            input_domain != SARBM3D_INPUT_DOMAIN
            or output_domain != SARBM3D_OUTPUT_DOMAIN
            or conversion != SARBM3D_DOMAIN_CONVERSION
        ):
            raise ValueError(f"Prediction MAT domain declaration mismatch for {pair_id}")
        amplitude_min = _mat_scalar(
            prediction_payload, "prediction_amplitude_min", prediction_path
        )
        amplitude_max = _mat_scalar(
            prediction_payload, "prediction_amplitude_max", prediction_path
        )
        if amplitude_min > amplitude_max:
            raise ValueError(f"Prediction amplitude range is reversed for {pair_id}")
        inference_seconds = _mat_scalar(
            prediction_payload, "inference_seconds", prediction_path
        )
        if inference_seconds < 0:
            raise ValueError(f"Negative inference time for pair_id={pair_id}")
        manifest_rows.append(
            {
                "protocol_id": PROTOCOL_ID,
                "formal_run": formal,
                "method": "sar_bm3d",
                "variant": "v1.0",
                "pair_id": pair_id,
                "L": look,
                "input_mat_path": job["mat_path"],
                "input_mat_sha256": job["input_mat_sha256"].lower(),
                "input_noisy_field": job["noisy_field"],
                "prediction_path": job["output_relative_path"],
                "prediction_sha256": sha256_file(prediction_path),
                "height": prediction.shape[0],
                "width": prediction.shape[1],
                "inference_seconds": inference_seconds,
                "raw_min": float(np.min(prediction)),
                "raw_max": float(np.max(prediction)),
                "prediction_amplitude_raw_min": amplitude_min,
                "prediction_amplitude_raw_max": amplitude_max,
                "input_domain": input_domain,
                "output_domain": output_domain,
                "domain_conversion": conversion,
                "job_index": index,
            }
        )

    output_root.mkdir(parents=True, exist_ok=True)
    prediction_manifest = output_root / "prediction_manifest.csv"
    _atomic_csv(manifest_rows, prediction_manifest)
    completion = {
        "protocol_id": PROTOCOL_ID,
        "formal_run": formal,
        "method": "sar_bm3d",
        "variant": "v1.0",
        "seed": None,
        "adaptation": None,
        "sarbm3d_input_domain": SARBM3D_INPUT_DOMAIN,
        "scored_output_domain": SARBM3D_OUTPUT_DOMAIN,
        "domain_conversion": SARBM3D_DOMAIN_CONVERSION,
        "completed_jobs": len(manifest_rows),
        "expected_jobs": expected_jobs,
        "job_manifest_sha256": sha256_file(jobs_file),
        "job_metadata_sha256": sha256_file(metadata_path),
        "pair_manifest_sha256": metadata.get("pair_manifest_sha256"),
        "prediction_manifest": prediction_manifest.name,
        "prediction_manifest_sha256": sha256_file(prediction_manifest),
        "pretest_gate": pretest_gate,
    }
    atomic_json_dump(completion, completion_path)
    return completion


def main() -> int:
    args = parse_args()
    completion = finalize_raw_predictions(
        args.jobs,
        args.fixed_pair_root,
        args.prediction_root,
        formal=not args.nonformal_smoke,
    )
    print(json.dumps(completion, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
