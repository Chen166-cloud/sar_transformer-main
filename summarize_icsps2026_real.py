"""Build predeclared paired parent-cluster comparisons for formal real SAR."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from experiment_protocol import sha256_file
from verify_icsps2026_artifact import verify as verify_artifact


PROTOCOL_ID = "ICSPS26-FROZEN-v2"
CORE_METRICS = (
    "ratio_mean_bias",
    "ratio_acf_sidelobe_energy",
    "sobel_gc_noisy_output",
)
Q_METRICS = ("qpsnr_gt16", "qssim_gt16")
LOWER_IS_BETTER = {
    "ratio_mean_bias",
    "ratio_acf_sidelobe_energy",
}
DIAGNOSTIC_METRICS = {"sobel_gc_noisy_output"}


def parse_key_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected METHOD_KEY=/path/to/real_per_patch.csv")
    key, path = value.split("=", 1)
    if not key or not path:
        raise argparse.ArgumentTypeError("Expected METHOD_KEY=/path/to/real_per_patch.csv")
    return key, Path(path).resolve()


def parse_comparison(value: str) -> tuple[str, str]:
    parts = value.split(":")
    if len(parts) != 2 or not all(parts):
        raise argparse.ArgumentTypeError("Expected TARGET_KEY:BASELINE_KEY")
    return parts[0], parts[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=parse_key_path, required=True)
    parser.add_argument(
        "--compare", action="append", type=parse_comparison, required=True,
        help=(
            "Predeclared TARGET_KEY:BASELINE_KEY; directional metrics orient positive "
            "toward target, while Sobel-GC remains descriptive"
        ),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260905)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _finite(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _canonical_identity_value(field: str, value: object) -> str | int | None:
    text = "" if value is None else str(value).strip()
    if field == "method":
        if not text:
            raise ValueError("method is missing")
        return text.lower().replace("-", "_").replace(" ", "_")
    if field == "seed":
        if text.lower() in {"", "none", "null"}:
            return None
        try:
            parsed = int(text)
        except ValueError as error:
            raise ValueError(f"seed is not an integer: {value!r}") from error
        if str(parsed) != text:
            raise ValueError(f"seed is not a canonical integer: {value!r}")
        return parsed
    normalized = text.lower()
    return None if normalized in {"", "none", "null"} else normalized


def _identity(mapping: dict[str, Any]) -> dict[str, object]:
    fields = ("method", "variant", "seed", "adaptation")
    return {
        field: _canonical_identity_value(field, mapping.get(field))
        for field in fields
    }


def read_and_validate(
    key: str, path: Path
) -> tuple[list[dict[str, str]], dict[str, Any], dict[str, object]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "protocol_id", "method", "variant", "seed", "adaptation",
        "sample_id", "parent_id", "split", "formal_run", "has_valid_gt16",
        *CORE_METRICS, *Q_METRICS,
    }
    if not rows or required.difference(rows[0]):
        raise ValueError(f"{key} is empty or lacks fields: {sorted(required.difference(rows[0] if rows else {}))}")
    ids = [row["sample_id"] for row in rows]
    parents = {row["parent_id"] for row in rows}
    if len(rows) != 592 or len(parents) != 148 or len(ids) != len(set(ids)):
        raise ValueError(f"{key} must contain 592 unique patches from 148 parents")
    parent_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        parent_counts[row["parent_id"]] += 1
    if set(parent_counts.values()) != {4}:
        raise ValueError(f"{key} must contain exactly four patches per parent")
    if any(
        row["protocol_id"] != PROTOCOL_ID
        or row["split"] != "test"
        or not _bool(row["formal_run"])
        for row in rows
    ):
        raise ValueError(f"{key} contains non-formal or non-test rows")
    identity_fields = ("method", "variant", "seed", "adaptation")
    row_identity_tuples = {
        tuple(_canonical_identity_value(field, row.get(field)) for field in identity_fields)
        for row in rows
    }
    if len(row_identity_tuples) != 1:
        raise ValueError(f"{key} contains mixed method/variant/seed/adaptation identities")
    row_identity = dict(
        zip(("method", "variant", "seed", "adaptation"), next(iter(row_identity_tuples)))
    )
    for row in rows:
        for metric in CORE_METRICS:
            if _finite(row[metric]) is None:
                raise ValueError(f"{key}/{row['sample_id']} has non-finite {metric}")
        if _bool(row["has_valid_gt16"]):
            for metric in Q_METRICS:
                if _finite(row[metric]) is None:
                    raise ValueError(f"{key}/{row['sample_id']} has invalid GT16 {metric}")

    aggregate_path = path.parent / "aggregate.json"
    if not aggregate_path.is_file():
        raise FileNotFoundError(f"{key} has no sibling aggregate.json: {aggregate_path}")
    with aggregate_path.open("r", encoding="utf-8") as handle:
        aggregate = json.load(handle)
    expected = {
        "protocol_id": PROTOCOL_ID,
        "formal_run": True,
        "split": "test",
        "num_patches": 592,
        "num_parent_clusters": 148,
    }
    mismatch = {
        field: {"expected": value, "found": aggregate.get(field)}
        for field, value in expected.items() if aggregate.get(field) != value
    }
    if mismatch:
        raise ValueError(f"{key} aggregate is not a formal ICSPS result: {mismatch}")
    artifacts = aggregate.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError(f"{key} aggregate lacks its artifact hash chain")
    declared_path = (aggregate_path.parent / str(artifacts.get("real_per_patch_csv", ""))).resolve()
    if declared_path != path.resolve():
        raise ValueError(f"{key} aggregate points to a different real_per_patch CSV")
    actual_hash = sha256_file(path)
    if artifacts.get("real_per_patch_sha256") != actual_hash:
        raise ValueError(f"{key} real_per_patch CSV SHA-256 does not match aggregate")
    aggregate_identity = _identity(aggregate)
    if aggregate_identity != row_identity:
        raise ValueError(
            f"{key} aggregate/row identity mismatch: "
            f"aggregate={aggregate_identity}, rows={row_identity}"
        )
    verify_artifact(
        "real", aggregate_path,
        expected_method=row_identity["method"],
        expected_variant=row_identity["variant"],
        expected_seed=row_identity["seed"],
        expected_adaptation=row_identity["adaptation"],
    )
    return rows, aggregate, aggregate_identity


def paired_parent_bootstrap(
    target_rows: list[dict[str, str]],
    baseline_rows: list[dict[str, str]],
    metric: str,
    samples: int,
    seed: int,
) -> dict[str, float | int | str]:
    target = {row["sample_id"]: row for row in target_rows}
    baseline = {row["sample_id"]: row for row in baseline_rows}
    if set(target) != set(baseline):
        raise ValueError("Paired inputs do not have identical sample_id sets")
    grouped: dict[str, list[float]] = defaultdict(list)
    paired_count = 0
    for sample_id in sorted(target):
        left, right = target[sample_id], baseline[sample_id]
        if left["parent_id"] != right["parent_id"]:
            raise ValueError(f"Parent mismatch for sample_id={sample_id}")
        if metric in Q_METRICS and not (
            _bool(left["has_valid_gt16"]) and _bool(right["has_valid_gt16"])
        ):
            continue
        left_value, right_value = _finite(left[metric]), _finite(right[metric])
        if left_value is None or right_value is None:
            if metric in Q_METRICS:
                continue
            raise ValueError(f"Non-finite paired core metric for sample_id={sample_id}")
        raw_difference = left_value - right_value
        advantage = -raw_difference if metric in LOWER_IS_BETTER else raw_difference
        grouped[left["parent_id"]].append(advantage)
        paired_count += 1
    if not grouped:
        raise ValueError(f"No common valid pairs for {metric}")
    parent_ids = sorted(grouped)
    parent_sums = np.asarray([sum(grouped[key]) for key in parent_ids], dtype=np.float64)
    parent_counts = np.asarray([len(grouped[key]) for key in parent_ids], dtype=np.float64)
    rng = np.random.default_rng(seed)
    replicates = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        selected = rng.integers(0, len(parent_ids), size=len(parent_ids))
        replicates[index] = float(
            np.sum(parent_sums[selected]) / np.sum(parent_counts[selected])
        )
    observed = float(np.sum(parent_sums) / np.sum(parent_counts))
    return {
        "metric": metric,
        "advantage_definition": (
            "target_minus_baseline_descriptive_difference"
            if metric in DIAGNOSTIC_METRICS
            else ("baseline_minus_target" if metric in LOWER_IS_BETTER else "target_minus_baseline")
        ),
        "positive_favors": (
            "not_defined_diagnostic" if metric in DIAGNOSTIC_METRICS else "target"
        ),
        "advantage": observed,
        "advantage_ci95_low": float(np.percentile(replicates, 2.5)),
        "advantage_ci95_high": float(np.percentile(replicates, 97.5)),
        "paired_patches": paired_count,
        "paired_parent_clusters": len(parent_ids),
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "cluster": "parent_id; all child patches retained",
        "pair_scope": (
            "common valid GT16 intersection" if metric in Q_METRICS else "all formal test patches"
        ),
    }


def main() -> int:
    args = parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("bootstrap-samples must be positive")
    inputs = dict(args.input)
    if len(inputs) != len(args.input):
        raise ValueError("Duplicate --input method key")
    if len(set(args.compare)) != len(args.compare):
        raise ValueError("Duplicate --compare declaration")
    missing_keys = sorted({key for pair in args.compare for key in pair}.difference(inputs))
    if missing_keys:
        raise ValueError(f"Comparison keys are absent from --input: {missing_keys}")

    loaded = {key: read_and_validate(key, path) for key, path in inputs.items()}
    pretest_bindings = {
        key: {
            field: aggregate["pretest_gate"].get(field)
            for field in (
                "pretest_seal_sha256", "real_qc_manifest_sha256",
                "real_qc_csv_sha256", "validation_selection_sha256",
                "figure_selection_sha256", "real_use_gt16",
                "real_evaluation_scope", "real_valid_quasi_reference",
            )
        }
        for key, (_, aggregate, _) in loaded.items()
    }
    if len({json.dumps(value, sort_keys=True) for value in pretest_bindings.values()}) != 1:
        raise ValueError(f"Formal real inputs use different pretest seals/QC: {pretest_bindings}")
    common_pretest_binding = next(iter(pretest_bindings.values()))
    if (
        common_pretest_binding.get("real_use_gt16") is not False
        or common_pretest_binding.get("real_evaluation_scope") != "no_reference_only"
        or common_pretest_binding.get("real_valid_quasi_reference") != 0
    ):
        raise ValueError("Formal real summary requires the frozen no-GT16 pretest binding")
    for target_key, baseline_key in args.compare:
        if "noisy" in {
            loaded[target_key][2]["method"], loaded[baseline_key][2]["method"]
        }:
            raise ValueError(
                "Noisy input cannot be used in a paired comparison because its "
                "ratio and Sobel-GC relationships are tautological"
            )
    reference = None
    for key, (rows, _, _) in loaded.items():
        identity = {row["sample_id"]: row["parent_id"] for row in rows}
        if reference is None:
            reference = identity
        elif identity != reference:
            raise ValueError(f"{key} has a different sample/parent mapping")

    result_rows = []
    skipped_q_metrics = [
        {
            "target": target_key,
            "baseline": baseline_key,
            "metric": metric,
            "reason": "disabled by ICSPS26-FROZEN-v2 no-GT16 protocol",
        }
        for target_key, baseline_key in args.compare
        for metric in Q_METRICS
    ]
    for target_key, baseline_key in args.compare:
        for metric in CORE_METRICS:
            result = paired_parent_bootstrap(
                loaded[target_key][0], loaded[baseline_key][0], metric,
                args.bootstrap_samples, args.bootstrap_seed,
            )
            result_rows.append(
                {"target_key": target_key, "baseline_key": baseline_key, **result}
            )

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    paired_path = output / "real_paired_parent_bootstrap.csv"
    with paired_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result_rows[0]))
        writer.writeheader()
        writer.writerows(result_rows)
    provenance = {
        "protocol_id": PROTOCOL_ID,
        "formal_run": True,
        "pretest_binding": common_pretest_binding,
        "inputs": {
            key: {
                "per_patch_csv": str(path),
                "per_patch_sha256": sha256_file(path),
                "aggregate_json": str(path.parent / "aggregate.json"),
                "aggregate_sha256": sha256_file(path.parent / "aggregate.json"),
                "verified_identity": loaded[key][2],
            }
            for key, path in inputs.items()
        },
        "comparisons": [
            {"target": target, "baseline": baseline} for target, baseline in args.compare
        ],
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "positive_favors_by_metric": {
            metric: ("not_defined_diagnostic" if metric in DIAGNOSTIC_METRICS else "target")
            for metric in CORE_METRICS
        },
        "real_evaluation_scope": "no_reference_only",
        "forbidden_paper_metrics": list(Q_METRICS),
        "q_metric_scope": "disabled by frozen no-GT16 protocol",
        "skipped_q_metrics": skipped_q_metrics,
        "paired_csv": paired_path.name,
        "paired_csv_sha256": sha256_file(paired_path),
    }
    with (output / "summary_provenance.json").open("x", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    print(json.dumps(provenance, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
