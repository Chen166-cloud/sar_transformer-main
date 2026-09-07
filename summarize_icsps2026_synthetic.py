"""Rebuild the UCM main table and paired source-cluster confidence intervals."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiment_protocol import sha256_file
from verify_icsps2026_artifact import verify as verify_artifact


METRICS = ("psnr", "ssim")
SUMMARY_METRICS = METRICS
FORMAL_LOOKS = {1, 2, 4, 8}


def parse_key_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected METHOD_KEY=/path/to/per_image.csv")
    key, path = value.split("=", 1)
    if not key or not path:
        raise argparse.ArgumentTypeError("Expected METHOD_KEY=/path/to/per_image.csv")
    return key, Path(path).resolve()


def parse_comparison(value: str) -> tuple[str, str]:
    parts = value.split(":")
    if len(parts) != 2 or not all(parts):
        raise argparse.ArgumentTypeError("Expected OURS_KEY:BASELINE_KEY")
    return parts[0], parts[1]


def resolve_comparison_specs(
    comparisons: list[tuple[str, str]],
    ours_key: str | None,
    baseline_key: str | None,
) -> list[tuple[str, str]]:
    if (ours_key is None) != (baseline_key is None):
        raise ValueError("Supply both --ours-key and --baseline-key, or neither")
    resolved = list(comparisons)
    if ours_key is not None and baseline_key is not None:
        resolved.append((ours_key, baseline_key))
    if len(set(resolved)) != len(resolved):
        raise ValueError("Duplicate predeclared comparison")
    return resolved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=parse_key_path, required=True)
    parser.add_argument("--ours-key")
    parser.add_argument("--baseline-key")
    parser.add_argument(
        "--compare", action="append", type=parse_comparison, default=[],
        help="Repeatable predeclared OURS_KEY:BASELINE_KEY comparison",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260905)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--allow-nonformal", action="store_true")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with open(path, "r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows


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


def validate_input_identity(key: str, rows: list[dict[str, str]]) -> dict[str, object]:
    fields = ("method", "variant", "seed")
    missing = set(fields).difference(rows[0])
    if missing:
        raise ValueError(f"{key} lacks identity fields: {sorted(missing)}")
    identities = {
        tuple(_canonical_identity_value(field, row.get(field)) for field in fields)
        for row in rows
    }
    if len(identities) != 1:
        raise ValueError(f"{key} contains mixed method/variant/seed identities")
    identity = next(iter(identities))
    return dict(zip(fields, identity))


def validate_input_aggregate(
    key: str,
    per_image_path: Path,
    row_identity: dict[str, object],
    *,
    formal_required: bool,
) -> tuple[Path, dict[str, object]]:
    aggregate_path = per_image_path.parent / "aggregate.json"
    if not aggregate_path.is_file():
        raise FileNotFoundError(f"{key} has no sibling aggregate.json: {aggregate_path}")
    with aggregate_path.open("r", encoding="utf-8") as handle:
        aggregate = json.load(handle)
    if not isinstance(aggregate, dict):
        raise ValueError(f"{key} aggregate must be a JSON object")
    if aggregate.get("per_image_sha256") != sha256_file(per_image_path):
        raise ValueError(f"{key} per_image CSV SHA-256 does not match aggregate")
    if formal_required:
        expected = {
            "protocol_id": "ICSPS26-FROZEN-v2",
            "formal_run": True,
            "dataset": "ucm_test",
            "pairs": 8_400,
            "sources": 2_100,
        }
        mismatch = {
            field: {"expected": value, "found": aggregate.get(field)}
            for field, value in expected.items()
            if aggregate.get(field) != value
        }
        if mismatch:
            raise ValueError(f"{key} aggregate is not a complete formal UCM result: {mismatch}")
    aggregate_identity = {
        field: _canonical_identity_value(field, aggregate.get(field))
        for field in ("method", "variant", "seed")
    }
    if aggregate_identity != row_identity:
        raise ValueError(
            f"{key} aggregate/row identity mismatch: "
            f"aggregate={aggregate_identity}, rows={row_identity}"
        )
    if formal_required:
        verify_artifact(
            "ucm", aggregate_path,
            expected_method=row_identity["method"],
            expected_variant=row_identity["variant"],
            expected_seed=row_identity["seed"],
        )
    return aggregate_path, aggregate_identity


def pretest_binding(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        aggregate = json.load(handle)
    gate = aggregate.get("pretest_gate")
    if not isinstance(gate, dict):
        raise ValueError(f"Formal aggregate lacks pretest credential: {path}")
    return {
        "pretest_seal_sha256": gate.get("pretest_seal_sha256"),
        "ucm_test_manifest_sha256": gate.get("ucm_test_manifest_sha256"),
        "validation_selection_sha256": gate.get("validation_selection_sha256"),
        "figure_selection_sha256": gate.get("figure_selection_sha256"),
    }


def cell_macro(rows: list[dict[str, str]], metric: str) -> float:
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["class_name"], int(row["L"]))].append(float(row[metric]))
    return float(np.mean([np.mean(values) for values in grouped.values()]))


def validate_formal_ucm_rows(key: str, rows: list[dict[str, str]]) -> None:
    required = {
        "protocol_id", "formal_run", "dataset", "pair_id", "source_id",
        "class_name", "L", *SUMMARY_METRICS,
    }
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"{key} lacks required fields: {sorted(missing)}")
    if len(rows) != 8400 or any(row["formal_run"].lower() != "true" for row in rows):
        raise ValueError(f"{key} is not a complete 8400-pair formal result")
    if any(
        row["protocol_id"] != "ICSPS26-FROZEN-v2"
        or row["dataset"] != "ucm_test"
        for row in rows
    ):
        raise ValueError(f"{key} contains non-UCM or wrong-protocol rows")
    grouped: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        looks = int(row["L"])
        if looks not in FORMAL_LOOKS:
            raise ValueError(f"{key} contains unexpected L={looks}")
        if not all(np.isfinite(float(row[metric])) for metric in SUMMARY_METRICS):
            raise ValueError(f"{key} contains a non-finite paper metric")
        grouped[row["class_name"]][row["source_id"]].append(looks)
    if len(grouped) != 21:
        raise ValueError(f"{key} must contain 21 UCM classes, found {len(grouped)}")
    for class_name, sources in grouped.items():
        if len(sources) != 100:
            raise ValueError(
                f"{key} class {class_name!r} must contain 100 sources, found {len(sources)}"
            )
        if any(sorted(looks) != sorted(FORMAL_LOOKS) for looks in sources.values()):
            raise ValueError(
                f"{key} class {class_name!r} does not contain exactly one row per frozen L"
            )


def paired_cluster_bootstrap(
    ours: list[dict[str, str]],
    baseline: list[dict[str, str]],
    metric: str,
    samples: int,
    seed: int,
) -> dict[str, float | int | str]:
    ours_by_pair = {row["pair_id"]: row for row in ours}
    baseline_by_pair = {row["pair_id"]: row for row in baseline}
    if set(ours_by_pair) != set(baseline_by_pair):
        raise ValueError("Paired bootstrap inputs do not contain identical pair_id sets")
    by_class_source: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for pair_id in sorted(ours_by_pair):
        ours_row = ours_by_pair[pair_id]
        baseline_row = baseline_by_pair[pair_id]
        identity = (ours_row["source_id"], ours_row["class_name"], ours_row["L"])
        other_identity = (
            baseline_row["source_id"], baseline_row["class_name"], baseline_row["L"]
        )
        if identity != other_identity:
            raise ValueError(f"Metadata mismatch for pair {pair_id}")
        difference = float(ours_row[metric]) - float(baseline_row[metric])
        by_class_source[ours_row["class_name"]][ours_row["source_id"]].append(difference)
    if len(by_class_source) != 21:
        raise ValueError(f"Formal UCM bootstrap requires 21 classes, found {len(by_class_source)}")
    class_vectors = []
    for class_name in sorted(by_class_source):
        source_values = by_class_source[class_name]
        if len(source_values) != 100 or any(len(values) != 4 for values in source_values.values()):
            raise ValueError(
                f"Class {class_name} must have 100 sources with four L values each"
            )
        class_vectors.append(
            np.asarray([np.mean(source_values[key]) for key in sorted(source_values)], dtype=np.float64)
        )
    observed_raw = float(np.mean([np.mean(values) for values in class_vectors]))
    rng = np.random.default_rng(seed)
    replicates = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        class_means = []
        for values in class_vectors:
            sampled = rng.integers(0, len(values), size=len(values))
            class_means.append(float(np.mean(values[sampled])))
        replicates[index] = float(np.mean(class_means))
    favorable = replicates
    return {
        "metric": metric,
        "raw_difference_definition": "ours_minus_baseline",
        "raw_difference": observed_raw,
        "advantage_definition": "ours_minus_baseline",
        "advantage": observed_raw,
        "advantage_ci95_low": float(np.percentile(favorable, 2.5)),
        "advantage_ci95_high": float(np.percentile(favorable, 97.5)),
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "cluster": "UCM source within class; all four L retained",
    }


def main() -> int:
    args = parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("bootstrap-samples must be positive")
    comparison_specs = resolve_comparison_specs(
        args.compare, args.ours_key, args.baseline_key
    )
    inputs = dict(args.input)
    if len(inputs) != len(args.input):
        raise ValueError("Duplicate --input method key")
    all_rows = {key: read_rows(path) for key, path in inputs.items()}
    identities = {
        key: validate_input_identity(key, rows) for key, rows in all_rows.items()
    }
    reference_ids = None
    for key, rows in all_rows.items():
        pair_ids = {row["pair_id"] for row in rows}
        if len(rows) != len(pair_ids):
            raise ValueError(f"Duplicate pair_id in {key}")
        if not args.allow_nonformal:
            validate_formal_ucm_rows(key, rows)
        if reference_ids is None:
            reference_ids = pair_ids
        elif pair_ids != reference_ids:
            raise ValueError(f"Pair set mismatch for {key}")
    aggregates = {
        key: validate_input_aggregate(
            key, inputs[key], identities[key], formal_required=not args.allow_nonformal
        )
        for key in inputs
    }
    common_pretest_binding = None
    if not args.allow_nonformal:
        bindings = {key: pretest_binding(value[0]) for key, value in aggregates.items()}
        distinct = {json.dumps(value, sort_keys=True) for value in bindings.values()}
        if len(distinct) != 1:
            raise ValueError(f"Formal inputs use different pretest seals/manifests: {bindings}")
        common_pretest_binding = next(iter(bindings.values()))

    summary_rows = []
    for key, rows in all_rows.items():
        for looks in sorted({int(row["L"]) for row in rows}):
            subset = [row for row in rows if int(row["L"]) == looks]
            summary_rows.append(
                {
                    "method_key": key,
                    **identities[key],
                    "L": looks,
                    "pairs": len(subset),
                    **{metric: cell_macro(subset, metric) for metric in SUMMARY_METRICS},
                }
            )
        summary_rows.append(
            {
                "method_key": key,
                **identities[key],
                "L": "macro",
                "pairs": len(rows),
                **{metric: cell_macro(rows, metric) for metric in SUMMARY_METRICS},
            }
        )

    provenance = {
        "protocol_id": "ICSPS26-FROZEN-v2",
        "input_keys_are_labels_only": True,
        "inputs": {
            key: {
                "per_image_csv": str(path),
                "per_image_sha256": sha256_file(path),
                "verified_row_identity": identities[key],
                "aggregate_json": str(aggregates[key][0]),
                "aggregate_sha256": sha256_file(aggregates[key][0]),
            }
            for key, path in inputs.items()
        },
        "formal_required": not args.allow_nonformal,
        "pretest_binding": common_pretest_binding,
    }
    missing_keys = sorted(
        {key for comparison in comparison_specs for key in comparison}.difference(all_rows)
    )
    if missing_keys:
        raise ValueError(f"Comparison keys are absent from --input: {missing_keys}")
    comparisons = [
        {
            "ours_key": ours_key,
            "baseline_key": baseline_key,
            **paired_cluster_bootstrap(
                all_rows[ours_key], all_rows[baseline_key], metric,
                args.bootstrap_samples, args.bootstrap_seed,
            ),
        }
        for ours_key, baseline_key in comparison_specs
        for metric in METRICS
    ]
    if comparison_specs:
        provenance["predeclared_comparisons"] = [
            {
                "ours": ours_key,
                "baseline": baseline_key,
                "bootstrap_samples": args.bootstrap_samples,
                "bootstrap_seed": args.bootstrap_seed,
            }
            for ours_key, baseline_key in comparison_specs
        ]
    if args.ours_key is not None:
        # Retain the old singular provenance key for downstream compatibility.
        provenance["predeclared_comparison"] = provenance["predeclared_comparisons"][-1]
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    main_summary_path = output / "ucm_main_summary.csv"
    paired_summary_path = output / "ucm_paired_cluster_bootstrap.csv"
    with open(main_summary_path, "x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    if comparisons:
        with open(paired_summary_path, "x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]))
            writer.writeheader()
            writer.writerows(comparisons)
    provenance["outputs"] = {
        "ucm_main_summary_csv": main_summary_path.name,
        "ucm_main_summary_sha256": sha256_file(main_summary_path),
        "ucm_paired_cluster_bootstrap_csv": (
            paired_summary_path.name if comparisons else None
        ),
        "ucm_paired_cluster_bootstrap_sha256": (
            sha256_file(paired_summary_path) if comparisons else None
        ),
    }
    with open(output / "summary_provenance.json", "x", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    print(json.dumps(provenance, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
