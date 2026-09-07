"""Summarize three-seed UCM runs and bootstrap seed-averaged paired effects."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiment_protocol import sha256_file
from summarize_icsps2026_synthetic import (
    METRICS,
    cell_macro,
    paired_cluster_bootstrap,
    parse_comparison,
    parse_key_path,
    read_rows,
    validate_formal_ucm_rows,
    validate_input_aggregate,
    validate_input_identity,
    pretest_binding,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--member", action="append", type=parse_key_path, required=True,
        help="Repeat three times per group as GROUP=/path/to/per_image.csv",
    )
    parser.add_argument(
        "--compare", action="append", type=parse_comparison, required=True,
        help="Repeatable predeclared TARGET_GROUP:BASELINE_GROUP",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260905)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def average_seed_rows(seed_rows: list[list[dict[str, str]]]) -> list[dict[str, object]]:
    """Average metrics over seeds for each frozen pair while preserving metadata."""

    if not seed_rows:
        raise ValueError("No seed rows supplied")
    indexed = [{row["pair_id"]: row for row in rows} for rows in seed_rows]
    pair_ids = set(indexed[0])
    if any(set(values) != pair_ids for values in indexed[1:]):
        raise ValueError("Seed members do not contain identical pair_id sets")
    output = []
    for pair_id in sorted(pair_ids):
        rows = [values[pair_id] for values in indexed]
        identity = (rows[0]["source_id"], rows[0]["class_name"], rows[0]["L"])
        if any((row["source_id"], row["class_name"], row["L"]) != identity for row in rows[1:]):
            raise ValueError(f"Seed metadata mismatch for pair_id={pair_id}")
        output.append(
            {
                "pair_id": pair_id,
                "source_id": identity[0],
                "class_name": identity[1],
                "L": identity[2],
                **{
                    metric: float(np.mean([float(row[metric]) for row in rows]))
                    for metric in METRICS
                },
            }
        )
    return output


def main() -> int:
    args = parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("bootstrap-samples must be positive")
    if len(set(args.compare)) != len(args.compare):
        raise ValueError("Duplicate --compare declaration")

    grouped_paths: dict[str, list[Path]] = defaultdict(list)
    for key, path in args.member:
        grouped_paths[key].append(path)
    required_groups = {key for comparison in args.compare for key in comparison}
    missing = sorted(required_groups.difference(grouped_paths))
    if missing:
        raise ValueError(f"Comparison groups have no members: {missing}")

    group_data: dict[str, dict[str, object]] = {}
    all_pair_ids: set[str] | None = None
    common_pretest_binding: dict[str, object] | None = None
    for key, paths in grouped_paths.items():
        if len(paths) != 3 or len(set(paths)) != 3:
            raise ValueError(f"Group {key!r} must contain exactly three distinct CSV paths")
        members = []
        for path in paths:
            rows = read_rows(path)
            validate_formal_ucm_rows(f"{key}:{path}", rows)
            identity = validate_input_identity(f"{key}:{path}", rows)
            aggregate_path, _ = validate_input_aggregate(
                f"{key}:{path}", path, identity, formal_required=True
            )
            member_binding = pretest_binding(aggregate_path)
            if common_pretest_binding is None:
                common_pretest_binding = member_binding
            elif member_binding != common_pretest_binding:
                raise ValueError("All multi-seed members must use one pretest seal/manifest")
            pair_ids = {row["pair_id"] for row in rows}
            if all_pair_ids is None:
                all_pair_ids = pair_ids
            elif pair_ids != all_pair_ids:
                raise ValueError("All seed groups must use the identical frozen pair set")
            members.append(
                {
                    "path": path,
                    "rows": rows,
                    "identity": identity,
                    "aggregate_path": aggregate_path,
                }
            )
        method_variants = {
            (item["identity"]["method"], item["identity"]["variant"])
            for item in members
        }
        seeds = {item["identity"]["seed"] for item in members}
        if len(method_variants) != 1 or seeds != {42, 43, 44}:
            raise ValueError(
                f"Group {key!r} must be one method/variant with seeds 42/43/44; "
                f"found identities={method_variants}, seeds={seeds}"
            )
        members.sort(key=lambda item: int(item["identity"]["seed"]))
        method, variant = next(iter(method_variants))
        group_data[key] = {
            "method": method,
            "variant": variant,
            "members": members,
            "averaged_rows": average_seed_rows([item["rows"] for item in members]),
        }

    summary_rows = []
    for key, group in group_data.items():
        members = group["members"]
        for looks in (1, 2, 4, 8, "macro"):
            for metric in METRICS:
                values = []
                for item in members:
                    rows = item["rows"]
                    subset = rows if looks == "macro" else [row for row in rows if int(row["L"]) == looks]
                    values.append(cell_macro(subset, metric))
                summary_rows.append(
                    {
                        "protocol_id": "ICSPS26-FROZEN-v2",
                        "group_key": key,
                        "method": group["method"],
                        "variant": group["variant"] or "",
                        "L": looks,
                        "metric": metric,
                        "seed42": values[0],
                        "seed43": values[1],
                        "seed44": values[2],
                        "seed_mean": statistics.mean(values),
                        "seed_sample_sd": statistics.stdev(values),
                        "seed_count": 3,
                    }
                )

    comparison_rows = []
    for target, baseline in args.compare:
        for metric in METRICS:
            comparison_rows.append(
                {
                    "protocol_id": "ICSPS26-FROZEN-v2",
                    "target_group": target,
                    "baseline_group": baseline,
                    "seed_aggregation": "per-pair mean over seeds 42/43/44 before bootstrap",
                    **paired_cluster_bootstrap(
                        group_data[target]["averaged_rows"],
                        group_data[baseline]["averaged_rows"],
                        metric,
                        args.bootstrap_samples,
                        args.bootstrap_seed,
                    ),
                }
            )

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    summary_path = output / "ucm_multiseed_summary.csv"
    with summary_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    comparison_path = output / "ucm_multiseed_paired_bootstrap.csv"
    with comparison_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0]))
        writer.writeheader()
        writer.writerows(comparison_rows)
    provenance = {
        "protocol_id": "ICSPS26-FROZEN-v2",
        "formal_run": True,
        "required_seeds": [42, 43, 44],
        "groups": {
            key: {
                "method": group["method"],
                "variant": group["variant"],
                "members": [
                    {
                        "seed": item["identity"]["seed"],
                        "per_image_csv": str(item["path"]),
                        "per_image_sha256": sha256_file(item["path"]),
                        "aggregate_json": str(item["aggregate_path"]),
                        "aggregate_sha256": sha256_file(item["aggregate_path"]),
                    }
                    for item in group["members"]
                ],
            }
            for key, group in group_data.items()
        },
        "comparisons": [
            {"target": target, "baseline": baseline} for target, baseline in args.compare
        ],
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "bootstrap_unit": "UCM source within class with all four L retained",
        "pretest_binding": common_pretest_binding,
        "summary_csv": summary_path.name,
        "summary_csv_sha256": sha256_file(summary_path),
        "paired_csv": comparison_path.name,
        "paired_csv_sha256": sha256_file(comparison_path),
    }
    with (output / "summary_provenance.json").open("x", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    print(json.dumps(provenance, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
