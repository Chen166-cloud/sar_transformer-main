"""Evaluate fixed-ROI ENL and paper-defined M/EPI on real SAR outputs.

This is a supplementary classical-metric evaluator.  It consumes the frozen
real-SAR QC/ROI manifest and either evaluates the Noisy identity baseline or
``*_prediction.npy`` arrays exported by ``evaluate_icsps2026_real.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from classic_sar_metrics import edge_preservation_index, m_index_paper
from icsps2026_metrics import enl_roi_details, parent_cluster_bootstrap
from prepare_icsps2026_real import apply_fixed_mapping, load_real_array, sha256_file


def _bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _artifact_path(manifest_path: Path, manifest: dict, key: str) -> Path:
    path = (manifest_path.parent / manifest["artifacts"][key]).resolve()
    expected = manifest["artifacts"].get(key + "_sha256")
    if expected and sha256_file(path) != expected:
        raise AssertionError(f"QC artifact SHA-256 mismatch: {path}")
    return path


def _safe_stem(sample_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id)


def _prediction(
    noisy: np.ndarray,
    sample_id: str,
    prediction_dir: Path | None,
) -> np.ndarray:
    if prediction_dir is None:
        return noisy.copy()
    path = prediction_dir / f"{_safe_stem(sample_id)}_prediction.npy"
    if not path.is_file():
        raise FileNotFoundError(path)
    value = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32).squeeze()
    if value.shape != noisy.shape:
        raise ValueError(f"Prediction shape mismatch for {sample_id}: {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"Prediction contains NaN/Inf: {path}")
    return np.clip(value, 0.0, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--qc-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method-label", default="Noisy")
    parser.add_argument(
        "--prediction-dir",
        help="Directory containing <sample_id>_prediction.npy; omit for Noisy",
    )
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--allow-nonformal-count", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("bootstrap-samples must be positive")
    dataset_root = Path(args.dataset_root).resolve()
    manifest_path = Path(args.qc_manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {output_dir}")
    prediction_dir = Path(args.prediction_dir).resolve() if args.prediction_dir else None
    if prediction_dir is not None and not prediction_dir.is_dir():
        raise FileNotFoundError(prediction_dir)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    qc_csv = _artifact_path(manifest_path, manifest, "qc_csv")
    roi_csv = _artifact_path(manifest_path, manifest, "roi_csv")
    qc_rows = [
        row
        for row in _read_csv(qc_csv)
        if row["split"] == args.split and _bool(row["valid_no_reference"])
    ]
    parent_count = len({row["parent_id"] for row in qc_rows})
    if (
        args.split == "test"
        and not args.allow_nonformal_count
        and (len(qc_rows), parent_count) != (592, 148)
    ):
        raise ValueError(
            "Formal real test requires 592 patches from 148 parents; "
            f"found {len(qc_rows)} patches from {parent_count} parents"
        )

    sample_ids = {row["sample_id"] for row in qc_rows}
    rois_by_sample: dict[str, list[dict[str, int | str]]] = defaultdict(list)
    for row in _read_csv(roi_csv):
        if row["sample_id"] in sample_ids:
            rois_by_sample[row["sample_id"]].append(
                {
                    "roi_id": row["roi_id"],
                    "x": int(row["x"]),
                    "y": int(row["y"]),
                    "width": int(row["width"]),
                    "height": int(row["height"]),
                }
            )
    expected_rois = int(manifest["enl_roi_policy"]["num_rois"])
    bad_counts = {
        sample_id: len(rois_by_sample.get(sample_id, []))
        for sample_id in sample_ids
        if len(rois_by_sample.get(sample_id, [])) != expected_rois
    }
    if bad_counts:
        raise ValueError(f"Frozen ROI count mismatch: {list(sorted(bad_counts.items()))[:5]}")

    mapping = manifest["numeric_mapping"]
    per_patch: list[dict[str, object]] = []
    per_roi: list[dict[str, object]] = []
    for index, qc in enumerate(qc_rows, start=1):
        noisy_raw, _ = load_real_array(dataset_root / qc["noisy_path"], "noisy")
        noisy = apply_fixed_mapping(noisy_raw, mapping)
        prediction = _prediction(noisy, qc["sample_id"], prediction_dir)
        roi_details = enl_roi_details(prediction, rois_by_sample[qc["sample_id"]])
        roi_enl_values = [float(detail["enl"]) for detail in roi_details]
        for detail in roi_details:
            per_roi.append(
                {
                    "method": args.method_label,
                    "sample_id": qc["sample_id"],
                    "parent_id": qc["parent_id"],
                    "split": qc["split"],
                    "roi_id": detail["roi_id"],
                    "x": detail["x"],
                    "y": detail["y"],
                    "width": detail["width"],
                    "height": detail["height"],
                    "mean": detail["mean"],
                    "variance": detail["variance"],
                    "enl": detail["enl"],
                }
            )

        m_result = m_index_paper(
            noisy,
            prediction,
            window=25,
            tolerance=0.03,
            levels=8,
            shuffles=100,
            seed=args.seed,
        )
        per_patch.append(
            {
                "method": args.method_label,
                "sample_id": qc["sample_id"],
                "parent_id": qc["parent_id"],
                "split": qc["split"],
                "fixed_roi_count": len(roi_enl_values),
                "enl_fixed_roi_mean": float(np.mean(roi_enl_values)),
                "epi": edge_preservation_index(noisy, prediction),
                "m_index": m_result.value,
                "m_valid": m_result.valid,
                "m_invalid_reason": m_result.reason,
                "m_selected_windows": m_result.selected_windows,
                "m_total_windows": m_result.total_windows,
                "m_first_order_residual": m_result.first_order_residual,
                "m_delta_h": m_result.delta_h,
                "m_original_homogeneity": m_result.original_homogeneity,
                "m_shuffled_homogeneity_mean": m_result.shuffled_homogeneity_mean,
                "m_ratio_denominator_floor_fraction": (
                    m_result.ratio_denominator_floor_fraction
                ),
            }
        )
        if index % 100 == 0 or index == len(qc_rows):
            print(f"evaluated {index}/{len(qc_rows)}", flush=True)

    output_dir.mkdir(parents=True)
    patch_path = output_dir / "per_patch.csv"
    with patch_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_patch[0]))
        writer.writeheader()
        writer.writerows(per_patch)
    roi_path = output_dir / "roi_regions.csv"
    with roi_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_roi[0]))
        writer.writeheader()
        writer.writerows(per_roi)

    roi_values = np.asarray([float(row["enl"]) for row in per_roi], dtype=np.float64)
    epi_values = np.asarray([float(row["epi"]) for row in per_patch], dtype=np.float64)
    valid_m_rows = [row for row in per_patch if _bool(row["m_valid"])]
    paper_table_row = {
        "Method": args.method_label,
        "ENL": (
            f"{np.median(roi_values):.3f} "
            f"[{np.percentile(roi_values, 25):.3f}, "
            f"{np.percentile(roi_values, 75):.3f}]"
        ),
        "M_down": (
            f"{np.mean([float(row['m_index']) for row in valid_m_rows]):.4f}"
            if valid_m_rows
            else "N/A"
        ),
        "EPI_up": f"{np.mean(epi_values):.4f}",
    }
    table_path = output_dir / "paper_table.csv"
    with table_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paper_table_row))
        writer.writeheader()
        writer.writerow(paper_table_row)

    summary = {
        "schema_version": 1,
        "method": args.method_label,
        "split": args.split,
        "num_patches": len(per_patch),
        "num_parent_clusters": parent_count,
        "num_fixed_rois": len(per_roi),
        "enl_fixed_roi": {
            "mean": float(np.mean(roi_values)),
            "median": float(np.median(roi_values)),
            "q1": float(np.percentile(roi_values, 25)),
            "q3": float(np.percentile(roi_values, 75)),
            "display": (
                f"{np.median(roi_values):.3f} "
                f"[{np.percentile(roi_values, 25):.3f}, "
                f"{np.percentile(roi_values, 75):.3f}]"
            ),
        },
        "epi": {
            "definition": "Remote Sensing 2024 Eq.16; diagonal neighbours per author EPD.m",
            "mean": float(np.mean(epi_values)),
            "median": float(np.median(epi_values)),
            "min": float(np.min(epi_values)),
            "max": float(np.max(epi_values)),
            "identity_baseline_is_tautological": prediction_dir is None,
        },
        "m_index": {
            "definition": "Remote Sensing 2017 Eq.1-3; strict no-fallback implementation",
            "parameters": {
                "window": 25,
                "tolerance": 0.03,
                "levels": 8,
                "shuffles": 100,
                "seed": args.seed,
            },
            "valid_patches": len(valid_m_rows),
            "invalid_patches": len(per_patch) - len(valid_m_rows),
            "invalid_reasons": dict(Counter(str(row["m_invalid_reason"]) for row in per_patch if not _bool(row["m_valid"]))),
            "mean": (
                float(np.mean([float(row["m_index"]) for row in valid_m_rows]))
                if valid_m_rows
                else None
            ),
            "median": (
                float(np.median([float(row["m_index"]) for row in valid_m_rows]))
                if valid_m_rows
                else None
            ),
        },
        "paper_table_row": paper_table_row,
        "qc_manifest_sha256": sha256_file(manifest_path),
        "qc_csv_sha256": sha256_file(qc_csv),
        "roi_manifest_sha256": sha256_file(roi_csv),
        "artifacts": {
            "per_patch_csv": patch_path.name,
            "per_patch_sha256": sha256_file(patch_path),
            "roi_regions_csv": roi_path.name,
            "roi_regions_sha256": sha256_file(roi_path),
            "paper_table_csv": table_path.name,
            "paper_table_sha256": sha256_file(table_path),
        },
    }
    summary_path = output_dir / "summary.json"
    with summary_path.open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
