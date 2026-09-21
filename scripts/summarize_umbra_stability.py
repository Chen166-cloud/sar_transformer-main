"""Summarize cross-scene residual diagnostics for the Umbra stability runs.

These are no-reference diagnostics, not PSNR/SSIM substitutes.  They describe
the noisy/output ratio and log-domain edge agreement so the report can discuss
cross-scene behavior without inventing a speckle-free ground truth.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "output" / "umbra_stability_3scenes"
SCENES = {
    "01_busan": "Busan Port",
    "02_bangkok": "Bangkok Suvarnabhumi Airport",
    "03_newark": "Newark Port",
}
RUNTIME_SOURCES = {
    "SAR-BM3D": ("sarbm3d/summary.json", ("inference_seconds",)),
    "SAR2SAR": ("sar2sar/summary.json", ("inference_seconds",)),
    "SDUDNet": ("sdudnet_db_display/run.json", ("forward_seconds_including_first_call",)),
    "Trans-SAR": ("transsar/run.json", ("tiling", "forward_seconds")),
    "CL-SAR": ("cl_sar/run.json", ("forward_seconds_including_first_call",)),
    "MERLIN": ("merlin_stride64_weighted/run.json", ("inference", "primary_elapsed_seconds")),
    "MuLoG-DRUNet": ("mulog_drunet/run.json", ("forward_seconds_including_first_call",)),
}


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    x -= x.mean()
    y -= y.mean()
    denominator = np.sqrt(np.dot(x, x) * np.dot(y, y))
    return float(np.dot(x, y) / denominator) if denominator > 0 else 0.0


def winsorize(array: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    bounds = np.percentile(array[np.isfinite(array)], [low, high])
    return np.clip(array, bounds[0], bounds[1])


def ratio_diagnostics(noisy: np.ndarray, ratio_db: np.ndarray) -> dict[str, float]:
    values = ratio_db[np.isfinite(ratio_db)].astype(np.float64, copy=False)
    q01, q25, q50, q75, q99 = np.percentile(values, [1, 25, 50, 75, 99])
    mad = np.median(np.abs(values - q50))

    residual = winsorize(ratio_db.astype(np.float64, copy=False))
    noisy_db = 10.0 * np.log10(np.maximum(noisy.astype(np.float64, copy=False), 1e-12))
    noisy_db = winsorize(noisy_db)
    output_db = noisy_db - residual

    neighbor_h = pearson(residual[:, :-1], residual[:, 1:])
    neighbor_v = pearson(residual[:-1, :], residual[1:, :])
    noisy_gradient = np.hypot(*np.gradient(noisy_db))
    output_gradient = np.hypot(*np.gradient(output_db))
    return {
        "ratio_median_db": float(q50),
        "ratio_iqr_db": float(q75 - q25),
        "ratio_robust_sigma_db": float(1.4826 * mad),
        "ratio_p01_db": float(q01),
        "ratio_p99_db": float(q99),
        "ratio_neighbor_corr": float((neighbor_h + neighbor_v) / 2.0),
        "ratio_vs_noisy_log_corr": pearson(residual, noisy_db),
        "log_gradient_corr_with_noisy": pearson(output_gradient, noisy_gradient),
    }


def nested_value(payload: dict[str, object], keys: tuple[str, ...]) -> float:
    value: object = payload
    for key in keys:
        if not isinstance(value, dict):
            raise KeyError(keys)
        value = value[key]
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    experiment_root = args.root.resolve()

    rows: list[dict[str, object]] = []
    runtimes: list[dict[str, object]] = []
    for slug, expected_name in SCENES.items():
        scene_root = experiment_root / slug
        roi_report_path = scene_root / "roi" / "run.json"
        figure_root = scene_root / "experiment" / "figure3_style"
        manifest_path = figure_root / "manifest.json"
        if not roi_report_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(f"incomplete scene: {scene_root}")
        roi_report = json.loads(roi_report_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not manifest.get("complete"):
            raise RuntimeError(f"figure manifest is incomplete: {manifest_path}")
        noisy = np.load(scene_root / "roi" / "noisy_intensity.npy", allow_pickle=False)
        for method, files in manifest["ratios"]["files"].items():
            ratio_path = figure_root / files["npy"]
            ratio = np.load(ratio_path, allow_pickle=False)
            metrics = ratio_diagnostics(noisy, ratio)
            rows.append({
                "scene_slug": slug,
                "scene": manifest.get("scene", expected_name),
                "method": method,
                "roi_row": roi_report["roi"]["row_start"],
                "roi_col": roi_report["roi"]["col_start"],
                **metrics,
            })
        runs_root = scene_root / "experiment" / "runs"
        for method, (relative_path, keys) in RUNTIME_SOURCES.items():
            payload = json.loads((runs_root / relative_path).read_text(encoding="utf-8"))
            runtimes.append({
                "scene_slug": slug,
                "scene": manifest.get("scene", expected_name),
                "method": method,
                "inference_seconds": nested_value(payload, keys),
                "timing_scope": "model/algorithm inference recorded by the adapter; excludes common rendering and most process startup",
            })

    methods = sorted({str(row["method"]) for row in rows})
    aggregate: list[dict[str, object]] = []
    metric_names = [
        "ratio_median_db",
        "ratio_iqr_db",
        "ratio_robust_sigma_db",
        "ratio_neighbor_corr",
        "ratio_vs_noisy_log_corr",
        "log_gradient_corr_with_noisy",
    ]
    for method in methods:
        subset = [row for row in rows if row["method"] == method]
        record: dict[str, object] = {"method": method, "scene_count": len(subset)}
        for name in metric_names:
            values = np.asarray([float(row[name]) for row in subset], dtype=np.float64)
            record[f"{name}_mean"] = float(values.mean())
            record[f"{name}_range"] = float(values.max() - values.min())
        aggregate.append(record)

    csv_path = experiment_root / "stability_diagnostics.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    output = {
        "status": "complete",
        "scope": "no-reference diagnostics; not PSNR/SSIM and not a surrogate ground truth score",
        "definitions": {
            "ratio": "10*log10(noisy/output)",
            "ratio_median_db": "robust radiometric shift indicator; zero is not by itself proof of correctness",
            "ratio_iqr_db": "middle-50% residual spread",
            "ratio_neighbor_corr": "mean horizontal/vertical Pearson correlation of winsorized ratio dB; lower absolute values mean a whiter residual, not necessarily better restoration",
            "ratio_vs_noisy_log_corr": "residual leakage diagnostic",
            "log_gradient_corr_with_noisy": "edge/noise agreement diagnostic; higher values may also retain speckle",
        },
        "per_scene": rows,
        "cross_scene": aggregate,
        "runtimes": runtimes,
        "csv": str(csv_path),
    }
    json_path = experiment_root / "stability_diagnostics.json"
    json_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"json": str(json_path), "csv": str(csv_path), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
