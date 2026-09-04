"""Aggregate completed P0 evaluation runs, including per-L and macro metrics."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from ablation_config import ABLATION_PRESETS


RUN_PATTERN = re.compile(r"^(?P<ablation>.+)_seed(?P<seed>\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", default="test_results_repro/nwpu_global_L_p0_ablation")
    parser.add_argument(
        "--output", default="test_results_repro/nwpu_global_L_p0_ablation_summary.csv"
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.evaluation_root).resolve()
    grouped: dict[str, list[dict]] = defaultdict(list)
    expected_seeds = set(args.seeds)
    for path in sorted(root.glob("*/summary.json")):
        match = RUN_PATTERN.match(path.parent.name)
        if match is None:
            continue
        ablation = match.group("ablation")
        seed = int(match.group("seed"))
        if ablation not in ABLATION_PRESETS or seed not in expected_seeds:
            continue
        with path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        grouped[ablation].append({"seed": seed, **summary})

    missing = []
    for ablation in ABLATION_PRESETS:
        present = {row["seed"] for row in grouped[ablation]}
        for seed in sorted(expected_seeds - present):
            missing.append(f"{ablation}_seed{seed}")
    if missing:
        raise RuntimeError("Incomplete matrix; missing: " + ", ".join(missing))

    metrics = ("prediction_psnr_mean", "prediction_ssim_mean")
    rows = []
    for ablation in ABLATION_PRESETS:
        records = sorted(grouped[ablation], key=lambda row: row["seed"])
        row = {
            "ablation": ablation,
            "seeds": ",".join(str(record["seed"]) for record in records),
            "num_seeds": len(records),
        }
        for metric in metrics:
            values = np.asarray([record[metric] for record in records], dtype=np.float64)
            row[metric] = float(np.mean(values))
            row[metric.replace("_mean", "_std")] = float(np.std(values, ddof=1))
        if any("by_L" in record for record in records):
            expected_labels = {"L1", "L2", "L4", "L8"}
            if not all(set(record.get("by_L", {})) == expected_labels for record in records):
                raise RuntimeError(f"Incomplete per-L summaries for {ablation}")
            for label in ("L1", "L2", "L4", "L8"):
                for metric in metrics:
                    values = np.asarray(
                        [record["by_L"][label][metric] for record in records],
                        dtype=np.float64,
                    )
                    row[f"{label}_{metric}"] = float(np.mean(values))
                    row[f"{label}_{metric.replace('_mean', '_std')}"] = float(
                        np.std(values, ddof=1)
                    )
            for metric in metrics:
                values = np.asarray(
                    [record["macro_average"][metric] for record in records],
                    dtype=np.float64,
                )
                row[f"macro_{metric}"] = float(np.mean(values))
                row[f"macro_{metric.replace('_mean', '_std')}"] = float(
                    np.std(values, ddof=1)
                )
        rows.append(row)

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
