"""Tune one global Lee-MMSE window on NWPU validation and evaluate UCM-21."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import uniform_filter

from experiment_protocol import atomic_json_dump, prepare_run_directory, sha256_file
from icsps2026_protocol import (
    ICSPSFixedPairDataset,
    read_csv_records,
    validate_pair_manifest,
)
from sar_metrics import psnr, ssim
from verify_icsps2026_pretest_gate import verify as verify_pretest_gate


PROTOCOL_ID = "ICSPS26-FROZEN-v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/icsps2026_frozen_v2.json")
    parser.add_argument("--fixed-pair-root", required=True)
    parser.add_argument("--nwpu-val-manifest", required=True)
    parser.add_argument("--ucm-test-manifest", required=True)
    parser.add_argument("--windows", type=int, nargs="+", default=(5, 7, 9, 11))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-root", required=True, help="Pretest-registration root")
    parser.add_argument("--nonformal-smoke", action="store_true")
    parser.add_argument("--max-val-pairs", type=int, default=0)
    parser.add_argument("--max-test-pairs", type=int, default=0)
    return parser.parse_args()


def lee_mmse(noisy: np.ndarray, looks: int, window: int, eps: float = 1e-12) -> np.ndarray:
    """Lee local-statistics MMSE estimate for unit-mean Gamma intensity speckle."""
    image = np.asarray(noisy, dtype=np.float64)
    if image.ndim != 2 or looks <= 0 or window <= 1 or window % 2 == 0:
        raise ValueError("Expected a 2-D image, positive looks, and odd window > 1")
    local_mean = uniform_filter(image, size=window, mode="reflect")
    local_second = uniform_filter(image * image, size=window, mode="reflect")
    observed_variance = np.maximum(local_second - local_mean * local_mean, 0.0)
    speckle_variance = 1.0 / float(looks)
    signal_variance = np.maximum(
        (observed_variance - local_mean * local_mean * speckle_variance)
        / (1.0 + speckle_variance),
        0.0,
    )
    weight = signal_variance / np.maximum(observed_variance, eps)
    estimate = local_mean + np.clip(weight, 0.0, 1.0) * (image - local_mean)
    return np.clip(estimate, 0.0, 1.0).astype(np.float32)


def scalar(value: Any) -> Any:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().reshape(-1)[0].item()
    except ImportError:
        pass
    if isinstance(value, (list, tuple)):
        return value[0]
    return value


def array(value: Any) -> np.ndarray:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(value)


def class_L_macro(rows: list[dict[str, Any]], metric: str) -> float:
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["class_name"]), int(row["L"]))].append(float(row[metric]))
    return float(np.mean([np.mean(values) for values in grouped.values()]))


def main() -> int:
    args = parse_args()
    pretest_gate = verify_pretest_gate(args.run_root)
    if args.max_val_pairs < 0 or args.max_test_pairs < 0:
        raise ValueError("max-val-pairs and max-test-pairs must be non-negative")
    if any(window <= 1 or window % 2 == 0 for window in args.windows):
        raise ValueError("All Lee windows must be odd and greater than 1")
    if (args.max_val_pairs or args.max_test_pairs) and not args.nonformal_smoke:
        raise ValueError("Truncation requires --nonformal-smoke")
    with open(args.config, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    config = payload.get("effective_protocol", payload)
    if config.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(f"Expected {PROTOCOL_ID}, found {config.get('protocol_id')!r}")
    artifact_protocol_id = str(config.get("artifact_protocol_id", PROTOCOL_ID))
    val_rows = read_csv_records(args.nwpu_val_manifest)
    test_rows = read_csv_records(args.ucm_test_manifest)
    if sha256_file(args.ucm_test_manifest) != pretest_gate["ucm_test_manifest_sha256"]:
        raise ValueError("UCM manifest does not match the pretest-seal binding")
    validate_pair_manifest(
        val_rows,
        expected_dataset="NWPU-RESISC45",
        expected_classes=int(config["nwpu"]["expected_classes"]),
        expected_per_class_per_look=int(config["nwpu"]["validation_per_class"]),
        looks=config["looks"], output_root=args.fixed_pair_root,
        synthesis_seed=int(config["seeds"]["validation"]),
    )
    validate_pair_manifest(
        test_rows,
        expected_dataset="UCMerced_LandUse",
        expected_classes=int(config["ucm"]["expected_classes"]),
        expected_per_class_per_look=int(config["ucm"]["expected_per_class"]),
        looks=config["looks"], output_root=args.fixed_pair_root,
        synthesis_seed=int(config["seeds"]["ucm_test"]),
    )
    val_dataset = ICSPSFixedPairDataset(
        args.fixed_pair_root, val_rows,
        protocol_id=artifact_protocol_id, verify_hashes=True,
    )
    test_dataset = ICSPSFixedPairDataset(
        args.fixed_pair_root, test_rows,
        protocol_id=artifact_protocol_id, verify_hashes=True,
    )
    if not args.nonformal_smoke and (len(val_dataset), len(test_dataset)) != (900, 8400):
        raise ValueError(
            f"Formal Lee run requires 900/8400 val/test pairs, found {len(val_dataset)}/{len(test_dataset)}"
        )
    tuning_rows = []
    for window in sorted(set(args.windows)):
        candidate_rows = []
        for index in range(len(val_dataset)):
            if args.max_val_pairs and index >= args.max_val_pairs:
                break
            sample = val_dataset[index]
            noisy = np.squeeze(array(sample["noisy"]))
            clean = np.squeeze(array(sample["clean"]))
            looks = int(scalar(sample["L"]))
            prediction = lee_mmse(noisy, looks, window)
            candidate_rows.append(
                {
                    "class_name": str(scalar(sample["class_name"])),
                    "L": looks,
                    "mse": float(np.mean((prediction.astype(np.float64) - clean) ** 2)),
                    "psnr": psnr(prediction, clean),
                    "ssim": ssim(prediction, clean),
                }
            )
        tuning_rows.append(
            {
                "window": window,
                "pairs": len(candidate_rows),
                "val_macro_mse": class_L_macro(candidate_rows, "mse"),
                "val_macro_psnr": class_L_macro(candidate_rows, "psnr"),
                "val_macro_ssim": class_L_macro(candidate_rows, "ssim"),
            }
        )
    selected = min(tuning_rows, key=lambda row: (row["val_macro_mse"], row["window"]))
    run_config = {
        "protocol_id": PROTOCOL_ID,
        "artifact_protocol_id": artifact_protocol_id,
        "formal_run": not args.nonformal_smoke,
        "dataset": "ucm_test",
        "method": "lee_mmse",
        "method_label": "Lee-MMSE",
        "variant": f"window_{selected['window']}",
        "seed": None,
        "adaptation": None,
        "look_mode": "oracle nominal L",
        "candidate_windows": sorted(set(args.windows)),
        "selection_dataset": "NWPU fixed validation",
        "selection_metric": "class-by-L macro post-clip MSE",
        "test_dataset": "UCM-21 fixed external test",
        "implementation_note": "local-statistics Lee MMSE; not Refined Lee edge-directed filtering",
        "pretest_gate": pretest_gate,
    }
    output = prepare_run_directory(
        args.output_dir,
        run_config,
        source_files=(
            __file__, "icsps2026_protocol.py", "sar_metrics.py",
            args.config, args.nwpu_val_manifest, args.ucm_test_manifest,
        ),
    )
    with open(output / "nwpu_window_tuning.csv", "x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(tuning_rows[0]))
        writer.writeheader()
        writer.writerows(tuning_rows)

    rows = []
    for index in range(len(test_dataset)):
        if args.max_test_pairs and index >= args.max_test_pairs:
            break
        sample = test_dataset[index]
        noisy = np.squeeze(array(sample["noisy"]))
        clean = np.squeeze(array(sample["clean"]))
        looks = int(scalar(sample["L"]))
        prediction = lee_mmse(noisy, looks, int(selected["window"]))
        rows.append(
            {
                "protocol_id": PROTOCOL_ID,
                "artifact_protocol_id": artifact_protocol_id,
                "formal_run": not args.nonformal_smoke,
                "dataset": "ucm_test",
                "method": "lee_mmse",
                "variant": f"window_{selected['window']}",
                "seed": "",
                "pair_id": str(scalar(sample["pair_id"])),
                "source_id": str(scalar(sample["source_id"])),
                "class_name": str(scalar(sample["class_name"])),
                "L": looks,
                "psnr": psnr(prediction, clean),
                "ssim": ssim(prediction, clean),
            }
        )
    per_image_path = output / "per_image.csv"
    with open(per_image_path, "x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    aggregate = {
        "protocol_id": PROTOCOL_ID,
        "artifact_protocol_id": artifact_protocol_id,
        "formal_run": not args.nonformal_smoke,
        "dataset": "ucm_test",
        "method": "lee_mmse",
        "method_label": "Lee-MMSE",
        "variant": f"window_{selected['window']}",
        "seed": None,
        "adaptation": None,
        "look_mode": "oracle nominal L",
        "selected_window": int(selected["window"]),
        "selection": selected,
        "pairs": len(rows),
        "sources": len({row["source_id"] for row in rows}),
        "by_L_class_macro": {
            str(looks): {
                metric: class_L_macro([row for row in rows if row["L"] == looks], metric)
                for metric in ("psnr", "ssim")
            }
            for looks in sorted({int(row["L"]) for row in rows})
        },
        "macro": {
            metric: class_L_macro(rows, metric)
            for metric in ("psnr", "ssim")
        },
        "nwpu_val_manifest_sha256": sha256_file(args.nwpu_val_manifest),
        "ucm_test_manifest_sha256": sha256_file(args.ucm_test_manifest),
        "per_image_sha256": sha256_file(per_image_path),
        "pretest_gate": pretest_gate,
    }
    atomic_json_dump(aggregate, output / "aggregate.json")
    print(json.dumps(aggregate, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
