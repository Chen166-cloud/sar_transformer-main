"""Validate and score precomputed outputs from the official SAR-BM3D package."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from experiment_protocol import atomic_json_dump, prepare_run_directory, sha256_file
from finalize_icsps2026_sarbm3d_raw import (
    SARBM3D_DOMAIN_CONVERSION,
    SARBM3D_INPUT_DOMAIN,
    SARBM3D_OUTPUT_DOMAIN,
)
from icsps2026_protocol import (
    ICSPSFixedPairDataset,
    read_csv_records,
    validate_pair_manifest,
)
from sar_metrics import psnr, ssim
from verify_icsps2026_artifact import verify as verify_artifact
from verify_icsps2026_pretest_gate import verify as verify_pretest_gate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/icsps2026_frozen_v2.json")
    parser.add_argument("--fixed-pair-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--prediction-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-root", required=True, help="Pretest-registration root")
    parser.add_argument("--package-archive-sha256", required=True)
    parser.add_argument("--package-function-sha256", required=True)
    parser.add_argument("--nonformal-smoke", action="store_true")
    parser.add_argument("--max-pairs", type=int, default=0)
    return parser.parse_args()


def scalar(value):
    try:
        import torch
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().reshape(-1)[0].item()
    except ImportError:
        pass
    if isinstance(value, (list, tuple)):
        return value[0]
    return value


def as_array(value) -> np.ndarray:
    try:
        import torch
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(value)


def mat_text(payload: dict, field: str, path: Path) -> str:
    if field not in payload:
        raise KeyError(f"Missing MAT variable {field!r}: {path}")
    values = np.asarray(payload[field]).squeeze()
    if values.dtype.kind not in {"U", "S"}:
        raise ValueError(f"MAT variable {field!r} is not text: {path}")
    result = "".join(
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values.reshape(-1).tolist()
    )
    if not result:
        raise ValueError(f"MAT variable {field!r} is empty: {path}")
    return result


def macro(rows, metric):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["class_name"], row["L"])].append(float(row[metric]))
    return float(np.mean([np.mean(values) for values in grouped.values()]))


def main() -> int:
    args = parse_args()
    pretest_gate = verify_pretest_gate(args.run_root)
    if args.max_pairs < 0:
        raise ValueError("--max-pairs must be non-negative")
    for label, digest in (
        ("package archive", args.package_archive_sha256),
        ("SARBM3D_v10.m", args.package_function_sha256),
    ):
        if len(digest) != 64 or any(character not in "0123456789abcdefABCDEF" for character in digest):
            raise ValueError(f"{label} SHA-256 must contain 64 hexadecimal characters")
    if args.max_pairs and not args.nonformal_smoke:
        raise ValueError("--max-pairs requires --nonformal-smoke")
    with open(args.config, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    config = payload.get("effective_protocol", payload)
    if config.get("protocol_id") != "ICSPS26-FROZEN-v2":
        raise ValueError("Wrong protocol config")
    artifact_protocol_id = config.get("artifact_protocol_id", config["protocol_id"])
    manifest_rows = read_csv_records(args.manifest)
    if sha256_file(args.manifest) != pretest_gate["ucm_test_manifest_sha256"]:
        raise ValueError("UCM manifest does not match the pretest-seal binding")
    validate_pair_manifest(
        manifest_rows,
        expected_dataset="UCMerced_LandUse",
        expected_classes=int(config["ucm"]["expected_classes"]),
        expected_per_class_per_look=int(config["ucm"]["expected_per_class"]),
        looks=config["looks"], output_root=args.fixed_pair_root,
        synthesis_seed=int(config["seeds"]["ucm_test"]),
    )
    dataset = ICSPSFixedPairDataset(
        args.fixed_pair_root, manifest_rows,
        protocol_id=artifact_protocol_id, verify_hashes=True,
    )
    if not args.nonformal_smoke and len(dataset) != 8400:
        raise ValueError(f"Formal UCM SAR-BM3D evaluation requires 8400 pairs, found {len(dataset)}")
    prediction_root = Path(args.prediction_root).resolve()
    raw_verification = verify_artifact(
        "sarbm_raw", prediction_root / "completion.json",
        expected_method="sar_bm3d", expected_variant="v1.0",
        expected_seed="", expected_adaptation="",
    )
    if raw_verification["pretest_seal_sha256"] != pretest_gate["pretest_seal_sha256"]:
        raise ValueError("SAR-BM3D raw predictions use a different pretest seal")
    rows = []
    for index in range(len(dataset)):
        if args.max_pairs and index >= args.max_pairs:
            break
        sample = dataset[index]
        pair_id = str(scalar(sample["pair_id"]))
        prediction_path = prediction_root / "predictions" / f"{pair_id}.mat"
        if not prediction_path.is_file():
            raise FileNotFoundError(f"Missing SAR-BM3D prediction: {prediction_path}")
        payload = loadmat(prediction_path)
        if "prediction" not in payload:
            raise KeyError(f"Missing prediction field in {prediction_path}")
        raw_prediction = np.asarray(payload["prediction"], dtype=np.float32).squeeze()
        input_domain = mat_text(payload, "input_domain", prediction_path)
        output_domain = mat_text(payload, "output_domain", prediction_path)
        conversion = mat_text(payload, "domain_conversion", prediction_path)
        if (
            input_domain != SARBM3D_INPUT_DOMAIN
            or output_domain != SARBM3D_OUTPUT_DOMAIN
            or conversion != SARBM3D_DOMAIN_CONVERSION
        ):
            raise ValueError(f"SAR-BM3D domain declaration mismatch for {pair_id}")
        noisy = np.squeeze(as_array(sample["noisy"]))
        clean = np.squeeze(as_array(sample["clean"]))
        if raw_prediction.shape != clean.shape or not np.isfinite(raw_prediction).all():
            raise ValueError(f"Invalid prediction shape/values for {pair_id}")
        prediction = np.clip(raw_prediction, 0.0, 1.0)
        rows.append(
            {
                "protocol_id": "ICSPS26-FROZEN-v2",
                "artifact_protocol_id": artifact_protocol_id,
                "formal_run": not args.nonformal_smoke,
                "dataset": "ucm_test",
                "method": "sar_bm3d",
                "variant": "v1.0",
                "seed": "",
                "pair_id": pair_id,
                "source_id": str(scalar(sample["source_id"])),
                "class_name": str(scalar(sample["class_name"])),
                "L": int(scalar(sample["L"])),
                "psnr": psnr(prediction, clean),
                "ssim": ssim(prediction, clean),
                "raw_min": float(np.min(raw_prediction)),
                "raw_max": float(np.max(raw_prediction)),
                "postclip_fraction": float(np.mean((raw_prediction < 0.0) | (raw_prediction > 1.0))),
                "prediction_sha256": sha256_file(prediction_path),
                "input_domain": input_domain,
                "output_domain": output_domain,
                "domain_conversion": conversion,
            }
        )
    if not rows:
        raise RuntimeError("No SAR-BM3D predictions evaluated")
    run_config = {
        "protocol_id": "ICSPS26-FROZEN-v2",
        "formal_run": not args.nonformal_smoke,
        "dataset": "ucm_test",
        "method": "sar_bm3d",
        "method_label": "SAR-BM3D",
        "variant": "v1.0",
        "seed": None,
        "adaptation": None,
        "version": "1.0",
        "look_mode": "oracle nominal L",
        "source_pair_domain": "linear_normalized_intensity_0_1",
        "sarbm3d_input_domain": SARBM3D_INPUT_DOMAIN,
        "scored_output_domain": SARBM3D_OUTPUT_DOMAIN,
        "domain_conversion": SARBM3D_DOMAIN_CONVERSION,
        "package_archive_sha256": args.package_archive_sha256,
        "package_function_sha256": args.package_function_sha256,
        "opencv_runtime": "package-bundled lib_opencv210/glnxa64 prepended to LD_LIBRARY_PATH",
        "manifest_sha256": sha256_file(args.manifest),
        "pretest_gate": pretest_gate,
        "raw_completion_sha256": raw_verification["json_sha256"],
    }
    output = prepare_run_directory(
        args.output_dir,
        run_config,
        source_files=(__file__, "sar_metrics.py", args.config, args.manifest),
    )
    per_image_path = output / "per_image.csv"
    with open(per_image_path, "x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    looks_values = sorted({int(row["L"]) for row in rows})
    aggregate = {
        **run_config,
        "pairs": len(rows),
        "sources": len({row["source_id"] for row in rows}),
        "by_L_class_macro": {
            str(looks): {
                metric: macro([row for row in rows if row["L"] == looks], metric)
                for metric in ("psnr", "ssim")
            }
            for looks in looks_values
        },
        "macro": {
            metric: macro(rows, metric)
            for metric in ("psnr", "ssim")
        },
        "per_image_sha256": sha256_file(per_image_path),
    }
    atomic_json_dump(aggregate, output / "aggregate.json")
    print(json.dumps(aggregate, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
