"""Evaluate learned checkpoints or the noisy input on a frozen real-SAR QC manifest.

Reported no-reference metrics are ratio-mean bias, ratio ACF sidelobe energy,
fixed-ROI ENL details, and Sobel gradient correlation (Sobel-GC).  Sobel-GC is
explicitly *not* RGPI.  This evaluator does not report the repository's
unvalidated historical M-index implementation.  ICSPS26-FROZEN-v2 is no-GT16;
the optional qPSNR/qSSIM compatibility code is only for nonformal or future
separately-versioned protocols and must remain unused in formal paper artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
import torch

from experiment_protocol import (
    atomic_json_dump,
    load_checkpoint_strict,
    prepare_run_directory,
    seed_everything,
    validate_completed_checkpoint,
)
from icsps2026_metrics import (
    enl_roi_details,
    intensity_ratio,
    parent_cluster_bootstrap,
    ratio_acf_sidelobe_energy_from_ratio,
    sobel_gradient_correlation,
)
from numeric_domain import INTENSITY_DOMAIN
from model_registry import (
    SUPPORTED_METHODS,
    build_model,
    extract_checkpoint_model_metadata,
)
from prepare_icsps2026_real import apply_fixed_mapping, load_real_array, sha256_file
from sar_metrics import psnr, ssim
from verify_icsps2026_pretest_gate import verify as verify_pretest_gate


PROTOCOL_ID = "ICSPS26-FROZEN-v2"


PER_IMAGE_METRICS = (
    "qpsnr_gt16",
    "qssim_gt16",
    "ratio_mean_bias",
    "ratio_acf_sidelobe_energy",
    "sobel_gc_noisy_output",
    "noisy_enl_roi_median",
    "output_enl_roi_median",
)


def _bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _finite_float(value: object) -> float | None:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if np.isfinite(converted) else None


def _safe_stem(sample_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id)


def _write_gray(path: Path, image: np.ndarray) -> None:
    output = np.round(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    if not cv2.imwrite(str(path), output):
        raise IOError(f"Failed to write {path}")


def _device(requested: str) -> torch.device:
    requested_device = torch.device(requested)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was explicitly requested but is unavailable. Use --device cpu only for an "
            "intentional CPU smoke run."
        )
    return requested_device


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _artifact_path(manifest_path: Path, manifest: dict, key: str) -> Path:
    relative = manifest["artifacts"][key]
    path = (manifest_path.parent / relative).resolve()
    expected = manifest["artifacts"].get(key + "_sha256")
    if expected and sha256_file(path) != expected:
        raise AssertionError(f"QC artifact SHA-256 mismatch: {path}")
    return path


def _source_path(dataset_root: Path, relative: str) -> Path:
    path = (dataset_root / relative).resolve()
    try:
        path.relative_to(dataset_root)
    except ValueError as error:
        raise ValueError(f"QC path escapes dataset root: {relative}") from error
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _checkpoint_model(
    checkpoint: Path,
    device: torch.device,
    method: str,
    variant: str | None,
    sar_cam_root: str | None,
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    checkpoint_payload = torch.load(
        checkpoint, map_location="cpu", weights_only=False
    )
    if not isinstance(checkpoint_payload, Mapping):
        raise ValueError("Checkpoint must be a metadata-bearing mapping")
    model_identity = extract_checkpoint_model_metadata(checkpoint_payload)
    model = build_model(
        method,
        variant=variant,
        numeric_domain=INTENSITY_DOMAIN,
        external_root=sar_cam_root,
        checkpoint_metadata=checkpoint_payload,
    )
    metadata = load_checkpoint_strict(model, checkpoint, map_location="cpu")
    run_config = metadata.get("run_config") if isinstance(metadata, dict) else None
    checkpoint_domain = run_config.get("numeric_domain") if isinstance(run_config, dict) else None
    if checkpoint_domain is not None and checkpoint_domain != INTENSITY_DOMAIN:
        raise ValueError(
            f"Checkpoint numeric domain is {checkpoint_domain!r}, expected {INTENSITY_DOMAIN!r}"
        )
    model.to(device).eval()
    return model, metadata, model_identity


def _validate_adaptation(
    requested: str,
    model_identity: Mapping[str, Any],
    checkpoint_metadata: Mapping[str, Any],
) -> None:
    """Prevent the display-level adaptation flag from relabeling a checkpoint."""
    run_config = checkpoint_metadata.get("run_config")
    declared = run_config.get("adaptation") if isinstance(run_config, Mapping) else None
    if requested == "ams":
        if model_identity["method"] != "ours" or model_identity["variant"] != "full":
            raise ValueError("AMS evaluation is defined only for the Ours Full model")
        if declared != "ams":
            raise ValueError(
                "--adaptation ams requires checkpoint run_config.adaptation='ams'"
            )
    elif declared == "ams":
        raise ValueError("An AMS checkpoint cannot be evaluated or labeled as --adaptation base")


def _validate_formal_checkpoint(
    adaptation: str,
    checkpoint_metadata: Mapping[str, Any],
    seed: int,
) -> None:
    run_config = checkpoint_metadata.get("run_config")
    if not isinstance(run_config, Mapping):
        raise ValueError("A formal real-SAR evaluation requires checkpoint run_config metadata")
    expected = {
        "protocol_id": PROTOCOL_ID,
        "formal_run": True,
        "numeric_domain": INTENSITY_DOMAIN,
        "seed": seed,
    }
    expected_format = (
        "icsps26-ams-v1" if adaptation == "ams" else "icsps26-resumable-v1"
    )
    mismatches = {
        key: {"expected": value, "found": run_config.get(key)}
        for key, value in expected.items()
        if run_config.get(key) != value
    }
    if checkpoint_metadata.get("checkpoint_format") != expected_format:
        mismatches["checkpoint_format"] = {
            "expected": expected_format,
            "found": checkpoint_metadata.get("checkpoint_format"),
        }
    if adaptation == "base" and run_config.get("target_updates") != 100_000:
        mismatches["target_updates"] = {
            "expected": 100_000,
            "found": run_config.get("target_updates"),
        }
    if adaptation == "ams" and run_config.get("epochs") != 8:
        mismatches["epochs"] = {"expected": 8, "found": run_config.get("epochs")}
    if mismatches:
        raise ValueError(f"Checkpoint is not compatible with a formal real-SAR run: {mismatches}")


def _default_method_label(
    model_identity: Mapping[str, Any], adaptation: str
) -> str:
    method = str(model_identity["method"])
    variant = model_identity.get("variant")
    if method == "ours":
        label = "DFNG-SARNet"
        if variant not in (None, "full"):
            label += f" ({variant})"
        if adaptation == "ams":
            label += "+AMS"
        return label
    if method == "transsar_v2":
        return "TransSARV2"
    if method == "sar_cam":
        return "SAR-CAM"
    raise ValueError(f"Unsupported registry method in checkpoint: {method!r}")


def _quartiles(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return float("nan"), float("nan"), float("nan")
    array = np.asarray(values, dtype=np.float64)
    return (
        float(np.median(array)),
        float(np.percentile(array, 25)),
        float(np.percentile(array, 75)),
    )


def _parent_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["parent_id"])].append(row)
    output = []
    for parent_id in sorted(grouped):
        children = grouped[parent_id]
        item: dict[str, object] = {
            "method": children[0]["method"],
            "variant": children[0]["variant"],
            "method_label": children[0]["method_label"],
            "adaptation": children[0]["adaptation"],
            "seed": children[0]["seed"],
            "formal_run": children[0]["formal_run"],
            "split": children[0]["split"],
            "category": "|".join(
                sorted({str(child.get("category", "")) for child in children if child.get("category")})
            ),
            "parent_id": parent_id,
            "num_child_patches": len(children),
        }
        for metric in PER_IMAGE_METRICS:
            values = [value for row in children if (value := _finite_float(row.get(metric))) is not None]
            item[metric + "_mean"] = float(np.mean(values)) if values else ""
        output.append(item)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--qc-manifest", required=True, help="real_qc_manifest.json")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--run-root",
        help="Required pretest-registration root whenever --split test is used",
    )
    parser.add_argument(
        "--method", required=True, choices=("noisy", *SUPPORTED_METHODS)
    )
    parser.add_argument("--variant", default=None, help="Ours variant; omitted means full")
    parser.add_argument("--sar-cam-root", help="Pinned official SAR-CAM Git checkout")
    parser.add_argument(
        "--method-label",
        default=None,
        help="Display-only label; default is derived from verified checkpoint metadata",
    )
    parser.add_argument("--adaptation", choices=("base", "ams"), required=True)
    parser.add_argument("--split", choices=("train", "val", "test", "unassigned"), default="test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--nonformal-smoke", action="store_true")
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--ratio-display-min", type=float, default=0.5)
    parser.add_argument("--ratio-display-max", type=float, default=1.5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pretest_gate = None
    if args.split == "test":
        if not args.run_root:
            raise ValueError("Every real-test evaluation requires --run-root for the pretest gate")
        pretest_gate = verify_pretest_gate(args.run_root)
    if args.bootstrap_samples <= 0:
        raise ValueError("bootstrap-samples must be positive")
    if args.ratio_display_max <= args.ratio_display_min:
        raise ValueError("ratio-display-max must exceed ratio-display-min")
    if args.max_images < 0:
        raise ValueError("max-images must be non-negative")
    if args.max_images and not args.nonformal_smoke:
        raise ValueError("--max-images requires --nonformal-smoke")
    if not args.nonformal_smoke and args.split != "test":
        raise ValueError("A formal real-SAR evaluation must use --split test")
    if args.method == "noisy":
        if args.checkpoint:
            raise ValueError("The noisy-input baseline does not accept --checkpoint")
        if args.variant is not None or args.sar_cam_root is not None:
            raise ValueError("The noisy-input baseline accepts neither variant nor SAR-CAM options")
        if args.adaptation != "base":
            raise ValueError("The noisy-input baseline must use --adaptation base")
    elif not args.checkpoint:
        raise ValueError(f"{args.method} requires --checkpoint")
    seed_everything(args.seed)
    device = torch.device("cpu") if args.method == "noisy" else _device(args.device)
    dataset_root = Path(args.dataset_root).resolve()
    manifest_path = Path(args.qc_manifest).resolve()
    checkpoint_path = Path(args.checkpoint).resolve() if args.checkpoint else None
    with manifest_path.open("r", encoding="utf-8") as handle:
        qc_manifest = json.load(handle)
    qc_csv = _artifact_path(manifest_path, qc_manifest, "qc_csv")
    roi_csv = _artifact_path(manifest_path, qc_manifest, "roi_csv")
    if pretest_gate is not None:
        if sha256_file(manifest_path) != pretest_gate["real_qc_manifest_sha256"]:
            raise ValueError("Real-SAR QC manifest does not match the pretest-seal binding")
        if sha256_file(qc_csv) != pretest_gate["real_qc_csv_sha256"]:
            raise ValueError("Real-SAR QC CSV does not match the pretest-seal binding")
    mapping = qc_manifest["numeric_mapping"]
    qc_rows = [
        row for row in _read_csv(qc_csv)
        if row["split"] == args.split and _bool(row["valid_no_reference"])
    ]
    if args.max_images:
        qc_rows = qc_rows[: args.max_images]
    if not qc_rows:
        raise RuntimeError(f"No valid samples for split={args.split!r}")
    sample_ids = [row["sample_id"] for row in qc_rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise AssertionError("QC manifest contains duplicate sample IDs")
    if not args.nonformal_smoke:
        parent_count = len({row["parent_id"] for row in qc_rows})
        if (len(qc_rows), parent_count) != (592, 148):
            raise ValueError(
                "Formal real-SAR test requires 592 valid patches from 148 parents; "
                f"found {len(qc_rows)} patches from {parent_count} parents"
            )
    for row in qc_rows:
        noisy_path = _source_path(dataset_root, row["noisy_path"])
        if sha256_file(noisy_path) != row["noisy_sha256"]:
            raise AssertionError(f"Noisy source SHA-256 mismatch: {noisy_path}")
        if _bool(row["valid_quasi_reference"]) and row["gt16_path"]:
            gt16_path = _source_path(dataset_root, row["gt16_path"])
            if sha256_file(gt16_path) != row["gt16_sha256"]:
                raise AssertionError(f"GT16 source SHA-256 mismatch: {gt16_path}")

    sample_id_set = set(sample_ids)
    rois_by_sample: dict[str, list[dict]] = defaultdict(list)
    for row in _read_csv(roi_csv):
        if row["sample_id"] in sample_id_set:
            rois_by_sample[row["sample_id"]].append(
                {
                    "roi_id": row["roi_id"],
                    "x": int(row["x"]),
                    "y": int(row["y"]),
                    "width": int(row["width"]),
                    "height": int(row["height"]),
                }
            )
    if not args.nonformal_smoke:
        expected_roi_count = int(qc_manifest["enl_roi_policy"]["num_rois"])
        bad_roi_counts = {
            sample_id: len(rois_by_sample.get(sample_id, []))
            for sample_id in sample_ids
            if len(rois_by_sample.get(sample_id, [])) != expected_roi_count
        }
        if bad_roi_counts:
            examples = list(sorted(bad_roi_counts.items()))[:5]
            raise ValueError(
                "Formal real-SAR evaluation requires the frozen ROI count for every "
                f"patch; expected {expected_roi_count}, examples={examples}"
            )
    model: torch.nn.Module | None
    if checkpoint_path is None:
        model = None
        checkpoint_metadata: dict[str, Any] = {}
        model_identity: dict[str, Any] = {
            "method": "noisy",
            "class_name": None,
            "variant": None,
        }
        method = "noisy"
        variant = None
        method_label = args.method_label or "Noisy input"
    else:
        model, checkpoint_metadata, model_identity = _checkpoint_model(
            checkpoint_path,
            device,
            method=args.method,
            variant=args.variant,
            sar_cam_root=args.sar_cam_root,
        )
        _validate_adaptation(args.adaptation, model_identity, checkpoint_metadata)
        if not args.nonformal_smoke:
            validate_completed_checkpoint(checkpoint_path, PROTOCOL_ID)
            _validate_formal_checkpoint(
                args.adaptation, checkpoint_metadata, args.seed
            )
        method = str(model_identity["method"])
        variant = model_identity.get("variant")
        method_label = args.method_label or _default_method_label(
            model_identity, args.adaptation
        )
    run_config = vars(args).copy()
    run_config.update(
        {
            "method": method,
            "variant": variant,
            "method_label": method_label,
            "method_label_source": (
                "user_display_override"
                if args.method_label
                else ("built_in_noisy_baseline" if model is None else "checkpoint_metadata")
            ),
            "method_label_is_display_only": True,
            "model_identity": model_identity,
            "numeric_domain": INTENSITY_DOMAIN,
            "numeric_mapping": mapping,
            "metric_protocol": "icsps2026_real_v1",
            "gt16_semantics": "multitemporal_quasi_reference",
            "m_index_reported": False,
            "rgpi_reported": False,
            "gradient_metric_name": "Sobel-GC (not RGPI)",
            "bootstrap_cluster": "parent_id",
            "evaluated_sample_count": len(qc_rows),
            "evaluated_parent_count": len({row["parent_id"] for row in qc_rows}),
            "formal_run": not args.nonformal_smoke,
            "effective_device": str(device),
            "pretest_gate": pretest_gate,
        }
    )
    output = prepare_run_directory(
        args.output_dir,
        run_config,
        source_files=[
            __file__, "icsps2026_metrics.py", "prepare_icsps2026_real.py",
            "model_registry.py", "transform_main.py", "numeric_domain.py", "sar_metrics.py",
            "experiment_protocol.py", manifest_path, qc_csv, roi_csv,
        ],
        checkpoint_path=checkpoint_path,
    )
    image_dir = output / "images"
    raw_dir = output / "raw_arrays"
    if args.save_images:
        image_dir.mkdir()
        raw_dir.mkdir()

    per_image_rows = []
    roi_detail_rows = []
    for index, qc in enumerate(qc_rows, start=1):
        noisy_raw, _ = load_real_array(
            _source_path(dataset_root, qc["noisy_path"]), "noisy"
        )
        noisy = apply_fixed_mapping(noisy_raw, mapping)
        if model is None:
            prediction = noisy.copy()
        else:
            tensor = torch.from_numpy(noisy)[None, None].to(device)
            with torch.inference_mode():
                prediction_tensor = model(tensor)
            prediction = prediction_tensor.detach().float().cpu().numpy()[0, 0]
        if not np.isfinite(prediction).all():
            raise RuntimeError(f"Model produced NaN/Inf for {qc['sample_id']}")
        prediction = np.clip(prediction, 0.0, 1.0).astype(np.float32)

        ratio, floor_fraction = intensity_ratio(noisy, prediction)
        ratio_mean = float(np.mean(ratio))
        frozen_rois = rois_by_sample.get(qc["sample_id"], [])
        noisy_details = enl_roi_details(noisy, frozen_rois) if frozen_rois else []
        output_details = enl_roi_details(prediction, frozen_rois) if frozen_rois else []
        noisy_enl = [float(row["enl"]) for row in noisy_details]
        output_enl = [float(row["enl"]) for row in output_details]
        noisy_median, noisy_q1, noisy_q3 = _quartiles(noisy_enl)
        output_median, output_q1, output_q3 = _quartiles(output_enl)
        for noisy_roi, output_roi in zip(noisy_details, output_details):
            if noisy_roi["roi_id"] != output_roi["roi_id"]:
                raise AssertionError("Noisy/output ROI ordering differs")
            roi_detail_rows.append(
                {
                    "method": method,
                    "variant": variant or "",
                    "method_label": method_label,
                    "adaptation": args.adaptation,
                    "seed": args.seed,
                    "formal_run": not args.nonformal_smoke,
                    "sample_id": qc["sample_id"],
                    "parent_id": qc["parent_id"],
                    "split": qc["split"],
                    "category": qc.get("category", ""),
                    "roi_id": noisy_roi["roi_id"],
                    "x": noisy_roi["x"],
                    "y": noisy_roi["y"],
                    "width": noisy_roi["width"],
                    "height": noisy_roi["height"],
                    "variance_ddof": noisy_roi["variance_ddof"],
                    "noisy_mean": noisy_roi["mean"],
                    "noisy_variance": noisy_roi["variance"],
                    "noisy_enl": noisy_roi["enl"],
                    "output_mean": output_roi["mean"],
                    "output_variance": output_roi["variance"],
                    "output_enl": output_roi["enl"],
                }
            )

        qpsnr = qssim = float("nan")
        has_valid_gt16 = _bool(qc["valid_quasi_reference"]) and bool(qc["gt16_path"])
        if has_valid_gt16:
            gt_raw, _ = load_real_array(
                _source_path(dataset_root, qc["gt16_path"]), "gt16"
            )
            gt16 = apply_fixed_mapping(gt_raw, mapping)
            qpsnr = psnr(prediction, gt16)
            qssim = ssim(prediction, gt16)

        row = {
            "protocol_id": PROTOCOL_ID,
            "method": method,
            "variant": variant or "",
            "method_label": method_label,
            "adaptation": args.adaptation,
            "seed": args.seed,
            "formal_run": not args.nonformal_smoke,
            "sample_id": qc["sample_id"],
            "parent_id": qc["parent_id"],
            "split": qc["split"],
            "category": qc.get("category", ""),
            "noisy_path": qc["noisy_path"],
            "gt16_path": qc["gt16_path"],
            "has_valid_gt16": has_valid_gt16,
            "qpsnr_gt16": qpsnr,
            "qssim_gt16": qssim,
            "ratio_mean": ratio_mean,
            "ratio_mean_bias": abs(ratio_mean - 1.0),
            "ratio_acf_sidelobe_energy": ratio_acf_sidelobe_energy_from_ratio(ratio),
            "ratio_denominator_floor_fraction": floor_fraction,
            "sobel_gc_noisy_output": sobel_gradient_correlation(noisy, prediction),
            "enl_roi_count": len(noisy_enl),
            "noisy_enl_roi_median": noisy_median,
            "noisy_enl_roi_q1": noisy_q1,
            "noisy_enl_roi_q3": noisy_q3,
            "output_enl_roi_median": output_median,
            "output_enl_roi_q1": output_q1,
            "output_enl_roi_q3": output_q3,
        }
        if not args.nonformal_smoke:
            for metric in (
                "ratio_mean_bias",
                "ratio_acf_sidelobe_energy",
                "sobel_gc_noisy_output",
            ):
                if _finite_float(row[metric]) is None:
                    raise RuntimeError(
                        f"Formal real-SAR metric {metric} is non-finite for "
                        f"sample_id={qc['sample_id']}; refusing to publish aggregate"
                    )
        per_image_rows.append(row)

        if args.save_images:
            stem = _safe_stem(qc["sample_id"])
            np.save(raw_dir / f"{stem}_prediction.npy", prediction)
            np.save(raw_dir / f"{stem}_ratio.npy", ratio.astype(np.float32))
            _write_gray(image_dir / f"{stem}_noisy.png", noisy)
            _write_gray(image_dir / f"{stem}_prediction.png", prediction)
            ratio_display = (ratio - args.ratio_display_min) / (
                args.ratio_display_max - args.ratio_display_min
            )
            _write_gray(image_dir / f"{stem}_ratio_fixed_range.png", ratio_display)
        print(
            json.dumps(
                {"image": index, "total": len(qc_rows), "sample_id": qc["sample_id"]},
                sort_keys=True,
            ),
            flush=True,
        )

    per_image_path = output / "real_per_patch.csv"
    with per_image_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_image_rows[0]))
        writer.writeheader()
        writer.writerows(per_image_rows)
    roi_path = output / "real_enl_per_roi.csv"
    roi_fields = [
        "method", "variant", "method_label", "adaptation", "seed", "formal_run", "sample_id",
        "parent_id", "split", "category",
        "roi_id", "x", "y", "width", "height", "variance_ddof", "noisy_mean",
        "noisy_variance", "noisy_enl", "output_mean", "output_variance", "output_enl",
    ]
    with roi_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=roi_fields)
        writer.writeheader()
        writer.writerows(roi_detail_rows)
    parent_rows = _parent_rows(per_image_rows)
    parent_path = output / "real_by_parent.csv"
    with parent_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(parent_rows[0]))
        writer.writeheader()
        writer.writerows(parent_rows)

    summary_metrics = {}
    for metric in PER_IMAGE_METRICS:
        if any(_finite_float(row.get(metric)) is not None for row in per_image_rows):
            summary_metrics[metric] = parent_cluster_bootstrap(
                per_image_rows,
                value_key=metric,
                samples=args.bootstrap_samples,
                seed=args.seed,
            )
    enl_summary = {}
    for metric in ("noisy_enl", "output_enl"):
        values = [float(row[metric]) for row in roi_detail_rows if np.isfinite(float(row[metric]))]
        if not values:
            continue
        result = parent_cluster_bootstrap(
            roi_detail_rows,
            value_key=metric,
            samples=args.bootstrap_samples,
            seed=args.seed,
        )
        result["roi_q1"] = float(np.percentile(values, 25))
        result["roi_q3"] = float(np.percentile(values, 75))
        result["aggregation_note"] = "ENL is diagnostic; higher alone does not imply better quality"
        enl_summary[metric] = result
    aggregate = {
        "schema_version": 1,
        "method": method,
        "variant": variant,
        "method_label": method_label,
        "method_label_source": (
            "user_display_override"
            if args.method_label
            else ("built_in_noisy_baseline" if model is None else "checkpoint_metadata")
        ),
        "method_label_is_display_only": True,
        "adaptation": args.adaptation,
        "model_identity": model_identity,
        "seed": args.seed,
        "protocol_id": PROTOCOL_ID,
        "formal_run": not args.nonformal_smoke,
        "split": args.split,
        "num_patches": len(per_image_rows),
        "num_parent_clusters": len(parent_rows),
        "num_valid_gt16_pairs": sum(bool(row["has_valid_gt16"]) for row in per_image_rows),
        "qc_manifest_sha256": sha256_file(manifest_path),
        "qc_csv_sha256": sha256_file(qc_csv),
        "parent_cluster_bootstrap": summary_metrics,
        "enl_roi_summary": enl_summary,
        "checkpoint_metadata": {
            key: checkpoint_metadata[key]
            for key in (
                "checkpoint_format", "epoch", "global_step", "model_metadata",
                "run_config", "metrics",
            )
            if key in checkpoint_metadata
        },
        "metric_interpretation": {
            "q_prefix": "GT16 multitemporal quasi-reference only",
            "ratio_mean_bias": "lower is better",
            "ratio_acf_sidelobe_energy": "lower is better",
            "sobel_gc_noisy_output": "descriptive Sobel-GC; not EPI or RGPI",
            "enl": "diagnostic only; inspect jointly with ratio metrics and images",
            "m_index": "not reported because reference-regression validation is incomplete",
            "rgpi": "not implemented or reported",
            "noisy_identity_baseline": (
                "ratio and Sobel-GC are identity diagnostics except at denominator-floor "
                "zero pixels; report N/A and do not rank them"
                if model is None else "not applicable"
            ),
        },
        "artifacts": {
            "real_per_patch_csv": per_image_path.name,
            "real_per_patch_sha256": sha256_file(per_image_path),
            "real_enl_per_roi_csv": roi_path.name,
            "real_enl_per_roi_sha256": sha256_file(roi_path),
            "real_by_parent_csv": parent_path.name,
            "real_by_parent_sha256": sha256_file(parent_path),
        },
        "pretest_gate": pretest_gate,
    }
    atomic_json_dump(aggregate, output / "aggregate.json")
    print(json.dumps(aggregate, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
