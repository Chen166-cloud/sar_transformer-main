"""Export pre-registered RealSAR evidence panels from completed formal runs.

Only ``dataset=RealSAR`` rows already sealed in
``RUN_ROOT/records/figure_selection.csv`` are eligible.  The exporter rebuilds
their normalized inputs from the frozen QC mapping, reconstructs predictions
from completion-covered checkpoints, and publishes fixed-range crops and ratio
images with an auditable SHA-256 chain.  It has no result-based selector.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image

from experiment_protocol import (
    atomic_json_dump,
    load_checkpoint_strict,
    seed_everything,
    sha256_file,
    validate_completed_checkpoint,
)
from icsps2026_metrics import intensity_ratio
from icsps2026_pretest import (
    csv_rows,
    select_registered_real_samples,
    validate_crop_fields,
)
from model_registry import build_model, validate_checkpoint_model_metadata
from numeric_domain import INTENSITY_DOMAIN
from prepare_icsps2026_real import apply_fixed_mapping, load_real_array
from verify_icsps2026_pretest_gate import verify as verify_pretest_gate


PROTOCOL_ID = "ICSPS26-FROZEN-v2"
RATIO_DISPLAY_MIN = 0.5
RATIO_DISPLAY_MAX = 1.5
REQUIRED_BASE_METHODS = {
    "transsar_v2": ("transsar_v2", None),
    "ours_log_only": ("ours", "log_only"),
    "ours_full": ("ours", "full"),
}
OPTIONAL_BASE_METHODS = {"sar_cam": ("sar_cam", None)}
OPTIONAL_AMS_METHODS = {"ours_full_ams": ("ours", "full")}
DISPLAY_LABELS = {
    "noisy": "Noisy",
    "transsar_v2": "TransSARV2",
    "ours_log_only": "Ours (log-only)",
    "ours_full": "Ours (full)",
    "sar_cam": "SAR-CAM",
    "ours_full_ams": "Ours (full) + AMS",
}


def _parse_key_path(
    value: str, allowed: Mapping[str, tuple[str, str | None]], label: str
) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"Expected {label}_KEY=/path/to/checkpoint_best.pth")
    key, raw_path = value.split("=", 1)
    key = key.strip()
    if key not in allowed or not raw_path.strip():
        raise argparse.ArgumentTypeError(
            f"Expected one of {sorted(allowed)} as {label}_KEY and a non-empty path"
        )
    return key, Path(raw_path).expanduser().resolve()


def parse_base_key_path(value: str) -> tuple[str, Path]:
    return _parse_key_path(
        value, {**REQUIRED_BASE_METHODS, **OPTIONAL_BASE_METHODS}, "METHOD"
    )


def parse_ams_key_path(value: str) -> tuple[str, Path]:
    return _parse_key_path(value, OPTIONAL_AMS_METHODS, "AMS_METHOD")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--real-protocol-root", required=True)
    parser.add_argument(
        "--method",
        action="append",
        type=parse_base_key_path,
        required=True,
        help=(
            "Repeat METHOD_KEY=checkpoint_best.pth. Required keys: transsar_v2, "
            "ours_log_only, ours_full; sar_cam is optional."
        ),
    )
    parser.add_argument(
        "--ams",
        action="append",
        type=parse_ams_key_path,
        default=[],
        help="Optional: ours_full_ams=/path/to/completed AMS checkpoint_best.pth",
    )
    parser.add_argument("--sar-cam-root")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _safe_token(value: object, fallback: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip()).strip("._-")
    return token or fallback


def _resolve_inside(root: Path, relative: object, label: str) -> Path:
    text = str(relative)
    if not text or Path(text).is_absolute():
        raise ValueError(f"{label} must be a non-empty relative path")
    path = (root / text).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{label} escapes its declared root: {text!r}") from error
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def validate_real_qc_chain(
    dataset_root: Path,
    real_protocol_root: Path,
    gate: Mapping[str, object],
    figure_rows: Sequence[Mapping[str, object]],
) -> tuple[dict[str, Any], Path, list[dict[str, str]], list[tuple[dict[str, object], dict[str, object]]]]:
    """Validate the frozen QC chain and resolve only registered real samples."""

    manifest_path = real_protocol_root / "real_qc_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest_hash = sha256_file(manifest_path)
    if manifest_hash != gate.get("real_qc_manifest_sha256"):
        raise ValueError("REAL_PROTOCOL_ROOT manifest differs from the sealed pretest binding")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("Real-SAR QC manifest has the wrong schema")
    if manifest.get("dataset_root_name") != dataset_root.name:
        raise ValueError(
            "Dataset root basename differs from the frozen QC manifest: "
            f"{dataset_root.name!r} != {manifest.get('dataset_root_name')!r}"
        )
    expected_shape = manifest.get("expected_shape")
    if expected_shape != [256, 256]:
        raise ValueError(f"Formal RealSAR figures require frozen 256x256 data, found {expected_shape}")
    mapping = manifest.get("numeric_mapping")
    if not isinstance(mapping, dict) or mapping.get("per_image_percentile_normalization") is not False:
        raise ValueError("Real-SAR QC manifest lacks the frozen dataset-level numeric mapping")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("Real-SAR QC manifest lacks artifact hashes")
    artifact_paths: dict[str, Path] = {}
    for key in ("qc_csv", "roi_csv", "alignment_csv"):
        path = _resolve_inside(real_protocol_root, artifacts.get(key, ""), key)
        if sha256_file(path) != artifacts.get(f"{key}_sha256"):
            raise ValueError(f"Real-SAR QC artifact SHA-256 mismatch: {key}")
        artifact_paths[key] = path
    qc_path = artifact_paths["qc_csv"]
    if sha256_file(qc_path) != gate.get("real_qc_csv_sha256"):
        raise ValueError("Real-SAR QC CSV differs from the sealed pretest binding")
    qc_rows = csv_rows(qc_path)
    valid_test = [
        row
        for row in qc_rows
        if row.get("split") == "test"
        and row.get("valid_no_reference", "").strip().lower() == "true"
    ]
    test_sample_ids = [str(row.get("sample_id", "")) for row in valid_test]
    parent_counts = Counter(row.get("parent_id", "") for row in valid_test)
    if (
        len(valid_test) != 592
        or len(set(test_sample_ids)) != 592
        or "" in test_sample_ids
        or len(parent_counts) != 148
        or "" in parent_counts
        or set(parent_counts.values()) != {4}
    ):
        raise ValueError("Formal RealSAR QC must contain 592 valid test patches from 148 parents")
    matched = select_registered_real_samples(figure_rows, qc_rows)
    return manifest, qc_path, qc_rows, matched


def load_registered_noisy_sample(
    dataset_root: Path,
    mapping: Mapping[str, Any],
    registration: Mapping[str, object],
    qc: Mapping[str, object],
) -> dict[str, Any]:
    path = _resolve_inside(dataset_root, qc.get("noisy_path", ""), "noisy_path")
    if sha256_file(path) != qc.get("noisy_sha256"):
        raise ValueError(f"Noisy source SHA-256 mismatch for sample_id={qc.get('sample_id')}")
    native, field = load_real_array(path, "noisy")
    if field != qc.get("noisy_field"):
        raise ValueError(f"Noisy source field changed for sample_id={qc.get('sample_id')}")
    if native.shape != (256, 256) or not np.isfinite(native).all():
        raise ValueError(f"Invalid registered noisy array for sample_id={qc.get('sample_id')}")
    if str(native.dtype) != str(qc.get("noisy_dtype")):
        raise ValueError(f"Noisy source dtype changed for sample_id={qc.get('sample_id')}")
    for name, observed in (("noisy_min", np.min(native)), ("noisy_max", np.max(native))):
        try:
            expected = float(str(qc.get(name, "")))
        except ValueError as error:
            raise ValueError(f"Invalid frozen {name} for sample_id={qc.get('sample_id')}") from error
        if not np.isclose(float(observed), expected, rtol=1e-7, atol=1e-8):
            raise ValueError(f"Noisy source range changed for sample_id={qc.get('sample_id')}")
    normalized = apply_fixed_mapping(native, dict(mapping))
    crop = validate_crop_fields(
        registration, height=normalized.shape[0], width=normalized.shape[1]
    )
    return {
        "native": np.ascontiguousarray(native),
        "noisy": np.ascontiguousarray(normalized, dtype=np.float32),
        "source_path": path,
        "source_sha256": sha256_file(path),
        "source_field": field,
        "crop": crop,
    }


def _validate_base_checkpoint(
    key: str, checkpoint_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    method, variant = {**REQUIRED_BASE_METHODS, **OPTIONAL_BASE_METHODS}[key]
    completion = validate_completed_checkpoint(checkpoint_path, PROTOCOL_ID)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"{key} checkpoint must be a metadata-bearing mapping")
    run_config = checkpoint.get("run_config")
    if not isinstance(run_config, dict):
        raise ValueError(f"{key} checkpoint lacks run_config")
    expected = {
        "protocol_id": PROTOCOL_ID,
        "artifact_protocol_id": PROTOCOL_ID,
        "formal_run": True,
        "numeric_domain": INTENSITY_DOMAIN,
        "method": method,
        "variant": variant,
        "seed": 42,
        "target_updates": 100_000,
    }
    mismatch = {
        field: {"expected": value, "found": run_config.get(field)}
        for field, value in expected.items()
        if run_config.get(field) != value
    }
    if run_config.get("adaptation") is not None:
        mismatch["adaptation"] = {"expected": None, "found": run_config.get("adaptation")}
    if checkpoint.get("checkpoint_format") != "icsps26-resumable-v1":
        mismatch["checkpoint_format"] = {
            "expected": "icsps26-resumable-v1",
            "found": checkpoint.get("checkpoint_format"),
        }
    step = checkpoint.get("global_step")
    if type(step) is not int or step < 0 or step > 100_000 or step % 5_000:
        mismatch["global_step"] = {
            "expected": "integer validation step in {0,5000,...,100000}",
            "found": step,
        }
    best = completion.get("best")
    if not isinstance(best, dict) or best.get("global_step") != step:
        mismatch["completion_best_step"] = {
            "expected": step,
            "found": best.get("global_step") if isinstance(best, dict) else best,
        }
    elif (
        not np.isfinite(float(best.get("val_macro_mse", np.nan)))
        or float(best["val_macro_mse"]) < 0.0
    ):
        mismatch["completion_best_val_macro_mse"] = {
            "expected": "finite and non-negative",
            "found": best.get("val_macro_mse"),
        }
    checkpoint_hash = sha256_file(checkpoint_path)
    if completion.get("checkpoint_best_sha256") != checkpoint_hash:
        mismatch["checkpoint_best_sha256"] = {
            "expected": checkpoint_hash,
            "found": completion.get("checkpoint_best_sha256"),
        }
    if completion.get("completed_updates") != 100_000 or completion.get("target_updates") != 100_000:
        mismatch["completion_updates"] = {
            "expected": "100000/100000",
            "found": f"{completion.get('completed_updates')}/{completion.get('target_updates')}",
        }
    if mismatch:
        raise ValueError(f"{key} is not a compatible completed formal base run: {mismatch}")
    validate_checkpoint_model_metadata(checkpoint, method, variant)
    return checkpoint, completion


def _validate_ams_checkpoint(
    key: str, checkpoint_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    method, variant = OPTIONAL_AMS_METHODS[key]
    completion = validate_completed_checkpoint(checkpoint_path, PROTOCOL_ID)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"{key} checkpoint must be a metadata-bearing mapping")
    run_config = checkpoint.get("run_config")
    if not isinstance(run_config, dict):
        raise ValueError(f"{key} checkpoint lacks run_config")
    expected = {
        "protocol_id": PROTOCOL_ID,
        "artifact_protocol_id": PROTOCOL_ID,
        "formal_run": True,
        "numeric_domain": INTENSITY_DOMAIN,
        "method": method,
        "variant": variant,
        "adaptation": "ams",
        "seed": 42,
        "epochs": 8,
        "selection": "lowest fixed-mask real-validation loss over adapted epochs 1-8",
    }
    mismatch = {
        field: {"expected": value, "found": run_config.get(field)}
        for field, value in expected.items()
        if run_config.get(field) != value
    }
    if checkpoint.get("checkpoint_format") != "icsps26-ams-v1":
        mismatch["checkpoint_format"] = {
            "expected": "icsps26-ams-v1",
            "found": checkpoint.get("checkpoint_format"),
        }
    epoch = checkpoint.get("epoch")
    if type(epoch) is not int or epoch < 1 or epoch > 8:
        mismatch["epoch"] = {"expected": "integer in [1,8]", "found": epoch}
    best = completion.get("best")
    if not isinstance(best, dict) or best.get("epoch") != epoch:
        mismatch["completion_best_epoch"] = {
            "expected": epoch,
            "found": best.get("epoch") if isinstance(best, dict) else best,
        }
    elif not np.isfinite(float(best.get("fixed_val_masked_loss", np.nan))):
        mismatch["completion_best_fixed_val_masked_loss"] = {
            "expected": "finite",
            "found": best.get("fixed_val_masked_loss"),
        }
    checkpoint_hash = sha256_file(checkpoint_path)
    if completion.get("checkpoint_best_sha256") != checkpoint_hash:
        mismatch["checkpoint_best_sha256"] = {
            "expected": checkpoint_hash,
            "found": completion.get("checkpoint_best_sha256"),
        }
    if completion.get("completed_epochs") != 8:
        mismatch["completion_state"] = {
            "expected": "8 completed epochs with a real-validation selected best",
            "found": {"completed_epochs": completion.get("completed_epochs")},
        }
    if mismatch:
        raise ValueError(f"{key} is not a compatible completed formal AMS run: {mismatch}")
    validate_checkpoint_model_metadata(checkpoint, method, variant)
    return checkpoint, completion


def build_completed_model(
    key: str,
    checkpoint_path: Path,
    adaptation: str,
    device: torch.device,
    sar_cam_root: str | None,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    if checkpoint_path.name != "checkpoint_best.pth":
        raise ValueError(f"{key} must use checkpoint_best.pth")
    if adaptation == "base":
        checkpoint, completion = _validate_base_checkpoint(key, checkpoint_path)
        method, variant = {**REQUIRED_BASE_METHODS, **OPTIONAL_BASE_METHODS}[key]
    elif adaptation == "ams":
        checkpoint, completion = _validate_ams_checkpoint(key, checkpoint_path)
        method, variant = OPTIONAL_AMS_METHODS[key]
    else:  # defensive; CLI never reaches this branch
        raise ValueError(f"Unsupported adaptation: {adaptation}")
    model = build_model(
        method,
        variant=variant,
        external_root=sar_cam_root,
        checkpoint_metadata=checkpoint,
    )
    load_checkpoint_strict(model, checkpoint_path, map_location="cpu")
    model.to(device).eval()
    return model, {
        "method": method,
        "variant": variant,
        "adaptation": adaptation,
        "seed": 42,
        "selection_step_or_epoch": (
            checkpoint.get("global_step") if adaptation == "base" else checkpoint.get("epoch")
        ),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "completion": str(checkpoint_path.parent / "completion.json"),
        "completion_sha256": sha256_file(checkpoint_path.parent / "completion.json"),
        "completion_selection": completion.get("best"),
    }


def _save_gray_png(array: np.ndarray, path: Path) -> None:
    display = np.rint(np.clip(array, 0.0, 1.0) * 65_535.0).astype(np.uint16)
    Image.fromarray(display).save(path, format="PNG")


def _save_ratio_png(array: np.ndarray, path: Path) -> None:
    from matplotlib import colormaps

    normalized = (
        np.clip(array, RATIO_DISPLAY_MIN, RATIO_DISPLAY_MAX) - RATIO_DISPLAY_MIN
    ) / (RATIO_DISPLAY_MAX - RATIO_DISPLAY_MIN)
    rgba = colormaps["coolwarm"](normalized, bytes=True)
    Image.fromarray(np.asarray(rgba, dtype=np.uint8), mode="RGBA").save(path, format="PNG")


def _save_panel(
    arrays: Mapping[str, np.ndarray],
    ratios: Mapping[str, np.ndarray],
    crop: tuple[int, int, int, int],
    title: str,
    png_path: Path,
    pdf_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x, y, width, height = crop
    keys = list(arrays)
    figure, axes = plt.subplots(
        2,
        len(keys),
        figsize=(2.15 * len(keys), 4.25),
        squeeze=False,
        constrained_layout=True,
    )
    ratio_image = None
    for column, key in enumerate(keys):
        image_crop = np.clip(arrays[key], 0.0, 1.0)[y : y + height, x : x + width]
        ratio_crop = ratios[key][y : y + height, x : x + width]
        axes[0, column].imshow(image_crop, cmap="gray", vmin=0.0, vmax=1.0)
        axes[0, column].set_title(DISPLAY_LABELS.get(key, key), fontsize=9)
        ratio_image = axes[1, column].imshow(
            ratio_crop,
            cmap="coolwarm",
            vmin=RATIO_DISPLAY_MIN,
            vmax=RATIO_DISPLAY_MAX,
        )
        axes[1, column].set_title("Noisy / estimate", fontsize=8)
        for row in (0, 1):
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
    assert ratio_image is not None
    colorbar = figure.colorbar(ratio_image, ax=axes[1, :].tolist(), fraction=0.018, pad=0.01)
    colorbar.set_label(f"fixed range [{RATIO_DISPLAY_MIN}, {RATIO_DISPLAY_MAX}]", fontsize=8)
    figure.suptitle(title, fontsize=10)
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight", metadata={"Creator": __file__})
    plt.close(figure)


def main() -> int:
    args = parse_args()
    run_root = Path(args.run_root).expanduser().resolve()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    real_protocol_root = Path(args.real_protocol_root).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    if not dataset_root.is_dir() or not real_protocol_root.is_dir():
        raise FileNotFoundError("dataset-root and real-protocol-root must exist")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite RealSAR figure directory: {output}")
    base_methods = dict(args.method)
    ams_methods = dict(args.ams)
    if len(base_methods) != len(args.method) or len(ams_methods) != len(args.ams):
        raise ValueError("Duplicate --method/--ams key")
    missing = sorted(set(REQUIRED_BASE_METHODS).difference(base_methods))
    if missing:
        raise ValueError(f"Missing required completed formal base methods: {missing}")
    if "sar_cam" in base_methods and not args.sar_cam_root:
        raise ValueError("--sar-cam-root is required when sar_cam is exported")

    gate = verify_pretest_gate(run_root)
    figure_selection = run_root / "records" / "figure_selection.csv"
    figure_rows = csv_rows(figure_selection)
    manifest, qc_path, _, matched = validate_real_qc_chain(
        dataset_root, real_protocol_root, gate, figure_rows
    )
    validation_selection = run_root / "records" / "validation_selection.csv"
    strongest = [
        row
        for row in csv_rows(validation_selection)
        if row.get("selection_role") == "strongest_non_ours_baseline"
    ][0]
    strongest_key = strongest["method"]
    if strongest_key not in base_methods:
        raise ValueError(f"Figure export must include sealed strongest baseline: {strongest_key}")
    if sha256_file(base_methods[strongest_key]) != strongest["checkpoint_sha256"]:
        raise ValueError("Strongest-baseline checkpoint differs from the sealed selection")

    selected: list[dict[str, Any]] = []
    for index, (registration, qc) in enumerate(matched, start=1):
        loaded = load_registered_noisy_sample(
            dataset_root, manifest["numeric_mapping"], registration, qc
        )
        selected.append(
            {
                "index": index,
                "registration": registration,
                "qc": qc,
                **loaded,
                "raw_predictions": {"noisy": loaded["noisy"]},
            }
        )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested for RealSAR figure export but CUDA is unavailable")
    seed_everything(42)
    method_provenance: dict[str, Any] = {}
    method_specs = [
        *((key, path, "base") for key, path in args.method),
        *((key, path, "ams") for key, path in args.ams),
    ]
    for key, checkpoint_path, adaptation in method_specs:
        model, provenance = build_completed_model(
            key, checkpoint_path, adaptation, device, args.sar_cam_root
        )
        method_provenance[key] = provenance
        for sample in selected:
            tensor = torch.from_numpy(sample["noisy"])[None, None].to(device)
            with torch.inference_mode():
                prediction_tensor = model(tensor)
            if prediction_tensor.shape != tensor.shape or not bool(
                torch.isfinite(prediction_tensor).all().item()
            ):
                raise ValueError(
                    f"{key} produced invalid output for sample_id={sample['qc']['sample_id']}"
                )
            sample["raw_predictions"][key] = np.ascontiguousarray(
                prediction_tensor.detach().cpu().numpy()[0, 0].astype(np.float32, copy=False)
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        ordered_series = [
            "noisy",
            *[key for key, _ in args.method],
            *[key for key, _ in args.ams],
        ]
        manifest_rows: list[dict[str, Any]] = []
        panel_rows: list[dict[str, str]] = []
        for sample in selected:
            registration = sample["registration"]
            qc = sample["qc"]
            x, y, width, height = sample["crop"]
            sample_name = (
                f"sample_{sample['index']:02d}_"
                f"{_safe_token(qc['sample_id'], 'sample')}"
            )
            sample_dir = staging / sample_name
            sample_dir.mkdir()
            native_path = sample_dir / "noisy_source_native.npy"
            np.save(native_path, sample["native"], allow_pickle=False)
            clipped_arrays: dict[str, np.ndarray] = {}
            ratios: dict[str, np.ndarray] = {}
            for key in ordered_series:
                raw = sample["raw_predictions"][key]
                clipped = np.clip(raw, 0.0, 1.0).astype(np.float32, copy=False)
                ratio, floor_fraction = intensity_ratio(sample["noisy"], clipped)
                clipped_arrays[key] = clipped
                ratios[key] = np.ascontiguousarray(ratio, dtype=np.float32)
                method_dir = sample_dir / key
                method_dir.mkdir()
                raw_path = method_dir / "full_raw.npy"
                postclip_path = method_dir / "full_postclip_0_1.npy"
                ratio_path = method_dir / "ratio_noisy_over_estimate.npy"
                crop_path = method_dir / "crop_gray_0_1.png"
                ratio_png = method_dir / "crop_ratio_fixed_0_5_1_5.png"
                np.save(raw_path, raw, allow_pickle=False)
                np.save(postclip_path, clipped, allow_pickle=False)
                np.save(ratio_path, ratios[key], allow_pickle=False)
                _save_gray_png(clipped[y : y + height, x : x + width], crop_path)
                _save_ratio_png(ratios[key][y : y + height, x : x + width], ratio_png)
                identity = method_provenance.get(key, {})
                manifest_rows.append(
                    {
                        "protocol_id": PROTOCOL_ID,
                        "registration_row": sample["index"],
                        "figure": registration.get("figure", ""),
                        "panel": registration.get("panel", ""),
                        "sample_id": qc["sample_id"],
                        "parent_id": qc["parent_id"],
                        "region_type": registration["region_type"],
                        "selection_rule": registration["selection_rule"],
                        "crop_x": x,
                        "crop_y": y,
                        "crop_width": width,
                        "crop_height": height,
                        "series_key": key,
                        "display_label": DISPLAY_LABELS.get(key, key),
                        "method": identity.get("method", "noisy"),
                        "variant": identity.get("variant") or "",
                        "adaptation": identity.get("adaptation", "base"),
                        "seed": identity.get("seed", 42),
                        "checkpoint_sha256": identity.get("checkpoint_sha256", ""),
                        "noisy_source_sha256": sample["source_sha256"],
                        "full_raw_npy": str(raw_path.relative_to(staging)),
                        "full_raw_npy_sha256": sha256_file(raw_path),
                        "full_postclip_npy": str(postclip_path.relative_to(staging)),
                        "full_postclip_npy_sha256": sha256_file(postclip_path),
                        "ratio_npy": str(ratio_path.relative_to(staging)),
                        "ratio_npy_sha256": sha256_file(ratio_path),
                        "crop_gray_png": str(crop_path.relative_to(staging)),
                        "crop_gray_png_sha256": sha256_file(crop_path),
                        "crop_ratio_png": str(ratio_png.relative_to(staging)),
                        "crop_ratio_png_sha256": sha256_file(ratio_png),
                        "raw_min": float(np.min(raw)),
                        "raw_max": float(np.max(raw)),
                        "postclip_fraction": float(np.mean((raw < 0.0) | (raw > 1.0))),
                        "ratio_denominator_floor_fraction": floor_fraction,
                    }
                )
            panel_png = sample_dir / "comparison_panel.png"
            panel_pdf = sample_dir / "comparison_panel.pdf"
            _save_panel(
                clipped_arrays,
                ratios,
                sample["crop"],
                (
                    f"RealSAR | {registration['region_type']} | "
                    f"sample={qc['sample_id']} | pre-registered crop"
                ),
                panel_png,
                panel_pdf,
            )
            panel_rows.append(
                {
                    "sample_id": qc["sample_id"],
                    "source_native_npy": str(native_path.relative_to(staging)),
                    "source_native_npy_sha256": sha256_file(native_path),
                    "png": str(panel_png.relative_to(staging)),
                    "png_sha256": sha256_file(panel_png),
                    "pdf": str(panel_pdf.relative_to(staging)),
                    "pdf_sha256": sha256_file(panel_pdf),
                }
            )

        manifest_output = staging / "figure_manifest.csv"
        with manifest_output.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
            writer.writeheader()
            writer.writerows(manifest_rows)
        provenance = {
            "protocol_id": PROTOCOL_ID,
            "formal_run": True,
            "selection_policy": "sealed pre-registered RealSAR rows only; no result-based selection",
            "pretest_gate": gate,
            "run_root": str(run_root),
            "dataset_root": str(dataset_root),
            "real_protocol_root": str(real_protocol_root),
            "real_qc_manifest": str(real_protocol_root / "real_qc_manifest.json"),
            "real_qc_manifest_sha256": sha256_file(
                real_protocol_root / "real_qc_manifest.json"
            ),
            "real_qc_csv": str(qc_path),
            "real_qc_csv_sha256": sha256_file(qc_path),
            "figure_selection": str(figure_selection),
            "figure_selection_sha256": sha256_file(figure_selection),
            "validation_selection": str(validation_selection),
            "validation_selection_sha256": sha256_file(validation_selection),
            "registered_region_types_required": ["homogeneous", "structured"],
            "selected_samples": len(selected),
            "method_order": ordered_series,
            "checkpoints": method_provenance,
            "numeric_mapping": manifest["numeric_mapping"],
            "display": {
                "grayscale_range": [0.0, 1.0],
                "grayscale_png_encoding": "uint16 fixed linear mapping",
                "ratio_definition": "normalized noisy / max(post-clipped estimate, 1e-6)",
                "ratio_display_range": [RATIO_DISPLAY_MIN, RATIO_DISPLAY_MAX],
                "ratio_colormap": "coolwarm",
                "noisy_ratio_interpretation": (
                    "identity diagnostic only; it must not be ranked against learned methods"
                ),
                "model_display_operation": "clip raw prediction to [0,1]",
            },
            "source_code": str(Path(__file__).resolve()),
            "source_code_sha256": sha256_file(__file__),
            "shared_implementation_sha256": {
                name: sha256_file(Path(__file__).resolve().with_name(name))
                for name in (
                    "evaluate_icsps2026_real.py",
                    "icsps2026_pretest.py",
                    "icsps2026_metrics.py",
                    "prepare_icsps2026_real.py",
                    "model_registry.py",
                    "numeric_domain.py",
                )
            },
            "panels": panel_rows,
        }
        provenance_path = staging / "provenance.json"
        atomic_json_dump(provenance, provenance_path)
        completion = {
            "protocol_id": PROTOCOL_ID,
            "formal_run": True,
            "state": "complete",
            "selected_samples": len(selected),
            "exported_series": len(ordered_series),
            "figure_manifest": manifest_output.name,
            "figure_manifest_sha256": sha256_file(manifest_output),
            "provenance": provenance_path.name,
            "provenance_sha256": sha256_file(provenance_path),
            "pretest_seal_sha256": gate["pretest_seal_sha256"],
        }
        atomic_json_dump(completion, staging / "completion.json")
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite RealSAR figure directory: {output}")
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(completion, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
