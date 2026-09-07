"""Export pre-registered UCM evidence panels from completed formal runs.

This entry point deliberately has no image-quality-based selection option.  It
accepts only the UCM rows frozen in ``RUN_ROOT/records/figure_selection.csv``,
resolves them against the prepared fixed-pair manifest, and reconstructs each
learned output from a checkpoint covered by a formal completion marker.
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
from scipy.io import loadmat

from experiment_protocol import (
    atomic_json_dump,
    load_checkpoint_strict,
    seed_everything,
    sha256_file,
    validate_completed_checkpoint,
)
from finalize_icsps2026_sarbm3d_raw import (
    SARBM3D_DOMAIN_CONVERSION,
    SARBM3D_INPUT_DOMAIN,
    SARBM3D_OUTPUT_DOMAIN,
)
from icsps2026_pretest import (
    canonical_int as _canonical_int,
    select_registered_ucm_pairs as _select_registered_ucm_pairs,
    validate_crop_fields,
)
from icsps2026_protocol import (
    ICSPSFixedPairDataset,
    read_csv_records,
    resolve_portable,
    validate_pair_manifest,
)
from model_registry import build_model, validate_checkpoint_model_metadata
from numeric_domain import INTENSITY_DOMAIN
from verify_icsps2026_artifact import verify as verify_artifact
from verify_icsps2026_pretest_gate import verify as verify_pretest_gate


PROTOCOL_ID = "ICSPS26-FROZEN-v2"
REQUIRED_METHODS = {
    "transsar_v2": ("transsar_v2", None),
    "ours_wout_all_fdr": ("ours", "wout_all_fdr"),
    "ours_full": ("ours", "full"),
}
OPTIONAL_METHODS = {"sar_cam": ("sar_cam", None)}
DISPLAY_LABELS = {
    "clean": "Clean",
    "noisy": "Noisy",
    "transsar_v2": "TransSARV2",
    "ours_wout_all_fdr": "Ours w/o all FDR",
    "ours_full": "Ours (full)",
    "sar_cam": "SAR-CAM",
    "sar_bm3d": "SAR-BM3D",
}


def parse_key_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected METHOD_KEY=/path/to/checkpoint_best.pth")
    key, raw_path = value.split("=", 1)
    key = key.strip()
    if key not in {*REQUIRED_METHODS, *OPTIONAL_METHODS} or not raw_path.strip():
        allowed = sorted({*REQUIRED_METHODS, *OPTIONAL_METHODS})
        raise argparse.ArgumentTypeError(
            f"Expected one of {allowed} as METHOD_KEY and a non-empty checkpoint path"
        )
    return key, Path(raw_path).expanduser().resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--prep-root", required=True)
    parser.add_argument(
        "--method",
        action="append",
        type=parse_key_path,
        required=True,
        help=(
            "Repeat METHOD_KEY=checkpoint_best.pth. Required keys: "
            "transsar_v2, ours_wout_all_fdr, ours_full; sar_cam is optional."
        ),
    )
    parser.add_argument("--sar-cam-root")
    parser.add_argument(
        "--sarbm3d-root",
        help="Optional finalized formal SAR-BM3D raw-prediction directory",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _safe_token(value: object, fallback: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip()).strip("._-")
    return token or fallback


def select_registered_ucm_pairs(
    figure_rows: Sequence[Mapping[str, object]],
    manifest_rows: Sequence[Mapping[str, object]],
    *,
    required_looks: set[int] | None = None,
) -> list[tuple[dict[str, object], dict[str, object]]]:
    """Compatibility wrapper around the shared seal/export semantic validator."""

    if required_looks not in (None, {1, 4}):
        raise ValueError("The frozen protocol requires exactly the L=1/L=4 strata")
    return _select_registered_ucm_pairs(figure_rows, manifest_rows)


def _load_and_validate_prepared_data(
    prep_root: Path,
) -> tuple[dict[str, Any], Path, list[dict[str, str]]]:
    config_path = prep_root / "config_snapshot.json"
    manifest_path = prep_root / "ucm_test_manifest.csv"
    hashes_path = prep_root / "artifact_hashes.json"
    for path in (config_path, manifest_path, hashes_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing prepared-data artifact: {path}")
    with config_path.open("r", encoding="utf-8") as handle:
        snapshot = json.load(handle)
    config = snapshot.get("effective_protocol")
    if not isinstance(config, dict) or config.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Prepared config snapshot is not ICSPS26-FROZEN-v2")
    if config.get("artifact_protocol_id") != PROTOCOL_ID or config.get("mode") != "formal":
        raise ValueError("Figure export requires a formal prepared-data snapshot")
    with hashes_path.open("r", encoding="utf-8") as handle:
        frozen_hashes = json.load(handle)
    if (
        frozen_hashes.get("state") != "complete"
        or frozen_hashes.get("protocol_id") != PROTOCOL_ID
    ):
        raise ValueError("Prepared-data artifact hash index is not a completed formal run")
    critical = frozen_hashes.get("critical_files", {})
    if not isinstance(critical, dict) or critical.get(manifest_path.name) != sha256_file(manifest_path):
        raise ValueError("UCM manifest is not covered by the prepared-data hash chain")
    manifest_rows = read_csv_records(manifest_path)
    validate_pair_manifest(
        manifest_rows,
        expected_dataset="UCMerced_LandUse",
        expected_classes=int(config["ucm"]["expected_classes"]),
        expected_per_class_per_look=int(config["ucm"]["expected_per_class"]),
        looks=config["looks"],
        output_root=prep_root,
        verify_file_hashes=False,
        synthesis_seed=int(config["seeds"]["ucm_test"]),
    )
    if len(manifest_rows) != 8_400:
        raise ValueError(f"Formal UCM manifest must contain 8400 pairs, found {len(manifest_rows)}")
    return config, manifest_path, manifest_rows


def _checkpoint_model(
    method_key: str,
    checkpoint_path: Path,
    device: torch.device,
    sar_cam_root: str | None,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    expected_method, expected_variant = {
        **REQUIRED_METHODS,
        **OPTIONAL_METHODS,
    }[method_key]
    if checkpoint_path.name != "checkpoint_best.pth":
        raise ValueError(
            f"{method_key} must use the validation-selected checkpoint_best.pth"
        )
    completion = validate_completed_checkpoint(checkpoint_path, PROTOCOL_ID)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"{method_key} checkpoint must be a metadata-bearing mapping")
    run_config = checkpoint.get("run_config")
    if not isinstance(run_config, dict):
        raise ValueError(f"{method_key} checkpoint lacks run_config")
    expected = {
        "protocol_id": PROTOCOL_ID,
        "artifact_protocol_id": PROTOCOL_ID,
        "formal_run": True,
        "method": expected_method,
        "variant": expected_variant,
        "seed": 42,
        "target_updates": 100_000,
        "numeric_domain": INTENSITY_DOMAIN,
    }
    mismatch = {
        key: {"expected": value, "found": run_config.get(key)}
        for key, value in expected.items()
        if run_config.get(key) != value
    }
    if checkpoint.get("checkpoint_format") != "icsps26-resumable-v1":
        mismatch["checkpoint_format"] = {
            "expected": "icsps26-resumable-v1",
            "found": checkpoint.get("checkpoint_format"),
        }
    step = checkpoint.get("global_step")
    if type(step) is not int or not 0 <= step <= 100_000 or step % 5_000:
        mismatch["global_step"] = {
            "expected": "integer validation step in {0,5000,...,100000}",
            "found": step,
        }
    if completion.get("completed_updates") != 100_000 or completion.get("target_updates") != 100_000:
        mismatch["completion_updates"] = {
            "expected": "100000/100000",
            "found": f"{completion.get('completed_updates')}/{completion.get('target_updates')}",
        }
    completion_best = completion.get("best")
    if not isinstance(completion_best, dict) or completion_best.get("global_step") != step:
        mismatch["completion_best_step"] = {
            "expected": step,
            "found": (
                completion_best.get("global_step")
                if isinstance(completion_best, dict)
                else completion_best
            ),
        }
    elif (
        not np.isfinite(float(completion_best.get("val_macro_mse", np.nan)))
        or float(completion_best["val_macro_mse"]) < 0.0
    ):
        mismatch["completion_best_val_macro_mse"] = {
            "expected": "finite and non-negative",
            "found": completion_best.get("val_macro_mse"),
        }
    if mismatch:
        raise ValueError(f"{method_key} is not a compatible completed formal run: {mismatch}")
    validate_checkpoint_model_metadata(checkpoint, expected_method, expected_variant)
    model = build_model(
        expected_method,
        variant=expected_variant,
        external_root=sar_cam_root,
        checkpoint_metadata=checkpoint,
    )
    load_checkpoint_strict(model, checkpoint_path, map_location="cpu")
    model.to(device).eval()
    return model, {
        "method": expected_method,
        "variant": expected_variant,
        "seed": 42,
        "global_step": step,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "completion": str(checkpoint_path.parent / "completion.json"),
        "completion_sha256": sha256_file(checkpoint_path.parent / "completion.json"),
    }


def _load_sarbm3d_selected(
    root: Path,
    pair_rows: Sequence[Mapping[str, object]],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    completion_path = root / "completion.json"
    if not completion_path.is_file():
        raise FileNotFoundError(f"SAR-BM3D completion marker is missing: {completion_path}")
    verify_artifact(
        "sarbm_raw", completion_path, expected_method="sar_bm3d",
        expected_variant="v1.0", expected_seed="", expected_adaptation="",
    )
    with completion_path.open("r", encoding="utf-8") as handle:
        completion = json.load(handle)
    expected = {
        "protocol_id": PROTOCOL_ID,
        "formal_run": True,
        "method": "sar_bm3d",
        "variant": "v1.0",
        "completed_jobs": 8_400,
        "expected_jobs": 8_400,
        "sarbm3d_input_domain": SARBM3D_INPUT_DOMAIN,
        "scored_output_domain": SARBM3D_OUTPUT_DOMAIN,
        "domain_conversion": SARBM3D_DOMAIN_CONVERSION,
    }
    mismatch = {
        key: {"expected": value, "found": completion.get(key)}
        for key, value in expected.items()
        if completion.get(key) != value
    }
    if mismatch:
        raise ValueError(f"SAR-BM3D raw completion is not formal/complete: {mismatch}")
    manifest_path = resolve_portable(root, str(completion.get("prediction_manifest", "")))
    if sha256_file(manifest_path) != completion.get("prediction_manifest_sha256"):
        raise ValueError("SAR-BM3D prediction manifest SHA-256 mismatch")
    rows = read_csv_records(manifest_path)
    pair_ids = [str(row.get("pair_id", "")) for row in rows]
    if len(rows) != 8_400 or len(set(pair_ids)) != 8_400 or "" in pair_ids:
        raise ValueError("SAR-BM3D prediction manifest is not a unique 8400-pair result")
    if any(
        row.get("protocol_id") != PROTOCOL_ID
        or not _bool(row.get("formal_run"))
        or row.get("method") != "sar_bm3d"
        or row.get("variant") != "v1.0"
        for row in rows
    ):
        raise ValueError("SAR-BM3D prediction manifest contains mixed/non-formal identity")
    if Counter(_canonical_int(row["L"], "L") for row in rows) != Counter(
        {1: 2_100, 2: 2_100, 4: 2_100, 8: 2_100}
    ):
        raise ValueError("SAR-BM3D prediction manifest has the wrong per-look counts")
    by_pair = {row["pair_id"]: row for row in rows}
    predictions: dict[str, np.ndarray] = {}
    for pair in pair_rows:
        pair_id = str(pair["pair_id"])
        if pair_id not in by_pair:
            raise ValueError(f"SAR-BM3D result lacks registered pair_id={pair_id}")
        row = by_pair[pair_id]
        if row.get("input_mat_sha256") != pair.get("mat_sha256"):
            raise ValueError(f"SAR-BM3D input hash mismatch for pair_id={pair_id}")
        if _canonical_int(row["L"], "L") != _canonical_int(pair["global_L"], "global_L"):
            raise ValueError(f"SAR-BM3D look mismatch for pair_id={pair_id}")
        if (
            row.get("input_domain") != SARBM3D_INPUT_DOMAIN
            or row.get("output_domain") != SARBM3D_OUTPUT_DOMAIN
            or row.get("domain_conversion") != SARBM3D_DOMAIN_CONVERSION
        ):
            raise ValueError(f"SAR-BM3D domain metadata mismatch for pair_id={pair_id}")
        prediction_path = resolve_portable(root, row["prediction_path"])
        if sha256_file(prediction_path) != row.get("prediction_sha256"):
            raise ValueError(f"SAR-BM3D prediction hash mismatch for pair_id={pair_id}")
        payload = loadmat(
            prediction_path,
            variable_names=[
                "prediction", "pair_id", "looks", "input_domain",
                "output_domain", "domain_conversion",
            ],
        )
        prediction = np.asarray(payload.get("prediction"), dtype=np.float32).squeeze()
        if prediction.shape != (256, 256) or not np.isfinite(prediction).all():
            raise ValueError(f"Invalid SAR-BM3D prediction for pair_id={pair_id}")
        embedded_pair = np.asarray(payload.get("pair_id")).squeeze()
        if embedded_pair.dtype.kind not in {"U", "S"}:
            raise ValueError(f"SAR-BM3D pair_id metadata is invalid for {pair_id}")
        pieces = embedded_pair.reshape(-1).tolist()
        embedded_text = "".join(
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in pieces
        )
        if embedded_text != pair_id:
            raise ValueError(f"SAR-BM3D embedded pair_id mismatch for {pair_id}")
        embedded_look = float(np.asarray(payload.get("looks")).squeeze())
        if embedded_look != _canonical_int(pair["global_L"], "global_L"):
            raise ValueError(f"SAR-BM3D embedded L mismatch for {pair_id}")
        for field, expected_value in (
            ("input_domain", SARBM3D_INPUT_DOMAIN),
            ("output_domain", SARBM3D_OUTPUT_DOMAIN),
            ("domain_conversion", SARBM3D_DOMAIN_CONVERSION),
        ):
            values = np.asarray(payload.get(field)).squeeze()
            if values.dtype.kind not in {"U", "S"}:
                raise ValueError(f"SAR-BM3D embedded {field} is invalid for {pair_id}")
            actual_value = "".join(
                value.decode("utf-8") if isinstance(value, bytes) else str(value)
                for value in values.reshape(-1).tolist()
            )
            if actual_value != expected_value:
                raise ValueError(f"SAR-BM3D embedded {field} mismatch for {pair_id}")
        predictions[pair_id] = np.ascontiguousarray(prediction)
    return predictions, {
        "root": str(root),
        "completion": str(completion_path),
        "completion_sha256": sha256_file(completion_path),
        "prediction_manifest": str(manifest_path),
        "prediction_manifest_sha256": sha256_file(manifest_path),
    }


def validate_crop(array: np.ndarray, registration: Mapping[str, object]) -> tuple[int, int, int, int]:
    return validate_crop_fields(
        registration, height=int(array.shape[0]), width=int(array.shape[1])
    )


def _save_gray_png(array: np.ndarray, path: Path) -> None:
    display = np.rint(np.clip(array, 0.0, 1.0) * 65_535.0).astype(np.uint16)
    Image.fromarray(display).save(path, format="PNG")


def _save_error_png(array: np.ndarray, path: Path) -> None:
    from matplotlib import colormaps

    rgba = colormaps["magma"](np.clip(array, 0.0, 1.0), bytes=True)
    Image.fromarray(np.asarray(rgba, dtype=np.uint8), mode="RGBA").save(path, format="PNG")


def _save_panel(
    arrays: Mapping[str, np.ndarray],
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
    clean = np.clip(arrays["clean"], 0.0, 1.0)
    figure, axes = plt.subplots(
        2,
        len(keys),
        figsize=(2.15 * len(keys), 4.25),
        squeeze=False,
        constrained_layout=True,
    )
    error_image = None
    for column, key in enumerate(keys):
        clipped = np.clip(arrays[key], 0.0, 1.0)
        crop_image = clipped[y : y + height, x : x + width]
        crop_error = np.abs(crop_image - clean[y : y + height, x : x + width])
        axes[0, column].imshow(crop_image, cmap="gray", vmin=0.0, vmax=1.0)
        axes[0, column].set_title(DISPLAY_LABELS.get(key, key), fontsize=9)
        error_image = axes[1, column].imshow(
            crop_error, cmap="magma", vmin=0.0, vmax=1.0
        )
        axes[1, column].set_title("Absolute error", fontsize=8)
        for row in (0, 1):
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
    assert error_image is not None
    figure.colorbar(error_image, ax=axes[1, :].tolist(), fraction=0.018, pad=0.01)
    figure.suptitle(title, fontsize=10)
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight", metadata={"Creator": __file__})
    plt.close(figure)


def _publish_directory(staging: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite figure output directory: {output}")
    os.replace(staging, output)


def main() -> int:
    args = parse_args()
    run_root = Path(args.run_root).expanduser().resolve()
    prep_root = Path(args.prep_root).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite figure output directory: {output}")
    methods = dict(args.method)
    if len(methods) != len(args.method):
        raise ValueError("Duplicate --method key")
    missing = sorted(set(REQUIRED_METHODS).difference(methods))
    if missing:
        raise ValueError(f"Missing required completed formal methods: {missing}")
    if "sar_cam" in methods and not args.sar_cam_root:
        raise ValueError("--sar-cam-root is required when sar_cam is exported")

    gate = verify_pretest_gate(run_root)
    validation_selection = run_root / "records" / "validation_selection.csv"
    validation_rows = read_csv_records(validation_selection)
    strongest_rows = [
        row
        for row in validation_rows
        if row.get("selection_role", "").strip() == "strongest_non_ours_baseline"
    ]
    # The gate has already checked uniqueness and syntax; bind the declared
    # identity/hash to the checkpoint that will actually produce the panel.
    strongest = strongest_rows[0]
    strongest_key = strongest["method"].strip().lower().replace("-", "_")
    if strongest_key not in methods:
        raise ValueError(
            "The predeclared strongest non-ours baseline must be included in figure export: "
            f"{strongest_key}"
        )
    if sha256_file(methods[strongest_key]) != strongest["checkpoint_sha256"].strip().lower():
        raise ValueError(
            "The provided strongest-baseline checkpoint does not match "
            "validation_selection.csv"
        )
    _, manifest_path, manifest_rows = _load_and_validate_prepared_data(prep_root)
    if sha256_file(manifest_path) != gate.get("ucm_test_manifest_sha256"):
        raise ValueError("PREP_ROOT UCM manifest differs from the sealed pretest binding")
    if sha256_file(manifest_path) != gate["ucm_test_manifest_sha256"]:
        raise ValueError("Figure PREP_ROOT does not match the pretest-seal UCM manifest")
    figure_selection = run_root / "records" / "figure_selection.csv"
    figure_rows = read_csv_records(figure_selection)
    matched = select_registered_ucm_pairs(figure_rows, manifest_rows)
    pair_rows = [pair for _, pair in matched]
    selected_dataset = ICSPSFixedPairDataset(
        prep_root,
        pair_rows,
        protocol_id=PROTOCOL_ID,
        verify_hashes=True,
    )
    selected: list[dict[str, Any]] = []
    for index, (registration, pair) in enumerate(matched, start=1):
        sample = selected_dataset[index - 1]
        clean = sample["clean"].numpy()[0]
        noisy = sample["noisy"].numpy()[0]
        if clean.shape != (256, 256) or noisy.shape != (256, 256):
            raise ValueError(f"Registered pair {pair['pair_id']} is not 256x256")
        crop = validate_crop(clean, registration)
        selected.append(
            {
                "index": index,
                "registration": registration,
                "pair": pair,
                "crop": crop,
                "arrays": {
                    "clean": np.ascontiguousarray(clean, dtype=np.float32),
                    "noisy": np.ascontiguousarray(noisy, dtype=np.float32),
                },
            }
        )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested for figure reconstruction but CUDA is unavailable")
    seed_everything(42)
    checkpoint_provenance: dict[str, Any] = {}
    for method_key, checkpoint_path in args.method:
        model, provenance = _checkpoint_model(
            method_key, checkpoint_path, device, args.sar_cam_root
        )
        checkpoint_provenance[method_key] = provenance
        for sample in selected:
            tensor = torch.from_numpy(sample["arrays"]["noisy"])[None, None].to(device)
            with torch.inference_mode():
                prediction_tensor = model(tensor)
            if prediction_tensor.shape != tensor.shape or not bool(
                torch.isfinite(prediction_tensor).all().item()
            ):
                raise ValueError(
                    f"{method_key} produced invalid output for pair_id={sample['pair']['pair_id']}"
                )
            prediction = prediction_tensor.detach().cpu().numpy()[0, 0].astype(
                np.float32, copy=False
            )
            sample["arrays"][method_key] = np.ascontiguousarray(prediction)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    sarbm3d_provenance = None
    if args.sarbm3d_root:
        sarbm3d_predictions, sarbm3d_provenance = _load_sarbm3d_selected(
            Path(args.sarbm3d_root).expanduser().resolve(), pair_rows
        )
        for sample in selected:
            sample["arrays"]["sar_bm3d"] = sarbm3d_predictions[
                str(sample["pair"]["pair_id"])
            ]

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)
    )
    try:
        manifest_output_rows: list[dict[str, Any]] = []
        panels: list[dict[str, str]] = []
        ordered_series = [
            "clean",
            "noisy",
            *[key for key, _ in args.method],
            *(["sar_bm3d"] if args.sarbm3d_root else []),
        ]
        for sample in selected:
            pair = sample["pair"]
            registration = sample["registration"]
            x, y, width, height = sample["crop"]
            sample_name = (
                f"sample_{sample['index']:02d}_"
                f"{_safe_token(pair['pair_id'], 'pair')}_L{pair['global_L']}"
            )
            sample_dir = staging / sample_name
            sample_dir.mkdir()
            arrays = {key: sample["arrays"][key] for key in ordered_series}
            clean_display = np.clip(arrays["clean"], 0.0, 1.0)
            for key, raw_array in arrays.items():
                method_dir = sample_dir / key
                method_dir.mkdir()
                full_path = method_dir / "full_raw.npy"
                crop_path = method_dir / "crop_gray_0_1.png"
                error_path = method_dir / "crop_abs_error_0_1.png"
                error_array_path = method_dir / "crop_abs_error.npy"
                np.save(full_path, np.asarray(raw_array, dtype=np.float32), allow_pickle=False)
                clipped = np.clip(raw_array, 0.0, 1.0)
                crop_array = clipped[y : y + height, x : x + width]
                error_array = np.abs(
                    crop_array - clean_display[y : y + height, x : x + width]
                ).astype(np.float32, copy=False)
                np.save(error_array_path, error_array, allow_pickle=False)
                _save_gray_png(crop_array, crop_path)
                _save_error_png(error_array, error_path)
                identity = checkpoint_provenance.get(key, {})
                recorded_method = identity.get("method", key)
                recorded_variant = identity.get("variant") or ""
                if key == "sar_bm3d":
                    recorded_method, recorded_variant = "sar_bm3d", "v1.0"
                manifest_output_rows.append(
                    {
                        "protocol_id": PROTOCOL_ID,
                        "registration_row": sample["index"],
                        "figure": registration.get("figure", ""),
                        "panel": registration.get("panel", ""),
                        "pair_id": pair["pair_id"],
                        "source_id": pair["source_id"],
                        "class_name": pair["class_name"],
                        "L": pair["global_L"],
                        "crop_x": x,
                        "crop_y": y,
                        "crop_width": width,
                        "crop_height": height,
                        "selection_rule": registration.get("selection_rule", ""),
                        "series_key": key,
                        "display_label": DISPLAY_LABELS.get(key, key),
                        "method": recorded_method,
                        "variant": recorded_variant,
                        "seed": identity.get("seed", ""),
                        "checkpoint_sha256": identity.get("checkpoint_sha256", ""),
                        "fixed_pair_mat_sha256": pair["mat_sha256"],
                        "full_raw_npy": str(full_path.relative_to(staging)),
                        "full_raw_npy_sha256": sha256_file(full_path),
                        "crop_gray_png": str(crop_path.relative_to(staging)),
                        "crop_gray_png_sha256": sha256_file(crop_path),
                        "crop_abs_error_npy": str(error_array_path.relative_to(staging)),
                        "crop_abs_error_npy_sha256": sha256_file(error_array_path),
                        "crop_abs_error_png": str(error_path.relative_to(staging)),
                        "crop_abs_error_png_sha256": sha256_file(error_path),
                        "raw_min": float(np.min(raw_array)),
                        "raw_max": float(np.max(raw_array)),
                        "postclip_fraction": float(
                            np.mean((raw_array < 0.0) | (raw_array > 1.0))
                        ),
                    }
                )
            panel_png = sample_dir / "comparison_panel.png"
            panel_pdf = sample_dir / "comparison_panel.pdf"
            panel_title = (
                f"{pair['class_name']} | source={pair['source_id']} | "
                f"L={pair['global_L']} | pre-registered crop"
            )
            _save_panel(arrays, sample["crop"], panel_title, panel_png, panel_pdf)
            panels.append(
                {
                    "pair_id": str(pair["pair_id"]),
                    "png": str(panel_png.relative_to(staging)),
                    "png_sha256": sha256_file(panel_png),
                    "pdf": str(panel_pdf.relative_to(staging)),
                    "pdf_sha256": sha256_file(panel_pdf),
                }
            )

        manifest_output = staging / "figure_manifest.csv"
        with manifest_output.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(manifest_output_rows[0]))
            writer.writeheader()
            writer.writerows(manifest_output_rows)
        provenance = {
            "protocol_id": PROTOCOL_ID,
            "formal_run": True,
            "selection_policy": "pre-registered rows only; no result-based selection",
            "pretest_gate": gate,
            "run_root": str(run_root),
            "prep_root": str(prep_root),
            "prepared_config": str(prep_root / "config_snapshot.json"),
            "prepared_config_sha256": sha256_file(prep_root / "config_snapshot.json"),
            "ucm_manifest": str(manifest_path),
            "ucm_manifest_sha256": sha256_file(manifest_path),
            "figure_selection": str(figure_selection),
            "figure_selection_sha256": sha256_file(figure_selection),
            "validation_selection": str(validation_selection),
            "validation_selection_sha256": sha256_file(validation_selection),
            "strongest_non_ours_baseline_key": strongest_key,
            "required_registered_looks": [1, 4],
            "selected_pairs": len(selected),
            "method_order": ordered_series,
            "learned_checkpoints": checkpoint_provenance,
            "sarbm3d": sarbm3d_provenance,
            "display": {
                "grayscale_range": [0.0, 1.0],
                "grayscale_png_encoding": "uint16 fixed linear mapping",
                "absolute_error_range": [0.0, 1.0],
                "absolute_error_colormap": "magma",
                "model_display_operation": "clip raw prediction to [0,1]",
            },
            "source_code": str(Path(__file__).resolve()),
            "source_code_sha256": sha256_file(__file__),
            "panels": panels,
        }
        provenance_path = staging / "provenance.json"
        atomic_json_dump(provenance, provenance_path)
        completion = {
            "protocol_id": PROTOCOL_ID,
            "formal_run": True,
            "state": "complete",
            "selected_pairs": len(selected),
            "exported_series": len(ordered_series),
            "figure_manifest": manifest_output.name,
            "figure_manifest_sha256": sha256_file(manifest_output),
            "provenance": provenance_path.name,
            "provenance_sha256": sha256_file(provenance_path),
        }
        atomic_json_dump(completion, staging / "completion.json")
        _publish_directory(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(completion, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
