"""Prepare an immutable QC/ROI manifest for ICSPS 2026 real-SAR evaluation.

The script never moves or rewrites source imagery.  It freezes one dataset-level
numeric mapping and noisy-only ENL ROI coordinates for reuse by every method.
ICSPS26-FROZEN-v2 explicitly disables GT16.  The explicit ``--gt16-root``
compatibility path exists only for nonformal or future separately-versioned
protocol work; GT16 is never auto-discovered and is never exact ground truth.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from scipy.io import loadmat

from icsps2026_metrics import select_homogeneous_rois


SUPPORTED_EXTENSIONS = {".mat", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
SPLITS = ("train", "val", "test")
PARENT_PATTERN = re.compile(
    r"^(?P<parent_row>-?\d+)_(?P<parent_col>-?\d+)_y-?\d+_x-?\d+$",
    re.IGNORECASE,
)
PAIR_TOKEN_PATTERN = re.compile(
    r"(?:^|[_\-.])(noisy|gt[_\-.]?16|ground[_\-.]?truth[_\-.]?16)(?:$|[_\-.])",
    re.IGNORECASE,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _shape_text(shape: tuple[int, ...]) -> str:
    return "x".join(str(value) for value in shape)


def _single_channel(array: np.ndarray, path: Path) -> np.ndarray:
    image = np.asarray(array)
    while image.ndim > 2 and 1 in image.shape:
        image = np.squeeze(image, axis=list(image.shape).index(1))
    if image.ndim == 3 and image.shape[-1] in (3, 4):
        rgb = image[..., :3]
        if not (
            np.array_equal(rgb[..., 0], rgb[..., 1])
            and np.array_equal(rgb[..., 0], rgb[..., 2])
        ):
            raise ValueError(f"Real SAR input is not single-channel: {path} {image.shape}")
        image = rgb[..., 0]
    if image.ndim != 2:
        raise ValueError(f"Expected a 2-D real-SAR array in {path}, got {image.shape}")
    return image


def load_real_array(path: Path, role: str) -> tuple[np.ndarray, str]:
    """Load a real-SAR array without applying any per-image normalization."""

    if role not in {"noisy", "gt16"}:
        raise ValueError("role must be noisy or gt16")
    if path.suffix.lower() == ".mat":
        payload = loadmat(path)
        preferred = (
            ("noisy", "sar", "image", "img", "data", "input")
            if role == "noisy"
            else ("gt16", "gt_16", "groundtruth16", "ground_truth_16", "reference", "image", "img", "data")
        )
        keys_by_lower = {key.lower(): key for key in payload if not key.startswith("__")}
        for candidate in preferred:
            key = keys_by_lower.get(candidate)
            if key is not None:
                return _single_channel(payload[key], path), key
        candidate_arrays = [
            (key, value)
            for key, value in payload.items()
            if not key.startswith("__") and np.asarray(value).ndim in (2, 3)
        ]
        if len(candidate_arrays) == 1:
            key, value = candidate_arrays[0]
            return _single_channel(value, path), key
        raise KeyError(
            f"No unambiguous {role} array in {path}; fields="
            f"{sorted(key for key in payload if not key.startswith('__'))}"
        )

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return _single_channel(image, path), "image_file"


def canonical_pair_key(relative_path: str | Path) -> str:
    """Create a conservative, case-insensitive key for mechanical pairing."""

    path = Path(relative_path)
    parts = [part.lower() for part in path.with_suffix("").parts]
    parts = [part for part in parts if part not in {"noisy", "gt16", "gt_16"}]
    if not parts:
        raise ValueError(f"Cannot derive pair key from {relative_path}")
    stem = PAIR_TOKEN_PATTERN.sub("_", parts[-1]).strip("_.-")
    parts[-1] = stem
    return "/".join(parts)


def parent_id_from_sample_name(name: str) -> str:
    stem = Path(name).stem
    match = PARENT_PATTERN.match(stem)
    if match:
        return f"{match.group('parent_row')}_{match.group('parent_col')}"
    return stem


def apply_fixed_mapping(array: np.ndarray, mapping: dict, tolerance: float = 1e-6) -> np.ndarray:
    """Apply a dataset-level affine mapping, rejecting material out-of-range data."""

    image = np.asarray(array, dtype=np.float32)
    if not np.isfinite(image).all():
        raise ValueError("Cannot map an array containing NaN or Inf")
    offset = float(mapping["offset"])
    scale = float(mapping["scale"])
    if scale <= 0:
        raise ValueError("Mapping scale must be positive")
    mapped = (image - offset) / scale
    low = float(np.min(mapped))
    high = float(np.max(mapped))
    if low < -tolerance or high > 1.0 + tolerance:
        raise ValueError(
            f"Dataset mapping {mapping['mode']} produces range [{low}, {high}], outside [0,1]"
        )
    return np.clip(mapped, 0.0, 1.0).astype(np.float32)


def choose_dataset_mapping(
    noisy_range: tuple[float, float],
    gt16_range: tuple[float, float] | None,
    mode: str,
    linear_offset: float,
    linear_scale: float,
) -> dict:
    """Choose one mapping for every Noisy and GT16 array in the dataset."""

    role_ranges = [noisy_range] + ([gt16_range] if gt16_range is not None else [])
    if mode == "auto" and gt16_range is not None:
        noisy_unit = noisy_range[0] >= -1e-6 and noisy_range[1] <= 1.0 + 1e-6
        gt_unit = gt16_range[0] >= -1e-6 and gt16_range[1] <= 1.0 + 1e-6
        if noisy_unit != gt_unit:
            raise ValueError(
                "Noisy and GT16 appear to use different numeric scales; choose an explicit "
                "dataset-level mapping after auditing the source metadata"
            )
    global_low = min(item[0] for item in role_ranges)
    global_high = max(item[1] for item in role_ranges)
    if mode == "auto":
        if global_low >= -1e-6 and global_high <= 1.0 + 1e-6:
            mode, offset, scale = "unit_float", 0.0, 1.0
        elif global_low >= -1e-6 and global_high <= 255.0 + 1e-6:
            mode, offset, scale = "uint8_255", 0.0, 255.0
        else:
            raise ValueError(
                f"Raw dataset range [{global_low}, {global_high}] is not an unambiguous "
                "[0,1] or [0,255] representation; use --mapping-mode linear explicitly"
            )
    elif mode == "unit_float":
        offset, scale = 0.0, 1.0
    elif mode == "uint8_255":
        offset, scale = 0.0, 255.0
    elif mode == "linear":
        offset, scale = float(linear_offset), float(linear_scale)
    else:
        raise ValueError(f"Unsupported mapping mode: {mode}")
    mapping = {
        "mode": mode,
        "offset": offset,
        "scale": scale,
        "formula": "mapped=(raw-offset)/scale",
        "raw_global_min": global_low,
        "raw_global_max": global_high,
        "per_image_percentile_normalization": False,
    }
    # Validate the global extrema before touching individual samples.
    apply_fixed_mapping(np.asarray([[global_low, global_high]], dtype=np.float64), mapping)
    return mapping


def alignment_audit(noisy: np.ndarray, gt16: np.ndarray) -> dict[str, float]:
    """Return zero-lag smoothed correlation and phase-correlation displacement."""

    first = np.asarray(noisy, dtype=np.float32)
    second = np.asarray(gt16, dtype=np.float32)
    if first.shape != second.shape:
        raise ValueError(f"Alignment inputs differ in shape: {first.shape} != {second.shape}")
    first_smooth = cv2.GaussianBlur(first, (0, 0), sigmaX=1.5, sigmaY=1.5)
    second_smooth = cv2.GaussianBlur(second, (0, 0), sigmaX=1.5, sigmaY=1.5)
    a = first_smooth.astype(np.float64) - float(np.mean(first_smooth))
    b = second_smooth.astype(np.float64) - float(np.mean(second_smooth))
    denominator = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
    zero_lag = float(np.sum(a * b) / denominator) if denominator > 1e-12 else float("nan")

    height, width = first.shape
    window = cv2.createHanningWindow((width, height), cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(first_smooth, second_smooth, window)
    return {
        "alignment_score": float(np.clip(zero_lag, -1.0, 1.0)),
        "phase_shift_x": float(dx),
        "phase_shift_y": float(dy),
        "phase_response": float(response),
    }


def _collect_supported(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def _find_named_directory(root: Path, candidates: Iterable[str]) -> Path | None:
    children = {path.name.lower(): path for path in root.iterdir() if path.is_dir()}
    for candidate in candidates:
        if candidate.lower() in children:
            return children[candidate.lower()]
    return None


def _relative_inside(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"Source path must be inside dataset root: {path}") from error


def _records_from_split_manifest(dataset_root: Path, noisy_root: Path, manifest_path: Path) -> list[dict]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    files = manifest.get("files")
    if not isinstance(files, dict) or any(split not in files for split in SPLITS):
        raise ValueError("Split manifest must contain train/val/test file lists")
    records = []
    seen = set()
    for split in SPLITS:
        for relative in files[split]:
            relative_path = Path(relative)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError(f"Unsafe split-manifest path: {relative}")
            candidates = [dataset_root / relative_path, noisy_root / relative_path]
            path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
            resolved = str(path.resolve())
            if resolved in seen:
                raise ValueError(f"Noisy file is listed more than once: {path}")
            seen.add(resolved)
            records.append({"split": split, "noisy_path": path})
    return records


def _records_from_directory(dataset_root: Path, noisy_root: Path, default_split: str) -> list[dict]:
    records = []
    for path in _collect_supported(noisy_root):
        relative = path.relative_to(noisy_root)
        inferred = relative.parts[0].lower() if len(relative.parts) > 1 else ""
        split = inferred if inferred in SPLITS else default_split
        records.append({"split": split, "noisy_path": path})
    return records


def _build_gt_indices(gt16_root: Path | None) -> tuple[dict[str, Path], dict[str, list[Path]]]:
    if gt16_root is None:
        return {}, {}
    exact: dict[str, Path] = {}
    basenames: dict[str, list[Path]] = defaultdict(list)
    for path in _collect_supported(gt16_root):
        relative = path.relative_to(gt16_root)
        key = canonical_pair_key(relative)
        if key in exact:
            raise ValueError(f"Duplicate GT16 pair key {key!r}: {exact[key]} and {path}")
        exact[key] = path
        basenames[Path(key).name].append(path)
    return exact, dict(basenames)


def _match_gt(
    noisy_path: Path,
    noisy_root: Path,
    exact: dict[str, Path],
    basenames: dict[str, list[Path]],
) -> tuple[Path | None, str]:
    try:
        relative = noisy_path.relative_to(noisy_root)
    except ValueError:
        relative = Path(noisy_path.name)
    key = canonical_pair_key(relative)
    if key in exact:
        return exact[key], "relative_key"
    matches = basenames.get(Path(key).name, [])
    if len(matches) == 1:
        return matches[0], "unique_basename_key"
    if len(matches) > 1:
        return None, "ambiguous_basename_key"
    return None, "missing"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-manifest")
    parser.add_argument("--noisy-root")
    parser.add_argument("--gt16-root")
    parser.add_argument(
        "--disable-gt16",
        action="store_true",
        help=(
            "Do not auto-discover or use a GT16 directory. This is the frozen "
            "ICSPS 2026 no-GT16 branch; q-reference metrics remain unavailable."
        ),
    )
    parser.add_argument(
        "--default-split", choices=("unassigned",) + SPLITS, default="unassigned",
        help="Used only when no split manifest/path component supplies a split",
    )
    parser.add_argument(
        "--mapping-mode", choices=("auto", "unit_float", "uint8_255", "linear"), default="auto"
    )
    parser.add_argument("--linear-offset", type=float, default=0.0)
    parser.add_argument("--linear-scale", type=float, default=1.0)
    parser.add_argument("--expected-height", type=int, default=256)
    parser.add_argument("--expected-width", type=int, default=256)
    parser.add_argument("--alignment-score-min", type=float, default=0.5)
    parser.add_argument("--alignment-max-shift", type=float, default=1.0)
    parser.add_argument("--roi-size", type=int, default=32)
    parser.add_argument("--num-rois", type=int, default=5)
    parser.add_argument("--roi-stride", type=int, default=32)
    parser.add_argument("--roi-min-mean", type=float, default=0.03)
    parser.add_argument("--roi-max-clipped-fraction", type=float, default=0.05)
    return parser.parse_args()


def _metadata(path: Path, role: str) -> dict:
    array, field = load_real_array(path, role)
    finite = bool(np.isfinite(array).all())
    finite_values = array[np.isfinite(array)]
    minimum = float(np.min(finite_values)) if finite_values.size else float("nan")
    maximum = float(np.max(finite_values)) if finite_values.size else float("nan")
    return {
        "array": array,
        "field": field,
        "shape": tuple(array.shape),
        "dtype": str(array.dtype),
        "minimum": minimum,
        "maximum": maximum,
        "finite": finite,
        "constant": bool(finite and maximum <= minimum),
    }


def _range_from_metadata(items: list[dict]) -> tuple[float, float]:
    finite = [item for item in items if item.get("finite")]
    if not finite:
        raise ValueError("No finite arrays are available to determine dataset mapping")
    return min(item["minimum"] for item in finite), max(item["maximum"] for item in finite)


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")

    noisy_root = (
        Path(args.noisy_root).resolve()
        if args.noisy_root
        else (_find_named_directory(dataset_root, ("Noisy", "noisy")) or dataset_root)
    )
    if not noisy_root.is_dir():
        raise FileNotFoundError(noisy_root)
    if args.disable_gt16 and args.gt16_root:
        raise ValueError("--disable-gt16 and --gt16-root are mutually exclusive")
    if args.disable_gt16:
        gt16_root = None
    elif args.gt16_root:
        gt16_root = Path(args.gt16_root).resolve()
    else:
        # Fail-safe default: GT16 is never discovered implicitly.  A future,
        # separately versioned protocol must opt in with an explicit path.
        gt16_root = None
    if gt16_root is not None and not gt16_root.is_dir():
        raise FileNotFoundError(gt16_root)

    if args.split_manifest:
        split_manifest = Path(args.split_manifest).resolve()
        records = _records_from_split_manifest(dataset_root, noisy_root, split_manifest)
    else:
        split_manifest = None
        records = _records_from_directory(dataset_root, noisy_root, args.default_split)
    if not records:
        raise RuntimeError(f"No supported real-SAR files found below {noisy_root}")

    gt_exact, gt_basenames = _build_gt_indices(gt16_root)
    noisy_metadata = []
    gt_metadata_by_path: dict[str, dict] = {}
    for record in records:
        path = record["noisy_path"]
        try:
            metadata = _metadata(path, "noisy")
            metadata["load_error"] = ""
        except Exception as error:  # QC must record bad samples instead of aborting.
            metadata = {"load_error": f"{type(error).__name__}: {error}", "finite": False}
        noisy_metadata.append(metadata)
        gt_path, pair_rule = _match_gt(path, noisy_root, gt_exact, gt_basenames)
        record["gt16_path"] = gt_path
        record["pair_rule"] = pair_rule
        if gt_path is not None and str(gt_path) not in gt_metadata_by_path:
            try:
                gt_item = _metadata(gt_path, "gt16")
                gt_item["load_error"] = ""
            except Exception as error:
                gt_item = {"load_error": f"{type(error).__name__}: {error}", "finite": False}
            gt_metadata_by_path[str(gt_path)] = gt_item

    # A quasi-reference may not be silently reused for multiple noisy samples.
    matched_gt_counts = Counter(
        str(record["gt16_path"])
        for record in records
        if record.get("gt16_path") is not None
    )
    for record in records:
        gt_path = record.get("gt16_path")
        if gt_path is not None and matched_gt_counts[str(gt_path)] != 1:
            record["gt16_path"] = None
            record["pair_rule"] = "gt16_reused_by_multiple_noisy"

    usable_noisy = [item for item in noisy_metadata if not item.get("load_error")]
    usable_gt = [item for item in gt_metadata_by_path.values() if not item.get("load_error")]
    noisy_range = _range_from_metadata(usable_noisy)
    gt_range = _range_from_metadata(usable_gt) if usable_gt else None
    mapping = choose_dataset_mapping(
        noisy_range,
        gt_range,
        args.mapping_mode,
        args.linear_offset,
        args.linear_scale,
    )
    mapping["noisy_raw_min"] = noisy_range[0]
    mapping["noisy_raw_max"] = noisy_range[1]
    mapping["gt16_raw_min"] = gt_range[0] if gt_range else None
    mapping["gt16_raw_max"] = gt_range[1] if gt_range else None

    qc_rows = []
    roi_rows = []
    sample_ids = set()
    for record, noisy_meta in zip(records, noisy_metadata):
        noisy_path = record["noisy_path"]
        noisy_relative = _relative_inside(noisy_path, dataset_root)
        sample_key = canonical_pair_key(
            noisy_path.relative_to(noisy_root) if noisy_path.is_relative_to(noisy_root) else noisy_path.name
        )
        sample_id = sample_key.replace("/", "__")
        if sample_id in sample_ids:
            raise ValueError(f"Duplicate sample_id generated: {sample_id}")
        sample_ids.add(sample_id)
        parent_id = parent_id_from_sample_name(noisy_path.name)
        gt_path = record["gt16_path"]
        gt_meta = gt_metadata_by_path.get(str(gt_path), {}) if gt_path is not None else {}

        no_ref_reasons = []
        qref_reasons = []
        mapped_noisy = None
        mapped_gt = None
        if noisy_meta.get("load_error"):
            no_ref_reasons.append("noisy_load_error")
        else:
            if not noisy_meta["finite"]:
                no_ref_reasons.append("noisy_nonfinite")
            if noisy_meta["constant"]:
                no_ref_reasons.append("noisy_constant")
            if args.expected_height > 0 and noisy_meta["shape"][0] != args.expected_height:
                no_ref_reasons.append("unexpected_height")
            if args.expected_width > 0 and noisy_meta["shape"][1] != args.expected_width:
                no_ref_reasons.append("unexpected_width")
            try:
                mapped_noisy = apply_fixed_mapping(noisy_meta["array"], mapping)
            except ValueError:
                no_ref_reasons.append("noisy_mapping_out_of_range")

        pair_exists = gt_path is not None
        alignment = {
            "alignment_score": float("nan"),
            "phase_shift_x": float("nan"),
            "phase_shift_y": float("nan"),
            "phase_response": float("nan"),
        }
        if not pair_exists:
            qref_reasons.append(record["pair_rule"])
        elif gt_meta.get("load_error"):
            qref_reasons.append("gt16_load_error")
        else:
            if not gt_meta["finite"]:
                qref_reasons.append("gt16_nonfinite")
            if gt_meta["constant"]:
                qref_reasons.append("gt16_constant")
            if noisy_meta.get("shape") != gt_meta.get("shape"):
                qref_reasons.append("shape_mismatch")
            try:
                mapped_gt = apply_fixed_mapping(gt_meta["array"], mapping)
            except ValueError:
                qref_reasons.append("gt16_mapping_out_of_range")
            if mapped_noisy is not None and mapped_gt is not None and not qref_reasons:
                alignment = alignment_audit(mapped_noisy, mapped_gt)
                if not np.isfinite(alignment["alignment_score"]):
                    qref_reasons.append("alignment_score_nonfinite")
                elif alignment["alignment_score"] < args.alignment_score_min:
                    qref_reasons.append("alignment_score_below_threshold")
                if max(abs(alignment["phase_shift_x"]), abs(alignment["phase_shift_y"])) > args.alignment_max_shift:
                    qref_reasons.append("alignment_shift_above_threshold")

        valid_no_ref = not no_ref_reasons
        if not valid_no_ref:
            qref_reasons.append("noisy_invalid")
        valid_qref = valid_no_ref and pair_exists and not qref_reasons
        valid_enl_rois = False
        enl_roi_count = 0
        roi_exclusion_reason = ""
        if valid_no_ref and mapped_noisy is not None:
            selected = select_homogeneous_rois(
                mapped_noisy,
                roi_size=args.roi_size,
                num_rois=args.num_rois,
                stride=args.roi_stride,
                min_mean=args.roi_min_mean,
                max_clipped_fraction=args.roi_max_clipped_fraction,
            )
            enl_roi_count = len(selected)
            valid_enl_rois = enl_roi_count == args.num_rois
            if len(selected) < args.num_rois:
                roi_exclusion_reason = "insufficient_enl_rois"
            for roi in selected:
                roi_rows.append(
                    {
                        "sample_id": sample_id,
                        "parent_id": parent_id,
                        "split": record["split"],
                        **roi,
                    }
                )

        qc_rows.append(
            {
                "sample_id": sample_id,
                "parent_id": parent_id,
                "split": record["split"],
                "category": "",
                "noisy_path": noisy_relative,
                "gt16_path": _relative_inside(gt_path, dataset_root) if gt_path else "",
                "pair_rule": record["pair_rule"],
                "noisy_field": noisy_meta.get("field", ""),
                "gt16_field": gt_meta.get("field", ""),
                "noisy_shape": _shape_text(noisy_meta.get("shape", ())),
                "gt16_shape": _shape_text(gt_meta.get("shape", ())),
                "noisy_dtype": noisy_meta.get("dtype", ""),
                "gt16_dtype": gt_meta.get("dtype", ""),
                "noisy_min": noisy_meta.get("minimum", ""),
                "noisy_max": noisy_meta.get("maximum", ""),
                "gt16_min": gt_meta.get("minimum", ""),
                "gt16_max": gt_meta.get("maximum", ""),
                "is_finite": bool(noisy_meta.get("finite", False)),
                "is_constant": bool(noisy_meta.get("constant", False)),
                "pair_exists": pair_exists,
                "shape_match": bool(pair_exists and noisy_meta.get("shape") == gt_meta.get("shape")),
                **alignment,
                "alignment_pass": valid_qref,
                "valid_no_reference": valid_no_ref,
                "valid_quasi_reference": valid_qref,
                "valid_enl_rois": valid_enl_rois,
                "enl_roi_count": enl_roi_count,
                "exclusion_reason": ";".join(no_ref_reasons),
                "qref_exclusion_reason": ";".join(qref_reasons),
                "roi_exclusion_reason": roi_exclusion_reason,
                "noisy_sha256": sha256_file(noisy_path),
                "gt16_sha256": sha256_file(gt_path) if gt_path else "",
            }
        )

    if not roi_rows:
        raise RuntimeError("No fixed ENL ROIs were selected")
    output_dir.mkdir(parents=True, exist_ok=True)
    qc_csv = output_dir / "real_qc_manifest.csv"
    roi_csv = output_dir / "real_enl_roi_manifest.csv"
    alignment_csv = output_dir / "real_alignment_audit.csv"
    with qc_csv.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(qc_rows[0]))
        writer.writeheader()
        writer.writerows(qc_rows)
    with roi_csv.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(roi_rows[0]))
        writer.writeheader()
        writer.writerows(roi_rows)
    alignment_fields = [
        "sample_id", "parent_id", "split", "noisy_path", "gt16_path", "pair_rule",
        "pair_exists", "shape_match", "alignment_score", "phase_shift_x", "phase_shift_y",
        "phase_response", "alignment_pass", "qref_exclusion_reason",
    ]
    with alignment_csv.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=alignment_fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in alignment_fields} for row in qc_rows)

    split_counts = {}
    for split in ("train", "val", "test", "unassigned"):
        subset = [row for row in qc_rows if row["split"] == split]
        if subset:
            split_counts[split] = {
                "assigned": len(subset),
                "valid_no_reference": sum(bool(row["valid_no_reference"]) for row in subset),
                "paired_gt16": sum(bool(row["pair_exists"]) for row in subset),
                "valid_quasi_reference": sum(bool(row["valid_quasi_reference"]) for row in subset),
                "valid_enl_rois": sum(bool(row["valid_enl_rois"]) for row in subset),
                "parent_clusters": len({row["parent_id"] for row in subset}),
            }
    matched_gt_paths = {
        str(record["gt16_path"].resolve())
        for record in records
        if record.get("gt16_path") is not None
    }
    all_gt_paths = {str(path.resolve()) for path in gt_exact.values()}
    unmatched_gt_paths = sorted(all_gt_paths - matched_gt_paths)
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root_name": dataset_root.name,
        "noisy_root": _relative_inside(noisy_root, dataset_root) if noisy_root != dataset_root else ".",
        "gt16_root": (
            _relative_inside(gt16_root, dataset_root) if gt16_root is not None else None
        ),
        "gt16_disabled_by_protocol": bool(args.disable_gt16),
        "split_manifest": str(split_manifest) if split_manifest else None,
        "split_manifest_sha256": sha256_file(split_manifest) if split_manifest else None,
        "numeric_mapping": mapping,
        "expected_shape": [args.expected_height, args.expected_width],
        "alignment_policy": {
            "score": "zero-lag Pearson correlation after Gaussian sigma=1.5 smoothing",
            "phase_method": "OpenCV phaseCorrelate with a 2-D Hann window",
            "minimum_score": args.alignment_score_min,
            "maximum_absolute_shift_pixels": args.alignment_max_shift,
            "quasi_reference_only": True,
        },
        "enl_roi_policy": {
            "selection_input": "noisy only",
            "score": "variance / mean^2 (squared coefficient of variation)",
            "roi_size": args.roi_size,
            "num_rois": args.num_rois,
            "stride": args.roi_stride,
            "minimum_mean": args.roi_min_mean,
            "maximum_clipped_fraction": args.roi_max_clipped_fraction,
            "semantic_homogeneity_requires_human_review": True,
        },
        "counts": {
            "total_samples": len(qc_rows),
            "total_parent_clusters": len({row["parent_id"] for row in qc_rows}),
            "gt16_files_discovered": len(gt_exact),
            "gt16_files_matched_one_to_one": len(matched_gt_paths),
            "gt16_files_unmatched": len(unmatched_gt_paths),
            "unmatched_gt16_examples": [
                _relative_inside(Path(path), dataset_root) for path in unmatched_gt_paths[:20]
            ],
            "pair_rules": dict(Counter(row["pair_rule"] for row in qc_rows)),
            "by_split": split_counts,
        },
        "artifacts": {
            "qc_csv": qc_csv.name,
            "qc_csv_sha256": sha256_file(qc_csv),
            "roi_csv": roi_csv.name,
            "roi_csv_sha256": sha256_file(roi_csv),
            "alignment_csv": alignment_csv.name,
            "alignment_csv_sha256": sha256_file(alignment_csv),
        },
        "metric_scope": {
            "implemented_here": ["QC", "fixed ENL ROI coordinates", "alignment audit"],
            "not_claimed": ["validated M-index", "RGPI", "speckle-free ground truth"],
        },
    }
    manifest_path = output_dir / "real_qc_manifest.json"
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"manifest": str(manifest_path), **manifest["counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
