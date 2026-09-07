#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$project_root"

: "${REAL_ROOT:?Set REAL_ROOT to the Mendeley fs455tz88y.1 data root}"
: "${REAL_PROTOCOL_ROOT:?Set REAL_PROTOCOL_ROOT to a new QC output directory}"
: "${REAL_SPLIT_MANIFEST:?Set REAL_SPLIT_MANIFEST to real_split_grouped_seed42.json}"

real_use_gt16=${REAL_USE_GT16:-0}
if [[ "$real_use_gt16" != "0" ]]; then
  echo "ICSPS26-FROZEN-v2 only permits REAL_USE_GT16=0; GT16 requires a new protocol and runner." >&2
  exit 1
fi
if [[ "$real_use_gt16" == "0" && -n "${REAL_GT16_ROOT:-}" ]]; then
  echo "REAL_GT16_ROOT is set while REAL_USE_GT16=0; unset it for the frozen no-GT16 branch." >&2
  exit 1
fi

if [[ ! -d "$REAL_ROOT" ]]; then
  echo "Real-SAR dataset root does not exist: $REAL_ROOT" >&2
  exit 1
fi
if [[ ! -f "$REAL_SPLIT_MANIFEST" ]]; then
  echo "Frozen real split manifest does not exist: $REAL_SPLIT_MANIFEST" >&2
  exit 1
fi
if [[ -f "$REAL_PROTOCOL_ROOT/real_qc_manifest.json" ]]; then
  python - \
    "$REAL_PROTOCOL_ROOT/real_qc_manifest.json" \
    "$REAL_SPLIT_MANIFEST" \
    "$REAL_ROOT" \
    "${REAL_NOISY_ROOT:-}" \
    "${REAL_GT16_ROOT:-}" \
    "$real_use_gt16" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1]).resolve()
split_path = Path(sys.argv[2]).resolve()
dataset_root = Path(sys.argv[3]).resolve()
explicit_noisy = Path(sys.argv[4]).resolve() if sys.argv[4] else None
explicit_gt16 = Path(sys.argv[5]).resolve() if sys.argv[5] else None
use_gt16 = sys.argv[6] == "1"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()

def named_child(names):
    children = {path.name.lower(): path for path in dataset_root.iterdir() if path.is_dir()}
    return next((children[name.lower()] for name in names if name.lower() in children), None)

noisy_root = explicit_noisy or named_child(("Noisy", "noisy")) or dataset_root
gt16_root = (
    explicit_gt16 or named_child(("GT16", "gt16", "GT_16", "gt_16"))
) if use_gt16 else None
for label, path in (("dataset", dataset_root), ("noisy", noisy_root)):
    if not path.is_dir():
        raise SystemExit(f"Current {label} root does not exist: {path}")
if gt16_root is not None and not gt16_root.is_dir():
    raise SystemExit(f"Current GT16 root does not exist: {gt16_root}")

def relative(path):
    if path is None:
        return None
    try:
        value = path.relative_to(dataset_root).as_posix()
    except ValueError as error:
        raise SystemExit(f"Current real-data subroot escapes REAL_ROOT: {path}") from error
    return value or "."

expected_roots = {
    "dataset_root_name": dataset_root.name,
    "noisy_root": relative(noisy_root),
    "gt16_root": relative(gt16_root),
    "gt16_disabled_by_protocol": not use_gt16,
}
root_mismatches = {
    key: {"expected": value, "found": manifest.get(key)}
    for key, value in expected_roots.items()
    if manifest.get(key) != value
}
if root_mismatches:
    raise SystemExit(
        "Existing real QC artifacts were prepared from different Noisy/GT16 roots: "
        f"{root_mismatches}. Use a new REAL_PROTOCOL_ROOT and rerun QC; do not overwrite."
    )
if manifest.get("split_manifest_sha256") != digest(split_path):
    raise SystemExit("Existing real QC artifacts use a different split manifest")
for name_key, hash_key in (
    ("qc_csv", "qc_csv_sha256"),
    ("roi_csv", "roi_csv_sha256"),
    ("alignment_csv", "alignment_csv_sha256"),
):
    artifacts = manifest["artifacts"]
    artifact_path = manifest_path.parent / artifacts[name_key]
    if not artifact_path.is_file() or digest(artifact_path) != artifacts[hash_key]:
        raise SystemExit(f"Existing real QC artifact failed hash verification: {artifact_path}")
print(f"Existing real QC artifacts verified: {manifest_path.parent}")
PY
  exit 0
fi
if [[ -e "$REAL_PROTOCOL_ROOT" && ! -d "$REAL_PROTOCOL_ROOT" ]]; then
  echo "REAL_PROTOCOL_ROOT exists but is not a directory: $REAL_PROTOCOL_ROOT" >&2
  exit 1
fi
if [[ -d "$REAL_PROTOCOL_ROOT" && -n "$(find "$REAL_PROTOCOL_ROOT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Refusing to overwrite partial/non-empty real protocol root: $REAL_PROTOCOL_ROOT" >&2
  exit 1
fi

pair_args=()
if [[ -n "${REAL_NOISY_ROOT:-}" ]]; then
  pair_args+=(--noisy-root "$REAL_NOISY_ROOT")
fi
pair_args+=(--disable-gt16)

python prepare_icsps2026_real.py \
  --dataset-root "$REAL_ROOT" \
  --split-manifest "$REAL_SPLIT_MANIFEST" \
  --output-dir "$REAL_PROTOCOL_ROOT" \
  "${pair_args[@]}"

echo "Inspect these before real test unlock:"
echo "  $REAL_PROTOCOL_ROOT/real_qc_manifest.json"
echo "  $REAL_PROTOCOL_ROOT/real_alignment_audit.csv"
echo "  $REAL_PROTOCOL_ROOT/real_enl_roi_manifest.csv"
