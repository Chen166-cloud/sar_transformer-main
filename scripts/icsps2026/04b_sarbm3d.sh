#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$project_root"
source "$project_root/scripts/icsps2026/sarbm3d_runtime.sh"

: "${PREP_ROOT:?Set PREP_ROOT to the formal prepared-data directory}"
: "${RUN_ROOT:?Set RUN_ROOT to the formal experiment output root}"
: "${SARBM3D_ROOT:?Set SARBM3D_ROOT to the extracted official v1.0 package directory}"
: "${SARBM3D_ARCHIVE:?Set SARBM3D_ARCHIVE to the downloaded official tar.gz}"

python verify_icsps2026_pretest_gate.py --run-root "$RUN_ROOT"

if ! command -v matlab >/dev/null 2>&1; then
  echo "MATLAB is not available on PATH; SAR-BM3D cannot run." >&2
  exit 1
fi
for required_path in "$PREP_ROOT/config_snapshot.json" "$PREP_ROOT/ucm_test_manifest.csv" "$SARBM3D_ARCHIVE"; do
  if [[ ! -f "$required_path" ]]; then
    echo "Required file does not exist: $required_path" >&2
    exit 1
  fi
done
if [[ ! -d "$SARBM3D_ROOT" ]]; then
  echo "SAR-BM3D package directory does not exist: $SARBM3D_ROOT" >&2
  exit 1
fi
mkdir -p "$RUN_ROOT/logs"
sarbm3d_prepare_runtime \
  "$SARBM3D_ROOT" "$RUN_ROOT/logs/sarbm3d_runtime_probe.log"

result_dir="$RUN_ROOT/ucm/sar_bm3d_v1"
if [[ -e "$result_dir/aggregate.json" ]]; then
  python verify_icsps2026_artifact.py --kind ucm --path "$result_dir/aggregate.json" \
    --expected-method sar_bm3d --expected-variant v1.0 \
    --expected-seed "" --expected-adaptation ""
  echo "SAR-BM3D evaluation already completed: $result_dir"
  exit 0
fi
if [[ -d "$result_dir" && -n "$(find "$result_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Partial SAR-BM3D evaluation directory cannot be overwritten: $result_dir" >&2
  echo "Move it aside after inspection, then rerun; raw predictions remain restartable." >&2
  exit 1
fi

job_root="$RUN_ROOT/sarbm3d_jobs"
mkdir -p "$job_root"
jobs="$job_root/ucm_jobs.csv"
if [[ ! -f "$jobs" ]]; then
  python export_icsps2026_sarbm3d_jobs.py \
    --fixed-pair-root "$PREP_ROOT" \
    --manifest "$PREP_ROOT/ucm_test_manifest.csv" \
    --run-root "$RUN_ROOT" \
    --output "$jobs"
fi

python - "$jobs.json" "$PREP_ROOT/ucm_test_manifest.csv" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

metadata_path, manifest_path = map(Path, sys.argv[1:])
if not metadata_path.is_file():
    raise SystemExit(f"Missing SAR-BM3D job metadata: {metadata_path}")
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
if metadata.get("pair_manifest_sha256") != digest or metadata.get("jobs") != 8400:
    raise SystemExit("Existing SAR-BM3D job list does not match the formal UCM manifest")
PY

matlab_quote() {
  sed "s/'/''/g" <<< "$1"
}
wrapper_dir=$(matlab_quote "$project_root/matlab")
package_dir=$(matlab_quote "$SARBM3D_ROOT")
jobs_path=$(matlab_quote "$jobs")
pair_root=$(matlab_quote "$PREP_ROOT")
prediction_root=$(matlab_quote "$RUN_ROOT/sarbm3d_raw_ucm")
if [[ -e "$RUN_ROOT/sarbm3d_raw_ucm/completion.json" ]]; then
  python verify_icsps2026_artifact.py \
    --kind sarbm_raw \
    --path "$RUN_ROOT/sarbm3d_raw_ucm/completion.json" \
    --expected-method sar_bm3d --expected-variant v1.0 \
    --expected-seed "" --expected-adaptation ""
  echo "Skipping verified SAR-BM3D raw predictions: $RUN_ROOT/sarbm3d_raw_ucm"
else
  matlab -batch "addpath('$wrapper_dir'); run_icsps2026_sarbm3d('$package_dir','$jobs_path','$pair_root','$prediction_root')"
  python finalize_icsps2026_sarbm3d_raw.py \
    --jobs "$jobs" \
    --fixed-pair-root "$PREP_ROOT" \
    --prediction-root "$RUN_ROOT/sarbm3d_raw_ucm"
  python verify_icsps2026_artifact.py \
    --kind sarbm_raw \
    --path "$RUN_ROOT/sarbm3d_raw_ucm/completion.json" \
    --expected-method sar_bm3d --expected-variant v1.0 \
    --expected-seed "" --expected-adaptation ""
fi

archive_sha=$(sha256sum "$SARBM3D_ARCHIVE" | awk '{print $1}')
function_sha=$(sha256sum "$SARBM3D_FUNCTION_FILE" | awk '{print $1}')
python evaluate_icsps2026_sarbm3d.py \
  --config "$PREP_ROOT/config_snapshot.json" \
  --fixed-pair-root "$PREP_ROOT" \
  --manifest "$PREP_ROOT/ucm_test_manifest.csv" \
  --prediction-root "$RUN_ROOT/sarbm3d_raw_ucm" \
  --package-archive-sha256 "$archive_sha" \
  --package-function-sha256 "$function_sha" \
  --run-root "$RUN_ROOT" \
  --output-dir "$result_dir"
python verify_icsps2026_artifact.py --kind ucm --path "$result_dir/aggregate.json" \
  --expected-method sar_bm3d --expected-variant v1.0 \
  --expected-seed "" --expected-adaptation ""

echo "Official SAR-BM3D v1.0 UCM evaluation completed."
