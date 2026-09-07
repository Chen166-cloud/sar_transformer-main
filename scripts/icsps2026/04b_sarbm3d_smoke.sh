#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$project_root"
source "$project_root/scripts/icsps2026/sarbm3d_runtime.sh"

: "${PREP_ROOT:?Set PREP_ROOT to the formal prepared-data directory}"
: "${SARBM3D_ROOT:?Set SARBM3D_ROOT to the extracted official v1.0 package directory}"
: "${SARBM3D_ARCHIVE:?Set SARBM3D_ARCHIVE to the downloaded official tar.gz}"
: "${SARBM3D_SMOKE_ROOT:?Set SARBM3D_SMOKE_ROOT to a disposable, initially absent directory}"
: "${RUN_ROOT:?Set RUN_ROOT to the formal experiment root containing pretest registration}"

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
if [[ -e "$SARBM3D_SMOKE_ROOT" ]]; then
  echo "Refusing to overwrite SAR-BM3D smoke root: $SARBM3D_SMOKE_ROOT" >&2
  exit 1
fi

mkdir -p "$SARBM3D_SMOKE_ROOT"
sarbm3d_prepare_runtime \
  "$SARBM3D_ROOT" "$SARBM3D_SMOKE_ROOT/sarbm3d_runtime_probe.log"
jobs="$SARBM3D_SMOKE_ROOT/jobs.csv"
prediction_root="$SARBM3D_SMOKE_ROOT/raw_predictions"
python export_icsps2026_sarbm3d_jobs.py \
  --fixed-pair-root "$PREP_ROOT" \
  --manifest "$PREP_ROOT/ucm_test_manifest.csv" \
  --run-root "$RUN_ROOT" \
  --output "$jobs" \
  --nonformal-smoke \
  --max-pairs 4

matlab_quote() {
  sed "s/'/''/g" <<< "$1"
}
wrapper_dir=$(matlab_quote "$project_root/matlab")
package_dir=$(matlab_quote "$SARBM3D_ROOT")
jobs_path=$(matlab_quote "$jobs")
pair_root=$(matlab_quote "$PREP_ROOT")
prediction_path=$(matlab_quote "$prediction_root")
LD_LIBRARY_PATH="$LD_LIBRARY_PATH" matlab -batch "addpath('$wrapper_dir'); run_icsps2026_sarbm3d('$package_dir','$jobs_path','$pair_root','$prediction_path')"

python finalize_icsps2026_sarbm3d_raw.py \
  --jobs "$jobs" \
  --fixed-pair-root "$PREP_ROOT" \
  --prediction-root "$prediction_root" \
  --nonformal-smoke

archive_sha=$(sha256sum "$SARBM3D_ARCHIVE" | awk '{print $1}')
function_sha=$(sha256sum "$SARBM3D_FUNCTION_FILE" | awk '{print $1}')
python evaluate_icsps2026_sarbm3d.py \
  --config "$PREP_ROOT/config_snapshot.json" \
  --fixed-pair-root "$PREP_ROOT" \
  --manifest "$PREP_ROOT/ucm_test_manifest.csv" \
  --prediction-root "$prediction_root" \
  --package-archive-sha256 "$archive_sha" \
  --package-function-sha256 "$function_sha" \
  --output-dir "$SARBM3D_SMOKE_ROOT/result" \
  --run-root "$RUN_ROOT" \
  --nonformal-smoke \
  --max-pairs 4

echo "Four-pair SAR-BM3D compatibility smoke passed. Never report these numbers."
