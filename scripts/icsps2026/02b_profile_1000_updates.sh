#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$project_root"

: "${NWPU_ROOT:?Set NWPU_ROOT}"
: "${PREP_ROOT:?Set PREP_ROOT to formal prepared data}"
: "${PROFILE_ROOT:?Set PROFILE_ROOT to a new disposable output directory}"
if [[ -e "$PROFILE_ROOT" ]]; then
  echo "Refusing to overwrite profile root: $PROFILE_ROOT" >&2
  exit 1
fi
mkdir -p "$PROFILE_ROOT"

run_profile() {
  local gpu=$1
  local method=$2
  local variant=$3
  local name=$method
  local variant_args=()
  if [[ -n "$variant" ]]; then
    name="${method}_${variant}"
    variant_args=(--variant "$variant")
  fi
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  python train_icsps2026.py \
    --config "$PREP_ROOT/config_snapshot.json" \
    --source-root "$NWPU_ROOT" \
    --source-manifest "$PREP_ROOT/source_manifest.csv" \
    --train-schedule "$PREP_ROOT/train_schedule_seed42.csv" \
    --fixed-pair-root "$PREP_ROOT" \
    --val-manifest "$PREP_ROOT/nwpu_validation_manifest.csv" \
    --method "$method" \
    "${variant_args[@]}" \
    --seed 42 \
    --device cuda \
    --workers 4 \
    --output-dir "$PROFILE_ROOT/$name" \
    --nonformal-smoke \
    --max-updates 1000 \
    --max-val-pairs 40
}

run_profile 0 ours full &
pid0=$!
run_profile 1 transsar_v2 "" &
pid1=$!
status=0
wait "$pid0" || status=1
wait "$pid1" || status=1
if [[ "$status" -ne 0 ]]; then
  exit 1
fi

echo "1000-update profiles complete. They are timing diagnostics, never paper results."
echo "Use completion.json and step_metrics.csv to estimate wall time before paid formal runs."
