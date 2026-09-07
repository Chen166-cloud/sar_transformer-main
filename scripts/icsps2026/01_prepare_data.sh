#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$project_root"

: "${NWPU_ROOT:?Set NWPU_ROOT to the 45-class NWPU-RESISC45 directory}"
: "${UCM_ROOT:?Set UCM_ROOT to the 21-class UCMerced_LandUse directory}"
: "${PREP_ROOT:?Set PREP_ROOT to a new protocol-artifact directory}"
: "${NWPU_MASTER_MANIFEST:?Set NWPU_MASTER_MANIFEST to the audited 560/140 master CSV/JSON}"
protocol_source_config="$project_root/configs/icsps2026_frozen_v2.json"

for required_path in "$NWPU_ROOT" "$UCM_ROOT"; do
  if [[ ! -d "$required_path" ]]; then
    echo "Required dataset directory does not exist: $required_path" >&2
    exit 1
  fi
done
if [[ ! -f "$NWPU_MASTER_MANIFEST" ]]; then
  echo "Frozen NWPU master manifest does not exist: $NWPU_MASTER_MANIFEST" >&2
  exit 1
fi

if [[ ! -d "$PREP_ROOT" ]]; then
  python prepare_icsps2026_data.py \
    --config "$protocol_source_config" \
    --nwpu-root "$NWPU_ROOT" \
    --ucm-root "$UCM_ROOT" \
    --output-root "$PREP_ROOT" \
    --run-seed 42 \
    --nwpu-master-manifest "$NWPU_MASTER_MANIFEST"
fi

python prepare_icsps2026_data.py \
  --config "$protocol_source_config" \
  --nwpu-root "$NWPU_ROOT" \
  --ucm-root "$UCM_ROOT" \
  --nwpu-master-manifest "$NWPU_MASTER_MANIFEST" \
  --output-root "$PREP_ROOT" \
  --run-seed 42 \
  --verify-only \
  --verify-source-files \
  --verify-pair-files

python generate_icsps2026_schedules.py \
  --config "$protocol_source_config" \
  --prepared-root "$PREP_ROOT" \
  --run-seeds 42 43 44

for seed in 42 43 44; do
  test -s "$PREP_ROOT/train_schedule_seed${seed}.csv"
done

echo "Formal synthetic artifacts verified at $PREP_ROOT"
