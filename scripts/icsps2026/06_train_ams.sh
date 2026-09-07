#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$project_root"

: "${REAL_ROOT:?Set REAL_ROOT to the real-SAR dataset root}"
: "${REAL_PROTOCOL_ROOT:?Set REAL_PROTOCOL_ROOT to the frozen QC directory}"
: "${PREP_ROOT:?Set PREP_ROOT to the frozen protocol-artifact directory}"
: "${RUN_ROOT:?Set RUN_ROOT to the formal experiment output root}"

protocol_config="$PREP_ROOT/config_snapshot.json"
if [[ ! -f "$protocol_config" ]]; then
  echo "Missing prepared protocol snapshot: $protocol_config" >&2
  exit 1
fi

seed=${AMS_SEED:-42}
base="$RUN_ROOT/train/ours_full/seed${seed}/checkpoint_best.pth"
base_completion="$RUN_ROOT/train/ours_full/seed${seed}/completion.json"
output="$RUN_ROOT/ams/ours_full/seed${seed}"
if [[ ! -f "$base" ]]; then
  echo "Missing Full base checkpoint: $base" >&2
  exit 1
fi
python verify_icsps2026_artifact.py --kind supervised --path "$base_completion" \
  --expected-method ours --expected-variant full --expected-seed "$seed" \
  --expected-adaptation ""
if [[ -e "$output/completion.json" ]]; then
  python verify_icsps2026_artifact.py --kind ams --path "$output/completion.json" \
    --expected-method ours --expected-variant full --expected-seed "$seed" \
    --expected-adaptation ams
  echo "AMS already completed: $output"
  exit 0
fi
resume_args=()
if [[ -f "$output/checkpoint_last.pth" ]]; then
  resume_args=(--resume "$output/checkpoint_last.pth")
elif [[ -e "$output" ]]; then
  echo "Incomplete AMS output has no checkpoint_last.pth: $output" >&2
  exit 1
fi

python train_icsps2026_ams.py \
  --config "$protocol_config" \
  --dataset-root "$REAL_ROOT" \
  --real-qc-manifest "$REAL_PROTOCOL_ROOT/real_qc_manifest.json" \
  --base-checkpoint "$base" \
  --output-dir "$output" \
  --seed "$seed" \
  --device "${SAR_DEVICE:-cuda}" \
  --workers "${SAR_WORKERS:-4}" \
  "${resume_args[@]}"
python verify_icsps2026_artifact.py --kind ams --path "$output/completion.json" \
  --expected-method ours --expected-variant full --expected-seed "$seed" \
  --expected-adaptation ams
