#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$project_root"

: "${NWPU_ROOT:?Set NWPU_ROOT to the NWPU class-folder directory}"
: "${PREP_ROOT:?Set PREP_ROOT to the formal prepared-data directory}"
: "${RUN_ROOT:?Set RUN_ROOT to the formal experiment output root}"

protocol_config="$PREP_ROOT/config_snapshot.json"
if [[ ! -f "$protocol_config" ]]; then
  echo "Missing prepared protocol snapshot: $protocol_config" >&2
  exit 1
fi

seed_list=${CORE_SEEDS:-42}
read -r -a seeds <<< "$seed_list"
if [[ -n "${CORE_METHOD_SPECS:-}" ]]; then
  read -r -a methods <<< "$CORE_METHOD_SPECS"
else
  methods=(
    "ours:full"
    "transsar_v2:"
    "ours:intensity_only"
    "ours:log_only"
    "ours:wout_all_fdr"
  )
fi
if [[ "${INCLUDE_SAR_CAM:-0}" == "1" ]]; then
  : "${SAR_CAM_ROOT:?Set SAR_CAM_ROOT when INCLUDE_SAR_CAM=1}"
  if [[ -z "${CORE_METHOD_SPECS:-}" ]]; then
    methods+=("sar_cam:")
  fi
fi
for spec in "${methods[@]}"; do
  if [[ "${spec%%:*}" == "sar_cam" ]]; then
    : "${SAR_CAM_ROOT:?Set SAR_CAM_ROOT when a SAR-CAM run is requested}"
  fi
done

mkdir -p "$RUN_ROOT/logs"
for seed in "${seeds[@]}"; do
  schedule="$PREP_ROOT/train_schedule_seed${seed}.csv"
  if [[ ! -f "$schedule" ]]; then
    echo "Missing schedule: $schedule" >&2
    exit 1
  fi
  for spec in "${methods[@]}"; do
    method=${spec%%:*}
    variant=${spec#*:}
    name="$method"
    variant_args=()
    if [[ -n "$variant" ]]; then
      name="${method}_${variant}"
      variant_args=(--variant "$variant")
    fi
    output="$RUN_ROOT/train/$name/seed${seed}"
    log="$RUN_ROOT/logs/train_${name}_seed${seed}.log"
    identity_args=(
      --expected-method "$method" --expected-variant "$variant"
      --expected-seed "$seed" --expected-adaptation ""
    )
    if [[ -e "$output/completion.json" ]]; then
      python verify_icsps2026_artifact.py --kind supervised \
        --path "$output/completion.json" "${identity_args[@]}"
      echo "Skipping completed run: $output"
      continue
    fi
    resume_args=()
    if [[ -f "$output/checkpoint_last.pth" ]]; then
      resume_args=(--resume "$output/checkpoint_last.pth")
    elif [[ -e "$output" ]]; then
      echo "Non-empty/incomplete output has no resumable checkpoint: $output" >&2
      exit 1
    fi
    external_args=()
    if [[ "$method" == "sar_cam" ]]; then
      external_args=(--sar-cam-root "$SAR_CAM_ROOT")
    fi
    python train_icsps2026.py \
      --config "$protocol_config" \
      --source-root "$NWPU_ROOT" \
      --source-manifest "$PREP_ROOT/source_manifest.csv" \
      --train-schedule "$schedule" \
      --fixed-pair-root "$PREP_ROOT" \
      --val-manifest "$PREP_ROOT/nwpu_validation_manifest.csv" \
      --method "$method" \
      "${variant_args[@]}" \
      "${external_args[@]}" \
      --seed "$seed" \
      --device "${SAR_DEVICE:-cuda}" \
      --workers "${SAR_WORKERS:-4}" \
      --output-dir "$output" \
      --tensorboard \
      "${resume_args[@]}" \
      2>&1 | tee -a "$log"
    python verify_icsps2026_artifact.py --kind supervised \
      --path "$output/completion.json" "${identity_args[@]}"
  done
done

echo "Requested core training matrix completed."
