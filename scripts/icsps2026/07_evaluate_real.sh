#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$project_root"

: "${REAL_ROOT:?Set REAL_ROOT to the real-SAR dataset root}"
: "${REAL_PROTOCOL_ROOT:?Set REAL_PROTOCOL_ROOT to the frozen QC directory}"
: "${RUN_ROOT:?Set RUN_ROOT to the formal experiment output root}"

python verify_icsps2026_pretest_gate.py --run-root "$RUN_ROOT"

seed=${REAL_EVAL_SEED:-42}
noisy_output="$RUN_ROOT/real/noisy_base/seed${seed}"
if [[ -e "$noisy_output/aggregate.json" ]]; then
  python verify_icsps2026_artifact.py --kind real --path "$noisy_output/aggregate.json" \
    --expected-method noisy --expected-variant "" --expected-seed "$seed" \
    --expected-adaptation base
  echo "Skipping completed real noisy-input evaluation: $noisy_output"
else
  python evaluate_icsps2026_real.py \
    --dataset-root "$REAL_ROOT" \
    --qc-manifest "$REAL_PROTOCOL_ROOT/real_qc_manifest.json" \
    --output-dir "$noisy_output" \
    --run-root "$RUN_ROOT" \
    --method noisy \
    --adaptation base \
    --split test \
    --seed "$seed" \
    --bootstrap-samples 10000
  python verify_icsps2026_artifact.py --kind real --path "$noisy_output/aggregate.json" \
    --expected-method noisy --expected-variant "" --expected-seed "$seed" \
    --expected-adaptation base
fi

specs=(
  "ours:full:base:$RUN_ROOT/train/ours_full/seed${seed}/checkpoint_best.pth"
  "ours:log_only:base:$RUN_ROOT/train/ours_log_only/seed${seed}/checkpoint_best.pth"
  "transsar_v2::base:$RUN_ROOT/train/transsar_v2/seed${seed}/checkpoint_best.pth"
  "ours:full:ams:$RUN_ROOT/ams/ours_full/seed${seed}/checkpoint_best.pth"
)
if [[ "${INCLUDE_SAR_CAM:-0}" == "1" ]]; then
  : "${SAR_CAM_ROOT:?Set SAR_CAM_ROOT when INCLUDE_SAR_CAM=1}"
  specs+=("sar_cam::base:$RUN_ROOT/train/sar_cam/seed${seed}/checkpoint_best.pth")
fi

for spec in "${specs[@]}"; do
  IFS=: read -r method variant adaptation checkpoint <<< "$spec"
  name="${method}_${adaptation}"
  variant_args=()
  if [[ -n "$variant" ]]; then
    name="${method}_${variant}_${adaptation}"
    variant_args=(--variant "$variant")
  fi
  output="$RUN_ROOT/real/$name/seed${seed}"
  identity_args=(
    --expected-method "$method" --expected-variant "$variant"
    --expected-seed "$seed" --expected-adaptation "$adaptation"
  )
  if [[ -e "$output/aggregate.json" ]]; then
    python verify_icsps2026_artifact.py --kind real \
      --path "$output/aggregate.json" "${identity_args[@]}"
    echo "Skipping completed real evaluation: $output"
    continue
  fi
  if [[ ! -f "$checkpoint" ]]; then
    echo "Missing checkpoint: $checkpoint" >&2
    exit 1
  fi
  checkpoint_dir=${checkpoint%/*}
  checkpoint_kind=supervised
  checkpoint_adaptation=""
  if [[ "$adaptation" == "ams" ]]; then
    checkpoint_kind=ams
    checkpoint_adaptation=ams
  fi
  python verify_icsps2026_artifact.py \
    --kind "$checkpoint_kind" \
    --path "$checkpoint_dir/completion.json" \
    --expected-method "$method" --expected-variant "$variant" \
    --expected-seed "$seed" --expected-adaptation "$checkpoint_adaptation"
  external_args=()
  if [[ "$method" == "sar_cam" ]]; then
    external_args=(--sar-cam-root "$SAR_CAM_ROOT")
  fi
  python evaluate_icsps2026_real.py \
    --dataset-root "$REAL_ROOT" \
    --qc-manifest "$REAL_PROTOCOL_ROOT/real_qc_manifest.json" \
    --checkpoint "$checkpoint" \
    --output-dir "$output" \
    --run-root "$RUN_ROOT" \
    --method "$method" \
    "${variant_args[@]}" \
    "${external_args[@]}" \
    --adaptation "$adaptation" \
    --split test \
    --seed "$seed" \
    --device "${SAR_DEVICE:-cuda}" \
    --bootstrap-samples 10000
  python verify_icsps2026_artifact.py --kind real \
    --path "$output/aggregate.json" "${identity_args[@]}"
done

echo "No-GT16 real noisy and learned-method evaluation completed. Use only the pre-registered real-SAR quality outputs."
