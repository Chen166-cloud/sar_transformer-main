#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$project_root"

: "${NWPU_ROOT:?Set NWPU_ROOT to the NWPU class-folder directory}"
: "${UCM_ROOT:?Set UCM_ROOT to the UCM class-folder directory}"
: "${REAL_ROOT:?Set REAL_ROOT to the real-SAR dataset root}"
: "${REAL_PROTOCOL_ROOT:?Set REAL_PROTOCOL_ROOT to the frozen real-SAR QC directory}"
: "${SMOKE_ROOT:?Set SMOKE_ROOT to a disposable, initially absent directory}"

if [[ ! -f "$REAL_PROTOCOL_ROOT/real_qc_manifest.json" ]]; then
  echo "Run 05_prepare_real.sh before the end-to-end smoke test." >&2
  exit 1
fi

prep_root="$SMOKE_ROOT/data"
run_root="$SMOKE_ROOT/runs"
if [[ -e "$SMOKE_ROOT" ]]; then
  echo "Refusing to overwrite smoke root: $SMOKE_ROOT" >&2
  exit 1
fi
mkdir -p "$SMOKE_ROOT"

python prepare_icsps2026_data.py \
  --nwpu-root "$NWPU_ROOT" \
  --ucm-root "$UCM_ROOT" \
  --output-root "$prep_root" \
  --run-seed 42 \
  --smoke

python prepare_icsps2026_data.py \
  --nwpu-root "$NWPU_ROOT" \
  --ucm-root "$UCM_ROOT" \
  --output-root "$prep_root" \
  --run-seed 42 \
  --smoke \
  --verify-only \
  --verify-source-files \
  --verify-pair-files

methods=(
  "ours:full"
  "ours:intensity_only"
  "ours:log_only"
  "ours:wout_all_fdr"
  "transsar_v2:"
)
if [[ "${INCLUDE_SAR_CAM:-0}" == "1" ]]; then
  : "${SAR_CAM_ROOT:?Set SAR_CAM_ROOT when INCLUDE_SAR_CAM=1}"
  methods+=("sar_cam:")
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
  external_args=()
  if [[ "$method" == "sar_cam" ]]; then
    external_args=(--sar-cam-root "$SAR_CAM_ROOT")
  fi
  python train_icsps2026.py \
    --config "$prep_root/config_snapshot.json" \
    --source-root "$NWPU_ROOT" \
    --source-manifest "$prep_root/source_manifest.csv" \
    --train-schedule "$prep_root/train_schedule_seed42.csv" \
    --fixed-pair-root "$prep_root" \
    --val-manifest "$prep_root/nwpu_validation_manifest.csv" \
    --method "$method" \
    "${variant_args[@]}" \
    "${external_args[@]}" \
    --seed 42 \
    --device "${SAR_DEVICE:-cuda}" \
    --workers "${SAR_WORKERS:-2}" \
    --output-dir "$run_root/train/$name" \
    --nonformal-smoke \
    --max-updates 2 \
    --max-val-pairs 4

  python evaluate_icsps2026_synthetic.py \
    --config "$prep_root/config_snapshot.json" \
    --dataset nwpu_validation \
    --fixed-pair-root "$prep_root" \
    --manifest "$prep_root/nwpu_validation_manifest.csv" \
    --method "$method" \
    "${variant_args[@]}" \
    "${external_args[@]}" \
    --checkpoint "$run_root/train/$name/checkpoint_best.pth" \
    --seed 42 \
    --device "${SAR_DEVICE:-cuda}" \
    --workers "${SAR_WORKERS:-2}" \
    --output-dir "$run_root/nwpu_validation/$name" \
    --nonformal-smoke \
    --max-pairs 4
done

# Exercise both real-SAR inference paths on the validation split before any
# expensive formal run.  No UCM or real-test prediction/metric is exposed before
# the pretest registration gate.  These four-patch outputs remain non-formal.
python evaluate_icsps2026_real.py \
  --dataset-root "$REAL_ROOT" \
  --qc-manifest "$REAL_PROTOCOL_ROOT/real_qc_manifest.json" \
  --output-dir "$run_root/real/noisy_base" \
  --method noisy \
  --adaptation base \
  --split val \
  --seed 42 \
  --bootstrap-samples 20 \
  --nonformal-smoke \
  --max-images 4 \
  --save-images

python evaluate_icsps2026_real.py \
  --dataset-root "$REAL_ROOT" \
  --qc-manifest "$REAL_PROTOCOL_ROOT/real_qc_manifest.json" \
  --checkpoint "$run_root/train/ours_full/checkpoint_best.pth" \
  --output-dir "$run_root/real/ours_full_base" \
  --method ours \
  --variant full \
  --adaptation base \
  --split val \
  --seed 42 \
  --device "${SAR_DEVICE:-cuda}" \
  --bootstrap-samples 20 \
  --nonformal-smoke \
  --max-images 4 \
  --save-images

echo "Smoke pipeline passed. Never copy numbers from $SMOKE_ROOT into the paper."
