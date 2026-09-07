#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$project_root"

: "${PREP_ROOT:?Set PREP_ROOT to the formal prepared-data directory}"
: "${RUN_ROOT:?Set RUN_ROOT to the formal experiment output root}"

protocol_config="$PREP_ROOT/config_snapshot.json"
if [[ ! -f "$protocol_config" ]]; then
  echo "Missing prepared protocol snapshot: $protocol_config" >&2
  exit 1
fi
python verify_icsps2026_pretest_gate.py --run-root "$RUN_ROOT"

run_shared=${EVAL_RUN_SHARED_BASELINES:-1}
run_learned=${EVAL_RUN_LEARNED:-1}
if [[ "$run_shared" != "0" && "$run_shared" != "1" ]]; then
  echo "EVAL_RUN_SHARED_BASELINES must be 0 or 1." >&2
  exit 2
fi
if [[ "$run_learned" != "0" && "$run_learned" != "1" ]]; then
  echo "EVAL_RUN_LEARNED must be 0 or 1." >&2
  exit 2
fi

seed_list=${EVAL_SEEDS:-42}
read -r -a seeds <<< "$seed_list"
methods=()
if [[ "$run_learned" == "1" ]]; then
  if [[ -n "${EVAL_METHOD_SPECS:-}" ]]; then
    read -r -a methods <<< "$EVAL_METHOD_SPECS"
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
    if [[ -z "${EVAL_METHOD_SPECS:-}" ]]; then
      methods+=("sar_cam:")
    fi
  fi
  for spec in "${methods[@]}"; do
    if [[ "${spec%%:*}" == "sar_cam" ]]; then
      : "${SAR_CAM_ROOT:?Set SAR_CAM_ROOT when a SAR-CAM evaluation is requested}"
    fi
  done
  if [[ "${#methods[@]}" -eq 0 ]]; then
    echo "Learned evaluation was enabled but no EVAL_METHOD_SPECS were supplied." >&2
    exit 2
  fi
fi

if [[ "$run_shared" == "1" ]]; then
  noisy_output="$RUN_ROOT/ucm/noisy"
  if [[ -e "$noisy_output/aggregate.json" ]]; then
    python verify_icsps2026_artifact.py --kind ucm --path "$noisy_output/aggregate.json" \
      --expected-method noisy --expected-variant "" --expected-seed 42 --expected-adaptation ""
    echo "Skipping verified noisy-input evaluation: $noisy_output"
  else
    python evaluate_icsps2026_synthetic.py \
      --config "$protocol_config" \
      --dataset ucm_test \
      --fixed-pair-root "$PREP_ROOT" \
      --manifest "$PREP_ROOT/ucm_test_manifest.csv" \
      --method noisy \
      --seed 42 \
      --run-root "$RUN_ROOT" \
      --device "${SAR_DEVICE:-cuda}" \
      --workers "${SAR_WORKERS:-4}" \
      --output-dir "$noisy_output"
    python verify_icsps2026_artifact.py --kind ucm --path "$noisy_output/aggregate.json" \
      --expected-method noisy --expected-variant "" --expected-seed 42 --expected-adaptation ""
  fi
fi

if [[ "$run_learned" == "1" ]]; then
  for seed in "${seeds[@]}"; do
    for spec in "${methods[@]}"; do
    method=${spec%%:*}
    variant=${spec#*:}
    name="$method"
    variant_args=()
    if [[ -n "$variant" ]]; then
      name="${method}_${variant}"
      variant_args=(--variant "$variant")
    fi
    checkpoint="$RUN_ROOT/train/$name/seed${seed}/checkpoint_best.pth"
    completion="$RUN_ROOT/train/$name/seed${seed}/completion.json"
    output="$RUN_ROOT/ucm/$name/seed${seed}"
    identity_args=(
      --expected-method "$method" --expected-variant "$variant"
      --expected-seed "$seed" --expected-adaptation ""
    )
    if [[ -e "$output/aggregate.json" ]]; then
      python verify_icsps2026_artifact.py --kind ucm \
        --path "$output/aggregate.json" "${identity_args[@]}"
      echo "Skipping completed evaluation: $output"
      continue
    fi
    if [[ ! -f "$checkpoint" ]]; then
      echo "Missing best checkpoint: $checkpoint" >&2
      exit 1
    fi
    python verify_icsps2026_artifact.py --kind supervised \
      --path "$completion" "${identity_args[@]}"
    external_args=()
    if [[ "$method" == "sar_cam" ]]; then
      external_args=(--sar-cam-root "$SAR_CAM_ROOT")
    fi
    python evaluate_icsps2026_synthetic.py \
      --config "$protocol_config" \
      --dataset ucm_test \
      --fixed-pair-root "$PREP_ROOT" \
      --manifest "$PREP_ROOT/ucm_test_manifest.csv" \
      --method "$method" \
      "${variant_args[@]}" \
      "${external_args[@]}" \
      --checkpoint "$checkpoint" \
      --seed "$seed" \
      --run-root "$RUN_ROOT" \
      --device "${SAR_DEVICE:-cuda}" \
      --workers "${SAR_WORKERS:-4}" \
      --output-dir "$output"
    python verify_icsps2026_artifact.py --kind ucm \
      --path "$output/aggregate.json" "${identity_args[@]}"
    done
  done
fi

if [[ "$run_shared" == "1" ]]; then
  lee_output="$RUN_ROOT/ucm/lee_mmse"
  if [[ -e "$lee_output/aggregate.json" ]]; then
    python verify_icsps2026_artifact.py --kind ucm --path "$lee_output/aggregate.json" \
      --expected-method lee_mmse --expected-seed "" --expected-adaptation ""
    echo "Skipping verified Lee-MMSE evaluation: $lee_output"
  else
    python evaluate_icsps2026_lee.py \
      --config "$protocol_config" \
      --fixed-pair-root "$PREP_ROOT" \
      --nwpu-val-manifest "$PREP_ROOT/nwpu_validation_manifest.csv" \
      --ucm-test-manifest "$PREP_ROOT/ucm_test_manifest.csv" \
      --run-root "$RUN_ROOT" \
      --output-dir "$lee_output"
    python verify_icsps2026_artifact.py --kind ucm --path "$lee_output/aggregate.json" \
      --expected-method lee_mmse --expected-seed "" --expected-adaptation ""
  fi
fi

echo "Requested learned/shared UCM evaluations completed."
echo "SAR-BM3D is separate because it requires MATLAB and acceptance of its nonprofit license."
