#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$project_root"

: "${RUN_ROOT:?Set RUN_ROOT to the formal experiment output root}"
if [[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" -ne 2 ]]; then
  echo "Expected two visible GPUs before launching dual evaluation queues." >&2
  exit 1
fi
if ! command -v setsid >/dev/null 2>&1; then
  echo "setsid (util-linux) is required for safe two-queue process cleanup." >&2
  exit 1
fi

default_gpu0_methods="ours:full ours:log_only"
default_gpu1_methods="transsar_v2: ours:intensity_only ours:wout_all_fdr"
gpu0_methods=${GPU0_EVAL_METHOD_SPECS:-$default_gpu0_methods}
gpu1_methods=${GPU1_EVAL_METHOD_SPECS:-$default_gpu1_methods}
if [[ "${INCLUDE_SAR_CAM:-0}" == "1" ]]; then
  if [[ -z "${GPU0_EVAL_METHOD_SPECS+x}" && -z "${GPU1_EVAL_METHOD_SPECS+x}" ]]; then
    gpu0_methods="$gpu0_methods sar_cam:"
  else
    echo "Custom evaluation queues selected; SAR-CAM is not auto-appended."
  fi
fi

declare -A seen_specs=()
for spec in $gpu0_methods $gpu1_methods; do
  if [[ -n "${seen_specs[$spec]+x}" ]]; then
    echo "The same method/variant appears in both evaluation queues: $spec" >&2
    exit 2
  fi
  seen_specs[$spec]=1
done
if [[ -z "$gpu0_methods" || -z "$gpu1_methods" ]]; then
  echo "Both GPU evaluation queues must be non-empty." >&2
  exit 2
fi

echo "Running shared Noisy and Lee-MMSE baselines once before learned queues."
CUDA_VISIBLE_DEVICES=0 EVAL_RUN_SHARED_BASELINES=1 EVAL_RUN_LEARNED=0 \
  bash scripts/icsps2026/04_evaluate_synthetic.sh

echo "GPU 0 evaluation queue: $gpu0_methods"
echo "GPU 1 evaluation queue: $gpu1_methods"
echo "Each process sees one independent 24-GiB GPU; no memory pooling or DDP is used."

setsid env \
  CUDA_VISIBLE_DEVICES=0 \
  SAR_DEVICE=cuda \
  OMP_NUM_THREADS=4 \
  MKL_NUM_THREADS=4 \
  EVAL_RUN_SHARED_BASELINES=0 \
  EVAL_RUN_LEARNED=1 \
  EVAL_METHOD_SPECS="$gpu0_methods" \
  bash scripts/icsps2026/04_evaluate_synthetic.sh &
pid_gpu0=$!

setsid env \
  CUDA_VISIBLE_DEVICES=1 \
  SAR_DEVICE=cuda \
  OMP_NUM_THREADS=4 \
  MKL_NUM_THREADS=4 \
  EVAL_RUN_SHARED_BASELINES=0 \
  EVAL_RUN_LEARNED=1 \
  EVAL_METHOD_SPECS="$gpu1_methods" \
  bash scripts/icsps2026/04_evaluate_synthetic.sh &
pid_gpu1=$!

terminate_workers() {
  kill -TERM -- "-$pid_gpu0" "-$pid_gpu1" 2>/dev/null || true
}
trap 'terminate_workers; wait "$pid_gpu0" "$pid_gpu1" 2>/dev/null || true; exit 130' INT TERM

set +e
wait -n -p completed_pid "$pid_gpu0" "$pid_gpu1"
first_status=$?
set -e
if [[ "$first_status" -ne 0 ]]; then
  terminate_workers
  wait "$pid_gpu0" "$pid_gpu1" 2>/dev/null || true
  echo "A GPU evaluation worker failed; the other queue was stopped." >&2
  exit "$first_status"
fi

if [[ "$completed_pid" == "$pid_gpu0" ]]; then
  remaining_pid=$pid_gpu1
else
  remaining_pid=$pid_gpu0
fi
if ! wait "$remaining_pid"; then
  echo "A GPU evaluation worker failed; inspect the output directories before resuming." >&2
  exit 1
fi
trap - INT TERM

echo "Both independent single-GPU learned evaluation queues completed."
