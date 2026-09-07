#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$project_root"

if [[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" -ne 2 ]]; then
  echo "Expected two visible GPUs before launching the two-worker matrix." >&2
  exit 1
fi
if ! command -v setsid >/dev/null 2>&1; then
  echo "setsid (util-linux) is required for safe two-queue process cleanup." >&2
  exit 1
fi

default_gpu0_methods="ours:full ours:log_only"
default_gpu1_methods="transsar_v2: ours:intensity_only ours:wout_all_fdr"
gpu0_methods=${GPU0_METHOD_SPECS:-$default_gpu0_methods}
gpu1_methods=${GPU1_METHOD_SPECS:-$default_gpu1_methods}
if [[ "${INCLUDE_SAR_CAM:-0}" == "1" ]]; then
  if [[ -z "${GPU0_METHOD_SPECS+x}" && -z "${GPU1_METHOD_SPECS+x}" ]]; then
    gpu0_methods="$gpu0_methods sar_cam:"
  else
    echo "Custom GPU queues selected; SAR-CAM is not auto-appended. Add sar_cam: explicitly if required."
  fi
fi

echo "GPU 0 queue: $gpu0_methods"
echo "GPU 1 queue: $gpu1_methods"
echo "Each process sees one 24-GiB GPU; batch size remains 1 and no DDP is used."

setsid env \
  CUDA_VISIBLE_DEVICES=0 \
  SAR_DEVICE=cuda \
  OMP_NUM_THREADS=4 \
  MKL_NUM_THREADS=4 \
  CORE_METHOD_SPECS="$gpu0_methods" \
  bash scripts/icsps2026/03_train_core.sh &
pid_gpu0=$!

setsid env \
  CUDA_VISIBLE_DEVICES=1 \
  SAR_DEVICE=cuda \
  OMP_NUM_THREADS=4 \
  MKL_NUM_THREADS=4 \
  CORE_METHOD_SPECS="$gpu1_methods" \
  bash scripts/icsps2026/03_train_core.sh &
pid_gpu1=$!

terminate_workers() {
  kill -TERM -- "-$pid_gpu0" "-$pid_gpu1" 2>/dev/null || true
}
trap 'terminate_workers; wait "$pid_gpu0" "$pid_gpu1" 2>/dev/null || true; exit 130' INT TERM

# Ubuntu 22.04 ships Bash 5.1, whose wait -n/-p support lets us stop the
# second paid GPU queue promptly if either queue fails.
set +e
wait -n -p completed_pid "$pid_gpu0" "$pid_gpu1"
first_status=$?
set -e
if [[ "$first_status" -ne 0 ]]; then
  terminate_workers
  wait "$pid_gpu0" "$pid_gpu1" 2>/dev/null || true
  echo "A GPU worker failed; the other queue was stopped. Inspect RUN_ROOT/logs, then rerun to resume." >&2
  exit "$first_status"
fi

if [[ "$completed_pid" == "$pid_gpu0" ]]; then
  remaining_pid=$pid_gpu1
else
  remaining_pid=$pid_gpu0
fi
if ! wait "$remaining_pid"; then
  echo "A GPU worker failed; inspect RUN_ROOT/logs before resuming." >&2
  exit 1
fi
trap - INT TERM

echo "Both single-GPU experiment queues completed."
