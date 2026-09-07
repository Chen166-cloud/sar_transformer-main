#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$project_root"

inventory_dir=${1:-server_inventory}
if [[ -e "$inventory_dir" ]]; then
  echo "Refusing to overwrite inventory directory: $inventory_dir" >&2
  exit 1
fi
mkdir -p "$inventory_dir"

nvidia-smi -q > "$inventory_dir/nvidia-smi-q.txt"
nvidia-smi > "$inventory_dir/nvidia-smi.txt"
lscpu > "$inventory_dir/lscpu.txt"
nproc > "$inventory_dir/nproc.txt"
getconf _NPROCESSORS_ONLN > "$inventory_dir/getconf-processors-onln.txt"
free -h > "$inventory_dir/free-h.txt"
df -hT > "$inventory_dir/df-hT.txt"
lsblk -o NAME,SIZE,FSTYPE,MOUNTPOINTS,MODEL > "$inventory_dir/lsblk.txt"
python --version > "$inventory_dir/python-version.txt" 2>&1
python -m pip --version > "$inventory_dir/pip-version.txt" 2>&1

# Containers can expose the host CPU topology in lscpu while enforcing a
# smaller cgroup quota.  Preserve both v2 and legacy v1 limits when present.
for cgroup_file in \
  /sys/fs/cgroup/cpu.max \
  /sys/fs/cgroup/cpuset.cpus.effective \
  /sys/fs/cgroup/memory.max \
  /sys/fs/cgroup/cpu/cpu.cfs_quota_us \
  /sys/fs/cgroup/cpu/cpu.cfs_period_us; do
  if [[ -f "$cgroup_file" ]]; then
    safe_name=${cgroup_file#/sys/fs/cgroup/}
    safe_name=${safe_name//\//-}
    cp "$cgroup_file" "$inventory_dir/cgroup-${safe_name}.txt"
  fi
done

python - <<'PY' > "$inventory_dir/torch-runtime.json"
import json
import torch

payload = {
    "torch": torch.__version__,
    "torch_cuda_build": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "visible_gpu_count": torch.cuda.device_count(),
    "gpus": [],
}
for index in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(index)
    payload["gpus"].append(
        {
            "index": index,
            "name": props.name,
            "vram_gib": props.total_memory / 1024**3,
            "compute_capability": f"{props.major}.{props.minor}",
        }
    )
print(json.dumps(payload, indent=2, sort_keys=True))
PY

if [[ -n "${SAR_DATA_WORK_ROOT:-}" ]]; then
  case "$SAR_DATA_WORK_ROOT" in
    /*) ;;
    *)
      echo "SAR_DATA_WORK_ROOT must be an absolute path: $SAR_DATA_WORK_ROOT" >&2
      exit 1
      ;;
  esac
  case "$SAR_DATA_WORK_ROOT" in
    /|/SET|/SET/*)
      echo "SAR_DATA_WORK_ROOT is unsafe or still a placeholder: $SAR_DATA_WORK_ROOT" >&2
      exit 1
      ;;
  esac
  if [[ ! -d "$SAR_DATA_WORK_ROOT" ]]; then
    echo "SAR_DATA_WORK_ROOT must already exist on the manually confirmed data mount: $SAR_DATA_WORK_ROOT" >&2
    exit 1
  fi
  df -hT "$SAR_DATA_WORK_ROOT" > "$inventory_dir/selected-data-filesystem.txt"
  df -PT "$SAR_DATA_WORK_ROOT" > "$inventory_dir/selected-data-filesystem-posix.txt"
  available_kib=$(df -Pk "$SAR_DATA_WORK_ROOT" | awk 'END {print $4}')
  minimum_gib=${SAR_MIN_FREE_GIB:-25}
  if [[ ! "$minimum_gib" =~ ^[0-9]+$ ]] || (( minimum_gib < 1 )); then
    echo "SAR_MIN_FREE_GIB must be a positive integer, found: $minimum_gib" >&2
    exit 1
  fi
  minimum_kib=$((minimum_gib * 1024 * 1024))
  if (( available_kib < minimum_kib )); then
    echo "Selected data filesystem has less than ${minimum_gib} GiB available." >&2
    exit 1
  fi
  if [[ -n "${SAR_EXPECTED_DATA_DEVICE:-}" ]]; then
    actual_device=$(df -P "$SAR_DATA_WORK_ROOT" | awk 'END {print $1}')
    if [[ "$actual_device" != "$SAR_EXPECTED_DATA_DEVICE" ]]; then
      echo "Data filesystem mismatch: expected $SAR_EXPECTED_DATA_DEVICE, found $actual_device" >&2
      exit 1
    fi
  else
    actual_device=$(df -P "$SAR_DATA_WORK_ROOT" | awk 'END {print $1}')
    echo "Selected data device: $actual_device" >&2
    echo "Confirm it against lsblk.txt; set SAR_EXPECTED_DATA_DEVICE and rerun for a strict mount check." >&2
  fi
else
  echo "SAR_DATA_WORK_ROOT was not set; inventory captured, but storage target was not validated." >&2
fi

echo "Server inventory written to $inventory_dir"
