#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$project_root"

for required_command in \
  nvidia-smi lscpu free df lsblk git curl unzip tar gzip rsync jq tmux setsid ldd \
  sha256sum md5sum; do
  if ! command -v "$required_command" >/dev/null 2>&1; then
    echo "Required server command is missing: $required_command" >&2
    exit 1
  fi
done

bash -n scripts/icsps2026/*.sh

python - <<'PY'
import json
import platform

import cv2
import numpy
import scipy
import torch
import torchvision

os_release = platform.freedesktop_os_release()
report = {
    "os_id": os_release.get("ID"),
    "os_version": os_release.get("VERSION_ID"),
    "python": platform.python_version(),
    "numpy": numpy.__version__,
    "scipy": scipy.__version__,
    "opencv": cv2.__version__,
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "cuda_runtime": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "gpu_count": torch.cuda.device_count(),
    "gpus": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
}
print(json.dumps(report, indent=2, sort_keys=True))
if os_release.get("ID") != "ubuntu" or os_release.get("VERSION_ID") != "22.04":
    raise SystemExit(
        "Expected Ubuntu 22.04, found "
        f"{os_release.get('ID')} {os_release.get('VERSION_ID')}"
    )
if platform.python_version_tuple()[:2] != ("3", "12"):
    raise SystemExit(f"Expected Python 3.12, found {platform.python_version()}")
if torch.__version__.split("+")[0] != "2.5.1":
    raise SystemExit(f"Expected PyTorch 2.5.1, found {torch.__version__}")
if torchvision.__version__.split("+")[0] != "0.20.1":
    raise SystemExit(f"Expected torchvision 0.20.1, found {torchvision.__version__}")
if torch.version.cuda != "12.4":
    raise SystemExit(
        f"Expected the image's PyTorch CUDA build 12.4, found {torch.version.cuda}; "
        "do not confuse this with the host driver's CUDA capability"
    )
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; do not start a formal 100k-update run")
if torch.cuda.device_count() != 2:
    raise SystemExit(
        f"Expected two visible RTX 4090 GPUs for this server profile, found {torch.cuda.device_count()}"
    )
for index in range(torch.cuda.device_count()):
    properties = torch.cuda.get_device_properties(index)
    if "RTX 4090" not in properties.name or properties.total_memory < 23 * 1024**3:
        raise SystemExit(
            f"GPU {index} does not match the declared 24-GB RTX 4090 profile: "
            f"{properties.name}, {properties.total_memory / 1024**3:.2f} GiB"
        )
    # Force a real CUDA context/allocation on each independently visible GPU.
    probe = torch.empty(1, device=f"cuda:{index}")
    del probe
PY

python -m compileall -q \
  ablation_config.py \
  experiment_protocol.py \
  icsps2026_protocol.py \
  icsps2026_metrics.py \
  model_registry.py \
  numeric_domain.py \
  sar_metrics.py \
  transform_main.py \
  prepare_icsps2026_data.py \
  generate_icsps2026_schedules.py \
  train_icsps2026.py \
  train_icsps2026_ams.py \
  evaluate_icsps2026_synthetic.py \
  evaluate_icsps2026_lee.py \
  evaluate_icsps2026_real.py \
  export_icsps2026_sarbm3d_jobs.py \
  finalize_icsps2026_sarbm3d_raw.py \
  evaluate_icsps2026_sarbm3d.py \
  icsps2026_pretest.py \
  seal_icsps2026_pretest.py \
  verify_icsps2026_pretest_gate.py \
  summarize_icsps2026_synthetic.py \
  summarize_icsps2026_multiseed.py \
  summarize_icsps2026_real.py \
  export_icsps2026_ucm_figures.py \
  export_icsps2026_real_figures.py \
  verify_icsps2026_artifact.py \
  prepare_icsps2026_real.py

python -m unittest discover -s tests -p 'test_*.py' -v

echo "Preflight passed. This checks software/model/protocol code, not dataset completeness."
