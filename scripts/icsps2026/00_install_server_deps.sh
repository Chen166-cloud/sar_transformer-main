#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$project_root"

python - <<'PY'
import sys
import torch

if sys.version_info[:2] != (3, 12):
    raise SystemExit(f"Expected Python 3.12, found {sys.version}")
if torch.__version__.split("+")[0] != "2.5.1":
    raise SystemExit(f"Expected preinstalled PyTorch 2.5.1, found {torch.__version__}")
if torch.version.cuda != "12.4":
    raise SystemExit(
        f"Expected the image's PyTorch CUDA build to be 12.4, found {torch.version.cuda}. "
        "This value is distinct from the host driver's reported CUDA capability."
    )
print("Base Python/PyTorch/CUDA-build versions match the declared image.")
PY

# The server description guarantees torch but does not state that torchvision
# is installed.  Install only the official matching cu124 torchvision wheel
# when it is missing or has the wrong release; never replace torch here.
if ! python - <<'PY'
try:
    import torchvision
except Exception as error:
    print(f"torchvision import failed: {type(error).__name__}: {error}")
    raise SystemExit(1)
if torchvision.__version__.split("+")[0] != "0.20.1":
    print(f"Expected torchvision 0.20.1, found {torchvision.__version__}")
    raise SystemExit(1)
print(f"Matching torchvision is already installed: {torchvision.__version__}")
PY
then
  python -m pip install --no-cache-dir --force-reinstall --no-deps \
    --index-url https://download.pytorch.org/whl/cu124 \
    torchvision==0.20.1
fi

python - <<'PY'
import torch
import torchvision

if torch.__version__.split("+")[0] != "2.5.1" or torch.version.cuda != "12.4":
    raise SystemExit("torch changed while preparing torchvision; stop and inspect the environment")
if torchvision.__version__.split("+")[0] != "0.20.1":
    raise SystemExit(f"Matching torchvision installation failed: {torchvision.__version__}")
print(f"Verified torch={torch.__version__}, torchvision={torchvision.__version__}")
PY

torch_before=$(python -c 'import torch; print(torch.__version__)')
torchvision_before=$(python -c 'import torchvision; print(torchvision.__version__)')
constraints_file=$(mktemp)
cleanup() {
  rm -f -- "$constraints_file"
}
trap cleanup EXIT
printf 'torch===%s\ntorchvision===%s\n' "$torch_before" "$torchvision_before" > "$constraints_file"
python -m pip install --no-cache-dir --upgrade-strategy only-if-needed \
  --constraint "$constraints_file" \
  -r docker/requirements.server-py312.txt
python -m pip check
torch_after=$(python -c 'import torch; print(torch.__version__)')
torchvision_after=$(python -c 'import torchvision; print(torchvision.__version__)')
if [[ "$torch_after" != "$torch_before" || "$torchvision_after" != "$torchvision_before" ]]; then
  echo "Dependency installation unexpectedly changed torch/torchvision:" >&2
  echo "  torch: $torch_before -> $torch_after" >&2
  echo "  torchvision: $torchvision_before -> $torchvision_after" >&2
  exit 1
fi
trap - EXIT
cleanup

echo "Python 3.12 experiment dependencies installed without replacing PyTorch."
