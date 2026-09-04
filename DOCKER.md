# SAR denoising Docker environment

## Why this image differs from `environment.yml`

The historical environment pins Python 3.6, PyTorch 1.7.1 and CUDA 11.0-era packages. The current workstation has an RTX 4070 Laptop GPU (Ada, compute capability 8.9), so that legacy stack is not a safe GPU runtime for new experiments.

This container uses PyTorch 2.2.2 with CUDA 12.1 and dependency versions that preserve the APIs used by the repository. `mmcv` is the 1.x lite package because the repository imports only `mmcv.cnn.ConvModule` and does not call compiled MMCV operators.

## One-time Windows prerequisite

Docker Desktop GPU containers require the WSL2 backend. If WSL reports `HCS_E_SERVICE_NOT_AVAILABLE`, open an Administrator PowerShell and run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
& 'D:\research\sar_transformer-main\docker\enable-wsl2-admin.ps1'
```

Restart Windows afterwards, then start Docker Desktop.

## Build and verify

From the repository directory:

```powershell
docker compose build sar
docker compose run --rm sar python docker/verify_environment.py
```

Successful verification must report:

- `cuda_available: true`
- `gpu: NVIDIA GeForce RTX 4070 Laptop GPU`
- input and output shape `[1, 1, 256, 256]`
- `output_finite: true`

The source tree is mounted at `/workspace`. Large datasets, results and checkpoints are excluded from the image build context but remain available through the bind mount when the container runs.

## Verified installation

Verified on 2026-09-03 with image `sar-denoise:cuda12.1`:

- Docker Desktop 4.41.2 / Engine 28.1.1
- Python 3.10.14
- PyTorch 2.2.2 / torchvision 0.17.2
- CUDA runtime 12.1
- NVIDIA GeForce RTX 4070 Laptop GPU
- Existing Base checkpoint loaded with zero missing or unexpected keys
- `1 × 1 × 256 × 256` model forward passed with finite output

Machine-readable results are stored in `docker/environment-verification.json`.
