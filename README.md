# Transformer based SAR image despeckling

This repository contains source code, configurations, documentation, and example
figures. Datasets, model checkpoints, generated experiment outputs, and local
transfer scripts are excluded from Git. Prepare the datasets and checkpoints
separately before training or evaluation.

For the original SAR-BM3D v1.0 on local Windows/MATLAB, see
[SAR-BM3D 本地复现说明](docs/SAR_BM3D_LOCAL.md).

For local SAR-CAM training/inference and SAR2SAR pretrained inference, see
[SAR-CAM 与 SAR2SAR 本地运行](docs/SAR_CAM_SAR2SAR_LOCAL.md).

For local SDUDNet inference with the authors' pretrained weights, see
[SDUDNet 本地运行](docs/SDUDNET_LOCAL.md).

For local CL-SAR and MuLoG-DRUNet inference with the authors' pretrained weights,
see [CL-SAR 本地运行](docs/CL_SAR_LOCAL.md) and
[MuLoG-DRUNet 本地运行](docs/MULOG_DRUNET_LOCAL.md).
Both adapters require an explicit intensity/amplitude input domain; MuLoG-DRUNet
also requires an explicit number of looks. Their local smoke tests do not change
the frozen ICSPS/GRSL evaluation protocol or establish a new performance ranking.
As of 2026-09-18, CL-SAR pretrained inference is verified; MuLoG-DRUNet's adapter
and offline tests are ready, but its official weight download and model inference
verification are still pending. See the method-specific document for status.

## Dataset location

All datasets are stored under [`datasets/`](datasets/README.md). Use
`datasets/<dataset-name>` for data paths; the original top-level compatibility
junctions have been removed. Docker uses `/workspace/datasets/` through the
project mount. See the dataset guide for each dataset's purpose.

> **Reproducibility update:** the historical commands below describe the upstream
> project only. `REPRODUCIBLE_EXPERIMENTS.md` documents the general historical
> cleanup, but **ICSPS 2026 formal runs must use only** the frozen ICSPS documents
> linked below and `scripts/icsps2026/`. In particular, do not train directly from
> `datasets/bsds500_synthetic_dataset/train` because that legacy directory contains
> official BSDS500 test images.


## Using the code:

The code is stable while using Python 3.6.13, CUDA >=10.1

- Clone this repository:
```bash
git clone git@gitee.com:chy66666/sar_transformer-main.git
cd sar_transformer-main
```

To install all the dependencies using conda:

```bash
conda env create -f environment.yml
conda activate sar
```

For the native Apple Silicon (`linux/arm64`) Docker environment, see
[APPLE_SILICON_DOCKER.md](APPLE_SILICON_DOCKER.md).

For the frozen ICSPS 2026 SAR-despeckling protocol and the Ubuntu 22.04 / dual
RTX 4090 server workflow, use
[docs/ICSPS2026_SAR去斑网络设计与实验方案.md](docs/ICSPS2026_SAR去斑网络设计与实验方案.md)
and
[docs/ICSPS2026_远程服务器运行手册.md](docs/ICSPS2026_远程服务器运行手册.md).
Those entry points target Python 3.12, PyTorch 2.5.1 + CUDA 12.4, and do not use
the historical Python 3.6 environment below.

If you prefer pip, install following versions:

```bash
timm==0.3.2
mmcv-full==1.2.7
torch==1.7.1
torchvision==0.8.2
opencv-python==4.5.1.48
```


## Creating synthetic data:
This network was trained synthetic SAR images generated using [BSD500](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/). To create the synthetic data use create_synthetic_data.py file.

## To  train the network:

```   
python train.py --batch_size 1 --epoch 400 --modelname "TransSARV2" --learning_rate 0.0002 --train_dataset "path_to_training_data" --val_dataset "path_to _validation_data" --direc "path_to_save_results" --crop 256
```

## To test the network:

```   
python test.py --loadmodel "./pretrained_models/model.pth" --save_path "./test_images/" --model "TransSARV2"
```
