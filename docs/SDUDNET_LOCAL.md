# SDUDNet 本地运行记录

已于 2026-09-08 在本机跑通作者发布的 **SDUDNet 网络和两套预训练权重**。本次完成预训练推理复现、项目数据接入和数值一致性检查；没有从头训练，也没有复现论文完整测试集的表格指标。

## 来源与版本

- 作者仓库：[BFY-official/SDUDNet](https://github.com/BFY-official/SDUDNet)。
- 对应论文：Fuyu Bo 等，*Speckle-Driven Unsupervised Despeckling for SAR Images*，IEEE JSTARS，2025；[IEEE 页面](https://ieeexplore.ieee.org/document/10999093/)，[DOI](https://doi.org/10.1109/JSTARS.2025.3568854)。本次实现依据作者公开代码；IEEE 页面访问时出现机器人验证。
- 固定提交：`0c799911e84ef79c5fbe6f58f44cb8ee033048a1`。
- 原样保留的代码：`external/SDUDNet/`；推理网络为 [`net/DNN.py`](https://github.com/BFY-official/SDUDNet/blob/0c799911e84ef79c5fbe6f58f44cb8ee033048a1/net/DNN.py)。
- 作者自带权重：`models/real.pth`、`models/synthetic.pth`。未训练或替换这些权重。
- 该提交未附 LICENSE 文件；本项目仅保留获取入口，外部源码和权重没有加入主项目 Git。

| 权重 | SHA-256 |
|---|---|
| real.pth | `3e1b9bc4eaaa01a81544e9b96c5e238415f7fd8af089881f3ae19e3469b7e69d` |
| synthetic.pth | `3d9c28c80fad49510240b9ed7fd589049870f0e789edd5ac5382147903a9486c` |

## 已配置的环境

- Windows，Python 3.12.13。
- PyTorch 2.5.1，CUDA 12.4，NVIDIA GeForce RTX 4070 Laptop GPU。
- NumPy 2.4.3、Pillow 10.4.0、SciPy 1.17.0。
- 入口：`.venv-sdudnet/Scripts/python.exe`。该环境通过 `--system-site-packages` 复用 `D:/develop/Anaconda/envs/pytorch_gpu/python.exe` 所在环境的 PyTorch/CUDA，单独安装 SciPy。因此依赖这个基础环境，不是可以脱离基础环境搬走的独立安装。
- 推理入口不依赖 OpenCV 或图形窗口；数值核对脚本额外使用基础环境已有的 torchvision 0.20.1。

## 直接运行

在 PowerShell 中执行以下命令即可用 `real.pth` 处理作者随附的 `01233.jpg`。当前环境已准备好，无需再次安装。

```powershell
& 'D:\research\sar_transformer-main\scripts\sdudnet\run_windows.ps1'
```

默认自动选择 CUDA，结果写入新的 `output/sdudnet_local/run_时间戳/`。可以用 `-Device cpu` 指定 CPU。

复用本项目已有的合成验证样本：

```powershell
Set-Location 'D:\research\sar_transformer-main'
& .\scripts\sdudnet\run_windows.ps1 `
  -Model synthetic `
  -InputPath 'datasets/NWPU_RESISC45_SAR_global_L_v2/val/L1/airplane/airplane_00003.mat' `
  -CleanPath 'datasets/NWPU_RESISC45_SAR_global_L_v2/val/L1/airplane/airplane_00003.mat'
```

处理自己的真实 SAR 数据：

```powershell
& 'D:\research\sar_transformer-main\scripts\sdudnet\run_windows.ps1' `
  -Model real `
  -InputPath 'D:\data\your_sar.mat' `
  -MatField noisy
```

其中 `D:\data\your_sar.mat` 是需要替换的示例路径。输入支持单通道灰度图片、二维 `.npy` 和普通 MATLAB `.mat`；MATLAB v7.3/HDF5 不在当前入口支持范围内。使用 MAT 的其他字段时指定 `-MatField`，参考图字段用 `-CleanMatField`，默认分别是 `noisy` 和 `clean`。每次处理一幅图像。

### 数值尺度

1. 8 位灰度 PNG/JPG 按作者 `ToTensor` 处理，除以 255；不做逐图拉伸。
2. MAT/NPY 的数组值原样读取。本项目上述样本已经在 `[0,1]` 范围内，直接输入。
3. 网络要求输入处于 `[0,1]`。其他已知尺度使用 `-Scale` 显式除法归一化；结果数组会乘回这个尺度。例如原值在 `0..65535` 的 16 位图像可用 `-Scale 65535`。
4. `-Scale` 对输入及参考图共同生效；不能给两者采用不同的逐图最大值。若计算非单位尺度数据的 PSNR，还需指定其原始单位下的 `-DataRange`，例如 `-Scale 65535 -DataRange 65535`。
5. 不自动做开方、平方、对数、分位数归一化或输入裁剪。公开测试代码采用灰度图数值，未给出足以确定任意传感器物理强度/幅度标定的说明。
6. 输出可能略小于 0 或大于输入显示范围。`.npy`、`.mat` 和 PSNR 保留这些原始预测；PNG 仅以固定范围裁剪显示，不能代替数值结果参与评估。

## 实测结果

下列均为整幅 256×256 图像，FP32、无额外缩放、无裁剪、无分块推理。耗时仅为一次网络前向，包含该进程首次调用的开销，不含读取文件和模型加载，不作为正式速度基准。

| 样本 | 权重 | 输入 PSNR | 输出 PSNR | 前向耗时 | 结果目录 |
|---|---|---:|---:|---:|---|
| 作者随附 `01233.jpg` | real | 无参考图 | 无参考图 | 0.142 s | `output/sdudnet_local/official_real/` |
| 项目 NWPU L1 `airplane_00003.mat` | synthetic | 12.8822 dB | 18.8980 dB | 0.147 s | `output/sdudnet_local/project_synthetic/` |
| 项目真实 SAR `01_homogeneous_2560_4096_y0_x256.mat` | real | 无参考图 | 无参考图 | 0.144 s | `output/sdudnet_local/project_real/` |

PSNR 使用 `10*log10(data_range^2 / MSE)`，此处 `data_range=1`，针对原始浮点输出计算。这是本地样本的结果；官方权重的训练数据和本项目验证样本不构成同一评估协议，因此不能据此宣称达到论文指标或与此前本地训练的其他算法作公平排名。

每个结果目录包含：

- `denoised.npy`：二维 float32 原始去斑结果。
- `result.mat`：`noisy`、`denoised`、`residual`（输入减输出）；提供参考图时还保存 `clean`。
- `denoised.png`、`comparison.png`：固定显示范围的预览图，所有面板使用相同尺度。
- `run.json`：代码提交、源码与权重校验和、输入路径与校验和、尺度、设备、耗时及可用指标。

## 与作者代码的一致性

本地入口直接导入原始 `DNN` 类，严格加载完整权重并采用第一个返回值 `out` 作为去斑图；第二个返回值是 `x-out`，不会误用为结果。网络注册参数量为 579,552，其中包含作者声明但未参与前向计算的参数，故不是全部参数均参与计算的 FLOPs 依据。

核对命令：

```powershell
Set-Location 'D:\research\sar_transformer-main'
& .\.venv-sdudnet\Scripts\python.exe .\scripts\sdudnet\verify_local.py
```

核对记录为 `output/sdudnet_local/verification.json`，本次结果：

- 作者 `ToTensor` 与本地 JPEG 读取的浮点输入逐元素完全相同。
- 按作者 `test.py` 的计算步骤直接调用模型，对 real 和 synthetic 两套权重与本地入口比较，最大绝对误差均为 **0**。
- 作者测试保留训练模式，本地调用 `eval()`；实际 DNN 没有启用 BatchNorm/Dropout 等随模式改变的层，上述核对同时验证了两种模式在样本上的输出一致。
- 173×211 非方形图像在 CPU 和 CUDA 上均成功运行，二者最大绝对误差 `1.96695e-6`。
- 原始受版本控制的源码和权重保持未修改。

核对针对原始测试脚本中的计算步骤；没有声称原始 GUI 脚本直接执行成功。作者 [`test.py`](https://github.com/BFY-official/SDUDNet/blob/0c799911e84ef79c5fbe6f58f44cb8ee033048a1/test.py) 写死的 `my_datasets/S4_L1.png` 并未在该提交中提供，因此使用实际附带的 `01233.jpg` 进行数值核对。本地入口还移除了固定 256×256 reshape 和强制 CUDA；没有修改网络结构。

## 重新安装及训练范围

若需要重新配置本机运行环境：

```powershell
& 'D:\research\sar_transformer-main\scripts\sdudnet\setup_windows.ps1'
```

其他安装路径使用 `-BasePython '完整路径\python.exe'` 指向已有可用 PyTorch 的 Python。安装入口会获取固定提交并安装少量适配依赖，不会替换基础环境的 PyTorch。若选择不同 PyTorch/CUDA 版本，应重新运行数值核对。

作者训练入口是 `external/SDUDNet/train.py`，需要自行准备 `train/final/` 下的灰度光学图和 `train/real_sar/` 下的 SAR 图；这些训练数据并未包含在仓库内。公开脚本还需要 TensorBoard 等训练依赖，本次没有配置或启动完整训练。

还需区分“复现公开实现”与“重新建立论文训练过程”：公开 [`Generator.py`](https://github.com/BFY-official/SDUDNet/blob/0c799911e84ef79c5fbe6f58f44cb8ee033048a1/net/Generator.py) 的 `forward(x,y)` 创建了 `cat([x,y])`，随后卷积实际仅使用 `x`，所以传入的 `y` 不影响该生成器的输出。这是当前公开代码本身的行为，本次没有自行修补或推断论文原意。仓库也存在尚未得到可见回复的[实现完整性询问](https://github.com/BFY-official/SDUDNet/issues/1)。这些情况不妨碍运行所附去斑权重，但不能据此宣称从头训练已严格复现论文。
