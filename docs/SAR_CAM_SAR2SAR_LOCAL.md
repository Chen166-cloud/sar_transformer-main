# SAR-CAM 与 SAR2SAR 本地运行

这两个方法都是需要训练权重的神经网络。本次分别核对作者代码和权重来源，再配置本地运行入口。SAR-CAM 使用用户确认的短训练方案；SAR2SAR 使用作者发布的权重。短训练用于验证流程，不能作为论文性能复现。

## 作者来源

### SAR-CAM

- [官方 GitHub](https://github.com/JK-the-Ko/SAR-CAM)
- [论文 DOI：10.1109/JSTARS.2021.3132027](https://doi.org/10.1109/JSTARS.2021.3132027)，Ko and Lee，IEEE JSTARS 15，3–19，2022。
- 固定源码提交：`ea5ee3bed00ab22735a7c87518fe5388c2d6c49a`。
- 本地作者源码：`external/SAR-CAM/`，MIT 许可。
- 截至本次检查，作者 README 仍将预训练模型列在 Future work，发布页仅有源码包。因此这里产生的权重明确标记为本地短训练，不冒充作者预训练模型。

### SAR2SAR

- [作者 GitHub 迁移说明](https://github.com/emanueledalsasso/SAR2SAR)
- [Télécom Paris 官方 GitLab](https://gitlab.telecom-paris.fr/RING/SAR2SAR)
- [论文 DOI：10.1109/JSTARS.2021.3071864](https://doi.org/10.1109/JSTARS.2021.3071864)，Dalsasso, Denis and Tupin，IEEE JSTARS 14，4321–4329，2021。
- 固定源码提交：`ca3c783333cbe072743f0b8060b5f168b9f0db11`。
- 官方 Single-Look 包：`network_weights/SAR2SAR-test.zip`，SHA-256：`70d03e0b664c45b9ddc4a31d67990c550cc51feb0aee68baef3a31c45f423bd3`。
- 本地作者文件：`external/SAR2SAR/`，GPLv3 许可。

## 数据含义

SAR-CAM 本次训练使用项目现有 NWPU 全局 L v2 的归一化强度配对数据。输入的数值域必须与训练权重一致；此流程不进行对数或开平方变换。原模型已在网络内部实现残差连接，输出就是恢复图像，不应再从输入中减一次。

SAR2SAR 的官方 Single-Look 测试说明要求 Sentinel-1 单通道**幅度**数组。它的预处理、对数归一化和逆变换应与官方权重配套。项目中预先归一化、裁剪到 [0,1] 的显示数据不能直接视为保留原始幅度标定的 Sentinel-1 数据。官方还提供独立 GRD 示例，本次 Single-Look 权重不自动代表 GRD 模式。

两个方法的本地入口及输出独立于项目现有冻结论文实验，不改变其数据划分、已有权重或历史结果。

## SAR-CAM：本地短训练与推理

环境位于 `.venv-sar-cam/`，继承本机 `pytorch_gpu` 环境的 PyTorch 2.5.1 / CUDA 12.4，再单独安装 MAT 文件所需的 SciPy。GPU 为 RTX 4070 Laptop 8 GB；原 Python 环境不被改写。

从任意 PowerShell 目录运行本次训练好的模型：

```powershell
& 'D:\research\sar_transformer-main\scripts\sar_cam\run_windows.ps1'
```

默认读取已留出的 `val/L1/airplane/airplane_00003.mat`，从 `output/sar_cam_local/training_demo/checkpoint_best.pth` 严格加载权重，在新的时间戳目录保存结果。有干净参考图时计算强度域 PSNR（data range=1，预测不裁剪）。

处理自己的归一化强度 MAT：

```powershell
& 'D:\research\sar_transformer-main\scripts\sar_cam\run_windows.ps1' -InputPath 'D:\data\sample.mat' -MatField noisy
```

PNG/TIFF 整数图按像素类型最大值归一化；MAT/NPY 保留数值并要求处于 [0,1]。对于尚未归一化的真实强度图，可显式加 `-Normalization percentile` 使用 1%/99% 分位数归一化，此时保存的输出是归一化强度，不恢复绝对辐射尺度。幅度图或 dB 图应在调用前转换为线性强度。

使用其他兼容权重时传 `-Weights`。普通原始 state_dict 还需要 `-CheckpointMetadata` 指定 JSON，其中记录源码提交、构造参数、数值域、权重 SHA-256 和训练来源。入口不会静默使用随机权重，也不会放宽网络参数匹配。

重新进行短训练：

```powershell
& 'D:\research\sar_transformer-main\scripts\sar_cam\train_windows.ps1' -Updates 200
```

采用未修改的原作者 SAR-CAM 网络（128 通道，3,317,284 个参数）和 DG+TV 损失，Adam 学习率 1e-4、weight decay 1e-5、TV 系数 2e-4；batch=2、64×64 随机裁剪，固定种子 20260908。从现有 NWPU v2 的训练划分随机选 512 对图，验证划分独立选 8 对图并使用中心裁剪。此短训练采用固定学习率，未执行论文的完整数据和训练日程。

训练目录保存 `checkpoint_best.pth`、`checkpoint_last.pth`、逐步日志、指标 CSV、训练/验证文件哈希清单和摘要。推理目录保存 `result.mat`、`prediction.npy`、输入/输出 PNG、对比图和 `summary.json`。PNG 仅为显示预览，完整预测数组不裁剪。

增加训练量并从本次的最后状态继续：

```powershell
& 'D:\research\sar_transformer-main\scripts\sar_cam\train_windows.ps1' -Resume 'D:\research\sar_transformer-main\output\sar_cam_local\training_demo\checkpoint_last.pth' -Updates 10000
```

这里的 `Updates` 是累计目标步数，示例从第 201 步继续到 10000 步。续训恢复优化器和随机数状态，要求样本选择、字段、种子、裁剪、batch 和验证间隔保持一致；仍使用本地适配器的固定学习率日程。需要更大数据子集或不同日程时应启动新的训练配置。完整作者训练脚本也保留在 `external/SAR-CAM/train.py`，其目录输入、依赖和 epoch 学习率日程见作者 README。

当前适配器会预读选定的 MAT 样本；默认 512 对约占 256 MB 数组内存。增加步数会继续训练这一子集；大幅增加样本数前需评估内存，不应直接将全部 NWPU 配对预读进笔记本内存。连续 4 步与 2+2 步续训已实测权重、Adam 状态和随机数状态一致；详细检查见 `output/sar_cam_local/resume_qa/summary.json`。

本次结果：独立 8 个 64×64 验证裁剪的平均 PSNR 为 **20.4064 → 28.0302 dB**；另一个完整 256×256 验证图为 **12.8822 → 20.3658 dB**。前者参与短训练的模型选择，后者用于完整图推理演示；都不是论文完整测试集结果。实际输出位于 `output/sar_cam_local/validation_sample/` 和 `output/sar_cam_local/real_sample/`。

运行环境需要重新检查时：

```powershell
& 'D:\research\sar_transformer-main\scripts\sar_cam\setup_windows.ps1'
```

## SAR2SAR：官方预训练权重推理

原作者软件基于 TensorFlow 1.13.1。本地独立环境 `.venv-sar2sar/` 使用 TensorFlow 2.16.2 的 `compat.v1` 在 Windows CPU 上读取作者保存的 `.meta` 网络结构和原始 checkpoint，不重新训练，也不重写 U-Net。依赖版本和安装包哈希已锁定，安装包的 SHA-256 对照官方 PyPI 元数据核验。

运行作者附带的 Sentinel-1 幅度样例：

```powershell
& 'D:\research\sar_transformer-main\scripts\sar2sar\run_windows.ps1'
```

默认对作者样例取中心 256×256 区域，便于快速确认环境。处理整幅作者样例可加 `-CropSize 0`。

本次已成功读取官方 `lely.npy`（500×500），对中心 256×256 区域完成推理，CPU 耗时约 **0.40 秒**（含网络图加载及权重恢复，不含 Python/TensorFlow 启动）。结果位于 `output/sar2sar_local/official_demo/`，有原始幅度/强度数值和去斑对比图。

完整 500×500 图像也已通过 25 个重叠块处理，耗时约 **2.75 秒**，结果在 `output/sar2sar_local/official_full/`。本地适配器与实际作者 `model.denoiser.test`、`u_net.py`、`utils.py` 在同一个 TensorFlow 兼容环境中对照：单块和 25 块输出均逐值一致，最大幅度误差为 **0.0**。作者文件保持原样；对照运行只在内存中适配旧版 TensorFlow 接口。完整记录见 `output/sar2sar_local/parity_validation/verification.json`。

处理自己的原始幅度 NPY（默认处理整幅输入）：

```powershell
& 'D:\research\sar_transformer-main\scripts\sar2sar\run_windows.ps1' -InputPath 'D:\data\sentinel1_amplitude.npy' -InputDomain amplitude
```

处理自己的线性强度 MAT（内部先开平方）：

```powershell
& 'D:\research\sar_transformer-main\scripts\sar2sar\run_windows.ps1' -InputPath 'D:\data\sentinel1_intensity.mat' -MatField noisy -InputDomain intensity
```

自定义图像不自动归一化或改变辐射数值尺度。支持二维 NPY、MAT 和灰度 TIFF/PNG；要求非负有限数值，至少 256×256。输入应符合官方 Single-Look Sentinel-1 模型的成像与数值条件。可显式通过 `-CropSize` 选择中心裁剪，`-Stride` 调整滑窗步长（默认 64，块大小 256），`-Threads` 调整 CPU 线程数。

作者的精确预处理为 `normalize_sar(A)/255`，其中 `normalize_sar` 先将幅度下限裁到 0.24，再取 log，用固定常数 `m=-1.429329123112601`、`M=10.089038980848645` 归一化。各块的原始网络输出在归一化对数域累计平均，然后按作者 `denormalize_sar` 将结果裁到 [0,1] 并反变换为幅度。本地保留该裁剪和运算顺序，不能改为先指数还原各块再平均。

输出保存在 `output/sar2sar_local/` 的独立目录，包括完整幅度/强度 `result.mat`、输入/输出 PNG、对比图、来源哈希与处理参数 JSON。预览图采用同一幅度显示范围；真实样例没有干净参考图，因此不报告 PSNR/SSIM。

重新检查运行环境和官方文件：

```powershell
& 'D:\research\sar_transformer-main\scripts\sar2sar\setup_windows.ps1'
```

在 Windows/Python 3.12 上使用锁定依赖文件；其他系统或 Python 版本应另建相应环境。作者文件清单和哈希位于 `external/SAR2SAR/provenance.json`，依赖验证位于 `external/SAR2SAR/dependency-verification.json`。
