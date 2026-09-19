# CL-SAR 本地预训练推理

2026-09-18 已完成作者正式权重的本地接入、Windows 命令入口和小样本数值核对。没有训练模型、没有跑全量真实测试集，也没有复现论文主表或得到本项目真实图质量排名。

## 来源与固定版本

- 论文：*Contrastive learning for real SAR image despeckling*，ISPRS Journal of Photogrammetry and Remote Sensing 218 (2024), 376–391；[DOI](https://doi.org/10.1016/j.isprsjprs.2024.11.003)。
- 作者仓库：[YangtianFang2002/CL-SAR-Despeckling](https://github.com/YangtianFang2002/CL-SAR-Despeckling)。
- 固定提交：`b12129d1d3448750b9098b239397587eeb359857`。
- 原样保留在 `external/CL-SAR/`；安装及每次推理检查来源、提交和受控文件是否修改。
- 使用 `experiments/MDN1-default/models/net_g_latest.pth`，40,347,040 bytes，SHA-256：`94765b1e6dfb7584842dbad4b3683a566f77c5b3d85595dbb8913f90fef8c80b`。
- 严格加载 checkpoint 的 `params`，模型为原始 `MDN1`，配置取自作者 `options/real/MDN1-default.yml`，注册参数量 10,038,249。
- 该提交未见顶层 LICENSE；第三方源码与权重仅下载到忽略目录，不纳入主项目版本控制。

适配直接执行未修改的 `MDN1_arch.py` 和 `arch_util.py`。导入时使用临时 Python 包命名空间以避开训练框架的附带依赖；唯一导入兼容项为网络前向未调用的日志函数，未替换任何网络层或数值计算。原始源码里的 `sys.path` 修改在导入后也会恢复。

## 安装与运行

当前机器环境已配置：Python 3.12.13、PyTorch 2.5.1 / CUDA 12.4、RTX 4070 Laptop GPU、NumPy 2.4.3、Pillow 10.4.0、SciPy 1.17.0。`.venv-cl-sar/` 通过 `--system-site-packages` 复用 `D:/develop/Anaconda/envs/pytorch_gpu/python.exe` 的 PyTorch/CUDA，只在自身环境安装 SciPy；未修改基础环境。它依赖基础环境，不能单独搬走。

重新配置：

```powershell
Set-Location 'D:\research\sar_transformer-main'
& .\scripts\cl_sar\setup_windows.ps1
```

也可用 `-BasePython '完整路径\python.exe'` 指向已有可用 PyTorch 的 Python。安装脚本不会覆盖已有不同提交或已修改的第三方目录。

运行已知强度域的项目合成样本：

```powershell
& .\scripts\cl_sar\run_windows.ps1 `
  -InputPath 'datasets/NWPU_RESISC45_SAR_global_L_v2/val/L1/airplane/airplane_00003.mat' `
  -InputDomain intensity
```

处理已确认是幅度的二维数组：

```powershell
& .\scripts\cl_sar\run_windows.ps1 `
  -InputPath 'D:\data\confirmed_amplitude.npy' `
  -InputDomain amplitude -Device cuda
```

示例 `D:\data\confirmed_amplitude.npy` 需替换为自己的文件。MAT 默认字段为 `noisy`，其他字段用 `-MatField`；CPU 用 `-Device cpu`；输出位置用 `-OutputPath`，必须为新目录或空目录。Python 入口的参数对应 `--input`、`--input-domain`、`--mat-field`、`--device`、`--output`、`--scale`，`--help` 不会加载或下载模型。

**`InputDomain` / `--input-domain` 必填，没有默认值。** 当前真实 MAT 的上游转换仍不明，不能因其位于 `[0,1]` 就认定强度或幅度，也不能通过选择一个能运行的参数替代物理域确认。本次没有对这些域未知真实 MAT 运行质量比较。

## 数值域与作者的实际后处理

支持实数二维 MAT、NPY 和单通道灰度图片；不静默把 RGB 转为灰度。普通 MATLAB MAT 使用 SciPy 读取，v7.3/HDF5 不在此入口范围。输入先转换为 float32，文件数值单位保留，**8 位图片也不会自动除以 255**。

输入强度时，流程为 `原值 / scale → sqrt → 作者幅度归一化 → MDN1 → 作者截断 → 幅度反归一化 → 平方 → 乘回 scale`。输入幅度时，去掉开方和末尾平方；主输出与输入保持相同数值域及单位。

作者的幅度归一化有两个容易漏掉的细节，均已保留：

1. 最小值是 **非零幅度的最小值**；分母为最大幅度减该值。
2. 归一化结果为 `abs((amplitude - min_nonzero) / (max - min_nonzero))`。零像素因此也不必映射为零，并且极端输入可以出现大于 1 的网络输入；入口不会擅自截断它。

全零图、只有单一正幅度的图会令作者归一化未定义，入口明确拒绝，不补造 epsilon 或替换成恒等输出；负值、复数、非二维和非有限输入也会拒绝。

原始 [`predict.py`](https://github.com/YangtianFang2002/CL-SAR-Despeckling/blob/b12129d1d3448750b9098b239397587eeb359857/basicsr/predict.py)先 `clamp(-20,20)`，随后调用 [`tensor2img`](https://github.com/YangtianFang2002/CL-SAR-Despeckling/blob/b12129d1d3448750b9098b239397587eeb359857/basicsr/utils/img_util.py)；后者默认把归一化幅度再次截断到 `[0,1]`，即便要求 float32 输出也一样。本入口的 `denoised.npy` 保留这个**作者推理后处理**，并在 `run.json` 记录截断比例；不将其误称为完全未截断的网络预测。`network_raw.npy` 单独保存截断前的归一化幅度网络输出。

`-Scale` 为已知输入单位的正除数，默认 1，输出乘回该值。PNG 始终使用所有面板共同的 `[0,scale]` 显示范围；不采用逐图最大值、分位数或作者用于显示的 `_view()` 独立拉伸。例如已确认物理含义的 8 位幅度文件可用 `-InputDomain amplitude -Scale 255`，浮点输出仍为原来的 0–255 单位。`run.json` 记录落在显示范围外的比例；PNG 显示裁剪不改变科学数组。

每次结果包括：

- `denoised.npy`：二维 float32，作者后处理结果，与输入同域、同单位。
- `network_raw.npy`：二维 float32，归一化幅度域网络原始输出，尚未截断或反归一化，不能直接与输入做差。
- `result.mat`：`noisy`、`denoised`、`residual`、`input_domain`、`output_domain` 和原始归一化幅度网络输出。
- `denoised.png`、`comparison.png`：同范围预览。
- `run.json`：输入文件/源码/权重 SHA-256、固定提交、域、归一化/截断规则、输出范围、设备及耗时。

## 本机验证记录

执行：

```powershell
& .\scripts\cl_sar\verify_windows.ps1 -Device cuda
```

输出 `output/cl_sar_local/verification.json`。核对脚本从固定版本原文件提取作者 `max_normalize`、`max_denormalize`、`normalizedAmp2intensity`、`tensor2img` 的函数定义，直接执行这些原始定义，并直接调用未修改的 MDN1 作为参考；没有声称原始 BasicSR 完整命令行或训练环境已跑通。

2026-09-18 的验证结果：

| 核对 | 结果 |
| --- | --- |
| 项目 256×256 合成强度 MAT 的作者预处理与本地输入 | 逐元素完全相同 |
| 作者直接 MDN1 与本地网络原始输出 | 最大绝对误差 0 |
| 作者后处理与 `denoised` 强度结果 | 最大绝对误差 0 |
| 幅度入口不再次开方；幅度输出平方与强度输出一致 | 最大绝对误差 0 |
| 47×65 含零非方形输入、原始 padding/crop | CPU/CUDA 均通过，作者流程核对误差 0 |
| 47×65 样例 CPU 与 CUDA 强度结果 | 最大绝对误差 6.5565×10⁻⁷ |
| 人工覆盖 `[-20,20]` / `[0,1]` 截断边界 | 与作者函数相同 |
| 验证前后第三方受控文件 | 均未修改 |

此外已实际执行 Windows 运行入口，处理 `airplane_00003.mat`，保存于 `output/cl_sar_local/project_synthetic_intensity/`。首轮单次 GPU 前向耗时约 0.762 s，作者 `[0,1]` 截断涉及 0.2365% 像素。此耗时含首次网络执行开销，不含文件读取/模型加载，不是稳定速度基准。

这些结果验证适配的数值路径与工程可用性，不证明在本项目真实 SAR 上的去斑质量或公平排名。当前仅接入默认正式模型，不配置全量数据下载、训练、分块推理、其他权重变体或任意传感器的标定转换。
