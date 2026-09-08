# SAR-BM3D 本地复现（Windows / MATLAB）

本地使用 GRIP-UNINA 原作者发布的 **SAR-BM3D v1.0 Windows x64** 软件包，调用原始 `SARBM3D_v10` 和随包 MEX。算法参数和作者文件保持原样。无需训练、GPU、PyTorch 或 Python 环境。

## 来源与版本

- 论文：S. Parrilli, M. Poderico, C. V. Angelino, L. Verdoliva, *A Nonlocal SAR Image Denoising Algorithm Based on LLMMSE Wavelet Shrinkage*, IEEE TGRS 50(2), 606–616, 2012。[DOI](https://doi.org/10.1109/TGRS.2011.2161586)
- [作者团队算法介绍](https://www.grip.unina.it/remote-sensing/despeckling)
- [官方 v1.0 下载目录](https://www.grip.unina.it/download/prog/SAR-BM3D/version_1.0/)、[Windows x64 压缩包](https://www.grip.unina.it/download/prog/SAR-BM3D/version_1.0/SARBM3D_v10_win64.zip)、[官方 README](https://www.grip.unina.it/download/prog/SAR-BM3D/version_1.0/README.txt)
- 版本发布日期：2013-07-31。压缩包 SHA-256：`6bad387f5b085cb01a0559064812de6e667bd0b0cc784b4611f03520594e7da6`。

该软件公开下载，但核心为编译好的 MEX，并非完整开放源代码。随包 `LICENSE.txt` 限非营利用途，禁止分发软件，公开使用结果时应引用原作者。官方压缩包和解压目录仅保存在被 Git 忽略的 `external/sarbm3d/`，本项目只添加独立调用脚本。

## 已准备的本地环境

- MATLAB R2024a，64 位 Windows。
- MATLAB 程序：`D:\Program Files\MATLAB\R2024a\bin\matlab.exe`。
- 官方软件：`external/sarbm3d/SARBM3D_v10_win64/`。
- OpenCV 2.1 DLL 随作者包提供，运行入口只设置当前子进程的搜索路径。
- VC++ 2010 x64 运行库已在本机找到；最终兼容性以实际 MEX 运行结果为准。
- `external/sarbm3d/provenance.json` 保存下载地址、压缩包哈希和每个作者文件的哈希。

## 本次实际验证（2026-09-08）

| 样例 | 设置 | 含噪 → 去斑 PSNR | 算法耗时 |
| --- | --- | --- | --- |
| 官方 Napoli | 256×256，L=1，幅度域，峰值 255 | 14.20 → 23.07 dB | 11.21 秒 |
| 固定种子合成图 | 128×128，L=4，强度域，范围 1 | 14.74 → 26.62 dB | 2.77 秒 |
| 本地真实 SAR MAT | 256×256，调用参数 L=1 | 无干净参考，不计算 | 11.61 秒 |

这些是本次本机运行值，时间仅包含算法推理，不含 MATLAB 启动。两行 PSNR 使用不同数值域，不宜相互比较。

- 官方样例结果：`output/sarbm3d_local/official_napoli_L1/`，另有带标签的幅度对比图 `comparison_labeled.png`。
- 合成样例结果：`output/sarbm3d_local/synthetic_L4/`。
- 真实样例结果：`output/sarbm3d_local/real_sample_L1/`；输入是 `datasets/real_sar_dataset/test_selected/all/01_homogeneous_2560_4096_y0_x256.mat`。这里用 L=1 验证本地数据流程，没有估计该图实际视数。
- `output/sarbm3d_local/verification.json` 保存独立数值核验和运行脚本哈希；`validation.log` 记录格式/数值域一致性、固定种子重复性及拒绝覆盖检查。

18 个解压文件与官方压缩包逐字节一致。所有主要结果尺寸正确且数值有限；独立重算 PSNR 与 MATLAB 相符；同一数值的 TIFF/MAT、幅度/强度入口输出一致；固定种子重复运行结果完全一致。

## 运行

在项目根目录 `D:\research\sar_transformer-main` 打开 PowerShell。

**运行作者自带 Napoli 样例：**

```powershell
.\scripts\sarbm3d\run_windows.ps1
```

默认读取作者的 `napoli.raw` 与 `napoli_noisy.raw`，大小为 256×256，L=1。直接使用原始幅度数据，不重新生成噪声。每次自动创建新的输出目录，避免覆盖已有结果。

**使用固定随机种子的合成样例，测试其他视数：**

```powershell
.\scripts\sarbm3d\run_windows.ps1 -Demo synthetic -Looks 4
```

128×128 强度图乘以 `Gamma(L,1/L)` 噪声，固定种子 20260908。噪声图可能超过 1，计算过程保留这些值。

**处理自己的 MAT 文件：**

```powershell
.\scripts\sarbm3d\run_windows.ps1 -InputPath 'D:\data\sample.mat' -MatField noisy -InputDomain intensity -Looks 1
```

**处理幅度图像：**

```powershell
.\scripts\sarbm3d\run_windows.ps1 -InputPath 'D:\data\amplitude.tif' -InputDomain amplitude -Looks 1
```

MAT 支持指定的顶层二维数值变量；PNG/TIFF 等图像通过 MATLAB `imread` 读取。灰度整数图、浮点图和 MAT 都保留原始数值尺度，仅转为双精度，输出单位与输入一致。RGB 图像转灰度，索引图按调色板转灰度；有 SAR 数值数据时优先使用 MAT 或浮点灰度 TIFF。当前入口要求图像至少 64×64，输入非负且有限。dB 图需要事先还原为线性值。

`-Looks` 是斑点噪声视数 L，取有限数且 L≥1。真实 SAR 应从产品信息或均匀区估计获取 L；默认 1 是调用默认值，不代表入口自动估计了视数。

可以用 `-OutputDir` 指定目录，用 `-MatlabExecutable` 指定 MATLAB 程序，用 `-PackageRoot` 指定另一个官方 Windows 软件目录。已有结果文件会触发拒绝覆盖。

## 输出与数值约定

默认输出在 `output/sarbm3d_local/run_时间戳/`：

| 文件 | 内容 |
| --- | --- |
| `result.mat` | 完整精度的输入与输出强度/幅度；有参考图时包含参考强度 |
| `input.png`、`denoised.png` | 使用同一显示范围的输入与去斑预览 |
| `comparison.png` | 有参考图时：参考、含噪、去斑；其他输入：含噪、去斑 |
| `summary.json` | MATLAB 版本、输入域、L、运行时间、数值范围、指标与显示参数 |
| `matlab_*.log` | 运行日志 |

作者接口明确输入/输出均为 **square root intensity（幅度）**。强度数据的调用为：

```matlab
denoised_intensity = SARBM3D_v10(sqrt(noisy_intensity), L).^2;
```

科学数值数组不裁剪。PNG 是显示预览，会按统一范围裁剪到可显示范围，不能用于替代原始数据计算指标。

官方 Napoli 样例另外报告幅度域 PSNR（峰值 255），便于对照作者示例。强度域 PSNR 的数据范围为 255²；合成示例的强度域数据范围为 1。指标域和 data range 记录在 JSON 中。真实输入没有干净参考图时不生成 PSNR/SSIM。

## 重新安装或验证来源

```powershell
.\scripts\sarbm3d\setup_windows.ps1
```

脚本从作者服务器下载并核对固定 SHA-256；已有软件时逐一验证作者文件与压缩包一致。它不会修改 MATLAB 安装目录。首次运行或 MATLAB 冷启动可能明显长于 JSON 中记录的算法推理时间。

本地入口独立于 `scripts/icsps2026/` 的既有冻结实验流程。这里的演示指标用于验证本地复现；论文完整数据集结果需按原实验协议另行运行。
