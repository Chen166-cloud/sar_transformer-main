# Umbra 三场景 SAR 去噪稳定性实验报告

实验日期：2026-09-21  
实验场景：釜山港、曼谷素万那普机场、纽瓦克港  
对比方法：SAR-BM3D、SAR2SAR、SDUDNet、Trans-SAR、CL-SAR、MERLIN、MuLoG-DRUNet

## 1. 实验目的与结论摘要

本实验在与论文 Fig. 3 相同来源的 Umbra Open Data 上增加 3 幅真实单视复数 SAR 场景，用同一套 7 方法适配器、相同的单视设置（`L=1`）和相同的 1024×1024 ROI 尺寸检查跨场景稳定性。

主要结论如下：

1. 三个场景的 SICD 复数数据、GEC 产品和元数据均已下载到 E 盘；SICD 均通过 multipart ETag 和 SHA-256 校验。
2. 三个场景共 21 次方法推理全部完成，主图均按 Fig. 3 的 2×4 排版输出为 2091×1345 像素、600 dpi，同时保留单图、ratio 图、科学数组和严格统一灰度版本。
3. 在没有无斑真值的前提下，不能报告可信的 PSNR/SSIM。本文采用视觉对比和 `10log10(noisy/output)` ratio 诊断讨论稳定性，不把这些无参考量解释为绝对复原质量。
4. SAR-BM3D 的跨场景中位 ratio 变化范围为 0.13 dB，视觉表现也较一致，是本组实验中最稳妥的传统基线，但平均推理时间约 208 s。
5. CL-SAR 和 SAR2SAR 对强散射点、道路及建筑轮廓保留较多；SAR2SAR 在部分区域可见结构化残差或孤立暗点。CL-SAR 的辐射偏移随场景变化大于 SAR-BM3D、SAR2SAR 和 SDUDNet。
6. Trans-SAR 的中位 ratio 跨场景范围达到 6.87 dB，且在三图中均有较明显的平滑，说明其由合成灰度训练域迁移到 Umbra X 波段 SICD 时稳定性不足。
7. MERLIN 原始科学输出存在约 11.50 dB 的平均中位亮度偏移，并表现出较强平滑；主图中的 MERLIN 仅为显示目的做中位数对齐，科学数组与 ratio 指标未被改动。
8. MuLoG-DRUNet 的中位 ratio 最稳定（范围 0.07 dB），但在纽瓦克低回波水面/港区和曼谷大面积停机坪上出现块状或分段式过平滑，因此“辐射稳定”不能等同于“视觉质量最佳”。

综合建议：论文正文若需要强调跨场景稳定性，可把 SAR-BM3D作为稳定传统基线，把 CL-SAR 和 SAR2SAR作为细节保留对照；Trans-SAR、MERLIN 和 MuLoG-DRUNet 的域偏移或过平滑现象应在讨论中如实说明。

## 2. 数据来源与真值说明

数据来自 [Umbra Open Data](https://umbra.space/open-data/)，授权为 CC BY 4.0。三幅图均为 X 波段、Spotlight、VV 极化的真实单视 SICD 复数数据；GEC 只用于数据核对和定位，没有作为去噪真值。

| 场景 | 获取日期 | 标称分辨率 | 轨道/侧视 | 入射角 | SICD 大小 | 场景内容 |
|---|---:|---:|---|---:|---:|---|
| Busan Port | 2025-05-21 | 0.35 m | ascending / left | 43.8° | 7.066 GB | 港池、船舶、道路与密集建筑 |
| Bangkok Suvarnabhumi Airport | 2025-10-31 | 0.25 m | descending / left | 43.9° | 12.522 GB | 停机坪、航站楼、飞机/廊桥、滑行线与低回波铺装 |
| Newark Port | 2025-05-03 | 0.35 m | ascending / left | 44.2° | 6.639 GB | 港口水道、集装箱区、道路与城市结构 |

SICD 中记录的像素采样和冲激响应宽度如下。它们比“0.25 m/0.35 m”标称档位更适合复现实验时使用。

| 场景 | 行/列采样间隔 (m) | 行/列冲激响应宽度 (m) | 完整影像尺寸 |
|---|---|---|---|
| Busan | 0.2187 / 0.1511 | 0.2422 / 0.1674 | 19008×46464 |
| Bangkok | 0.1566 / 0.1118 | 0.1735 / 0.1237 | 26620×58800 |
| Newark | 0.2204 / 0.1548 | 0.2440 / 0.1714 | 19008×43659 |

### 地面真值边界

Umbra 没有为这些真实场景提供同一时刻、同一几何条件下的无斑 reflectivity 真值。GEC 是同一含斑观测的地理编码显示产品，并非 clean ground truth；多时相平均也只能在配准、辐射归一化和变化掩膜后作为伪参考，不能写成真实地面真值。因此本报告不计算 PSNR、SSIM，也不把 ratio 统计当成全参考质量分数。

## 3. ROI 与实验协议

ROI 只根据 noisy 输入预览选择，未查看任何方法输出后再调整位置。

| 场景 | ROI 起点 (row, col) | 尺寸 | 中心经纬度 | 选择理由 |
|---|---|---:|---|---|
| Busan | (8992, 22720) | 1024×1024 | 35.105058, 129.082199 | 密集港区和城市结构，兼有强散射与低回波区域 |
| Bangkok | (12798, 28888) | 1024×1024 | 13.690679, 100.750020 | 航站楼、飞机/廊桥、滑行线及大面积低回波停机坪 |
| Newark | (8992, 5311) | 1024×1024 | 40.700749, -74.137853 | 港口水道、集装箱区、道路及低回波水面 |

统一科学输入为

```text
I = real(S)^2 + imag(S)^2
```

其中 `S` 为原始 SICD complex64 ROI，`I` 为 float32 线性强度。未增加辐射定标、空间重采样或每块独立归一化。

### 方法与适配器

| 方法 | 使用版本/权重 | 输入与关键设置 | 备注 |
|---|---|---|---|
| SAR-BM3D | 作者 MATLAB v1.0 | 强度→幅度，`L=1`，输出再平方为强度 | 官方 MEX，传统非局部基线 |
| SAR2SAR | 官方 TF checkpoint，commit `ca3c783` | 线性强度→幅度→官方 log 归一化，256 patch，stride 64 | CPU 推理；原模型面向单视 Sentinel-1 |
| SDUDNet | `real.pth`，commit `0c79991` | 固定 noisy dB 显示图经 uint8/255 输入 | 与 Fig. 3 已采用的公开 real-model 输入契约一致，科学辐射可比性有限 |
| Trans-SAR | TransSARV2，commit `b3ac845` | ROI 最大值固定缩放，幅度域，256 patch，无额外 8-bit 量化 | 官方权重由 BSDS 合成散斑训练，存在明显跨传感器域差异 |
| CL-SAR | MDN1-default，commit `b12129d` | 原始强度，官方幅度归一化与逆变换 | 官方预训练权重 |
| MERLIN | Spotlight checkpoint，commit `b9bf54a` | 直接输入复数实部/虚部，256 patch，stride 64，正 Hann 加权重叠 | 主图只做显示级中位数对齐；严格共享灰度图另存 |
| MuLoG-DRUNet | generic checkpoint，commit `f468573` | 强度，`L=1`，10 次迭代 | 官方 MuLoG + DRUNet 标量强度流程 |

显示采用 noisy 输入的 1% 和 99.7% dB 分位点作为同场景共享窗口。除主图中的 MERLIN 显示对齐外，没有对单个方法做独立拉伸。所有定量诊断均读取未做显示对齐的科学数组。

## 4. 实验环境

| 项目 | 配置 |
|---|---|
| 操作系统 | Windows 11 家庭中文版 |
| CPU | Intel Core i9-14900HX |
| 内存 | 31.6 GiB |
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU，8188 MiB |
| GPU 驱动 | 610.88 |
| MATLAB | R2024a |
| PyTorch（主要 CUDA 方法） | 2.5.1 + CUDA 12.4 |
| 图像尺寸 | 1024×1024 |

以下时间为各适配器记录的模型/算法推理时间，通常不含公共渲染和大部分进程启动时间，因此只用于同一机器上的量级比较。

| 方法 | 三场景平均时间 (s) | 最小–最大 (s) |
|---|---:|---:|
| SDUDNet | 0.37 | 0.35–0.37 |
| Trans-SAR | 1.07 | 0.97–1.15 |
| CL-SAR | 3.27 | 2.38–3.99 |
| MERLIN | 4.44 | 4.17–4.77 |
| MuLoG-DRUNet | 18.62 | 17.10–19.77 |
| SAR2SAR | 23.16 | 20.61–24.93 |
| SAR-BM3D | 207.81 | 197.08–221.27 |

## 5. 结果图

### 5.1 Busan Port

![Busan Port 7-method comparison](01_busan/experiment/figure3_style/figure3_style_7methods_600dpi.png)

釜山场景含密集建筑、道路、船舶和港池。SAR-BM3D 在均匀区域抑斑较稳定；CL-SAR 与 SAR2SAR 保留的细碎高频信息较多，其中 SAR2SAR 可见少量孤立暗点。Trans-SAR、MERLIN 和 MuLoG-DRUNet 的平滑更强。

### 5.2 Bangkok Suvarnabhumi Airport

![Bangkok airport 7-method comparison](02_bangkok/experiment/figure3_style/figure3_style_7methods_600dpi.png)

曼谷场景具有大面积低回波停机坪和细长滑行线，能明显区分平滑与结构保留。SAR-BM3D 保留主要直线和强散射结构；CL-SAR 对飞机/廊桥和建筑轮廓保留较多。SAR2SAR 保留较多细节但残差中存在规则结构；Trans-SAR、MERLIN 和 MuLoG-DRUNet 对小目标和弱线状结构的抑制更明显。

### 5.3 Newark Port

![Newark Port 7-method comparison](03_newark/experiment/figure3_style/figure3_style_7methods_600dpi.png)

纽瓦克场景同时包含水面、港区和道路。SAR-BM3D、SAR2SAR、SDUDNet 和 CL-SAR 在主要边缘位置保持相对稳定；MERLIN 和 Trans-SAR 的整体平滑较强。MuLoG-DRUNet 在低回波区域出现较明显的分段/块状外观，提示其在该场景中的正则化偏强。

## 6. Ratio 诊断与跨场景稳定性

ratio 定义为

```text
R_dB = 10*log10(noisy / output)
```

中位值为负表示输出中位强度高于 noisy；IQR 表示 ratio 中间 50% 的离散程度；相邻相关系数接近 0 表示残差在一阶邻域上更接近白化，但这些量均不能替代真值指标。

### 6.1 每场景诊断

| 场景 | 方法 | ratio 中位值 (dB) | ratio IQR (dB) | 相邻相关 |
|---|---|---:|---:|---:|
| Busan | SAR-BM3D | -1.28 | 6.04 | -0.02 |
| Busan | SAR2SAR | -2.68 | 6.32 | 0.04 |
| Busan | SDUDNet | 0.10 | 5.74 | -0.02 |
| Busan | Trans-SAR | -9.87 | 7.84 | 0.30 |
| Busan | CL-SAR | -2.11 | 7.67 | 0.14 |
| Busan | MERLIN | -11.01 | 8.14 | 0.35 |
| Busan | MuLoG-DRUNet | -1.62 | 6.88 | 0.06 |
| Bangkok | SAR-BM3D | -1.41 | 6.47 | 0.01 |
| Bangkok | SAR2SAR | -2.78 | 6.24 | -0.02 |
| Bangkok | SDUDNet | -0.13 | 6.42 | -0.01 |
| Bangkok | Trans-SAR | -3.00 | 6.85 | 0.05 |
| Bangkok | CL-SAR | -0.73 | 6.50 | 0.00 |
| Bangkok | MERLIN | -12.83 | 7.14 | 0.18 |
| Bangkok | MuLoG-DRUNet | -1.67 | 6.82 | 0.04 |
| Newark | SAR-BM3D | -1.38 | 6.34 | -0.01 |
| Newark | SAR2SAR | -2.26 | 6.34 | 0.00 |
| Newark | SDUDNet | -0.04 | 6.30 | -0.02 |
| Newark | Trans-SAR | -4.58 | 6.99 | 0.09 |
| Newark | CL-SAR | -1.11 | 6.94 | 0.04 |
| Newark | MERLIN | -10.65 | 7.41 | 0.23 |
| Newark | MuLoG-DRUNet | -1.68 | 6.87 | 0.05 |

### 6.2 跨场景汇总

| 方法 | 中位 ratio 平均值 (dB) | 三场景范围 (dB) | 平均 IQR (dB) | 平均相邻相关 |
|---|---:|---:|---:|---:|
| SAR-BM3D | -1.36 | 0.13 | 6.28 | -0.01 |
| SAR2SAR | -2.57 | 0.52 | 6.30 | 0.01 |
| SDUDNet | -0.02 | 0.23 | 6.16 | -0.02 |
| Trans-SAR | -5.82 | 6.87 | 7.23 | 0.15 |
| CL-SAR | -1.32 | 1.38 | 7.04 | 0.06 |
| MERLIN | -11.50 | 2.18 | 7.56 | 0.25 |
| MuLoG-DRUNet | -1.66 | 0.07 | 6.86 | 0.05 |

从中位 ratio 的跨场景范围看，MuLoG-DRUNet、SAR-BM3D、SDUDNet 和 SAR2SAR 的全局辐射行为较稳定；但 SDUDNet 使用显示域适配，MuLoG-DRUNet 又存在明显过平滑，因此不能仅按该列排序。SAR-BM3D 同时具备较小的跨场景偏移范围、接近零的一阶残差相关和较一致的视觉结果。Trans-SAR 的范围最大；MERLIN 的原始亮度偏移和残差空间相关均较明显。

## 7. 适合论文使用的表述

可在论文中写为：

> 为验证不同去斑方法在真实高分辨率 SAR 场景上的稳定性，进一步选取 Umbra Open Data 中的釜山港、曼谷素万那普机场和纽瓦克港三幅 X 波段 Spotlight VV 单视 SICD 图像进行测试。所有方法使用相同的 1024×1024 ROI 和 `L=1` 设置。由于真实观测不存在同场景无斑真值，本文不报告 PSNR/SSIM，而结合去斑图和 ratio 图进行定性分析。结果表明，SAR-BM3D 在三个场景中具有较一致的辐射响应和视觉表现；CL-SAR 与 SAR2SAR能够保留较多强散射结构；Trans-SAR 和 MERLIN 在跨传感器测试中表现出更明显的平滑或亮度偏移；MuLoG-DRUNet 虽具有稳定的全局 ratio 中位值，但在部分低回波区域出现过平滑。

不建议写“某方法在真实场景上获得最高 PSNR/SSIM”，也不建议把同一获取时刻的 GEC 或未经过变化掩膜的多时相平均称为 ground truth。

## 8. 输出文件与复现入口

数据根目录：

```text
E:\SAR_Data\Umbra\Stability_3Scenes
```

结果根目录：

```text
D:\research\sar_transformer-main\output\umbra_stability_3scenes
```

关键文件：

- `target_dataset_manifest.json`：三个目标场景的路径、大小和 SHA-256。
- `stability_diagnostics.csv`：21 组场景/方法 ratio 诊断，可直接导入 Excel。
- `stability_diagnostics.json`：诊断定义、逐场景结果、跨场景汇总与推理时间。
- 每个场景的 `roi/run.json`：SICD 元数据、ROI 坐标、输入尺度和输入文件校验值。
- 每个场景的 `experiment/experiment_state.json`：7 方法完成状态。
- 每个场景的 `experiment/figure3_style/manifest.json`：图像、ratio、科学数组和 SHA-256。
- 每个 `runs/<method>/run.json` 或 `summary.json`：代码版本、checkpoint、数值域、运行环境和用时。

复现脚本：

```text
scripts/prepare_umbra_stability_roi.py
scripts/run_umbra_stability_methods.py
scripts/render_umbra_stability_comparison.py
scripts/summarize_umbra_stability.py
```

## 9. 数据完整性

| 场景 | SICD SHA-256 | GEC SHA-256 |
|---|---|---|
| Busan | `c2bd8e247e6c79ff61970e559266249131bf2ac5f07bc38df86b67e0df8bbd33` | `d0c147ab2064f78324e3edba4dc3f3755e0a18904d705720a816ea4c67ea61f1` |
| Bangkok | `7c8132b05ea58874b0b8d53e9d9b05e886d9597c4ea69f8cadc1fd5a41e15985` | `943ff819a2d18c095ba3036ff89eb134cae4de2bf22ffd1d08b3fa6db6d803e3` |
| Newark | `b9c9185e2f147ae22ee602c86f5235cc1abdc7474fca7b675519516c2f5b43b9` | `a9cbb93f606f62819585f2a3be7a84e93e67ac23e1fe8f810674dbe38c10eaf7` |

## 10. 参考资料与代码

1. Umbra Open Data: <https://umbra.space/open-data/>
2. SAR-BM3D: Parrilli et al., *A Nonlocal SAR Image Denoising Algorithm Based on LLMMSE Wavelet Shrinkage*, DOI: <https://doi.org/10.1109/TGRS.2011.2161586>
3. SAR2SAR: Dalsasso et al., *SAR2SAR: a semi-supervised despeckling algorithm for SAR images*, DOI: <https://doi.org/10.1109/JSTARS.2021.3071864>; code: <https://gitlab.telecom-paris.fr/ring/sar2sar>
4. MERLIN: Dalsasso et al., *As if by Magic: Self-Supervised Training of Deep Despeckling Networks with MERLIN*, <https://arxiv.org/abs/2110.13148>; code: <https://github.com/hi-paris/deepdespeckling>
5. CL-SAR: Fang et al., *Contrastive learning for real SAR image despeckling*, DOI: <https://doi.org/10.1016/j.isprsjprs.2024.11.003>; code: <https://github.com/YangtianFang2002/CL-SAR-Despeckling>
6. Trans-SAR code: <https://github.com/malshaV/sar_transformer>
7. SDUDNet code: <https://github.com/BFY-official/SDUDNet>
8. MuLoG-DRUNet code: <https://gitlab.telecom-paris.fr/ring/mulog-drunet>

