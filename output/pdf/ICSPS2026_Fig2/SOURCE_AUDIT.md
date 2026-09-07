# Fig. 2 — FDR source and shape audit

核验日期：2026-09-07。范围仅为论文 Fig. 2 的 **Full-spectrum Fourier residual refinement (FDR)** 模块。本审计依据当前论文源码、正式模型注册与模型实现；未修改模型、配置、实验数据或训练任何模型。静态核验与绘图验证分开记录，实际运行和图像检查结果见本交付目录的验证记录。

## 1. 本图对应的论文与正式模型

- 论文源文件：`docs/ICSPS2026_LaTeX/ICSPS2026_paper.tex:96–121`。小节标题为 *Full-Spectrum Fourier Residual Refinement*；Fig. 2 标签为 `fig:fdr`，位于该小节公式与解释之后。图中使用正文简称 **FDR**，不使用历史实现类名代替论文模块名。
- 模型构造入口：`model_registry.py:330–357`。正式 `ours` 方法与 `full` 变体构造 `TransSARV2_DualFreqNG_Bottle`。
- 正式配置：`configs/icsps2026_frozen_v2.json:3–7` 指定 `ICSPS26-FROZEN-v2`、normalized linear intensity 与 `256×256` 输入。`ablation_config.py:19–28` 的 `FORMAL_VARIANTS['full']` 开启 `decoder_fdr` 与 `bottleneck_fdr`。
- 正式 wrapper：`transform_main.py:2125–2131` 将上述两个开关传入 `TransSARV2_FreqNG_Bottle`；该模块在 `2040–2048` 构造瓶颈与解码器，在 `2066–2071` 执行 encoder → deepest-feature bottleneck → decoder。
- 本图直接展开的实现为 **`transform_main.py:1406–1458`, `FFTRefineBlock`**。它与 `BottleneckRefine` 外层不是同一个模块，见第 5 节。

## 2. 精确 forward 连接与维度

图中采用 `H×W×C` 的读图顺序并省略 batch 维；PyTorch 实际采用 `B×C×H×W`。令 `W_f = floor(W/2)+1`。所有 convolution 的 stride 都为 1，因此本模块不改变输入空间尺寸。

| 步骤 / 图中节点 | 运算与输出（省略 batch） | 实现依据 |
|---|---|---|
| Input feature `F` | 实数 `H×W×C`；同时分流到 spatial、spectral 与 identity residual 三条路径 | `transform_main.py:1428–1429,1432,1458` |
| Spatial: depthwise `3×3` | `C→C`，`groups=C`，padding 1；实数 `H×W×C` | `transform_main.py:1410–1411` |
| Spatial: pointwise `1×1` | `C→C`；实数 `H×W×C` | `transform_main.py:1412` |
| Spatial: GELU | 得到 `F_s`；实数 `H×W×C` | `transform_main.py:1413,1429` |
| `rFFT2` | `torch.fft.rfft2(F, norm='ortho')`；复数 `H×W_f×C` | `transform_main.py:1431–1432` |
| `[Re(Q), Im(Q)]` | 沿 **channel** 拼接；实数 `H×W_f×2C`，不扩大频率网格尺寸 | `transform_main.py:1433` |
| Spectral: first `1×1` | 实值 convolution `2C→2C`；实数 `H×W_f×2C` | `transform_main.py:1416–1417,1440` |
| Spectral: GELU | 实数 `H×W_f×2C` | `transform_main.py:1418` |
| Spectral: second `1×1` | 实值 convolution `2C→2C`；实数 `H×W_f×2C` | `transform_main.py:1419` |
| Split + complex reconstruction | 沿 channel 均分成两个实数 `H×W_f×C` 张量，再以 `torch.complex(real, imag)` 得到复数 `H×W_f×C` | `transform_main.py:1442–1444` |
| `irFFT2` | `torch.fft.irfft2(X_new, s=F.shape[-2:], norm='ortho')`；得到实数 `F_f`，`H×W×C` | `transform_main.py:1445` |
| `[F_s, F_f]` | 沿 channel 拼接；实数 `H×W×2C` | `transform_main.py:1456` |
| Fusion: `1×1` | `2C→C`；实数 `H×W×C` | `transform_main.py:1422–1423` |
| Fusion: GELU | 实数 `H×W×C` | `transform_main.py:1424` |
| Fusion: `3×3` | `C→C`，padding 1；实数 `H×W×C` | `transform_main.py:1425` |
| Residual addition | `FDR(F) = F + fuse([F_s,F_f])`；实数 `H×W×C` | `transform_main.py:1456–1458` |

模块内不存在 normalization 层、dropout、pooling、upsampling、attention、gate 或可学习 residual scale。`ortho` 是 FFT normalization 参数，不应绘成 BatchNorm/LayerNorm。各 `Conv2d` 未显式设置 `bias=False`，因此使用 PyTorch 默认 bias；架构图无需逐个显示偏置，但不应宣称这些运算无偏置。

实现还保留老版本 PyTorch 的 `torch.rfft` / `torch.irfft` fallback（`transform_main.py:1435–1438,1446–1454`），使用 `normalized=True, onesided=True` 并显式恢复输入尺寸。图表示两条兼容实现共同对应的运算，不把软件版本检查画成科研模块分支。

## 3. “Full-spectrum”的准确含义与图示边界

1. `rfft2` 对两个空间维执行实输入二维 Fourier transform；其最后一维存储非冗余一侧，所以复数存储网格为 `H×(floor(W/2)+1)`，**不是 `H×W`**，也不是把两个空间维都折半。
2. “Full-spectrum”指不按频带筛选、截断、划分或只增强高频；实输入完整频谱由其非冗余表示及 Hermitian redundancy 描述。代码对所有存储的频率坐标使用同一组 channel mixing 权重。图中应同时保留 **full-spectrum** 论文命名和 **one-sided rFFT representation** / `W_f` 尺寸说明，避免把全频覆盖误画为全复数网格存储。
3. 两层 `1×1` convolution 的 kernel 在频率网格上只覆盖一个坐标。它们混合该坐标的 feature channels 及 real/imaginary components；参数在不同频率坐标共享，**没有学习的相邻频率卷积**。该陈述依据为 kernel size 1（`transform_main.py:1417,1419`）与正文 `ICSPS2026_paper.tex:106,116`。图中无需画邻频箭头或局部频谱卷积核。
4. 两层 mixer 与 GELU 都在拼接后的实数张量上执行。它不是 complex-valued convolution，也没有 magnitude/phase 分解、fftshift、频率 mask、频带 attention 或独立低频/高频支路。
5. `irfft2(..., s=(H,W))` 明确返回原空间尺寸的实数 feature。代码没有另写 Hermitian-symmetry constraint 或对学到的实虚部进行显式对称化；不应把“强制频谱共轭对称”画成额外模块。
6. “No convolution across neighboring frequency coordinates”限定于 **频域内的学习操作**。空间分支和 inverse FFT 之后的 `3×3` fusion convolution 确实跨相邻空间位置，不能将限定误写成整个 FDR 没有空间邻域卷积。

## 4. 六个实例的实际输入与 Fourier 尺寸

默认输入 `256×256` 时，各实例尺寸如下。图 2 使用符号 `H,W,C` 可完整说明同一结构，不必重复列出多尺度数据而占用单栏空间。

| Instance | Real input `H×W×C` | Complex rFFT `H×W_f×C` | Real/imag concat `H×W_f×2C` | 来源 |
|---|---:|---:|---:|---|
| Bottleneck inner FDR | `8×8×512` | `8×5×512` | `8×5×1024` | `transform_main.py:1474,2040–2043,2068–2069` |
| Decoder D4 | `16×16×320` | `16×9×320` | `16×9×640` | `transform_main.py:1686,1728–1734` |
| Decoder D3 | `32×32×128` | `32×17×128` | `32×17×256` | `transform_main.py:1687,1738–1744` |
| Decoder D2 | `64×64×64` | `64×33×64` | `64×33×128` | `transform_main.py:1688,1748–1754` |
| Decoder D1 | `128×128×32` | `128×65×32` | `128×65×64` | `transform_main.py:1689,1758–1764` |
| Decoder D0 | `256×256×16` | `256×129×16` | `256×129×32` | `transform_main.py:1690,1768–1770` |

每个实例单独构造，因而参数不在不同 decoder scales 或 bottleneck 之间共享。“Shared weights”在本图仅表示**同一个实例的同一 mixer 在频率坐标之间共享卷积参数**，不表示六个 FDR 使用一组模型权重。

## 5. 与外层 local–spectral bottleneck 的区别

`BottleneckRefine`（`transform_main.py:1460–1501`）包含另一个 local branch、一个完整 `FFTRefineBlock`、外层 fusion 和初始化为 `0.1` 的参数 `gamma`。其精确外层表达为：

```text
Bottleneck(F) = F + gamma_b × outer_fuse(local(F) + FDR(F)).
```

依据：local branch `1467–1471`；inner FDR `1474`；outer fusion `1477–1481`；`gamma` 初始化 `1483`；相加及外层 residual `1493–1501`。注意：外层 local 与 FDR 使用 **addition**，而 Fig. 2 内部的 spatial 与 spectral features 使用 **concatenation**。Fig. 2 展开 inner FDR，最终 residual 为系数 1 的 `F + out`，**不应添加 `gamma_b`、额外 local branch 或额外 outer projection**。正文 `ICSPS2026_paper.tex:94` 的 scale initialized to 0.1 指外层瓶颈；`97–116` 的 FDR 公式则与内部无 scale 的 residual 相符。

## 6. 论文一致性与本次绘图选择

对照当前英文正文 `docs/ICSPS2026_LaTeX/ICSPS2026_paper.tex:97–116` 与 `transform_main.py:1406–1458`，**未发现影响 Fig. 2 的实质运算顺序、连接或输出维度矛盾**。图 2 占位说明 `119` 已要求 spatial branch、rFFT2、real/imag concat、两次共享 `1×1` mixing、split、irFFT2、spatial/spectral concat、projection 与 residual addition。本图应将正文未展开的 projection kernel sizes 用代码事实补足为 `1×1 → GELU → 3×3`。

以下是容易误读而非新的实验结论：

- 论文“full-spectrum”并不意味着 rFFT 存储 `H×W` 个复数频率系数；图采用 `W_f=floor(W/2)+1` 澄清。
- 论文的瓶颈 residual scale 不属于 FDR 内部；图中省略它符合实际代码。
- Fig. 1 中 FDR 为一个紧凑节点；Fig. 2 专门解释该节点的内部分流、表示变换与融合，不重复全网架构、结果表或训练方案。
- 图 2 无需实验照片、checkpoint、预测图像或额外数值数据；结构全部由现有实现和论文充分支持，未虚构实验表现或频谱响应。

## 7. 最终尺寸与建议插入位置

论文使用 `\documentclass[conference]{IEEEtran}`（`docs/ICSPS2026_LaTeX/ICSPS2026_paper.tex:1`）。本地模板 `docs/ICSPS2026_LaTeX/IEEEtran.cls:1734–1735` 指定 `columnsep=1pc`、`textwidth=43pc`，故单栏宽度为 `(43pc−1pc)/2 = 21pc = 252 TeX pt = 88.5678705 mm`。Fig. 2 使用单栏 `figure`（`ICSPS2026_paper.tex:117`），交付图按 **88.568 mm** 设定宽度。

建议替换 `fig:fdr` 现有占位框，仍放在 *Full-Spectrum Fourier Residual Refinement* 小节的解释之后、*Guidance Modulation and Intensity Compensation* 小节之前。`1.15in` 是占位框高度，不是约束真实矢量图的高度；应以 `width=\columnwidth` 保持自然纵横比，以便字号与箭头在最终尺寸下清晰。其英文图注与可直接使用的 LaTeX 插入片段见同目录 README。

## 8. 核验版本指纹

以下 SHA-256 在本次审计时读取，用于定位实际依据；路径相对项目根目录。

| File | SHA-256 |
|---|---|
| `transform_main.py` | `9405BB05D68405848C9966F4B093EA078208AD70856D3583672C70A3AA7C82FB` |
| `docs/ICSPS2026_LaTeX/ICSPS2026_paper.tex` | `C8AEBB8FA1C0F6D0B404A3207C104837597E19A74435FA15090A163F7A0EC243` |
| `configs/icsps2026_frozen_v2.json` | `75177763E514BA7E25958C8AB6A041AB4D235CDDBDC9A37F61347D5E9A8AD423` |
| `model_registry.py` | `7F32FDD81F06C148D357EADB684D3F4A3DCD8E81B733EFD376A2D2A3A708F38A` |
| `ablation_config.py` | `C51FA89433D4F9F12E440FA1915FEE041C38FBE51ACCD89D1D793A4A7FBC3E29` |

注意：现有 Fig. 1 审计所记录的论文 hash 与本次当前论文 hash 不同；以上以本次读取到的当前文件为准，不沿用旧行号或旧 hash 推断本次论文状态。
