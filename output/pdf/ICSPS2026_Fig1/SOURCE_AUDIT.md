# Fig. 1 — source and architecture audit

核验日期：2026-09-07。本文档只核验绘制 Fig. 1 所需的模型、连接、表示域和尺寸，不改动模型、实验数据或训练配置。

## 1. 本图对应的正式模型

- 论文：`docs/ICSPS2026_LaTeX/ICSPS2026_paper.tex:50–56`，双栏 `figure*`，标签 `fig:overview`；Fig. 1 占位说明明确要求区分 inherited operators 与 extensions。
- 正式训练入口：`train_icsps2026.py:69,75–76,371–381`，默认配置 `configs/icsps2026_frozen_v2.json`，通过 `build_model(args.method, variant=args.variant, …)` 构造模型；正式运行矩阵 `scripts/icsps2026/03_train_core.sh:23` 包含 `ours:full`。
- 正式测试入口：`evaluate_icsps2026_synthetic.py:42,46–47,183–190` 使用同一 registry 并严格加载 checkpoint。
- 正式注册：`model_registry.py:330–357`，`build_model("ours", variant="full")` 构造 `TransSARV2_DualFreqNG_Bottle(variant="full", numeric_domain=INTENSITY_DOMAIN)`；`numeric_domain.py:16,19` 定义 `INTENSITY_DOMAIN="intensity_v1"`、`LOG_ALPHA=10.0`。
- `ablation_config.py:19–28` 的 `FORMAL_VARIANTS["full"]` 开启 log representation、intensity compensation、learned skip fusion、decoder FDR、guidance gates、local bottleneck 与 bottleneck FDR。
- `configs/icsps2026_frozen_v2.json:3–7` 指定 normalized linear intensity 和 `256×256`。最终图中尺寸采用 `H×W×C`，本审计下表采用 `C×H×W`；均省略 batch 维。

图不采用其他历史类、历史消融或显式 legacy numerical-domain 分支。

## 2. 经代码核对的 forward 路径

1. 原始输入 `Y` 裁剪到 `[0,1]`，得到 `x01`；`T_alpha(Y)=log(1+alpha Y)/log(1+alpha)`，`alpha=10`。依据：`transform_main.py:2141–2155`；`numeric_domain.py:94–99`。
2. 同一 log 输入同时送入五级 `EncoderTransformerV2` 和 `NoiseEstimator`（图文名称为 **Guidance-map generator**）。依据：`transform_main.py:2019–2036,2063–2066`。
3. 仅最深 encoder feature `x1[4]` 经过 **Local–spectral bottleneck**；其他四个 encoder feature 留给同分辨率 decoder skips。依据：`transform_main.py:2068–2071`。
4. 前四个 decoder stage 顺序严格为 **Upsample → Skip concatenation + learned fusion → FDR → Residual block → Guidance modulation**。Skip 分别来自 `E4,E3,E2,E1`。依据：`transform_main.py:1728–1766`。
5. 第五个 stage 为 **Upsample → FDR → Residual block → Guidance modulation**，全分辨率处没有 encoder skip，也没有 `FusionBlock`。依据：`transform_main.py:1768–1772`。
6. stage 输出经 stride-1 transposed convolution `16→8`（分辨率保持），再经过 `3×3` convolution `8→1` 和 **Sigmoid**，产生 bounded log estimate `Z_hat`。依据：`transform_main.py:1675,1774–1777,2050–2054,2071–2073,2122–2131`。
7. `X_tilde=T_alpha^-1(Z_hat)`，再将 `[Y,X_tilde]` 送入 **Intensity residual compensator**，得到 `R_c`；最终 `X_hat=clip(X_tilde+gamma_c R_c,0,1)`。原始强度旁路必须接到补偿器，而不是与 log estimate 直接相加。依据：`transform_main.py:1522–1541,2157–2166`；`numeric_domain.py:102–109`。

### 关键尺寸

| 图中节点 / 操作 | 输出 `C×H×W` | 代码依据 |
|---|---:|---|
| Noisy intensity / bounded log input | `1×256×256` | `configs/icsps2026_frozen_v2.json:3–7`; `transform_main.py:2143–2155` |
| Guidance map | `1×256×256` | `transform_main.py:1503–1519` |
| E1 | `32×128×128` | `transform_main.py:244–245,349–355` |
| E2 | `64×64×64` | `transform_main.py:246–247,357–363` |
| E3 | `128×32×32` | `transform_main.py:248–249,365–371` |
| E4 | `320×16×16` | `transform_main.py:250–251,373–379` |
| E5 / bottleneck | `512×8×8` | `transform_main.py:252–253,381–387,1460–1501` |
| D4, skip E4 | `320×16×16`; concatenation has `640` channels | `transform_main.py:1670,1680,1728–1736` |
| D3, skip E3 | `128×32×32`; concatenation has `256` channels | `transform_main.py:1671,1681,1738–1746` |
| D2, skip E2 | `64×64×64`; concatenation has `128` channels | `transform_main.py:1672,1682,1748–1756` |
| D1, skip E1 | `32×128×128`; concatenation has `64` channels | `transform_main.py:1673,1683,1758–1766` |
| D0, no skip | `16×256×256` | `transform_main.py:1674,1768–1772` |
| Full-resolution head feature | `8×256×256` | `transform_main.py:1675,1774–1777` |
| Bounded log prediction / inverse-mapped estimate | `1×256×256` | `transform_main.py:2050–2054,2072,2157–2160` |
| Compensator concatenation / hidden layers / residual | `2→32→32→1`, all `256×256` | `transform_main.py:1526–1541` |
| Restored intensity | `1×256×256` | `transform_main.py:2162–2166` |

Encoder heads are `[1,1,2,4,8]`, depths `[3,3,4,6,3]`, spatial-reduction ratios `[8,8,4,2,1]`; first patch embedding uses `7×7,stride 2`, subsequent embeddings use `3×3,stride 2`. Evidence: `transform_main.py:234–253,2021–2036`. These details belong in this audit or a module figure; including all of them in Fig. 1 would obscure the overall connection structure.

## 3. 颜色归属

“Inherited”依据本地正式注册的 `TransSARV2` baseline 与 proposed model 逐项比对。论文材料已有 Trans-SAR 归属说明（`ICSPS2026_paper.tex:94`; `docs/ICSPS2026_SAR去斑网络设计与实验方案.md:130–144`）；本次绘图审计未将重新取得官方仓库并作逐字 diff 描述为已完成的工作。

| 灰色：继承算子 | 证据 |
|---|---|
| Five-stage Transformer encoder, stage configuration, overlap embedding | Baseline `transform_main.py:1812–1815`; proposed `2021–2036`; shared implementation `234–394` |
| Five stride-2 transposed-convolution upsamplers | Baseline `1169–1177`; proposed `1670–1674`; operator `arch/trans_basenetworks.py:98–105` |
| Convolutional residual blocks | Baseline `1170–1178`; proposed `1700–1704`; operator `arch/trans_basenetworks.py:108–120` |
| Stride-1 `16→8` projection and final `8→1` convolution | Baseline `1179,1819`; proposed `1675,2050` |

| 彩色：本文使用的扩展 / 更改 | 证据 |
|---|---|
| Bounded log mapping, exact inverse, intensity output clipping | `transform_main.py:2146–2166`; `numeric_domain.py:94–109` |
| Local–spectral bottleneck and six FDR blocks | `transform_main.py:1406–1501,1685–1690,2040–2043` |
| Concatenation-based learned skip fusion | Proposed `1680–1683,1731,1741,1751,1761`; baseline uses residual block then skip **addition**, `1195–1205` |
| Guidance-map generator and five guidance modulations | `transform_main.py:1503–1519,1543–1566,1692–1697` |
| Sigmoid | Baseline uses **Tanh** `1821`; formal proposed branch uses **Sigmoid** `2122–2131,2053–2054` |
| Intensity residual compensator | `transform_main.py:1522–1541,2162–2163` |
| Encoder-frozen masked post-adaptation | `train_icsps2026_ams.py:301–311,452–460`; frozen config `ams` section |

`FusionBlock` 位于 `arch/trans_basenetworks.py:124–134` 不能作为“它已用于 baseline”的依据；baseline `convprojection_baseV2` 实际不调用它。算子定义文件名不是继承归属的判据。

## 4. 必须避免的图示误读与材料差异

与当前英文正文 `ICSPS2026_paper.tex:78–165` 对照，未发现影响 Fig. 1 的实质 forward 拓扑或关键 feature 尺寸矛盾。以下为历史命名、容易遗漏的细节和限定说明，不能当作新增实验事实：

- **Historical “DualDomainFusion” ≠ direct log/intensity fusion.** 源码类名及参数名 `log_branch_output` 仍带历史命名（`transform_main.py:1522,1535–1538`），但正式 wrapper 在 `2159` 执行 inverse，再于 `2163` 调用补偿器。英文正文 `133–145` 已正确说明同域补偿。图中采用 **Intensity residual compensator**。
- **Historical “NoiseEstimator” ≠ supervised noise estimate.** 代码名称 `NoiseEstimator` / `noise_prior`（`1503,2064`）不证明存在 noise-map 或 look-number supervision。当前正文 `131` 将其限定为 latent guidance map。图中采用 **Guidance-map generator** 与 **Guidance modulation**，不画输入 `L`。
- **Decoder ≠ additive baseline skip.** 本模型是 upsample 后 concat/fusion，再 FDR/RB/gate；baseline 是 upsample 后 RB 再 skip addition。若直接复用 Trans-SAR 架构箭头，会错画 decoder。依据分别为 `1728–1766` 与 `1195–1205`。
- **Bottleneck combines by addition, not concatenation.** `BottleneckRefine` 的 local output 与完整 `FDR(x)` **相加**后经 projection，并以 `x+gamma_b·out` 输出（`1493–1501`）。FDR 内部才是 spatial / spectral **concat**（`1456–1458`）；FDR 自带 identity residual，不能将瓶颈频域支路解释为纯 spectral increment。
- **There are six FDR instances.** One at `512×8×8`, then five at `320×16×16`, `128×32×32`, `64×64×64`, `32×128×128`, `16×256×256`。They have separate parameters. `transform_main.py:1474,1686–1690`; manuscript `120`。
- **Full-spectrum real–imaginary mixing, not band splitting.** `rfft2` 的 real/imag 沿 channel concat，`1×1` convolutions mix channels，然后 `irfft2`（`1431–1455`）。图中不画 high/low-frequency branches、magnitude/phase split 或 complex-valued convolution。
- **Full-resolution projection is not another 2× upsample.** `convd1x` 名称沿用 upsample class，但其 stride 为 1（`1675`），`16→8` channel reduction 保持 `256×256`。不能因此把输出画成 `512×512`。
- **Sigmoid is a changed head operator.** 正式主模型使用 sigmoid，legacy numerical-domain path 才使用 tanh；论文 `96,133` 与正式路径一致。图不能照搬 baseline 的 tanh。
- **Clipping details.** 正文 `145` 的“Final clamping”指 residual add 后输出 clamp。代码还有输入 clamp（`2146`）和 inverse helper 对 sigmoid 范围的防御性 clamp（`numeric_domain.py:108`）。后者在正式 sigmoid 输出下不改变值；不能将其解读为独立可训练模块或新的归一化。
- **Post-adaptation is a training stage, not an inference branch.** Mask 20% of noisy pixels; freeze only `log_branch.Tenc`; update the remaining learned modules. 依据：`train_icsps2026_ams.py:307–311,453,460`；正文 `155–167`。图中的 masked adaptation 应单独显示并说明 inference 使用 unmasked observation，不画成生成 final output 必经的第二次 forward。

## 5. 模板尺寸与建议插入位置

`ICSPS2026_paper.tex:1` 使用 `\documentclass[conference]{IEEEtran}`；Fig. 1 已为双栏 `figure*`（`51`）。本地 `IEEEtran.cls:1734–1735` 定义 `\columnsep=1pc`、`\textwidth=43pc`。按 TeX `1pt=1/72.27in`，`43pc=516pt=181.3532586mm`，故最终图宽采用 **181.353 mm**。单栏为 `21pc=88.568mm`，不建议把该多尺度总图压到单栏。

建议替换现有 `fig:overview` 占位图并保留标签，位置仍在 Introduction 贡献段之后、Related Work 之前；实际排版由 IEEE 双栏浮动体规则决定。图1表达全局路径、四条 encoder skips、六个 FDR 插入点、原始强度旁路、颜色归属与独立 adaptation 状态；FDR 详细内部运算留给 Fig. 2，避免把完整计算图堆入 Fig. 1。

## 6. 源文件指纹

以下 SHA-256 用于定位此次绘图依据的版本，路径相对项目根目录：

| File | SHA-256 |
|---|---|
| `model_registry.py` | `7F32FDD81F06C148D357EADB684D3F4A3DCD8E81B733EFD376A2D2A3A708F38A` |
| `ablation_config.py` | `C51FA89433D4F9F12E440FA1915FEE041C38FBE51ACCD89D1D793A4A7FBC3E29` |
| `transform_main.py` | `9405BB05D68405848C9966F4B093EA078208AD70856D3583672C70A3AA7C82FB` |
| `arch/trans_basenetworks.py` | `F91F04EBFE8455360F48E7F5C0BF38190E766FF98D66DF0F38B7EEA1F85DBE9C` |
| `numeric_domain.py` | `D250A6C6B95058E3D8BF711514BEBE8A254F7BDD2B3261AE1CB78241A0F170EE` |
| `configs/icsps2026_frozen_v2.json` | `75177763E514BA7E25958C8AB6A041AB4D235CDDBDC9A37F61347D5E9A8AD423` |
| `docs/ICSPS2026_LaTeX/ICSPS2026_paper.tex` | `96B61F08A2FDD62E9AA85C529DA93496232E022AB5030DFB22B7F623604C5F5D` |

本文件是静态源码审计；实际运行 forward 的结果及渲染检查应以同目录交付的验证记录为准。
