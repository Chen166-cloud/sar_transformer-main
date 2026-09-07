# Fig. 3 — representation-consistent reconstruction and AMS source audit

核验日期：2026-09-07。范围为当前论文 Fig. 3 的数值域重建链、masked post-adaptation（AMS）流程、冻结边界和数据入口。本文档记录源码与论文证据；绘图检查、动态验证和导出结果由本目录的其他交付文件记录。未修改模型、协议、论文、实验数据或已有 checkpoint，未重新训练。下列路径均相对于项目根目录 `D:/research/sar_transformer-main`；行号对应核验时的文件。

## 1. 论文版本与正式入口

- 本轮使用当前论文源码 `docs/ICSPS2026_LaTeX/ICSPS2026_paper.tex`：bounded log transform / inverse 位于第 65–87 行；重建与补偿位于第 125–136 行；AMS 公式与流程位于第 138–155 行；Fig. 3 占位、图注及标签 `fig:representation_ams` 位于第 157–162 行。不能沿用旧论文版本的标签或术语推断本图。
- 图中应保留两个面板：(a) 数值域重建，解释 exact inverse 在 intensity compensation **之前**；(b) AMS，解释原始 noisy target、20% mask、冻结/可训练模块、fixed-mask validation selection 与 unmasked inference。
- 正式模型入口 `model_registry.py:330–357`：`build_model("ours", variant="full", numeric_domain="intensity_v1")` 构造 `TransSARV2_DualFreqNG_Bottle`。
- `ablation_config.py:19–28` 的 `FORMAL_VARIANTS['full']` 使用 `representation='log'`，并开启 compensation、MSF、decoder FDR、guidance gate、local bottleneck 和 bottleneck FDR。
- `configs/icsps2026_frozen_v2.json:3–7` 使用 `ICSPS26-FROZEN-v2`、normalized linear intensity 和 `256×256` 输入。`numeric_domain.py:16,19` 给出模型内部标识 `intensity_v1` 和 `LOG_ALPHA=10.0`。
- AMS 正式启动脚本 `scripts/icsps2026/06_train_ams.sh:18–21,44–50` 默认 seed 42，从 `train/ours_full/seed42/checkpoint_best.pth` 初始化；训练器 `train_icsps2026_ams.py:279–304` 审核 base checkpoint 的正式协议、numeric domain、method 和 variant，再严格加载 Full。图不据历史文件名如 `TransSARV2_DualFreqNG_Bottle_AMS_best_e8.pth` 推断所选 epoch。

## 2. 面板 (a)：实际 forward、数值域与尺寸

图中的单通道图像均为 `H×W×1`；PyTorch 存储顺序为 `B×1×H×W`。正式输入为 `B×1×256×256`。`Y` 表示已在统一数据入口映射到 `[0,1]` 的 observed intensity；wrapper 仍执行防御性的 `clip(Y,0,1)`。可以在图中以输入范围标识该约束，无须展开为额外科研模块。

| 图中节点或连接 | 精确计算、数值域和尺寸 | 源码依据 |
|---|---|---|
| Observed intensity `Y` | `x01 = clamp(x,0,1)`；`B×1×256×256` | `transform_main.py:2141–2146` |
| Bounded log transform `T_alpha` | `log1p(alpha*Y)/log1p(alpha)`，`alpha=10`；归一化 log intensity `[0,1]`，形状不变 | `numeric_domain.py:19,94–99`; `transform_main.py:2150–2153` |
| Enhanced reconstruction branch | `log_branch(T_alpha(Y))`；encoder 和 guidance generator **并行接收同一 log input**；深层 feature 经 bottleneck，decoder 同时使用 encoder features 和 guidance | `transform_main.py:2063–2071,2155` |
| Bounded log prediction `Z_hat` | `Sigmoid(ConvLayer_3x3_8to1(decoder_feature))`；`B×1×256×256`，范围 `[0,1]` | `transform_main.py:2050–2054,2072,2122–2131` |
| Exact inverse `T_alpha^-1` | `expm1(Z_hat*log1p(alpha))/alpha`；实现先将 `Z_hat` clamp 到 `[0,1]`；恢复 preliminary **intensity** `X_tilde` | `numeric_domain.py:102–109`; `transform_main.py:2157–2160` |
| Channel concatenation `[Y,X_tilde]` | observation bypass 使用 `x01`；与 inverse-mapped estimate 沿 channel 拼接，得到 `B×2×256×256` | `transform_main.py:2162–2163,1540` |
| Intensity compensator `F_comp` | Conv `3×3`, `2→32` → GELU → Conv `3×3`, `32→32` → GELU → Conv `3×3`, `32→1`；各 stride 1、padding 1；residual `R_c` 为 `B×1×256×256` | `transform_main.py:1522–1532` |
| Scaled residual addition | `X_tilde + gamma_c*R_c`；`gamma_c` 是可学习 scalar，初始化 `0.1`；加法保留同尺寸 | `transform_main.py:1533,1540–1541` |
| Final clipping | `X_hat = clip(X_tilde+gamma_c*R_c,0,1)`；normalized intensity，`B×1×256×256` | `transform_main.py:2162–2166` |

必须保留的箭头：`Y` 分流到 log transform 与 concatenation；`X_tilde` 分流到 concatenation 与 residual addition；`[Y,X_tilde]` 进入 compensator；其 residual 经 `gamma_c` 缩放进入 addition；最后才 clip。不存在 `Y + Z_hat`、`Y + R_c` 的输出残差基底，也不存在把 log prediction 直接送入 compensator 的正式路径。

`R_c` 为 **signed intensity-domain correction**，最后一个 convolution 后无 sigmoid/tanh；它和未 clip 的 residual sum 不应标注为 `[0,1]`。`Y`、bounded log prediction、`X_tilde` 和最终 `X_hat` 才有对应范围约束。“同数值域”指 residual addition 的两项都处于 intensity representation，不意味着残差非负。

## 3. 模块连接与可训练边界

- `transform_main.py:2064,2066` 分别执行 `noise_estimator(x)` 和 `Tenc(x)`。它们共享输入；代码调用顺序不表示 guidance 由 encoder 或 decoder 输出生成。图若压缩主干，宜写 “Encoder + enhanced reconstruction (Fig. 1)” 或把 guidance 放在并行分支，不能画成 `Encoder → Decoder → Guidance`。
- Encoder 五级 feature 的实际维度为 `B×32×128×128`、`B×64×64×64`、`B×128×32×32`、`B×320×16×16`、`B×512×8×8`（`transform_main.py:2021–2036,1717–1722`）。Bottleneck 仅更新 `x1[4]`（`2068–2069`）；其余四级送 decoder skips。
- Guidance generator 为 `1→32→32→32→1` 的四个 `3×3` convolution，前三个后接 GELU、最后接 Sigmoid（`transform_main.py:1503–1519`），产生 `B×1×256×256` latent map。该图可只标 “Guidance”，不应沿用历史类名将其称为有真值监督的噪声图。
- Guidance 被送入五个 decoder gate（`transform_main.py:1736,1746,1756,1766,1772`），不是单个最后层 filter；末级输出再投影为 8 channels 并进入输出 head（`1774–1777,2050,2072`）。详细解码器和 FDR 结构已由 Fig. 1/2 解释，本图无需重复展开。
- AMS 仅将 `model.log_branch.Tenc.parameters()` 的 `requires_grad` 设为 `False`（`train_icsps2026_ams.py:305–308`），并在每个训练 epoch 执行 `model.train()` 后单独 `model.log_branch.Tenc.eval()`（`452–453`）。
- Adam 仅接收 `requires_grad=True` 的参数（`train_icsps2026_ams.py:310–314`）。因此 bottleneck、decoder（含其 FDR/gates）、guidance、output head、compensator 以及相应 learnable scales 均可更新；配置明确列于 `configs/icsps2026_frozen_v2.json:152–160`。
- Log、inverse、mask multiplication、concatenation、clipping 为无参数运算。“Frozen encoder”不等于整条 log branch 冻结，也不等于这些固定运算有冻结权重。面板 (b) 的颜色/图例必须分清 inherited learned encoder 的 frozen 状态与可训练重建模块；不要把 “all gray modules frozen” 泛化到其他面板的无参数节点。

## 4. 面板 (b)：AMS 的真实数据流

| 阶段 | 实际执行 | 证据 |
|---|---|---|
| Noisy observation | 使用 normalized real `Noisy`，不是 clean image 或 GT16；数据集返回 `image[None]` | `train_icsps2026_ams.py:92–101`; `docs/ICSPS2026_LaTeX/ICSPS2026_paper.tex:198–200` |
| Bernoulli mask | `M = (rand(shape, seeded_generator) < 0.2).float()`；`M=1` 表示删除/评价的位置；20% 为 Bernoulli 概率，并非每张严格固定像素数 | `train_icsps2026_ams.py:104–107`; `configs/icsps2026_frozen_v2.json:142–145` |
| Training corruption | `Y_m = Y*(1-M)`，masked positions 填 **zero intensity**，没有局部均值替换、像素置换或噪声填充 | `train_icsps2026_ams.py:458–460` |
| Entire-model input | `prediction = model(Y_m)`；**整个 Full 模型接收 `Y_m`**，包括面板 (a) 的 observation-to-compensator bypass；此时该 bypass 为 `Y_m` | `train_icsps2026_ams.py:460`; `transform_main.py:2146,2163` |
| Reconstruction loss inputs | 三个必要输入为 `prediction=f_theta(Y_m)`、original noisy target `Y`、mask `M`；目标仅在 `M=1` 处比较，`Y` 无直接通向模型/补偿器的未遮挡旁路 | `train_icsps2026_ams.py:110–111,461` |
| TV regularization | 对整个 predicted image 的水平、竖直相邻绝对差分别求 mean，再相加；未乘 mask | `train_icsps2026_ams.py:114–118,461` |
| Optimization | 8 epochs，batch size 1，Adam，learning rate `1e-6`、weight decay `1e-5`、TV coefficient `1e-3`；仅更新上述可训练参数 | `configs/icsps2026_frozen_v2.json:131–160`; `train_icsps2026_ams.py:216–220,310–314,443–467` |
| Training masks | seed 包含 protocol、`ams-train-mask`、seed、epoch、sample_id；不同 epoch 重采样，同时保持可复现 | `train_icsps2026_ams.py:61–63,458–459` |
| Validation masks | 每个 validation sample 的 seed 不含 epoch；初始时记录 sample_id、parent_id、seed、binary-mask SHA256，并在各 epoch 重用；验证也输入 `Y*(1-M)` | `train_icsps2026_ams.py:244–264,360–375,167–192` |
| Selection | 取 validation 样本平均的 `masked_l1 + lambda_tv*TV` 最低值；候选是 adapted epochs 1–8。epoch 0 仅记录 baseline，`is_best=False`；`best` 初始为 infinity | `train_icsps2026_ams.py:186–192,387–388,408–416,469–488`; `configs/icsps2026_frozen_v2.json:162–165` |
| Inference | chosen model 接收完整 normalized `Y`，不加 mask，不构造 loss、不再执行参数更新 | `evaluate_icsps2026_real.py:471–485`; `configs/icsps2026_frozen_v2.json:167` |

图中 loss 应有清楚的 original noisy target 支路与 mask 支路。训练/验证共用一个 “whole model `f_theta`” 可避免重复绘制架构，但必须保证 `Y_m` 是其唯一 image input。模型内部的 frozen encoder 与 trainable reconstruction 可以用成员分组表述，不应为了紧凑把不存在的串行连接画出来。由 validation 到 checkpoint selection 的箭头表示选择，不表示验证参数更新；由 selected checkpoint 到 inference 的箭头表示模型权重使用，不是图像特征拼接。

## 5. 已确认的描述差异与解释边界

### 5.1 AMS 分母：论文 `+epsilon` 与代码 `clamp_min(1)`

当前论文 `ICSPS2026_paper.tex:145–153` 的 AMS 公式将 masked L1 分母写成 `||M||_1 + epsilon`；**实际实现** `train_icsps2026_ams.py:110–111` 为：

```python
(torch.abs(prediction - target) * mask).sum() / mask.sum().clamp_min(1.0)
```

数学上对应 `||M*(prediction-Y)||_1 / max(||M||_1,1)`。对于 nonempty binary mask，代码分母就是 masked pixel count；空 mask 时设为 1，损失分子为零。它并不与任意非零 `epsilon` 的分母逐字/严格相等。图若写出完整公式，应依据执行代码使用 `max(||M||_1,1)`；若只标 “Masked L1 + 1e-3 TV”，仍须保留本差异说明。此处未擅自修改论文公式或训练实现。

### 5.2 历史名称不代表正式计算

- `DualDomainFusion.forward` 参数名 `log_branch_output` 和其历史类名（`transform_main.py:1522,1535–1541`）容易暗示 log/intensity 直接混合；正式 `intensity_v1` wrapper 传入的是 `branch_clean=T_alpha^-1(branch_output)`（`2157–2163`），因此应标注 **Intensity compensator**。旧 `legacy_amplitude_v0` 分支在 `2168–2176` 使用不同链路，不属于本图。
- `TransSARV2_FreqNG_Bottle` 单独构造时 `output_activation='tanh'`（`transform_main.py:2008`），但正式 wrapper 明确传入 `'sigmoid'`（`2122–2131`），因此论文与正式 sigmoid output 一致；不能由默认构造器值把本图改成 tanh。
- `NoiseEstimator` 是历史类名。正文将其界定为 unsupervised latent guidance（论文第 125–126 行），本图沿用 Guidance；本轮没有证据支持 noise-map targets 或 look-number supervision。

### 5.3 正式 real-data normalization 入口

实际 AMS `RealQCNoisyDataset.__getitem__` 调用 `apply_fixed_mapping(raw,self.mapping)`（`train_icsps2026_ams.py:95–98`）；train 和 validation 的 mapping 来自同一 QC manifest `numeric_mapping`（`266–267`）。评价入口同样调用该函数（`evaluate_icsps2026_real.py:340,475`）。

`prepare_icsps2026_real.py:127–144` 实施 dataset-level affine mapping `(raw-offset)/scale`，拒绝超出 `[0,1]` tolerance 的数据，再执行保护性 clipping。`147–196` 的 mapping selection 支持 retained normalized float、division by 255 或显式统一 linear mapping，并记录 `per_image_percentile_normalization=False`。论文第 200 行与此正式入口相符。

`numeric_domain.py:76–91` 另有 `normalize_real_intensity` helper，其 docstring 声称用于 AMS/evaluation，但该 helper 实施逐图 1st/99th percentile normalization；**当前正式 AMS/evaluation 并未调用它**。因此不能把其名字/docstring 当作实际入口，不能在本图绘入 percentile stretching。该 helper 的文档描述与正式调用关系不一致，已记录且未修改。绘图无需给定 raw 文件的具体 offset/scale；图从已 normalized 的 `Y` 开始，不能未经 QC manifest 证据断言所有 raw 文件实际都采用同一种 0–255 scale。

### 5.4 科研结论边界

AMS 是以原始 noisy observation 为目标的 constrained masked calibration。当前论文第 155 行明确不主张 independence guarantee；本图不增加 “unbiased clean target”“blind-spot network”“noise-free self-supervision” 或 “independent noise pairs” 等未经实现/实验支持的标签。验证 target 仍是 noisy `Y`；fixed validation mask 的作用是使 checkpoint comparison 的 corruption/target positions 跨 epoch 一致。

## 6. 本图的材料充分性与验证边界

本图是基于已核准数学运算和训练协议的矢量流程示意，不含真实重建可视化或实验曲线，因此不需要重新训练、访问 clean/GT16、捏造输出图像或读取未提供的 best-epoch 数值。源码、正式配置与论文足以支持两个面板。图中采用图标或抽象 masked grid 时须保持示意含义，不把其像素布局当作真实样本或实际保存的 validation mask。

本审计属于静态源码/论文核验。绘图包中若另附 forward、inverse round-trip、mask/loss、冻结边界和导出检查日志，其运行结果应以对应日志为准，不将随机初始化结构检查称为正式实验性能复现，也不由结构检查推断论文指标、checkpoint 数值或训练完成状态。
