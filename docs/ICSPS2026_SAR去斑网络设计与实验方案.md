# ICSPS 2026 SAR 去斑：完整网络设计与实验方案

> 协议编号：`ICSPS26-FROZEN-v2`  
> 目标专题：ICSPS 2026 Special Session 5 — *Synthetic Aperture Radar Signal, Information Processing Technology and Application*  
> 目标篇幅：6 页英文双栏论文（含图、表和参考文献）  
> 文档日期：2026-09-05  
> 研究范围：单极化、单幅、强度域 SAR 去斑；**不包含舰船检测或任何下游检测实验**  
> 本文档性质：网络实现、正式实验和英文论文写作的统一执行规范；表格中的空白项必须由新协议下的正式运行填充。

> **v2 变更记录（2026-09-05）：** 正式监督训练集由 NWPU 25,200/6,300 train/validation 缩减为按类别分层的 **900/225**；最大训练预算由 200,000 缩减为 **100,000 optimizer updates**。训练 Gamma 噪声改为由冻结 schedule 驱动的确定性在线生成，验证噪声保持固定落盘。UCM-21 外部测试与真实 SAR 协议不变。旧 v1 结果不得与 v2 主表混用。

---

## 0. 一页结论

### 0.1 投稿约束

ICSPS 2026 官方页面将 SAR/PolSAR signal and image processing 明确列为 Special Session 5 的征稿方向，因此单独聚焦 SAR 去斑与该专题直接匹配。官方投稿说明要求英文全文、不少于 5 个双栏整页；超过 6 页需支付额外页费。当前官方投稿页给出的截止日期为 **2026-09-10**，会议采用双盲评审。[ICSPS Special Session 5](https://www.icsps.org/special5.html)；[Submission Instruction](https://www.icsps.org/sub.html)；[Peer Review](https://www.icsps.org/peer_review.html)。

> 官网不同缓存页面曾显示过旧日期，最终提交前必须再次以 `sub.html` 和 EasyChair 为准，并保存带日期的页面截图。

### 0.2 冻结决策

| 项目 | 最终决策 |
|---|---|
| 推荐标题 | **Representation-Consistent Fourier Refinement for TransSAR-Based SAR Despeckling** |
| 备选标题 | *TransSAR-Based Despeckling with Multi-Scale Fourier Refinement and Constrained Real-Data Adaptation*；仅在 AMS 结果稳定时使用 |
| 主干归属 | 明确写成 **TransSARV2-derived**；五阶段 Transformer 编码器及基本上采样/残差重建算子属于 TransSAR |
| 本文重点 | 多尺度全复频谱残差细化；严格的 log-to-intensity 逆映射与同域残差补偿；受 source-domain retention 约束的真实域后适应 |
| 主合成退化 | 仅使用全局 $L\in\{1,2,4,8\}$ 的 unit-mean Gamma speckle；其他退化全部关闭 |
| 监督开发集 | `NWPU-Small-1125-v1`：每类 20 张训练、5 张验证，共 900/225 个独立源图 |
| 监督训练预算 | batch 1；所有正式可学习方法统一执行 100,000 optimizer updates |
| 外部测试 | 完整 UCMerced Land Use 21 类、2,100 个 clean 源图，每图固定生成四个 $L$，共 8,400 对 |
| 真实 SAR | parent-disjoint Noisy-only 划分；AMS 只读取 train/validation Noisy；正式论文不使用 GT16，也不报告任何准参考指标 |
| 核心基线 | Noisy、Lee-MMSE、SAR-BM3D、SAR-CAM、真正的 TransSARV2-retrained、Ours |
| 核心消融 | Intensity-only、Log-only、Full w/o all FDR、Full |
| 论文之外 | 舰船检测、旧 mixed-L 结果、跨协议抄表、当前混杂的 `wout_msf`、未经原论文公式核对的真实域指标数字 |

### 0.3 最安全的论文中心句

> We adopt the publicly released TransSARV2 architecture as the backbone, retaining its five-stage hierarchical Transformer encoder and basic convolutional reconstruction operators. On top of this backbone, we introduce learnable concatenation-based skip fusion, multi-scale full-spectrum complex Fourier refinement, and an explicit log-to-intensity residual compensation path.

该表述把 TransSAR 归属、本文增量和正确的数值域链一次说清，避免宣称“全新 Transformer 主干”或含义不准确的“dual-domain fusion”。

---

## 1. 论文定位、研究问题与创新边界

### 1.1 任务定义

本文研究从一幅归一化的单通道 SAR 强度图恢复后向散射强度估计：

\[
\mathbf y=\mathbf x\odot\mathbf n,
\qquad
\mathbf n\sim\operatorname{Gamma}(L,1/L),
\qquad
\mathbb E[\mathbf n]=1,
\quad
\operatorname{Var}(\mathbf n)=1/L,
\]

其中 $\mathbf y$ 为观测强度，$\mathbf x$ 为待估计反射率，$L$ 为等效视数。Gamma 模型用于受控合成实验；真实聚焦 SAR 中 speckle 往往具有空间相关性与信号依赖性，不能把合成假设外推为真实数据的精确生成模型。该认识与 IEEE GRSM 的 SAR 去斑综述一致。[Fracastoro et al., 2021](https://doi.org/10.1109/MGRS.2021.3070956)

NWPU-RESISC45 和 UCMerced 本身是**光学遥感场景数据集**。论文必须写“optical remote-sensing images corrupted with simulated SAR-like speckle”，不得将其称为真实 SAR 数据。

### 1.2 三个研究问题

- **RQ1：** 在相同 TransSARV2 主干、训练数据和优化预算下，多尺度复频谱残差细化是否能稳定改善不同 $L$ 下的恢复质量？
- **RQ2：** 先预测归一化 log 表示、再解析逆变换并进行原始强度同域补偿，是否优于纯 intensity 路径和无补偿的 log-only 路径？
- **RQ3：** 在不使用真实 clean target 的前提下，encoder-frozen masked adaptation 能否在相同的真实 SAR test IDs 上改善预注册的真实域指标？

### 1.3 可检验假设

- **H1：** Full 相比 `TransSARV2-retrained` 和 `Full w/o all FDR` 在 UCM-21 的 macro PSNR/SSIM 上均改善，且 paired source-cluster bootstrap 的 95% CI 不跨 0。
- **H2：** Full 相比 Log-only 在高亮散射区误差和整体 PSNR/SSIM 上更好，证明补偿发生在正确数值域的价值。
- **H3：** Ours+AMS 相比 Ours-base 改善真实 test 的至少两项主要指标；AMS 前后稳定性只在相同真实 SAR test IDs 上比较。

若某一假设不成立，论文应如实收缩 claim，而不是更换指标或测试子集。

### 1.4 结果到主张的门槛

| 拟写入论文的主张 | 最低证据门槛 | 未达到时的处理 |
|---|---|---|
| FDR 带来有效频域细化 | Full 相对 `w/o-all-FDR` 在 UCM 类别宏平均 PSNR/SSIM 上取得一致改善，且配对置信区间不跨零 | 删除“FDR 导致提升”的因果表述；若核心证据不足，则从标题中移除 Fourier |
| 同域强度补偿缓解表示不一致 | Full 优于 Log-only，同时 Intensity-only 结果能解释收益并非单纯来自输出头变化 | 从标题和贡献项中移除 representation-consistent compensation，只保留为实现细节 |
| AMS 改善真实场景泛化 | AMS 后至少一项预注册真实 SAR 指标显著改善 | 不把 AMS 写入标题、摘要和主要贡献，只作为负结果或补充实验 |
| 达到或超过 SOTA | 与公开方法在同一输入、同一测试样本、同一后处理和同一统计口径下比较，并由配对置信区间支持 | 改写为 competitive performance 或只陈述具体指标差异 |

### 1.5 三项贡献的推荐英文表述

1. **Transparent backbone extension.**  
   *Starting from the official TransSARV2 implementation, we add full-spectrum complex Fourier residual refinement at multiple decoder resolutions while preserving a clear attribution boundary between the inherited backbone and our extensions.*

2. **Representation-consistent reconstruction.**  
   *The bounded log-domain estimate is analytically mapped back to normalized intensity before being combined with an original-intensity residual, avoiding direct fusion between incompatible numerical representations.*

3. **Constrained target-domain calibration.**  
   *A lightweight encoder-frozen masked adaptation stage updates the reconstruction modules using only real intensity observations, with checkpoint selection based solely on the fixed real-SAR validation split.*

第三项仅在完整真实实验稳定后进入摘要和贡献列表；否则将 AMS 降为 supplementary-style analysis 或删除。

### 1.6 近期文献对原创性措辞的约束

| 相关工作 | 已覆盖内容 | 对本文的直接约束 | 是否做数值基线 |
|---|---|---|---|
| TransSAR, IGARSS 2022 | Transformer encoder、五阶段层次特征、CNN projection decoder | 主干必须归属 TransSAR；不能称本文首创 Transformer SAR 去斑 | **必须同协议重训** |
| SAR-CAM, JSTARS 2022 | 多尺度 encoder-decoder、attention、context block | 不能将“多尺度注意力”单独作为原创 | 合成表优先加入 |
| SAR-FDD, 2024 | 高频/低频分解与交互双分支 | 不能称首个频域 SAR 去斑 | 引用；无官方代码不抄数字 |
| DiffusionSAR, TGRS 2024 | log-Yeo–Johnson 变换、Gamma-oriented diffusion、real-data fine-tuning | 不能称首个 log 变换或真实域微调 | 引用 |
| DAT-Net, Remote Sensing 2025 | 动态门控、FFT 高频/低频专家、多尺度 Transformer encoder-decoder | “gate + Fourier + multiscale Transformer”的宽泛组合已存在 | **必须引用**；不跨协议抄数字 |
| Noise-Guided Transformer, 2025 | learned noise prior 引导与多尺度融合 | 未监督 guidance map 不可包装成首个 noise-guided Transformer | 引用 |
| Speckle2Self, 2025 | log intensity、互补掩码、自监督学习、相关性处理 | 当前简单 masking 不能声称首个或理论无偏 | 引用 |
| SDS-SAR, ISPRS JPRS 2026 | intensity-only 自监督理论、RA-SAMPLE 独立训练对 | AMS 只能定位为实用 post-adaptation，不是新自监督理论 | 真实表首选现代基线 |
| GLCNet/LGCN, 2026 | wavelet/frequency-spatial、blind spot、global-local fusion | 不能称首个 frequency-spatial self-supervision | 引用；仓库完整可跑后才比较 |
| Integrated gradient–spatial–frequency method, SPIC 2026 | 显式融合 gradient、spatial、frequency 三类信息 | 不能把一般性的 spatial-frequency fusion 声称为领域首创 | 引用；未发现作者官方代码 |

因此，安全的差异化不是“用了 FFT/门控/Transformer”，而是：

> **在 TransSAR-derived progressive decoder 的多个尺度上，对完整 rFFT 复频谱进行共享通道残差变换，并在解析 inverse-log 后执行 representation-consistent intensity compensation。**

---

## 2. TransSAR 归属与实现边界

### 2.1 必须写入论文的归属

TransSAR 论文由 Malsha V. Perera 等发表于 IGARSS 2022，其官方仓库以 MIT License 发布。[论文](https://doi.org/10.1109/IGARSS46834.2022.9884596)；[arXiv](https://arxiv.org/abs/2201.09355)；[官方代码](https://github.com/malshaV/sar_transformer)

本地代码审计表明，`EncoderTransformerV2`、`OverlapPatchEmbed`、`Mlp`、`Attention`、`Block`、`DWConv`、`convprojection_baseV2`、`TransSARV2`、`ConvLayer`、`UpsampleConvLayer` 和 `ResidualBlock` 与官方实现对应。因此：

本次归属审计以官方仓库提交 [`b3ac845`](https://github.com/malshaV/sar_transformer/commit/b3ac845f96f2332aa4f1af94b455f71630978b17) 为比对基准；投稿归档时应保存该 commit、本文仓库 commit 和保留的 MIT License。

- 五阶段 Transformer encoder、overlap patch embedding、spatial-reduction attention、DWConv-MLP 均为继承项；
- 通道数、深度、head 数和 SR ratio 均为继承配置；
- progressive upsampling、基础 ResidualBlock 和原始 additive skip 思路属于 TransSAR reconstruction framework；
- 本文不能写 “we propose a novel Transformer encoder-decoder”。

### 2.2 继承项与本地扩展项

| 类型 | 模块 |
|---|---|
| 继承自 TransSARV2 | 五阶段 encoder；overlap embedding；SR attention；DWConv-MLP；stage widths/depths；转置卷积上采样；基础 residual reconstruction operators |
| 本地实现扩展 | concat-based learned skip fusion；`FFTRefineBlock`；`BottleneckRefine`；guidance-map generator；guidance-conditioned modulation；normalized log/inverse-log；intensity residual compensator；AMS |

“本地扩展”仅表示其不存在于 TransSAR 官方仓库，不等于文献意义上的首次提出。投稿仓库必须保留原 MIT License 与版权声明。

### 2.3 架构图归属规范

- 灰色：TransSARV2 继承模块；
- 蓝色：本文 frequency refinement；
- 橙色：本文 log/inverse-log 与 intensity compensation；
- 绿色：guidance modulation；
- 图注必须写：*Gray components are inherited from TransSARV2, while colored components denote our extensions.*

---

## 3. 完整网络设计

### 3.1 正式模型实例

正式论文只对应以下配置：

```python
TransSARV2_DualFreqNG_Bottle(
    ablation="full",
    numeric_domain="intensity_v1",
)
```

类名中的 `Dual` 是历史命名。论文中统一称：

> **frequency-refined TransSAR extension with log-to-intensity residual compensation**

### 3.2 端到端链路

```text
normalized noisy intensity y
    -> clamp to [0, 1]
    -> normalized log1p T_alpha(y), alpha=10
    -> shared input to TransSARV2 encoder and guidance-map generator
    -> bottleneck spatial-frequency refinement
    -> five-stage progressive decoder:
       first four stages: upsample -> same-scale skip concat/fusion -> FDR -> ResidualBlock -> guidance modulation
       final full-resolution stage: upsample -> FDR -> ResidualBlock -> guidance modulation
    -> 3x3 head + sigmoid gives estimated normalized log intensity z_hat
    -> exact inverse T_alpha^{-1}(z_hat)
    -> intensity residual compensation conditioned on [y, x_tilde]
    -> clamp to [0, 1]
    -> restored normalized intensity x_hat
```

三个不可颠倒的事实：

1. guidance-map generator 的输入是 **log intensity**；
2. inverse-log 在 residual compensation **之前**；
3. 最终正式路径没有 Tanh，只有 log head 的 Sigmoid 和最终 clamp。

### 3.3 数值域变换

对归一化输入 $\mathbf y$ 先做：

\[
\mathbf y_{01}=\operatorname{clip}(\mathbf y,0,1).
\]

使用 $\alpha=10$ 的有界 `log1p`：

\[
\mathbf z=T_\alpha(\mathbf y_{01})
=\frac{\log(1+\alpha\mathbf y_{01})}{\log(1+\alpha)}.
\]

该映射把 ([0,1]) 精确映射到 ([0,1])，解析逆变换为：

\[
T_\alpha^{-1}(\widehat{\mathbf z})
=\frac{\exp\!\left(\widehat{\mathbf z}\log(1+\alpha)\right)-1}{\alpha}.
\]

但

\[
\log(1+\alpha xn)\neq\log(1+\alpha x)+\log n,
\]

所以不可声称该 normalized `log1p` “严格把乘性噪声转成加性 Gaussian noise”。推荐表述：

> *The normalized log transform compresses the dynamic range and partially reduces the signal dependence of speckle.*

### 3.4 TransSARV2 五阶段编码器

对 $1\times256\times256$ 的 $\mathbf z$，encoder 输出：

| Stage | Channels | Resolution | Heads | Depth | MLP ratio | SR ratio |
|---:|---:|---:|---:|---:|---:|---:|
| $E_1$ | 32 | $128\times128$ | 1 | 3 | 2 | 8 |
| $E_2$ | 64 | $64\times64$ | 1 | 3 | 2 | 8 |
| $E_3$ | 128 | $32\times32$ | 2 | 4 | 2 | 4 |
| $E_4$ | 320 | $16\times16$ | 4 | 6 | 2 | 2 |
| $E_5$ | 512 | $8\times8$ | 8 | 3 | 2 | 1 |

这组配置不得列为本文创新。

### 3.5 Bottleneck refinement

最深层特征 $\mathbf E_5$ 经过局部分支和 FDR 分支：

\[
\mathbf B_l=\operatorname{GELU}
\left(W_l^{1\times1}*\operatorname{DWConv}_{3\times3}(\mathbf E_5)\right),
\]

\[
\mathbf B_f=\operatorname{FDR}(\mathbf E_5),
\]

\[
\widetilde{\mathbf E}_5
=\mathbf E_5+\gamma_b W_{b,2}^{3\times3}*
\operatorname{GELU}\left(W_{b,1}^{1\times1}*(\mathbf B_l+\mathbf B_f)\right),
\]

其中可学习标量 $\gamma_b$ 初始化为 0.1。由于 `FDR` 自身已有 identity residual，$\mathbf B_f$ 不是“纯频域增量”。

### 3.6 Progressive decoder

前四个具有 encoder skip 的解码级严格采用：

\[
\text{Upsample}
\rightarrow \text{same-scale skip concat}
\rightarrow \text{FusionBlock}
\rightarrow \text{FDR}
\rightarrow \text{ResidualBlock}
\rightarrow \text{Guidance Modulation}.
\]

最后的全分辨率级没有 encoder skip 和 FusionBlock，其顺序为 `Upsample → FDR → ResidualBlock → Guidance Modulation`。

| Decoder stage | 输入与 skip | 输出尺寸 | FDR | Gate |
|---|---|---:|:---:|:---:|
| $D_4$ | $512@8^2\rightarrow320@16^2$, concat $E_4$ | $320@16^2$ | ✓ | ✓ |
| $D_3$ | $320@16^2\rightarrow128@32^2$, concat $E_3$ | $128@32^2$ | ✓ | ✓ |
| $D_2$ | $128@32^2\rightarrow64@64^2$, concat $E_2$ | $64@64^2$ | ✓ | ✓ |
| $D_1$ | $64@64^2\rightarrow32@128^2$, concat $E_1$ | $32@128^2$ | ✓ | ✓ |
| $D_0$ | $32@128^2\rightarrow16@256^2$，无 encoder skip | $16@256^2$ | ✓ | ✓ |
| Head feature | stride-1 transpose convolution | $8@256^2$ | – | – |

Concat fusion 对单个同分辨率 skip 进行学习融合，并非一次聚合全部尺度。论文名称用 *concatenation-based learned skip fusion*，不要称“首次提出 multi-scale fusion”。

### 3.7 Full-spectrum Fourier residual refinement（FDR）

给定 $\mathbf F\in\mathbb R^{B\times C\times H\times W}$，空间分支为：

\[
\mathbf F_s=\operatorname{GELU}
\left(W_s^{1\times1}*\operatorname{DWConv}_{3\times3}(\mathbf F)\right).
\]

频域分支先做正交归一化二维实 FFT：

\[
\mathbf Q=\operatorname{rFFT2}(\mathbf F)
\in\mathbb C^{B\times C\times H\times(\lfloor W/2\rfloor+1)}.
\]

拼接实部和虚部：

\[
\mathbf Q_c=[\Re(\mathbf Q),\Im(\mathbf Q)]
\in\mathbb R^{B\times2C\times H\times(\lfloor W/2\rfloor+1)}.
\]

用共享的 real-valued $1\times1$ 卷积进行谱通道混合：

\[
\widetilde{\mathbf Q}_c=
W_{f,2}^{1\times1}*
\operatorname{GELU}\left(W_{f,1}^{1\times1}*\mathbf Q_c\right).
\]

拆分为实部/虚部并逆变换：

\[
\mathbf F_f=\operatorname{irFFT2}
\left(\widetilde{\mathbf Q}_{\Re}+j\widetilde{\mathbf Q}_{\Im}\right).
\]

最后融合并残差输出：

\[
\operatorname{FDR}(\mathbf F)=\mathbf F+
W_{o,2}^{3\times3}*\operatorname{GELU}
\left(W_{o,1}^{1\times1}*[\mathbf F_s,\mathbf F_f]\right).
\]

完整模型包含六个 FDR：bottleneck 一个，decoder 的 $16^2,32^2,64^2,128^2,256^2$ 五个。

必须遵守以下技术边界：

- 可称 *full-spectrum complex Fourier refinement* 或 *real–imaginary spectral channel refinement*；
- 不可称 complex-valued convolutional network，所有学习卷积均为实数卷积；
- 不可称 magnitude-phase processing，实现处理的是 real/imaginary parts；
- 不可称 high-frequency enhancement，代码没有高低频分带；
- $1\times1$ 权重在频率坐标间共享，只混合当前频点的通道与实虚分量，不能声称学习相邻频率关系或 frequency-selective filter。

### 3.8 Learned guidance map 与特征调制

同一 log input $\mathbf z$ 输入四层卷积生成器：

\[
\mathbf P=G_\theta(\mathbf z)\in[0,1]^{B\times1\times H\times W}.
\]

每个 decoder 尺度将 $\mathbf P$ 双线性缩放后，与特征拼接：

\[
\mathbf g_s=\tanh\left(C_{g,2}^{3\times3}
\left(\operatorname{GELU}\left(C_{g,1}^{1\times1}
([\mathbf D_s,\mathbf P_s])\right)\right)\right),
\]

\[
\mathbf D'_s=\mathbf D_s\odot(1+\gamma_s\mathbf g_s),
\]

其中 $\gamma_s$ 初始化为 0.1，但训练后没有范围约束。因此只可称 *learned guidance-map generator* 和 *guidance-conditioned feature modulation*；没有 noise-map 标签或 $L$ 监督，不能称“noise estimator”“estimated speckle map”或“look-number estimator”。

### 3.9 Log prediction、inverse-log 与强度补偿

Decoder head 输出：

\[
\widehat{\mathbf z}=\sigma(C_{\rm head}(\mathbf D_0)).
\]

先执行精确逆映射：

\[
\widetilde{\mathbf x}=T_\alpha^{-1}(\widehat{\mathbf z}).
\]

再由轻量补偿器接收两个**同处 intensity domain** 的输入：

\[
\mathbf r_c=F_{\rm comp}([\mathbf y_{01},\widetilde{\mathbf x}]),
\]

\[
\widehat{\mathbf x}=\operatorname{clip}_{[0,1]}
\left(\widetilde{\mathbf x}+\gamma_c\mathbf r_c\right),
\]

其中 $F_{\rm comp}$ 为 `2→32→32→1` 的三层 $3\times3$ CNN，前两层后接 GELU，$\gamma_c$ 初始化为 0.1。

因此当前 `DualDomainFusion` 类在论文中必须改称：

> *intensity-domain residual compensator* 或 *log-to-intensity residual compensation*

不能写成两条完整 restoration branches，也不能写成 log/intensity feature fusion。

### 3.10 监督目标

统一在归一化强度域监督：

\[
\mathcal L_{\rm sup}
=\underbrace{\frac{1}{HW}\|\widehat{\mathbf x}-\mathbf x\|_2^2}_{\mathcal L_{\rm MSE}}
+\lambda_{\rm TV}^{\rm sup}\underbrace{\operatorname{TV}(\widehat{\mathbf x})}_{\text{mean TV}}.
\]

该损失是标准重建损失，不列为创新。$\lambda_{\rm TV}^{\rm sup}$ 的候选集、验证依据和最终取值统一在第 8.1 节说明。

### 3.11 AMS：受约束真实域后适应

训练时以 0.2 的概率独立生成 Bernoulli 掩膜，只在被遮挡位置回归原 noisy observation。掩膜输入、预测强度与目标函数统一写为：

\[
\begin{gathered}
\mathbf y_m=\mathbf y\odot(1-\mathbf M),\qquad
\widehat{\mathbf x}_m=f_\theta(\mathbf y_m),\\
\mathcal L_{\rm AMS}
=\frac{\|\mathbf M\odot(\widehat{\mathbf x}_m-\mathbf y)\|_1}
{\max(\|\mathbf M\|_1,1)}
+\lambda_{\rm AMS}\operatorname{TV}(\widehat{\mathbf x}_m).
\end{gathered}
\]

冻结 `model.log_branch.Tenc`，其余 bottleneck、decoder、guidance、head 与 compensation 模块仍可更新。准确名称是 **encoder-frozen masked post-adaptation**，而不是严格的 decoder-only training 或 blind-spot network。

每个候选权重的运行仍在固定 real-validation masks 上按 masked loss 最低选择 epoch；跨权重的选择规则见第 8.3 节。

AMS 没有解决聚焦 SAR speckle 的空间相关性，也没有证明 masked target 与输入独立，故不能声称理论无偏。Speckle2Self 与 SDS-SAR 已专门讨论自监督与相关性问题，本文应把 AMS 定位为轻量部署校准。

### 3.12 源码映射

| 论文对象 | 当前源码位置 | 状态/注意事项 |
|---|---|---|
| 正式 wrapper | `transform_main.py::TransSARV2_DualFreqNG_Bottle` | Full 与正式消融统一入口 |
| 原始 TransSARV2 baseline | `transform_main.py::TransSARV2` | `model_registry.py` 直接构造真正 baseline |
| TransSAR-derived branch | `transform_main.py::TransSARV2_FreqNG_Bottle` | guidance 输入为 log image |
| FDR | `transform_main.py::FFTRefineBlock` | real/imag 拼接，非 mag/phase |
| Bottleneck | `transform_main.py::BottleneckRefine` | 内含一个 FDR |
| Guidance generator | `transform_main.py::NoiseEstimator` | 论文不要沿用历史类名 |
| Intensity compensator | `transform_main.py::DualDomainFusion` | 历史类名；实际是同域补偿 |
| Guidance gate | `transform_main.py::NoiseGuidedGate` | $\gamma$ 无范围约束 |
| Decoder | `transform_main.py::convprojection_freq_ng` | 五级 progressive decoder |
| Log/inverse-log | `numeric_domain.py:94` | $\alpha=10$ |
| Ablation presets | `ablation_config.py::FORMAL_VARIANTS` | 已锁定 Full、Intensity-only、Log-only、w/o-all-FDR |
| 统一训练/评测 | `train_icsps2026.py`、`evaluate_icsps2026_synthetic.py` | update-based、strict checkpoint、固定 manifest |
| Real metrics | `evaluate_icsps2026_real.py`、`classic_sar_metrics.py` | 论文只使用 ENL、文献定义的 M 与 EPI；任何内部 ratio/ACF/Sobel 诊断都不进入论文表，M/EPI 实现须先通过原公式核对 |
| AMS | `train_icsps2026_ams.py` | validation mask seed/hash 按 sample 固定并随 run 归档 |

---

## 4. 实验总体设计

### 4.1 证据层级

| 层级 | 数据与目的 | 是否用于主结论 |
|---|---|:---:|
| Development validation | NWPU Gamma-only validation；按固定 update grid 选 checkpoint、调 Lee 窗口 | 否，只报告协议与必要附录数字 |
| Primary external synthetic test | UCM-21 Gamma-only；检验跨数据集、跨 $L$ | **是，合成主表** |
| Real parent-disjoint test | 真实 Noisy-only；base 与 AMS、ratio image 评价 | **是，真实主表** |
| Compound stress test | 现有 global-L v2 的多退化数据 | 可选，只给一个 stress macro 列 |
| Historical/smoke | 中期报告旧数字、mixed-L、短 smoke run | **否** |

禁止在 UCM 或 real test 上选 checkpoint、调窗口、选择 ROI 规则、选择显示样例或决定指标参数。

### 4.2 数据角色汇总

| 数据 | 原始性质 | clean 源数 | noisy pairs | 角色 |
|---|---|---:|---:|---|
| NWPU train | 光学遥感 | **900（45×20）** | 100,000 次冻结 schedule 中的在线 pair presentation；不预存 noisy 数组 | 监督训练 |
| NWPU validation | 光学遥感 | **225（45×5）** | **900**；每源四个固定 $L$ | 选择与调参 |
| UCM-21 | 光学遥感 | 2,100（21×100） | 8,400；每源四个 $L$ | 外部合成测试 |
| Mendeley `fs455tz88y.1` | 真实 SAR intensity；传感器/构建元数据需随下载版本锁定 | 1,469 个 512 parent | 5,876 个 256 child patches | 真实域训练/验证/测试 |

NWPU-RESISC45 官方论文确认其含 31,500 张、45 类、每类 700 张图像。[Cheng et al., 2017](https://doi.org/10.1109/JPROC.2017.2675998)  UCMerced 数据由 UC Merced Computer Vision Lab 发布。[UCM dataset page](https://vision.ucmerced.edu/datasets/)

表中的 clean 源数、pair presentation 数和 optimizer update 数是三个不同口径。在线重新采样 Gamma 噪声不会增加独立场景数；100,000 次训练输入也不得写成“100,000 张独立训练图像”。相对 v1，独立 train/validation 源图均减少 96.43%，而最大更新次数减少 50%。

---

## 5. 合成实验协议

### 5.1 新建 Gamma-only 数据版本

现有 `NWPU_RESISC45_SAR_global_L_v2` 在 Gamma speckle 后还加入 clean-dependent gain、bright impulses、Gaussian noise、shift mixing 和条纹，不能作为“仅由 $L$ 控制”的主实验。

必须新建且不得覆盖旧数据。v2 使用独立的数据/manifest 标识：

```text
datasets/NWPU_RESISC45_SAR_gamma_only_small1125_v1/
```

对 RGB optical source (I)：

1. 校验源图为原生 $256\times256$，正式生成时不重采样；若发现非预期尺寸则 fail 并审计来源；
2. 灰度化 $G=0.299R+0.587G+0.114B$，并确定性映射到 $[0,1]$；
3. clean intensity proxy 定义为 $X=G^2$；
4. 采样 $N\sim\operatorname{Gamma}(L,1/L)$；
5. 得到 $Y=\operatorname{clip}(XN,0,1)$；训练 pair 按冻结 schedule 在线生成，validation/test noisy arrays 固定落盘；
6. 对训练 schedule 和固定 validation/test pairs 记录 clipping 前后最大值及 `saturation_rate = mean(X*N > 1)`。

论文应写“unit-mean Gamma before clipping”。不要把 clip 后噪声仍描述为严格 unit-mean。

### 5.2 NWPU 划分与随机性

v2 的 development pool 命名为 `NWPU-Small-1125-v1`。从已经修复 exact-content leakage 的 NWPU master split 中进行确定性分层子采样：

该规模不是事后按结果选择：DAT-Net 报告从 NWPU 每类取 25 张并作 8:2 development split，对应约 900/225；原始 TransSAR 仅使用 450/50 张 BSD 图像。v2 因而采用与近期 NWPU Transformer 工作一致、同时比 TransSAR 更保守的 900/225 固定分层协议。[DAT-Net](https://doi.org/10.3390/rs17173031)；[TransSAR](https://arxiv.org/abs/2201.09355)

- master `split_seed=42`，独立的 `subset_seed=20260905`；抽样前先按规范化相对路径排序；
- 从 master-train 每类固定抽取 20 张，共 **900 train unique sources**；
- 从 master-validation 每类固定抽取 5 张，共 **225 validation unique sources**；
- 每类其余 675 张不进入 v2 development protocol，loader 必须通过 manifest 白名单拒绝读取；
- 固定保存 `source_manifest.csv`；正式实验期间不得换图、补图或逐 epoch 重划 train/validation；
- source path 与 source-content SHA-256 在 train/validation 间的交集均为 0；manifest 另保存 selection rank、parent split 和自身 SHA-256。

训练采用 **deterministic online Gamma**，不为 900 张源图只生成一次固定噪声。每个正式 run 在开始前生成长度为 100,000 的 `train_schedule.csv`：

```text
global_step, source_id, class_name, source_occurrence,
L, speckle_seed, d4_id, protocol_id, run_seed
```

冻结约束：

- 四个 $L\in\{1,2,4,8\}$ 在 100,000 行中各出现 **25,000** 次；
- 900 个 source 的出现次数尽可能平衡，每个 source 恰好出现 111 或 112 次；各类别总呈现次数之差不超过 1；
- `speckle_seed` 由 `stable_hash(synthesis_seed, run_seed, source_id, source_occurrence, L)` 生成，不依赖目录遍历或 worker 数；
- `d4_id` 仅指定无插值的 paired dihedral transform；恢复训练必须从 checkpoint 的下一 schedule 行继续；
- 同一 `run_seed` 下，SAR-CAM、TransSARV2、Ours 和所有消融逐 `global_step` 使用同一 source、$L$、Gamma realization 与 D4；
- `synthesis_seed=20260904`；主 run `run_seed=42`，补充 run 为 43、44。

validation 的每个 source 固定生成四个 $L$，即每个 $L$ 225 对、总计 **900 fixed pairs**；不做随机增广。验证种子定义为：

```text
seed_i = stable_hash(validation_seed, source_id, L)
validation_seed = 20260904
```

source 与固定 validation manifest 至少保存：

```text
pair_id, source_id, source_relative_path, class_name, split,
global_L, clean_path, noisy_path, rng_seed, selection_rank, parent_split,
source_sha256, clean_sha256, noisy_sha256,
preclip_max, saturation_rate, numeric_domain, protocol_id
```

训练项不保存 `noisy_path`，而由 `train_schedule.csv` 的 `speckle_seed` 在线重建；固定 validation/test 项必须保存 `noisy_path` 与 `noisy_sha256`。训练 schedule 和 source/validation manifest 的 SHA-256 均写入 run config 与 checkpoint。

### 5.3 UCM-21 外部测试

- 完整使用 21 类 × 100 图，不只选 3 类；
- 每个 source 固定生成 $L=1,2,4,8$，共 8,400 pairs；
- `synthesis_seed=20260905`；
- clean 与 noisy 处理必须与 NWPU 一致；
- 在生成前核对 UCM 与 NWPU train/validation 的 exact-content SHA-256 交集为 0，并以 perceptual hash 只做近重复审计；
- UCM 不参与模型、checkpoint/update step、超参、Lee window 或可视化样例筛选；
- 3 个场景类只可作为预注册后的展示样例，不能代替全量测试；
- 报告每个 $L$ 与四个 $L$ 的不加权 macro；另输出 class-macro 供审计。

### 5.4 Compound stress test

把现有 `global_L_v2` 明确改称：

> **compound-degradation stress validation**

它包含：1.20× strong-target gain、稀疏高亮 impulse、$\sigma=0.015$ additive Gaussian、`(1,2)` shift 与 0.08 mixing、行列偏置、低频条纹、多次 clipping。

使用规则：

- Gamma-only 训练的 checkpoint 直接推理，不在 stress data 上重训；
- 正文空间紧时只报告 `TransSARV2` 与 `Ours` 的 stress macro；
- 不能用它支持“对不同视数的纯 speckle 建模”结论；
- 原 4×4 mixed-L 结果从论文删除。

---

## 6. 消融实验：必须先修正语义

### 6.1 当前 preset 的真实含义

| 当前 preset | 代码实际关闭内容 | 可否直接按旧名称发表 |
|---|---|---|
| `full` | 无 | 可以 |
| `wout_frequency` | 只关闭 decoder 五个 FDR；bottleneck FDR 仍在 | 只能叫 **w/o decoder FDR** |
| `wout_gate` | guidance generator + 五个 gates | 基本为单因素，但不是标题贡献 |
| `wout_fusion` | 只关闭 inverse-log 后的 intensity compensator | 改名 **w/o intensity compensation / Log-only** |
| `wout_msf` | 同时移除 4 个 FusionBlock、5 个 decoder ResidualBlock，并把 concat skip 改成 addition | **不可作为单因素 MSF 消融** |
| `wout_bottleneck` | 整个 bottleneck refine，包括 local 与 FDR | 是组合消融，不等于只去 FFT |

### 6.2 正式核心消融矩阵

| Variant | Network input | Log transform | Inverse-log | Intensity compensation | Bottleneck FDR | Decoder FDR | 目的 |
|---|---|:---:|:---:|:---:|:---:|:---:|---|
| Intensity-only | intensity | × | – | × | ✓ | ✓ | 判断 log 表示是否有益 |
| Log-only | log intensity | ✓ | ✓ | × | ✓ | ✓ | 隔离同域补偿价值 |
| Full w/o all FDR | log intensity | ✓ | ✓ | ✓ | × | × | 隔离全部 Fourier refinement |
| Full | log intensity | ✓ | ✓ | ✓ | ✓ | ✓ | 完整方法 |

主比较表另列真正 `TransSARV2-retrained`；它不是任一 proposed-model ablation 的别名。

正式代码已实现并用测试锁定两项结构开关：

1. 给 `BottleneckRefine` 添加可独立关闭 FDR 的开关，使 `Full w/o all FDR` 同时关闭 6 个 FDR；
2. 增加 intensity-only representation 路径，保持 decoder 容量、head 和训练预算不变。

### 6.3 辅助消融

若篇幅和算力允许，仅以 seed 42 放入补充表：`w/o gate`、`w/o bottleneck`。当前 `wout_msf` 不修正则完全删除。不要做 mask-ratio sweep、未见 $L=3,6$ 或大量组合消融来挤占主证据。

---

## 7. 基线协议

### 7.1 合成主表

| 方法 | 来源 | 使用方式 | 公平性标记 |
|---|---|---|---|
| Noisy | – | 原始 noisy input | – |
| Lee-MMSE | 传统局部统计滤波 | 窗口从 `{5,7,9,11}` 在 NWPU validation 选一个全局值 | 明确不是 edge-directed Refined Lee；使用 nominal $L$ 时标 `oracle-L` |
| SAR-BM3D | TGRS 2012 | 作者 MATLAB 实现；强度开方后输入，输出平方回强度；输入 nominal $L$ | `oracle-L` |
| SAR-CAM | JSTARS 2022 | 官方 PyTorch；同 Gamma-only train/val、同 update budget 重训 | `blind, retrained` |
| TransSARV2 | IGARSS 2022 | 本地真正 `TransSARV2` 类；同协议重训 | `blind, retrained` |
| Ours | 本文 | 一个模型覆盖四个 $L$ | `blind, one model` |

`TransSARV2-retrained` 的正式 runner 必须直接实例化 `transform_main.py:1792` 的原始类，不经过本文的 log、FDR、guidance 或 compensator。它接收/监督同一 `[0,1]` intensity pairs，保留官方类中的 Tanh head；训练 loss 作用于 raw output，保存图像和统一指标前才 clip 到 `[0,1]`。若另将 head 改为 Sigmoid，必须单独命名 `TransSARV2-Sigmoid`，不能用它替代“原始 TransSARV2”而不说明修改。

SAR-BM3D 的作者发布包由 University of Naples Federico II GRIP 提供；正式 wrapper 锁定 2013-07-31 的 v1.0 Linux 包，并单独记录下载档案 SHA-256、`SARBM3D_v10.m` SHA-256 与 nonprofit-only license 接受情况。作者函数声明输入和输出均为 square-root intensity，因此唯一正式链为：归一化线性强度 $Y\in[0,1]$ 开方成 amplitude，调用 `SARBM3D_v10(sqrt(Y),L)`，把返回的 fixed-point amplitude 平方回 raw intensity，最后只在统一指标入口 clip 到 $[0,1]$；不得把 intensity 直接送入作者函数。[论文](https://doi.org/10.1109/TGRS.2011.2161586)；[作者 v1.0 README](https://www.grip.unina.it/download/prog/SAR-BM3D/version_1.0/README.txt)；[作者 v1.0 代码档案](https://www.grip.unina.it/download/prog/SAR-BM3D/version_1.0/)  SAR-CAM 有作者官方训练与测试脚本，正式 runner 锁定作者仓库 commit `ea5ee3bed00ab22735a7c87518fe5388c2d6c49a` 并使用官方 `channels=128` 构造参数。[论文](https://doi.org/10.1109/JSTARS.2021.3132027)；[官方仓库](https://github.com/JK-the-Ko/SAR-CAM)

### 7.2 真实 SAR 表

最低配置：Noisy、TransSARV2、Ours Log-only、Ours-base、Ours+AMS。Log-only 用于在真实域支撑 H2。本文不使用 GT16，且没有独立依据冻结真实数据的 effective look，因此 Lee-MMSE 与 SAR-BM3D 不进入真实主表；二者在合成 UCM 主表仍是必须基线。

现代真实域基线优先顺序：

1. **SDS-SAR**：输入条件最匹配，官方仓库现有源码目录、预训练权重与数据链接，但 README 说明代码仍会逐步更新；完成 end-to-end smoke test 并锁定 commit 后才纳入。[论文](https://doi.org/10.1016/j.isprsjprs.2025.11.025)；[仓库](https://github.com/YYF121/SDS-SAR)
2. **CL-SAR**：有官方代码和 checkpoint；若用 pretrained weight，必须注明其外部训练数据，不可声称同训练数据公平比较。[论文](https://doi.org/10.1016/j.isprsjprs.2024.11.003)；[仓库](https://github.com/YangtianFang2002/CL-SAR-Despeckling)
3. **GLCNet/LGCN**：当前仓库 README 仍写训练、测试流程和权重待发布；只引用，除非 smoke test 证明确实可完整复现。[论文](https://doi.org/10.1016/j.jag.2026.105135)；[仓库](https://github.com/yangyang12318/LGCN)

可把 [Practical SAR Despeckling Codes](https://github.com/YangtianFang2002/Practical-SAR-Despeckling-Codes) 作为兼容性和复现实用参考，但其中标为 `PyTorch Re` 的项目是第三方复现，表中必须与作者官方实现区分。

### 7.3 不做强行数值比较的方法

- MERLIN、Sublook2Sublook：需要 complex SLC 或 sublooks；
- SAR2SAR：需要多时相 noisy pairs；
- Speckle2Void：正式流程含 decorrelation，与当前未做相同预处理的 intensity input 不等价；
- 无作者可运行代码的 SAR-FDD、DAT-Net、DiffusionSAR、Noise-Guided Transformer：引用其思想和 claim boundary，不抄论文表中的异协议数字。

---

## 8. 训练与模型选择

### 8.1 统一监督训练配置

| 项目 | 冻结值 |
|---|---|
| 输入/crop | $1\times256\times256$ normalized intensity |
| Batch size | 1 |
| Train sources | 900（45 类×20）；由冻结 `source_manifest.csv` 白名单读取 |
| Validation | 225 sources × 4 fixed $L$ = 900 fixed pairs |
| Train augmentation | paired dihedral transforms：水平/垂直翻转与 0°/90°/180°/270° 整数旋转；不插值 |
| Optimizer | Adam |
| Initial LR | $10^{-3}$ |
| Weight decay | $10^{-5}$ |
| LR scheduler | `ReduceLROnPlateau(mode=min, factor=0.5, patience=4, min_lr=1e-6)`；patience 按完整验证事件计数，并按 `val_macro_mse` 更新 |
| Loss | MSE + $\lambda_{\rm TV}^{\rm sup}$ mean-TV |
| TV 权重选择 | 在 synthetic validation 比较 $\{0,10^{-2},3\times10^{-2},5\times10^{-2},8\times10^{-2},10^{-1}\}$；$3\times10^{-2}$ 的 macro PSNR 和 SSIM 最高，故选定并固定用于全部监督测试 |
| 固定正式预算 | **100,000 successful optimizer updates** |
| 完整验证 | 每 **5,000** updates，共 20 次；另在 step 0 记录初始模型指标 |
| 选择规则 | NWPU validation 上 post-clip linear-intensity 的最低 `val_macro_mse`；TV 项不参与选模；并列时选更早 step |
| 正式 seeds | 42；时间允许再补 43、44 |
| Metric domain | linear normalized intensity，`data_range=1`，border=0 |

`val_macro_mse` 先在每个 source/$L$ 上计算，再对 45 个类别和四个 $L$ 等权汇总。UCM 和真实 SAR 绝不参与 checkpoint 选择。每个 run 同时保存 `checkpoint_best.pth` 与 `checkpoint_last.pth`；checkpoint 必须包含模型、optimizer、scheduler、global step、RNG 状态和 train-schedule SHA-256。

所有可学习方法和消融必须完整执行 100k successful updates，并得到相同的 900/225 source manifest、同一 `run_seed` 对应的 schedule/Gamma realization/D4、validation grid 与 selection rule；不做 method-specific early stopping，也不让某个方法因为收敛慢而得到更多训练。传统方法只允许在固定 NWPU validation 上选择一次全局参数。

旧 `BSD_SAR` 的任意角度旋转与 affine augmentation 会对 clean/noisy 同时插值，并改变在线 Gamma realization 的空间统计。正式 runner 已改为只做无插值的 paired dihedral transforms；validation/test 不做随机 augmentation。

batch 1 下，100k updates 等于 100k 次样本呈现，约为 $100000/900=111.11$ 次名义 source-pool 遍历；论文应报告 updates，不写成整数 epochs。开始正式矩阵前先用 1,000 updates 测 wall-clock。若 100k × 核心配置仍无法在截止日前完成，必须先升版为 `ICSPS26-FROZEN-v3`，冻结新的统一预算，并把所有受影响方法从头重跑；不能运行中途对不同方法采用不同预算。

远程正式环境按 RTX 4090 24 GiB ×2、16-vCPU quota、120 GB RAM 设计。两卡只承载两个相互独立的 batch-1 run/evaluation queue；每个进程通过 `CUDA_VISIBLE_DEVICES` 只看到一张物理卡，不使用 DDP/DataParallel，也不把 24+24 GiB 视为单进程 48 GiB。软件记录分别保存驱动版本与 `nvidia-smi` 的最高 CUDA capability、PyTorch 2.5.1 的 `torch.version.cuda==12.4` build，二者不得混写。

### 8.2 随机种子层级

- 第一优先：所有核心方法与核心消融 seed 42；
- 第二优先：Full 和 TransSARV2 补 seeds 43、44；
- 第三优先：Log-only 与 Full w/o all FDR 补 seeds 43、44；
- 单 seed 时不得写 training-run `mean±std`。

### 8.3 AMS 配置

| 项目 | 值 |
|---|---|
| Epochs | 8 |
| Batch/crop | 1 / 256 |
| LR / weight decay | $10^{-6}/10^{-5}$ |
| Mask ratio | 0.2 |
| Loss | masked L1 + $\lambda_{\rm AMS}$ mean-TV |
| TV 权重选择 | 在 real-SAR validation 比较 $\{0,10^{-4},10^{-3},10^{-2},10^{-1}\}$；按 M-index/EPI 联合排序选择 $10^{-3}$，ENL 仅用于诊断平滑程度，并固定用于全部 AMS 测试 |
| Frozen | TransSAR-derived encoder (`log_branch.Tenc`) |
| Trainable | bottleneck、decoder、guidance、head、compensator |
| Selection | 固定 real-validation masks 上 masked loss 最低；不读取合成 validation/test 结果 |
| Inference | 完整未遮挡输入，不做 mask |

正式 runner 已为每个 real-validation sample 固定 mask seed，并把 seed 与 mask SHA-256 写入 `ams_validation_masks.csv`。所有 epoch 复用同一组 masks；若 resume 时清单缺失或不一致，必须停止运行。

---

## 9. 真实 SAR 实验协议

### 9.1 数据版本与术语

锁定 Mendeley Data：[`fs455tz88y.1`](https://data.mendeley.com/datasets/fs455tz88y/1)，DOI `10.17632/fs455tz88y.1`。该版本说明包含 `Noisy`、`GT1` 和 `GT16`，但本论文只传输和使用冻结 split 对应的 `Noisy` patches。

不要与 Data in Brief 对应的另一数据版本 `2xf5v5pwkr.1` 混写；后者论文描述同一区域多时相 Sentinel-1 VV 数据经配准与融合，并把值 rescale 到 0–255，但其 DOI 和目录组织不同。[Data in Brief article](https://doi.org/10.1016/j.dib.2024.110065)

在任何 real test 输出产生前，已冻结 **no-GT16** 分支。GT16 不进入数据传输、QC 有效集、指标、表格或图片；所有准参考指标均不报告，也不留下可供事后选择的空结果槽。

### 9.2 Parent-disjoint 划分

沿用已审计 manifest：

| Split | 512 parent groups | 256 child patches |
|---|---:|---:|
| Train | 1,175 | 4,700 |
| Validation | 146 | 584 |
| Test assigned | 148 | 592 |
| Total | 1,469 | 5,876 |

- `group_seed=42`；
- 同一 512 parent 的四个 child patches 不得跨 split；
- 三组 parent 交集必须为 0；
- 这是**同一区域内 parent-disjoint**，不能宣称 cross-scene、cross-region 或 cross-sensor generalization。

### 9.3 先建立 QC manifest

历史 `RealSARDataset` 会静默跳过 constant/invalid 文件，容易造成 assigned 与 evaluated 计数不一致。正式流程已不再依赖该隐式行为，而是先生成唯一 QC manifest：

```text
sample_id, parent_id, split, noisy_path, gt16_path(empty),
shape, dtype, min, max, is_finite, is_constant,
pair_exists, alignment_score, validity, exclusion_reason,
noisy_sha256, gt16_sha256(empty)
```

所有方法只评价 `valid_no_reference=true` 的相同样本集合。当前本机原始 Noisy 副本的实测 QC 为 592/592 test patches 通过 no-reference 和固定 ROI 检查，对应 148 parents；远端正式 QC 必须复现该数量，否则停止并调查传输或数据版本。no-GT16 分支要求 `gt16_root=null` 且 `valid_quasi_reference=0`。论文中 assigned/valid 数必须从最终远端 QC 原始记录填写，不得预填假定排除数。

### 9.4 固定数值映射

当前逐图 1%/99% percentile normalization 会改变图间尺度。正式方案改为：

- uint8：固定除以 255；
- 已在 ([0,1]) 的 float：保持不变；
- 其他范围：立即 fail，并由数据审计决定全数据集统一映射；
- 所有 Noisy patches 使用完全相同、预先声明的 deterministic mapping；
- 显示 stretch 只用于论文可视化，绝不进入正式指标。

如果无法证明原始图的 radiometric calibration，论文只称 normalized-intensity restoration，不声称 absolute radiometric fidelity。

### 9.5 no-GT16 强制项

- 服务器环境固定 `REAL_USE_GT16=0` 并 `unset REAL_GT16_ROOT`；
- 数据准备显式传入 `--disable-gt16`，即使目录中意外出现 `GT16/` 也不会自动使用；
- 论文及机器可读模板不包含任何准参考指标行；
- 若未来决定加入 GT16，必须升协议版本并重跑全部受影响的真实域 QC、评测、汇总和图片，不能追加到当前结果。

### 9.6 AMS 泄漏规则

- AMS train 只读取 train/Noisy；
- AMS validation 只读取 validation/Noisy 和固定 masks；
- GT1/GT16 不读取，也不进入 loss、checkpoint selection、early stopping 或超参选择；
- real test Noisy 在所有设计决定冻结后一次性解锁；
- base 与 AMS 必须在同一 valid test IDs 上评估。

---

## 10. 指标定义

### 10.1 合成主指标

1. **PSNR↑**：逐图在 linear normalized intensity 上计算，`data_range=1`；
2. **SSIM↑**：统一 11×11 Gaussian window 实现。

不得在模型输出上先做可视化 stretch 再算指标。

### 10.2 真实主指标

| 指标 | 方向 | 说明 |
|---|:---:|---|
| ENL median [IQR] | 诊断 | 只在固定 homogeneous ROIs；不标“越大越好” |
| $\mathcal M$-index | ↓ | 按 ratio-image 一阶统计残差与 Haralick homogeneity 残差的原论文定义；同时报告 valid/total |
| EPI | ↑ | 按预先固定邻接方向的文献公式计算原始含斑图与去斑图的边缘差分保持比例 |

M-index 采用 Gómez、Ospina 与 Frery 的无参考 $\mathcal M$ 指标，不得使用历史 fallback 或自行改变缩放方式。[原论文](https://doi.org/10.3390/rs9040389)；[作者后续统计代码](https://github.com/Raydonal/Statistical-Behaviour-M-Index) EPI 使用文献给出的相邻像素绝对差分比，不把 Sobel 梯度相关性或 RGPI 改名为 EPI。[EPI 公式出处，Eq. (16)](https://doi.org/10.3390/rs16111992)

Ratio map 只作为预注册的定性图，不再从中另造论文评价指标。2025 年的 ratio-image quality assessment 说明，单纯方差抑制与 ratio image 恢复到无结构随机 speckle 并不是同一件事，因此 ENL 必须与 M、EPI 和固定样例共同解释。[Vásquez-Salazar et al., 2025](https://doi.org/10.3390/rs17244048)

### 10.3 指标实现的来源校验规则

正式表只能填写按上述论文公式计算且通过手算/参考输出回归的 ENL、M 与 EPI。`sar_metrics.py` 中历史 Sobel 相关性和无有效窗口时强行选取候选的 M fallback 均不得用于论文；也不得把任何内部 ratio/ACF/Sobel 诊断改名后写入主表。M 无有效窗口时记录 `N/A` 与 valid/total，不补造数值。

### 10.4 ENL ROI 规则

- 只根据 noisy input 和预注册规则选 ROI 一次；
- 固定尺寸、数量、坐标与最小均值阈值；
- 所有方法复用同一坐标；
- 不根据某个输出重新选择更平滑的区域；
- 报告 ROI-level median [IQR]，并与 ratio/edge 指标一起解释；高 ENL 可能来自过度平滑。

---

## 11. 统计分析

### 11.1 UCM-21 paired cluster bootstrap

同一 clean source 的四个 $L$ 不是独立样本。使用 10,000 次 stratified paired cluster bootstrap：

1. 在每个 UCM 类别内有放回采样 source IDs；
2. 每个采样 source 保留其四个 $L$ 与全部方法的 paired results；
3. 先计算 class×$L$ 均值；
4. 再对 21 类与四个 $L$ 做等权 macro；
5. 对 `Ours − strongest baseline` 报告 2.5%/97.5% percentile CI。

PSNR 的差值以 dB 直接配对；SSIM 直接配对；error 指标使用 `baseline − ours` 或清楚标注方向，避免符号混乱。

### 11.2 真实 SAR parent-cluster bootstrap

对真实主指标，以 148 个 test parent groups 为 cluster，保留每个 parent 的全部有效 child patches，做 10,000 次 paired bootstrap。不能把 592 个 child patches 当作相互独立样本。

### 11.3 多 seed 报告

- 三 seed 完成时：表中报告 seed-level mean ± sample SD；
- paired test 时，先对每个 source 的三 seed metric 求平均，再做 source-cluster bootstrap；
- 只有 seed 42 时：明确写 *single training run*，只报告测试源 bootstrap CI；该 CI 不代表初始化不确定性；
- 禁止把逐图标准差伪装成训练 seed 标准差。

### 11.4 预注册 primary external comparator

在看 Ours 的 test 差值前，预定义 primary external comparator：默认为同协议重训的 TransSARV2。若要用“development validation 上 strongest non-ours”的表述，SAR-CAM 等所有候选方法必须在 UCM 解锁前完成训练和 validation，再冻结对手。真实域 ENL/M/EPI 使用同一预声明比较对象。不能在 UCM/real test 上看完结果后再把 `max macro PSNR` 的方法宣布为预注册 baseline，也不要事后只选择最容易显著的对手。

正式实现使用一次性 `pretest_seal.json`：在尚无任何 test-derived 输出时，语义核验并绑定实际 best checkpoint/completion/validation 行、UCM 8,400-pair manifest、RealSAR 592-patch/148-parent QC、预注册的 UCM $L=1/L=4$ crop 与 RealSAR homogeneous/structured crop。所有 test evaluator、汇总器和论文图 exporter 必须携带并交叉核对同一 seal、manifest 与选择表 SHA-256；不能只用一个格式正确的时间戳或伪造的 64 位字符串解锁。

---

## 12. 不纳入本文的测量

本次六页会议论文不测量、不报告 Params、MACs、latency 或 peak memory，也不为它们保留正式运行或记录步骤。

---

## 13. 结果表结构

### 表 1：UCM-21 外部 Gamma-only 主比较

| Method | Training / Look mode | $L=1$ PSNR / SSIM | $L=2$ | $L=4$ | $L=8$ | Macro PSNR / SSIM |
|---|---|---:|---:|---:|---:|---:|
| Noisy | – |  |  |  |  |  |
| Lee-MMSE | oracle-$L$ |  |  |  |  |  |
| SAR-BM3D | oracle-$L$ |  |  |  |  |  |
| SAR-CAM | blind, retrained |  |  |  |  |  |
| TransSARV2 | blind, retrained |  |  |  |  |  |
| Ours | blind, one model |  |  |  |  |  |

表注：给出 Ours 相对 strongest baseline 的 source-cluster bootstrap 95% CI；`oracle-L` 不是 blind method。

### 表 2：表示链与频域模块核心消融

| Variant | Log | Inverse | Intensity Comp. | Bottleneck FDR | Decoder FDR | Macro PSNR / SSIM |
|---|:---:|:---:|:---:|:---:|:---:|---:|
| Intensity-only | × | – | × | ✓ | ✓ |  |
| Log-only | ✓ | ✓ | × | ✓ | ✓ |  |
| Full w/o all FDR | ✓ | ✓ | ✓ | × | × |  |
| Full | ✓ | ✓ | ✓ | ✓ | ✓ |  |

### 表 3：真实 SAR 与 AMS

| Method | External training / Adaptation | Test patches / parents | ENL median [IQR] | $\mathcal M$↓ | M valid/total | EPI↑ |
|---|---|---:|---:|---:|---:|---:|
| Noisy | – |  |  | N/A$^*$ | 0 /  | 1.0000$^*$ |
| TransSARV2 | synthetic only |  |  |  |  |  |
| Ours-base | synthetic only |  |  |  |  |  |
| Ours+AMS | encoder frozen |  |  |  |  |  |

表注必须说明论文采用 no-GT16 的 no-reference 真实域协议、real split 是 within-scene parent-disjoint，并报告 valid/excluded 样本数。$^*$ Noisy pass-through 的 M 无有效 ratio window，EPI=1 是恒等检查，均不得当成可排序性能。

### 表 4：协议与可复现性审计（建议放补充材料或代码仓库）

| Check | NWPU | UCM | Real SAR | Required result |
|---|---:|---:|---:|---|
| Source path overlap | train/validation | – | parent overlap | 0 |
| Content SHA overlap | train/validation；UCM cross-check | NWPU cross-check | exact duplicate overlap | 0 |
| Missing files |  |  |  | 0 |
| Duplicate IDs |  |  |  | 0 |
| Source counts | 900 train / 225 validation | 2,100 | 按 parent manifest | 与协议一致 |
| Per-$L$ counts | train schedule 25,000/$L$；validation 225/$L$ | 2,100/$L$ | – | 与协议一致 |
| Schedule replay | 100k rows + SHA-256 | – | – | bitwise same pair stream |
| Saturation rate |  |  | – | 每个 $L$ 报告 |
| Invalid/constant | – | – |  | 固定排除清单 |
| Checkpoint strict load |  |  |  | 通过 |

---

## 14. 论文图片建议

### 图 1：总体网络与 TransSAR 归属

**图 1：TransSARV2-derived SAR despeckling architecture with multi-scale full-spectrum Fourier refinement and log-to-intensity residual compensation.**

绘制要求：

- 左端显示 $y_{01}\rightarrow T_\alpha(y)$；
- 上支路为五阶段 TransSAR encoder，标注 `32@128², 64@64², 128@32², 320@16², 512@8²`；
- 同一 log input 分出 guidance-map generator；
- decoder 每级显示 `Up → Concat/Fusion → FDR → RB → Gate`；
- head 后显示 `Sigmoid → exact inverse-log`；
- 最后显示 `[original intensity, restored intensity] → residual compensator → clamp`；
- 灰色表示继承自 TransSAR，彩色表示本文扩展；
- 不出现最终 Tanh，不画成两条对等 restoration backbones。

### 图 2：FDR 与正确数值域链

**图 2：Full-spectrum rFFT residual refinement and representation-consistent log-to-intensity reconstruction.**

建议做成 (a)(b)：

- (a) 空间 DWConv 分支；rFFT2；real/imag concat；共享 1×1 spectral channel mixing；irFFT2；空间-频域 concat；residual；
- (b) $y_{01}\rightarrow z\rightarrow\widehat z\rightarrow T^{-1}(\widehat z)\rightarrow[ y_{01},\widetilde x ]\rightarrow\widehat x$；
- 图注注明没有显式 high/low band split，学习卷积为 real-valued。

若版面紧张，将图 2 作为图 1 的右下 inset。

### 图 3：跨视数定量曲线

**图 3：Controlled-look generalization on UCM-21 across $L=1,2,4,8$.**

双联图：

- (a) PSNR vs. $L$；
- (b) SSIM vs. $L$；
- 方法颜色全篇固定；error bar 不是逐图标准差。

### 图 4：外部合成定性比较

**图 4：External UCM-21 despeckling examples under severe and moderate speckle, with matched crops and absolute-error maps.**

布局：Clean / Noisy / SAR-BM3D / SAR-CAM / TransSARV2 / Ours / Error。至少选 $L=1$ 与 $L=4$；样例 ID 在跑 test 前预注册。所有图使用同一 intensity/display mapping、同一 crop 和误差色条；局部框展示细线、规则纹理与亮散射邻域，不做舰船检测框。

### 图 5：真实 SAR、ratio image 与 AMS

**图 5：Parent-disjoint real-SAR results before and after masked adaptation, together with ratio images.**

布局：Noisy / TransSARV2 / Ours Log-only / Ours-base / Ours+AMS，以及各自 $Y/\widehat X$ ratio map。固定两类区域：homogeneous + structured urban/linear feature。Ratio map 使用以 1 为中心的相同色阶；caption 明确真实域没有 clean/quasi-reference，所有结论来自固定 no-reference 诊断与全测试集统计。

6 页版面优先级：图 1（必须）> 图 4/5 合并定性图（必须）> 图 3（强烈建议）> 独立图 2（可并入图 1）。

---

## 15. 六页英文论文结构

| 部分 | 建议篇幅 | 核心内容 |
|---|---:|---|
| Abstract + Index Terms | 0.25 页 | 问题、TransSAR 归属、两项核心设计、synthetic/real 证据；不堆模块名 |
| I. Introduction | 0.55 页 | domain gap、频谱/数值域问题、三项贡献 |
| II. Related Work | 0.45 页 | supervised/Transformer；frequency-aware；real/self-supervised 三段 |
| III. Method | 1.45–1.65 页 | 问题模型、总体链、FDR、intensity compensation、AMS；图 1/2 |
| IV. Experiments | 2.4–2.6 页 | 数据/协议、基线、实现、主表、消融、真实评价、定性图 |
| V. Conclusion | 0.15–0.20 页 | 有限结论与 within-scene 限制 |
| References | 余量 | 约 20–25 篇，优先主流与直接近邻 |

Related Work 建议仅三段：

1. SAR-BM3D、SAR-CAM 到 TransSAR/DAT-Net/Noise-Guided Transformer；
2. SAR-FDD、DAT-Net、GLCNet 与其他 spatial-frequency work；
3. SAR2SAR/MERLIN、CL-SAR/DiffusionSAR、Speckle2Self/SDS-SAR。

方法章节不要逐层复述 TransSAR encoder；用一个表或一句话给出配置，把版面留给 FDR、数值域链和消融可检验性。

---

## 16. 投稿前执行顺序

### 当前工作区实况

截至 2026-09-05，本机 checkout 已有 31,500 张原始 NWPU 图像、5,876 个 real-Noisy patches、canonical NWPU master manifest 和 real grouped split；用户已在任何 test 输出产生前决定本论文不补充 GT16，因此不计算或报告任何 GT16-based quasi-reference metric。用户随后决定只采用公认评价指标并取消全部复杂度测量，合成域只报告 PSNR/SSIM，AMS 前后只比较真实 SAR。当前仍没有新协议下可写入论文的正式训练/评测输出，因此本次属于正式运行与预测试封存前的协议修订；远端不得沿用旧 schema 的结果目录。数据协议、统一 trainer/evaluator、AMS、传统基线 wrapper、服务器脚本与记录模板已经实现。仍须在远端先完成 GPU/CUDA、实际数据盘、迁移 hash、formal data generation/verification 和端到端 smoke，再启动训练矩阵。中期报告与旧 compound 数据数字不能自动升级为正式结果。

协议冻结后如必须改动数据、指标、预算或 split，应把编号提升为 `ICSPS26-FROZEN-v3`，写明变更原因，并从受影响实验的第一个方法开始统一重跑。

### P0：任何正式训练前必须完成

- [x] 已实现 Gamma-only NWPU/UCM generator 与 manifests；所有额外退化为 0；远端仍需正式生成/复核；
- [x] 已锁定 `NWPU-Small-1125-v1`：每类严格 20 train + 5 validation，未选样本不能被 loader 读取；
- [x] 已实现并测试 100k-row canonical schedule：每个 $L$ 25k，source/class 呈现差至多 1；
- [x] 已实现 NWPU split path/content audit、UCM–NWPU exact/pHash audit；
- [x] 已实现 manifests、SHA-256、可重放在线 Gamma、固定 validation/test MAT，以及每个正式 run 全 100k updates 的分 $L$ clipping/saturation 累计；
- [x] runner 已支持真正 `TransSARV2`、模型身份元数据和 strict checkpoint/resume；
- [x] 正式训练 augmentation 已锁定为无插值 paired D4；
- [x] 已实现 intensity-only 与 true `w/o all FDR`；
- [x] 已实现 real QC、固定 dataset-level mapping 与固定 ROI；正式分支显式禁用 GT16，远端需复核 `gt16_root=null` 和 q-valid count=0；
- [ ] 正式 ENL/M/EPI 实现已逐项与所引论文公式核对并通过回归；内部诊断量不进入论文；
- [x] 已固定并归档 AMS validation mask seed/hash；
- [x] 已实现 UCM source / real parent cluster bootstrap；
- [ ] 远端保存实际源码 bundle hash、环境 inventory、数据 hashes 与正式 checkpoint hashes。

### P1：不可删除的核心实验

1. Full seed 42；
2. 真正 TransSARV2 seed 42；
3. Log-only seed 42；
4. Intensity-only seed 42；
5. Full w/o all FDR seed 42；
6. UCM-21 全量四个 $L$ 外测；
7. Lee-MMSE 与 SAR-BM3D；
8. 同一 real test 上 TransSARV2、Log-only、Ours-base 与 Ours+AMS；
9. ratio maps（仅定性）与同一真实 test 上的 fixed-ROI ENL、M、EPI。

### P2：强加分项

1. Full 与 TransSARV2 seeds 43/44；
2. SAR-CAM 同协议重训；
3. SDS-SAR 或 CL-SAR 真实域比较；
4. 核心消融 seeds 43/44。

### P3：只在有余力时

- compound stress macro；
- `w/o gate`、`w/o bottleneck` 单 seed；
- 更丰富的频谱/ratio 可视化。

### 立即删除

- 舰船检测与任何下游 detector；
- 旧 mixed-L 表；
- 当前混杂 `wout_msf`；
- mask-ratio sweep；
- 未见 $L=3,6$；
- 输入条件不一致的 SLC/sublook 方法数值表；
- 中期报告旧 compound 数据上的数字。

---

## 17. 可复现输出结构

每次 run 使用独立、不可覆盖目录：

```text
experiments_icsps26/
  logs/
  records/
    *.csv                       # 11 张必填模板，逐表见远程手册§9
    ICSPS2026_实验记录表.md
    pretest_registration.sha256
    test_unlock_utc.txt
    pretest_seal.json
  train/<method>/seed<seed>/
    run_config.json
    step_metrics.csv
    train_progress.jsonl
    checkpoint_best.pth
    checkpoint_last.pth
    completion.json
    tensorboard/
  ucm/noisy/
    run_config.json
    per_image.csv
    aggregate.json
  ucm/<learned-method>/seed<seed>/
    run_config.json
    per_image.csv
    aggregate.json
  ucm/lee_mmse/
    run_config.json
    nwpu_window_tuning.csv
    per_image.csv
    aggregate.json
  ucm/sar_bm3d_v1/
    run_config.json
    per_image.csv
    aggregate.json
  sarbm3d_raw_ucm/
    prediction_manifest.csv
    completion.json
  ams/ours_full/seed<seed>/
    run_config.json
    ams_validation_masks.csv
    epoch_metrics.csv
    checkpoint_best.pth
    checkpoint_last.pth
    completion.json
  real/<method>_<adaptation>/seed<seed>/
    run_config.json
    real_per_patch.csv
    real_by_parent.csv
    real_enl_per_roi.csv
    aggregate.json
  summary/ucm_<summary-id>/
    ucm_main_summary.csv
    ucm_paired_cluster_bootstrap.csv
    summary_provenance.json
  summary/real_<summary-id>/
    real_paired_parent_bootstrap.csv
    summary_provenance.json
  paper_figures/ucm_pre_registered/
    figure_manifest.csv
    provenance.json
    completion.json
  paper_figures/real_pre_registered/
    figure_manifest.csv
    provenance.json
    completion.json
```

UCM `per_image.csv` 至少含：

```text
method, seed, pair_id, source_id, class_name, L,
psnr, ssim
```

`real_per_patch.csv` 至少含：

```text
method, adaptation, seed, sample_id, parent_id,
output_enl_roi_median, m_index, m_valid, epi
```

所有 aggregate 表必须能从 per-image/per-patch CSV 完全重建。

---

## 18. Claim 黑名单与推荐替换

| 不得使用 | 推荐替换 |
|---|---|
| the first Transformer-based SAR despeckling network | a TransSARV2-derived despeckling model |
| a novel Transformer backbone | an extension of the public TransSARV2 backbone |
| the first frequency-domain / dual-domain method | multi-scale full-spectrum Fourier refinement in a TransSAR-derived decoder |
| high-frequency enhancement | complex-spectrum residual refinement |
| frequency-selective filtering | shared spectral channel mixing |
| complex-valued convolution | real-valued mixing of concatenated real/imaginary spectra |
| magnitude–phase refinement | real–imaginary spectrum refinement |
| log Gaussianizes speckle | log1p stabilizes dynamic range and partially reduces signal dependence |
| noise estimator / accurate noise prior | learned guidance map / latent modulation map |
| decoder-only AMS | encoder-frozen adaptation of reconstruction modules |
| unbiased self-supervision | practical masked post-adaptation |
| GT16 / clean ground truth | 本论文不使用；不得在结果表中暗示存在真实 clean reference |
| cross-scene/cross-sensor generalization | within-scene parent-disjoint evaluation |
| radiometrically faithful | normalized-intensity consistent |
| state of the art | statistically better under the shared protocol；只有结果支持时使用 |

---

## 19. 主要参考文献与开源资源

### 19.1 会议、基础综述与主干

1. [ICSPS 2026 Special Session 5](https://www.icsps.org/special5.html).
2. [ICSPS 2026 Submission Instruction](https://www.icsps.org/sub.html).
3. [ICSPS Peer Review Process](https://www.icsps.org/peer_review.html).
4. G. Fracastoro *et al.*, “Deep Learning Methods for Synthetic Aperture Radar Image Despeckling: An Overview of Trends and Perspectives,” *IEEE Geoscience and Remote Sensing Magazine*, 2021. [DOI](https://doi.org/10.1109/MGRS.2021.3070956).
5. M. V. Perera *et al.*, “Transformer-based SAR Image Despeckling,” IGARSS 2022. [DOI](https://doi.org/10.1109/IGARSS46834.2022.9884596); [arXiv](https://arxiv.org/abs/2201.09355); [official code](https://github.com/malshaV/sar_transformer).

### 19.2 对比与近邻方法

6. S. Parrilli *et al.*, “A Nonlocal SAR Image Denoising Algorithm Based on LLMMSE Wavelet Shrinkage,” *IEEE TGRS*, 2012（SAR-BM3D）. [DOI](https://doi.org/10.1109/TGRS.2011.2161586); [author code v1.0](https://www.grip.unina.it/download/prog/SAR-BM3D/version_1.0/).
7. J. Ko and S. Lee, “SAR Image Despeckling Using Continuous Attention Module,” *IEEE JSTARS*, 2022. [DOI](https://doi.org/10.1109/JSTARS.2021.3132027); [official code](https://github.com/JK-the-Ko/SAR-CAM).
8. X. Zhao *et al.*, “Synthetic Aperture Radar Image Despeckling Based on a Deep Learning Network Employing Frequency Domain Decomposition,” *Electronics*, 2024（SAR-FDD）. [Article/DOI](https://doi.org/10.3390/electronics13030490).
9. “Despeckling SAR Images with Log-Yeo–Johnson Transformation and Conditional Diffusion Models,” *IEEE TGRS*, 2024. [DOI](https://doi.org/10.1109/TGRS.2024.3419083).
10. Y. Shen *et al.*, “DATNet: Dynamic Adaptive Transformer Network for SAR Image Denoising,” *Remote Sensing*, 2025. [Article/DOI](https://doi.org/10.3390/rs17173031).
11. L. Zhang *et al.*, “Effective SAR Image Despeckling Using Noise-Guided Transformer and Multi-Scale Feature Fusion,” *Remote Sensing*, 2025. [Article/DOI](https://doi.org/10.3390/rs17233863).
12. H. Lin *et al.*, “Speckle2Self: Learning Self-Supervised Despeckling with Attention Mechanism for SAR Images,” *Remote Sensing*, 2025. [Article/DOI](https://doi.org/10.3390/rs17233840).
13. Y. Fang *et al.*, “Contrastive Learning for Real SAR Image Despeckling,” *ISPRS JPRS*, 2024. [DOI](https://doi.org/10.1016/j.isprsjprs.2024.11.003); [official code](https://github.com/YangtianFang2002/CL-SAR-Despeckling).
14. L. Chen *et al.*, “Self-supervised Despeckling Based Solely on SAR Intensity Images: A General Strategy,” *ISPRS JPRS*, 2026. [DOI](https://doi.org/10.1016/j.isprsjprs.2025.11.025); [official code/data/weights](https://github.com/YYF121/SDS-SAR).
15. Y. Yang *et al.*, “Self-supervised Global–Local Collaborative Network for Real SAR Despeckling,” *International Journal of Applied Earth Observation and Geoinformation*, 2026（GLCNet）. [DOI](https://doi.org/10.1016/j.jag.2026.105135); [project repository](https://github.com/yangyang12318/LGCN).
16. A. Saha *et al.*, “Integrated Multi-channel Approach for Speckle Noise Reduction in SAR Imagery Using Gradient, Spatial, and Frequency Analysis,” *Signal Processing: Image Communication*, 2026. [DOI](https://doi.org/10.1016/j.image.2025.117406).
17. [Practical SAR Despeckling Codes](https://github.com/YangtianFang2002/Practical-SAR-Despeckling-Codes).

### 19.3 指标与数据

18. L. Gomez, R. Ospina, and A. C. Frery, “Unassisted Quantitative Evaluation of Despeckling Filters,” *Remote Sensing*, 2017. [DOI](https://doi.org/10.3390/rs9040389); [related author code](https://github.com/Raydonal/Statistical-Behaviour-M-Index).
19. “A No-Reference Edge-Preservation Assessment Index for SAR Image Filters under a Bayesian Framework Based on the Ratio Gradient,” *Remote Sensing*, 2022. [DOI](https://doi.org/10.3390/rs14040856).
20. G. Cheng, J. Han, and X. Lu, “Remote Sensing Image Scene Classification: Benchmark and State of the Art,” *Proceedings of the IEEE*, 2017（NWPU-RESISC45）. [DOI](https://doi.org/10.1109/JPROC.2017.2675998); [arXiv](https://arxiv.org/abs/1703.00121).
21. [UC Merced Land Use Dataset — official laboratory page](https://vision.ucmerced.edu/datasets/).
22. R. Vasquez *et al.*, “Labeled dataset for despeckling SAR imagery,” Mendeley Data, V1, 2024. [DOI/data](https://doi.org/10.17632/fs455tz88y.1).
23. R. D. Vásquez-Salazar *et al.*, “Labeled dataset for training despeckling filters for SAR imagery,” *Data in Brief*, 2024. [DOI](https://doi.org/10.1016/j.dib.2024.110065)；注意其数据 DOI 为 `2xf5v5pwkr.1`，不能与第 22 项混同。
24. R. D. Vásquez-Salazar *et al.*, “Quality Assessment of Despeckling Filters Based on the Analysis of Ratio Images,” *Remote Sensing*, 2025. [DOI](https://doi.org/10.3390/rs17244048).

---

## 20. 最终 Definition of Done

只有同时满足下列条件，论文中的“正式结果”才成立：

- [ ] 标题、摘要、方法图和正文均明确 TransSARV2 归属；
- [ ] NWPU source manifest 精确为 900 train / 225 validation，且训练 schedule 精确为 100k rows；
- [ ] 所有可学习基线和消融共享相同 100k update 上限、同 run-seed 数据流、5k 验证网格和选模规则；
- [ ] 主合成数据只有 Gamma speckle，且报告 clipping/saturation；
- [ ] UCM 使用全部 21 类和全部四个 $L$；
- [ ] 真正 TransSARV2 在同协议、同预算下重训；
- [ ] 核心消融为单因素且 `w/o all FDR` 确实关闭全部六个 FDR；
- [ ] 模型链是 `Sigmoid → inverse-log → intensity compensation → clamp`；
- [ ] 不把 guidance map 称为受监督 noise estimate；
- [ ] real split parent-disjoint，所有方法使用同一 valid IDs；
- [ ] `REAL_USE_GT16=0`，QC 记录 `gt16_root=null`；论文、主表与图中均无 GT16-based quasi-reference metrics；
- [ ] ENL/M/EPI 均与所引原论文公式一致；没有把内部诊断量改名写入正式表；
- [ ] 聚类 bootstrap 单位为 UCM source / real parent；
- [ ] 单 seed 不报告伪造的 run-level mean±std；
- [ ] 主表数字均来自 per-image CSV，可由脚本重建；
- [ ] 中期报告历史结果不混入正式表；
- [ ] 全文没有舰船检测内容；
- [ ] 双盲版本清除姓名、单位、致谢、仓库用户名及 PDF metadata 中的作者身份线索；
- [ ] 论文压到 6 页且所有图中文字在双栏打印尺寸下可读。

做到这些后，论文的评审叙事将非常清楚：**不是重新命名 TransSAR，也不是笼统堆叠 FFT、门控和多尺度模块，而是在可追溯的 TransSARV2 主干上验证一个表示一致的频谱细化与真实域校准方案。**
