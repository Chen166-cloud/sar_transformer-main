# 真实 SAR 图像去斑：强方法与可复现性调研

检索与仓库核查日期：2026-09-18。

## 1. 结论

针对本项目现有的单通道真实 SAR 图块，优先新增 **CL-SAR、MuLoG-DRUNet**；保留 **SDUDNet、SAR2SAR、SAR-BM3D**，补充 **NL-SAR** 作为传统强参照。若希望覆盖单图自监督路线，再加入 **S3DIP**。如果取得保留实部、虚部的复数 SLC，则应优先加入 **MERLIN**。

这是综合真实数据证据、公开实现、权重和输入条件后的复现优先级，不是同一测试集上的性能排名。本次没有运行新模型，也没有得到本项目上的新实验指标。当前证据不足以指定一个跨传感器、成像模式和场景都最好的方法。

2026 年值得关注的 SDS-SAR、GLCNet、SemDNet、MambaDIP，并不都已具备完整复现条件。其中 SDS-SAR 的公开测试入口仍是中值滤波占位实现，不能以仓库存在为由把它算作已复现的论文方法。较新的 Closed-Form Nonlocal Shrinkage 已有完整算法代码，但截至检索日仍属预印本，适合探索性对比。

## 2. 检索范围与判断依据

检索覆盖 IEEE TGRS/JSTARS/GRSL、ISPRS Journal of Photogrammetry and Remote Sensing、International Journal of Applied Earth Observation and Geoinformation、Pattern Recognition Letters、Expert Systems with Applications、Remote Sensing、Sensors，以及作者机构、作者 GitHub/GitLab 和 arXiv。较早的成熟方法用于对照，重点补查 2024—2026 年成果。

核验区分三件事：

- **效果证据**：真实传感器图像、near-real 伪参考数据、光学图加模拟斑点，不能混成同一类实验。
- **复现材料**：核查实际模型、训练/推理入口、权重文件与预处理；README 中“将发布”或网盘链接不等于完整实现已经可用。
- **数据适配**：区分强度、幅度、复数 SLC、极化协方差，以及训练与推理各自的数据要求。

部分出版社全文访问受限，结论仅采用能核查的论文正文、官方摘要或作者公开材料；未核实到代码不等于证明作者从未公开。仓库判断为静态审查，不能替代端到端复现。网盘数据未逐一下载验证。

## 3. 优先复现的方法

| 方法 | 正式论文 | 真实 SAR 与复现依据 | 对本项目的判断 |
| --- | --- | --- | --- |
| **CL-SAR** | ISPRS JPRS，2024，[论文](https://doi.org/10.1016/j.isprsjprs.2024.11.003) | Sentinel-1 near-real 及 Capella-X、TerraSAR-X 实验；[作者代码](https://github.com/YangtianFang2002/CL-SAR-Despeckling)包含训练、推理、配置及实际模型文件 | **优先新增**，补足面向真实相关斑点、跨传感器泛化的深度方法 |
| **MuLoG-DRUNet** | TGRS，2024，[论文](https://doi.org/10.1109/TGRS.2024.3432180) | 空间相关斑点下的统计模型与即插即用去噪；[官方发布](https://gitlab.telecom-paris.fr/ring/mulog-drunet)有模型、源码压缩包及示例 | **优先新增**；源码支持二维单强度，并非只能处理 PolSAR；需要合理的 looks 参数 |
| **SDUDNet** | JSTARS，2025，[论文](https://doi.org/10.1109/JSTARS.2025.3568854) | [作者仓库](https://github.com/BFY-official/SDUDNet)有网络和实际 `real.pth`、`synthetic.pth`；论文包含真实图无参考评价 | **保留**；真实权重公开，但不据少量真实图和自然图像质量指标断言全局最佳 |
| **SAR2SAR** | JSTARS，2021，[论文](https://doi.org/10.1109/JSTARS.2021.3071864) | 真实多时相自监督训练；[官方代码](https://gitlab.telecom-paris.fr/RING/SAR2SAR)，另有 [PyTorch 推理与权重](https://github.com/hi-paris/deepdespeckling) | **保留**；训练使用多时相，推理只需单幅幅度图 |
| **MERLIN** | TGRS，2022，[论文](https://doi.org/10.1109/TGRS.2021.3128621) | 复数 SAR 自监督；[作者参与维护的软件包](https://github.com/hi-paris/deepdespeckling)提供不同成像模式的模型权重 | **有 SLC 时优先**；现有单通道 MAT 不满足官方输入要求 |
| **S3DIP** | Pattern Recognition Letters，2025，[论文](https://doi.org/10.1016/j.patrec.2025.02.021) | [作者代码](https://github.com/IAPP-Group/S3DIP)有逐图优化、停止准则和 COSMO-SkyMed 真实样例；不需要预训练权重 | **第二批新增**；代表单图自监督，但计算和预处理成本较高 |
| **NL-SAR** | TGRS，2015 | [作者官方项目页](https://www.charles-deledalle.fr/pages/nlsar.php)提供源码、二进制和文档，支持单强度及多通道 SAR | **值得补充的传统强基线**；需核对 looks、相关斑和接口 |
| **SAR-BM3D** | TGRS，2012 | [官方研究软件](https://www.grip.unina.it/download/prog/SAR-BM3D/version_1.0/)仍提供 Windows/Linux 包 | **保留传统参照**；需报告 looks 及处理域，不能只以视觉平滑程度评判 |

### 3.1 CL-SAR：目前最值得优先补入的深度学习候选

论文 Table 2 的 near-real 平均结果如下。这是同一篇论文的报告值，不是本项目复测值；训练监督和数据使用方式也并非完全相同。

| 方法 | PSNR / dB | SSIM |
| --- | ---: | ---: |
| MRDDANet | 25.1188 | 0.8268 |
| SAR-CAM | 25.1649 | 0.8168 |
| CL-SAR | 25.3411 | 0.8305 |

CL-SAR 相对 SAR-CAM 的增量是约 0.176 dB 和 0.0137 SSIM。它支持“值得加入强对照”，而非“远超所有方法”。near-real 的参考图来自真实观测的构建流程，并非无噪声物理真值；这组数值不能与其他论文的真实图无参考指标直接排序。[论文及 Table 2](https://www.sciencedirect.com/science/article/abs/pii/S0924271624004118)

复现核查：仓库包含两份约 40.3 MB 的实际 `.pth` 文件。官方推理读取强度，开方为幅度，逐图 min–max 归一化，再反归一化并平方输出强度；不能把已经是幅度或显示变换后的 MAT 再当强度处理。[推理源码](https://github.com/YangtianFang2002/CL-SAR-Despeckling/blob/main/basicsr/predict.py)

作者 README 同时列出不同表示域的指标，明确 `psnr_view`、`ssim_view` 对应论文真实 Sentinel-1 评价。不能把其约 40 dB 的另一组数值当成上述 25.34 dB 的直接提升。[作者说明](https://github.com/YangtianFang2002/CL-SAR-Despeckling)

### 3.2 MuLoG-DRUNet：不能因标题写 PolSAR 就排除

正式论文题为 *Robustness to Spatially Correlated Speckle in Plug-and-Play PolSAR Despeckling*。它把适应空间相关噪声的去噪网络放入 MuLoG 框架，而非直接对 SAR 套用通用高斯去噪器；作者报告可跨传感器及 SAR/PolSAR/InSAR 等形式使用。[作者机构论文页](https://researchportal.ip-paris.fr/en/publications/robustness-to-spatially-correlated-speckle-in-plug-and-play-polsa/)

已检查官方 `py_functions.zip` 内的 `mulog.py`：二维数组走 `D=1` 分支，输出仍为二维，因此可以作为单强度候选。发布包还有 `models.zip`、`generic_model.pth` 的加载代码与 notebook；`L` 需要由数据的视数/有效视数设定，不应在元数据不明时默认全部 `L=1`。[官方发布](https://gitlab.telecom-paris.fr/ring/mulog-drunet)

独立的 CAID 2025 论文在 UMBRA、CAPELLA 真实 SLC 上使用公开预训练模型，Table III 报告 MuLoG-DRUNet 的 ENL 为 367.7、MERLIN 为 257.6、SAR-CAM 为 121.9。这是较强抑斑能力的证据；ENL 不衡量细节与旁瓣，不能据此宣布 MuLoG-DRUNet 总体质量第一。[独立实验原文](https://caid-conference.eu/wp-content/uploads/2025/11/CAID_2025_paper_15-1.pdf)

### 3.3 其余方法的主要复现条件

- **SDUDNet**：使用真实数据权重。原始 `test.py` 默认载入 `synthetic.pth`，固定 256×256，并含需调整的路径；不能只运行默认脚本就认定复现真实实验。论文真实实验规模较小，且描述真实训练数据取自 Speckle2Void 示例；若采用同源测试图，应检查重叠。[官方入口](https://github.com/BFY-official/SDUDNet/blob/main/test.py)、[原论文公开全文](https://www.researchgate.net/publication/391687101_Speckle-Driven_Unsupervised_Despeckling_for_SAR_Images)
- **MERLIN**：官方推理要求 `[H,W,2]` 实部/虚部；已有 spotlight、stripmap、Sentinel TOPS 权重。不能从强度图“补出”真实相位。**SAR2SAR** 的推理不要求时间序列，但输入幅度和预训练模型的尺度/成像模式仍需匹配。[实现与模型](https://github.com/hi-paris/deepdespeckling)
- **S3DIP**：逐图优化，示例配置为 10,000 次迭代；官方流程使用 SAR-BM3D/FANS 引导以及与 looks 匹配的噪声直方图。它不需预训练，却不等于不需传统先验或廉价推理。[真实数据脚本与说明](https://github.com/IAPP-Group/S3DIP)
- **Speckle2Void**（TGRS 2022）：[官方代码](https://github.com/diegovalsesia/speckle2void)确有 TF1 模型、训练/预测 notebook 和 checkpoint。网络处理标量图，但原始真实数据流程包含依赖复数 SLC 的 decorrelation；仅有 intensity 时需重新核实预处理，不能无条件照搬。[论文](https://doi.org/10.1109/TGRS.2021.3065461)

## 4. 最新论文与实际代码交付：重点区别

| 方法 | 论文时间与入口 | 截至检索日的核查结果 | 建议 |
| --- | --- | --- | --- |
| **SDS-SAR** | ISPRS JPRS 231，2026，[论文](https://doi.org/10.1016/j.isprsjprs.2025.11.025) | [仓库](https://github.com/YYF121/SDS-SAR)列出数据/权重链接，但实际测试调用占位模型，只做 3×3 中值滤波，未加载 checkpoint | 论文重点关注；**暂不列已可复现方法** |
| **GLCNet，仓库名 LGCN** | JAG 146，2026，[论文](https://doi.org/10.1016/j.jag.2026.105135) | [仓库](https://github.com/yangyang12318/LGCN)有真实网络和训练代码，但未确认权重发布；README 仍说待接收后提供流程，测试配置/数据类还存在明显缺口 | 有竞争力的真实多传感器论文候补，先补齐正式发布材料 |
| **SemDNet** | ESWA 296，2026，[论文](https://doi.org/10.1016/j.eswa.2025.129200) | [仓库](https://github.com/BFY-official/SemDNet)部分模型公开，三个阶段权重未提供；测试脚本用随机张量替代语义输入 | 部分代码，不是完整可用系统 |
| **MambaDIP** | JSTARS，2026-06，[论文](https://doi.org/10.1109/JSTARS.2026.3702550) | 已确认正式论文；本次未定位可核验的官方实现。包含去斑与增强，评价目标需区分 | 跟踪论文，暂不承诺直接复现 |
| **Closed-Form Nonlocal Shrinkage** | arXiv:2608.15028，v2 为 2026-09-06，[论文](https://arxiv.org/abs/2608.15028) | [作者代码](https://github.com/Teriri1999/Geometry-Calibrated-Closed-Form-Shrinkage)包含完整确定性算法、真实/合成 runner 与样例，不需训练；非商业研究许可 | 最新、易开展探索的候选；尚非经同行评议确认的领先结论 |
| **Speckle2Self（SAR 版本）** | Remote Sensing 17(23)，2025，[论文](https://www.mdpi.com/2072-4292/17/23/3840) | Transformer 自监督、含真实实验；本次未找到对应官方模型仓库/权重 | 不要混用同名医学超声项目的代码 |
| **DeCo-Despeckle** | JSTARS，2025，[论文](https://doi.org/10.1109/JSTARS.2025.3625996) | 本次未定位官方源码/权重；near-real 无配对训练仍使用高质量参考域，并非只需一张真实噪声图 | 论文候选；搜索结果中的 Pixel2Pixel 不是它的实现 |
| **ECDM** | Remote Sensing，2025，[论文](https://www.mdpi.com/2072-4292/17/17/2970) | 合成预训练、真实多时相微调的扩散路线；本次未确认可运行的官方发布 | 需代码及训练数据条件，不以摘要速度提升直接下结论 |

SDS-SAR 的判断来自实际源码，而非仅凭文件名。核查版本 `82541722491054b1ca6605012e0561bcf0cc1d6b`：[模型文件](https://github.com/YYF121/SDS-SAR/blob/82541722491054b1ca6605012e0561bcf0cc1d6b/src/models/placeholder_model.py)、[测试入口](https://github.com/YYF121/SDS-SAR/blob/82541722491054b1ca6605012e0561bcf0cc1d6b/scripts/test.py)。这一结论只针对该公开版本，不判断论文算法本身的效果。

Closed-Form Nonlocal Shrinkage 论文自报在 5 个传感器、6 种真实配置上平均 ratio 偏差最小。其[真实 runner](https://github.com/Teriri1999/Geometry-Calibrated-Closed-Form-Shrinkage/blob/main/scripts/run_real.py)对 miniSAR 使用额外尺度设置，真实/合成配置也有差异，因此应固定参数协议后实测，不能沿用“所有传感器完全无例外”的理解。

GLCNet 测试入口引用未导入/定义的数据类，默认配置文件也未在发布目录找到；SemDNet 的 `test_refinednet.py` 则以随机六通道张量充当语义信息。本次未运行，宜表述为交付缺口，不扩大成“论文方法无效”。[GLCNet 测试入口](https://github.com/yangyang12318/LGCN/blob/main/submit_test/test.py)、[SemDNet 测试入口](https://github.com/BFY-official/SemDNet/blob/main/test_refinednet.py)

## 5. 扩散、Mamba 与复数/多通道方向的补充

### 扩散模型不能仅凭架构判定优于成熟方法

**SAR-DDPM**（GRSL 2023）有[官方训练代码](https://github.com/malshaV/SAR_DDPM)；**SAR-DDPM-Aggregation** 对应 Sensors 2025 的 *On Denoising Diffusion Probabilistic Models for Synthetic Aperture Radar Despeckling*，有[作者实现](https://github.com/asp6244/SAR-DDPM-Aggregation)。两者展示的通用预训练起点不能直接当成已训练好的 SAR 去斑 checkpoint；本次未确认即用的 SAR 专用权重。

后者的论文明确讨论了生成结果可能更锐利，却出现幻觉及定量结果逊于普通 U-Net 的情况。因此应把辐射和结构保真放在“看起来清楚”之前。[Sensors 原论文](https://www.mdpi.com/1424-8220/25/7/2149)

另检索到 **DiSpeckle**（[TGRS 2025](https://doi.org/10.1109/TGRS.2025.3630133)）、**RSS-Net**（[JSTARS，2025 DOI](https://doi.org/10.1109/JSTARS.2025.3647971)）、**DeepMURE**（[ICIP 2025](https://doi.org/10.1109/ICIP55913.2025.11084619)）、**Self-Supervised Score-Based Despeckling via Log-Domain Transformation**（[2026 预印本](https://arxiv.org/abs/2601.14334)）。本次未确认对应官方完整可运行发布，列观察清单，不据此认定不可复现或效果不好。

### 有复数、多通道数据时另设比较组

**MuChaPro / Just Project!**（TGRS 2025）通过多通道投影、单通道去斑与协方差重建处理多通道 SLC，是值得阅读的新方向；但本次未核验完整独立方法仓库，不能把 MERLIN 包的成熟度等同于 MuChaPro 全流程已交付。[论文全文](https://arxiv.org/html/2408.11531v2)、[正式论文](https://doi.org/10.1109/TGRS.2025.3531957)

**PolMERLIN**（GRSL 2024）也不能直接作为最强结论：它的多极化实虚独立性假设受到正式评论质疑，且本次未核验官方代码/权重。[原论文](https://arxiv.org/html/2401.07503v1)、[评论 DOI](https://doi.org/10.1109/LGRS.2024.3387994)

**Sublook2Sublook**（[TGRS 2024](https://doi.org/10.1109/TGRS.2024.3397815)）依赖 SLC 子视构造，本次也未确认完整公开实现。上述方法不宜和仅有单幅强度图的方法强行放入同输入条件排行榜。

**SAR-DUCK** 是另一项值得记录的 2026 年新进展：[RING 官方仓库](https://gitlab.telecom-paris.fr/ring/sar-duck)已有 MERLIN 和不确定性模型、代码及样例，对应[2026 年 8 月预印本](https://telecom-paris.hal.science/hal-05595508)。但源码去斑阶段仍调用 MERLIN，新贡献主要是不确定性量化与辐射变化检测；当前模型面向经过特定预处理的 Sentinel-1 IW1 VV 复数数据，不能当成单强度去斑的新冠军。

## 6. 当前项目如何开展有说服力的对比

### 先处理现有数据域的不确定性

项目已有的[真实数据审计](D:/research/sar_transformer-main/output/real_sar_quality_audit_20260907/README.md)记录：5,876 个 MAT 图块的灰度层次有限，其中 5,626 个满足开方型离散格点；虽然协议称其为 intensity，原始数据到 MAT 的完整转换过程仍未追溯。

因此，接入前应优先核查强度/幅度、缩放、裁剪及原始位深。不能凭算法要求直接平方或开方“修复”。如果这一步不清楚，CL-SAR、SAR2SAR 和 MuLoG-DRUNet 之间的差异可能包含数值域错配，而非算法能力差异。当前数据来源的传感器/成像模式信息也不足，不适合直接假定与某个预训练模型匹配。

### 建议的最小充分比较组

| 层级 | 方法 | 目的 |
| --- | --- | --- |
| 现有参照 | 本项目方法、Trans-SAR、SAR2SAR、SDUDNet、SAR-BM3D | 保持既有研究上下文；复核统一评价入口 |
| 第一批新增 | CL-SAR、MuLoG-DRUNet | 增加真实相关斑点学习与统计模型/PnP 两条强路线 |
| 第二批新增 | NL-SAR、S3DIP | 增加可靠传统非局部方法及单图自监督方法 |
| 探索组 | Closed-Form Nonlocal Shrinkage | 检验最新免训练方法，明确预印本身份 |
| 有新数据后 | MERLIN；再考虑多通道方法 | 在 SLC/极化数据上独立建立匹配输入条件的测试组 |

SAR-CAM 本地已有适配代码，但历史短程演示训练不能替代论文级完整复现；其[官方仓库](https://github.com/JK-the-Ko/SAR-CAM)有训练/测试代码，预训练模型仍列为后续工作。若纳入主表，应明确是本地重训、训练预算与数据划分，而非声称使用作者正式权重。

### 评价协议

1. 固定测试父图与同质区域，按父图聚合，防止同一大图的相邻 patch 泄漏到训练和测试。
2. 同时报告 ratio 图及其结构残留、同质区域 ENL、均值/辐射保持、边缘和点目标保持。ENL 越高可能只是更平滑，不能单独定胜负；NIQE/BRISQUE 也仅作辅助。[真实 SAR ratio 评价研究](https://www.mdpi.com/2072-4292/17/24/4048)
3. 无干净参考时不报告“真实 PSNR”作为真值指标；若使用多时相均值或其他伪参考，明确其构建方式与残余误差。
4. 固定强度/幅度转换与显示范围，禁止为单一模型单独调节对比度后据视觉选优；保留浮点输出。
5. 对依赖 `L` 的方法使用统一、由输入确定的估计规则，并报告敏感性；模型选择和阈值不使用测试真值。
6. 记录仓库提交、权重校验值、配置、依赖、预处理、硬件和运行时间；将 S3DIP 的逐图优化耗时与前向网络推理明确区分。

这些是下一轮实验的建议，不是本次已经实施的实验。

## 7. 权威资源入口与许可说明

- [Télécom Paris RING 官方代码组](https://gitlab.telecom-paris.fr/groups/ring)：SAR2SAR、MuLoG 等统计与自监督方法的重要来源。
- [HI! PARIS deepdespeckling](https://github.com/hi-paris/deepdespeckling)：MERLIN、SAR2SAR 的 PyTorch 推理和模型。
- [GRIP / Università di Napoli 去斑专题](https://www.grip.unina.it/remote-sensing/despeckling)：传统、学习方法与真实 SAR 评价资源。
- [Charles Deledalle 官方方法页](https://www.charles-deledalle.fr/pages/nlsar.php)：NL-SAR 及相关研究软件。
- [IEEE IGARSS 2026 官方教程](https://2026.ieeeigarss.org/tutorials.php)：物理一致 SAR 学习与去斑方向；用于理解研究问题，不是性能排行榜。
- [Practical SAR Despeckling Codes](https://github.com/YangtianFang2002/Practical-SAR-Despeckling-Codes)：实用检索入口，但其中标注重实现的方法不能当成全部由原作者提供。

“公开代码”与“许可完整的开源软件”需区分：deepdespeckling、SAR-CAM 及上述两个 SAR-DDPM 仓库有 MIT 标注；NL-SAR 使用 CeCILL；SAR-BM3D 官方包具有非营利限制，Closed-Form Nonlocal Shrinkage 为非商业研究许可。CL-SAR、SDUDNet、MuLoG-DRUNet 等本次未确认完整顶层许可，研究复现前仍应检查用途限制，不能默认允许商业使用。[SAR-BM3D 官方许可说明入口](https://www.grip.unina.it/download/prog/SAR-BM3D/version_1.0/README.txt)

核查版本记录：CL-SAR `b12129d1d3448750b9098b239397587eeb359857`；SDS-SAR `82541722491054b1ca6605012e0561bcf0cc1d6b`；LGCN `274af0e58d3dd565fd1f7a24d8f21105687636e7`；SAR-DDPM `6a14fe99cb3aaf2f106d551efae2b08f0ad1f9d7`；S3DIP `ea018dab23ef44ec2106b182661d3bec921fc0d4`。仓库以后更新时，需要重新核对上述复现状态。

## 8. 一句话建议

先确认当前 MAT 的物理数值域，再接入 **CL-SAR + MuLoG-DRUNet**；用已有 **SAR2SAR + SDUDNet + SAR-BM3D** 保持对照完整，视资源增加 **NL-SAR/S3DIP**。有原始复数数据时加入 **MERLIN**。不要把发布时间最新、README 写了权重、或某项 ENL 最高，直接当作真实 SAR 效果最好的证据。
