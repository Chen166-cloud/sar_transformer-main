# ICSPS2026 论文逻辑与表述检查记录

检查日期：2026-09-08。对象为当前主稿、五张结果表、方法实现、训练配置和重点引用；修订已写入 `ICSPS2026_paper.tex` 并重新生成 `ICSPS2026_paper.pdf`。

## 总体判断

论文从合成数据监督训练、频域重建与强度补偿，到真实 SAR 掩码适应的主线可以成立。原稿存在方法表述与实现不一致、消融因果解释过强、评价指标含义不够准确等问题，已修正可确定的部分。五张表的实验数字与最新源表一致，正文差值、排名和三场景 Average 未发现新的算术错误。

作者进一步提供两篇指标原文，并确认真实 SAR 两张表实际采用其中的指标实现。文献来源与两表指标体系已确认，不再将“是否混用不同 EPI/M 实现”列为待确认事项。本次原文核对另外发现仓库 M 函数与原文打印公式的归一化差异；M 有效样本数、外部基线与场景子集记录也尚未齐全，详见下文。

## 作者本轮确认

- 三场景 UCMerced 使用 **L=4**，以最后一次更正为准；先前的 L=1 回答已被覆盖。
- Residential 对应 **dense residential**，每类 100 个样本。
- 真实 SAR 的 ENL 报告**均值**，不改成中位数。
- Tables IV/V 的 EPI 与 M 实际采用下列两篇文献中的指标实现。该确认适用于两张真实 SAR 表，不再依据仓库旧脚本推测实验使用了另一套指标。
- 实验数字以 `docs/实验结果表.md` 为准。本轮没有修改该文件，也没有重新计算模型输出。

源 Markdown 的旧说明仍含 `ENL median [IQR]`，且尚未补写 L=4/dense residential；论文已按作者本轮确认写明。真实三场景 AMS Average 保持 **72.7568 / 1.454366 / 0.798884**。

## 已修正的问题

| 位置 | 原问题 | 本轮处理 |
|---|---|---|
| 摘要、引言 | `blind` 易与盲点训练混淆；将整段适应称为 `lightweight` 缺少效率测量支持 | 摘要明确为无需输入视数；适应过程改用冻结编码器这一可验证描述，补偿器描述为浅层结构 |
| II-B | 将整个输出路径称为解析可逆，但后续还有学习补偿与裁剪 | 将解析可逆性限定于 bounded log 变换及其逆，不再暗示整网可逆 |
| II-B | DATNet 仅被概括为动态门控，遗漏与本稿较接近的频域设计 | 补入 Fourier 频带分解和频域专家，并准确区别本稿的逐频率坐标通道混合 |
| III-A，式 (1) | Gamma 第二参数是 scale 还是 rate 未明示 | 明确使用 shape/scale，使单位均值与方差 1/L 的说明无歧义 |
| III-B | 瓶颈和解码器模块的关系过于简略，且把功能动机写成必然效果 | 说明瓶颈包含额外局部残差包装与可学习缩放；五个解码器 FDR 使用相同基本运算、独立参数 |
| III-D | 将 guidance 称为 `unsupervised`，忽略其接受总体监督损失的梯度 | 改为与恢复网络联合学习、没有专门噪声图或视数标签的潜在引导图 |
| III-D，式 (6) | 由缩放初值 0.1 推断补偿始终很小，并暗示补偿器不会再执行去噪 | 说明初值不构成幅值约束；仅陈述逆变换后的强度残差补偿和最终裁剪 |
| III-E，式 (7) | 使用平方误差和的写法，与实际 mean MSE 不一致 | 加入 1/(HW)，并定义 TV 为水平、垂直邻差绝对值均值之和 |
| III-E，式 (8) | 掩码分母写成 sum + epsilon，与实现不符；“移除 20%”容易理解为固定比例 | 分母改成 max(sum, 1)；说明每个像素独立以 0.2 概率掩码 |
| III-E | 固定验证掩码被解释成保持 noisy target 不变 | 改为保持每轮验证所评价的像素位置一致；保留训练重采样、推理不掩码的区别 |
| IV-A、IV-H | “一次采集”缺少当前数据来源证据 | 改为同一数据集合内的父图隔离测试，保留不支持跨采集、跨区域或跨传感器结论的范围说明 |
| IV-A、IV-B | 非 256×256 来源的处理未写清；训练设置笼统覆盖全部学习方法；学习率像固定值 | 补明合成前双线性调整尺寸；训练预算限定于合成监督模型；给出初始学习率和 plateau 调度配置，patience=4 不写成第 4 次未改善即衰减 |
| IV-B、Table II、IV-D | Ours (Full)/Ours-base 命名关系及三场景协议不够明确 | 说明两者为适应前配置；三场景标题、表注、行名和分析统一 L=4/dense residential，与四视数宏平均分开解释 |
| IV-B、IV-F、Tables IV/V | ENL 聚合不明确；将 EPI 称为边缘能量并直接等同于真实边缘恢复 | 明确 ENL 均值；按正式实现描述相邻对角像素绝对差分和之比，指出残余斑点也会抬高 EPI；改写 `edge energy`，联合解释 ENL/M/EPI 与比值图 |
| IV-E | 声称增益并非来自容量增加；由有限消融推断互补/交互作用 | 改为已测配置内的贡献证据，明确移除模块也改变容量，当前设计不能分离参数量因素或量化交互作用 |
| IV-C、全文衔接、参考文献 [17] | 多处重复解释同一证据范围；个别主语不清；文献标题漏括号 | 合并重复的跨视数讨论，明确“本方法达到最高 PSNR”；将标题修正为 `filtered (Pol)SAR images` |

方法修订主要对照：[网络实现](D:/research/sar_transformer-main/transform_main.py:1416)、[监督 MSE](D:/research/sar_transformer-main/train_icsps2026.py:623)、[TV 定义](D:/research/sar_transformer-main/train_icsps2026.py:96)、[AMS 掩码损失](D:/research/sar_transformer-main/train_icsps2026_ams.py:104)、[学习率调度](D:/research/sar_transformer-main/train_icsps2026.py:387)、[消融配置](D:/research/sar_transformer-main/ablation_config.py:19)。本轮仅修改论文，未修改这些实现。

## EPI 与 M 原文核对（作者补充后更新）

| 指标 | 作者提供的原文 | 核对结果 |
|---|---|---|
| EPI | [Ma 等，Remote Sensing 2024, 16, 1992](D:/Users/Chen/Downloads/remotesensing-16-01992-v2.pdf) | 第 9 页第 3.2 节式 (16)：滤波图与原始噪声图中对应相邻像素的绝对差总和之比。现有文献 [17] 正确。 |
| M | [Gomez 等，Remote Sensing 2017, 9, 389](<D:/Users/Chen/Downloads/remotesensing-09-00389 (1).pdf>) | 第 8–9 页式 (1)–(3)：组合比值图的一阶统计偏差与随机置换所揭示的结构残余；理想值为 0，越小越好。现有文献 [16] 正确。 |

EPI 的分子是恢复图的相邻差分，分母是原始 speckled intensity 的相邻差分。第 9 页式 (17) 是另一项 EPD，不能与 EPI 混写。原文式 (16) 没有强制规定相邻方向；论文中的对角邻接来自本地实现约定。EPI 也不是 Sobel 梯度相关或干净参考边缘准确率，因此保留联合解释指标的文字。[本地 EPI](D:/research/sar_transformer-main/classic_sar_metrics.py:40)

M 使用 noisy/filtered 比值图：一阶项比较局部 ENL 与原始 ENL，并衡量比值均值偏离 1 的程度；结构项比较原始和随机置换比值图的 Haralick homogeneity。原文第 9 页给出窗口边长 25、容差 0.03、8 级量化和 100 次随机置换。正文已展开这两项含义，并明确两张真实 SAR 表使用上述 M/EPI 定义。

## 仍需记录说明的事项

### 1. M 一阶残差的具体归一化

所提供 M 原文第 8 页式 (1) 打印为 `r = 0.5 × sum_i(r_ENL(i) + r_mu(i))`；仓库当前函数采用 `0.5 × mean_i(...)`，一阶项相差选中均质窗口数 n 这一因子。n 是筛选得到的区域数，不是固定超参数；全文未找到该一阶项另除以 n 的说明。[当前函数第 203 行](D:/research/sar_transformer-main/classic_sar_metrics.py:203)

同时，原文自身的展示存在尺度不一致：第 9 页式 (3) 写 `M = r + delta_h`，但第 21 页 Table 11 默认 FANS 行的 `r=0.4833`、`delta_h=20.89`、`M=10.6867` 约等于两项相加后除以 2。因此，不能只凭打印公式擅自改写本实验数值或“纠正”实现。

这项发现说明当前仓库函数与原文打印式不能直接称为逐式一致，**不否定作者确认的指标来源，也不能据此判断已报告数据错误**。本次保留表格与代码，正文采用明确的组成与方向说明；复现记录宜保留实际使用的归一化约定、脚本版本及逐图结果。

### 2. 优先核实：M-index 的有效样本数

正式评估只对有效 M-index 求均值，因此“各方法输入同一批 592 个 patch”不必然等于“各方法 M 在同一有效集合求均值”。需提供逐方法 valid/592，并检查共同有效集合上的结论是否一致。[汇总代码](D:/research/sar_transformer-main/evaluate_icsps2026_real_enl_m_epi.py:210)

已有 Noisy 记录的 M 有效数为 0/592；其 ENL 均值约为 26.25318，而中位数约为 24.93025，前者对应表 IV 的 26.25。这支持 Noisy 行的均值说明，不能替代其他方法的逐图结果核验。[Noisy 汇总](D:/research/sar_transformer-main/records/icsps2026_real_noisy_classic_metrics/summary.json:10)

### 3. 补全外部基线与 60-patch 子集来源

真实 SAR 中 SAR2SAR/SDUDNet 的训练数据、checkpoint、是否额外适应，以及 SAR-BM3D 的真实视数设定或估计方式仍未充分披露。已删除“所有学习方法采用同一监督设置”的泛化表述，但可复现性还需要实际记录。

60-patch 子集还需父图清单、与 592-patch 测试集的包含关系、选择规则及模型来源。仓库的一个选择脚本按输入结构评分选每类前 20 张，不能据此未经溯源写成随机抽样。当前论文将其保留为单独的描述性场景评价。[选择实现](D:/research/sar_transformer-main/select_real_sar_60.py:342)

### 4. 补全三场景 UCMerced 的结果生成链

L=4 和 dense residential 已由作者确认，已不属于待确认项。仍需将源表对应到具体测试清单、checkpoint 和 SSIM 实现：旧三类脚本对整幅高斯滤波结果平均，正式指标实现使用有效窗口并去除 5 像素边界，二者可能产生不同 SSIM。应核实实际使用哪一个，不能由表值反推。[三类实现](D:/research/sar_transformer-main/test_ucmerced_3classes.py:94)、[正式 SSIM](D:/research/sar_transformer-main/sar_metrics.py:46)

## 重点引用核查

- Ma 2024 的主要新指标并非经典 EPI，但其第 3.2 节式 (16) 确实列出经典 EPI，因此该引用可保留，并已精确到公式。项目采用对角邻接约定，不声称直接复制作者另一个 EPD-ROA 函数。[期刊原文](https://www.mdpi.com/2072-4292/16/11/1992)
- DATNet 的 Fourier 频带处理有原文依据；SAR-FDD 的低/高频交互不能改称与本稿相同的 FFT 通道混合。[DATNet](https://www.mdpi.com/2072-4292/17/17/3031)、[SAR-FDD](https://www.mdpi.com/2079-9292/13/3/490)
- 所引真实数据页面支持 Noisy/参考数据的说明，但未充分证明所有 Noisy 父图来自一次采集，因此已收紧来源描述。[Mendeley 数据页](https://data.mendeley.com/datasets/fs455tz88y/1)、[EUSAR 出版商摘要](https://www.vde-verlag.de/proceedings-en/456286095.html)
- SAR2SAR 的多时相训练、Speckle2Void 的盲点与去相关、SDS-SAR 的强度输入与相关性感知采样概述未发现与重点来源矛盾。[SAR2SAR](https://arxiv.org/abs/2006.15037)、[Speckle2Void](https://arxiv.org/abs/2007.02075)、[SDS-SAR](https://www.sciencedirect.com/science/article/pii/S0924271625004642)

引用检查覆盖与核心论证相关的来源和书目信息，不代表已逐字阅读全文中的全部 21 篇参考文献。SDUDNet 的作者项目可访问，但本次未获得可读的 IEEE 全文，因此保留一般性的 unpaired clean/noisy 表述。[作者项目](https://github.com/BFY-official/SDUDNet)

## 编译与保留检查

- PDF 共 **7 页**，五张表数字及五幅图内容保留；未修改模型、训练代码和源结果表。
- Tables I/II/III/IV/V 分别位于第 **4/5/5/6/6 页**；Table IV 的满栏宽、数值右对齐和左对齐表注保留。
- 已检查逐页渲染，无内容遮挡、表格越界或缺失；公式、图表与 21 个文献条目引用均可解析。
- 最终稳定编译日志无错误、缺失引用、Overfull 或 Underfull hbox。保留第 4 页已有的一项 Underfull vbox 提示，视觉检查无溢出。
- 校验记录：`output/pdf/ICSPS2026_LogicReview/validation.json`；最终稳定日志保存在同目录。

当前文稿已根据作者确认明确两张真实 SAR 表的指标来源。后续记录应重点补明 M 的实际归一化约定与有效样本数；不再要求重复确认 EPI/M 采用哪一篇文献。
