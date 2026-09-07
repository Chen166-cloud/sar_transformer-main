# ICSPS 2026 论文包装任务提示词（交给 Codex）

> **协议更新：** 本文件只能在下列正式协议约束下使用：`ICSPS26-FROZEN-v2`、UCM 全 21 类/2,100 sources/8,400 pairs、RealSAR no-GT16。数字只能来自通过 artifact verifier 的正式 CSV/JSON，不得使用中期报告、smoke 或历史实验值。合成域只报告公认的 PSNR/SSIM，不测 Params、MACs 或 latency；AMS 前后只比较真实 SAR。如下文与 [网络设计与实验方案](./ICSPS2026_SAR去斑网络设计与实验方案.md)、[远程运行手册](./ICSPS2026_远程服务器运行手册.md) 或 [实验记录表](./ICSPS2026_实验记录表.md) 冲突，以这三份为准。

你现在是一名熟悉 IEEE/信号处理/遥感图像恢复论文写作的资深科研工程师。你的任务不是简单“缩写我的中期报告”，而是基于我提供的 **Markdown 协议文档、完整项目源码、训练/测试日志、通过验证的正式实验 CSV/JSON、模型权重说明与已冻结图片**，将当前 SAR 去斑工作重新组织为一篇适合投稿 **ICSPS 2026 Special Session 5: Synthetic Aperture Radar Signal, Information Processing Technology and Application** 的英文会议论文。中期报告只能用作背景，不能作为实验数字来源。

目标是：**优先提高论文的可审查性、可信度、主题匹配度和录用概率。**

---

## 1. 投稿目标与硬约束

1. 目标会议：ICSPS 2026。
2. 优先投稿专题：Special Session 5 — Synthetic Aperture Radar Signal, Information Processing Technology and Application。
3. 论文方向必须明确定位为：**SAR image processing / SAR despeckling / machine learning for SAR signal and image processing**。
4. 正文使用英文。
5. 按 ICSPS 官方双栏模板组织，目标控制在 **5–6 个完整页面（包括图、表、参考文献）**。不要为了保留中期报告内容而写成“大而全”的学位论文风格。
6. 论文只写已经被源码、日志和通过 artifact verifier 的正式实验输出支持的内容。**严禁编造或修改网络实现、参数、数据划分、实验值、参考文献或 SOTA 结论**。只允许在不改变值和统计口径的前提下调整有效数字、排版和表述。
7. 若资料之间存在冲突，必须先审计并报告冲突。

---

## 2. 论文核心定位

这篇 ICSPS 论文不要以“Noise-Guided Transformer”作为主创新，也不要在标题中突出 Noise-Guided / Noise Estimator。

建议将全文收敛到以下三条技术主线：

### Contribution 1 — Frequency-enhanced reconstruction
在 Transformer-CNN 编解码主干的解码阶段引入 FFT-based frequency refinement，使网络显式处理 SAR 去斑中“speckle 高频扰动”和“真实边缘/纹理高频结构”难以区分的问题。

### Contribution 2 — Log-intensity restoration with original-intensity residual compensation
利用 SAR speckle 的乘性退化特性，在 log-intensity 表示中承担主要恢复任务，并使用 original-intensity 输入进行轻量残差补偿，以减少 log 变换造成的局部辐射强度和结构细节偏移。

注意：如果源码实际结构只是“log 域主支路 + 原始输入残差融合”，论文中必须这样准确描述，不要夸大为两个彼此独立的完整去斑器。

### Contribution 3 — Self-supervised real-SAR adaptation
在没有 clean real-SAR target 的条件下，通过 masked reconstruction 对监督预训练模型进行真实 SAR 域自监督适应，用于缓解基于模拟 speckle 训练与真实 SAR 之间的 domain gap。

Noise-aware prior / noise gate、多尺度融合和 bottleneck refinement 可以作为支撑模块和消融项，但不要把它们全部提升成同等级的“核心贡献”。

---

## 3. 推荐论文标题方向

优先采用以下标题语义，不必机械照抄，但不要偏离：

**Frequency-Enhanced Dual-Domain SAR Despeckling with Self-Supervised Real-Data Adaptation**

如果源码审计后发现“Dual-Domain”容易造成两个完整分支的误解，则改成更严谨的：

**Frequency-Enhanced SAR Despeckling with Log-Intensity Fusion and Self-Supervised Real-Data Adaptation**

模型内部名称可以继续保留 `DFNG-SARNet`，但标题和摘要不要突出 `Noise-Guided`。

---

## 4. 在正式写论文前，必须先做源码与实验审计（但不要运行源码，本地非运行环境）

请先遍历项目源码、配置、日志文件，并生成 `evidence_audit.md`。至少完成以下审计。

### 4.1 网络结构审计

从源码确认并列出：

- 实际模型入口类和 forward 路径；
- Transformer encoder 的 stage 数、embed dimensions、heads、depths、sr ratios；
- decoder 的真实实现；
- FFT/frequency refinement 模块实际计算流程；
- noise-aware prior / gate 的实际输入和输出；
- bottleneck refinement；
- `DualDomainFusion` 的真实实现；
- log transform、inverse transform、residual scale `gamma`；
- 最终激活函数；
- 训练与推理时输入输出的数值域。

任何中期报告描述若与源码不一致，以**实际最终实验所使用的源码**为准，并明确列出差异。

### 4.2 数据划分泄漏审计

重点检查真实 SAR 数据是否由同一幅大图裁出的相邻 patch 随机分散到了 train/val/test。

如果文件名或 manifest 中含有原图 ID、产品 ID、轨道、日期、patch 坐标 `x/y` 等信息：

- 建立 `scene_id / product_id / acquisition_id`；
- 检查同源 patch 是否跨集合；
- 如果存在泄漏，优先重新进行 group-wise split；
- 正式论文主结果应优先使用按原始影像/产品/场景隔离的数据划分；
- 不得把可能泄漏的随机 patch split 包装成严格独立测试。

### 4.3 实验数值一致性审计

所有数字必须从远程服务器产生并通过校验的 formal CSV/JSON 及其 SHA-256 证据链引用。如果资料冲突，停止写结论并报告冲突；中期报告、smoke/profile 输出或旧协议数字不得填入正式论文。

### 4.4 数值域审计

必须确认：

- 输入是 amplitude、intensity 还是 log-intensity；
- `[0,1]` 归一化的具体公式；
- `clean = G^2` 等操作是否与代码一致；
- 模型输出是否可能为负；
- 若使用 `Tanh`，如何保证 ENL 和 ratio image 等强度域指标计算时输入为正；
- PSNR/SSIM 与真实 SAR 指标分别在哪个域计算；
- 可视化是否进行了 sqrt / log / percentile stretch。

论文必须使用统一、可复现的定义。

---

## 5. 术语必须严格修正

1. NWPU-RESISC45 和 UCMerced_LandUse 原始数据是光学遥感场景图像。如果它们只是被加入模拟乘性 speckle，则论文写成：
   - `optical remote-sensing images corrupted with simulated multiplicative speckle`；
   - 或 `synthetically corrupted remote-sensing images`。
   不要直接声称它们本身是 SAR 数据。

2. 如果 `NoiseEstimator` 没有 noise ground truth 监督，不要写成 accurate noise estimation。优先称为：
   - `noise-aware prior map`；
   - `noise-aware modulation map`。

3. `AMS` 是真实 SAR 自监督 adaptation strategy，不是一个新的独立网络结构。统一写成：
   - `DFNG-SARNet`：基础网络；
   - `DFNG-SARNet + self-supervised adaptation`：真实域适应后的实验版本。

4. 不要把 `ENL ↑` 单独解释成“图像质量一定更好”。必须联合文献定义的 M、EPI 和预注册视觉结果讨论，避免过平滑导致虚假 ENL 提升；不得加入自制替代指标。

---

## 6. 实验部分的最优组织方式

ICSPS 只有 5–6 页，因此不要照搬中期报告的全部实验演进史。正文实验建议只保留四组证据。

### Experiment A — Main comparison
优先整理已有横向对比结果。如果源码/实验目录里还没有横向对比结果，则不要虚构，生成 `missing_experiments.md` 明确指出最少还需要补的实验。

优先基线集合：

- Lee filter；
- SAR-BM3D；
- TransSARV2（按项目中的原始归属和统一 100k 训练链）；
- Ours Full；
- 资源允许时把 pinned official SAR-CAM 作为 P2 增强比较。

Noisy 只是 identity/reference row。未进入冻结代码、数据与统一评价链的 Speckle2Void 或其他方法不临时加入本次正式结果。

要求统一数据、输入尺寸、评价指标和可视化设置。

### Experiment B — Compact ablation
只保留最能解释论文贡献的消融：

- Intensity-only；
- Log-only；
- Full w/o all FDR；
- Full。

不要为了展示“做了很多实验”而塞入过多模型变体。

### Experiment C — Cross-dataset generalization
使用 UCMerced 全部 21 类、2,100 个固定 source，对每个 source 在 $L\in\{1,2,4,8\}$ 下各生成一对，共 8,400 个固定测试对，证明模型在未参与主训练的数据来源上的恢复能力。

注意论文措辞必须明确这些是对光学遥感图像施加模拟 SAR speckle 后形成的外部测试数据，而不是“真实 SAR 泛化”。

### Experiment D — Real-SAR adaptation
比较：

- noisy input；
- DFNG-SARNet；
- DFNG-SARNet + self-supervised adaptation。

本协议不使用 GT16，不报告任何 GT16-based quasi-reference metric。真实 SAR 指标和可视化必须遵循最终实验方案与记录表。

如果真实 SAR 数据存在 patch-level leakage，先修复数据划分，再将新的 group-split 结果作为论文主结果。

---

## 7. 可视化必须精简但有说服力

正文只保留高信息密度图。

### Figure 1 — Overall method
一张图说明完整方法：

`SAR intensity → log transform → Transformer encoder → frequency-enhanced multi-scale decoder → noise-aware modulation → log-domain restoration → original-intensity residual fusion → output`

并在旁边用小模块表示 real-SAR masked self-supervised adaptation。

不要再分别画 MSFormer、MSFF、多个历史版本的完整网络图。

### Figure 2 — Qualitative comparison
选择 2–3 个最有代表性的样例，统一显示动态范围和 crop 区域：

- clean/noisy（合成场景）；
- representative baselines；
- proposed method；
- local zoom；
- real SAR 时增加 ratio map。

crop 优先覆盖：建筑边缘、细纹理、强散射点。

---

## 8. 推荐表格布局

目标控制为 3 张核心表：

### Table 1 — Main quantitative comparison
方法 vs 四个 look 和 macro PSNR/SSIM（UCM 合成数据）。不加入自定义频谱指标或复杂度列。

### Table 2 — Ablation
Intensity-only / Log-only / Full w/o all FDR / Full。

### Table 3 — Real SAR adaptation
Noisy / TransSARV2 / Ours-base / Ours+AMS，按中期报告既定的 ENL、M、EPI 表结构记录；不得加入自定义指标或时间/复杂度列，也不得从 no-GT16 数据生成准参考质量指标。M 与 EPI 只有在实现与所引原论文公式核对一致后才能写入正式数值，不能把内部诊断量改名冒充。

如果页面不足，训练参数、完整数据集说明和次要辅助结果不要单独占大表，但不得删除上述四个预注册消融变体中的核心证据。

---

## 9. 论文结构与篇幅

请按会议论文而不是学位论文组织：

### Abstract
约 160–220 words。
必须包含：problem → method → three key ideas → synthetic/real evaluation → 1–2 个最可靠的量化结果 → conclusion。
不要在摘要里罗列所有模块名。

### 1. Introduction
约 0.7–0.9 页。
逻辑：

1. SAR multiplicative speckle 的挑战；
2. CNN/Transformer 方法的局限；
3. 单一空间域难以处理高频噪声与高频结构冲突；
4. synthetic-to-real domain gap；
5. 本文方法；
6. 三条 contributions。

不要写过长的 SAR 教科书背景。

### 2. Related Work
约 0.4–0.6 页，可与 Introduction 合并以节省版面。
覆盖：traditional despeckling、deep SAR despeckling、Transformer-based despeckling、self-supervised real-SAR adaptation。
必须讨论与最近 noise-guided Transformer / multi-scale fusion SAR 工作的区别，避免审稿人认为高度重合。

### 3. Proposed Method
约 1.6–2.0 页。
只讲最终模型：

- overall pipeline；
- frequency-enhanced decoder；
- log-intensity + original-intensity residual fusion；
- noise-aware gate（简述）；
- self-supervised real-SAR adaptation。

中期报告中的模型迭代历史不要进入正文。

### 4. Experiments
约 2.0–2.4 页。
顺序：

1. datasets & implementation（精简）；
2. main comparison；
3. ablation；
4. cross-dataset generalization；
5. real-SAR adaptation；
6. qualitative analysis。

### 5. Conclusion
约 0.2–0.3 页。
只总结已被实验支持的结论，不写宏大的未来 Agent 方向。

---

## 10. 写作风格要求

1. 使用 IEEE/信号处理会议常见的克制学术英语。
2. 少用 `novel`, `significant`, `superior`, `state-of-the-art` 等强词，除非实验真的支持。
3. 不使用“the proposed method perfectly solves...”一类夸张表达。
4. 每个技术设计都回答两个问题：
   - 为什么 SAR speckle 需要它？
   - 哪个实验直接证明它有效？
5. 所有 contribution 都必须在 ablation 或 real-SAR experiment 中找到对应证据。
6. 不要把多个普通模块堆叠包装成多个“创新点”。

---

## 11. 参考文献要求

1. 优先使用原论文、IEEE/Elsevier/Springer/Remote Sensing 官方来源。
2. 不得编造 DOI、作者、年份和期刊。
3. 如果当前环境不能联网核验文献，就保留明确 TODO，不要生成看似真实但未经验证的 BibTeX。
4. Related Work 至少覆盖：
   - Lee / classical despeckling；
   - SAR-BM3D；
   - Transformer-based SAR despeckling；
   - Speckle2Void / SAR2SAR 等真实 SAR 自监督工作；
   - 2025 noise-guided Transformer + multi-scale fusion 的强相关工作。

---

## 12. 最终交付物

请按以下顺序工作，不要一上来直接生成一篇“看起来完整”的论文。

### 第一步：审计
生成：

`evidence_audit.md`

内容包括：

- 最终模型真实结构；
- 中期报告 vs 源码差异；
- 数据划分检查；
- 指标实现与数值域检查；
- 所有实验结果的文件来源；
- 相互冲突的数值；
- 能写进论文的已验证结论；
- 不能写进论文的未经验证结论。

### 第二步：论文证据矩阵
生成：

`claim_evidence_matrix.md`

每一行包含：

`论文 claim | 对应源码 | 对应实验文件 | 对应表/图 | 是否可复现 | 风险`

确保每一个 contribution 都有证据支持。

### 第三步：论文结构
生成：

`paper_outline_icsps2026.md`

给出标题、摘要逻辑、各 section、每节目标字数/版面、图表位置和 contribution 对应关系。

### 第四步：正文
基于 ICSPS 官方模板生成最终英文论文草稿，优先输出 LaTeX：

- `main.tex`
- `references.bib`
- `figures/`

如果项目中已有 Word 模板工作流，也可以额外生成可粘贴到 Word 模板的正文，但 LaTeX 版本作为主稿。

### 第五步：投稿前检查
生成：

`submission_checklist.md`

检查：

- 是否 5–6 页；
- 是否选择 Special Session 5；
- 图中文字是否清晰；
- 所有数字是否能追溯到结果文件；
- 是否存在数据泄漏；
- 是否与 2025 noise-guided Transformer 工作充分区分；
- 是否包含必要 baseline；
- 是否存在未核验引用；
- 摘要、标题、contribution 是否一致；
- 正文是否出现中期报告式的冗余章节。

---

## 13. 最重要的执行原则

**不要把我的中期报告机械翻译成英文。**

你要把它重新设计成一篇短、聚焦、有清晰证据链的 ICSPS SAR 专题会议论文：

`SAR multiplicative speckle problem → frequency-domain structure modeling → log/intensity complementary restoration → self-supervised real-SAR adaptation → controlled ablation + external comparison + real-SAR evidence`。

可以改善图表排版、有效数字显示和学术表述，但不得更改、挑选或“美化”实验数值。

如果源码对应函数与报告不一致，先停下论文结论生成，完成审计并明确告诉我哪里不一致、应该采用哪套可复现结果，以及是否需要重新运行实验。
