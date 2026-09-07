> 2026-09-07 紧凑排版更新：当前 `draw_fig3.py`、SVG、PDF、PNG 和 `figure_manifest.json` 已按本轮要求更新。下面原交付说明及旧 layout proof/export QA 作为历史记录保留，其旧尺寸与布局描述不再适用于当前图。当前论文、实际尺寸、编译及验收结果见 [本轮报告](../ICSPS2026_LayoutRefinement/REPORT.md)。

# ICSPS 2026 — Fig. 3: Representation-consistent reconstruction and AMS

本交付完成指定初稿 `output/pdf/ICSPS2026_英文论文_LaTeX无图初稿.pdf` 中的 Fig. 3，依据当前论文源码、正式 Full 模型、数值域函数和 AMS 协议绘制可编辑矢量流程图。面板 (a) 解释 bounded log prediction 如何先精确逆变换，再进行同 intensity 域的残差补偿；面板 (b) 解释 noisy observation、mask、模型、masked objective、参数更新、固定验证选模和完整输入推理之间的关系。本图不重复 Fig. 1 的完整主干、Fig. 2 的 FDR 内部结构或实验表格。

现有源码、论文和配置足以支持两个面板，无需额外图像素材、clean target、预测文件或 checkpoint。没有捏造实验图像、性能指标或所选 best epoch；没有修改模型、协议、论文、数据，也没有重新训练。

## 交付文件

| 文件 | 用途 |
|---|---|
| `fig3_representation_ams.svg` | 可编辑矢量母版，主文字、变量和下标保留文本；16 个模块的框和文字已分组 |
| `fig3_representation_ams.pdf` | 单栏排版用矢量 PDF，嵌入 Arial/STIX 字体 |
| `fig3_representation_ams.png` | 已核实为 600 dpi、2092×4300 px 的高清预览 |
| `fig3_actual_size_150dpi.png` | 523×1075 px 的实际单栏尺寸预览 |
| `fig3_svg_render_check.png` | 独立 SVG 渲染器生成的 523×1075 px 校验图 |
| `fig3_layout_proof.pdf` | 带英文图注的单栏尺寸排版检查页，不是整篇论文重编译结果 |
| `draw_fig3.py` | 绘图、SVG 文字定位和模块分组代码 |
| `verify_workflow.py` / `workflow_trace.json` | 正式模型及原 AMS 函数的结构/流程核验代码与实际运行记录 |
| `render_svg_preview.cjs` | 独立 SVG 渲染检查代码 |
| `make_layout_proof.py` / `export_qa.json` | PDF/SVG 导出、排版 proof 检查代码与记录 |
| `figure_manifest.json` | 图宽、字体、线宽、文字溢出检查、依赖版本与图源 SHA-256 |
| `SOURCE_AUDIT.md` | 论文、配置、源码路径及行号；精确尺寸、连接和差异说明 |
| `caption.txt` / `fig3_insert.tex` | 英文图注及替换原占位图的 LaTeX 片段 |
| `requirements-drawing.txt` / `requirements-qa.txt` / `requirements-structure.txt` | 分别用于绘图、导出排版检查、可选模型结构核验的依赖 |

## 最终尺寸、样式与编辑

- 按当前 `IEEEtran` conference 模板的单栏宽输出：**88.5678705 mm × 182.0333333 mm**。宽度为 `21pc = 252 TeX pt`，换算为 PDF points 后约 `251.05853 pt`；图中不将两种 point 单位混用。模板依据见 `docs/ICSPS2026_LaTeX/IEEEtran.cls:1734–1735`。
- 全部图内标注为英文，字体为 Arial。基础文字为 **8–9.2 pt**，数学上下标按数学排版规则缩小；加法运算符为 **12 pt**。模块轮廓线 **0.75 pt**，箭头 **0.85 pt**。
- 绿色表示数值域重建/补偿，紫色表示 AMS，灰色块明确标记被冻结的 inherited Transformer encoder。面板 (a) 中 “Log predictor + sigmoid” 为 Fig. 1 重建主干的汇总，不暗示整条 log branch 被冻结。固定的 log/inverse/clip 运算没有可学习参数。
- SVG 可用 Inkscape 编辑；16 个模块的框和文字已分组。主文字、变量和下标保留为文本；**共 10 个小数学符号（帽号、波浪号及少量运算符）采用 STIX 真实字形轮廓生成矢量路径**，避免未安装 STIX 的编辑器错误映射。这些小符号不能作为普通文字直接替换；不应将 SVG 描述为所有数学符号均可直接改字。编辑电脑应安装 Arial，避免主字体替换造成错位。复杂数学标签建议在 `draw_fig3.py` 中修改后重新导出；手工移动模块时需同时调整箭头，箭头不是自动跟随的连接器。
- 图注未烘焙到图中，由论文正文排版。SVG/PDF 为结构图母版和投稿文件；PNG 用于预览。

## 材料来源与事实依据

1. **论文：** `docs/ICSPS2026_LaTeX/ICSPS2026_paper.tex:65–87,124–162`，对应 bounded log/inverse、*Guidance Modulation and Intensity Compensation*、*Supervised Training and Masked Post-Adaptation* 小节。Fig. 3 标签为 `fig:representation_ams`。
2. **模型入口与配置：** `model_registry.py:330–357` 的 `build_model("ours", variant="full", numeric_domain="intensity_v1")`；`ablation_config.py:19–28` 的正式 Full 变体；`configs/icsps2026_frozen_v2.json` 的 `256×256` normalized intensity 输入及 AMS 设置。
3. **数值域路径：** `numeric_domain.py:94–109` 和 `transform_main.py:2141–2166`。`alpha=10`，sigmoid 预测归一化 log intensity，精确 inverse 后得到 preliminary intensity，再与 observation 拼接补偿并进行同域 residual addition，最后 clip 到 `[0,1]`。
4. **补偿器：** `transform_main.py:1522–1541`。通道拼接 `[Y,X_tilde]` 得到两个 channels；三层 `3×3` convolution 为 `2→32→32→1`，前两层后接 GELU；learnable `gamma_c` 初始化为 `0.1`。残差基底为 `X_tilde`；signed residual `R_c` 没有 sigmoid/tanh 范围约束。
5. **AMS：** `train_icsps2026_ams.py:104–118,167–192,244–264,301–314,443–488`。Bernoulli(0.2) 以 20% 概率将像素置零；训练 mask 的 seed 包含 epoch，验证每样本的固定 mask 跨 epoch 重用；仅 encoder 冻结。验证以完整 `masked L1 + TV` 选择 adapted epochs 1–8 中最低损失的 checkpoint，epoch 0 不作为 candidate；推理无 mask。
6. **真实强度入口：** `train_icsps2026_ams.py:92–98,266–267`、`prepare_icsps2026_real.py:127–196` 以及 `evaluate_icsps2026_real.py:340,475–485`。训练、验证、评价共享 QC manifest 的 dataset-level affine mapping。本图从已 normalized 的 intensity `Y` 开始，不绘入 per-image percentile stretching。

所有单通道图像（`Y`、`Y_m`、`Z_hat`、`X_tilde`、`R_c`、`X_hat`）的正式尺寸为 `B×1×256×256`；补偿器 concat 输入为 `B×2×256×256`。`R_c` 和 clip 前的相加结果不应被误标成 `[0,1]`。本图省略重复尺寸，以将单栏空间留给数值域和训练流程；完整中间尺寸见 `SOURCE_AUDIT.md` 和 `workflow_trace.json`。

## 必须保留的连接与已发现差异

**AMS 输入遮挡作用于整个模型。** 实际调用为 `prediction = model(Y * (1-M))`，因此面板 (b) 的 whole model 接收 `Y_m`，面板 (a) 内部通向 compensator 的 observation bypass 此时同样为 `Y_m`。原始 noisy `Y` 仅从 target 支路进入 masked loss，mask `M` 也单独进入 loss；不能增加一条未遮挡 `Y` 返回补偿器的连接。图中 `P=f_theta(Y_m)` 的 TV 作用于整个预测，不只作用于 mask 区域。

**论文 AMS 分母与实现有一处明确差异。** 当前论文第 149–151 行使用 `||M||_1 + epsilon`，而实际 `train_icsps2026_ams.py:110–111` 使用 `mask.sum().clamp_min(1.0)`，即 `max(||M||_1,1)`。非空 binary mask 时实现分母就是 masked pixel count；空 mask 使用 1。两者不能宣称严格相同。图中标注 `Masked L1(P,Y;M) + 1e-3 TV(P)`，其 Masked L1 应以本交付审计所列的执行代码定义为准；未擅改论文公式或训练事实。

**历史 helper 与名字不能替代实际入口。** `numeric_domain.normalize_real_intensity` 的 docstring 声称服务 AMS/evaluation，但该 helper 的逐图百分位映射不在当前正式 AMS/evaluation 调用链中；正式入口使用 QC fixed mapping。历史 `DualDomainFusion` 参数名 `log_branch_output` 也不代表本图直接混合 log/intensity，正式 wrapper 已先 inverse；历史 `NoiseEstimator` 对应正文 unsupervised Guidance，不能据此新增噪声真值监督。

灰色 encoder 与紫色 trainable reconstruction 在面板 (b) 内是参数分组，不是串行模块链。guidance 与 encoder 接收同一 log input，guidance 驱动 decoder gates；没有 `Encoder → Decoder → Guidance` 的计算顺序。图不主张 noisy target 无偏、noise independence 或新的 blind-spot network。

## 复现命令

以下命令从项目根目录 `D:\research\sar_transformer-main` 运行。绘图不依赖 PyTorch、checkpoint 或真实图像；结构核验依赖模型运行环境。无需执行任何训练脚本。

### 生成 SVG、PDF 与 PNG

```powershell
python -m pip install -r output/pdf/ICSPS2026_Fig3/requirements-drawing.txt
python output/pdf/ICSPS2026_Fig3/draw_fig3.py --font Arial --dpi 600
```

默认输出到脚本所在交付目录，可用 `--output-dir` 指定另一输出目录。当前绘图环境使用 Matplotlib **3.8.4**、fonttools **4.51.0**，Arial 字体文件为 `C:\Windows\Fonts\arial.ttf`。脚本要求指定主字体存在，避免静默替换；导出 SVG 后自动分组模块、处理文字定位，并将 10 个 STIX 小数学符号转为准确的矢量轮廓。变量与下标继续保留文本；PDF 保留嵌入的 Arial/STIX 字体。

### 重复结构和 AMS 流程核验（可选）

在已安装项目模型依赖的 Python 环境中运行：

```powershell
python output/pdf/ICSPS2026_Fig3/verify_workflow.py
```

如需建立核验环境：

```powershell
python -m pip install -r output/pdf/ICSPS2026_Fig3/requirements-structure.txt
python output/pdf/ICSPS2026_Fig3/verify_workflow.py --threads 4
```

本机可复用已有隔离补充依赖目录：

```powershell
python output/pdf/ICSPS2026_Fig3/verify_workflow.py --dependency-dir tmp/fig1_runtime --threads 4
```

`tmp/fig1_runtime` 是本机可选路径，不是交付所必需的目录；正常安装依赖后可省略。记录使用 Python **3.12.3**、PyTorch **2.9.1**、torchvision **0.24.1**、timm **1.0.22** 和 NumPy **1.26.4**。

核验使用随机初始化 Full 和合成输入，执行两次 CPU inference，分别检查完整输入和 masked 输入；直接执行从原 AMS 源码提取、未修改函数体的纯 mask/loss helper 与冻结循环，避免导入完整训练入口。它不读取 dataset/checkpoint，不创建 optimizer、不执行 backward/step，也不训练模型。冻结状态检查只作用于该次临时随机模型的参数标志。

### 独立 SVG 渲染与 PDF 排版检查（可选）

SVG 校验依赖 Node.js 和 sharp，本脚本使用的 sharp 版本为 **0.35.4**：

```powershell
npm install --prefix tmp/fig3_svg_runtime sharp@0.35.4
$env:SHARP_MODULE = (Resolve-Path 'tmp/fig3_svg_runtime/node_modules/sharp').Path
node output/pdf/ICSPS2026_Fig3/render_svg_preview.cjs
Remove-Item Env:SHARP_MODULE
```

导出检查及带图注 proof：

```powershell
python -m pip install -r output/pdf/ICSPS2026_Fig3/requirements-qa.txt
python output/pdf/ICSPS2026_Fig3/make_layout_proof.py
```

上述脚本仅创建/更新本图交付和临时依赖文件；图源变化时，应先重新核对论文和源码，再重绘。`figure_manifest.json` 和 `workflow_trace.json` 保留来源 hashes。

## 验证记录与使用边界

- 已实际运行正式模型/AMS helper 核验，**88 项检查全部通过**。完整输入及 masked 输入的 residual reconstruction 和最终 clip reconstruction 最大绝对误差均为 **0**；log/inverse round-trip 最大绝对误差为约 **1.19e-7**。完整证据见 `workflow_trace.json`。
- 已实际打开 `fig3_actual_size_150dpi.png` 作独立人工检查：模块、公式和箭头清晰，未见文字重叠、明显裁切或错误连接；训练图明确保留原始 noisy target 和 mask 的独立 loss 输入。自动文字边界检查无 overflow，见 `figure_manifest.json`。
- 最新 PDF/PNG 和独立 sharp SVG 预览均已复核；STIX 小符号轮廓化修复仅作用于显示，不改变数学计算或图示连接。独立 SVG 渲染、PDF 导出和带图注单栏 proof 的逐项检查结果见 `export_qa.json`；图像对应 `fig3_svg_render_check.png`、`fig3_layout_proof.pdf`。单图 proof 用于检查自然尺寸、图注与页面容纳，不应被当作已重编译整篇论文。
- 这些运行只核对结构、数值变换、mask/loss、冻结边界和导出质量；随机诊断张量不是论文恢复效果，不能据其数值推断训练性能或 best checkpoint。

## 英文图注与建议插入位置

替换当前 *Supervised Training and Masked Post-Adaptation* 小节末尾、*Experiments* 之前的 Fig. 3 占位图，保留 `fig:representation_ams` 标签。使用单栏 `figure` 和 `width=\columnwidth`，按自然高度插入；不要使用 `figure*`，也不要将宽高都强制设为原占位框。

现有 **1.25in（31.75 mm）** 是草稿占位高度，无法容纳本图所需流程和可读字号。真实图高约 **182.03 mm**，加图注后会影响浮动位置；未擅自改变正文、缩小全文字号或重编译整篇论文。完成替换后需在完整稿件中确认浮动位置与总页数。

`caption.txt` 为不含手工 Fig. 3 编号的英文图注：

> Representation-consistent output reconstruction and masked post-adaptation (AMS). (a) A bounded log prediction is exactly inverted before residual compensation and addition in the intensity domain. (b) During adaptation, the whole model receives the masked observation Y_m, including the compensator; the original observation supplies the noisy target. Only the Transformer encoder is frozen. Training masks are resampled, validation masks are fixed for checkpoint selection, and inference uses the complete unmasked observation.

`fig3_insert.tex` 可替换原 `figure` 环境。片段中的图像路径为 `../../output/pdf/ICSPS2026_Fig3/fig3_representation_ams.pdf`，按从 `docs/ICSPS2026_LaTeX` 编译设置；若编译工作目录不同，请对应调整路径。纯文本图注采用 ASCII `Y_m` 以便 proof 排版，LaTeX 片段使用正式数学下标。
