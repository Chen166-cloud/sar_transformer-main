# ICSPS 2026 — Fig. 2: Full-spectrum Fourier residual refinement

本交付完成英文初稿中的 Fig. 2，依据当前论文与正式模型源码绘制可编辑矢量结构图。图中展开 Fig. 1 的 FDR 节点，清楚呈现空间/频谱并行分支、实虚部通道混合、逆变换、融合及 identity residual；不重复全网结构或实验表格。无需额外照片、预测结果或 checkpoint，现有材料足以支撑本图，**无缺失素材**。

## 交付文件

| 文件 | 用途 |
|---|---|
| `fig2_fdr_block.svg` | 可编辑矢量母版；文字保持文本，10 个模块的框与文字已分组 |
| `fig2_fdr_block.pdf` | 论文排版用矢量 PDF |
| `fig2_fdr_block.png` | 600 dpi、2092×3433 px 高清预览 |
| `fig2_actual_size_150dpi.png` | 150 dpi 单栏尺寸预览 |
| `fig2_svg_render_check.png` | 使用独立 SVG 渲染器导出的校验预览 |
| `fig2_layout_proof.pdf` | 带英文图注的单栏尺寸排版检查页；不是整篇论文重编译结果 |
| `draw_fig2.py` | 绘图及 SVG 可编辑分组/数学字符定位代码 |
| `verify_fdr.py` / `fdr_forward_trace.json` | 正式模型只读 forward 结构核验代码与实际运行记录 |
| `render_svg_preview.cjs` | 独立 SVG 渲染检查代码 |
| `make_layout_proof.py` / `export_qa.json` | PDF/SVG 导出检查、排版 proof 代码与记录 |
| `figure_manifest.json` | 尺寸、字体、线宽、依赖版本、溢出检查与图源 hash |
| `SOURCE_AUDIT.md` | 论文/配置/源码路径、准确行号、尺寸与事实边界的详细审计 |
| `caption.txt` / `fig2_insert.tex` | 英文图注与可直接替换占位图的 LaTeX 片段 |
| `requirements-drawing.txt` / `requirements-qa.txt` / `requirements-structure.txt` | 分别用于绘图、导出排版检查、可选模型结构核验的依赖版本 |

## 尺寸、样式与编辑

- 按当前 `IEEEtran` conference 模板设为单栏：**88.5678705 mm × 145.3444444 mm**。模板依据为 `docs/ICSPS2026_LaTeX/IEEEtran.cls:1734–1735`，单栏宽 `21pc = 252 TeX pt`。
- 统一英文标注、Arial 字体；正文基础字号为 **8–9.2 pt**，数学上下标按排版规则缩小。模块轮廓线 **0.75 pt**，箭头 **0.85 pt**。输出加号为较大的运算符。
- 颜色延续 Fig. 1 中 FDR 使用的橙色。淡橙色填充区显示卷积/GELU 处理；白底橙边显示 Fourier 表示变换、复数重建或特征拼接；深灰用于正文、连线与 identity residual。
- SVG 文字可编辑，模块框和文字已分成 10 组。数学标签的定位字符仍为文本；复杂公式的整体改写建议修改 `draw_fig2.py` 后重新生成，以保持上下标布局。模块移动后需同时调整连接箭头。
- 图中以 `H×W×C` 标注形状，省略 batch；代码实际为 `B×C×H×W`。`W_f = floor(W/2)+1` 是 rFFT 最后一维的单侧存储宽度。
- SVG 为便于编辑保留 Arial 字体引用；编辑电脑须安装 Arial，避免字体替换造成错位。PDF 用于最终排版，PNG 仅供预览。

## 图源与结构事实

1. **论文：** `docs/ICSPS2026_LaTeX/ICSPS2026_paper.tex:96–121`，对应指定初稿 `output/pdf/ICSPS2026_英文论文_LaTeX无图初稿.pdf` 的 Fig. 2 和 *Full-Spectrum Fourier Residual Refinement* 小节。
2. **正式入口：** `model_registry.py:330–357` 的 `build_model("ours", variant="full", numeric_domain="intensity_v1")`；配置为 `configs/icsps2026_frozen_v2.json`，正式变体开关见 `ablation_config.py:19–28`。
3. **直接图源：** `transform_main.py:1406–1458` 的 `FFTRefineBlock`。空间分支为 DWConv `3×3 → 1×1 conv → GELU`；频谱分支为 `rFFT2 → [Re,Im] → 1×1 conv → GELU → 1×1 conv → split/complex reconstruction → irFFT2`；两支路通道拼接后经 `1×1 conv → GELU → 3×3 conv`，最后加回输入。
4. **实例位置：** 瓶颈内 1 个、解码器内 5 个，共 6 个 FDR 实例。不同实例参数独立；“shared over coordinates”仅表示同一个实例内的 `1×1` convolution 参数在频率坐标之间共享。
5. **事实边界：** “Full-spectrum”没有按频带筛选或只增强高频；rFFT 存储非冗余的单侧表示。频谱支路的学习操作为实值通道/实虚部混合，不是相邻频率卷积、复值卷积、幅相处理或高低频分解。最终融合后的 `3×3` convolution 仍在空间邻域上运算。
6. **瓶颈边界：** Fig. 2 只展开 `FFTRefineBlock`。`BottleneckRefine` 的外层 local branch、projection 与初始化为 `0.1` 的 `gamma` 不在该块内部，故不出现在本图。完整依据见 `SOURCE_AUDIT.md`。

当前正文与代码未发现影响 Fig. 2 的实质连接、运算顺序或尺寸矛盾。此图不绘制任何实验统计或恢复效果；没有修改论文、模型、实验数据、配置，也没有重新训练。

## 复现命令

以下命令从项目根目录 `D:\research\sar_transformer-main` 运行。绘图无需 PyTorch 或 checkpoint；核验正式 forward 时才需要项目模型依赖。

### 生成 SVG、PDF 与 PNG

在已安装 Arial 的 Python 环境中安装本次绘图使用的版本，并运行：

```powershell
python -m pip install -r output/pdf/ICSPS2026_Fig2/requirements-drawing.txt
python output/pdf/ICSPS2026_Fig2/draw_fig2.py
```

可选参数：

```powershell
python output/pdf/ICSPS2026_Fig2/draw_fig2.py --font Arial --dpi 600
```

本机实际绘图 Python 为 `D:\develop\Anaconda\python.exe`，Matplotlib **3.8.4**；字体文件为 `C:\Windows\Fonts\arial.ttf`。脚本在字体不存在时会报错，避免静默替换。SVG 导出后会自动进行模块分组，并将数学公式中多位置 `<tspan>` 展开为单字定位，避免部分 SVG 渲染器忽略多位置列表而导致数学字符重叠。

### 重复模型结构核验（可选）

使用已装好项目依赖的环境运行：

```powershell
python output/pdf/ICSPS2026_Fig2/verify_fdr.py
```

本次核验记录使用 Python **3.12.3**、PyTorch **2.9.1**、torchvision **0.24.1**、timm **1.0.22**、NumPy **1.26.4**。如果需要建立隔离环境，可安装以下依赖；已有可用模型环境则不必重复安装：

```powershell
python -m pip install -r output/pdf/ICSPS2026_Fig2/requirements-structure.txt
```

本机复用了 Fig. 1 核验时的临时补充依赖目录，等价命令为：

```powershell
python output/pdf/ICSPS2026_Fig2/verify_fdr.py --dependency-dir tmp/fig1_runtime
```

`tmp/fig1_runtime` 是本机可选临时路径，非交付依赖；把依赖正常安装到 Python 环境后即可省略。核验固定随机种子，执行一次正式 `ours:full` 模型 CPU inference，记录 6 个实际 FDR 路径，并独立重构各块进行比较；另有一个小型奇数宽度输入核验 `irfft2` 显式输出尺寸。**随机权重和合成输入仅用于结构核验，不是实验结果，不加载 checkpoint，不进行训练。**

### 独立 SVG 渲染与 PDF 排版检查（可选）

独立 SVG 检查使用 Node.js 与 `sharp`；本次版本为 sharp **0.35.4**，其 librsvg 为 **2.62.91**：

```powershell
npm install --prefix tmp/fig2_svg_runtime sharp@0.35.4
$env:SHARP_MODULE = (Resolve-Path 'tmp/fig2_svg_runtime/node_modules/sharp').Path
node output/pdf/ICSPS2026_Fig2/render_svg_preview.cjs
Remove-Item Env:SHARP_MODULE
```

PDF/SVG 导出与单栏排版 proof 检查使用 `pypdf` 和 `reportlab`：

```powershell
python -m pip install -r output/pdf/ICSPS2026_Fig2/requirements-qa.txt
python output/pdf/ICSPS2026_Fig2/make_layout_proof.py
```

这些操作仅写本图交付或临时依赖文件。`draw_fig2.py` 会根据当前图源文件重新记录 SHA-256；若未来模型或论文改变，应先复核审计再重新绘图。

## 已完成的核验

- 正式模型 6 个 FDR 实例及其内部中间张量与图中连接一致。**155 项检查全部通过**；实际 forward 与独立重构比较的最大绝对误差为 **0**，记录见 `fdr_forward_trace.json`。
- 已打开实际导出的图片检查文字、箭头、留白、遮挡和裁切。模块文字边界自动检查无溢出，见 `figure_manifest.json`。
- 曾发现独立 SVG 渲染时数学 `<tspan>` 的多位置字符重叠，现已通过绘图代码自动转换为单字定位修正，并使用独立渲染图复核。
- 同时交付单栏宽度的 PDF 排版 proof 与导出检查记录。**未重编译整篇论文**，因此不把单图 proof 当作最终论文的分页验证。

## 英文图注与插入位置

建议保留 `fig:fdr` 标签，将本图放在 *Full-Spectrum Fourier Residual Refinement* 小节解释之后、*Guidance Modulation and Intensity Compensation* 之前，替换当前 Fig. 2 占位图。使用单栏 **`figure`**，不要改为 `figure*`。现有 `1.15in` 是占位框高度，无法容纳清晰的完整模块；插入真实图时仅指定 `width=\columnwidth`，保持自然高度。

英文图注见 `caption.txt`，不含手工 Fig. 2 编号，编号由 LaTeX 自动生成：

> Full-spectrum Fourier residual refinement (FDR). A depthwise-pointwise spatial branch complements a spectral branch. The one-sided rFFT representation retains all nonredundant frequencies. Shared real-valued 1x1 convolutions mix channels and real-imaginary components independently at each frequency coordinate, without convolution across neighboring frequencies. After inverse rFFT, spatial and spectral features are concatenated, projected, and added to the input.

纯文本图注用 ASCII 连字符和 `1x1`，避免 proof 的基础字体替换；LaTeX 片段使用正式的破折号和 `\(1\times1\)` 数学排版。`fig2_insert.tex` 可直接替换原 `figure` 环境。图像相对路径按从 `docs/ICSPS2026_LaTeX` 编译设置；如果编译工作目录不同，请对应调整路径。图注随正文排版，而不是烘焙到图像内。由于图自然高度约 **145.34 mm**，替换后需在完整稿件中复核浮动位置与总页数。
