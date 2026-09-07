> 2026-09-07 紧凑排版更新：当前 `draw_fig1.py`、SVG、PDF、PNG 和 `figure_manifest.json` 已按本轮要求更新。下面原交付说明及旧 layout proof/export QA 作为历史记录保留，其旧尺寸与布局描述不再适用于当前图。当前论文、实际尺寸、编译及验收结果见 [本轮报告](../ICSPS2026_LayoutRefinement/REPORT.md)。

# Fig. 1 交付说明

本次仅完成指定初稿的 Fig. 1。已阅读原 PDF 第 2 页、对应 LaTeX 方法段、正式配置与模型 forward，完成源码核验、矢量绘图、单次结构前向验证、PDF 渲染与实际版心尺寸检查。未更改论文正文、模型、配置、实验数据或 checkpoint，未重新训练。

## 配图方案与插入位置

沿用初稿的双栏 `figure*` 和 `fig:overview` 标签，在 Introduction 贡献说明后替换现有占位图；IEEE 双栏浮动规则可能将其放到下一页页首。

- **(a) Representation-consistent reconstruction**：用五级 encoder 和五级 decoder 概括主路径，明确四条同尺度 skip、瓶颈及五个 decoder 的六处 FDR、共享 guidance、原始强度旁路与 inverse 后补偿。公共 decoder 运算顺序只展开一次，避免展示完整计算图。基础算子灰色；新增或更改的运算按 Fourier、skip fusion、guidance、representation 配色。
- **(b) Masked post-adaptation**：独立表示仅训练时的 20% mask、encoder freezing 和其余重建模块更新。推理使用不加 mask 的完整输入。

本图解释方法连接和归属，不重复实验表格。FDR 内部的 FFT / real–imaginary mixing 细节留给初稿的 Fig. 2；本次未制作 Fig. 2 或其他图。

## 可直接使用的文件

| 文件 | 用途 |
|---|---|
| `fig1_overall_architecture.pdf` | 正式排版用，全矢量，字体已嵌入 |
| `fig1_overall_architecture.svg` | 可编辑矢量，文字保留为文本、框与箭头保留为路径 |
| `fig1_overall_architecture.png` | 高清预览，4283 × 3600 px，600 dpi |
| `fig1_actual_size_150dpi.png` | 依据相同物理尺寸输出的较小预览；屏幕显示缩放受软件影响 |
| `fig1_layout_proof.pdf` | Letter 纸张中的实际双栏宽度排版检查页，含英文图注；不作为论文图文件插入 |
| `fig1_insert.tex` | 替换原 Fig. 1 环境的 LaTeX 片段，含正式英文图注 |
| `caption.txt` | 英文图注纯文本 |
| `draw_fig1.py` | 绘图源码，负责全部 SVG / PDF / PNG |
| `make_layout_proof.py` | 矢量/字体检查与实际尺寸排版检查页源码 |
| `verify_forward.py` / `forward_trace.json` | 可复现结构核验与实际运行记录 |
| `SOURCE_AUDIT.md` | 模型入口、连接、特征尺寸、灰彩归属与具体源码行号 |
| `figure_manifest.json` / `export_qa.json` | 物理尺寸、字体、依赖版本、源文件 SHA-256 与导出检查 |

## 模板尺寸与编辑

本地 `docs/ICSPS2026_LaTeX/IEEEtran.cls:1735` 定义 `textwidth=43pc`。本图按 **181.353 mm 宽、152.400 mm 高** 输出，插入使用 `width=\textwidth`。PDF MediaBox 为 514.072 × 432 PDF points，不使用会改变最终物理宽度的自动紧裁切。

英文基础字体为 Arial；正文标注基础字号 8–10 pt，数学上下标按公式缩放，数学符号使用嵌入的字形。模块边框 0.75 pt、箭头 0.85 pt。颜色只是归属提示，同时保留名称、图例和连接，避免仅依赖颜色读图。

该图含完整多尺度连接，**不能压入原来 0.92 inch 高的占位框**。替换时删除占位框，保留自然高度，不指定 `height`，不压到单栏。图和图注总高约 168 mm；最终整篇论文需要正常重排。本次交付的是图片及独立尺寸检查页，没有声称已重新编译原论文或核定最终总页数。

SVG 中有 80 个可编辑文本元素，没有内嵌位图。可用矢量编辑器修改。编辑机器需要 Arial；若没有该字体，重新绘图时可以显式选择已安装的 Liberation Sans，不能在投稿前任由未知字体替换。PDF 中字体已嵌入，排版优先使用 PDF。

## 复现命令

以下命令均从项目根目录 `D:\research\sar_transformer-main` 执行。图不依赖实验图片、模型权重或网络推理；仅绘图只需要 Matplotlib 和所选字体。

```powershell
python -m pip install -r output/pdf/ICSPS2026_Fig1/requirements-drawing.txt
python output/pdf/ICSPS2026_Fig1/draw_fig1.py
```

可选：生成排版检查页并验证导出。

```powershell
python -m pip install -r output/pdf/ICSPS2026_Fig1/requirements-qa.txt
python output/pdf/ICSPS2026_Fig1/make_layout_proof.py
pdftoppm -png -r 150 -singlefile output/pdf/ICSPS2026_Fig1/fig1_overall_architecture.pdf tmp/pdfs/fig1_qa/figure_pdf_final_150dpi
```

可选：在项目模型环境中重复正式 Full 的单次 CPU forward 核验。

```powershell
python -m pip install -r output/pdf/ICSPS2026_Fig1/requirements-structure.txt
python output/pdf/ICSPS2026_Fig1/verify_forward.py
```

本机实际使用 `D:\develop\Anaconda\python.exe` 绘图，Matplotlib 3.8.4；结构验证使用同一个 Python，以及 `tmp/fig1_runtime` 中隔离安装的补充依赖：

```powershell
python output/pdf/ICSPS2026_Fig1/verify_forward.py --dependency-dir tmp/fig1_runtime
```

本机 PDF 检查页脚本实际使用 Codex 已提供的 Python（pypdf 6.10.0 / reportlab 4.4.9）：

```powershell
& 'C:\Users\Chen\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' output/pdf/ICSPS2026_Fig1/make_layout_proof.py
```

其他系统可省略本机绝对 Python 路径，在安装对应依赖后运行同名脚本。绘图支持 `--font "Liberation Sans"`、`--dpi 600` 和 `--output-dir`。保留项目目录关系即可复现源文件指纹记录。

## 依据与一致性结论

使用 `build_model("ours", variant="full", numeric_domain="intensity_v1")`，不采用历史 legacy/tanh 分支。当前英文正文与正式代码未发现影响 Fig. 1 的主要拓扑或尺寸矛盾。

具体区别已记录于 `SOURCE_AUDIT.md`：历史 `DualDomainFusion` 实际在 inverse-log 后进行强度补偿；`NoiseEstimator` 在图文中称 latent guidance generator；`FusionBlock` 定义在基础模块文件不等于 baseline 实际使用；Sigmoid 与 baseline 的 Tanh 不同，必须标色；`16→8` projection 不扩大空间尺寸。

“Inherited”归属通过本地正式 TransSARV2 baseline 与 proposed model 逐项核对。没有把外部上游仓库的版本逐字比对描述为本次已完成事项。具体文件、行号和 SHA-256 见审计文档。

## 已完成检查

- 正式 Full 模型以确定性合成输入运行一次 CPU forward。30 项断言通过；六个 FDR 各执行一次；encoder、skip concat、decoder、guidance、head、compensator 尺寸均记录。输入到 guidance 的是 log 表示，补偿器收到原始强度及 inverse 后估计。
- 该结构核验使用随机初始化权重，没有加载实验 checkpoint，输出值不作为复现实验指标或去斑效果。
- 绘图脚本已实际执行；模块文字边界检查无溢出。
- PDF 无位图，SVG 无位图且保留文字；PDF 字体均嵌入。
- 用 Poppler 从最终 PDF 渲染 150 dpi 预览，并打开检查；另打开 Letter 纸张中的实际宽度排版检查页。修复过第四条 skip 被瓶颈框遮住、旁路文字压线、瓶颈计数与 skip 标注过近的问题；最终检查未见文字遮挡、错误连接或裁切。
- 在 181.353 mm 宽度下检查整体和图注。该检查针对生成图与尺寸检查页，不等于重新编译完整论文。

Fig. 1 所需材料齐全，无需补充文件。
