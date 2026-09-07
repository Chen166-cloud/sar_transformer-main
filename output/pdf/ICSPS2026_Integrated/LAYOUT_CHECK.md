# 图1-3整稿集成与排版检查

## 交付与范围

- 论文入口：`docs/ICSPS2026_LaTeX/ICSPS2026_paper.tex`。
- 编译成稿：同目录 `ICSPS2026_paper.pdf`，全文 **8页**。
- 图1-3已替换占位框，正式英文图注和原有交叉引用标签已保留。
- 五张论文图片都位于该目录的 `figures/`，上传整个目录即可取得完整图像依赖。
- 本次接续已有图4、图5和对应文字。未改图4/5、表格数值、公式、研究结论、模型或实验数据。

## 采用的版式

沿用原版 `IEEEtran.cls` conference 模式：Letter、10 pt正文、双栏。
图注由 IEEEtran 统一排版；未引入 caption/subcaption/geometry 等样式覆盖，
也未缩小正文、压缩页边距或把图片拉伸到占位框高度。

| 图片 | 页码 | 版式 | 宽×高（mm，图像本身） |
|---|---:|---|---|
| 图1：总体架构 | 2 | 双栏通栏 | 181.353×152.400 |
| 图2：FDR | 3 | 单栏 | 88.568×145.344 |
| 图3：重建与AMS | 4 | 单栏 | 88.568×182.033 |
| 图4：模拟噪声对比 | 6 | 双栏通栏 | 使用已有图原尺寸 |
| 图5：真实SAR对比 | 7 | 双栏通栏 | 使用已有图原尺寸 |

图1-3均直接使用交付的矢量PDF，五张嵌入文件与各自交付原件的SHA-256相同。
图内Arial和正文Times分别承担图示标注与论文阅读，两者没有相互覆盖。
图1-3的基础字号仍不小于8 pt，数学上下标按原图比例保留。

## 已处理的问题

1. 原占位高度不足以容纳最终图片：改用 `\textwidth` / `\columnwidth`，保持自然比例。
2. Tectonic使用XeTeX内核时，IEEEtran在T1编码载入前初始化Times，产生TU字体回退警告。
   将 `\RequirePackage[T1]{fontenc}` 放到文档类之前，沿用模板的Times字族，消除字体回退警告。
3. 旧参考文献换栏点只适用于此前版式。改为第15条开始右栏，平衡第8页两栏。
4. 清除已无调用的占位图宏，并同步README、编译命令和图路径，避免上传后缺图。
5. 引言首段只增加英文单词的可选断词位置，以改善两条明显松散的行；不改变词义或字号。

## 检查记录

编译器：Tectonic 0.17.0，自动重复排版直到交叉引用收敛。
结构和导出检查由 `check_paper.py` 记录到 `paper_qa.json`，最终编译日志随本目录保留。
页面渲染在 `page_previews/`，使用Poppler按120 dpi生成，供检查整页布局。

检查包括：五图文件一致性、图表和公式标签、字体嵌入、文字是否超出页面、
缺失字符/引用/图片、浮动体过大、Overfull、正文保留和末页栏高差。
正文保留检查以本次开始前的稿件副本为基准，排除三幅替换图及必要排版指令后比较。

最终日志仅保留一条 `Underfull hbox (badness 1430)`，位于第6页真实SAR对比段。
实际检查该行只有轻微字距伸展，没有溢出或遮挡；未通过提高日志阈值隐藏提示。
没有Overfull、缺字、字体回退、未定义引用或过大浮动体警告。

所有实际绘制正文和图中文字的字体均已嵌入。既有图4、图5各包含一处ReportLab
默认声明的Helvetica资源；它们没有绘制任何文字，因此未更改原图来清理该声明。
`paper_qa.json` 同时保留完整字体资源清单和实际使用状态，避免将未使用资源误报为可见文字缺字体。

已打开检查全部8页：图1-3清晰、箭头/数学符号完整、图注不重叠，图4/5与表格正常；
各页没有超界文字、裁切或空白页。图2和图3位于对应方法内容同页，未延迟到独立图页。

本次保留完整正文和既有图片后为8页。会议是否允许该页数需另按投稿类别确认；
本任务未删减研究内容以压缩页数。此前核对发现的AMS分母公式与代码写法差异
仍保留在图3的 `SOURCE_AUDIT.md` 中，本次未更改该公式。

## 复现

从项目根目录运行：

```powershell
& 'docs/ICSPS2026_LaTeX/build.ps1'
python -m pip install -r output/pdf/ICSPS2026_Integrated/requirements-qa.txt
python output/pdf/ICSPS2026_Integrated/check_paper.py
pdftoppm -png -r 120 docs/ICSPS2026_LaTeX/ICSPS2026_paper.pdf output/pdf/ICSPS2026_Integrated/page_previews/page
```

编译脚本会寻找PATH中的Tectonic或项目已有的 `tmp/latex_runtime/tectonic.exe`；
也可传入 `-TectonicPath` 指定其位置。图像无需重绘，不依赖模型环境或checkpoint。
Overleaf主文件设为 `ICSPS2026_paper.tex`，采用pdfLaTeX；本机实际核验的编译器为上述Tectonic版本。
