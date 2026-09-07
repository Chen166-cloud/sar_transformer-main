# ICSPS 2026 单栏 2 行×4 列最终版

按用户最新更正，Fig. 4 和 Fig. 5 采用单栏 **2 行×4 列**。每张独立原图分别等比缩小，不拉伸、不重采样。

## 当前文件

- 论文源码：`ICSPS2026_paper.tex`。
- 当前 PDF：`ICSPS2026_paper.pdf`，已经使用原 `build.ps1` / Tectonic 0.17.0 实际编译并成功覆盖。
- 上一轮副本：`ICSPS2026_paper_layout_2x4.pdf` 被查看器占用，仍保留 Table I 位于页顶的布局。当前页底版请打开原名 `ICSPS2026_paper.pdf`。
- `ICSPS2026_paper_layout_4x2.pdf` 和 `LAYOUT_4X2_REPORT.md` 是上一轮历史版本，不代表当前布局。

## 布局与内容

- 单图宽 `0.24\columnwidth`，约 21.257×21.257 mm；原始图片均为 256×256。
- 相邻列间距 `0.01\columnwidth`；总图宽 `0.99\columnwidth`，约 87.683 mm，小于 88.568 mm 栏宽。
- 标签统一为 8 pt、两行显示，第一行为子图编号，第二行为完整方法名；两行图片之间仅增加 3 pt 间距。
- Fig. 4 顺序：第一行 Clean / Noisy / LEE / SAR-BM3D；第二行 SAR-CAM / Trans-SAR / Ours / Ours ratio。
- Fig. 5 顺序：第一行 Noisy / SAR-BM3D / SAR2SAR / SDUDNet；第二行 Trans-SAR / Ours-base / Ours+AMS / Ours+AMS ratio。
- Fig. 4 位于第 5 页左栏，与 Synthetic Comparison 同页。
- Fig. 5 位于第 6 页左栏，与 Real-SAR Adaptation 的后续讨论相邻；REFERENCES 从同页右栏开始。Fig. 5 连同 caption 完整位于参考文献之前。
- 原始 16 张图片文件没有修改，最终 PDF 的内嵌像素逐一与源图匹配，红框和两张 ratio 图保留。

Fig. 1–3 沿用前轮已核验设置：Fig. 1 为第 2 页双栏原尺寸；Fig. 2 为第 3 页单栏 0.94 栏宽及安全上下裁白；Fig. 3 为第 4 页单栏 0.98 栏宽及安全上下裁白。
Table I 按后续要求移至第 4 页底部，保持横跨双栏；使用 `table*` 的 `[!b]` 参数，并新增 `stfloats` 支持双栏页底浮动。Table I–III 的数据、表题、标签、注释、宽度与字号保持不变；Table II/III 位置也未改动。

## 编译和验证

- 原始版本 7 页，上一轮 4×2 版本 7 页，本次最终版本仍为 7 页。
- 最终稳定日志：无编译错误、无 Undefined references、无 Missing citation、无 Overfull；存在 1 条 Underfull vbox（第 4 页），无 Underfull hbox。
- 为后续 Table I 页底跨双栏要求新增 `stfloats` package（编译环境提供 v3.3）；未启用基线伸缩设置。没有修改 `.sty` 或 `IEEEtran.cls`，没有使用 `[H]`、浮动屏障、负间距或全局文字压缩。
- 采用 IEEEtran 自带的 `\IEEEtriggeratref{16}` 平衡末页参考文献两栏，仅改变换栏位置，文献内容和编号不变。
- 正文、所有公式及编号、section/subsection、ref/eqref/cite/label、三表数据和 21 条参考文献均通过保留核验。
- 已查看编译后的页面渲染，重点复核第 4–7 页的图像、标签、红框、栏边及参考文献位置。

审计和最终日志：`../../output/pdf/ICSPS2026_Layout2x4/`。
本轮修改前备份和页面渲染：`../../tmp/pdfs/layout_2x4/`。

## 当前实际 LaTeX 代码

共用宏：

```latex
\newcommand{\comparisonpanel}[3]{%
  \begin{minipage}[t]{0.24\columnwidth}
    \centering
    \includegraphics[width=\linewidth]{#1}\par
    \vspace{2pt}%
    {\footnotesize\strut #2\par\strut #3\par}
  \end{minipage}%
}
```

Fig. 4：

```latex
\begin{figure}[!t]
  \centering
  \begin{tabular}{@{}c@{\hspace{0.01\columnwidth}}c@{\hspace{0.01\columnwidth}}c@{\hspace{0.01\columnwidth}}c@{}}
    \comparisonpanel{figures/fig4/00_Clean.png}{(a)}{Clean} &
    \comparisonpanel{figures/fig4/01_Noisy.png}{(b)}{Noisy} &
    \comparisonpanel{figures/fig4/LEE-MMSE.png}{(c)}{LEE} &
    \comparisonpanel{figures/fig4/SAR-BM3D.png}{(d)}{SAR-BM3D} \\[3pt]
    \comparisonpanel{figures/fig4/SAR-CAM.png}{(e)}{SAR-CAM} &
    \comparisonpanel{figures/fig4/SAR-Trans.png}{(f)}{Trans-SAR} &
    \comparisonpanel{figures/fig4/Ours-base.png}{(g)}{Ours} &
    \comparisonpanel{figures/fig4/Ours-base_ratio.png}{(h)}{Ours ratio}
  \end{tabular}
  \caption{Qualitative comparison on one UCMerced buildings scene; LEE denotes Lee-MMSE. All panels retain the same field of view and original grayscale renderings. The Ours ratio in (h) is interpreted separately from the restored images.}
  \label{fig:synthetic_qual}
\end{figure}
```

Fig. 5：

```latex
\begin{figure}[!t]
  \centering
  \begin{tabular}{@{}c@{\hspace{0.01\columnwidth}}c@{\hspace{0.01\columnwidth}}c@{\hspace{0.01\columnwidth}}c@{}}
    \comparisonpanel{figures/fig5/noisy.png}{(a)}{Noisy} &
    \comparisonpanel{figures/fig5/SAR-BM3D.png}{(b)}{SAR-BM3D} &
    \comparisonpanel{figures/fig5/SAR2SAR.png}{(c)}{SAR2SAR} &
    \comparisonpanel{figures/fig5/SDUDNet.png}{(d)}{SDUDNet} \\[3pt]
    \comparisonpanel{figures/fig5/sar-trans.png}{(e)}{Trans-SAR} &
    \comparisonpanel{figures/fig5/Ours-base.png}{(f)}{Ours-base} &
    \comparisonpanel{figures/fig5/Ours-ams.png}{(g)}{Ours+AMS} &
    \comparisonpanel{figures/fig5/Ours-ams-ratio.png}{(h)}{Ours+AMS ratio}
  \end{tabular}
  \caption{Qualitative comparison on one real-SAR scene. Ours-base is the pre-adaptation model. Original renderings, including the red box for visual inspection in (a), are retained. The Ours+AMS ratio in (h) is interpreted separately from restored images. No clean reference is shown.}
  \label{fig:real_qual}
\end{figure}
```
