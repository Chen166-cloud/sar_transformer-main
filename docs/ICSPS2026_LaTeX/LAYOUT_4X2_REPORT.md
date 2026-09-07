> 历史版本：用户随后将要求更正为单栏 2 行×4 列。当前源码和原名 PDF 已更新；请参阅 `LAYOUT_2X4_REPORT.md`。

# ICSPS 2026 单栏 4×2 版面优化报告

最终 PDF：`ICSPS2026_paper.pdf`。
原名文件的占用已解除，已使用项目原有 `build.ps1` 再次编译并成功覆盖。`ICSPS2026_paper_layout_4x2.pdf` 保留相同布局作为额外副本。Fig. 4/5 各自使用 8 张独立原图，每张等比显示为约 42.51×42.51 mm，分别排成单栏 4 行×2 列。

## 修改范围

- 修改论文源文件：`ICSPS2026_paper.tex`。
- 新增 `figures/fig4/` 与 `figures/fig5/`，各含 8 张从原始素材目录逐字节复制的 PNG；没有重新生成、重采样或修改图像。
- 更新 `README.md`，新增本报告。
- 没有修改 `.sty`、`IEEEtran.cls`、任何原有 PDF/SVG 图或绘图脚本。
- 没有新增 LaTeX package，没有使用 `[H]`、`placeins`、`FloatBarrier` 或负间距。
- 正文字号、行距、纸张、页边距和 section/subsection 结构不变。

## 逐图结果

| 图 | 最终布局 | 调整 | PDF 页 |
|---|---|---|---|
| Fig. 1 | 双栏 `figure*`，`width=\textwidth` | 原图、尺寸和 caption 全部保留；原图边缘已经紧凑 | 2 |
| Fig. 2 | 单栏，`0.94\columnwidth` | 矢量 PDF 仅裁白：下 5bp、上 6bp；含图像外框高度约减少 8.5%，完整保留运算和说明 | 3 |
| Fig. 3 | 单栏，`0.98\columnwidth` | 矢量 PDF 仅裁白：下 2bp、上 5bp；高度约减少 3.8%，保留两条流程与所有回路 | 4 |
| Fig. 4 | 单栏 4 行×2 列 | 8 张独立原图；每张宽 0.48 栏宽，列间距 0.02 栏宽，图组总宽 0.98 栏宽；8pt 标签 | 5 右栏 |
| Fig. 5 | 单栏 4 行×2 列 | 同 Fig. 4；保留红框、灰度、ratio，caption 保留 pre-adaptation / visual inspection / no clean reference | 6 右栏 |

Fig. 4 源码紧跟 Synthetic Comparison 中第一次讨论该图的段落。
Fig. 5 源码紧跟 Real-SAR Adaptation 中 “Fig. 5 illustrates...” 的整段文字；最终与 IV-E 同第 6 页。
REFERENCES 从第 7 页开始，因此 Fig. 5 完整位于参考文献之前。

原始 Fig. 2/3 不是 TikZ，且现有矢量图内部已经较紧凑。本次不重绘、不移动或合并运算框，仅使用安全的外围裁白和保守比例调整。

## 浮动与表格

第一轮仅转换图组并移到首次讨论段落后，Fig. 4/5 仍被第 5 页顶部的跨栏 Table I 推迟，Fig. 5 仍落在 References 开始之后。
第二轮提前声明 Table I，使其正常浮动到第 4 页页顶，释放第 5 页的完整单栏高度。之后 Fig. 4/5 自然落到第 5/6 页，无需屏障。

Table I 使用 `tabular*`，表体和 note 同为 `0.86\textwidth`；Table III 表体和 note 同为 `0.82\columnwidth`；Table II 保留原样。
三表数据、标题、标签、字号和行高不变，检查均未越过栏边。

## 编译与验证

- 修改前：7 页；修改后：7 页。
- 使用项目现有 Tectonic 0.17.0，完成稳定引用所需的自动多轮编译。
- 最终稳定日志：无编译错误、无 Undefined references、无 Missing citation、无 Overfull hbox/vbox。
- 有 3 条 Underfull vbox 警告，位于第 4、5、6 页的浮动排版；最终日志无 Underfull hbox。
- 逐页渲染检查了全部 7 页，特别复查第 4–7 页；图组对齐，所有 16 张子图存在，没有明显大面积异常空白。
- 公式 1–8、图号 1–5、表号 I–III 和参考文献编号 1–21 保持正确。
- 保留 7 页以维持原有正文及图内文字的可读性，没有通过模板或全局文字压缩追求 6 页。

实际编译命令（仓库根目录）：

```powershell
& tmp/latex_runtime/tectonic.exe -X compile docs/ICSPS2026_LaTeX/ICSPS2026_paper.tex --outdir tmp/pdfs/layout_4x2/after --keep-logs --keep-intermediates
```

首次运行 `build.ps1` 时，编译成功，但最后一步覆盖被占用的旧 PDF 失败。随后用同一编译器完成第二轮编译，生成 `ICSPS2026_paper_layout_4x2.pdf`。用户再次确认排版要求后，已重新运行原 `build.ps1`；文件占用已解除，原名 `ICSPS2026_paper.pdf` 成功更新并复核。

验证材料：`../../output/pdf/ICSPS2026_Layout4x2/`（修改前快照、最终日志、辅助文件、审计结果）。
页面渲染：`../../tmp/pdfs/layout_4x2/after/page-1.png` 至 `page-7.png`。

## Fig. 4 / Fig. 5 实际排版代码

两个 figure 共用的宏：

```latex
\newcommand{\comparisonpanel}[2]{%
  \begin{minipage}[t]{0.48\columnwidth}
    \centering
    \includegraphics[width=\linewidth]{#1}\par
    \vspace{2pt}%
    {\footnotesize\strut #2\par}
  \end{minipage}%
}
```

Fig. 4：

```latex
\begin{figure}[!t]
  \centering
  \begin{tabular}{@{}c@{\hspace{0.02\columnwidth}}c@{}}
    \comparisonpanel{figures/fig4/00_Clean.png}{(a) Clean} &
    \comparisonpanel{figures/fig4/01_Noisy.png}{(b) Noisy} \\[2pt]
    \comparisonpanel{figures/fig4/LEE-MMSE.png}{(c) LEE} &
    \comparisonpanel{figures/fig4/SAR-BM3D.png}{(d) SAR-BM3D} \\[2pt]
    \comparisonpanel{figures/fig4/SAR-CAM.png}{(e) SAR-CAM} &
    \comparisonpanel{figures/fig4/SAR-Trans.png}{(f) Trans-SAR} \\[2pt]
    \comparisonpanel{figures/fig4/Ours-base.png}{(g) Ours} &
    \comparisonpanel{figures/fig4/Ours-base_ratio.png}{(h) Ours ratio}
  \end{tabular}
  \caption{Qualitative comparison on one UCMerced buildings scene; LEE denotes Lee-MMSE. All panels retain the same field of view and original grayscale renderings. The Ours ratio in (h) is interpreted separately from the restored images.}
  \label{fig:synthetic_qual}
\end{figure}
```

Fig. 5：

```latex
\begin{figure}[!t]
  \centering
  \begin{tabular}{@{}c@{\hspace{0.02\columnwidth}}c@{}}
    \comparisonpanel{figures/fig5/noisy.png}{(a) Noisy} &
    \comparisonpanel{figures/fig5/SAR-BM3D.png}{(b) SAR-BM3D} \\[2pt]
    \comparisonpanel{figures/fig5/SAR2SAR.png}{(c) SAR2SAR} &
    \comparisonpanel{figures/fig5/SDUDNet.png}{(d) SDUDNet} \\[2pt]
    \comparisonpanel{figures/fig5/sar-trans.png}{(e) Trans-SAR} &
    \comparisonpanel{figures/fig5/Ours-base.png}{(f) Ours-base} \\[2pt]
    \comparisonpanel{figures/fig5/Ours-ams.png}{(g) Ours+AMS} &
    \comparisonpanel{figures/fig5/Ours-ams-ratio.png}{(h) Ours+AMS ratio}
  \end{tabular}
  \caption{Qualitative comparison on one real-SAR scene. Ours-base is the pre-adaptation model. Original renderings, including the red box for visual inspection in (a), are retained. The Ours+AMS ratio in (h) is interpreted separately from restored images. No clean reference is shown.}
  \label{fig:real_qual}
\end{figure}
```
