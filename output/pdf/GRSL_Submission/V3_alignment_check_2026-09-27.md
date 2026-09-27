# GRSL LaTeX 与最终 V3 Word 对齐核查

2026-09-28 后续格式复核已恢复 LaTeX 参考文献和表注的 8 pt 字号，并同步精简 Word/LaTeX 正文以保持 5 页；同时修正标题斜体和参考文献格式。最新核验见 [IEEE GRSL 格式复核](GRSL_format_check_2026-09-28.md)。下文保留此前对齐工作的历史记录。

核查日期：2026-09-27。最终内容依据：`D:\Users\Chen\Desktop\V3修改答复.docx` 的 Word 最终显示视图。对应主稿：`docs/GRSL_LaTeX/GRSL_paper.tex`。`docs/ICSPS2026_LaTeX/` 是旧会议稿，不参与本次对齐。

## 结论

发现 LaTeX 式 (2) 比 Word 多写了 `\alpha=10`，已按 Word 删除并重新编译。当前 GRSL LaTeX 主稿的可见论文内容与最终 Word 对齐。指定 Word 文件已接受全部修订并关闭修订跟踪。两份实际 PDF 均为 5 页；浮动图表的落页位置不同。

## 核查范围与结果

| 项目 | 结果 |
| --- | --- |
| 标题、作者、通讯信息、摘要、关键词 | 一致 |
| 正文 | 39 段顺序对应，文字内容一致；其中 2 段仅行内数学的存储形式不同 |
| 公式 | 编号 (1)–(8) 对应；式 (2) 差异已修正 |
| 表格 | 5 张表的表题、表注、全部 132 个小数及加粗/下划线标记一致 |
| 主图 | 3 幅图及图注对应；Fig. 2/3 的 16 张面板与 Word 内嵌 PNG 逐字节相同，Fig. 1 视觉内容相同 |
| 参考文献 | 22 条顺序与内容对应，21 个 DOI 序列一致 |
| 主稿 PDF | Word 只读导出 5 页；LaTeX 重编译 5 页，无编译错误、未定义引用或 Overfull |

LaTeX PDF 的 Fig. 3 在第 5 页，Word PDF 在第 4 页；这属于排版浮动位置差异。2026-09-27 修正式 (2) 时，LaTeX 仅第 1 页的公式发生变化。2026-09-28 修正指标表述后，两版均保持 5 页；LaTeX 仅第 3 页变化，其他页面与上版渲染逐像素相同，未见裁切或重叠。

## 最终 Word 修订状态

指定路径的 Word 文件现已接受全部修订并保存。处理前，OOXML 有 58 个 `w:ins` 插入节点，Microsoft Word 正文 `Revisions.Count` 为 48；这两个数字采用不同计数口径，不应混称。处理后，所有 Word OOXML 部件中的修订节点为 **0**，Microsoft Word 显示修订跟踪关闭；当前 `word/settings.xml` 未启用 `trackRevisions`。2026-09-27 接受修订前后严格比较可见内容：293 段、5 张表、161 个单元格、70 个 OMML 数学对象、17 个图片引用和 17 个媒体文件完全一致。核查记录见 `tmp/grsl_v3_compare_20260927/visible_content_final_report.json`。

## 2026-09-28 指标表述修正

此前两版正文均写有“Eq. (16)”，但本稿只编号到式 (8)，容易误认为引用本稿公式。现已按作者确认的措辞同步修改 Word 和 LaTeX：并列说明真实 SAR 评价使用 M-index 与 EPI，并明确本研究的 EPI 实现为去噪输出中对角相邻像素的绝对差之和与噪声输入中对应绝对差之和的比值。正文不再出现该误指，问题已解决。

## 已更新的文件

- `docs/GRSL_LaTeX/GRSL_paper.tex` 与重编译的 `GRSL_paper.pdf`。
- `D:\Users\Chen\Desktop\V3修改答复.docx` 已接受修订、关闭修订跟踪，并同步修正指标段落；除该段外，当前可见内容与修正前一致。
- `output/pdf/GRSL_Submission/GRSL_paper.pdf`、`GRSL_LaTeX_V3.zip`、主稿页面预览及 `validation.json`。现有 V3 ZIP 包含与 Word 相同的 `fig5_v3/` 八张 1024×1024 图片。独立补充材料不在本次主稿对齐范围内。
- `docs/GRSL_LaTeX/README.md` 和 `GRSL_REVISION_NOTES.md` 中关于补充图已被正文引用的不准确说法已修正。

同目录的无版本号 `GRSL_LaTeX.zip` 为历史包；本次对齐后的源码包是 `GRSL_LaTeX_V3.zip`。

逐段提取、公式 XML、只读 Word 导出 PDF 和比较脚本保存在仓库的 `tmp/grsl_v3_compare_20260927/`，供复核使用。
