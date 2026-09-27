# IEEE GRSL 投稿格式复核

复核日期：2026-09-28。对象为最终 Word 主稿与 `docs/GRSL_LaTeX/GRSL_paper.tex` 及其 PDF。

本目录最新版主稿为 `GRSL_paper.pdf`，旧同名 PDF 已于 2026-09-28 成功覆盖。它与 `GRSL_paper.updated.pdf`、源码目录中的 `GRSL_paper.pdf` 及更新后的 `GRSL_LaTeX_V3.zip` 内主稿逐字节一致。当前整套页面预览位于 `format-checked-pages/`。

## 官方依据

- [GRSL Checklist for Authors](https://www.grss-ieee.org/publications/checklist-for-authors/)：采用期刊模板；总长不超过 5 页（含参考文献）；公式编号一致并融入句子；图中文字在印刷尺寸下可读。
- [IEEE 作者模板入口](https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/authoring-tools-and-templates/tools-for-ieee-authors/ieee-article-templates/)：使用期刊模板。当前 LaTeX 为 IEEEtran V1.8b 的 journal 模式，默认 10 pt、Letter、双栏。
- [IEEE Editorial Style Manual](https://journals.ieeeauthorcenter.ieee.org/wp-content/uploads/sites/7/IEEE-Editorial-Style-Manual-for-Authors.pdf)：同级小节标题使用斜体；图题位于图下、表题位于表上；图表与公式编号应一致。短文不要求引言首字下沉。
- [IEEE 图像分辨率与尺寸](https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/create-graphics-for-your-article/resolution-and-size/)：灰度栅格图应高于 300 dpi，图内文字在最终尺寸下应清晰。
- [IEEE Reference Guide](https://ieeeauthorcenter.ieee.org/wp-content/uploads/IEEE-Reference-Guide.pdf)：六位及以下作者全部列出，采用完整的期刊文章或数据集引用格式。

## 本次修正

1. 移除 LaTeX 参考文献的 `\scriptsize` 覆盖，恢复 IEEEtran 默认 8 pt；四处表注也由 7 pt 恢复到 8 pt。正文保留模板的 10 pt 字号、行距、栏宽和页边距。
2. 精简约 160 个英文词的重复表达，以保持 5 页。Word 与 LaTeX 同步。实验数字、数学表达、表格数据、图片、引用顺序及技术限制保持原有含义。
3. 统一 Word 小节 “C. Supervision and Masked Adaptation” 的斜体格式；将五个 Index Terms 按字母顺序排列；修正一处未闭合括号、个别单复数和逗号连接句问题。
4. 文献 [1] 和 [19] 列全六位作者；[19] 补充数据集发布日期；[10] 和 [12] 使用正式文章编号 5215417、5204017。
5. 修正后 Synthetic Comparison 的相关句子不再被跨页图表隔开。

## 核验项目

| 项目 | 结果 |
| --- | --- |
| 主稿页数 | Word 与 LaTeX 均为 5 页，包含 22 条参考文献 |
| 纸张与分栏 | Letter，双栏；LaTeX 使用 IEEEtran 默认尺寸 |
| 字号 | 正文 10 pt；图表标题、表体、表注和参考文献约 8 pt |
| 图中文字 | Fig. 1 主标签 8.5–10 pt；小号上/下标随数学符号排版 |
| 图片 | Fig. 1 为矢量；Fig. 2 面板约 306 dpi；Fig. 3 面板约 1223 dpi |
| 图表公式 | 3 幅主图、5 张表、8 个编号公式，均有正文对应引用 |
| PDF 字体 | 所有字体均嵌入 |
| 编译 | 无错误、未定义引用或 Overfull；本次正常字号编译无 Underfull 提示 |
| 版面 | 逐页检查文字、公式、图表、标题、页尾和参考文献 |
| Word 修订状态 | 修订 0、批注 0、修订跟踪关闭 |

39 段正文逐段对照通过，其中两段的行内数学分别采用普通文本与公式对象存储，已核对其可见内容。五张表的全部 132 个小数及表体文字顺序一致；22 条参考文献、12 项图表标题和表注、标题/作者/摘要/关键词均对应。Word 的 70 个数学对象、5 张表和 17 个媒体文件保留；LaTeX 的公式块、行内数学序列、表体、图片路径和引用顺序与本次调整前一致。

独立补充材料不在本次主稿对齐范围内。此复核针对论文文件的格式与排版；投稿系统中的作者资料等字段需要与主稿一致。

本次修改前备份、逐项文字修改记录及 PDF 核验数据保存在 `tmp/grsl_format_audit_20260928/`。
