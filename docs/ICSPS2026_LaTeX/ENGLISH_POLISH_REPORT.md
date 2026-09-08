# 全文英文润色说明

日期：2026-09-08。当前稿件：`ICSPS2026_paper.tex`；对应 PDF：`ICSPS2026_paper.pdf`，共 7 页。

本轮在已完成的数据修正、逻辑核查和排版修复基础上，润色摘要、引言、相关工作、方法、实验分析、结论及全部图注，并整理部分表下注释。实验数据以 `../实验结果表.md` 和作者后续确认的实验条件为准。

## 主要修改

| 部分 | 本轮处理 |
|---|---|
| 摘要 | 按问题、方法、定量证据和结论重组，突出四种视数下的 PSNR 排名、FDR 消融差值及真实 SAR 适应效果。 |
| 引言与相关工作 | 明确与 Trans-SAR 的继承关系，把方法分组与当前研究问题联系起来，减少背景、贡献和相关工作之间的重复。 |
| 方法 | 按输入、运算和输出顺序解释表示转换、FDR、guidance、强度补偿与 AMS；统一术语，拆分长句，保持技术限定和可复现细节。 |
| 实验结果 | 先交代主要发现，再给具体差值及解释；区分整体评估、场景评估和消融比较，避免逐格复述表格。 |
| 讨论与结论 | 集中说明评估边界，减少各段反复出现的防御性解释；总结得到的证据，同时保留尚未验证的推广条件。 |
| 图注与版面 | 精简图注，保留独立阅读所需的信息；修复均值连写造成的单词间距拉大。保持 IEEE 字体、页边距及既有图后间距修复。 |

## 结果表达如何得到加强

本轮用可核对的比较代替笼统的优势描述：

- 合成实验明确写出：在所比较方法中，同一模型在四种视数下均获得最高 PSNR 点估计；同时说明 SSIM 在不同视数下的排名差异。
- 消融实验明确写出：移除全部 FDR 后，PSNR 下降 1.6791 dB、SSIM 下降 0.0179，并说明比较对应的模块配置。
- 592 个真实 SAR 测试块上，AMS 相对 Ours-base 的 M-index 降低 26.62%，EPI 提高 94.23%；最优排名限定于表中的滤波输出。
- 三类真实场景均表现出 M-index 降低和 EPI 提高，将这一一致趋势与平滑程度和边缘保留共同解释。

这些表述强化了已有证据的呈现，没有新增实验、改变数据或添加统计显著性、普遍领先及跨传感器泛化等未经验证的结论。

## 独立表达与重复控制

改写以段落功能和论证顺序为单位，删除重复说明、压缩套话并重建句式，保留固定科学术语和引文归属。这种处理遵循 Purdue 关于理解后重新组织表达的建议，以及 Harvard 关于用当前论证组织来源信息的建议。[Purdue 改写指南](https://owl.purdue.edu/owl/research_and_citation/using_research/quoting_paraphrasing_and_summarizing/paraphrasing.html)、[Harvard 来源使用指南](https://usingsources.fas.harvard.edu/summarizing-paraphrasing-and-quoting)

本轮未将论文提交给外部查重服务，也未运行 iThenticate 或 Turnitin。因此，不能给出实际相似度或承诺降低百分比。方法名、公式、必要术语和参考文献保留原有准确表达。

## 实际查阅的权威写作来源

以下 8 项核心来源均已打开并读取相应正文或 PDF 内容。它们用于指导英文组织和改写，不替代论文中的研究文献，也不作为新的科学结论来源。

| 来源 | 对本稿采用的建议 |
|---|---|
| [IEEE Author Center：Structure Your Paper](https://conferences.ieeeauthorcenter.ieee.org/write-your-paper/structure-your-paper/) | 摘要独立且简明；引言说明具体问题；结果解释与证据相符。 |
| [Springer Nature：Six ways to improve your science writing](https://communities.springernature.com/posts/six-ways-to-improve-your-science-writing) | 使用具体主语和动词，以明确事实替代空泛修饰，删除无作用的重复。 |
| [Elsevier Researcher Academy：Elements of Style for Writing Scientific Journal Articles](https://researcheracademy.elsevier.com/uploads/2017-11/Elements%20of%20Style%20for%20Writing.pdf) | 一个段落围绕一个主题，注意信息承接、指代和时态；重点读取 PDF 第 4–7 页。 |
| [Elsevier：Manuscript preparation](https://www.elsevier.com/publishing/publish-in-a-journal/manuscript-preparation) | 遵循目标刊会格式，以清楚且有组织的文字报告研究。 |
| [Purdue OWL：Paraphrase—Write It in Your Own Words](https://owl.purdue.edu/owl/research_and_citation/using_research/quoting_paraphrasing_and_summarizing/paraphrasing.html) | 理解原意后重新组织语言，再核对意义和引用。 |
| [Harvard：Summarizing, Paraphrasing, and Quoting](https://usingsources.fas.harvard.edu/summarizing-paraphrasing-and-quoting) | 选择当前论证所需的内容，明确来源在段落中的作用。 |
| [Harvard：What Constitutes Plagiarism?](https://usingsources.fas.harvard.edu/what-constitutes-plagiarism-0) | 避免片段拼接和仅替换同义词；保持引文范围清楚。 |
| [UNC Writing Center：Writing Concisely](https://writingcenter.unc.edu/tips-and-tools/conciseness-handout/) | 删除冗余、笨重短语和套话，同时保留影响含义的限定。 |

## 数据和技术含义保护

相对于本轮开始时的备份，自动对比确认：5 张表的表体、8 个公式、21 条参考文献、全部引用键及其出现次数均未变化。5 幅图仅修改图注，图的引用路径、布局参数和标签保持一致；标题、关键词、字体和页边距设置也保持一致。末页现按作者最终确认采用自然分栏，先排满左栏，再续排右栏。

以下作者确认的条件继续适用：

- UCMerced 三场景使用 **L=4**，住宅类别为 **dense residential**，每类 100 个样本。
- 真实 SAR 的 ENL 报告为**均值**。
- 三场景真实 SAR 中，Ours+AMS 的 Average 为 **72.7568 / 1.454366 / 0.798884**；Ours-base 为 **597.0446 / 1.740321 / 0.372438**。
- 两组真实 SAR 评价沿用作者确认的 Ma et al. (2024) EPI 与 Gomez et al. (2017) M-index 定义；固定 ROI 的描述仍只适用于 592 个测试块的评估。

先前核查中的实验来源记录和待补充的复现信息仍见 `LOGIC_LANGUAGE_REVIEW.md`。英文润色没有新增实验，不能替代这些材料的补充。

## 编译与版面检查

截图反馈后的局部修正：将 II-B 的标题改为 `Frequency Processing and Image Representation`，避免标题中的省略式连字符；将正文 `\texttt{Noisy}` 改为普通字体的小写 `noisy`。两处均保持原意。重新编译并检查第 1、4 页，页数、图表所在页及数据保护检查均保持通过。

第一轮留白修正：仅在表 V 末尾加入局部 `\par\vspace{-8pt}`，将表下注释到正文的可见间距从约 7.8 mm 收紧至 5.0 mm；将参考文献换栏点从第 [2] 条前移至第 [3] 条前，使末页两栏底部高度差从约 12.9 mm 缩小到 6.1 mm。第 6、7 页已检查，前 5 页的文字、坐标和字体与该次修改前完全一致。该阶段记录为 `../../output/pdf/ICSPS2026_EnglishPolish/spacing_validation.json`。

随后曾试用 `flushend` 自动对齐两栏，该阶段核验记录为 `../../output/pdf/ICSPS2026_EnglishPolish/column_balance_validation.json`，已由下述最终偏好取代。

作者最终澄清：左栏排满后再续到右栏，右栏允许留空。因此当前稿件已删除 `flushend`，也未设置人工换栏点。末页左栏自然排至参考文献 [3]，右栏从 [4] 接续至 [21]，保留完整条目。共 7 页，前 6 页文字、坐标和字体与该次调整前完全一致，正文与数据未改。当前记录为 `../../output/pdf/ICSPS2026_EnglishPolish/natural_columns_validation.json`。

图引用间距修正：本地官方模板 `official_conference_101719.tex` 第 217–218 行明确要求句首也使用 `Fig.` 缩写。引言中的 `Fig.~\ref{fig:overview}` 本身正确，但内部空格随两端对齐伸长；现仅将此引用包在 `\mbox{...}` 中，固定内部自然间距。句意、引用编号和段落行数保持不变，第 2–7 页文字坐标与本次调整前完全一致。检查记录为 `../../output/pdf/ICSPS2026_EnglishPolish/fig_reference_validation.json`。

- 主 PDF 已成功覆盖更新，共 **7 页**。
- 全部 7 页已渲染检查；无编译错误、未定义引用、缺失引文、Overfull 或 Underfull hbox 警告。
- 第 4 页保留一个 Underfull vbox 警告；对应页面已目视检查，未出现此前截图中的大段图后空白。
- 表 I–V 分别位于第 **4、5、5、6、6** 页；图 1–5 分别位于第 **2、3、4、5、6** 页。
- 最终 PDF 副本、页面预览、编译日志及逐项核验记录保存在 `../../output/pdf/ICSPS2026_EnglishPolish/`。
