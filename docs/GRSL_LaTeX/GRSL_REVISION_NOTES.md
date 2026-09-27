# GRSL 精简与核验说明

核查日期：2026-09-19。

## 本次结果

从原会议稿另建独立 GRSL 目录，主稿压缩至 **5 页（含参考文献）**。V3 同步后的作者顺序为 Hongyu Chen、Peng Liu（Senior Member, IEEE）；通讯作者为 Peng Liu，邮箱 pliu@fudan.edu.cn。全体作者单位为 Key Laboratory of Information Science of Electromagnetic Waves, Fudan University, Shanghai, China。姓名、顺序、会员等级及通讯信息已在主稿和补充材料中同步。保持原论文标题；转投期刊不要求仅为“缩写”而改文件名或缩成陌生缩略语。

主要改动是压缩文字和重组内容：相关工作并入引言，集中交代研究问题与新增操作；方法保留实现所需的条件和公式，删去流程的重复解释；实验设置统一说明，分析围绕改进幅度、不同指标的取舍和结论边界展开。V3 Word 新增的 Fig. 3 面板也已按文档中的可见顺序、标签和图注同步；未扩展或改写其方法解释。

## 官方依据

1. [GRSL Checklist for Authors](https://www.grss-ieee.org/publications/checklist-for-authors/) 是主要依据：总长最多 5 页，包括参考文献及可选作者简介；关键方法、实验和结果必须在主稿内；引言应明确问题、相关研究与贡献；补充材料只承载非关键内容，并在正文引用。
2. [GRSL 刊物主页](https://www.grss-ieee.org/publications/geoscience-and-remote-sensing-letters/) 再次给出 5 页上限，并要求清楚说明实验数据、方法及条件。
3. [GRSS Peer Review](https://www.grss-ieee.org/publications/author-resources/peer-review/) 说明采用 single-anonymous 评审，因此使用用户提供的真实署名。没有补写未知共同作者、基金或通信地址。
4. [IEEE Tools for Authors](https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/authoring-tools-and-templates/tools-for-ieee-authors/) 推荐使用 Template Selector。本次核实了官方模板指引，但没有取得可核查的最新 GRSL 专用模板包；实际采用已有 IEEEtran 的 journal 模式，未声称下载了新版专用模板。
5. [GRSL Submission Hints](https://www.grss-ieee.org/publications/grsl-submission-hints/) 源于 2015 年社论，只将两栏排版和不要篡改样式硬塞页数的原则作为补充依据；旧投稿网站、附件形式及旧流程没有当作当前强制要求。

当前公开官网要求已用于本次改稿；登录后投稿系统的全部字段和上传校验未在本次操作中逐项验证。

## 阅读的相关原文与编辑借鉴

以下文献仅用于研究短文组织方式，不复制其措辞，也未为增加引用数量而加入主稿。

- **Enhanced Deep Learning SAR Despeckling Networks Based on SAR Assessing Metrics**，IEEE GRSL，vol. 22，2025，article 4009305，5 页，DOI 10.1109/LGRS.2025.3577907。[作者机构库出版版本全文](https://accedacris.ulpgc.es/bitstream/10553/141828/1/Enhanced_Deep_Learning_on_SAR_Assessing_Metrics.pdf)。借鉴相关工作并入引言、共同实验条件集中说明、真实 SAR 指标联合解释的组织方式。
- **Despeckling Sentinel-1 GRD Images by Deep-Learning and Application to Narrow River Segmentation**，IGARSS 2021，pp. 2995–2998，DOI 10.1109/IGARSS47720.2021.9554350。[IP Paris 机构成果记录](https://researchportal.ip-paris.fr/en/publications/despeckling-sentinel-1-grd-images-by-deep-learning-and-applicatio/)，[作者全文](https://arxiv.org/pdf/2102.00692)。这是一篇会议论文，借鉴其明确区分继承方法与适配部分的写法；本稿没有据此声称进行了下游分割实验。
- **SAR Image Despeckling Through Convolutional Neural Networks**，IGARSS 2017，pp. 5438–5441，DOI 10.1109/IGARSS.2017.8128234。[那不勒斯大学机构成果记录](https://iris.unina.it/handle/11588/703221)，[作者项目页](https://grip-unina.github.io/SAR-CNN/)，[作者全文](https://arxiv.org/pdf/1704.00275)。这也是会议论文，借鉴其将模拟与真实 SAR 的评价条件分开、让图与正文承担不同说明任务的组织方式。

## 图表对应关系

| 原稿内容 | GRSL 位置 | 保留情况 |
| --- | --- | --- |
| Table I：UCM-21 四视数比较 | 主稿 Table I | 全表原样保留 |
| Table II：UCMerced 三场景 | 主稿 Table II | 全表原样保留，L=4，dense residential |
| Table III：消融 | 主稿 Table III | 全表原样保留 |
| Table IV：592 个真实 SAR 测试块 | 主稿 Table IV | 全表原样保留，ENL 报均值 |
| Table V：真实 SAR 三场景 | 主稿 Table V | 全表原样保留，使用已修正 Average |
| Fig. 1：总体架构 | 主稿 Fig. 1 | 按用户要求重绘为三个分区，保持结构与运算含义；图注同步更新 |
| Fig. 2：FDR 局部结构 | 补充 Fig. S1 | 原图和图注保留；关键运算仍在正文 |
| Fig. 3：重构及 AMS 路径 | 补充 Fig. S2 | 原图和图注保留；关键条件仍在正文 |
| Fig. 4：模拟数据可视化 | 主稿 Fig. 2 | 8 个面板、图注和尺寸设置原样保留 |
| Fig. 5：真实 SAR 可视化 | 主稿 Fig. 3 | 已按 V3 Word 同步 8 个 1024×1024 面板，保持原标签、图注与 2×4 排列 |

补充文件共 1 页，当前主稿未引用 Supplementary Figs. S1 and S2。它只提供主稿已完整说明的运算示意，未迁出实验结果或关键方法描述。

## 事实与解释边界

- 保留 8,400 个 UCM-21 固定测试对、L=1/2/4/8，以及模型不输入 L 的条件。
- 保留 UCMerced 三场景各 100 张、L=4、dense residential 的条件。
- 真实 SAR 训练、验证和测试按父图划分；592 块与三场景 60 块分别总结，未混合统计。
- ENL 均值、M-index 与 EPI 定义和原始来源保留。本文的 EPI 实现为去噪输出与噪声输入的对角相邻像素绝对差求和之比，未替换成 Sobel 梯度相关。
- Table V 的 Ours+AMS Average 保留为 ENL **72.7568**、M-index **1.454366**、EPI **0.798884**。
- 保留 ENL 不能单独代表图像质量、残余斑点也可能提高 EPI、相关斑点下掩码目标不保证无偏等限制。
- 消融只支持所测配置内的条件贡献；未声称已证明模块协同、排除了参数量影响，或实现跨传感器泛化。
- 使用 mean squared intensity error 与公式一致；明确由最低验证 MSE 选择检查点。

## 编译与完整性验证

- 原 ICSPS 目录全部 **36 个文件**的文件列表和 SHA-256 与修改前一致。
- 5 张表的 tabular 表体逐字符一致，包括所有数字、精度、粗体和下划线。
- Fig. 2 的 8 个实验面板保持不变；Fig. 3 的 8 个面板与 V3 Word 的 `image10.png` 至 `image17.png` 逐一 SHA-256 一致。旧 Fig. 3 资源保留在 `figures/fig5/` 以便回溯，新资源独立存放在 `figures/fig5_v3/`。
- 8 个公式去除空白后逐字符一致。
- V3 同步后共有 22 条参考文献，新增 Fast Fourier Convolution 条目并按正文首次引用顺序编号；LaTeX 自动交叉引用可解析。此前正文中的“Eq. (16)”误指已于 2026-09-28 修正，见末尾复核记录。
- 主稿 5 页，补充方法图 1 页。作者信息更新后，编译无错误、未定义引用或 Overfull；参考文献中有 3 处 Underfull hbox 提示，已通过页面预览核对，无裁切或重叠。保留原稿正文和模板尺寸，未为消除提示改动实验内容。
- 已重新编译并渲染检查主稿全部 5 页：新 Fig. 3 面板、原标签和图注均清晰，无文字裁切、图表重叠、未定义引用或 Overfull。作者邮箱完整显示，末页保留先左后右的自然排版。

可复查的 PDF、页面预览和校验记录存放于仓库 `output/pdf/GRSL_Submission/`。

## 2026-09-27 与最终 V3 Word 复核

以 `D:\Users\Chen\Desktop\V3修改答复.docx` 的最终显示内容为准，逐段、逐图表、逐公式复核 GRSL 主稿。LaTeX 式 (2) 原先在分式后额外写有 `\alpha=10`，现已按 Word 删除；该值仍在 Fig. 1 和实验设置中明确给出。主稿与 Word 均为 5 页，3 幅主图、5 张表、8 个编号公式、22 条参考文献对应；Fig. 2、Fig. 3 的 16 张面板图片与 Word 嵌入文件逐字节一致。

2026-09-28 已按作者确认的措辞修正此前的“Eq. (16)”误指：Word 与 LaTeX 正文均并列说明 M-index 和 EPI，并明确对角相邻像素绝对差求和比为本研究的 EPI 实现。指定路径的 Word 文件此前已接受全部修订并关闭修订跟踪：全部 OOXML 部件中的修订节点为 0，Microsoft Word 显示修订跟踪关闭，当前 `word/settings.xml` 未启用 `trackRevisions`。接受修订前曾有 58 个 `w:ins` 插入节点；Microsoft Word 正文 `Revisions.Count` 为 48，两者计数口径不同。2026-09-27 接受修订前后严格比较可见内容，293 段、5 张表、161 个单元格、70 个 OMML 数学对象及 17 个图片引用/媒体文件完全一致。详情见 `output/pdf/GRSL_Submission/V3_alignment_check_2026-09-27.md`。

## 2026-09-28 IEEE GRSL 格式复核

重新对照 GRSL 官方清单及 IEEE 模板检查后，撤销参考文献的 `\scriptsize` 覆盖，并将四处表注恢复到 8 pt。正文保持 IEEEtran 默认的 10 pt 字号、行距和页面尺寸。通过精简约 160 个英文词的重复表述维持主稿 5 页，Word 同步；实验数字、公式、图表、引用顺序和技术限制保留。Word 小节 C 的斜体格式也已统一，五个 Index Terms 已按字母顺序排列。

参考文献 [1]、[19] 列全六位作者，[19] 补充数据集发布日期，[10]、[12] 使用正式文章编号。新版正常字号 PDF 共 5 页，字体全部嵌入，编译无错误、未定义引用、Overfull 或 Underfull。逐页核对未见裁切、重叠或缺字；此前 Synthetic Comparison 的句子跨页并被图表隔断的问题已消除。完整记录见 `output/pdf/GRSL_Submission/GRSL_format_check_2026-09-28.md`。独立补充材料本次未修改。
