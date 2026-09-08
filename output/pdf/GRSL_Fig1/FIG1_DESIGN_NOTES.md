# 图 1 重绘说明

本次仅更新 GRSL 稿件的图 1、图注及配套说明。原 ICSPS 目录未写入；所有实验结果图、结果表、实验数字、正文方法和作者信息保持不变。

## 设计结果

图 1 采用三个分区：**(a) 网络总览、(b) 单级解码顺序、(c) 强度重构**。编码器和相同尺度的解码器上下对齐，让四条 skip 直接连接对应阶段。主干采用深色实线，跳连为蓝色实线，引导图分发为紫色虚线。图形采用平面矢量模块、统一边距和线宽，减少交叉线与重复文字。

灰色标记继承的基本操作；FDR、skip fusion、guidance 和表征变换分别使用淡暖色、蓝色、紫色及青绿色。颜色之外保留模块名称、线型及文字说明，不依靠颜色单独表达含义。Projection 保持灰色，Sigmoid 单独标为新增表征操作，避免将整个预测头误标为新贡献。

共享潜变量图记为 **G**，每级的 guidance modulation 记为 **GM**。这一区分避免将“共享同一引导图”误解为“各级调制模块共享参数”。六个 FDR 的参数独立性另行说明。

图注同步解释分区、拼接跳连、GM 含义，以及先逆变换、再强度补偿和裁剪的顺序。未将其他论文中的模块、实验或创新结论移入本方法。

## 官方绘图规范

核查日期：2026-09-08。

- [GRSL 作者检查表 §4.4](https://www.grss-ieee.org/publications/checklist-for-authors/)：图应在刊印尺寸下清晰可读，优先采用矢量格式，配色考虑色觉差异，图注能独立说明图意。本次采用最终双栏宽度绘制，以 8.5–10 pt 为主要文字大小。
- [IEEE Resolution and Size](https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/create-graphics-for-your-article/resolution-and-size/)：PDF 等矢量格式适合图形缩放；双栏常用宽度为约 182 mm。新版 PDF 与旧图保持相同物理尺寸，避免通过缩小文字挤压篇幅。
- [Nature Research Figure Guide](https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/)：借鉴统一字体、可编辑文字、无阴影装饰、避免文字重叠和矢量导出的通用原则。**Nature 自身的 5–7 pt 字号规格未用于本 GRSL 图**；本稿以 GRSL 的可读性要求为准。

## 实际查看的论文架构图

以下来源用于学习组织方法；没有复制其图形资产或逐项照搬布局。

| 原文与一手来源 | 实际查看位置 | 本次采用的组织思路 |
| --- | --- | --- |
| Restormer，CVPR 2022：[CVF 官方条目](https://openaccess.thecvf.com/content/CVPR2022/html/Zamir_Restormer_Efficient_Transformer_for_High-Resolution_Image_Restoration_CVPR_2022_paper.html)、[作者公开稿](https://arxiv.org/pdf/2111.09881) | Fig. 2，PDF 第 3 页 | 网络总览与局部操作展开分层，共用模块颜色，旁路避开主要文字 |
| Uformer，CVPR 2022：[CVF 官方 PDF](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Uformer_A_General_U-Shaped_Transformer_for_Image_Restoration_CVPR_2022_paper.pdf)、[作者公開稿](https://arxiv.org/pdf/2106.03106) | Fig. 2、3，PDF 第 3、4 页 | 将不同抽象层级分区，缩短模块标签，明确多尺度路径 |
| Simple Baselines for Image Restoration / NAFNet，ECCV 2022：[ECVA 官方 PDF](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136670017.pdf)、[作者项目](https://github.com/megvii-research/NAFNet) | Fig. 2，PDF 第 4 页 | 统一模块尺寸与尺度对应关系，避免没有信息作用的立体装饰 |
| SwinIR，ICCV Workshops 2021：[作者公开稿](https://arxiv.org/pdf/2108.10257) | Fig. 2，PDF 第 3 页 | 整体流程、重复模块和内部操作分层；明确恢复路径。这是 Workshop 论文，未当作主会论文表述 |
| Learning Enriched Features for Fast Image Restoration and Enhancement / MIRNet-v2，IEEE TPAMI，DOI 10.1109/TPAMI.2022.3167175：[作者论文主页](https://www.waqaszamir.com/publication/zamir-2022-mirnetv2/)、[作者公开稿](https://www.waqaszamir.com/publication/zamir-2022-mirnetv2/zamir-2022-mirnetv2.pdf) | Fig. 1，PDF 第 3 页 | 完整恢复流程、重复组和多尺度块分层；同类运算同色，重复结构只展开一次。该文是图像恢复网络，未表述为 Transformer 论文 |

研究时已渲染查看上述架构图，部分 CVF PDF 的直接访问受限时使用作者公开稿，不声称读到了不可访问的付费版本。

## 实现核对

- E1–E5：128²×32、64²×64、32²×128、16²×320、8²×512。
- Bottleneck 保持 8²×512；解码方向为 D4→D3→D2→D1→D0，输出依次为 16²×320、32²×128、64²×64、128²×32、256²×16。
- 四条 skip 为 E1→D1、E2→D2、E3→D3、E4→D4；D0 无 skip。
- 解码顺序为上采样→拼接和融合（D1–D4）→FDR→残差卷积→GM。
- G 从 log 输入生成，同一图缩放后送五级；GM 是每级独立的调制模块。
- 一处 bottleneck FDR 与五处 decoder FDR 参数独立；bottleneck 的 Local + FDR 表示正文已定义的局部与频域组合，不是裸 FDR。
- 预测头先给出有界 log 预测，经过精确逆变换，再利用原强度 Y 做补偿，最后 clip。补偿内部算式仍见正文式(6)和补充 Fig. S2。

核对来源为仓库 `transform_main.py`、`arch/trans_basenetworks.py` 和主稿方法。重绘没有改动网络代码。

## 可编辑文件与质量检查

- `figures/fig1_overall_architecture.pdf`：论文使用的矢量图，约 182×92.4 mm。
- `figures/fig1_overall_architecture.svg`：文字可编辑的矢量版本。
- `figures/draw_fig1.py`：可复现绘图源码，使用 Matplotlib 3.8.4 和 Arial；也可指定其他已安装字体。
- `figures/figure_manifest.json`：尺寸、字体、文件信息及结构核对记录。
- 仓库 `output/pdf/GRSL_Fig1/`：600 dpi PNG、150 dpi 刊印尺寸预览及独立矢量文件。

脚本检查模块内文字和画布边界；PDF 中无嵌入光栅图像，字体嵌入为 Type0/TrueType，而非 Type3。已按实际双栏尺寸检查图形及在主稿中的呈现。主稿保持 5 页，原来的 5 张实验表和两组 16 个结果面板均保留。
