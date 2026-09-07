# 图 1 / 图 3 限定范围优化与验收报告

已直接修改当前论文工程并完成 Tectonic 编译、PDF 渲染和逐页检查。论文由 **8 页变为 7 页**；保留图 3 的两个子图。未调整会议模板字号、版心、行距、文档类或图注格式。

## 修改文件

路径均相对于 `D:/research/sar_transformer-main/`。

| 文件或目录 | 修改内容 |
|---|---|
| `docs/ICSPS2026_LaTeX/ICSPS2026_paper.tex` | 重写图 1/3 图注；保留原 label；统一 `Fig.~\ref`；补偿流程指向图 3(a)，AMS 指向图 3(b)；移除旧参考文献断栏指令 |
| `docs/ICSPS2026_LaTeX/ICSPS2026_paper.pdf` | 重新编译的 7 页论文 |
| `docs/ICSPS2026_LaTeX/figures/draw_fig1.py`、`draw_fig3.py` | 新增随论文交付的可编辑绘图脚本副本，与现有源同步 |
| `docs/ICSPS2026_LaTeX/figures/fig1_overall_architecture.svg`、`fig3_representation_ams.svg` | 新增可编辑矢量 SVG |
| `docs/ICSPS2026_LaTeX/figures/fig1_overall_architecture.pdf`、`fig3_representation_ams.pdf` | 论文实际引用的矢量文件 |
| `output/pdf/ICSPS2026_Fig1/`、`ICSPS2026_Fig3/` | 修改现有 `draw_fig*.py`；重新生成对应 PDF、SVG、600 dpi PNG、150 dpi 预览和 `figure_manifest.json`；同步 `caption.txt`、`fig*_insert.tex`；README 标明旧 proof/QA 为历史记录 |
| `docs/ICSPS2026_LaTeX/README.md`、`LAYOUT_REFINEMENT_REPORT.md` | 更新构建说明、尺寸、页码和验收记录 |
| `output/pdf/ICSPS2026_LayoutRefinement/` | 保存修改前基准、渲染页、当前编译日志/aux、结构核验脚本与 JSON、正文差异和本报告 |

绘图源沿用已有 Matplotlib 方案和配色，无截图修补或整图栅格化。原先另一个 `scripts/icsps2026/plot_fig1_trans_sar_architecture.py` 是早期版本，并非论文当时使用的图源，因此未修改。

## 图示调整

**图 1：** 删除底部 AMS 子图、(a)/(b) 分区标题和独立解码器操作顺序横带。将输出投影、sigmoid、逆变换、补偿和裁剪的逐框路径合并成 “Bounded-log prediction head” 与 “Intensity reconstruction”。保留输入尺寸及通道数，减少重复空间尺寸。

保留五级编码器、局部—频谱瓶颈、D4–D0、E4→D4 / E3→D3 / E2→D2 / E1→D1 四条跳跃连接、六处独立参数 FDR、对数输入生成的共享引导图及五级调制、原始强度 Y 旁路和最终强度输出。D0 没有 skip/fusion。解码器采用中性外框，内部灰色继承算子与彩色新增操作分别标识，未将整个解码器归为继承模块。

**图 3(a)：** 保留精确逆变换 `Xtilde = T_alpha^-1(Zhat)`、补偿器输入 `[Y, Xtilde]`、输出 `Rc`、`Xtilde + gamma_c Rc` 同强度域相加及后续 `[0,1]` 裁剪。Y 旁路与 Xtilde 加法旁路均保留。移除框内实现超参数，正文参数不变。

**图 3(b)：** 保留 Bernoulli M、Ym、全模型掩码输入、仅 Transformer 编码器冻结、其余重建模块可训练、掩码位置 L1 与全输出 TV、训练重采样、固定验证掩码选模及无掩码推理。原始 Y 和 M 只通向监督目标/位置；没有向网络内部额外接入未掩码 Y。图注明确 (a) 的一般输入 Y 在 AMS 调用时由 Ym 替代，包括补偿旁路。删除框内 epochs/Adam/学习率信息，正文保留。

## 编译及尺寸

继续使用既有构建脚本及 **Tectonic 0.17.0**，未更换引擎：

```powershell
.\docs\ICSPS2026_LaTeX\build.ps1
```

脚本实际调用：

```text
tectonic.exe -X compile ICSPS2026_paper.tex --outdir D:\research\sar_transformer-main\tmp\pdfs\paper_fig123\build --keep-logs --keep-intermediates
```

修改前先保存了现有 8 页 PDF、LaTeX、绘图脚本和五幅图，并用原命令成功重编译基准。最终构建退出码为 0，交叉引用经自动多轮编译稳定。

| 图 | 最终页码 | 最终 PDF 中实际宽 × 高，不含图注 | 矢量源页面宽 × 高 | 普通标签 |
|---|---:|---|---|---|
| 图 1 | 2 | **181.371 × 92.437 mm** | 181.353 × 92.428 mm | 主要 9 pt，最小 8.5 pt |
| 图 3 | 4 | **88.573 × 133.358 mm** | 88.568 × 133.350 mm | 主要 8.8 pt，最小 8.2 pt |

实际尺寸通过最终 PDF 的图形 Form 及完整放置矩阵测得。Tectonic 的极小等比舍入使其与源页面尺寸略有不同；LaTeX 仍只设置 `width=\textwidth` / `width=\columnwidth`，没有同时固定宽高。原图形高度为 152.400 / 182.033 mm；通过布局减少约 60 / 49 mm。

图 1 仍为双栏 `[!t]`，图 3 仍为单栏 `[!t]`。紧凑图形导致原 `\IEEEtriggeratref{15}` 把文献推到额外一页；移除这条原手动断栏指令后文献自然接续，页数变为 7。参考文献内容完全不变，未增加负间距、强制分页或新宏包。

## 验收结果

使用 Poppler 渲染基准全部 8 页及最终全部 7 页，检查了重新分页、图注间距和页尾。图 1/3 同时按最终排版尺寸及较高分辨率检查；独立复核了第 2、4、5、7 页。未见图内文字重叠、箭头穿字、边框裁切、错误分支或图注碰撞。最终第 7 页两栏文献排布完整，旧多余页与异常空白已消除。

`check_refinement.py` 的检查通过：8 个公式环境、全部原行内数学、3 个表格、参考文献、图 2/4/5 的完整环境和文件哈希、IEEEtran 和主文件前导设置均保持不变。已人工核对剩余正文差异，只有两处图 3 引用调整及上述断栏指令移除；统一 Figure→Fig. 的格式修改不改变正文含义。模型、指标实现、数据与实验配置未修改。

无未定义/重复 label、`??`、缺图、字体警告或 overfull box。图 1/3 没有栅格图像对象，所有可见字体均嵌入。源脚本与论文目录副本、导出 PDF 与实际引用 PDF 的哈希一致。

**保留的警告：** 最终日志有一条 `Underfull \vbox (badness 10000)`，对应第 5 页（源文件第 228 行附近）。该页仅有轻微段间留白，未见严重空白或分页异常。本轮保留模板排版，不通过改变模板参数消除该警告。

## 既有源文件不一致（未修改）

1. `ICSPS2026_paper.tex:152` 的 AMS 分母为 `||M||_1 + epsilon`；`train_icsps2026_ams.py:111` 使用 `mask.sum().clamp_min(1.0)`。图内保留通用 “Masked L1 + TV”，没有替作者改动公式或代码。
2. `transform_main.py:1535–1541` 的历史变量名 `log_branch_output` 与正式调用实际传入的逆变换强度不一致；`:2159–2163` 的计算连接正确。`DualDomainFusion` 和 `NoiseEstimator` 也是历史命名，本轮未据此增加架构或监督。
3. `numeric_domain.py:81` 的历史说明声称 percentile normalization helper 用于 AMS/evaluation；正式路径在 `train_icsps2026_ams.py:96` 和 `evaluate_icsps2026_real.py:475` 调用固定映射。图从已归一化强度开始，不涉及该文档差异。

复查入口：`qa.json` 保存结构与尺寸检查，`textual_diff.patch` 保存范围内正文差异，`visual_review.json` 保存最终人工视觉验收及文件哈希；`before/` 保存基准，`after/pages/` 保存最终 7 页预览。


## 后续箭头修正

用户指出的图 3 目标输入箭头确有局部几何问题，现已修正并独立检查 SVG/PDF。此前“未见箭头问题”的记录未覆盖该处异常，以 `output/pdf/ICSPS2026_LayoutRefinement/arrow_fix/qa.json` 为准。图形尺寸、文字、方法内容和 7 页分页不变。原论文 PDF 被占用，新版已保存为 `docs/ICSPS2026_LaTeX/ICSPS2026_paper_arrowfix.pdf`；原文件仍保留修正前版本。
