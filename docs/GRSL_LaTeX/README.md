# GRSL 精简稿

本目录是独立的 GRSL 稿件，原始 `docs/ICSPS2026_LaTeX` 目录未修改。

## 文件

- `GRSL_paper.tex` / `GRSL_paper.pdf`：主稿，**5 页，包含全部参考文献**。
- `GRSL_method_diagrams.tex` / `GRSL_method_diagrams.pdf`：1 页补充方法示意图，正文已引用 Fig. S1、S2。
- `IEEEtran.cls`、`figures/`：本地模板类及全部原始图形资源。
- `GRSL_REVISION_NOTES.md`：精简策略、图表对应关系、官方及论文来源、校验结果。
- `build.ps1`：同时生成主稿和补充方法图 PDF。
- `FIG1_DESIGN_NOTES.md`：图 1 的绘图依据、重绘说明及可编辑资源。

作者顺序：Hongyu Chen；Peng Liu（Senior Member, IEEE）。通讯作者：Peng Liu，邮箱 pliu@fudan.edu.cn。

全体作者单位：Key Laboratory of Information Science of Electromagnetic Waves, Fudan University, Shanghai, China。

## 内容保留

5 张实验结果表及其所有数值、精度、粗体和下划线均保留在主稿中。两组实验结果图的全部 16 个面板、图注及绘图尺寸设置原样保留。总体架构图仍为 Fig. 1，现已按用户要求重绘为“网络总览、解码顺序、强度重构”三个分区；原 Fig. 4、5 在主稿中依次编号为 Fig. 2、3。

原 Fig. 2、3 是局部方法示意图，现作为补充 Fig. S1、S2 单独保存。其必要运算、训练条件及 8 个公式均已写入主稿，主稿不依赖补充文件来说明关键方法或结果。22 条参考文献全部保留，并按正文首次引用顺序编号。

采用 IEEEtran 期刊模式，保留标准 10 pt 正文字体及页边距。末页按先填左栏、再排右栏的顺序流动。

## 编译

在本仓库根目录执行：

```powershell
& .\docs\GRSL_LaTeX\build.ps1
```

脚本优先查找已安装的 Tectonic，否则使用仓库 `tmp/latex_runtime/tectonic.exe`。其他电脑可显式传入 `-TectonicPath`。

也可将整个目录上传至 Overleaf，选择 `GRSL_paper.tex` 为主文件、使用 XeLaTeX 编译；补充材料以 `GRSL_method_diagrams.tex` 为主文件另行编译。两份文稿都使用同一目录内的图形资源，无需原 ICSPS 目录。

已在本机通过 Tectonic 编译并逐页检查。其他 TeX 发行版的换行和浮动位置可能略有差别，重新编译后应核对主稿仍为 5 页。

## 投稿文件区分

主稿与补充方法图是两份独立 PDF。补充图提供可选的直观说明；不能把两份 PDF 合并后当作 5 页主稿。提交时按 GRSL 投稿系统的文件类别分别上传，README 和改稿说明无需当作论文正文上传。本次仅完成稿件准备，未进行线上投稿。
