# ICSPS 2026 — Figure 4

图 4 已插入 `docs/ICSPS2026_LaTeX/ICSPS2026_paper.tex`，标签保持
`fig:synthetic_qual`，编译后位于第 5 页，全文 6 页。

## 排版与文件

- 2×4，按从左到右、从上到下排列：Clean、Noisy、LEE、SAR-BM3D、SAR-CAM、Trans-SAR、Ours、Ours ratio。
- 方法名称按作者在本次任务中确认的重命名使用；排版交付不构成方法身份核验。
- `fig4_buildings.pdf`：用于论文的图文件，约 181.35×101.89 mm，适配 IEEEtran 双栏通栏宽度。
- `fig4_buildings.png`：600 dpi 合成图导出；`fig4_buildings_preview.png`：150 dpi 预览。
- `source_images/`：8 张原始灰度图的副本，均为 256×256。
- `caption.txt` / `fig4_insert.tex`：最终英文图注及完整 LaTeX 插入片段。
- `figure_manifest.json`：尺寸、排列、文件哈希和图像嵌入检查记录。
- `paper_qa.json`：整篇论文的编译与视觉检查记录。

原图以无损方式嵌入 PDF，未修改灰度、对比度或图像内容。文字使用嵌入的
Arial 8.5 pt。600 dpi 指合成文件的导出分辨率，原图信息仍为 256×256；
按通栏尺寸排版时，各原图有效分辨率约为 149 dpi。

## 正文修改

将原先建议的双场景、跨 look 展示改为单一 buildings 场景的实际对比。
补入 LEE，说明 Ours ratio 中仍可见建筑轮廓，将定性观察与表 I 的四 look
定量结果区分。未添加未经确认的 look number、PSNR/SSIM 或逐图排名。
图注由 LaTeX 排版，不烘焙到图文件中。

更新后移除了原稿 `IEEEtriggeratref{12}` 的手工换栏，以免完整图 4
插入后把参考文献推到多余一页。其余图仍保持现有稿件状态。

## 重建

依赖：Python、Pillow、reportlab、pypdf、Poppler 的 pdftoppm，以及 Arial 字体。
从此目录运行：

```powershell
python build_fig4.py
```

需要再次同步到论文时，从项目根目录运行：

```powershell
python output/pdf/ICSPS2026_Fig4/build_fig4.py --install-dir docs/ICSPS2026_LaTeX/figures
```

完整论文可从 `docs/ICSPS2026_LaTeX` 用 pdfLaTeX 或 Tectonic 编译。
上传 Overleaf 时保留该目录下的 `figures` 文件夹。
