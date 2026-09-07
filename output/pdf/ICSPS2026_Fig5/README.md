# ICSPS 2026 — Figure 5

图 5 已插入 `docs/ICSPS2026_LaTeX/ICSPS2026_paper.tex`，保留标签
`fig:real_qual`。当前编译稿共 7 页，图 4 在第 5 页，图 5 在第 6 页。

## 排列与显示

采用与图 4 相同的 2×4 通栏排版：

| 上排 | Noisy | SAR-BM3D | SAR2SAR | SDUDNet |
|---|---|---|---|---|
| 下排 | Trans-SAR | Ours-base | Ours+AMS | Ours+AMS ratio |

方法标签按作者确认的重命名使用。本次只进行图像组合和论文排版，
不进行历史输出、checkpoint 或模型身份追溯。

8 张原图均为 256×256；Noisy 原有的红框及 RGB 颜色保留，其余原始灰度
显示保持不变。未增加 ROI 框、裁剪、对比度调整或局部放大图。
末格仅显示作者提供的 Ours+AMS ratio，不增加 Ours-base ratio。

图尺寸约 181.35×101.89 mm，Arial 字号 8.5 pt，适配 IEEEtran 通栏宽度。
PDF 无损嵌入原图并使用矢量文字。600 dpi PNG 是合成图导出分辨率，
原图信息仍为 256×256；通栏排版下有效原图分辨率约 149 dpi。

## 交付文件

- `fig5_real_sar.pdf`：论文插图。
- `fig5_real_sar.png`：600 dpi 导出。
- `fig5_real_sar_preview.png`：150 dpi 预览。
- `source_images/`：原始图片副本。
- `caption.txt` / `fig5_insert.tex`：最终图注和 LaTeX 片段。
- `build_fig5.py` / `figure_manifest.json`：组合脚本与排版记录。
- `paper_qa.json`：编译、引用和版面检查记录。

## 论文更新

将双场景及成对 ratio 的占位说明替换为单场景实图；图中与正文统一使用
Ours-base，与表 III 一致。正文描述 AMS 后的边界和亮结构变化，也说明
ratio 中仍有可见场景结构。表格数值和定量 ROI 选择协议保留。
参考文献换栏位置设为第 17 条，以平衡当前第 7 页的两栏；其他图完成后，
可按最终分页重新调整该设置。

## 重建

依赖：Python、Pillow、reportlab、Poppler 的 pdftoppm 和 Arial 字体。
从项目根目录运行：

```powershell
python output/pdf/ICSPS2026_Fig5/build_fig5.py --install-dir docs/ICSPS2026_LaTeX/figures
```

从 `docs/ICSPS2026_LaTeX` 用 Tectonic 或 pdfLaTeX 编译论文。
上传 Overleaf 时保留 `figures` 文件夹；图 4 和图 5 均已包含在交付包中。
