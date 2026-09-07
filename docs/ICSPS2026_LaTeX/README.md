# ICSPS 2026 LaTeX manuscript with Figures 1-5 and scene-wise results

## Current manuscript with scene-wise experiments

Main manuscript: `ICSPS2026_paper.tex`.
**Current compiled PDF: `ICSPS2026_paper_with_average.pdf` (7 pages).**
The existing `build.ps1` / Tectonic workflow has compiled the manuscript with both scene-wise tables, all source Average values, and their English analysis from `../实验结果表.md`. The canonical `ICSPS2026_paper.pdf` is currently locked by a viewer and retains the preceding version without the real-scene Average rows, so the updated build is saved as `ICSPS2026_paper_with_average.pdf`. Use this latest PDF; the named layout copies are historical versions. A later `build.ps1` run can update the canonical PDF once the viewer releases it.

- Section IV-D and Table II add UCMerced agricultural, buildings, and residential results (100 samples per group), including the reported average of 29.0909 dB / 0.7757 for Ours-base.
- Section IV-G and Table V add real-SAR homogeneous, structural, and texture results (20 patches per group), including the source Average rows for 60 patches, comparing Ours-base and Ours+AMS with ENL, M-index, and EPI.
- The conclusion now reflects both scene evaluations. Their summaries remain separate from the four-look UCM-21 macro benchmark and the 592-patch real-SAR benchmark; the fixed-ROI description is explicitly scoped to the latter.
- All included scene values retain the source precision. Original experiment tables, equations, figures, and bibliography entries are unchanged.

As confirmed by the author, `../实验结果表.md` is authoritative for all scene and Average values. Table V reproduces both Average rows exactly: Ours-base ENL/M-index/EPI = 597.0446/1.740321/0.372438 and Ours+AMS = 72.7568/1.454366/0.798884, with Noisy ENL = 23.7477 for 60 patches. The table and discussion incorporate the author's corrected source values. The source Markdown has not been edited by this manuscript update. The supplied scene summaries do not specify their look aggregation or establish that they use the main benchmark's fixed-ROI protocol, so the text does not assume either.

| Figure | Current placement | PDF page |
|---|---|---|
| 1 | Original full-width vector PDF, double column | 2 |
| 2 | Original vector PDF, safely trimmed top/bottom, 0.94 column width | 3 |
| 3 | Original vector PDF, safely trimmed top/bottom, 0.98 column width | 4 |
| 4 | Eight independent PNGs, **2 rows by 4 columns within one column** | 5, left column |
| 5 | Eight independent PNGs, **2 rows by 4 columns within one column** | 6, right column |

Each comparison panel is proportionally scaled to 0.24 column width (approximately 21.257 mm square); the original 256-by-256 pixels are unchanged. Two-line labels remain 8 pt. All eight panels, both ratio images, and Figure 5(a)'s red inspection box are retained.

Table I remains across both columns at the bottom of page 4. Tables II (UCM scenes) and III (ablation) are in the right column of page 5; Tables IV (592 real patches) and V (real scenes) are in the left and right columns of page 6, respectively. The new tables use the existing IEEEtran, booktabs, and 8 pt table style.

REFERENCES starts in the left column of page 7. IEEEtran's native `\IEEEtriggeratref{3}` balances the final columns after the added content. Template geometry, global type size, line spacing, and reference numbering are unchanged.

Build using `build.ps1` as documented below. Final checks have no errors, undefined references, missing citations, Overfull warnings, or Underfull hbox warnings; the existing Underfull vbox remains on page 4. The rendered experiment pages and final bibliography were visually checked. Scene-update verification is saved under `../../output/pdf/ICSPS2026_SceneResults/`; the earlier layout reports describe historical versions.

The previous `ICSPS2026_paper_layout_4x2.pdf` and `LAYOUT_4X2_REPORT.md` are historical files, superseded by this 2-row by 4-column version.

## Original double-column 2-by-4 layout record (historical)


The record below describes the earlier PDF, not the current 4-by-2 layout.

Main manuscript: `ICSPS2026_paper.tex`
Compiled manuscript: `ICSPS2026_paper.pdf` (7 pages, with the updated title and corrected Figure 3 target arrow). The file lock has been released and the canonical PDF has been updated successfully. `ICSPS2026_paper_arrowfix.pdf` retains the previous-title version.

The directory includes the exact `IEEEtran.cls` distributed in the official
ICSPS 2026 LaTeX package. `official_conference_101719.tex` is retained only as
an unedited template reference; it is not the manuscript entry point.

Compile locally with:

```bash
tectonic ICSPS2026_paper.tex
```

On Windows, the repository also includes a build wrapper:

```powershell
.\build.ps1
# Or provide an explicit compiler path:
.\build.ps1 -TectonicPath 'C:\path\to\tectonic.exe'
```

The wrapper checks PATH and then the repository's existing
`tmp/latex_runtime/tectonic.exe`. It keeps logs and auxiliary files under
`tmp/pdfs/paper_fig123/build` and updates the PDF beside the manuscript.
The verified compiler is Tectonic 0.17.0. The T1 font encoding is loaded
before IEEEtran initializes Times to avoid XeTeX/Tectonic font fallback.

For Overleaf, upload this directory and set `ICSPS2026_paper.tex` as the main
document with the pdfLaTeX compiler. Include the `figures` subdirectory when
uploading. Figure 4 is integrated as `figures/fig4_buildings.pdf`: a 2-by-4
comparison of one buildings scene, with the Ours ratio image in panel (h).
Its caption and the Synthetic Comparison discussion describe this single
scene. Figure 5 is integrated as `figures/fig5_real_sar.pdf`: one real-SAR
scene in the same 2-by-4 layout, with Ours+AMS ratio in panel (h). The original
red annotation in Noisy is retained. Its caption and the Real-SAR Adaptation
discussion describe the single scene and the visible effect of AMS.

Figures 1-3 now use their delivered vector PDFs at natural size. All five
figures are local to this directory; no `../../output` graphics path is
needed for compilation. IEEEtran controls captions and numbering, and no
caption-style or page-geometry overrides have been added.

| Figure | File in `figures/` | Width | Verified page |
|---|---|---|---|
| 1 | `fig1_overall_architecture.pdf` | text width, 181.353 mm | 2 |
| 2 | `fig2_fdr_block.pdf` | column width, 88.568 mm | 3 |
| 3 | `fig3_representation_ams.pdf` | column width, 88.568 mm | 4 |
| 4 | `fig4_buildings.pdf` | text width, 181.353 mm | 5 |
| 5 | `fig5_real_sar.pdf` | text width, 181.353 mm | 7 |

Figures 1 and 3 have been compactly redrawn from their existing editable
Matplotlib sources. Copies of `draw_fig1.py`, `draw_fig3.py`, and both editable
SVGs are included in `figures/` beside the PDFs actually used by the manuscript.
Figure 2 is unchanged. Figure 1 is a single architecture diagram; AMS appears
only in Figure 3(b). The old reference break at entry 15 was removed because
it left an unnecessary final page after this layout change.

The final placed Figure 1 measures 181.371 × 92.437 mm (page 2); Figure 3
measures 88.573 × 133.358 mm (page 4), excluding captions. Labels are mainly
9 pt / 8.8 pt, with minimum ordinary labels 8.5 pt / 8.2 pt respectively.
The slight difference from native widths comes from Tectonic's PDF placement
rounding; the manuscript still uses only `width=\textwidth` / `\columnwidth`.

To regenerate the editable vector exports from the repository root:

```powershell
python output/pdf/ICSPS2026_Fig1/draw_fig1.py
python output/pdf/ICSPS2026_Fig3/draw_fig3.py
Copy-Item output/pdf/ICSPS2026_Fig1/fig1_overall_architecture.pdf docs/ICSPS2026_LaTeX/figures/
Copy-Item output/pdf/ICSPS2026_Fig3/fig3_representation_ams.pdf docs/ICSPS2026_LaTeX/figures/
.\docs\ICSPS2026_LaTeX\build.ps1
```

The script copies under `figures/` are identical and also accept `--output-dir`.
Drawing requires Matplotlib and fontTools with Arial installed (or use the
`--font` option). The verified drawing runtime is Matplotlib 3.8.4.

The current baseline, final page renders, strict preservation checks and report
are in `../../output/pdf/ICSPS2026_LayoutRefinement/`. See also
`LAYOUT_REFINEMENT_REPORT.md`. The original integrated QA and prior figure proof
PDFs remain historical records, not checks of the current layout. The final
build has one Underfull vbox warning on page 5, with no visible serious blank
region; no overfull, missing references, missing images or font warnings remain.
The recorded AMS denominator discrepancy remains unchanged. All formulas,
tables, bibliography, model code, template settings and Figure 2/4/5 contents
are preserved.

Figure 4 sources, a reproducible composition script, a PNG preview, and the
source-pixel verification record are in `../../output/pdf/ICSPS2026_Fig4`.
The original 256-by-256 images are embedded losslessly; exporting the composite
preview at 600 dpi does not increase their native information content.

Figure 5 sources, composition script, caption, and previews are in
`../../output/pdf/ICSPS2026_Fig5`. Labels follow the author-confirmed renamed
filenames. The work covers composition and manuscript layout; no local model
identity or historical-output validation is performed for Figure 5.
