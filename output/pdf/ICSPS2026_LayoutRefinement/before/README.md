# ICSPS 2026 LaTeX manuscript with Figures 1-5

Main manuscript: `ICSPS2026_paper.tex`
Compiled manuscript: `ICSPS2026_paper.pdf` (8 pages in the verified build).

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
| 4 | `fig4_buildings.pdf` | text width, 181.353 mm | 6 |
| 5 | `fig5_real_sar.pdf` | text width, 181.353 mm | 7 |

Figures 1-3 retain their original 8 pt or larger base labels and embedded
fonts. Their source code, SVGs, captions, and model/source audits are in
`../../output/pdf/ICSPS2026_Fig1`, `ICSPS2026_Fig2`, and `ICSPS2026_Fig3`.
The original figure placeholders and their unused macro have been removed.
The reference-column break is set at entry 15 for the current eight-page
layout; review that setting after changing text or figure sizes.

Full-paper checks, page previews, and the QA script are in
`../../output/pdf/ICSPS2026_Integrated`. The eight-page result is a layout
check, not verification of a conference page allowance. No model code,
experimental results, table values, or existing Figures 4-5 were changed
as part of integrating Figures 1-3. The previously documented AMS loss
denominator difference is unchanged; this task does not revise that formula.

Figure 4 sources, a reproducible composition script, a PNG preview, and the
source-pixel verification record are in `../../output/pdf/ICSPS2026_Fig4`.
The original 256-by-256 images are embedded losslessly; exporting the composite
preview at 600 dpi does not increase their native information content.

Figure 5 sources, composition script, caption, and previews are in
`../../output/pdf/ICSPS2026_Fig5`. Labels follow the author-confirmed renamed
filenames. The work covers composition and manuscript layout; no local model
identity or historical-output validation is performed for Figure 5.
