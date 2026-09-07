# Figure 1: Trans-SAR-derived network overview

## Delivered files

- `fig1_trans_sar_architecture.svg`: editable vector artwork with live text and no embedded raster image.
- `../../pdf/fig1_trans_sar_architecture.pdf`: vector PDF at the IEEE two-column width, 7.16 × 2.72 in.
- `fig1_trans_sar_architecture.png`: 600-dpi preview, 4296 × 1632 px.
- `../../../scripts/icsps2026/plot_fig1_trans_sar_architecture.py`: deterministic Matplotlib drawing source with a preflight source-code check.
- `../../../scripts/icsps2026/requirements-figures.txt`: pinned plotting dependency.

No experimental measurements or checkpoint outputs are used in this architecture figure.

## Reproduce

From the repository root:

```bash
python3 -m venv .venv-figures
.venv-figures/bin/python -m pip install -r scripts/icsps2026/requirements-figures.txt
.venv-figures/bin/python scripts/icsps2026/plot_fig1_trans_sar_architecture.py
```

The script verifies identifying fragments in the current model, registry, formal variant, numerical-domain, and AMS sources before it draws. Use `--skip-source-check` only when rendering a historical checkout intentionally.

## Audited model path and source mapping

The figure represents the formal `ours/full` registry path, namely
`TransSARV2_DualFreqNG_Bottle(variant="full", numeric_domain="intensity_v1")`.
The paper-facing name remains **Trans-SAR**; code identifiers are retained here only for reproducibility.

| Figure element | Code or paper evidence |
|---|---|
| Formal model entry | `model_registry.py:348-361`; full switches in `ablation_config.py:19-28` |
| Bounded log and exact inverse, α=10 | `numeric_domain.py:16-19,94-109`; wrapper order in `transform_main.py:2141-2166` |
| Five-stage encoder and nominal 256-pixel feature sizes | `transform_main.py:234-253,345-394,2021-2037` |
| Bottleneck local/FDR refinement | `transform_main.py:1406-1501,2039-2069` |
| Four same-scale concat/fusion skips and five-stage decoder | `transform_main.py:1655-1777` |
| Five decoder FDR blocks | `transform_main.py:1685-1690,1728-1772` |
| Guidance generator and modulation | `transform_main.py:1503-1519,1543-1566,1692-1697` |
| Projection, 3×3 head, and Sigmoid for the formal intensity path | `transform_main.py:1675,1774-1775,2050-2056,2120-2132` |
| Intensity-domain compensator | `transform_main.py:1522-1541,2157-2166` |
| AMS freeze boundary | `train_icsps2026_ams.py:301-313`; manuscript `docs/ICSPS2026_LaTeX/ICSPS2026_paper.tex:155` |
| Inherited versus extended components | `docs/ICSPS2026_SAR去斑网络设计与实验方案.md:130-154` |

The nominal feature shapes shown for a `1 × 256 × 256` input are:

- Encoder: `E1 32@128²`, `E2 64@64²`, `E3 128@32²`, `E4 320@16²`, `E5 512@8²`.
- Bottleneck: `512@8²`, containing FDR 1/6.
- Decoder: `D4 320@16²`, `D3 128@32²`, `D2 64@64²`, `D1 32@128²`, `D0 16@256²`, containing FDR 2/6 through 6/6.
- Head feature: a stride-1 transposed convolution produces `8@256²`; the 3×3 reconstruction convolution produces one channel.
- Guidance map: `1@256²`, resized to each of the five decoder resolutions.

## Consistency findings

The current manuscript's architectural description, feature sizes, six FDR positions, guidance input, inverse-before-compensation order, and AMS freeze boundary agree with the formal source path. The following distinctions are recorded so the figure is not interpreted beyond the code:

1. The design document shows a direct constructor with `ablation="full"`, whereas the formal registry passes `variant="full"`. For the full model these resolve to the same enabled components; the figure follows the registry entry actually used by the formal runners.
2. The repository class literally named `TransSAR` is a different four-stage historical model. The five-stage backbone used here is the repository's `TransSARV2` implementation. The visible figure uses the requested paper name **Trans-SAR**, while this provenance note preserves the exact class identity.
3. The manuscript summarizes the output as a “sigmoid head.” The code contains a stride-1 `16→8` projection, a 3×3 `8→1` convolution, and then Sigmoid. The figure exposes all three operations without changing the manuscript's claim.
4. Gray denotes inheritance, not freezing. Only the encoder is frozen during AMS; inherited decoder operators remain trainable. The dashed AMS boundary therefore uses a separate visual encoding.

No local formal checkpoint is required to draw this structural figure. A dynamic PyTorch hook check could not be run in the current lightweight plotting environment because PyTorch is not installed; the dimensions were instead cross-checked from the locked 256×256 configuration, convolution strides, decoder source comments, and exact forward sequence. No uncertain dimension is included.

## Recommended placement and LaTeX

Place the figure at the top of the Methods page, immediately after the overview paragraph in **Trans-SAR-Derived Reconstruction Path** and before the detailed FDR subsection. Use the two-column `figure*` environment:

```latex
\begin{figure*}[!t]
  \centering
  \includegraphics[width=\textwidth]{figures/fig1_trans_sar_architecture.pdf}
  \caption{Overview of the Trans-SAR-derived despeckling network. The normalized log-intensity image is encoded at five resolutions and reconstructed through four same-scale skip fusions and a five-stage progressive decoder. FDR is applied once at the $8\times8$ bottleneck and at all five decoder resolutions, while a latent guidance map modulates each decoder stage. The sigmoid log-domain estimate is analytically inverted before intensity-domain residual compensation. Gray components are inherited from Trans-SAR, and colored components denote our extensions. During AMS, only the encoder is frozen.}
  \label{fig:overview}
\end{figure*}
```

## Visual QA

- The plotting source completed its architecture preflight check.
- The PDF is a one-page vector file at exactly 515.52 × 195.84 pt (7.16 × 2.72 in).
- The SVG contains live text and no `<image>` raster payload.
- The 600-dpi PNG and a fresh 240-dpi rasterization of the PDF were opened and visually checked for clipped text, obscured arrows, inconsistent stage order, and excess whitespace.
- The PDF was also embedded at `\textwidth` in the repository's official `IEEEtran.cls` with Tectonic; the compilation reported no overfull boxes, and the rendered conference page was visually checked.
