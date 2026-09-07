#!/usr/bin/env python3
"""Draw the ICSPS 2026 Trans-SAR-derived network overview (Fig. 1).

The diagram is intentionally generated from an audited, paper-facing summary
of the ``ours/full`` configuration.  It does not import PyTorch, so the figure
can be reproduced in a lightweight plotting environment.  Before drawing, the
script checks identifying source fragments to guard against silent model drift.

Outputs
-------
* SVG with live text for editing
* vector PDF at IEEE two-column width
* 600-dpi PNG preview
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch


# IEEEtran's two-column text block is approximately 7.16 in wide.  The compact
# height leaves room for the remaining four planned figures in a 6--7 page paper.
FIGURE_WIDTH_IN = 7.16
FIGURE_HEIGHT_IN = 2.72

COLORS = {
    "ink": "#263238",
    "muted": "#5E6A71",
    "line": "#4F5B62",
    "inherited_fill": "#E7E9EC",
    "inherited_edge": "#69737A",
    "frequency_fill": "#DCEBFA",
    "frequency_edge": "#2F6FAE",
    "fusion_fill": "#EEE3F8",
    "fusion_edge": "#7851A9",
    "guidance_fill": "#DDF2E5",
    "guidance_edge": "#2E7D5A",
    "numeric_fill": "#FBE8D2",
    "numeric_edge": "#C66B1E",
    "neutral_fill": "#FAFBFC",
    "neutral_edge": "#8A959C",
    "white": "#FFFFFF",
}


ENCODER_STAGES = (
    ("E1", "32 × 128²"),
    ("E2", "64 × 64²"),
    ("E3", "128 × 32²"),
    ("E4", "320 × 16²"),
    ("E5", "512 × 8²"),
)

DECODER_STAGES = (
    # name, feature size, FDR index, has a same-scale encoder skip
    ("D4", "320 × 16²", "FDR 2/6", True),
    ("D3", "128 × 32²", "FDR 3/6", True),
    ("D2", "64 × 64²", "FDR 4/6", True),
    ("D1", "32 × 128²", "FDR 5/6", True),
    ("D0", "16 × 256²", "FDR 6/6", False),
)


def verify_sources(repo_root: Path) -> None:
    """Fail early if the audited architecture identifiers have drifted."""

    requirements = {
        "model_registry.py": (
            "TransSARV2_DualFreqNG_Bottle(",
            "variant=ours_variant",
        ),
        "ablation_config.py": (
            '"representation": "log"',
            '"compensation": True',
            '"decoder_fdr": True',
            '"bottleneck_fdr": True',
        ),
        "numeric_domain.py": (
            "LOG_ALPHA = 10.0",
            "def log_transform_01_torch",
            "def inverse_log_transform_01_torch",
        ),
        "transform_main.py": (
            "class TransSARV2_DualFreqNG_Bottle",
            "class TransSARV2_FreqNG_Bottle",
            "class convprojection_freq_ng",
            "class BottleneckRefine",
            "class FFTRefineBlock",
            "class NoiseGuidedGate",
            "class NoiseEstimator",
            "embed_dims=[32, 64, 128, 320, 512]",
            "x1[4] = self.bottleneck_refine(x1[4])",
            "clean = self.dual_fusion(x01, branch_clean)",
        ),
        "train_icsps2026_ams.py": (
            "model.log_branch.Tenc.parameters()",
            "parameter.requires_grad = False",
        ),
    }
    problems: list[str] = []
    for relative_path, fragments in requirements.items():
        path = repo_root / relative_path
        if not path.is_file():
            problems.append(f"missing {relative_path}")
            continue
        text = path.read_text(encoding="utf-8")
        for fragment in fragments:
            if fragment not in text:
                problems.append(f"{relative_path}: missing {fragment!r}")
    if problems:
        joined = "\n  - ".join(problems)
        raise RuntimeError(f"Architecture source check failed:\n  - {joined}")


def rounded_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    face: str,
    edge: str,
    linewidth: float = 0.85,
    radius: float = 0.75,
    zorder: int = 3,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.12,rounding_size={radius}",
        linewidth=linewidth,
        edgecolor=edge,
        facecolor=face,
        joinstyle="round",
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def box_text(
    ax: plt.Axes,
    x: float,
    y: float,
    lines: Sequence[str],
    *,
    fontsize: float = 6.6,
    color: str | None = None,
    weights: Sequence[str] | None = None,
    line_gap: float = 1.55,
    zorder: int = 5,
) -> None:
    color = color or COLORS["ink"]
    weights = weights or tuple("normal" for _ in lines)
    center_offset = (len(lines) - 1) * line_gap / 2
    for idx, (line, weight) in enumerate(zip(lines, weights)):
        ax.text(
            x,
            y + center_offset - idx * line_gap,
            line,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight=weight,
            color=color,
            zorder=zorder,
        )


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str | None = None,
    linewidth: float = 0.9,
    style: str = "-|>",
    connectionstyle: str = "arc3,rad=0",
    linestyle: str = "-",
    mutation_scale: float = 7.5,
    zorder: int = 2,
) -> FancyArrowPatch:
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=mutation_scale,
        linewidth=linewidth,
        color=color or COLORS["line"],
        connectionstyle=connectionstyle,
        linestyle=linestyle,
        shrinkA=0,
        shrinkB=0,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def routed_arrow(
    ax: plt.Axes,
    vertices: Iterable[tuple[float, float]],
    *,
    color: str,
    linewidth: float = 0.9,
    linestyle: str = "-",
    zorder: int = 1,
) -> None:
    points = list(vertices)
    path = MplPath(points, [MplPath.MOVETO] + [MplPath.LINETO] * (len(points) - 1))
    ax.add_patch(
        PathPatch(
            path,
            fill=False,
            edgecolor=color,
            linewidth=linewidth,
            linestyle=linestyle,
            capstyle="round",
            joinstyle="round",
            zorder=zorder,
        )
    )
    arrow(ax, points[-2], points[-1], color=color, linewidth=linewidth, zorder=zorder + 0.1)


def draw_decoder_stage(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    name: str,
    size: str,
    fdr_index: str,
    has_fusion: bool,
) -> None:
    rounded_box(
        ax,
        x,
        y,
        w,
        h,
        face=COLORS["neutral_fill"],
        edge=COLORS["neutral_edge"],
    )
    ax.text(
        x + w / 2,
        y + h - 1.35,
        f"{name}  ·  {size}",
        ha="center",
        va="center",
        fontsize=6.35,
        fontweight="bold",
        color=COLORS["ink"],
        zorder=5,
    )
    ax.text(
        x + w / 2,
        y + h - 3.10,
        fdr_index,
        ha="center",
        va="center",
        fontsize=5.85,
        color=COLORS["frequency_edge"],
        fontweight="bold",
        zorder=5,
    )

    # The ordered color rail encodes the exact mixed inherited/extended stage.
    rail_x = x + 0.45
    rail_y = y + 0.43
    rail_w = w - 0.90
    rail_h = 1.13
    sequence = [
        ("Up", "inherited_fill", "inherited_edge"),
        *(([("Fuse", "fusion_fill", "fusion_edge")]) if has_fusion else []),
        ("FDR", "frequency_fill", "frequency_edge"),
        ("RB", "inherited_fill", "inherited_edge"),
        ("GM", "guidance_fill", "guidance_edge"),
    ]
    gap = 0.15
    segment_w = (rail_w - gap * (len(sequence) - 1)) / len(sequence)
    for idx, (label, fill_key, edge_key) in enumerate(sequence):
        sx = rail_x + idx * (segment_w + gap)
        ax.add_patch(
            Rectangle(
                (sx, rail_y),
                segment_w,
                rail_h,
                facecolor=COLORS[fill_key],
                edgecolor=COLORS[edge_key],
                linewidth=0.55,
                zorder=4,
            )
        )
        ax.text(
            sx + segment_w / 2,
            rail_y + rail_h / 2,
            label,
            ha="center",
            va="center",
            fontsize=5.25,
            color=COLORS["ink"],
            zorder=5,
        )


def draw_figure() -> plt.Figure:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.0,
            "axes.linewidth": 0.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)
    ax.axis("off")

    top_y, top_h = 31.45, 5.55

    # Input and representation path.
    input_spec = [
        (0.8, 6.7, COLORS["neutral_fill"], COLORS["neutral_edge"], ("Noisy SAR", "intensity y")),
        (8.65, 6.3, COLORS["numeric_fill"], COLORS["numeric_edge"], ("Clamp", "[0, 1]")),
        (16.15, 7.5, COLORS["numeric_fill"], COLORS["numeric_edge"], ("Bounded log", "Tα · α=10")),
    ]
    for x, w, face, edge, label in input_spec:
        rounded_box(ax, x, top_y, w, top_h, face=face, edge=edge)
        box_text(ax, x + w / 2, top_y + top_h / 2, label, fontsize=6.15, weights=("bold", "normal"))
    for (x1, w1, *_), (x2, *_rest) in zip(input_spec, input_spec[1:]):
        arrow(ax, (x1 + w1, top_y + top_h / 2), (x2, top_y + top_h / 2))

    # Five-stage inherited encoder.  The separate dashed boundary is the AMS
    # freeze status; gray alone denotes provenance, not trainability.
    encoder_x = [25.55, 38.35, 51.15, 63.95, 76.75]
    encoder_w = [10.85, 10.85, 10.85, 10.85, 9.35]
    for (name, size), x, w in zip(ENCODER_STAGES, encoder_x, encoder_w):
        rounded_box(
            ax,
            x,
            top_y,
            w,
            top_h,
            face=COLORS["inherited_fill"],
            edge=COLORS["inherited_edge"],
        )
        box_text(ax, x + w / 2, top_y + top_h / 2, (name, size), fontsize=6.35, weights=("bold", "normal"))
    arrow(ax, (23.65, top_y + top_h / 2), (encoder_x[0], top_y + top_h / 2))
    for idx in range(len(encoder_x) - 1):
        arrow(
            ax,
            (encoder_x[idx] + encoder_w[idx], top_y + top_h / 2),
            (encoder_x[idx + 1], top_y + top_h / 2),
        )

    freeze_boundary = FancyBboxPatch(
        (24.75, 30.67),
        62.08,
        7.35,
        boxstyle="round,pad=0.18,rounding_size=0.9",
        facecolor="none",
        edgecolor=COLORS["inherited_edge"],
        linewidth=0.8,
        linestyle=(0, (3.0, 2.0)),
        zorder=2.5,
    )
    ax.add_patch(freeze_boundary)
    ax.text(
        25.05,
        38.45,
        "AMS: encoder frozen",
        ha="left",
        va="center",
        fontsize=6.2,
        color=COLORS["muted"],
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5},
    )

    # Colored bottleneck extension.  FDR remains nested because it is not a
    # pure FFT branch: the source block also contains spatial and residual paths.
    bottleneck_x, bottleneck_w = 87.35, 11.85
    rounded_box(
        ax,
        bottleneck_x,
        top_y,
        bottleneck_w,
        top_h,
        face=COLORS["frequency_fill"],
        edge=COLORS["frequency_edge"],
    )
    box_text(
        ax,
        bottleneck_x + bottleneck_w / 2,
        top_y + top_h / 2,
        ("Bottleneck", "Local + FDR 1/6", "512 × 8²"),
        fontsize=5.9,
        weights=("bold", "bold", "normal"),
        line_gap=1.42,
    )
    arrow(
        ax,
        (encoder_x[-1] + encoder_w[-1], top_y + top_h / 2),
        (bottleneck_x, top_y + top_h / 2),
    )

    # Progressive decoder, right-to-left, aligned with four same-scale skips.
    dec_y, dec_h, dec_w = 21.55, 6.55, 11.65
    decoder_positions = {
        "D4": 63.95,
        "D3": 51.15,
        "D2": 38.35,
        "D1": 25.55,
        "D0": 12.75,
    }
    for name, size, fdr, has_fusion in DECODER_STAGES:
        draw_decoder_stage(
            ax,
            decoder_positions[name],
            dec_y,
            dec_w,
            dec_h,
            name,
            size,
            fdr,
            has_fusion,
        )

    arrow(
        ax,
        (bottleneck_x + bottleneck_w / 2, top_y),
        (decoder_positions["D4"] + dec_w, dec_y + dec_h / 2),
        connectionstyle="arc3,rad=-0.24",
        linewidth=1.0,
    )
    decoder_order = ["D4", "D3", "D2", "D1", "D0"]
    for current, following in zip(decoder_order, decoder_order[1:]):
        arrow(
            ax,
            (decoder_positions[current], dec_y + dec_h / 2),
            (decoder_positions[following] + dec_w, dec_y + dec_h / 2),
            linewidth=1.0,
        )

    # Four inherited skip routes terminate at the colored learned-fusion stages.
    skip_pairs = [
        (3, "D4"),
        (2, "D3"),
        (1, "D2"),
        (0, "D1"),
    ]
    for encoder_index, decoder_name in skip_pairs:
        ex = encoder_x[encoder_index] + encoder_w[encoder_index] / 2
        dx = decoder_positions[decoder_name] + dec_w / 2
        arrow(
            ax,
            (ex, top_y),
            (dx, dec_y + dec_h),
            color=COLORS["inherited_edge"],
            linewidth=0.75,
            mutation_scale=6.3,
        )
    ax.text(
        57.55,
        29.25,
        "4 same-scale concat/fusion skips",
        ha="center",
        va="center",
        fontsize=5.8,
        color=COLORS["fusion_edge"],
        fontweight="bold",
    )

    # Guidance branch: the generator reads the same bounded log image, and the
    # resulting 1-channel map is resized independently for all five gates.
    guide_x, guide_y, guide_w, guide_h = 79.9, 12.75, 19.0, 5.5
    rounded_box(
        ax,
        guide_x,
        guide_y,
        guide_w,
        guide_h,
        face=COLORS["guidance_fill"],
        edge=COLORS["guidance_edge"],
    )
    box_text(
        ax,
        guide_x + guide_w / 2,
        guide_y + guide_h / 2,
        ("Guidance-map generator", "1→32→32→32→1 · P: 1 × 256²"),
        fontsize=5.8,
        weights=("bold", "normal"),
    )
    routed_arrow(
        ax,
        [
            (16.15 + 7.5 / 2, top_y),
            (16.15 + 7.5 / 2, 18.95),
            (guide_x - 1.1, 18.95),
            (guide_x - 1.1, guide_y + guide_h / 2),
            (guide_x, guide_y + guide_h / 2),
        ],
        color=COLORS["guidance_edge"],
        linewidth=0.85,
        zorder=1.2,
    )
    ax.text(
        66.2,
        18.78,
        "same bounded-log input z",
        ha="center",
        va="center",
        fontsize=5.7,
        color=COLORS["guidance_edge"],
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.7},
    )

    bus_y = 20.05
    ax.plot(
        [18.55, guide_x + 2.0],
        [bus_y, bus_y],
        color=COLORS["guidance_edge"],
        linewidth=0.82,
        linestyle=(0, (2.1, 1.6)),
        zorder=1.1,
    )
    arrow(
        ax,
        (guide_x + 2.0, guide_y + guide_h),
        (guide_x + 2.0, bus_y),
        color=COLORS["guidance_edge"],
        linewidth=0.82,
        linestyle=(0, (2.1, 1.6)),
        mutation_scale=6.0,
        zorder=1.2,
    )
    for decoder_name in ("D4", "D3", "D2", "D1", "D0"):
        cx = decoder_positions[decoder_name] + dec_w / 2
        arrow(
            ax,
            (cx, bus_y),
            (cx, dec_y),
            color=COLORS["guidance_edge"],
            linewidth=0.74,
            linestyle=(0, (2.1, 1.6)),
            mutation_scale=6.0,
            zorder=1.2,
        )
    ax.text(
        75.8,
        bus_y + 0.45,
        "bilinear resize",
        ha="center",
        va="bottom",
        fontsize=5.55,
        color=COLORS["guidance_edge"],
    )

    # Output-domain path.  The gray projection/convolution is inherited; the
    # sigmoid, inverse transform, and intensity compensation are extensions.
    out_y, out_h = 5.95, 5.3
    output_blocks = [
        (12.8, 12.0, COLORS["inherited_fill"], COLORS["inherited_edge"], ("Projection + Conv", "16→8→1 · 3×3")),
        (26.15, 8.05, COLORS["numeric_fill"], COLORS["numeric_edge"], ("Sigmoid", "log estimate")),
        (35.55, 10.8, COLORS["numeric_fill"], COLORS["numeric_edge"], ("Exact Tα⁻¹", "to intensity")),
        (47.7, 17.0, COLORS["numeric_fill"], COLORS["numeric_edge"], ("Intensity residual", "compensator [y₀₁, x̃]")),
        (66.05, 8.0, COLORS["numeric_fill"], COLORS["numeric_edge"], ("Final", "clamp")),
        (75.4, 11.9, COLORS["neutral_fill"], COLORS["neutral_edge"], ("Restored SAR", "intensity x̂")),
    ]
    for x, w, face, edge, labels in output_blocks:
        rounded_box(ax, x, out_y, w, out_h, face=face, edge=edge)
        box_text(ax, x + w / 2, out_y + out_h / 2, labels, fontsize=5.85, weights=("bold", "normal"))
    for (x1, w1, *_), (x2, *_rest) in zip(output_blocks, output_blocks[1:]):
        arrow(ax, (x1 + w1, out_y + out_h / 2), (x2, out_y + out_h / 2), linewidth=0.9)

    # Route D0 to its inherited final feature projection without overlapping
    # the central guidance arrow.
    routed_arrow(
        ax,
        [
            (decoder_positions["D0"] + 1.1, dec_y),
            (10.2, dec_y - 1.1),
            (10.2, out_y + out_h / 2),
            (12.8, out_y + out_h / 2),
        ],
        color=COLORS["line"],
        linewidth=0.95,
        zorder=2.0,
    )

    # Original normalized intensity bypass into the compensator.  It originates
    # after clamp, not from the unclipped raw input.
    clamp_center = 8.65 + 6.3 / 2
    comp_center = 47.7 + 17.0 / 2
    routed_arrow(
        ax,
        [
            (clamp_center, top_y),
            (5.6, top_y - 1.6),
            (5.6, 3.8),
            (comp_center, 3.8),
            (comp_center, out_y),
        ],
        color=COLORS["numeric_edge"],
        linewidth=0.85,
        zorder=1.0,
    )
    ax.text(
        30.0,
        4.08,
        "normalized intensity bypass y₀₁",
        ha="center",
        va="bottom",
        fontsize=5.65,
        color=COLORS["numeric_edge"],
    )

    # Compact provenance/operator legend.  All colored modules are trainable in
    # AMS; inherited decoder operators remain trainable despite their gray fill.
    legend_y = 1.18
    legend_items = [
        (2.0, "inherited_fill", "inherited_edge", "Inherited Trans-SAR operator"),
        (27.0, "frequency_fill", "frequency_edge", "FDR / bottleneck"),
        (45.5, "fusion_fill", "fusion_edge", "Learned skip fusion"),
        (64.8, "guidance_fill", "guidance_edge", "Guidance modulation"),
        (80.0, "numeric_fill", "numeric_edge", "Log / intensity compensation"),
    ]
    for x, fill_key, edge_key, label in legend_items:
        ax.add_patch(
            Rectangle(
                (x, legend_y - 0.55),
                2.0,
                1.1,
                facecolor=COLORS[fill_key],
                edgecolor=COLORS[edge_key],
                linewidth=0.65,
                zorder=4,
            )
        )
        ax.text(
            x + 2.5,
            legend_y,
            label,
            ha="left",
            va="center",
            fontsize=5.75,
            color=COLORS["ink"],
            zorder=5,
        )
    ax.text(
        98.9,
        38.45,
        "AMS trainable: bottleneck · decoder · guidance · head · compensator",
        ha="right",
        va="center",
        fontsize=6.0,
        color=COLORS["frequency_edge"],
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5},
    )

    fig.subplots_adjust(left=0.003, right=0.997, top=0.986, bottom=0.012)
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    script_path = Path(__file__).resolve()
    default_root = script_path.parents[2]
    parser.add_argument("--repo-root", type=Path, default=default_root)
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=default_root / "output" / "figures" / "icsps2026",
    )
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=default_root / "output" / "pdf",
    )
    parser.add_argument(
        "--skip-source-check",
        action="store_true",
        help="Draw even if audited source identifiers cannot be found.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    if not args.skip_source_check:
        verify_sources(repo_root)

    args.image_dir.mkdir(parents=True, exist_ok=True)
    args.pdf_dir.mkdir(parents=True, exist_ok=True)
    stem = "fig1_trans_sar_architecture"

    figure = draw_figure()
    svg_path = args.image_dir / f"{stem}.svg"
    png_path = args.image_dir / f"{stem}.png"
    pdf_path = args.pdf_dir / f"{stem}.pdf"

    common_metadata = {
        "Title": "Trans-SAR-derived despeckling network overview",
        "Author": "Reproducible figure generated from the audited project source",
        "Subject": "ICSPS 2026 Figure 1",
    }
    figure.savefig(svg_path, format="svg", facecolor="white", metadata={"Title": common_metadata["Title"]})
    figure.savefig(pdf_path, format="pdf", facecolor="white", metadata=common_metadata)
    figure.savefig(png_path, format="png", dpi=600, facecolor="white", metadata={"Title": common_metadata["Title"]})
    plt.close(figure)

    print(f"source check: {'skipped' if args.skip_source_check else 'passed'}")
    print(f"SVG: {svg_path}")
    print(f"PDF: {pdf_path}")
    print(f"PNG: {png_path}")


if __name__ == "__main__":
    main()
