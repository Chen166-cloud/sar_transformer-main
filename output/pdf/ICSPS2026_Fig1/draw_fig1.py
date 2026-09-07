"""Editable, publication-size Fig. 1; no model or experimental data are changed.

Run: python output/pdf/ICSPS2026_Fig1/draw_fig1.py
Requires matplotlib (validated with 3.8.4). All positions are PDF points.
The bundled IEEEtran.cls declares textwidth=43pc, i.e. 516 TeX points.
"""
from pathlib import Path
import argparse
import json
import hashlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
from matplotlib.path import Path as MplPath
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
W = 516 / 72.27 * 72  # 43 pc, converted from TeX pt to PDF bp
H = 432
INK = "#242B32"
GRAY = ("#E9ECEF", "#68727D")
BLUE = ("#E7F1FA", "#27669A")
ORANGE = ("#FFF0D9", "#A8620C")
PURPLE = ("#F0EAF7", "#795798")
GREEN = ("#E6F3EE", "#267461")
WHITE = ("#FFFFFF", "#66717B")


def make_figure(output_dir, font="Arial", dpi=600):
    font_path = font_manager.findfont(font, fallback_to_default=False)
    plt.rcParams.update({
        "font.family": font, "font.size": 8.5,
        "svg.fonttype": "none", "svg.hashsalt": "icsps2026-fig1",
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "mathtext.fontset": "custom", "mathtext.rm": font,
        "mathtext.it": font + ":italic", "mathtext.bf": font + ":bold",
        "axes.unicode_minus": False,
    })
    fig = plt.figure(figsize=(W / 72, H / 72), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, W), ylim=(H, 0))
    ax.axis("off")
    text_boxes = []

    def label(x, y, text, size=8.5, color=INK, weight="normal", ha="center"):
        t = ax.text(x, y, text, fontsize=size, color=color, fontweight=weight,
                    ha=ha, va="center", linespacing=1.2, zorder=6)
        return t

    def box(x, y, w, h, text="", style=WHITE, size=8.5, weight="normal", radius=2.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle=f"round,pad=0,rounding_size={radius}",
                     facecolor=style[0], edgecolor=style[1], linewidth=.75, zorder=3))
        if text:
            t = label(x+w/2, y+h/2, text, size=size, weight=weight)
            text_boxes.append((t, (x, y, w, h), text))

    def arrow(points, color=INK, lw=.85, dashed=False, head=True):
        verts = [(float(x), float(y)) for x, y in points]
        path = MplPath(verts, [MplPath.MOVETO] + [MplPath.LINETO]*(len(verts)-1))
        ax.add_patch(FancyArrowPatch(path=path, arrowstyle="-|>" if head else "-",
                     mutation_scale=6, linewidth=lw, color=color,
                     linestyle=(0, (3, 2)) if dashed else "solid",
                     capstyle="round", joinstyle="round", zorder=2))

    def dot(x, y, color=INK):
        ax.plot(x, y, "o", ms=2.2, color=color, zorder=4)

    # Attribution legend: operator origin, rather than optimization status.
    label(15, 11, "(a) Representation-consistent reconstruction", 10, weight="bold", ha="left")
    for x, style, text in [
        (15, GRAY, "Inherited"), (104, ORANGE, "Fourier"),
        (187, BLUE, "Skip fusion"), (282, PURPLE, "Guidance"),
        (376, GREEN, "Representation"),
    ]:
        ax.add_patch(Rectangle((x, 25), 10, 8, facecolor=style[0], edgecolor=style[1], lw=.65))
        label(x+15, 29, text, 8.2, ha="left")

    # Same normalized log input is sent to both encoder and guidance branch.
    box(15, 48, 80, 32, "Noisy intensity $Y$\n$256\\times256\\times1$", WHITE)
    box(113, 48, 112, 32, "Bounded log $T_{\\alpha}$\n$\\alpha=10$", GREEN)
    box(309, 48, 173, 32, "Guidance-map generator\n$G$: single-channel latent map", PURPLE)
    arrow([(95,64),(113,64)])
    arrow([(225,64),(309,64)], PURPLE[1])
    dot(239,64, PURPLE[1])
    # Branch originates after T_alpha, not before it.
    arrow([(239,64),(239,89),(59,89),(59,100)])
    label(386, 91, "Bilinear resize at each decoder scale", 8, color=PURPLE[1])

    xs = [15, 113, 211, 309, 407]
    enc_shapes = [(128,32),(64,64),(32,128),(16,320),(8,512)]
    for i,(x,(s,c)) in enumerate(zip(xs,enc_shapes),1):
        box(x,100,88,33, f"Encoder $E_{i}$\n${s}\\times{s}\\times{c}$", GRAY)
        if i < 5:
            arrow([(x+88,116.5),(xs[i],116.5)])
    for i in range(4):
        start=xs[i]+44
        target=xs[i+1]+18
        lane = 190 if i == 3 else 167
        arrow([(start,133),(start,lane),(target,lane),(target,199)],BLUE[1])
        label((start+target)/2,lane-7,"skip",8,color=BLUE[1])

    box(407,145,88,39,"Local-spectral\nbottleneck (1 FDR)\n$8\\times8\\times512$",ORANGE,8.2)
    arrow([(451,133),(451,145)])
    arrow([(451,184),(451,199)])

    # Decoder cards are stage-level abstractions; the exact common order is
    # expanded below. Skip arrows terminate at a stage's fusion input.
    dec_shapes=[(256,16),(128,32),(64,64),(32,128),(16,320)]
    for i,(x,(s,c)) in enumerate(zip(xs,dec_shapes)):
        box(x,199,88,45,style=WHITE)
        label(x+44,210,f"Decoder $D_{i}$",8.5,weight="bold")
        label(x+44,222,f"${s}\\times{s}\\times{c}$",8.3)
        for off,style,txt in [(0,ORANGE,"FDR"),(44,PURPLE,"G")]:
            ax.add_patch(Rectangle((x+off+.4,232),43.2,11.5,fc=style[0],ec="none",zorder=4))
            label(x+off+22,238,txt,8,color=style[1])
        if i < 4:
            arrow([(xs[i+1],213),(x+88,213)])
    # Guidance is shared across five scales; FDR parameters are stage-specific.
    arrow([(482,64),(505,64),(505,255),(81,255)],PURPLE[1],dashed=True,head=False)
    for x in xs:
        arrow([(x+66,255),(x+66,244)],PURPLE[1],dashed=True)
        dot(x+66,255,PURPLE[1])

    # The inherited operators in a decoder stage remain gray.
    label(15,279,"$D_4$-$D_1$",8.7,weight="bold",ha="left")
    stage_boxes = [
        (62,52,"Up $\\times2$",GRAY), (125,95,"Concat + fusion",BLUE),
        (231,43,"FDR",ORANGE), (285,88,"Residual block",GRAY),
        (384,111,"Guidance modulation",PURPLE),
    ]
    for k,(x,w,txt,style) in enumerate(stage_boxes):
        box(x,269,w,20,txt,style,8.2)
        if k:
            px,pw,*_=stage_boxes[k-1]
            arrow([(px+pw,279),(x,279)])
    label(15,301,"$D_0$: same order, without encoder skip or fusion.",8,ha="left")
    label(495,301,"Shapes: $H\\times W\\times C$",8,ha="right")

    # Final projection uses stride 1 (not another spatial upsampling).
    box(15,317,102,32,"Projection + conv\n$16\\rightarrow8\\rightarrow1$ channels",GRAY,8.2)
    box(130,317,55,32,"Sigmoid\n$\\widehat{Z}$",GREEN,8.4)
    box(204,317,79,32,"Exact inverse\n$T_{\\alpha}^{-1}$",GREEN,8.5)
    box(313,317,110,32,"Intensity compensation\n$\\widetilde{X}+\\gamma_c F_{\\rm comp}([Y,\\widetilde{X}])$",GREEN,8.1)
    box(441,317,54,32,"Clip $[0,1]$\n$\\widehat{X}$",GREEN,8.4)
    arrow([(15,213),(9,213),(9,333),(15,333)])
    for a,b in [(117,130),(185,204),(283,313),(423,441)]:
        arrow([(a,333),(b,333)])
    label(298,322,"$\\widetilde{X}$",8.5)
    # The bypass carries the bounded original intensity, not a second network.
    arrow([(40,80),(3,80),(3,358),(368,358),(368,349)],GREEN[1])
    label(140,365,"Original intensity $Y$",8.1,color=GREEN[1])

    # A separate training-only panel. This is the same complete model f_theta,
    # including the masked input used by its intensity compensator.
    ax.plot([15,495],[371,371],color="#C9CDD2",lw=.6)
    label(15,382,"(b) Masked post-adaptation (training only)",9.3,weight="bold",ha="left")
    box(15,394,58,26,"Real SAR\n$Y$",WHITE,8.1)
    box(88,394,105,26,"20% Bernoulli mask\n$Y_m=Y\\odot(1-M)$",PURPLE,8.1)
    box(208,394,129,26,"Same model $f_{\\theta}(Y_m)$\nEncoder frozen",WHITE,8.1)
    box(352,394,143,26,"Masked $L_1$ to $Y$ + $10^{-3}$ TV\nUpdate reconstruction modules",PURPLE,8.1)
    for a,b in [(73,88),(193,208),(337,352)]:
        arrow([(a,407),(b,407)],PURPLE[1],dashed=True)
    label(495,382,"Inference: no mask",8.2,ha="right",color=PURPLE[1])

    output_dir.mkdir(parents=True,exist_ok=True)
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    overflow=[]
    for t,(x,y,w,h),txt in text_boxes:
        bb=t.get_window_extent(renderer).transformed(ax.transData.inverted())
        if bb.x0<x+.6 or bb.x1>x+w-.6 or min(bb.y0,bb.y1)<y+.4 or max(bb.y0,bb.y1)>y+h-.4:
            overflow.append(txt)
    if overflow:
        raise RuntimeError("Text does not fit its module: "+repr(overflow))
    basename=output_dir/"fig1_overall_architecture"
    for ext in ("svg","pdf","png"):
        kwargs={"dpi":dpi,"facecolor":"white"}
        if ext=="pdf":
            kwargs["metadata"]={"Title":"Fig. 1. Representation-consistent Fourier refinement",
                                "Author":"", "Creator":"draw_fig1.py / Matplotlib",
                                "CreationDate":None,"ModDate":None}
        if ext=="svg":
            kwargs["metadata"]={"Date":None}
        fig.savefig(basename.with_suffix("."+ext),**kwargs)
    fig.savefig(output_dir/"fig1_actual_size_150dpi.png",dpi=150,facecolor="white")
    source_paths=["transform_main.py","arch/trans_basenetworks.py","model_registry.py",
                  "ablation_config.py","numeric_domain.py","train_icsps2026_ams.py",
                  "configs/icsps2026_frozen_v2.json","docs/ICSPS2026_LaTeX/IEEEtran.cls",
                  "docs/ICSPS2026_LaTeX/ICSPS2026_paper.tex"]
    manifest={"figure":"Fig. 1","width_mm":W/72*25.4,"height_mm":H/72*25.4,
              "width_tex_pt":516,"width_pdf_pt":W,"minimum_base_font_pt":8,
              "font":font,"font_file_used":font_path,"png_dpi":dpi,
              "line_width_pt":{"module":.75,"arrow":.85},
              "matplotlib_version":matplotlib.__version__,"boxed_text_overflows":overflow,
              "data":"Architecture only; no experimental image or metric is plotted.",
              "source_sha256":{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in source_paths}}
    (output_dir/"figure_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    plt.close(fig)
    print(json.dumps({"output_dir":str(output_dir),"width_mm":manifest["width_mm"],
                      "height_mm":manifest["height_mm"],"text_overflows":overflow}))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=HERE)
    parser.add_argument("--font",default="Arial",help="Installed font family; use Liberation Sans on Linux.")
    parser.add_argument("--dpi",type=int,default=600)
    args=parser.parse_args()
    make_figure(args.output_dir,args.font,args.dpi)
