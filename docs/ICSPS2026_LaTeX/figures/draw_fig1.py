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
ROOT = next(p for p in HERE.parents if (p / "transform_main.py").is_file())
W = 516 / 72.27 * 72  # 43 pc, converted from TeX pt to PDF bp
H = 262
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

    # Gray marks inherited operators; white decoder cards contain mixed origins.
    for x, style, txt in [
        (15, GRAY, "Inherited"), (104, ORANGE, "Fourier"),
        (187, BLUE, "Skip fusion"), (282, PURPLE, "Guidance"),
        (376, GREEN, "Representation"),
    ]:
        ax.add_patch(Rectangle((x, 4), 10, 8, facecolor=style[0], edgecolor=style[1], lw=.65))
        label(x+15, 8, txt, 8.5, ha="left")

    box(15,23,90,29,"Intensity $Y$ in $[0,1]$\n$256\\times256\\times1$",WHITE,9)
    box(126,23,112,29,"Bounded log $T_{\\alpha}$\n$\\alpha=10$",GREEN,9)
    box(309,23,186,29,"Shared guidance-map generator\n$G$: single-channel latent map",PURPLE,9)
    arrow([(105,37.5),(126,37.5)])
    arrow([(238,37.5),(309,37.5)],PURPLE[1])
    dot(251,37.5,PURPLE[1])
    arrow([(251,37.5),(251,60),(59,60),(59,69)])

    xs=[15,113,211,309,407]
    for i,(x,c) in enumerate(zip(xs,[32,64,128,320,512]),1):
        box(x,69,88,27,f"Encoder $E_{i}$\n{c} channels",GRAY,9)
        if i<5:
            arrow([(x+88,82.5),(xs[i],82.5)])
    # E1->D1, E2->D2, E3->D3, E4->D4. D0 has no encoder skip.
    for i in range(4):
        start=xs[i]+44
        target=xs[i+1]+17
        lane=132 if i==3 else 115
        arrow([(start,96),(start,lane),(target,lane),(target,139)],BLUE[1])
        label(375 if i==3 else (start+target)/2,lane-7,"skip fusion",8.5,color=BLUE[1])

    box(407,104,88,27,"Local-spectral\nbottleneck: FDR",ORANGE,9)
    arrow([(451,96),(451,104)])
    arrow([(451,131),(451,139)])

    # Each card retains its own FDR and guidance modulation. These compact
    # origin bands identify components, not a separate operation-order strip.
    for i,(x,c) in enumerate(zip(xs,[16,32,64,128,320])):
        box(x,139,88,47,style=WHITE)
        label(x+44,147.5,f"$D_{i}$  /  {c} channels",9,weight="bold")
        ax.add_patch(Rectangle((x+.5,156),87,14,fc=GRAY[0],ec="none",zorder=4))
        label(x+44,163,"Up / residual",8.5,color=GRAY[1])
        for off,style,txt in [(0,ORANGE,"FDR"),(44,PURPLE,"G")]:
            ax.add_patch(Rectangle((x+off+.5,171),43,14.5,fc=style[0],ec="none",zorder=4))
            label(x+off+22,178.5,txt,9,color=style[1])
        if i<4:
            arrow([(xs[i+1],147.5),(x+88,147.5)])
    arrow([(495,37.5),(506,37.5),(506,195),(81,195)],PURPLE[1],dashed=True,head=False)
    for x in xs:
        arrow([(x+66,195),(x+66,186)],PURPLE[1],dashed=True)
        dot(x+66,195,PURPLE[1])
    label(304,203,"$G$: resized at each scale; FDR parameters are independent",8.5,color=PURPLE[1])

    # Two compact output modules; detailed same-domain arithmetic is in Fig. 3.
    # The prediction card keeps the inherited projection gray and sigmoid green.
    box(25,213,161,29,style=WHITE)
    label(105.5,221.5,"Bounded-log prediction head",9)
    ax.add_patch(Rectangle((25.5,228),85,13.5,fc=GRAY[0],ec="none",zorder=4))
    ax.add_patch(Rectangle((110.5,228),75,13.5,fc=GREEN[0],ec="none",zorder=4))
    label(68,235,"Projection",8.5,color=GRAY[1])
    label(148,235,"Sigmoid",8.5,color=GREEN[1])
    box(247,213,152,29,"Intensity reconstruction",GREEN,9)
    box(430,213,65,29,"Intensity $\\widehat{X}$",WHITE,9)
    arrow([(15,147.5),(9,147.5),(9,227.5),(25,227.5)])
    arrow([(186,227.5),(247,227.5)])
    label(216.5,218,"$\\widehat{Z}$",9,color=GREEN[1])
    arrow([(399,227.5),(430,227.5)])
    # The normalized intensity bypass does not come from the log input.
    arrow([(39,52),(3,52),(3,258),(323,258),(323,242)],GREEN[1])
    label(161,249,"Original intensity $Y$",8.5,color=GREEN[1])
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
    svg_path = basename.with_suffix(".svg")
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
    fig.savefig(output_dir/"fig1_actual_size_150dpi.png",dpi=150,facecolor="white")
    source_paths=["transform_main.py","arch/trans_basenetworks.py","model_registry.py",
                  "ablation_config.py","numeric_domain.py","train_icsps2026_ams.py",
                  "configs/icsps2026_frozen_v2.json","docs/ICSPS2026_LaTeX/IEEEtran.cls",
                  "docs/ICSPS2026_LaTeX/ICSPS2026_paper.tex"]
    manifest={"figure":"Fig. 1","width_mm":W/72*25.4,"height_mm":H/72*25.4,
              "width_tex_pt":516,"width_pdf_pt":W,"minimum_base_font_pt":8.5,
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
