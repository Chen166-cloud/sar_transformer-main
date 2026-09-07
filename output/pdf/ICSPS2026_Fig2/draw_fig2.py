"""Publication-size, editable Fig. 2 for the actual FFTRefineBlock.

Run from the repository root:
    python output/pdf/ICSPS2026_Fig2/draw_fig2.py
Only figure deliverables are written. No model or experiment is modified.
"""
from pathlib import Path
import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.path import Path as MplPath

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
WIDTH = 252 / 72.27 * 72  # IEEEtran columnwidth=21pc, in PDF points
HEIGHT = 412
INK = "#242B32"
ORANGE = "#A8620C"
FILL = "#FFF0D9"


def group_svg_modules(path, modules):
    """Keep each module's box and text together for convenient SVG editing."""
    ns="http://www.w3.org/2000/svg"
    ET.register_namespace("",ns)
    ET.register_namespace("xlink","http://www.w3.org/1999/xlink")
    ET.register_namespace("dc","http://purl.org/dc/elements/1.1/")
    ET.register_namespace("cc","http://creativecommons.org/ns#")
    ET.register_namespace("rdf","http://www.w3.org/1999/02/22-rdf-syntax-ns#")
    tree=ET.parse(path)
    root=tree.getroot()
    # Matplotlib's mathtext groups non-adjacent glyphs into tspans with x lists.
    # Some librsvg-based editors/renderers ignore these lists and overlap the
    # glyphs. Expand only these runs into individually positioned editable spans.
    parents={child:parent for parent in root.iter() for child in parent}
    for span in list(root.iter("{"+ns+"}tspan")):
        xpos=span.get("x","").split()
        if len(xpos)<=1:
            continue
        chars=list(span.text or "")
        ypos=span.get("y","0").split()
        if len(xpos)!=len(chars) or len(ypos) not in (1,len(chars)):
            raise RuntimeError("Unexpected SVG mathtext glyph positioning")
        parent=parents[span]
        at=list(parent).index(span)
        parent.remove(span)
        for j,(ch,x) in enumerate(zip(chars,xpos)):
            attrib=dict(span.attrib)
            attrib.update(x=x,y=ypos[0] if len(ypos)==1 else ypos[j])
            part=ET.Element(span.tag,attrib)
            part.text=ch
            parent.insert(at+j,part)
    for module in modules:
        parents={child:parent for parent in root.iter() for child in parent}
        parts=[e for e in root.iter() if e.get("id") in {module+"-box",module+"-text"}]
        if len(parts)!=2 or parents[parts[0]] is not parents[parts[1]]:
            raise RuntimeError("Cannot group SVG module: "+module)
        parent=parents[parts[0]]
        group=ET.SubElement(parent,"{"+ns+"}g",{"id":module})
        for part in parts:
            parent.remove(part)
            group.append(part)
    tree.write(path,encoding="utf-8",xml_declaration=True)


def draw(output_dir, font="Arial", dpi=600):
    font_file=font_manager.findfont(font,fallback_to_default=False)
    plt.rcParams.update({
        "font.family":font,"font.size":8.5,"svg.fonttype":"none",
        "svg.hashsalt":"icsps2026-fig2-fdr","pdf.fonttype":42,"ps.fonttype":42,
        "mathtext.fontset":"custom","mathtext.rm":font,
        "mathtext.it":font+":italic","mathtext.bf":font+":bold",
    })
    fig=plt.figure(figsize=(WIDTH/72,HEIGHT/72),facecolor="white")
    ax=fig.add_axes([0,0,1,1])
    ax.set(xlim=(0,WIDTH),ylim=(HEIGHT,0))
    ax.axis("off")
    modules=[]
    checks=[]

    def text(x,y,s,size=8.5,color=INK,weight="normal",ha="center",gid=None):
        artist=ax.text(x,y,s,fontsize=size,color=color,fontweight=weight,
                       va="center",ha=ha,linespacing=1.15,zorder=6)
        if gid:
            artist.set_gid(gid)
        return artist

    def box(name,x,y,w,h,s,learned=False,size=8.3):
        p=FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0,rounding_size=2.4",
                        facecolor=FILL if learned else "white",edgecolor=ORANGE,
                        linewidth=.75,zorder=3)
        p.set_gid(name+"-box")
        ax.add_patch(p)
        t=text(x+w/2,y+h/2,s,size,gid=name+"-text")
        checks.append((t,(x,y,w,h),name))
        modules.append(name)

    def arrow(points,head=True,color=INK,dashed=False):
        path=MplPath(points,[MplPath.MOVETO]+[MplPath.LINETO]*(len(points)-1))
        ax.add_patch(FancyArrowPatch(path=path,arrowstyle="-|>" if head else "-",
                     mutation_scale=6,linewidth=.85,color=color,capstyle="round",
                     joinstyle="round",linestyle=(0,(2,2)) if dashed else "solid",zorder=2))

    text(125,12,r"Input $F$: $H\times W\times C$",9.2,weight="bold")
    arrow([(125,20),(125,28)],head=False)
    arrow([(125,28),(64,28),(64,78)])
    arrow([(125,28),(180,28),(180,52)])
    ax.plot(125,28,"o",ms=2.2,color=INK,zorder=5)
    ax.plot(64,28,"o",ms=2.2,color=INK,zorder=5)
    text(34,43,"Spatial",8.5,weight="bold")
    text(212,43,"Spectral",8.5,weight="bold")

    # Spatial convolution sequence (all intermediate widths C).
    box("spatial-dwconv",25,78,78,21,r"DWConv $3\times3$",True)
    box("spatial-pointwise",25,110,78,21,r"$1\times1$ conv: $C\to C$",True)
    box("spatial-gelu",25,142,78,20,"GELU",True)
    arrow([(64,99),(64,110)])
    arrow([(64,131),(64,142)])

    # FFT uses all independent frequency coefficients, with one-sided storage.
    box("rfft",124,52,112,28,
        "rFFT2 (ortho)\n"+r"$Q$: $H\times W_f\times C$ (complex)",False,8.1)
    box("real-imag-concat",124,92,112,28,
        "Concat Re / Im\n"+r"$H\times W_f\times2C$ (real)",False,8.3)
    arrow([(180,80),(180,92)])
    box("frequency-mixing",124,132,112,43,
        r"$1\times1$ conv: $2C\to2C$"+"\nGELU\n"+r"$1\times1$ conv: $2C\to2C$",True,8.4)
    arrow([(180,120),(180,132)])
    box("complex-reconstruction",124,187,112,21,
        r"Split $\to$ Re $+\,i$ Im",False,8.4)
    arrow([(180,175),(180,187)])
    box("irfft",124,220,112,28,
        "irFFT2 (ortho)\n"+r"output size: $(H,W)$",False,8.4)
    arrow([(180,208),(180,220)])

    # Annotation occupies unused space, clear of the feature and residual paths.
    text(70,204,"Frequency mixers:\nshared over coordinates;\nno convolution across\nneighboring frequencies.",8,color=ORANGE)

    # Merge real spatial features. No learned gamma exists in FFTRefineBlock.
    arrow([(64,162),(64,171),(20,171),(20,255),(91,255),(91,268)])
    arrow([(180,248),(180,255),(160,255),(160,268)])
    text(49,247,r"$F_s$",9)
    text(203,255,r"$F_f$",9)
    box("spatial-spectral-concat",69,268,114,24,
        r"Concat $[F_s,F_f]$"+"\n"+r"$H\times W\times2C$",False,8.3)
    box("output-projection",69,304,114,43,
        r"$1\times1$ conv: $2C\to C$"+"\nGELU\n"+r"$3\times3$ conv: $C\to C$",True,8.4)
    arrow([(126,292),(126,304)])
    arrow([(126,347),(126,355)])
    ax.add_patch(Circle((126,362),7,facecolor="white",edgecolor=ORANGE,linewidth=.75,zorder=4))
    text(126,362,"+",12)
    arrow([(64,28),(9,28),(9,362),(119,362)])
    text(42,352,r"Identity $F$",8.2)
    arrow([(126,369),(126,375)])
    text(126,383,r"FDR$(F)$: $H\times W\times C$",9.2,weight="bold")
    text(126,401,r"$W_f$ = floor($W$/2) + 1; batch omitted",8)

    output_dir.mkdir(parents=True,exist_ok=True)
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    overflows=[]
    for t,(x,y,w,h),name in checks:
        bb=t.get_window_extent(renderer).transformed(ax.transData.inverted())
        if bb.x0<x+.5 or bb.x1>x+w-.5 or min(bb.y0,bb.y1)<y+.3 or max(bb.y0,bb.y1)>y+h-.3:
            overflows.append(name)
    if overflows:
        raise RuntimeError("Module text overflow: "+repr(overflows))
    basename=output_dir/"fig2_fdr_block"
    for ext in ("svg","pdf","png"):
        kwargs={"dpi":dpi,"facecolor":"white"}
        if ext=="pdf":
            kwargs["metadata"]={"Title":"Fig. 2. Full-spectrum Fourier residual refinement",
                                "Creator":"draw_fig2.py / Matplotlib","CreationDate":None,"ModDate":None}
        if ext=="svg":
            kwargs["metadata"]={"Date":None}
        fig.savefig(basename.with_suffix("."+ext),**kwargs)
    group_svg_modules(basename.with_suffix(".svg"),modules)
    fig.savefig(output_dir/"fig2_actual_size_150dpi.png",dpi=150,facecolor="white")
    paths=["transform_main.py","model_registry.py","ablation_config.py",
           "configs/icsps2026_frozen_v2.json","docs/ICSPS2026_LaTeX/ICSPS2026_paper.tex",
           "docs/ICSPS2026_LaTeX/IEEEtran.cls","output/pdf/ICSPS2026_英文论文_LaTeX无图初稿.pdf"]
    manifest={"figure":"Fig. 2","width_mm":WIDTH/72*25.4,"height_mm":HEIGHT/72*25.4,
              "width_tex_pt":252,"width_pdf_pt":WIDTH,"minimum_base_font_pt":8,
              "font":font,"font_file":font_file,"dpi":dpi,"matplotlib":matplotlib.__version__,
              "line_width_pt":{"module":.75,"arrow":.85},"text_overflows":overflows,
              "svg_grouped_modules":modules,"shape_notation":"H x W x C; batch omitted",
              "data":"Architecture only; no experimental image or metric is plotted.",
              "source_sha256":{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}}
    (output_dir/"figure_manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps({k:manifest[k] for k in ("figure","width_mm","height_mm","text_overflows")},indent=2))
    plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=HERE)
    parser.add_argument("--font",default="Arial")
    parser.add_argument("--dpi",type=int,default=600)
    args=parser.parse_args()
    draw(args.output_dir,args.font,args.dpi)
