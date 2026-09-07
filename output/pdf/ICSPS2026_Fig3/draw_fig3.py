"""Editable Fig. 3: representation-consistent output and masked adaptation.

Run: python output/pdf/ICSPS2026_Fig3/draw_fig3.py
No model, configuration, experimental data, or manuscript is changed.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.path import Path as MplPath
from fontTools.ttLib import TTFont
from fontTools.pens.svgPathPen import SVGPathPen

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
WIDTH=252/72.27*72
HEIGHT=516
INK="#242B32"
GREEN=("#E6F3EE","#267461")
PURPLE=("#F0EAF7","#795798")
GRAY=("#E9ECEF","#68727D")
WHITE=("#FFFFFF","#66717B")


def editable_svg(path,modules):
    """Preserve module groups and use renderer-compatible math positioning."""
    ns="http://www.w3.org/2000/svg"
    for prefix,uri in [("",ns),("xlink","http://www.w3.org/1999/xlink"),
                       ("dc","http://purl.org/dc/elements/1.1/"),
                       ("cc","http://creativecommons.org/ns#"),
                       ("rdf","http://www.w3.org/1999/02/22-rdf-syntax-ns#")]:
        ET.register_namespace(prefix,uri)
    tree=ET.parse(path)
    root=tree.getroot()
    parents={child:parent for parent in root.iter() for child in parent}
    for span in list(root.iter("{"+ns+"}tspan")):
        xs=span.get("x","").split()
        if len(xs)<=1:
            continue
        chars=list(span.text or "")
        ys=span.get("y","0").split()
        if len(xs)!=len(chars) or len(ys) not in (1,len(chars)):
            raise RuntimeError("Unexpected SVG mathtext positioning")
        parent=parents[span]
        index=list(parent).index(span)
        parent.remove(span)
        for j,(char,x) in enumerate(zip(chars,xs)):
            attrs=dict(span.attrib)
            attrs.update(x=x,y=ys[0] if len(ys)==1 else ys[j])
            part=ET.Element(span.tag,attrs)
            part.text=char
            parent.insert(index+j,part)
    # Small mathematical accents and symbols use STIX fallback glyphs. Outline
    # only these glyphs so that hats, tildes, membership and mask operators are
    # portable even when an SVG editor does not have STIX installed. Main
    # labels, variables, channel counts, and subscripts remain editable text.
    parents={child:parent for parent in root.iter() for child in parent}
    outline_count=0
    for span in list(root.iter("{"+ns+"}tspan")):
        match=re.search(r"([0-9.]+)px '([^']+)'",span.get("style",""))
        if not match or not match.group(2).startswith("STIX"):
            continue
        size,family=float(match.group(1)),match.group(2)
        font_path=font_manager.findfont(font_manager.FontProperties(family=family),fallback_to_default=False)
        face=TTFont(font_path)
        glyphset=face.getGlyphSet()
        cmap=face.getBestCmap()
        scale=size/face["head"].unitsPerEm
        x,y=float(span.get("x","0")),float(span.get("y","0"))
        text_node=parents[span]
        container=parents[text_node]
        for char in span.text or "":
            glyph_name=cmap[ord(char)]
            pen=SVGPathPen(glyphset)
            glyphset[glyph_name].draw(pen)
            ET.SubElement(container,"{"+ns+"}path",{
                "d":pen.getCommands(),"transform":f"translate({x},{y}) scale({scale},{-scale})",
                "data-math-symbol":char,"data-font":family})
            x+=face["hmtx"].metrics[glyph_name][0]*scale
            outline_count+=1
        text_node.remove(span)
        face.close()
    for name in modules:
        parents={child:parent for parent in root.iter() for child in parent}
        parts=[e for e in root.iter() if e.get("id") in {name+"-box",name+"-text"}]
        assert len(parts)==2 and parents[parts[0]] is parents[parts[1]]
        parent=parents[parts[0]]
        group=ET.SubElement(parent,"{"+ns+"}g",{"id":name})
        for part in parts:
            parent.remove(part)
            group.append(part)
    tree.write(path,encoding="utf-8",xml_declaration=True)
    return outline_count


def draw(output_dir,font="Arial",dpi=600):
    font_file=font_manager.findfont(font,fallback_to_default=False)
    plt.rcParams.update({"font.family":font,"font.size":8.3,"svg.fonttype":"none",
                        "svg.hashsalt":"icsps2026-fig3","pdf.fonttype":42,"ps.fonttype":42,
                        "mathtext.fontset":"custom","mathtext.rm":font,
                        "mathtext.it":font+":italic","mathtext.bf":font+":bold",
                        "mathtext.fallback":"stix"})
    fig=plt.figure(figsize=(WIDTH/72,HEIGHT/72),facecolor="white")
    ax=fig.add_axes([0,0,1,1])
    ax.set(xlim=(0,WIDTH),ylim=(HEIGHT,0))
    ax.axis("off")
    checks=[]
    modules=[]

    def text(x,y,s,size=8.3,color=INK,weight="normal",ha="center",rotation=0,gid=None):
        artist=ax.text(x,y,s,fontsize=size,color=color,fontweight=weight,va="center",ha=ha,
                       linespacing=1.15,rotation=rotation,zorder=6)
        if gid:
            artist.set_gid(gid)
        return artist

    def box(name,x,y,w,h,s,style=WHITE,size=8.3):
        p=FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0,rounding_size=2.4",
                        facecolor=style[0],edgecolor=style[1],linewidth=.75,zorder=3)
        p.set_gid(name+"-box")
        ax.add_patch(p)
        t=text(x+w/2,y+h/2,s,size,gid=name+"-text")
        checks.append((t,(x,y,w,h),name))
        modules.append(name)

    def arrow(points,color=INK,dashed=False,head=True):
        path=MplPath(points,[MplPath.MOVETO]+[MplPath.LINETO]*(len(points)-1))
        ax.add_patch(FancyArrowPatch(path=path,arrowstyle="-|>" if head else "-",
                     mutation_scale=6,linewidth=.85,color=color,capstyle="round",joinstyle="round",
                     linestyle=(0,(3,2)) if dashed else "solid",zorder=2))

    def dot(x,y,color=INK):
        ax.plot(x,y,"o",ms=2.2,color=color,zorder=5)

    # (a) The original observation is fused only after the exact inverse.
    text(9,12,"(a) Representation-consistent output",9.2,weight="bold",ha="left")
    box("observed-intensity",9,35,58,37,"Intensity $Y$\n$[0,1]$",GREEN)
    box("bounded-log",79,35,65,37,"Bounded log\n$T_{\\alpha}$, $\\alpha=10$",GREEN,8.2)
    box("log-predictor",155,35,87,37,"Log predictor\n+ sigmoid\n$\\widehat{Z}\\in[0,1]$",WHITE,8.2)
    arrow([(67,53.5),(79,53.5)])
    arrow([(144,53.5),(155,53.5)])
    box("inverse-log",165,88,77,32,"Exact inverse\n$\\widetilde{X}=T_{\\alpha}^{-1}(\\widehat{Z})$",GREEN,8.2)
    arrow([(203.5,72),(203.5,88)])
    box("intensity-concat",93,88,59,32,"Concat\n$[Y,\\widetilde{X}]$",GREEN)
    arrow([(165,104),(152,104)],GREEN[1])
    dot(158,104,GREEN[1])
    arrow([(31,72),(31,81),(122.5,81),(122.5,88)],GREEN[1])
    box("compensator",9,88,71,44,"Compensator\n$2\\to32\\to32\\to1$\n$3\\times3$ convs",GREEN,8)
    arrow([(93,104),(80,104)],GREEN[1])
    arrow([(158,104),(158,133),(123,133),(123,150)],GREEN[1])
    text(173,132,r"$\widetilde{X}$",8.5,color=GREEN[1])
    arrow([(44.5,132),(44.5,157),(116,157)],GREEN[1])
    text(78,148,r"$\gamma_c R_c$",8.6,color=GREEN[1])
    ax.add_patch(Circle((123,157),7,facecolor="white",edgecolor=GREEN[1],linewidth=.75,zorder=4))
    text(123,157,"+",12)
    box("output-clipping",163,142,79,30,"Clip $[0,1]$\nIntensity $\\widehat{X}$",GREEN)
    arrow([(130,157),(163,157)],GREEN[1])
    text(125,185,"Intensity + intensity; $\\gamma_c$ initialized to 0.1",8.1,color=GREEN[1])
    ax.plot([9,242],[196,196],color="#C9CDD2",lw=.6)

    # (b) The complete model gets Ym, including its compensator observation path.
    text(9,209,"(b) Masked post-adaptation (AMS)",9.2,weight="bold",ha="left")
    text(125,223,"Training: new mask per image and epoch",8.1,color=PURPLE[1])
    box("real-observation",9,239,47,25,"Real SAR\n$Y$",WHITE,8.2)
    box("training-mask",82,233,160,37,
        "$M\\sim\\mathrm{Bernoulli}(0.2)$\n$Y_m=Y\\odot(1-M)$",PURPLE,8.7)
    arrow([(56,251.5),(82,251.5)])
    arrow([(162,270),(162,285)])
    text(178,278,r"$Y_m$",8.5)
    frame=FancyBboxPatch((58,285),184,66,boxstyle="round,pad=0,rounding_size=3",
                        facecolor="white",edgecolor=WHITE[1],linewidth=.75,zorder=0)
    frame.set_gid("full-model-status-frame")
    ax.add_patch(frame)
    text(150,296,"Full model $f_{\\theta}(Y_m)$ (panel a)",8.5,weight="bold")
    box("frozen-encoder",64,305,172,14,"Frozen: Transformer encoder",GRAY,8)
    box("trainable-reconstruction",64,324,172,24,
        "Trainable: bottleneck, decoder, guidance;\nhead and intensity compensator",PURPLE,8)
    arrow([(150,351),(150,370)])
    text(164,360,"$P$",8.7)
    box("masked-objective",58,370,184,34,
        "Masked $L_1(P,Y;M)$\n$+\\,10^{-3}\\,\\mathrm{TV}(P)$",PURPLE,8.5)
    # M is an input of the masked loss, not another input of the predictor.
    arrow([(242,251.5),(248,251.5),(248,385.5),(242,385.5)],PURPLE[1])
    text(245,361,"$M$",8.3,color=PURPLE[1],ha="right")
    # The unmasked noisy observation goes only to the target port.
    arrow([(32.5,264),(32.5,412),(110,412),(110,404)])
    text(23,337,"Noisy target $Y$",8.1,rotation=90)
    # Dashed optimization path reaches only the trainable parameter group.
    arrow([(58,389),(49,389),(49,336),(64,336)],PURPLE[1],dashed=True)
    text(43,366,"update",8,color=PURPLE[1],rotation=90)
    text(145,422,"8 epochs; Adam, learning rate $10^{-6}$",8.1,color=PURPLE[1])
    # Validation runs the same masked loss with fixed per-sample masks. It is
    # not driven by the training loss shown above; no train-loss arrow is drawn.
    box("validation-selection",9,434,233,31,
        "Fixed validation mask per sample\nSelect lowest validation loss over epochs 1-8",PURPLE,8.2)
    text(9,476,"Inference (no mask)",8.3,weight="bold",ha="left")
    box("inference-observation",9,483,66,25,"Unmasked $Y$",WHITE,8.2)
    box("selected-model",88,483,80,25,"Selected model\n$f_{\\theta^*}$",WHITE,8.2)
    box("inference-output",181,483,61,25,"Intensity $\\widehat{X}$",GREEN,8.2)
    arrow([(75,495.5),(88,495.5)])
    arrow([(168,495.5),(181,495.5)])
    arrow([(128,465),(128,483)],PURPLE[1],dashed=True)

    output_dir.mkdir(parents=True,exist_ok=True)
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    overflows=[]
    for t,(x,y,w,h),name in checks:
        bounds=t.get_window_extent(renderer).transformed(ax.transData.inverted())
        if bounds.x0<x+.5 or bounds.x1>x+w-.5 or min(bounds.y0,bounds.y1)<y+.3 or max(bounds.y0,bounds.y1)>y+h-.3:
            overflows.append(name)
    if overflows:
        raise RuntimeError("Module text overflow: "+repr(overflows))
    base=output_dir/"fig3_representation_ams"
    for ext in ("svg","pdf","png"):
        kwargs={"dpi":dpi,"facecolor":"white"}
        if ext=="pdf":
            kwargs["metadata"]={"Title":"Fig. 3. Representation-consistent reconstruction and AMS",
                                "Creator":"draw_fig3.py / Matplotlib","CreationDate":None,"ModDate":None}
        elif ext=="svg":
            kwargs["metadata"]={"Date":None}
        fig.savefig(base.with_suffix("."+ext),**kwargs)
    outlined_symbols=editable_svg(base.with_suffix(".svg"),modules)
    fig.savefig(output_dir/"fig3_actual_size_150dpi.png",dpi=150,facecolor="white")
    sources=["transform_main.py","numeric_domain.py","model_registry.py","ablation_config.py",
             "train_icsps2026_ams.py","prepare_icsps2026_real.py","evaluate_icsps2026_real.py",
             "configs/icsps2026_frozen_v2.json","docs/ICSPS2026_LaTeX/ICSPS2026_paper.tex",
             "docs/ICSPS2026_LaTeX/IEEEtran.cls","output/pdf/ICSPS2026_英文论文_LaTeX无图初稿.pdf"]
    manifest={"figure":"Fig. 3","width_mm":WIDTH/72*25.4,"height_mm":HEIGHT/72*25.4,
              "width_tex_pt":252,"width_pdf_pt":WIDTH,"minimum_base_font_pt":8,
              "font":font,"font_file":font_file,"dpi":dpi,"matplotlib":matplotlib.__version__,
              "line_width_pt":{"module":.75,"arrow":.85},"text_overflows":overflows,
              "svg_grouped_modules":modules,"svg_outlined_math_symbols":outlined_symbols,
              "data":"Architecture and protocol only; no experimental images or metrics are plotted.",
              "source_sha256":{s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources}}
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
