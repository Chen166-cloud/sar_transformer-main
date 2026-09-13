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
ROOT=next(p for p in HERE.parents if (p/"model_registry.py").is_file())
WIDTH=252/72.27*72
HEIGHT=378
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
    plt.rcParams.update({"font.family":font,"font.size":8.8,"svg.fonttype":"none",
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

    def text(x,y,s,size=8.8,color=INK,weight="normal",ha="center",rotation=0,gid=None):
        artist=ax.text(x,y,s,fontsize=size,color=color,fontweight=weight,va="center",ha=ha,
                       linespacing=1.15,rotation=rotation,zorder=6)
        if gid:
            artist.set_gid(gid)
        return artist

    def box(name,x,y,w,h,s,style=WHITE,size=8.8):
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

    # (a) Compact, editable same-domain reconstruction. The observation input
    # is the generic model input: AMS substitutes Ym for it in panel (b).
    text(9,11,"(a) Representation-consistent",9.2,weight="bold",ha="left")
    text(9,22,"output reconstruction",9.2,weight="bold",ha="left")
    box("observed-intensity",9,34,49,27,"Intensity $Y$",GREEN)
    box("bounded-log",71,34,67,27,"Bounded log\n$T_{\\alpha}$",GREEN)
    box("log-predictor",151,34,91,27,
        "Log predictor\n+ sigmoid $\\to\\widehat{Z}$",WHITE)
    arrow([(58,47.5),(71,47.5)])
    arrow([(138,47.5),(151,47.5)])
    box("inverse-log",162,73,80,31,
        "Exact inverse\n$\\widetilde{X}=T_{\\alpha}^{-1}(\\widehat{Z})$",GREEN,8.6)
    arrow([(203.5,61),(203.5,73)])
    box("compensator",9,75,137,28,
        "Intensity compensation\n$R_c=F_{\\mathrm{comp}}([Y,\\widetilde{X}])$",GREEN)
    # Only intensity-valued Y and Xtilde reach the compensator.
    arrow([(33.5,61),(33.5,68),(71,68),(71,75)],GREEN[1])
    arrow([(162,89),(146,89)],GREEN[1])
    # Keep the uncompensated estimate as the other residual-addition input.
    dot(155,89,GREEN[1])
    arrow([(155,89),(155,110),(119,110),(119,116)],GREEN[1])
    arrow([(43,103),(43,123),(112,123)],GREEN[1])
    text(78,114,r"$\gamma_c R_c$",8.8,color=GREEN[1])
    ax.add_patch(Circle((119,123),7,facecolor="white",edgecolor=GREEN[1],linewidth=.75,zorder=4))
    text(119,123,"+",12)
    box("output-clipping",164,109,78,28,"Clip $[0,1]$\nIntensity $\\widehat{X}$",GREEN)
    arrow([(126,123),(164,123)],GREEN[1])
    text(125,145,r"Intensity sum: $\widetilde{X}+\gamma_c R_c$",8.8,color=GREEN[1])
    ax.plot([9,242],[154,154],color="#C9CDD2",lw=.6)

    # (b) Only masked Ym enters the complete model, including the observation
    # path of the compensator above. Unmasked Y is solely a loss target.
    text(9,165,"(b) Masked post-adaptation (AMS)",9.2,weight="bold",ha="left")
    box("real-observation",9,182,44,27,"Real SAR\n$Y$",WHITE,8.6)
    box("training-mask",80,178,162,34,
        "$M\\sim\\mathrm{Bernoulli}(0.2)$\n$Y_m=Y\\odot(1-M)$",PURPLE,8.9)
    arrow([(53,195.5),(80,195.5)])
    arrow([(162,212),(162,223)])
    frame=FancyBboxPatch((58,223),184,57,boxstyle="round,pad=0,rounding_size=3",
                        facecolor="white",edgecolor=WHITE[1],linewidth=.75,zorder=0)
    frame.set_gid("full-model-status-frame")
    ax.add_patch(frame)
    text(150,232,"Whole model $f_{\\theta}(Y_m)$",8.8,weight="bold")
    box("frozen-encoder",64,240,172,14,"Frozen: Transformer encoder",GRAY,8.4)
    box("trainable-reconstruction",64,258,172,19,
        "Trainable: bottleneck, decoder, guidance;\nhead and intensity compensator",PURPLE,8.2)
    arrow([(150,280),(150,288)])
    text(163,284,"$P$",8.8)
    box("masked-objective",58,288,184,38,
        "Masked $L_1(P,Y;M)$\n$+\\,\\lambda_{\\mathrm{AMS}}\\,\\mathrm{TV}(P)$",PURPLE,8.2)
    # Mask selects supervised pixels; it is not an additional model input.
    arrow([(242,195),(248,195),(248,306),(242,306)],PURPLE[1])
    text(246,286,"$M$",8.5,color=PURPLE[1],ha="right")
    # Enter the lower-left loss port, 7 pt below the update route. A full
    # horizontal terminal segment avoids an arrowhead overrunning a short bend.
    arrow([(31,209),(31,317),(58,317)])
    text(22,266,"Target $Y$",8.6,rotation=90)
    # Only the trainable parameter group receives the dashed update arrow.
    arrow([(58,310),(49,310),(49,267.5),(64,267.5)],PURPLE[1],dashed=True)
    text(41,289,"update",8.2,color=PURPLE[1],rotation=90)
    # Fixed validation masks select a checkpoint, independently of training.
    text(125,332,"Train: resampled masks. Validation: fixed masks;",8.4,color=PURPLE[1])
    text(125,342,"select checkpoint by validation loss.",8.4,color=PURPLE[1])
    # The inference row is intentionally compact and still explicitly unmasked.
    box("inference-observation",9,350,66,24,"Inference\nUnmasked $Y$",WHITE,8.6)
    box("selected-model",88,350,80,24,"Selected model\n$f_{\\theta^*}$",WHITE,8.6)
    box("inference-output",181,350,61,24,"Intensity $\\widehat{X}$",GREEN,8.6)
    arrow([(75,362),(88,362)])
    arrow([(168,362),(181,362)])

    output_dir.mkdir(parents=True,exist_ok=True)
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    overflows=[]
    for t,(x,y,w,h),name in checks:
        bounds=t.get_window_extent(renderer).transformed(ax.transData.inverted())
        if bounds.x0<x+.5 or bounds.x1>x+w-.5 or min(bounds.y0,bounds.y1)<y+.3 or max(bounds.y0,bounds.y1)>y+h-.3:
            overflows.append((name, (bounds.x0, bounds.y0, bounds.x1, bounds.y1), (x,y,w,h)))
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
    svg_path=base.with_suffix(".svg")
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines())+"\n",encoding="utf-8")
    fig.savefig(output_dir/"fig3_actual_size_150dpi.png",dpi=150,facecolor="white")
    sources=["transform_main.py","numeric_domain.py","model_registry.py","ablation_config.py",
             "train_icsps2026_ams.py","prepare_icsps2026_real.py","evaluate_icsps2026_real.py",
             "configs/icsps2026_frozen_v2.json","docs/ICSPS2026_LaTeX/ICSPS2026_paper.tex",
             "docs/ICSPS2026_LaTeX/IEEEtran.cls","output/pdf/ICSPS2026_英文论文_LaTeX无图初稿.pdf"]
    manifest={"figure":"Fig. 3","width_mm":WIDTH/72*25.4,"height_mm":HEIGHT/72*25.4,
              "width_tex_pt":252,"width_pdf_pt":WIDTH,"minimum_base_font_pt":8.2,"default_label_font_pt":8.8,
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
