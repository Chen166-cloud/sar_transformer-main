"""Validate Fig. 2 vector exports and place it at actual IEEEtran column width.

python output/pdf/ICSPS2026_Fig2/make_layout_proof.py
Requires pypdf and reportlab. Does not modify or recompile the manuscript.
"""
from pathlib import Path
import io
import json
import importlib.metadata
import xml.etree.ElementTree as ET
from pypdf import PdfReader, PdfWriter, Transformation
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle

HERE=Path(__file__).resolve().parent
COL=252/72.27*72
TEXT=516/72.27*72
page=PdfReader(HERE/"fig2_fdr_block.pdf").pages[0]
width,height=float(page.mediabox.width),float(page.mediabox.height)
assert abs(width-COL)<.001
assert len(page.images)==0
fonts=[]
for font_ref in page["/Resources"]["/Font"].values():
    font=font_ref.get_object()
    for child_ref in font.get("/DescendantFonts",[font]):
        child=child_ref.get_object()
        desc=child.get("/FontDescriptor")
        embedded=bool(desc and any(k in desc.get_object() for k in ("/FontFile","/FontFile2","/FontFile3")))
        assert embedded
        fonts.append({"name":str(child.get("/BaseFont")),"embedded":embedded,"subtype":str(child.get("/Subtype"))})
svg=ET.parse(HERE/"fig2_fdr_block.svg")
ns={"s":"http://www.w3.org/2000/svg"}
text_count=len(svg.findall(".//s:text",ns))
image_count=len(svg.findall(".//s:image",ns))
assert text_count>20 and image_count==0
assert all(len(e.get("x","").split())<=1 for e in svg.findall(".//s:tspan",ns))
manifest=json.loads((HERE/"figure_manifest.json").read_text(encoding="utf-8"))
group_ids={g.get("id") for g in svg.findall(".//s:g",ns)}
assert all(m in group_ids for m in manifest["svg_grouped_modules"])
trace=json.loads((HERE/"fdr_forward_trace.json").read_text(encoding="utf-8"))
assert trace["all_checks_passed"] and trace["registered_FDR_instance_count"]==6

left=(612-TEXT)/2
right=left+TEXT-COL
top=54
bottom=792-top-height
buffer=io.BytesIO()
c=canvas.Canvas(buffer,pagesize=(612,792))
c.setTitle("Fig. 2 actual-size single-column placement proof")
c.setFont("Helvetica",8)
c.setFillColorRGB(.35,.35,.35)
c.drawString(left,765,"Layout proof: IEEEtran conference, 21 pc column width, 100% print scale")
caption=(HERE/"caption.txt").read_text(encoding="utf-8").strip()
caption_style=ParagraphStyle("caption",fontName="Times-Roman",fontSize=8,leading=9.5)
p=Paragraph("Fig. 2. "+caption,caption_style)
_,caption_h=p.wrap(COL,200)
p.drawOn(c,left,bottom-6-caption_h)

# The right-hand notes are solely for the proof and never enter the figure.
notes_style=ParagraphStyle("notes",fontName="Helvetica",fontSize=9,leading=13,textColor="#30363B")
notes=[
    "<b>Single-column placement</b>",
    "Final figure size: 88.568 x 145.344 mm.<br/>Insert with width=columnwidth and preserve its natural aspect ratio.",
    "<b>Location in the manuscript</b>",
    "Replace the current Fig. 2 placeholder (fig:fdr), after the Full-Spectrum Fourier Residual Refinement subsection and before Guidance Modulation and Intensity Compensation.",
    "<b>What the diagram explains</b>",
    "Two feature branches, one-sided Fourier storage, real-imaginary channel mixing, inverse FFT, spatial fusion, and the identity residual.",
    "Orange retains the FDR color from Fig. 1. Shaded boxes contain convolutional processing; white boxes show transforms, reshaping, concatenation, or addition.",
    "<b>Verified deliverables</b>",
    "PDF: vectors with embedded fonts.<br/>SVG: editable text and 10 grouped modules.<br/>Model check: 155 passed checks across the six real model instances and one odd-width diagnostic.",
    "<b>Scope</b>",
    "This is a size proof, not a recompiled manuscript. The original 1.15-inch placeholder height is insufficient; final paper pagination must be checked after replacement.",
]
y=738
for note in notes:
    para=Paragraph(note,notes_style)
    _,h=para.wrap(COL,500)
    y-=h
    para.drawOn(c,right,y)
    y-=12
c.setFont("Helvetica",8)
c.drawString(left,54,"Print at 100% without 'Fit to page'. Model, experiments, and manuscript remain unchanged.")
c.save()
proof=PdfReader(io.BytesIO(buffer.getvalue())).pages[0]
proof.merge_transformed_page(page,Transformation().translate(left,bottom))
writer=PdfWriter()
writer.add_page(proof)
with (HERE/"fig2_layout_proof.pdf").open("wb") as out:
    writer.write(out)
result={"pdf_width_pt":width,"pdf_height_pt":height,"pdf_raster_image_count":len(page.images),
        "embedded_fonts":fonts,"svg_editable_text_elements":text_count,"svg_raster_image_count":image_count,
        "svg_grouped_module_count":len(manifest["svg_grouped_modules"]),"svg_math_position_lists_expanded":True,
        "column_width_mm":COL/72*25.4,"figure_height_mm":height/72*25.4,
        "caption_height_pt":caption_h,"proof_page_size_pt":[612,792],
        "model_checks_passed":trace["check_count"],"model_fdr_instances":trace["registered_FDR_instance_count"],
        "pypdf_version":importlib.metadata.version("pypdf"),"reportlab_version":importlib.metadata.version("reportlab"),
        "manual_review":"Final PDF and independently rendered grouped SVG were opened and inspected; see README.md."}
(HERE/"export_qa.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
print(json.dumps(result,indent=2))
