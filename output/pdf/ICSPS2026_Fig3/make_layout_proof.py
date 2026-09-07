"""Validate Fig. 3 vector files and build a single-column actual-size proof.

python output/pdf/ICSPS2026_Fig3/make_layout_proof.py
Requires pypdf and reportlab. The manuscript is not changed or recompiled.
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
page=PdfReader(HERE/"fig3_representation_ams.pdf").pages[0]
width,height=float(page.mediabox.width),float(page.mediabox.height)
assert abs(width-COL)<.001 and len(page.images)==0
fonts=[]
for ref in page["/Resources"]["/Font"].values():
    font=ref.get_object()
    for child_ref in font.get("/DescendantFonts",[font]):
        child=child_ref.get_object()
        descriptor=child.get("/FontDescriptor")
        embedded=bool(descriptor and any(k in descriptor.get_object() for k in ("/FontFile","/FontFile2","/FontFile3")))
        assert embedded
        fonts.append({"name":str(child.get("/BaseFont")),"embedded":embedded,"subtype":str(child.get("/Subtype"))})
svg=ET.parse(HERE/"fig3_representation_ams.svg")
ns={"s":"http://www.w3.org/2000/svg"}
text_count=len(svg.findall(".//s:text",ns))
image_count=len(svg.findall(".//s:image",ns))
assert text_count>30 and image_count==0
spans=svg.findall(".//s:tspan",ns)
assert all(len(e.get("x","").split())<=1 for e in spans)
assert not any("STIX" in e.get("style","") or "cmex" in e.get("style","") or "cmsy" in e.get("style","") for e in spans)
manifest=json.loads((HERE/"figure_manifest.json").read_text(encoding="utf-8"))
ids={e.get("id") for e in svg.findall(".//s:g",ns)}
assert all(m in ids for m in manifest["svg_grouped_modules"])
symbols=[p for p in svg.findall(".//s:path",ns) if p.get("data-math-symbol")]
assert len(symbols)==manifest["svg_outlined_math_symbols"]
trace=json.loads((HERE/"workflow_trace.json").read_text(encoding="utf-8"))
assert trace["all_checks_passed"]

left=(612-TEXT)/2
right=left+TEXT-COL
bottom=792-54-height
buffer=io.BytesIO()
c=canvas.Canvas(buffer,pagesize=(612,792))
c.setTitle("Fig. 3 actual-size single-column placement proof")
c.setFont("Helvetica",8)
c.setFillColorRGB(.35,.35,.35)
c.drawString(left,765,"Layout proof: IEEEtran conference, 21 pc column width, 100% print scale")
caption=(HERE/"caption.txt").read_text(encoding="utf-8").strip()
style=ParagraphStyle("caption",fontName="Times-Roman",fontSize=8,leading=9.5)
para=Paragraph("Fig. 3. "+caption,style)
_,caption_height=para.wrap(COL,200)
para.drawOn(c,left,bottom-6-caption_height)
assert bottom-6-caption_height>72, "Figure and caption do not fit this proof page."

notes_style=ParagraphStyle("notes",fontName="Helvetica",fontSize=9,leading=13,textColor="#30363B")
notes=[
    "<b>Single-column figure</b>",
    "Size: 88.568 x 182.033 mm.<br/>Keep width=columnwidth and natural height. The original 1.25-inch placeholder cannot retain readable labels at this level of detail.",
    "<b>Panel (a): numerical domain</b>",
    "The bounded log prediction is analytically inverted. Both the residual base and the compensator's inputs are intensities. Clipping follows the residual addition.",
    "<b>Panel (b): training and inference</b>",
    "The entire model receives Y_m during adaptation, including its observation bypass. Original Y supplies only the noisy loss target. M selects loss pixels. The shaded parameter groups show optimization status, not serial forward stages.",
    "Validation reuses a fixed mask for each image and compares the same masked L1 plus TV objective. Inference uses the selected model without a mask.",
    "<b>Paper/code difference recorded</b>",
    "The paper writes sum(M) + epsilon in the masked-L1 denominator; the actual code uses max(sum(M), 1). The diagram names the implemented masked objective; SOURCE_AUDIT.md records the exact evidence. Neither code nor paper was altered.",
    "<b>Verified outputs</b>",
    "88 passed checks, including two full-model forwards.<br/>PDF: vectors and embedded fonts.<br/>SVG: 16 grouped modules; main text remains editable. Ten special math glyphs are portable vector outlines.",
    "<b>Scope</b>",
    "This proof verifies final figure size, not full-manuscript pagination. No experimental data, checkpoint, training, or model edit was used for the figure.",
]
y=738
for note in notes:
    p=Paragraph(note,notes_style)
    _,h=p.wrap(COL,600)
    y-=h
    p.drawOn(c,right,y)
    y-=12
c.setFont("Helvetica",8)
c.drawString(left,54,"Print at 100% without 'Fit to page'. Original manuscript, models, and experimental data are unchanged.")
c.save()
proof=PdfReader(io.BytesIO(buffer.getvalue())).pages[0]
proof.merge_transformed_page(page,Transformation().translate(left,bottom))
writer=PdfWriter()
writer.add_page(proof)
with (HERE/"fig3_layout_proof.pdf").open("wb") as handle:
    writer.write(handle)
result={"pdf_width_pt":width,"pdf_height_pt":height,"pdf_raster_image_count":len(page.images),
        "embedded_fonts":fonts,"svg_editable_text_elements":text_count,"svg_raster_image_count":image_count,
        "svg_grouped_module_count":len(manifest["svg_grouped_modules"]),
        "svg_outlined_math_symbol_count":len(symbols),"svg_math_position_lists_expanded":True,
        "column_width_mm":COL/72*25.4,"figure_height_mm":height/72*25.4,
        "caption_height_pt":caption_height,"proof_page_size_pt":[612,792],
        "model_checks_passed":trace["passed_count"],"model_forward_count":trace["model_inference_count"],
        "pypdf":importlib.metadata.version("pypdf"),"reportlab":importlib.metadata.version("reportlab"),
        "manual_review":"Final PDF and independently rendered SVG were opened and visually inspected; see README.md."}
(HERE/"export_qa.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
print(json.dumps(result,indent=2))
