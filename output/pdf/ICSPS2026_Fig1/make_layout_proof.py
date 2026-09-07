"""Verify vector exports and create an IEEEtran-width page proof.

Run after draw_fig1.py: python output/pdf/ICSPS2026_Fig1/make_layout_proof.py
Requires pypdf and reportlab. The original manuscript is never modified.
"""
import io
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import importlib.metadata
from pypdf import PdfReader, PdfWriter, Transformation
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle

HERE = Path(__file__).resolve().parent
W = 516 / 72.27 * 72
page = PdfReader(HERE / "fig1_overall_architecture.pdf").pages[0]
width, height = float(page.mediabox.width), float(page.mediabox.height)
assert abs(width-W) < .001
assert len(page.images) == 0, "Architecture PDF must contain no raster images."
fonts = []
for ref in page["/Resources"]["/Font"].values():
    font=ref.get_object()
    descendants=font.get("/DescendantFonts",[font])
    for child_ref in descendants:
        child=child_ref.get_object()
        desc=child.get("/FontDescriptor")
        embedded=bool(desc and any(k in desc.get_object() for k in ("/FontFile","/FontFile2","/FontFile3")))
        assert embedded, f"Unembedded font: {child.get('/BaseFont')}"
        fonts.append({"name":str(child.get("/BaseFont")),"embedded":embedded,"subtype":str(child.get("/Subtype"))})
svg=ET.parse(HERE/"fig1_overall_architecture.svg")
ns={"s":"http://www.w3.org/2000/svg"}
text_count=len(svg.findall(".//s:text",ns))
image_count=len(svg.findall(".//s:image",ns))
assert text_count>50 and image_count==0
trace=json.loads((HERE/"forward_trace.json").read_text(encoding="utf-8"))

# At 100% print scale, figure width equals the manuscript's 43pc text block.
left=(612-W)/2
top=54
bottom=792-top-height
buffer=io.BytesIO()
c=canvas.Canvas(buffer,pagesize=(612,792))
c.setTitle("Fig. 1 actual-size placement proof")
c.setFont("Helvetica",8)
c.setFillColorRGB(.4,.4,.4)
c.drawString(left,765,"Layout proof: IEEEtran conference, 43 pc text width, 100% scale")
caption=(HERE/"caption.txt").read_text(encoding="utf-8").strip()
style=ParagraphStyle("caption",fontName="Times-Roman",fontSize=8,leading=9.5)
p=Paragraph("Fig. 1. "+caption,style)
_,caption_h=p.wrap(W,100)
p.drawOn(c,left,bottom-6-caption_h)
c.setFont("Helvetica",8)
c.drawString(left,54,"Figure: 181.353 x 152.400 mm. Print without 'Fit to page'. Original manuscript unchanged.")
c.save()
proof_page=PdfReader(io.BytesIO(buffer.getvalue())).pages[0]
proof_page.merge_transformed_page(page,Transformation().translate(left,bottom))
writer=PdfWriter()
writer.add_page(proof_page)
with (HERE/"fig1_layout_proof.pdf").open("wb") as f:
    writer.write(f)
result={"pdf_width_pt":width,"pdf_height_pt":height,"pdf_raster_image_count":len(page.images),
        "embedded_fonts":fonts,"svg_editable_text_elements":text_count,"svg_raster_image_count":image_count,
        "proof_page_size_pt":[612,792],"proof_figure_width_mm":W/72*25.4,
        "caption_height_pt":caption_h,
        "pypdf_version":importlib.metadata.version("pypdf"),
        "reportlab_version":importlib.metadata.version("reportlab"),
        "manual_review":"See README.md for the final Poppler-rendered visual inspection record."}
(HERE/"export_qa.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
print(json.dumps(result,indent=2))
