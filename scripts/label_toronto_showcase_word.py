"""Add editable method labels below the Toronto showcase images in Word."""

from __future__ import annotations

import hashlib
from pathlib import Path
from zipfile import ZipFile

from lxml import etree


ROOT = Path(r"D:\research\sar_transformer-main")
SOURCE = ROOT / "output/toronto_gt_benchmark/Toronto_三场景_配对真值_图3式独立八图汇总.docx"
STAGED = ROOT / "output/toronto_gt_benchmark/_word_qa/showcase_labels/staged.docx"
SOURCE_SHA256 = "db958aa57c3b93f19c2cf0c2cea8901963eca3faaa1ec42cac14a441af505e91"
SCENES = ("01_city_roads", "02_waterfront", "03_farmland")
LABELS = (
    "(a) Noisy",
    "(b) Ground Truth",
    "(c) SAR-BM3D",
    "(d) SAR2SAR",
    "(e) SDUDNet",
    "(f) Trans-SAR",
    "(g) CL-SAR",
    "(h) γ-Bridge",
)
NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "pic": "http://schemas.openxmlformats.org/drawingml/2006/picture",
    "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
}


def qn(prefix: str, local: str) -> str:
    return f"{{{NS[prefix]}}}{local}"


def source_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def add_caption(cell: etree._Element, label: str, scene: str) -> None:
    if len(cell.xpath(".//a:blip", namespaces=NS)) != 1:
        raise ValueError(f"Expected exactly one image for {scene}: {label}")
    paragraphs = cell.findall("w:p", namespaces=NS)
    if len(paragraphs) != 1 or len(cell.xpath(".//w:t", namespaces=NS)) != 0:
        raise ValueError(f"Expected one image-only paragraph for {scene}: {label}")

    caption = etree.Element(qn("w", "p"))
    paragraph_properties = etree.SubElement(caption, qn("w", "pPr"))
    spacing = etree.SubElement(paragraph_properties, qn("w", "spacing"))
    spacing.set(qn("w", "before"), "0")
    spacing.set(qn("w", "after"), "0")
    spacing.set(qn("w", "line"), "240")
    spacing.set(qn("w", "lineRule"), "auto")
    alignment = etree.SubElement(paragraph_properties, qn("w", "jc"))
    alignment.set(qn("w", "val"), "center")

    run = etree.SubElement(caption, qn("w", "r"))
    run_properties = etree.SubElement(run, qn("w", "rPr"))
    fonts = etree.SubElement(run_properties, qn("w", "rFonts"))
    for script in ("ascii", "hAnsi", "cs"):
        fonts.set(qn("w", script), "Times New Roman")
    font_size = etree.SubElement(run_properties, qn("w", "sz"))
    font_size.set(qn("w", "val"), "15")  # 7.5 pt fits the narrow figure cells.
    text = etree.SubElement(run, qn("w", "t"))
    text.text = label
    cell.append(caption)

    doc_properties = cell.find(".//wp:docPr", namespaces=NS)
    if doc_properties is not None:
        doc_properties.set("name", f"{label} {scene}")
        doc_properties.set("descr", f"{label} image for Toronto scene {scene}")
    picture_properties = cell.find(".//pic:cNvPr", namespaces=NS)
    if picture_properties is not None:
        picture_properties.set("name", f"{label} {scene}")


def main() -> None:
    if source_sha256(SOURCE) != SOURCE_SHA256:
        raise ValueError("The source document changed; inspect it before adding labels")
    if STAGED.exists():
        raise FileExistsError(STAGED)

    with ZipFile(SOURCE, "r") as original:
        document = etree.fromstring(original.read("word/document.xml"))
        tables = document.xpath("/w:document/w:body/w:tbl", namespaces=NS)
        if len(tables) != 6 or len(document.xpath(".//a:blip", namespaces=NS)) != 24:
            raise ValueError("Expected three pages with eight image cells each")

        for scene_index, scene in enumerate(SCENES):
            rows = tables[2 * scene_index : 2 * scene_index + 2]
            cells = [cell for table in rows for cell in table.findall("w:tr/w:tc", namespaces=NS)]
            if len(cells) != 8:
                raise ValueError(f"Expected eight images in {scene}")
            for cell, label in zip(cells, LABELS, strict=True):
                add_caption(cell, label, scene)

        replacement = etree.tostring(
            document, encoding="UTF-8", xml_declaration=True, standalone="yes"
        )
        STAGED.parent.mkdir(parents=True, exist_ok=True)
        with ZipFile(STAGED, "w") as result:
            for entry in original.infolist():
                result.writestr(
                    entry,
                    replacement if entry.filename == "word/document.xml" else original.read(entry.filename),
                )

    print(STAGED)
    print("Added 24 editable captions; all embedded image bytes were preserved")


if __name__ == "__main__":
    main()
