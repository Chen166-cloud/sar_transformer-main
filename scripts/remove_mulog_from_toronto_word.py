"""Remove MuLoG-DRUNet panels from the Toronto Word comparison.

Each scene retains the seven earlier slots plus its Gamma-Bridge picture in
the eighth slot. The MuLoG media and its unused relationship are removed from
the package. Writes a staged DOCX for visual review.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from zipfile import ZipFile

from lxml import etree


ROOT = Path(r"D:\research\sar_transformer-main")
SOURCE = ROOT / "output/toronto_gt_benchmark/Toronto_三场景_配对真值_图3式独立八图汇总.docx"
STAGED = ROOT / "output/toronto_gt_benchmark/_word_qa/no_mulog/staged.docx"
EXPECTED_SHA256 = "b18e1c9eb5b6760ca3c5e21d2f6321d62d2b790c67b29cf81e07304833c9960d"
SCENES = ("01_urban", "02_coast_urban_rural", "03_water_islands_farmland")
NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "pic": "http://schemas.openxmlformats.org/drawingml/2006/picture",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
}


def qn(prefix: str, local: str) -> str:
    return f"{{{NS[prefix]}}}{local}"


def xml_bytes(root: etree._Element) -> bytes:
    return etree.tostring(root, encoding="UTF-8", xml_declaration=True, standalone="yes")


def main() -> None:
    with SOURCE.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError("Word document changed since the nine-image version was reviewed")
    if STAGED.exists():
        raise FileExistsError(STAGED)

    with ZipFile(SOURCE, "r") as source:
        document = etree.fromstring(source.read("word/document.xml"))
        relationships = etree.fromstring(source.read("word/_rels/document.xml.rels"))
        body = document.find("w:body", namespaces=NS)
        if body is None:
            raise ValueError("Word document has no body")
        tables = body.findall("w:tbl", namespaces=NS)
        if len(tables) != 9:
            raise ValueError(f"Expected three rows per scene, found {len(tables)} tables")

        dropped_parts: set[str] = set()
        dropped_rids: set[str] = set()
        expected_old_parts = {"word/media/image22.png", "word/media/image23.png", "word/media/image24.png"}
        expected_new_parts = {"word/media/image25.png", "word/media/image26.png", "word/media/image27.png"}
        rel_map = {relationship.get("Id"): relationship for relationship in relationships}

        for scene_index, scene in enumerate(SCENES):
            first, second, third = tables[3 * scene_index : 3 * scene_index + 3]
            if [len(row.findall("w:tr/w:tc", namespaces=NS)) for row in (first, second, third)] != [4, 4, 4]:
                raise ValueError(f"Unexpected four-column grid for {scene}")
            second_cells = second.findall("w:tr/w:tc", namespaces=NS)
            third_cells = third.findall("w:tr/w:tc", namespaces=NS)
            old_blips = second_cells[3].xpath(".//a:blip", namespaces=NS)
            gamma_blips = third_cells[0].xpath(".//a:blip", namespaces=NS)
            if len(old_blips) != 1 or len(gamma_blips) != 1:
                raise ValueError(f"Expected old and new images for {scene}")
            old_rid = old_blips[0].get(qn("r", "embed"))
            gamma_rid = gamma_blips[0].get(qn("r", "embed"))
            old_target = rel_map[old_rid].get("Target")
            gamma_target = rel_map[gamma_rid].get("Target")
            if old_target != f"media/image{22 + scene_index}.png" or gamma_target != f"media/image{25 + scene_index}.png":
                raise ValueError(f"Unexpected method mapping for {scene}")
            result_image = ROOT / "output/gamma_bridge_toronto_gt_20260926" / scene / "denoised.png"
            if source.read("word/" + gamma_target) != result_image.read_bytes():
                raise ValueError(f"Gamma-Bridge image differs from the scene result for {scene}")

            old_blips[0].set(qn("r", "embed"), gamma_rid)
            old_docpr = second_cells[3].find(".//wp:docPr", namespaces=NS)
            gamma_docpr = third_cells[0].find(".//wp:docPr", namespaces=NS)
            if old_docpr is None or gamma_docpr is None:
                raise ValueError(f"Missing Word picture metadata for {scene}")
            old_docpr.attrib.update(gamma_docpr.attrib)
            old_cnvpr = second_cells[3].find(".//pic:cNvPr", namespaces=NS)
            gamma_cnvpr = third_cells[0].find(".//pic:cNvPr", namespaces=NS)
            if old_cnvpr is not None and gamma_cnvpr is not None:
                old_cnvpr.attrib.update(gamma_cnvpr.attrib)

            spacer = third.getprevious()
            if spacer is None or spacer.tag != qn("w", "p"):
                raise ValueError(f"Missing third-row spacer for {scene}")
            body.remove(spacer)
            body.remove(third)
            relationships.remove(rel_map[old_rid])
            dropped_parts.add("word/" + old_target)
            dropped_rids.add(old_rid)

        if dropped_parts != expected_old_parts or dropped_rids != {"rId30", "rId31", "rId32"}:
            raise ValueError("Removed assets do not match the three MuLoG panels")
        if not expected_new_parts.issubset(set(source.namelist())):
            raise ValueError("Gamma-Bridge media parts are missing")
        if len(body.findall("w:tbl", namespaces=NS)) != 6:
            raise ValueError("Expected two image rows per scene after edit")
        if len(document.xpath(".//a:blip", namespaces=NS)) != 24:
            raise ValueError("Expected eight independent pictures per scene")

        replacements = {
            "word/document.xml": xml_bytes(document),
            "word/_rels/document.xml.rels": xml_bytes(relationships),
        }
        STAGED.parent.mkdir(parents=True, exist_ok=True)
        with ZipFile(STAGED, "w") as output:
            for info in source.infolist():
                if info.filename not in dropped_parts:
                    output.writestr(info, replacements.get(info.filename, source.read(info.filename)))

    print(STAGED)
    print("3 scenes; 8 independent pictures per scene; MuLoG-DRUNet media removed")


if __name__ == "__main__":
    main()
