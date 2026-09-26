"""Add scene-matched GammaBridge pictures to the Toronto image-only Word file.

The existing eight pictures per scene remain byte-for-byte unchanged. Each
new picture is an independent ninth image in a third, four-column row. This
builder writes a staged DOCX for visual review before replacing the requested
document.
"""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from lxml import etree


ROOT = Path(r"D:\research\sar_transformer-main")
DOC = ROOT / "output/toronto_gt_benchmark/Toronto_三场景_配对真值_图3式独立八图汇总.docx"
DATA = Path(r"E:\SAR_Data\Toronto_Paired_SAR\selected")
RESULTS = ROOT / "output/gamma_bridge_toronto_gt_20260926"
STAGED = ROOT / "output/toronto_gt_benchmark/_word_qa/gamma_bridge/staged.docx"
SCENES = ("01_urban", "02_coast_urban_rural", "03_water_islands_farmland")
NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "pic": "http://schemas.openxmlformats.org/drawingml/2006/picture",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "pr": "http://schemas.openxmlformats.org/package/2006/relationships",
    "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
}


def qn(prefix: str, local: str) -> str:
    return f"{{{NS[prefix]}}}{local}"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def xml_bytes(element: etree._Element) -> bytes:
    return etree.tostring(element, encoding="UTF-8", xml_declaration=True, standalone="yes")


def blip_rid(cell: etree._Element) -> str:
    blips = cell.xpath(".//a:blip", namespaces=NS)
    if len(blips) != 1:
        raise ValueError(f"Expected one image in cell, found {len(blips)}")
    return blips[0].get(qn("r", "embed"))


def main() -> None:
    if STAGED.exists():
        raise FileExistsError(f"Review or remove the prior staged output: {STAGED}")
    if not DOC.exists():
        raise FileNotFoundError(DOC)
    for scene in SCENES:
        run_path = RESULTS / scene / "run.json"
        panel = RESULTS / scene / "denoised.png"
        if not run_path.exists() or not panel.exists():
            raise FileNotFoundError(f"Missing GammaBridge result for {scene}")
        run = json.loads(run_path.read_text(encoding="utf-8"))
        expected_input = DATA / scene / "noisy_intensity.npy"
        if run["scene"] != scene or run["input_sha256"] != sha256(expected_input):
            raise ValueError(f"GammaBridge output does not match Toronto input for {scene}")
        if run["shape"] != [512, 512]:
            raise ValueError(f"Unexpected GammaBridge image dimensions for {scene}")

    with ZipFile(DOC, "r") as source:
        source_names = set(source.namelist())
        document = etree.fromstring(source.read("word/document.xml"))
        relationships = etree.fromstring(source.read("word/_rels/document.xml.rels"))
        rel_map = {rel.get("Id"): rel for rel in relationships}
        body = document.find("w:body", namespaces=NS)
        if body is None:
            raise ValueError("Word document has no body")
        tables = body.findall("w:tbl", namespaces=NS)
        if len(tables) != 6:
            raise ValueError(f"Expected exactly two four-image rows per scene; found {len(tables)} tables")
        if sum(len(table.xpath(".//a:blip", namespaces=NS)) for table in tables) != 24:
            raise ValueError("Existing Word image count is not 24")
        existing_docpr = [int(node.get("id")) for node in document.xpath(".//wp:docPr", namespaces=NS)]
        next_docpr = max(existing_docpr) + 1
        next_rid = max(int(rid[3:]) for rid in rel_map if re.fullmatch(r"rId\d+", rid)) + 1
        image_numbers = [int(match.group(1)) for name in source_names if (match := re.fullmatch(r"word/media/image(\d+)\.png", name))]
        next_image = max(image_numbers) + 1
        additions: dict[str, bytes] = {}

        for index, scene in enumerate(SCENES):
            first_row, second_row = tables[2 * index: 2 * index + 2]
            for row in (first_row, second_row):
                if len(row.findall("w:tr/w:tc", namespaces=NS)) != 4:
                    raise ValueError(f"Unexpected existing grid in {scene}")

            # Verify the first picture is the Noisy image for this exact scene.
            first_cell = first_row.find("w:tr/w:tc", namespaces=NS)
            rid = blip_rid(first_cell)
            target = rel_map[rid].get("Target")
            if not target or not target.startswith("media/"):
                raise ValueError(f"Unexpected first-picture relationship in {scene}")
            word_media = "word/" + target
            expected_noisy = ROOT / "output/toronto_gt_benchmark" / scene / "figures/panels/01_noisy.png"
            if source.read(word_media) != expected_noisy.read_bytes():
                raise ValueError(f"First Word image does not match {scene} Noisy panel")

            new_table = deepcopy(second_row)
            new_cells = new_table.findall("w:tr/w:tc", namespaces=NS)
            new_rid = f"rId{next_rid}"
            new_cells[0].xpath(".//a:blip", namespaces=NS)[0].set(qn("r", "embed"), new_rid)
            docpr = new_cells[0].find(".//wp:docPr", namespaces=NS)
            if docpr is None:
                raise ValueError("Picture lacks Word docPr element")
            docpr.set("id", str(next_docpr))
            docpr.set("name", f"Gamma-Bridge {scene}")
            docpr.set("descr", f"Gamma-Bridge denoised Toronto scene {scene}")
            cnvpr = new_cells[0].find(".//pic:cNvPr", namespaces=NS)
            if cnvpr is not None:
                cnvpr.set("id", str(next_docpr))
                cnvpr.set("name", f"Gamma-Bridge {scene}")

            # Keep matching cell geometry but leave the other three cells empty.
            for cell in new_cells[1:]:
                for para in cell.findall("w:p", namespaces=NS):
                    for child in list(para):
                        if child.tag != qn("w", "pPr"):
                            para.remove(child)
            if len(new_table.xpath(".//a:blip", namespaces=NS)) != 1:
                raise ValueError("The added row must contain exactly one independent picture")

            # Reuse the existing inter-row blank paragraph so the gap stays equal.
            spacer = second_row.getprevious()
            if spacer is None or spacer.tag != qn("w", "p"):
                raise ValueError(f"Expected a spacer before the second row in {scene}")
            new_spacer = deepcopy(spacer)
            second_row.addnext(new_spacer)
            new_spacer.addnext(new_table)

            image_path = f"word/media/image{next_image}.png"
            if image_path in source_names:
                raise ValueError(f"Image part already exists: {image_path}")
            additions[image_path] = (RESULTS / scene / "denoised.png").read_bytes()
            relationship = etree.Element(qn("pr", "Relationship"))
            relationship.set("Id", new_rid)
            relationship.set("Type", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image")
            relationship.set("Target", f"media/image{next_image}.png")
            relationships.append(relationship)
            next_rid += 1
            next_image += 1
            next_docpr += 1

        replacements = {
            "word/document.xml": xml_bytes(document),
            "word/_rels/document.xml.rels": xml_bytes(relationships),
        }
        STAGED.parent.mkdir(parents=True, exist_ok=True)
        with ZipFile(STAGED, "w") as result:
            for item in source.infolist():
                result.writestr(item, replacements.get(item.filename, source.read(item.filename)))
            for path, payload in additions.items():
                result.writestr(path, payload, compress_type=ZIP_DEFLATED)

    print(f"Staged {STAGED}")
    print("3 scenes, 27 independent images, 9 tables")


if __name__ == "__main__":
    main()
