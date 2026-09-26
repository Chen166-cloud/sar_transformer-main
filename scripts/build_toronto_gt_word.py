"""Create an image-only Toronto benchmark Word document from the retained template.

The template's document package is patched surgically so its page and image
layout remain intact. Each result stays a separate image object in Word.
"""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from lxml import etree


ROOT = Path(r"D:\research\sar_transformer-main")
REFERENCE = ROOT / "output/fig3_three_scenes_separate_8_20260925/三场景_图3式独立八图汇总.docx"
OUTPUT_ROOT = ROOT / "output/toronto_gt_benchmark"
OUTPUT = OUTPUT_ROOT / "Toronto_三场景_配对真值_图3式独立八图汇总.docx"
EXPECTED_REFERENCE_SHA256 = "CC49B62896495D1183E9E4C0A6D2F0075B156272797D13CE19A48FE5D5F65BC9"
SCENES = ("01_urban", "02_coast_urban_rural", "03_water_islands_farmland")
PANEL_NAMES = (
    "01_noisy.png",
    "02_ground_truth.png",
    "03_sar-bm3d.png",
    "04_sar2sar.png",
    "05_sdudnet.png",
    "06_trans-sar.png",
    "07_cl-sar.png",
    "08_mulog-drunet.png",
)
NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "pr": "http://schemas.openxmlformats.org/package/2006/relationships",
    "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
}


def qn(prefix: str, local: str) -> str:
    return f"{{{NS[prefix]}}}{local}"


def xml_bytes(element: etree._Element) -> bytes:
    return etree.tostring(element, encoding="UTF-8", xml_declaration=True, standalone="yes")


def image_rid(cell: etree._Element) -> str:
    blips = cell.xpath(".//a:blip", namespaces=NS)
    if len(blips) != 1:
        raise ValueError(f"Expected exactly one image in a table cell, found {len(blips)}")
    return blips[0].get(qn("r", "embed"))


def main() -> None:
    if sha256(REFERENCE.read_bytes()).hexdigest().upper() != EXPECTED_REFERENCE_SHA256:
        raise ValueError("Reference document changed; re-inspect its layout before building")

    all_inputs = [OUTPUT_ROOT / scene / "figures/panels" / name for scene in SCENES for name in PANEL_NAMES]
    missing = [str(path) for path in all_inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing panel images: " + ", ".join(missing))

    with ZipFile(REFERENCE, "r") as source:
        source_names = set(source.namelist())
        document = etree.fromstring(source.read("word/document.xml"))
        relationships = etree.fromstring(source.read("word/_rels/document.xml.rels"))
        rel_map = {rel.get("Id"): rel for rel in relationships}

        body = document.find("w:body", namespaces=NS)
        if body is None:
            raise ValueError("No document body")
        tables = body.findall("w:tbl", namespaces=NS)
        if len(tables) != 6:
            raise ValueError(f"Expected six source tables, found {len(tables)}")

        replacements: dict[str, bytes] = {}
        additions: dict[str, bytes] = {}
        next_rid = max(int(rid[3:]) for rid in rel_map if rid.startswith("rId") and rid[3:].isdigit()) + 1
        next_image_number = 22
        next_docpr_id = 22

        for scene_index, scene in enumerate(SCENES):
            first_table, second_table = tables[2 * scene_index : 2 * scene_index + 2]
            first_cells = first_table.findall("w:tr/w:tc", namespaces=NS)
            second_cells = second_table.findall("w:tr/w:tc", namespaces=NS)
            if (len(first_cells), len(second_cells)) != (4, 3):
                raise ValueError("The source image-grid shape has changed")
            panel_paths = [OUTPUT_ROOT / scene / "figures/panels" / name for name in PANEL_NAMES]

            for cell, panel in zip(first_cells + second_cells, panel_paths[:7], strict=True):
                rid = image_rid(cell)
                relationship = rel_map[rid]
                target = relationship.get("Target")
                if not target or not target.startswith("media/image") or not target.endswith(".png"):
                    raise ValueError(f"Unexpected image relationship target for {rid}: {target}")
                media_path = "word/" + target
                if media_path not in source_names or media_path in replacements:
                    raise ValueError(f"Invalid or repeated image target: {media_path}")
                replacements[media_path] = panel.read_bytes()

            second_row = second_table.find("w:tr", namespaces=NS)
            grid = second_table.find("w:tblGrid", namespaces=NS)
            if second_row is None or grid is None:
                raise ValueError("Second-row table lacks a row or grid")
            new_cell = deepcopy(second_cells[0])
            new_rid = f"rId{next_rid}"
            image_rid(new_cell)
            new_cell.xpath(".//a:blip", namespaces=NS)[0].set(qn("r", "embed"), new_rid)
            doc_pr = new_cell.find(".//wp:docPr", namespaces=NS)
            if doc_pr is not None:
                doc_pr.set("id", str(next_docpr_id))
                doc_pr.set("name", f"Picture {next_docpr_id}")
            second_row.append(new_cell)
            grid.append(deepcopy(grid[0]))

            media_path = f"word/media/image{next_image_number}.png"
            additions[media_path] = panel_paths[7].read_bytes()
            relationship = etree.Element(qn("pr", "Relationship"))
            relationship.set("Id", new_rid)
            relationship.set("Type", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image")
            relationship.set("Target", f"media/image{next_image_number}.png")
            relationships.append(relationship)
            next_rid += 1
            next_image_number += 1
            next_docpr_id += 1

        # Remove the reference's empty trailing space and legacy method labels.
        last_table = tables[-1]
        while last_table.getnext() is not None and last_table.getnext().tag != qn("w", "sectPr"):
            body.remove(last_table.getnext())

        replacements["word/document.xml"] = xml_bytes(document)
        replacements["word/_rels/document.xml.rels"] = xml_bytes(relationships)
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        with ZipFile(OUTPUT, "w") as result:
            for info in source.infolist():
                result.writestr(info, replacements.get(info.filename, source.read(info.filename)))
            for path, payload in additions.items():
                result.writestr(path, payload, compress_type=ZIP_DEFLATED)

    print(f"Created {OUTPUT}")
    print(f"Scenes: {len(SCENES)}; independent images: {len(replacements) - 2 + len(additions)}")


if __name__ == "__main__":
    main()
