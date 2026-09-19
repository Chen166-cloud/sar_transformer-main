"""Create a safe V3 copy with Figure 3 replaced by eight editable image objects.

The original document is never modified.  The existing Figure 3 OOXML layout
(two centered rows of four inline pictures) is preserved exactly; only the
eight PNG payloads are replaced and the two panel-label rows, two method-name
rows, and the Figure 3 caption are cleared.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import tempfile
from zipfile import ZIP_DEFLATED, ZipFile

from lxml import etree


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DOCX = Path(r"D:\Users\Chen\Desktop\V3.docx")
EXPECTED_SOURCE_SHA256 = (
    "6be7fc4d1cb59d9ca4a9804f9c163daad76c51b4f62c60baf77713c967b264ae"
)
EXPORT_ROOT = (
    PROJECT_ROOT
    / "output"
    / "umbra_buenos_aires_multimethod_1024"
    / "figure3"
    / "word_ready_no_text"
)
OUTPUT_DOCX = EXPORT_ROOT / "V3_Figure3_editable_no_text.docx"
PACKAGE_ZIP = EXPORT_ROOT / "figure3_editable_no_text_package.zip"

PANELS = (
    EXPORT_ROOT / "panels" / "a_noisy.png",
    EXPORT_ROOT / "panels" / "b_sar-bm3d.png",
    EXPORT_ROOT / "panels" / "c_sar2sar.png",
    EXPORT_ROOT / "panels" / "d_sdudnet.png",
    EXPORT_ROOT / "panels" / "e_trans-sar.png",
    EXPORT_ROOT / "panels" / "f_cl-sar.png",
    EXPORT_ROOT / "panels" / "g_merlin.png",
    EXPORT_ROOT / "panels" / "h_mulog-drunet.png",
)

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
NS = {"w": W_NS, "a": A_NS, "r": R_NS, "pr": PKG_REL_NS}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clear_paragraph_content(paragraph: etree._Element) -> None:
    """Remove visible paragraph content while retaining paragraph properties."""
    for child in list(paragraph):
        if child.tag != f"{{{W_NS}}}pPr":
            paragraph.remove(child)


def main() -> None:
    if not SOURCE_DOCX.is_file():
        raise FileNotFoundError(SOURCE_DOCX)
    source_hash = sha256(SOURCE_DOCX)
    if source_hash.lower() != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(
            "V3.docx has changed since the Figure 3 audit; refusing to edit an "
            f"unexpected source (SHA-256 {source_hash})."
        )
    for panel in PANELS:
        if not panel.is_file():
            raise FileNotFoundError(panel)

    EXPORT_ROOT.mkdir(parents=True, exist_ok=True)
    with ZipFile(SOURCE_DOCX, "r") as source_zip:
        document_root = etree.fromstring(source_zip.read("word/document.xml"))
        rels_root = etree.fromstring(source_zip.read("word/_rels/document.xml.rels"))

        rel_targets = {
            rel.get("Id"): rel.get("Target")
            for rel in rels_root.xpath("./pr:Relationship", namespaces=NS)
        }
        paragraphs = document_root.xpath(".//w:body//w:p", namespaces=NS)

        image_rows: list[tuple[int, etree._Element, list[str]]] = []
        for index, paragraph in enumerate(paragraphs):
            embeds = paragraph.xpath(".//a:blip/@r:embed", namespaces=NS)
            if len(embeds) == 4:
                targets = [rel_targets.get(embed, "") for embed in embeds]
                media_names = [Path(target).name for target in targets]
                if media_names in (
                    [f"image{i}.png" for i in range(10, 14)],
                    [f"image{i}.png" for i in range(14, 18)],
                ):
                    image_rows.append((index, paragraph, embeds))

        if len(image_rows) != 2:
            raise RuntimeError(f"Expected two Figure 3 image rows, found {len(image_rows)}")
        image_rows.sort(key=lambda item: item[0])

        # Clear (a)-(d), method names, (e)-(h), method names, and the Fig. 3 caption.
        first_index = image_rows[0][0]
        second_index = image_rows[1][0]
        clear_indices = [
            first_index + 1,
            first_index + 2,
            second_index + 1,
            second_index + 2,
            second_index + 3,
        ]
        expected_text_prefixes = ("(a)", "Noisy", "(e)", "Trans-SAR", "Fig. 3.")
        for index, prefix in zip(clear_indices, expected_text_prefixes, strict=True):
            text = "".join(paragraphs[index].xpath(".//w:t/text()", namespaces=NS))
            if not text.startswith(prefix):
                raise RuntimeError(
                    f"Unexpected Figure 3 text at XML paragraph {index}: {text!r}"
                )
            clear_paragraph_content(paragraphs[index])

        figure_embeds = image_rows[0][2] + image_rows[1][2]
        media_replacements: dict[str, bytes] = {}
        for embed, panel in zip(figure_embeds, PANELS, strict=True):
            target = rel_targets.get(embed)
            if not target:
                raise RuntimeError(f"Missing relationship target for {embed}")
            member = str((Path("word") / Path(target)).as_posix())
            media_replacements[member] = panel.read_bytes()

        document_bytes = etree.tostring(
            document_root,
            xml_declaration=True,
            encoding="UTF-8",
            standalone=True,
        )

        with tempfile.NamedTemporaryFile(
            prefix="v3_fig3_", suffix=".docx", dir=EXPORT_ROOT, delete=False
        ) as temp_stream:
            temp_path = Path(temp_stream.name)

        try:
            with ZipFile(temp_path, "w", compression=ZIP_DEFLATED) as output_zip:
                for info in source_zip.infolist():
                    if info.filename == "word/document.xml":
                        data = document_bytes
                    elif info.filename in media_replacements:
                        data = media_replacements[info.filename]
                    else:
                        data = source_zip.read(info.filename)
                    output_zip.writestr(info, data)
            shutil.move(str(temp_path), OUTPUT_DOCX)
        finally:
            temp_path.unlink(missing_ok=True)

    package_members = [
        OUTPUT_DOCX,
        EXPORT_ROOT / "figure3_2x4_no_text.png",
        EXPORT_ROOT / "README.md",
        EXPORT_ROOT / "manifest.json",
        *PANELS,
    ]
    with ZipFile(PACKAGE_ZIP, "w", compression=ZIP_DEFLATED) as package:
        for path in package_members:
            if not path.is_file():
                raise FileNotFoundError(path)
            package.write(path, path.relative_to(EXPORT_ROOT).as_posix())

    print(OUTPUT_DOCX)
    print(PACKAGE_ZIP)
    print(f"source_sha256={source_hash}")
    print(f"output_sha256={sha256(OUTPUT_DOCX)}")


if __name__ == "__main__":
    main()
