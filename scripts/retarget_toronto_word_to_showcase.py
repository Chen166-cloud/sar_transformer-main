"""Replace Toronto Word comparison scenes with the three selected showcase scenes.

Keeps eight independent image slots and the existing two-row layout per page.
The eighth slot is Gamma-Bridge; MuLoG-DRUNet is omitted by design.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

from lxml import etree
from PIL import Image


ROOT = Path(r"D:\research\sar_transformer-main")
SOURCE = ROOT / "output/toronto_gt_benchmark/Toronto_三场景_配对真值_图3式独立八图汇总.docx"
STAGED = ROOT / "output/toronto_gt_benchmark/_word_qa/showcase_scenes/staged.docx"
SHOWCASE = ROOT / "output/toronto_showcase_20260926"
GAMMA = ROOT / "output/gamma_bridge_probe_20260926"
DATA = Path(r"E:\SAR_Data\Toronto_Paired_SAR\showcase_selected_20260926")
SOURCE_SHA256 = "c62c366bd318210d1a2b64cef9c847a314e434f9d6ae6050c56d5ebda0544236"
SCENES = ("01_city_roads", "02_waterfront", "03_farmland")
PANEL_NAMES = (
    "01_noisy.png",
    "02_ground_truth.png",
    "03_sar-bm3d.png",
    "04_sar2sar.png",
    "05_sdudnet.png",
    "06_trans-sar.png",
    "07_cl-sar.png",
)
NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "pic": "http://schemas.openxmlformats.org/drawingml/2006/picture",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
}


def qn(prefix: str, local: str) -> str:
    return f"{{{NS[prefix]}}}{local}"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    if sha256(SOURCE) != SOURCE_SHA256:
        raise ValueError("The current Word document changed; re-inspect before replacing its images")
    if STAGED.exists():
        raise FileExistsError(STAGED)

    scene_images: dict[str, list[Path]] = {}
    for scene in SCENES:
        images = [SHOWCASE / scene / "figures/panels" / name for name in PANEL_NAMES]
        images.append(GAMMA / scene / "denoised.png")
        run_path = GAMMA / scene / "run.json"
        if any(not image.is_file() for image in images) or not run_path.is_file():
            raise FileNotFoundError(f"Missing showcase or Gamma-Bridge images for {scene}")
        run = json.loads(run_path.read_text(encoding="utf-8"))
        if run["scene"] != scene or run["input_sha256"] != sha256(DATA / scene / "noisy_intensity.npy"):
            raise ValueError(f"Gamma-Bridge result was not made from the selected input for {scene}")
        for image in images:
            with Image.open(image) as im:
                if im.size != (512, 512) or im.mode != "L":
                    raise ValueError(f"Expected a 512x512 grayscale panel: {image}")
        scene_images[scene] = images

    with ZipFile(SOURCE, "r") as original:
        document = etree.fromstring(original.read("word/document.xml"))
        relationships = etree.fromstring(original.read("word/_rels/document.xml.rels"))
        rel_map = {relation.get("Id"): relation.get("Target") for relation in relationships}
        tables = document.xpath("/w:document/w:body/w:tbl", namespaces=NS)
        if len(tables) != 6 or len(document.xpath(".//a:blip", namespaces=NS)) != 24:
            raise ValueError("Expected three pages with eight independent images each")
        replacements: dict[str, bytes] = {}

        for scene_index, scene in enumerate(SCENES):
            rows = tables[2 * scene_index : 2 * scene_index + 2]
            cells = [cell for row in rows for cell in row.findall("w:tr/w:tc", namespaces=NS)]
            if len(cells) != 8:
                raise ValueError(f"Unexpected Word grid for {scene}")
            for cell, image in zip(cells, scene_images[scene], strict=True):
                blips = cell.xpath(".//a:blip", namespaces=NS)
                if len(blips) != 1:
                    raise ValueError(f"Expected one editable image in each {scene} cell")
                rid = blips[0].get(qn("r", "embed"))
                target = rel_map.get(rid)
                if not target or not target.startswith("media/") or not target.endswith(".png"):
                    raise ValueError(f"Unexpected Word image relationship {rid}: {target}")
                member = "word/" + target
                if member in replacements:
                    raise ValueError(f"Two Word images share media part {member}")
                replacements[member] = image.read_bytes()

            gamma_cell = cells[-1]
            docpr = gamma_cell.find(".//wp:docPr", namespaces=NS)
            if docpr is None:
                raise ValueError(f"Missing Gamma-Bridge metadata in {scene}")
            docpr.set("name", f"Gamma-Bridge {scene}")
            docpr.set("descr", f"Gamma-Bridge denoised Toronto showcase scene {scene}")
            cnvpr = gamma_cell.find(".//pic:cNvPr", namespaces=NS)
            if cnvpr is not None:
                cnvpr.set("name", f"Gamma-Bridge {scene}")

        if len(replacements) != 24:
            raise ValueError("Not all 24 image parts were replaced")
        replacements["word/document.xml"] = etree.tostring(
            document, encoding="UTF-8", xml_declaration=True, standalone="yes"
        )
        STAGED.parent.mkdir(parents=True, exist_ok=True)
        with ZipFile(STAGED, "w") as result:
            for entry in original.infolist():
                result.writestr(entry, replacements.get(entry.filename, original.read(entry.filename)))

    print(STAGED)
    print("Three selected showcase scenes; 8 independent images each; Gamma-Bridge is last")


if __name__ == "__main__":
    main()
