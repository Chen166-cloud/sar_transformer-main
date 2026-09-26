"""Export three existing SAR comparisons as editable Word-style single panels.

The image pixels are copied from each comparison's caption-free panel files.
Only PNG DPI metadata is adjusted to match the physical size of Figure 3 in
V3修改答复.docx; no crop, interpolation, or intensity remapping is performed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "fig3_three_scenes_separate_8_20260925"
PANEL_NAMES = (
    "a_noisy.png",
    "b_sar-bm3d.png",
    "c_sar2sar.png",
    "d_sdudnet.png",
    "e_trans-sar.png",
    "f_cl-sar.png",
    "g_merlin.png",
    "h_mulog-drunet.png",
)
SCENES = (
    (
        "01_Hong_Kong_ground_v3",
        ROOT / "output" / "umbra_stability_ground_v3" / "01_Hong_Kong_Port"
        / "experiment" / "figure3_ground_equal_scale" / "panels_word_center_square",
        "Hong Kong v3: ground-proportion center square, 1024x1024 SICD method input",
    ),
    (
        "02_Nevada_from_report_v2",
        ROOT / "output" / "umbra_stability_new_3scenes" / "02_Tesla_Semi_Factory"
        / "experiment" / "figure3_style" / "panels",
        "Nevada v2: original SICD pixel grid, old report ROI",
    ),
    (
        "03_Melbourne_from_report_v2",
        ROOT / "output" / "umbra_stability_new_3scenes" / "03_Melbourne"
        / "experiment" / "figure3_style" / "panels",
        "Melbourne v2: original SICD pixel grid, old report ROI",
    ),
)

# wp:extent from Figure 3 in D:\Users\Chen\Desktop\V3修改答复.docx.
WORD_PICTURE_INCHES = 758190 / 914400


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Output already exists; refusing to overwrite: {OUTPUT}")
    missing = [str(folder / name) for _, folder, _ in SCENES for name in PANEL_NAMES if not (folder / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing source panels:\n" + "\n".join(missing))

    OUTPUT.mkdir(parents=True)
    record = {
        "reference_docx": r"D:\Users\Chen\Desktop\V3修改答复.docx",
        "reference_figure": "Figure 3; eight independent square PNGs, 4+4, with labels as separate Word text",
        "word_picture_inches": WORD_PICTURE_INCHES,
        "pixel_operation": "none; source pixels unchanged; PNG DPI metadata only",
        "scenes": [],
    }
    for slug, folder, description in SCENES:
        destination = OUTPUT / slug
        destination.mkdir()
        scene_record = {"folder": slug, "description": description, "files": []}
        for name in PANEL_NAMES:
            source = folder / name
            target = destination / name
            with Image.open(source) as original:
                original.load()
                if original.mode != "L" or original.width != original.height:
                    raise ValueError(f"Expected square grayscale source panel: {source}")
                dpi = round(original.width / WORD_PICTURE_INCHES)
                original.save(target, dpi=(dpi, dpi), optimize=True)
                expected = original.tobytes()
                size = original.size
            with Image.open(target) as exported:
                exported.load()
                if exported.mode != "L" or exported.size != size or exported.tobytes() != expected:
                    raise ValueError(f"Exported pixels differ from source: {target}")
                actual_dpi = [float(v) for v in exported.info.get("dpi", ())]
            scene_record["files"].append(
                {
                    "name": name,
                    "source": str(source),
                    "source_sha256": sha256(source),
                    "output_sha256": sha256(target),
                    "pixels": list(size),
                    "dpi_metadata": actual_dpi,
                }
            )
        record["scenes"].append(scene_record)
    (OUTPUT / "manifest.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
