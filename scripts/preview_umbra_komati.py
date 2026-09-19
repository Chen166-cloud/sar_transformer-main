"""Create display-only previews from the downloaded Umbra GEC numerical raster.

The provider originals are read-only. The provider's existing uint8 display
mapping is retained. PNGs must not replace the complex SAR in experiments.
"""

from pathlib import Path
import argparse
import json
import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roi", type=int, nargs=2, metavar=("X", "Y"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    scene = root / "datasets/umbra_open/komati_20230802"
    source = scene / "2023-08-02-19-35-16_UMBRA-05_GEC.tif"
    dest = scene / "previews"
    dest.mkdir(exist_ok=True)
    with Image.open(source) as image:
        arr = np.asarray(image)
        info = {"source": source.name, "width": image.width,
                "height": image.height, "mode": image.mode,
                "dtype": str(arr.dtype)}
    if arr.dtype != np.uint8 or arr.ndim != 2:
        raise ValueError("This preview script expects the inspected uint8 GEC product")
    display = Image.fromarray(arr)
    overview = display.copy()
    overview.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
    overview.save(dest / "overview.png")
    info.update({"display_only": True, "radiometric_transform": "none; provider uint8 mapping retained",
                 "overview_resampling": "LANCZOS thumbnail",
                 "display_limits": [0, 255],
                 "overview_size": list(overview.size), "despeckled": False})
    if args.roi:
        x, y = args.roi
        if x < 0 or y < 0 or x + 1024 > display.width or y + 1024 > display.height:
            raise ValueError("ROI is outside the original image")
        display.crop((x, y, x + 1024, y + 1024)).save(dest / "industrial_roi_1024.png")
        info["roi"] = {"x": x, "y": y, "width": 1024, "height": 1024,
                       "resampled": False, "same_stretch_as_overview": True}
    (dest / "preview_metadata.json").write_text(
        json.dumps(info, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
