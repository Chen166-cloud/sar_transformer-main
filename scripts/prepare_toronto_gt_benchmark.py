"""Prepare three paired Sentinel-1 Toronto validation patches for benchmarking."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image
from scipy.io import savemat


SOURCE_ROOT = Path(r"E:\SAR_Data\Toronto_Paired_SAR\validation_preview")
NOISY_FOLDER = SOURCE_ROOT / "b56730a7-6091-4b6e-9714-075862762917"
GT_FOLDER = SOURCE_ROOT / "b5a1fb7e-d358-47be-a8e0-3708a8c94121"
OUTPUT_ROOT = Path(r"E:\SAR_Data\Toronto_Paired_SAR\selected")

SCENES = (
    ("01_urban", "5120_10240.tiff", "dense Toronto urban fabric and transport corridors"),
    ("02_coast_urban_rural", "5120_15360.tiff", "Lake Ontario coast, urban area, and agricultural fields"),
    ("03_water_islands_farmland", "5120_25088.tiff", "water, islands/shoreline, and agricultural parcels"),
)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_grayscale(path: Path) -> np.ndarray:
    value = tifffile.imread(path)
    if value.ndim == 3:
        if value.shape[2] not in (3, 4) or not np.array_equal(value[..., 0], value[..., 1]):
            raise ValueError(f"Expected repeated grayscale channels: {path} {value.shape}")
        value = value[..., 0]
    if value.shape != (512, 512) or value.dtype != np.uint8:
        raise ValueError(f"Expected 512x512 uint8 image: {path} {value.shape} {value.dtype}")
    return np.ascontiguousarray(value, dtype=np.float32)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "dataset": "SAR despeckling filters dataset",
        "doi": "10.17632/2xf5v5pwkr.2",
        "paper_doi": "10.1016/j.dib.2024.110065",
        "sensor_product": "Sentinel-1 GRD-HD, VV",
        "reference": "ten-acquisition registered multitemporal average (pseudo-ground truth)",
        "source_folder_mapping": {
            "b56730a7-6091-4b6e-9714-075862762917": "Noisy_val",
            "b5a1fb7e-d358-47be-a8e0-3708a8c94121": "GTruth_val",
        },
        "scenes": [],
    }
    for scene_name, filename, description in SCENES:
        scene = OUTPUT_ROOT / scene_name
        scene.mkdir(parents=True, exist_ok=True)
        noisy_source = NOISY_FOLDER / filename
        gt_source = GT_FOLDER / filename
        noisy = read_grayscale(noisy_source)
        ground_truth = read_grayscale(gt_source)
        if noisy.shape != ground_truth.shape:
            raise ValueError(f"Pair shape mismatch: {filename}")

        shutil.copy2(noisy_source, scene / "noisy_original_rgb.tiff")
        shutil.copy2(gt_source, scene / "ground_truth_original_rgb.tiff")
        np.save(scene / "noisy_intensity.npy", noisy, allow_pickle=False)
        np.save(scene / "ground_truth.npy", ground_truth, allow_pickle=False)
        savemat(scene / "input.mat", {"noisy": noisy, "clean": ground_truth}, do_compression=True)
        Image.fromarray(noisy.astype(np.uint8)).save(scene / "noisy.png")
        Image.fromarray(ground_truth.astype(np.uint8)).save(scene / "ground_truth.png")
        record = {
            "scene": scene_name,
            "filename": filename,
            "description": description,
            "roi": {"row_start": int(filename.split("_")[0]),
                    "col_start": int(filename.split("_")[1].split(".")[0]),
                    "height": 512, "width": 512},
            "source": {"path": str(noisy_source), "sha256": sha256(noisy_source),
                       "product": "Sentinel-1 GRD-HD VV 8-bit rescaled intensity"},
            "ground_truth": {"path": str(gt_source), "sha256": sha256(gt_source),
                             "construction": "registered mean of ten acquisitions"},
            "scientific_domain": {"domain": "published uint8 intensity", "range": [0, 255],
                                  "spatial_resampling": False, "channel_conversion": "identical RGB channels -> first channel"},
            "frozen_method_adapter": {"intensity_scale": 255.0},
        }
        (scene / "run.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        record["directory"] = str(scene)
        manifest["scenes"].append(record)

    (OUTPUT_ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(OUTPUT_ROOT)


if __name__ == "__main__":
    main()
