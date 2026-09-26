"""Prepare frozen paired inputs for the three Toronto showcase scenes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
from scipy.io import savemat


ROOT = Path(r"D:\research\sar_transformer-main")
SOURCE = Path(r"E:\SAR_Data\Toronto_Paired_SAR\validation_full")
OUTPUT = Path(r"E:\SAR_Data\Toronto_Paired_SAR\showcase_selected_20260926")
SELECTION = ROOT / "output/toronto_showcase_20260926/selection/selected_scenes.json"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_gray(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        value = np.asarray(image)
    if value.ndim == 3:
        if value.shape[2] not in (3, 4):
            raise ValueError(f"Unexpected channels: {path} {value.shape}")
        if not np.array_equal(value[..., 0], value[..., 1]) or not np.array_equal(value[..., 0], value[..., 2]):
            raise ValueError(f"Published RGB channels are not repeated grayscale: {path}")
        value = value[..., 0]
    if value.shape != (512, 512) or value.dtype != np.uint8:
        raise ValueError(f"Expected 512x512 uint8: {path} {value.shape} {value.dtype}")
    return np.ascontiguousarray(value, dtype=np.float32)


def main() -> None:
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    source_manifest = json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8"))
    scenes = selection["scenes"]
    if len(scenes) != 3 or len({item["filename"] for item in scenes}) != 3:
        raise ValueError("Selection must have three distinct paired filenames")
    hashes = {
        kind: {item["filename"]: item["sha256"] for item in source_manifest["folders"][kind]["files"]}
        for kind in ("Noisy_val", "GTruth_val")
    }
    records = []
    for item in scenes:
        scene_name, name = item["scene"], item["filename"]
        row, col = (int(part) for part in Path(name).stem.split("_"))
        noisy_source = SOURCE / "Noisy_val" / name
        reference_source = SOURCE / "GTruth_val" / name
        for kind, path in (("Noisy_val", noisy_source), ("GTruth_val", reference_source)):
            if not path.is_file() or sha256(path) != hashes[kind][name]:
                raise ValueError(f"Source hash mismatch: {path}")
        noisy, reference = read_gray(noisy_source), read_gray(reference_source)
        if float(noisy.max()) != 255.0:
            raise ValueError(f"Current Trans-SAR adapter needs observed max 255: {name}")
        destination = OUTPUT / scene_name
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(noisy_source, destination / "noisy_original_rgb.tiff")
        shutil.copy2(reference_source, destination / "ground_truth_original_rgb.tiff")
        np.save(destination / "noisy_intensity.npy", noisy, allow_pickle=False)
        np.save(destination / "ground_truth.npy", reference, allow_pickle=False)
        savemat(destination / "input.mat", {"noisy": noisy, "clean": reference}, do_compression=True)
        Image.fromarray(noisy.astype(np.uint8)).save(destination / "noisy.png")
        Image.fromarray(reference.astype(np.uint8)).save(destination / "ground_truth.png")
        record = {
            "scene": scene_name,
            "label_zh": item["label_zh"],
            "filename": name,
            "visual_reason": item["visual_reason"],
            "roi": {"row_start": row, "col_start": col, "height": 512, "width": 512},
            "source": {
                "path": str(noisy_source),
                "sha256": hashes["Noisy_val"][name],
                "product": "Sentinel-1 GRD-HD VV, published 8-bit rescaled intensity",
            },
            "ground_truth": {
                "path": str(reference_source),
                "sha256": hashes["GTruth_val"][name],
                "construction": "registered average of ten acquisitions (pseudo-ground truth)",
            },
            "scientific_domain": {
                "domain": "published uint8 intensity",
                "range": [0, 255],
                "spatial_resampling": False,
                "channel_conversion": "identical RGB channels to first grayscale channel",
            },
            "frozen_method_adapter": {"intensity_scale": 255.0},
        }
        (destination / "run.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        records.append({**record, "directory": str(destination)})
        print(f"Prepared {scene_name}: {name}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset": "SAR despeckling filters dataset",
        "doi": "10.17632/2xf5v5pwkr.2",
        "split": "published validation subset",
        "selection_basis": selection["selection_basis"],
        "reference": "ten-acquisition registered average (pseudo-ground truth)",
        "scenes": records,
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(OUTPUT / "manifest.json")


if __name__ == "__main__":
    main()
