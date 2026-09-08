"""Local inference using SAR2SAR's original saved TensorFlow graph and weights.

Author implementation: E. Dalsasso, L. Denis, F. Tupin, SAR2SAR, GPL-3.0.
This wrapper supplies current TensorFlow compatibility and local file I/O.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
from PIL import Image, ImageDraw
from scipy.io import loadmat, savemat
import tensorflow as tf

from download_official import ARCHIVE_SHA256, COMMIT, DESTINATION, ROOT, sha256


def read_image(path: Path, field: str) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        value = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == ".mat":
        payload = loadmat(path, variable_names=[field])
        if field not in payload:
            raise ValueError(f"MAT file does not contain field {field!r}")
        value = payload[field]
    else:
        with Image.open(path) as image:
            value = np.asarray(image)
    value = np.asarray(value)
    if not np.issubdtype(value.dtype, np.number) or np.iscomplexobj(value):
        raise ValueError("Input must be a real numeric array")
    if value.ndim != 2 or min(value.shape) < 1:
        raise ValueError("Input must be a single-channel 2-D SAR array")
    value = value.astype(np.float64)
    if not np.isfinite(value).all() or (value < 0).any() or not (value > 0).any():
        raise ValueError("Input must be finite, nonnegative, with at least one positive pixel")
    return value


def preview(array: np.ndarray, upper: float) -> Image.Image:
    pixels = np.rint(np.clip(array / upper, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(pixels)


def validate_official_package() -> tuple[Path, dict]:
    provenance_path = DESTINATION / "provenance.json"
    if not provenance_path.exists():
        raise FileNotFoundError("Run scripts/sar2sar/setup_windows.ps1 first")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("source_commit") != COMMIT or provenance.get("archive_sha256") != ARCHIVE_SHA256:
        raise RuntimeError("SAR2SAR source provenance does not match this runner")
    for relative, expected in provenance["files_sha256"].items():
        path = DESTINATION / relative
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"Author file changed or missing: {path}")
    return DESTINATION / "SAR2SAR-test", provenance


def infer(amplitude: np.ndarray, package: Path, stride: int, threads: int):
    # The exact normalization and output tensor are bound from verified author
    # code in official_adapter; the U-Net architecture is imported from .meta.
    from official_adapter import normalize, denormalize, OUTPUT_TENSOR

    patch = 256
    height, width = amplitude.shape
    if min(height, width) < patch:
        raise ValueError("SAR2SAR requires at least 256 x 256 pixels; no implicit resizing is applied")
    normalized = normalize(amplitude)
    ys = list(range(0, height - patch + 1, stride))
    xs = list(range(0, width - patch + 1, stride))
    if ys[-1] != height - patch:
        ys.append(height - patch)
    if xs[-1] != width - patch:
        xs.append(width - patch)
    estimate = np.zeros((height, width), dtype=np.float64)
    count = np.zeros((height, width), dtype=np.int32)
    checkpoint = package / "checkpoint/SAR2SAR-tensorflow-98550"
    tf.compat.v1.disable_eager_execution()
    graph = tf.Graph()
    started = time.perf_counter()
    with graph.as_default():
        saver = tf.compat.v1.train.import_meta_graph(str(checkpoint) + ".meta", clear_devices=True)
    configuration = tf.compat.v1.ConfigProto(
        device_count={"GPU": 0}, intra_op_parallelism_threads=threads,
        inter_op_parallelism_threads=1, allow_soft_placement=True,
    )
    with tf.compat.v1.Session(graph=graph, config=configuration) as session:
        saver.restore(session, str(checkpoint))
        input_tensor = graph.get_tensor_by_name("noisy_image:0")
        training_tensor = graph.get_tensor_by_name("is_training:0")
        output_tensor = graph.get_tensor_by_name(OUTPUT_TENSOR)
        total = len(ys) * len(xs)
        completed = 0
        for y in ys:
            for x in xs:
                tile = normalized[y:y + patch, x:x + patch]
                prediction = session.run(output_tensor, {
                    input_tensor: tile[None, :, :, None].astype(np.float32),
                    training_tensor: False,
                })[0, :, :, 0]
                estimate[y:y + patch, x:x + patch] += prediction
                count[y:y + patch, x:x + patch] += 1
                completed += 1
                if completed == 1 or completed == total or completed % 10 == 0:
                    print(f"SAR2SAR patches: {completed}/{total}", flush=True)
    result = denormalize(estimate / count)
    if not np.isfinite(result).all() or (result < 0).any():
        raise RuntimeError("SAR2SAR produced invalid amplitude values")
    return result, {
        "patch_size": patch, "stride_size": stride, "patch_count": total,
        "minimum_patch_coverage": int(count.min()), "maximum_patch_coverage": int(count.max()),
        "inference_seconds": time.perf_counter() - started,
        "output_tensor": OUTPUT_TENSOR,
        "checkpoint_prefix": str(checkpoint),
        "checkpoint_sha256": {p.name: sha256(p) for p in checkpoint.parent.glob(checkpoint.name + ".*")},
        "graph_execution": "original TF1.13.1 .meta imported with TensorFlow compat.v1; no architecture rewrite",
    }


def main() -> None:
    from official_adapter import NORMALIZATION
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--input-domain", choices=["amplitude", "intensity"], default="amplitude")
    parser.add_argument("--mat-field", default="noisy")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--crop-size", type=int, default=None,
                        help="Centered crop (demo default 256; custom input default 0 = entire image)")
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.stride <= 256 or args.threads < 1:
        parser.error("stride must be 1..256 and threads must be positive")
    if args.crop_size is not None and args.crop_size != 0 and args.crop_size < 256:
        parser.error("crop-size must be 0 or at least 256")
    package, provenance = validate_official_package()
    official_demo = args.input is None
    crop_size = (256 if official_demo else 0) if args.crop_size is None else args.crop_size
    source = args.input or next(iter(sorted((package / "test_data").glob("*.npy"))), None)
    if source is None:
        raise FileNotFoundError("No official test image found; supply --input")
    source = source.resolve()
    domain = "amplitude" if official_demo else args.input_domain
    value = read_image(source, args.mat_field)
    source_shape = list(value.shape)
    crop = None
    if crop_size:
        if min(value.shape) < crop_size:
            raise ValueError("Input is smaller than requested crop-size")
        y, x = (np.array(value.shape) - crop_size) // 2
        crop = [int(y), int(x), crop_size, crop_size]
        value = value[y:y + crop_size, x:x + crop_size]
    amplitude = np.sqrt(value) if domain == "intensity" else value
    out = (args.output or ROOT / "output/sar2sar_local" /
           datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")).resolve()
    out.mkdir(parents=True, exist_ok=True)
    for filename in ("result.mat", "summary.json", "input.png", "denoised.png", "comparison.png"):
        if (out / filename).exists():
            raise FileExistsError(f"Refusing to overwrite {out / filename}")
    denoised, execution = infer(amplitude, package, args.stride, args.threads)
    noisy_intensity, denoised_intensity = amplitude ** 2, denoised ** 2
    if not np.isfinite(denoised_intensity).all():
        raise RuntimeError("Squared intensity contains non-finite values")
    upper = float(np.percentile(amplitude, 99))
    if upper <= 0:
        upper = float(amplitude.max())
    before, after = preview(amplitude, upper), preview(denoised, upper)
    before.save(out / "input.png")
    after.save(out / "denoised.png")
    comparison = Image.new("L", (amplitude.shape[1] * 2 + 8, amplitude.shape[0] + 24), 255)
    comparison.paste(before, (0, 24))
    comparison.paste(after, (amplitude.shape[1] + 8, 24))
    drawing = ImageDraw.Draw(comparison)
    drawing.text((5, 5), "Input amplitude", fill=0)
    drawing.text((amplitude.shape[1] + 13, 5), "SAR2SAR amplitude", fill=0)
    comparison.save(out / "comparison.png")
    summary = {
        "algorithm": "SAR2SAR", "model_status": "official_pretrained_checkpoint",
        "source_repository": provenance["source_repository"], "source_commit": COMMIT,
        "archive_sha256": ARCHIVE_SHA256, "paper_doi": provenance["paper_doi"],
        "license": "GPL-3.0", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "official_demo": official_demo, "input_path": str(source), "input_sha256": sha256(source),
        "source_image_size": source_shape, "crop_y_x_height_width": crop,
        "processed_image_size": list(amplitude.shape), "input_domain": domain,
        "input_numeric_values_preserved": True, "algorithm_io_domain": "linear_amplitude",
        "normalization": NORMALIZATION,
        "input_pixels_below_normalization_floor": int((amplitude < 0.24).sum()),
        "training_data_scope": "single-look Sentinel-1; use a GRD model for GRD data",
        "output_domains": ["linear_amplitude", "linear_intensity"],
        "display_amplitude_range": [0, upper], "png_display_only": True,
        "ground_truth_available": False, "psnr_ssim_available": False,
        "tensorflow_version": tf.__version__, "device": "CPU", "threads": args.threads,
        "wrapper_sha256": sha256(Path(__file__)), **execution,
        "normalization_adapter_sha256": sha256(Path(__file__).with_name("official_adapter.py")),
    }
    savemat(out / "result.mat", {
        "noisy_amplitude": amplitude, "denoised_amplitude": denoised,
        "noisy_intensity": noisy_intensity, "denoised_intensity": denoised_intensity,
        "summary_json": json.dumps(summary),
    }, do_compression=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Results: {out}", flush=True)


if __name__ == "__main__":
    main()
