"""Audit display-only stretch choices for the weighted-overlap MERLIN panel.

The preserved ``denoised.npy`` array is linear intensity and is never written.
This script creates only a comparison preview and a JSON audit under the MERLIN
run directory; it does not touch the Figure 3 renderer or its published panels.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = PROJECT_ROOT / "output" / "umbra_buenos_aires_multimethod_1024"
RUN_ROOT = EXPERIMENT_ROOT / "runs" / "merlin_stride64_weighted"
FIGURE3_ROOT = EXPERIMENT_ROOT / "figure3"
CL_SAR_ROOT = PROJECT_ROOT / "output" / "cl_sar_umbra_buenos_aires" / "figure3_clsar_final"
OUTPUT_ROOT = RUN_ROOT / "display_stretch_audit"
PREVIEW_PATH = OUTPUT_ROOT / "merlin_display_stretch_candidates.png"
AUDIT_PATH = OUTPUT_ROOT / "display_stretch_audit.json"

MERLIN_PATH = RUN_ROOT / "denoised.npy"
NOISY_PATH = RUN_ROOT / "noisy_intensity.npy"
MERLIN_RUN_PATH = RUN_ROOT / "run.json"
CL_SAR_RUN_PATH = CL_SAR_ROOT / "run.json"

# Selected from the noisy input by the original CL-SAR scene runner.
URBAN_CROP = {"x": 500, "y": 160, "width": 512, "height": 512}

CONTEXT_PANELS = [
    ("Noisy", "a_noisy.png"),
    ("SAR-BM3D", "b_sar-bm3d.png"),
    ("SAR2SAR", "c_sar2sar.png"),
    ("SDUDNet", "d_sdudnet.png"),
    ("Trans-SAR", "e_trans-sar.png"),
    ("CL-SAR", "f_cl-sar.png"),
    ("MuLoG-DRUNet", "h_mulog-drunet.png"),
]


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def intensity_db(array: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(array.astype(np.float64), np.finfo(np.float32).tiny))


def map_db(array_db: np.ndarray, low: float, high: float) -> np.ndarray:
    if not np.isfinite([low, high]).all() or high <= low:
        raise ValueError(f"Invalid dB limits: {low}, {high}")
    return np.rint(np.clip((array_db - low) / (high - low), 0.0, 1.0) * 255.0).astype(np.uint8)


def exact_rank_histogram_match(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Return a display-domain array with exactly the reference order statistics."""
    source_order = np.argsort(source.ravel(), kind="stable")
    reference_sorted = np.sort(reference.ravel())
    matched = np.empty(source.size, dtype=np.float64)
    matched[source_order] = reference_sorted
    return matched.reshape(source.shape)


def crop(array: np.ndarray) -> np.ndarray:
    x = URBAN_CROP["x"]
    y = URBAN_CROP["y"]
    w = URBAN_CROP["width"]
    h = URBAN_CROP["height"]
    return array[y : y + h, x : x + w]


def display_stats(pixels: np.ndarray) -> dict:
    values = pixels.astype(np.float64)
    gx = np.abs(np.diff(values, axis=1)).ravel()
    gy = np.abs(np.diff(values, axis=0)).ravel()
    gradients = np.concatenate((gx, gy))
    p05, median, p95 = np.percentile(values, [5.0, 50.0, 95.0])
    return {
        "mean_gray_8bit": float(values.mean()),
        "median_gray_8bit": float(median),
        "p05_gray_8bit": float(p05),
        "p95_gray_8bit": float(p95),
        "p05_to_p95_span_8bit": float(p95 - p05),
        "black_fraction": float(np.mean(values == 0)),
        "white_fraction": float(np.mean(values == 255)),
        "mean_absolute_neighbor_gradient_8bit": float(gradients.mean()),
        "p90_absolute_neighbor_gradient_8bit": float(np.percentile(gradients, 90.0)),
    }


def saturation_stats(display_db: np.ndarray, low: float, high: float) -> dict:
    return {
        "low_fraction": float(np.mean(display_db <= low)),
        "high_fraction": float(np.mean(display_db >= high)),
        "total_fraction": float(np.mean((display_db <= low) | (display_db >= high))),
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    source_stat_before = MERLIN_PATH.stat()
    source_hash_before = sha256(MERLIN_PATH)
    merlin_run = read_json(MERLIN_RUN_PATH)
    expected_hash = merlin_run["outputs"]["denoised.npy"]["sha256"]
    if source_hash_before != expected_hash:
        raise RuntimeError("MERLIN scientific-array hash no longer matches run.json")

    merlin = np.load(MERLIN_PATH, allow_pickle=False)
    noisy = np.load(NOISY_PATH, allow_pickle=False)
    for name, array in (("MERLIN", merlin), ("Noisy", noisy)):
        if array.shape != (1024, 1024):
            raise ValueError(f"{name}: expected (1024, 1024), got {array.shape}")
        if not np.isfinite(array).all() or np.any(array < 0):
            raise ValueError(f"{name}: invalid intensity values")

    merlin_db = intensity_db(merlin)
    noisy_db = intensity_db(noisy)
    clsar_run = read_json(CL_SAR_RUN_PATH)
    shared_low, shared_high = map(float, clsar_run["display"]["shared_db_limits"])
    computed_shared = np.percentile(noisy_db, [1.0, 99.7])
    if not np.allclose([shared_low, shared_high], computed_shared, rtol=0.0, atol=1e-6):
        raise RuntimeError("Stored Figure 3 limits do not match the noisy p1-p99.7 window")

    a_low, a_high = map(float, np.percentile(merlin_db, [1.0, 99.7]))
    b_low, b_high = map(float, np.percentile(merlin_db, [0.5, 99.5]))
    median_shift_db = float(np.median(merlin_db) - np.median(noisy_db))
    c_low = shared_low + median_shift_db
    c_high = shared_high + median_shift_db
    matched_db = exact_rank_histogram_match(merlin_db, noisy_db)

    candidates = [
        {
            "id": "current",
            "short_title": "Current: shared noisy window",
            "mapping_type": "affine dB window",
            "display_db": merlin_db,
            "limits_db": [shared_low, shared_high],
            "definition": "Figure 3 noisy p1-p99.7 limits applied unchanged to MERLIN",
            "assessment_zh": "无暗端饱和，但 MERLIN 的最低灰度仍约为 117；整幅图缺少黑场，因而显得发白。",
        },
        {
            "id": "A",
            "short_title": "A: MERLIN p1-p99.7",
            "mapping_type": "affine dB window",
            "display_db": merlin_db,
            "limits_db": [a_low, a_high],
            "definition": "MERLIN's own dB percentiles 1.0 and 99.7",
            "assessment_zh": "城市边缘对比明显增强且保留较多强散射尾部，但全图中位灰度偏低，与其余面板并排时显得过暗。",
        },
        {
            "id": "B",
            "short_title": "B: MERLIN p0.5-p99.5",
            "mapping_type": "affine dB window",
            "display_db": merlin_db,
            "limits_db": [b_low, b_high],
            "definition": "MERLIN's own dB percentiles 0.5 and 99.5",
            "assessment_zh": "候选中城市结构的显示对比最强，双端各约 0.5% 饱和；亮度仍低于图 3 其他面板，更适合细节插图。",
        },
        {
            "id": "C",
            "short_title": "C: noisy span + median shift",
            "mapping_type": "affine dB window",
            "display_db": merlin_db,
            "limits_db": [c_low, c_high],
            "definition": (
                "Preserve the noisy p1-p99.7 dB width and translate both limits by "
                "median(MERLIN_dB)-median(noisy_dB)"
            ),
            "assessment_zh": "把 MERLIN 中位灰度对齐 noisy，同时保持原图 40.83 dB 显示跨度；几乎不裁剪，城市结构不过度增强，最适合主图。",
        },
        {
            "id": "D",
            "short_title": "D: rank histogram match",
            "mapping_type": "nonlinear rank histogram match, then shared noisy affine window",
            "display_db": matched_db,
            "limits_db": [shared_low, shared_high],
            "definition": "Exact rank-order match of MERLIN dB values to the noisy dB histogram",
            "assessment_zh": "亮度与动态范围都接近 noisy，结构醒目；但非线性映射改变不同强度区间的相对对比，解释和复现成本最高，不建议用于主图。",
        },
    ]

    context = []
    for method, filename in CONTEXT_PANELS:
        path = FIGURE3_ROOT / "panels" / filename
        pixels = np.asarray(Image.open(path).convert("L"))
        stats = display_stats(pixels)
        context.append(
            {
                "method": method,
                "path": str(path.relative_to(PROJECT_ROOT)),
                "sha256": sha256(path),
                "pixels": pixels,
                "statistics": stats,
            }
        )
    context_medians = np.array([item["statistics"]["median_gray_8bit"] for item in context])
    context_median_of_medians = float(np.median(context_medians))
    context_min_median = float(context_medians.min())
    context_max_median = float(context_medians.max())

    candidate_records = {}
    for item in candidates:
        low, high = item["limits_db"]
        pixels = map_db(item["display_db"], low, high)
        full_stats = display_stats(pixels)
        urban_pixels = crop(pixels)
        urban_stats = display_stats(urban_pixels)
        full_saturation = saturation_stats(item["display_db"], low, high)
        urban_saturation = saturation_stats(crop(item["display_db"]), low, high)
        candidate_records[item["id"]] = {
            "short_title": item["short_title"],
            "mapping_type": item["mapping_type"],
            "definition": item["definition"],
            "limits_db": [float(low), float(high)],
            "window_width_db": float(high - low),
            "full_roi_saturation": full_saturation,
            "urban_crop_saturation": urban_saturation,
            "full_roi_display_statistics": full_stats,
            "urban_crop_display_statistics": urban_stats,
            "absolute_median_gray_distance_from_context_median": float(
                abs(full_stats["median_gray_8bit"] - context_median_of_medians)
            ),
            "visual_assessment_zh": item["assessment_zh"],
        }
        item["pixels"] = pixels
        item["full_saturation"] = full_saturation
        item["urban_stats"] = urban_stats

    fig = plt.figure(figsize=(18.0, 11.0), dpi=180, facecolor="white")
    outer = fig.add_gridspec(
        3,
        1,
        height_ratios=[1.0, 0.88, 0.53],
        left=0.028,
        right=0.985,
        top=0.905,
        bottom=0.055,
        hspace=0.25,
    )
    full_grid = outer[0].subgridspec(1, len(candidates), wspace=0.035)
    crop_grid = outer[1].subgridspec(1, len(candidates), wspace=0.035)
    context_grid = outer[2].subgridspec(1, len(context), wspace=0.035)

    fig.suptitle(
        "MERLIN display-only stretch audit — weighted-overlap scientific array unchanged",
        fontsize=19,
        fontweight="bold",
        y=0.973,
    )
    fig.text(
        0.5,
        0.937,
        (
            "Top: full 1024×1024 ROI (red box marks the input-selected urban crop).  "
            "Middle: identical 512×512 crop.  Bottom: current Figure 3 context panels."
        ),
        ha="center",
        va="center",
        fontsize=11,
        color="#333333",
    )

    x = URBAN_CROP["x"]
    y = URBAN_CROP["y"]
    w = URBAN_CROP["width"]
    h = URBAN_CROP["height"]
    for index, item in enumerate(candidates):
        pixels = item["pixels"]
        low, high = item["limits_db"]
        sat = item["full_saturation"]
        ax = fig.add_subplot(full_grid[0, index])
        ax.imshow(pixels, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.add_patch(
            plt.Rectangle((x, y), w, h, fill=False, edgecolor="#ef3b2c", linewidth=1.2)
        )
        ax.set_title(
            f"{item['short_title']}\n"
            f"[{low:.2f}, {high:.2f}] dB | clip {100*sat['low_fraction']:.3f}% / "
            f"{100*sat['high_fraction']:.3f}%",
            fontsize=10.5,
            pad=7,
        )
        ax.axis("off")

        city = crop(pixels)
        city_stats = item["urban_stats"]
        ax_crop = fig.add_subplot(crop_grid[0, index])
        ax_crop.imshow(city, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax_crop.set_title(
            f"urban median {city_stats['median_gray_8bit']:.0f} | "
            f"p05–p95 span {city_stats['p05_to_p95_span_8bit']:.0f} | "
            f"grad p90 {city_stats['p90_absolute_neighbor_gradient_8bit']:.0f}",
            fontsize=9.4,
            pad=5,
        )
        ax_crop.axis("off")

    for index, item in enumerate(context):
        ax = fig.add_subplot(context_grid[0, index])
        ax.imshow(item["pixels"], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        median = item["statistics"]["median_gray_8bit"]
        span = item["statistics"]["p05_to_p95_span_8bit"]
        ax.set_title(f"{item['method']}\nmedian {median:.0f} | span {span:.0f}", fontsize=8.8, pad=4)
        ax.axis("off")

    fig.text(0.006, 0.77, "Full ROI", rotation=90, va="center", ha="center", fontsize=11, fontweight="bold")
    fig.text(0.006, 0.45, "Urban crop", rotation=90, va="center", ha="center", fontsize=11, fontweight="bold")
    fig.text(0.006, 0.16, "Fig. 3 context", rotation=90, va="center", ha="center", fontsize=10, fontweight="bold")
    fig.text(
        0.5,
        0.018,
        (
            "Recommendation: C for the main figure (brightness correction only, affine dB slope preserved, "
            "negligible clipping). B is the stronger-contrast fallback; D is intentionally not recommended."
        ),
        ha="center",
        va="bottom",
        fontsize=10.5,
        color="#222222",
    )
    fig.savefig(PREVIEW_PATH, dpi=180, facecolor="white", metadata={"Software": "matplotlib"})
    plt.close(fig)
    # Keep the audit preview broadly readable by image viewers that reject an
    # RGBA PNG even when the rendered page itself is fully opaque.
    with Image.open(PREVIEW_PATH) as rendered:
        rgb_preview = rendered.convert("RGB")
    rgb_preview.save(PREVIEW_PATH, dpi=(180, 180), optimize=True)

    source_hash_after = sha256(MERLIN_PATH)
    source_stat_after = MERLIN_PATH.stat()
    if source_hash_after != source_hash_before:
        raise RuntimeError("MERLIN scientific array changed while producing the display audit")

    audit = {
        "scope": "Display-only stretch audit for the weighted-overlap MERLIN Figure 3 panel",
        "scientific_array_unchanged": True,
        "figure3_renderer_modified": False,
        "main_figure_modified": False,
        "display_arrays_persisted": False,
        "source": {
            "path": str(MERLIN_PATH.relative_to(PROJECT_ROOT)),
            "domain": "linear intensity",
            "shape": list(merlin.shape),
            "dtype": str(merlin.dtype),
            "sha256_before": source_hash_before,
            "sha256_after": source_hash_after,
            "run_json_expected_sha256": expected_hash,
            "size_bytes_before": source_stat_before.st_size,
            "size_bytes_after": source_stat_after.st_size,
            "mtime_ns_before": source_stat_before.st_mtime_ns,
            "mtime_ns_after": source_stat_after.st_mtime_ns,
            "linear_intensity": {
                "min": float(merlin.min()),
                "median": float(np.median(merlin)),
                "max": float(merlin.max()),
            },
            "db_percentiles_0_0p5_1_5_50_95_99_99p5_99p7_100": [
                float(value)
                for value in np.percentile(merlin_db, [0, 0.5, 1, 5, 50, 95, 99, 99.5, 99.7, 100])
            ],
        },
        "reference_noisy": {
            "path": str(NOISY_PATH.relative_to(PROJECT_ROOT)),
            "sha256": sha256(NOISY_PATH),
            "figure3_shared_limits_db": [shared_low, shared_high],
            "figure3_shared_window_width_db": float(shared_high - shared_low),
            "median_db": float(np.median(noisy_db)),
        },
        "urban_crop_xywh": URBAN_CROP,
        "metric_notes": {
            "saturation": "Fraction at or beyond the candidate dB clipping limits; low/high are reported separately.",
            "display_statistics": "Computed on rounded 8-bit preview pixels; p05-p95 span and neighbor gradients describe display-space visibility, not scientific image quality.",
            "context": "Existing Figure 3 PNG panels excluding MERLIN, all rendered with the noisy shared window.",
            "no_clean_reference": True,
            "psnr_ssim_not_used": True,
        },
        "context_panels": {
            "median_gray_range_8bit": [context_min_median, context_max_median],
            "median_of_panel_medians_8bit": context_median_of_medians,
            "panels": [
                {key: value for key, value in item.items() if key != "pixels"} for item in context
            ],
        },
        "median_shift_option_c_db": median_shift_db,
        "candidates": candidate_records,
        "recommendation": {
            "preferred": "C",
            "reason_zh": (
                "C 将 MERLIN 全图中位灰度从当前的 184 降至 120，落入其余图 3 面板的中位灰度范围 119–139；"
                "它沿用 noisy 的 40.83 dB 窗宽，只平移 10.15 dB，因而保持 MERLIN 原有 dB 对比斜率，"
                "高端裁剪约 0.003%，低端无裁剪。城市局部结构仍可辨，不会像 A/B 那样因窄窗被强行放大。"
            ),
            "stronger_contrast_fallback": "B",
            "fallback_reason_zh": (
                "若主观上仍希望更强的街区/建筑对比，B 比 A 更均衡：双端各约 0.5% 裁剪，"
                "城市局部 p05–p95 灰度跨度更大；但它的全图中位灰度明显低于其他面板。"
            ),
            "do_not_recommend": "D",
            "do_not_recommend_reason_zh": (
                "D 的非线性秩直方图匹配改变强弱散射区间的相对显示对比，难以用单一 dB 窗口说明；"
                "对解决单一亮度偏移而言复杂度过高。"
            ),
            "publication_disclosure_zh": (
                "若采用 C，应注明 MERLIN 面板使用独立显示窗：保持 noisy 窗宽并按 MERLIN/noisy 的 dB 中位数差平移；"
                "该操作仅用于显示，不改变或覆盖线性强度科学数组。"
            ),
        },
        "outputs": {
            "preview": {
                "path": str(PREVIEW_PATH.relative_to(PROJECT_ROOT)),
                "sha256": sha256(PREVIEW_PATH),
                "bytes": PREVIEW_PATH.stat().st_size,
            },
            "audit_json": str(AUDIT_PATH.relative_to(PROJECT_ROOT)),
        },
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    AUDIT_PATH.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "preview": str(PREVIEW_PATH),
                "audit": str(AUDIT_PATH),
                "recommendation": audit["recommendation"]["preferred"],
                "scientific_sha256_unchanged": source_hash_before == source_hash_after,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
