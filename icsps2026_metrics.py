"""Auditable metrics for the ICSPS 2026 SAR despeckling protocol.

This module intentionally does *not* implement or report M-index or RGPI.  The
repository's historical M implementation has not yet been numerically matched
to a reference implementation, and Sobel gradient correlation is not RGPI.

All functions operate on two-dimensional, finite, linear-intensity arrays.
Display stretching must never be applied before calling these functions.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.ndimage import sobel


DEFAULT_RATIO_EPS = 1e-6
DEFAULT_ACF_RADIUS = 5


def _as_finite_2d(image: np.ndarray, name: str = "image") -> np.ndarray:
    array = np.asarray(image, dtype=np.float64)
    array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return array


def _matching_arrays(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = _as_finite_2d(first, "first image")
    b = _as_finite_2d(second, "second image")
    if a.shape != b.shape:
        raise ValueError(f"Image shape mismatch: {a.shape} != {b.shape}")
    return a, b


def intensity_ratio(
    noisy: np.ndarray,
    prediction: np.ndarray,
    eps: float = DEFAULT_RATIO_EPS,
) -> tuple[np.ndarray, float]:
    """Return ``noisy / max(prediction, eps)`` and denominator-floor fraction."""

    if eps <= 0:
        raise ValueError("eps must be positive")
    observed, estimate = _matching_arrays(noisy, prediction)
    if float(np.min(observed)) < 0 or float(np.min(estimate)) < 0:
        raise ValueError("Intensity ratio requires non-negative arrays")
    clipped = estimate < eps
    ratio = observed / np.maximum(estimate, eps)
    if not np.isfinite(ratio).all():
        raise ValueError("Intensity ratio contains NaN or Inf")
    return ratio, float(np.mean(clipped))


def ratio_mean_bias(
    noisy: np.ndarray,
    prediction: np.ndarray,
    eps: float = DEFAULT_RATIO_EPS,
) -> float:
    """Absolute ratio-mean bias ``abs(mean(noisy/prediction) - 1)``."""

    ratio, _ = intensity_ratio(noisy, prediction, eps=eps)
    return float(abs(float(np.mean(ratio)) - 1.0))


def ratio_acf_sidelobe_energy_from_ratio(
    ratio: np.ndarray,
    radius: int = DEFAULT_ACF_RADIUS,
    eps: float = 1e-12,
) -> float:
    """Mean absolute normalized linear-ACF sidelobe in a central window.

    A zero-variance ratio has no measurable sidelobe structure and returns 0.
    The zero-lag coefficient is excluded; for radius 5 the mean contains 120
    coefficients, matching an 11 x 11 window minus its centre.
    """

    if radius <= 0:
        raise ValueError("radius must be positive")
    array = _as_finite_2d(ratio, "ratio")
    height, width = array.shape
    if height <= radius or width <= radius:
        raise ValueError(f"Ratio shape {array.shape} is too small for radius={radius}")
    centered = array - float(np.mean(array))
    zero_lag = float(np.sum(centered * centered))
    if zero_lag <= eps:
        return 0.0

    fft_shape = (2 * height - 1, 2 * width - 1)
    spectrum = np.fft.fft2(centered, s=fft_shape)
    correlation = np.fft.fftshift(np.fft.ifft2(spectrum * np.conj(spectrum)).real)
    correlation /= zero_lag
    cy, cx = height - 1, width - 1
    local = correlation[
        cy - radius : cy + radius + 1,
        cx - radius : cx + radius + 1,
    ].copy()
    local[radius, radius] = np.nan
    return float(np.nanmean(np.abs(local)))


def ratio_acf_sidelobe_energy(
    noisy: np.ndarray,
    prediction: np.ndarray,
    radius: int = DEFAULT_ACF_RADIUS,
    eps: float = DEFAULT_RATIO_EPS,
) -> float:
    ratio, _ = intensity_ratio(noisy, prediction, eps=eps)
    return ratio_acf_sidelobe_energy_from_ratio(ratio, radius=radius)


def sobel_gradient_correlation(
    noisy: np.ndarray,
    prediction: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """Pearson correlation of Sobel gradient magnitudes.

    This is a descriptive gradient-correlation diagnostic (``Sobel-GC``). It is
    **not** the ratio-gradient probability index (RGPI) and must not be labelled
    EPI/RGPI in paper tables.
    """

    observed, estimate = _matching_arrays(noisy, prediction)

    def magnitude(array: np.ndarray) -> np.ndarray:
        gx = sobel(array, axis=1, mode="reflect")
        gy = sobel(array, axis=0, mode="reflect")
        return np.hypot(gx, gy)

    first = magnitude(observed)
    second = magnitude(estimate)
    first -= float(np.mean(first))
    second -= float(np.mean(second))
    denominator = float(np.sqrt(np.sum(first * first) * np.sum(second * second)))
    if denominator <= eps:
        return float("nan")
    value = float(np.sum(first * second) / denominator)
    return float(np.clip(value, -1.0, 1.0))


def _roi_tuple(roi: Sequence[int] | Mapping[str, int]) -> tuple[str, int, int, int, int]:
    if isinstance(roi, Mapping):
        roi_id = str(roi.get("roi_id", ""))
        x = int(roi["x"])
        y = int(roi["y"])
        width = int(roi.get("width", roi.get("w", 0)))
        height = int(roi.get("height", roi.get("h", 0)))
    else:
        if len(roi) != 4:
            raise ValueError("ROI sequence must be (x, y, width, height)")
        roi_id = ""
        x, y, width, height = map(int, roi)
    if min(x, y) < 0 or width <= 0 or height <= 0:
        raise ValueError(f"Invalid ROI {(x, y, width, height)}")
    return roi_id, x, y, width, height


def enl_roi_details(
    image: np.ndarray,
    rois: Sequence[Sequence[int] | Mapping[str, int]],
    variance_ddof: int = 0,
    eps: float = 1e-12,
) -> list[dict[str, float | int | str]]:
    """Return auditable mean/variance/ENL values for every fixed ROI."""

    array = _as_finite_2d(image)
    if variance_ddof not in (0, 1):
        raise ValueError("variance_ddof must be 0 or 1")
    details: list[dict[str, float | int | str]] = []
    for index, roi in enumerate(rois):
        roi_id, x, y, width, height = _roi_tuple(roi)
        region = array[y : y + height, x : x + width]
        if region.shape != (height, width):
            raise ValueError(f"ROI {(x, y, width, height)} exceeds image {array.shape}")
        if region.size <= variance_ddof:
            raise ValueError("ROI is too small for requested variance_ddof")
        mean = float(np.mean(region))
        variance = float(np.var(region, ddof=variance_ddof))
        details.append(
            {
                "roi_id": roi_id or f"roi{index + 1:02d}",
                "x": x,
                "y": y,
                "width": width,
                "height": height,
                "mean": mean,
                "variance": variance,
                "variance_ddof": variance_ddof,
                "enl": float(mean * mean / max(variance, eps)),
            }
        )
    return details


def select_homogeneous_rois(
    noisy: np.ndarray,
    roi_size: int = 32,
    num_rois: int = 5,
    stride: int | None = None,
    min_mean: float = 0.03,
    max_clipped_fraction: float = 0.05,
    eps: float = 1e-12,
) -> list[dict[str, float | int | str]]:
    """Select deterministic noisy-only ROI coordinates using squared CV.

    Selection is method-independent and should be run once by the preparation
    script. The returned coordinates, scores, and noisy statistics must then be
    frozen and reused by every method. Human review is still recommended because
    low coefficient of variation does not prove semantic homogeneity.
    """

    array = _as_finite_2d(noisy, "noisy")
    if roi_size <= 1 or num_rois <= 0:
        raise ValueError("roi_size must exceed 1 and num_rois must be positive")
    stride = roi_size if stride is None else int(stride)
    if stride <= 0:
        raise ValueError("stride must be positive")
    height, width = array.shape
    candidates: list[dict[str, float | int | str]] = []
    for y in range(0, height - roi_size + 1, stride):
        for x in range(0, width - roi_size + 1, stride):
            region = array[y : y + roi_size, x : x + roi_size]
            mean = float(np.mean(region))
            variance = float(np.var(region))
            clipped_fraction = float(np.mean((region <= 0.0) | (region >= 1.0)))
            if mean < min_mean or clipped_fraction > max_clipped_fraction:
                continue
            score = variance / max(mean * mean, eps)
            candidates.append(
                {
                    "roi_id": "",
                    "x": x,
                    "y": y,
                    "width": roi_size,
                    "height": roi_size,
                    "selection_score_cv2": score,
                    "noisy_mean": mean,
                    "noisy_variance": variance,
                    "noisy_enl": float(mean * mean / max(variance, eps)),
                    "clipped_fraction": clipped_fraction,
                }
            )
    candidates.sort(
        key=lambda row: (
            float(row["selection_score_cv2"]),
            int(row["y"]),
            int(row["x"]),
        )
    )
    selected = candidates[:num_rois]
    for index, row in enumerate(selected, start=1):
        row["roi_id"] = f"roi{index:02d}"
    return selected


def parent_cluster_bootstrap(
    rows: Iterable[Mapping[str, object]],
    value_key: str,
    cluster_key: str = "parent_id",
    samples: int = 10_000,
    seed: int = 42,
) -> dict[str, float | int]:
    """Bootstrap a patch-level mean by resampling complete parent clusters."""

    if samples <= 0:
        raise ValueError("samples must be positive")
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        cluster = str(row[cluster_key])
        try:
            value = float(row[value_key])
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            grouped[cluster].append(value)
    grouped = {key: values for key, values in grouped.items() if values}
    if not grouped:
        raise ValueError(f"No finite values for {value_key!r}")

    cluster_ids = sorted(grouped)
    cluster_sums = np.asarray([np.sum(grouped[key]) for key in cluster_ids], dtype=np.float64)
    cluster_counts = np.asarray([len(grouped[key]) for key in cluster_ids], dtype=np.float64)
    all_values = np.concatenate(
        [np.asarray(grouped[key], dtype=np.float64) for key in cluster_ids]
    )
    rng = np.random.default_rng(seed)
    replicates = np.empty(samples, dtype=np.float64)
    cluster_count = len(cluster_ids)
    for index in range(samples):
        sampled = rng.integers(0, cluster_count, size=cluster_count)
        replicates[index] = float(
            np.sum(cluster_sums[sampled]) / np.sum(cluster_counts[sampled])
        )
    return {
        "num_patches": int(all_values.size),
        "num_parent_clusters": cluster_count,
        "patch_mean": float(np.mean(all_values)),
        "patch_median": float(np.median(all_values)),
        "parent_macro_mean": float(
            np.mean(cluster_sums / np.maximum(cluster_counts, 1.0))
        ),
        "ci95_low": float(np.percentile(replicates, 2.5)),
        "ci95_high": float(np.percentile(replicates, 97.5)),
        "bootstrap_samples": int(samples),
        "bootstrap_seed": int(seed),
    }
