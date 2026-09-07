"""Paper-faithful classical metrics for real-SAR despeckling evaluation.

This module implements two metrics that are intentionally kept separate from
the frozen ICSPS26 no-reference evaluator:

* EPI follows Equation (16) in Ma et al., Remote Sensing 2024, 16, 1992.
  The lower-right diagonal neighbour convention follows the authors' public
  ``EPD.m`` implementation.
* M follows Equations (1)--(3) and the parameter settings in Gomez et al.,
  Remote Sensing 2017, 9, 389.

The M index is undefined when the automatic textureless-window selection finds
no eligible window.  The paper does not define a fallback selection, so this
implementation never silently substitutes the closest windows.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


def _matching_finite_2d(
    noisy: np.ndarray, prediction: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    observed = np.asarray(noisy, dtype=np.float64).squeeze()
    estimate = np.asarray(prediction, dtype=np.float64).squeeze()
    if observed.ndim != 2 or estimate.ndim != 2:
        raise ValueError("EPI/M inputs must be two-dimensional")
    if observed.shape != estimate.shape:
        raise ValueError(f"Shape mismatch: {observed.shape} != {estimate.shape}")
    if not np.isfinite(observed).all() or not np.isfinite(estimate).all():
        raise ValueError("EPI/M inputs must contain only finite values")
    if float(np.min(observed)) < 0.0 or float(np.min(estimate)) < 0.0:
        raise ValueError("EPI/M require non-negative intensity images")
    return observed, estimate


def edge_preservation_index(
    noisy: np.ndarray,
    prediction: np.ndarray,
    *,
    eps: float = 1e-12,
) -> float:
    """Return the diagonal-neighbour EPI from Equation (16).

    Let ``I`` be the original speckled intensity image and ``u_hat`` the
    despeckled image.  With the authors' diagonal pairing convention,

    ``EPI = sum(abs(u_hat[:-1,:-1] - u_hat[1:,1:])) /
           sum(abs(I[:-1,:-1] - I[1:,1:]))``.

    A spatially constant noisy image has a zero denominator and therefore no
    defined EPI.
    """

    if eps <= 0:
        raise ValueError("eps must be positive")
    observed, estimate = _matching_finite_2d(noisy, prediction)
    if min(observed.shape) < 2:
        raise ValueError("EPI requires both image dimensions to be at least 2")
    denominator = float(
        np.sum(np.abs(observed[:-1, :-1] - observed[1:, 1:]), dtype=np.float64)
    )
    if denominator <= eps:
        return float("nan")
    numerator = float(
        np.sum(np.abs(estimate[:-1, :-1] - estimate[1:, 1:]), dtype=np.float64)
    )
    return numerator / denominator


def _integral_image(array: np.ndarray) -> np.ndarray:
    result = np.zeros((array.shape[0] + 1, array.shape[1] + 1), dtype=np.float64)
    result[1:, 1:] = np.cumsum(np.cumsum(array, axis=0), axis=1)
    return result


def _box_sum_valid(array: np.ndarray, height: int, width: int) -> np.ndarray:
    integral = _integral_image(array)
    return (
        integral[height:, width:]
        - integral[:-height, width:]
        - integral[height:, :-width]
        + integral[:-height, :-width]
    )


def _local_mean_variance(array: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    count = window * window
    total = _box_sum_valid(array, window, window)
    total_squared = _box_sum_valid(array * array, window, window)
    mean = total / count
    variance = np.maximum(total_squared / count - mean * mean, 0.0)
    return mean, variance


def _quantize_equal_width(array: np.ndarray, levels: int) -> np.ndarray:
    minimum = float(np.min(array))
    maximum = float(np.max(array))
    if maximum <= minimum:
        return np.zeros(array.shape, dtype=np.int16)
    scaled = (array - minimum) / (maximum - minimum)
    return np.minimum(np.floor(scaled * levels), levels - 1).astype(np.int16)


def _mean_horizontal_haralick_homogeneity(quantized: np.ndarray, window: int) -> float:
    """Mean local GLCM inverse-difference moment for offset (0, 1)."""

    pair_weights = 1.0 / (
        1.0 + np.square(quantized[:, :-1] - quantized[:, 1:], dtype=np.float64)
    )
    local = _box_sum_valid(pair_weights, window, window - 1) / (window * (window - 1))
    return float(np.mean(local))


@dataclass(frozen=True)
class MIndexResult:
    value: float
    valid: bool
    reason: str
    selected_windows: int
    total_windows: int
    first_order_residual: float
    delta_h: float
    original_homogeneity: float
    shuffled_homogeneity_mean: float
    window: int
    tolerance: float
    levels: int
    shuffles: int
    seed: int
    ratio_denominator_floor_fraction: float

    def to_dict(self) -> dict[str, float | int | bool | str]:
        return asdict(self)


def m_index_paper(
    noisy: np.ndarray,
    prediction: np.ndarray,
    *,
    window: int = 25,
    tolerance: float = 0.03,
    levels: int = 8,
    shuffles: int = 100,
    seed: int = 42,
    ratio_eps: float = 1e-6,
    variance_eps: float = 1e-12,
) -> MIndexResult:
    """Compute the M index from Gomez et al. without an invented fallback.

    The published experimental settings are ``window=25``,
    ``tolerance=0.03``, ``levels=8``, and ``shuffles=100``.  Quantization uses
    equal-width bins and the GLCM uses the conventional horizontal offset.
    ``value`` is NaN when no automatically selected textureless window exists.
    """

    observed, estimate = _matching_finite_2d(noisy, prediction)
    if window <= 1 or min(observed.shape) < window:
        raise ValueError("M-index window must fit inside both image dimensions")
    if not 0.0 <= tolerance < 1.0:
        raise ValueError("tolerance must lie in [0,1)")
    if levels < 2 or shuffles <= 0 or ratio_eps <= 0 or variance_eps <= 0:
        raise ValueError("levels/shuffles/eps parameters are invalid")

    floored = estimate < ratio_eps
    ratio = observed / np.maximum(estimate, ratio_eps)
    noisy_mean, noisy_variance = _local_mean_variance(observed, window)
    ratio_mean, ratio_variance = _local_mean_variance(ratio, window)
    noisy_enl = noisy_mean * noisy_mean / np.maximum(noisy_variance, variance_eps)
    ratio_enl = ratio_mean * ratio_mean / np.maximum(ratio_variance, variance_eps)
    r_enl = np.abs(noisy_enl - ratio_enl) / np.maximum(noisy_enl, variance_eps)
    r_mean = np.abs(1.0 - ratio_mean)
    finite = np.isfinite(r_enl) & np.isfinite(r_mean) & (noisy_enl > 0.0)
    selected = finite & (r_enl <= tolerance) & (r_mean <= tolerance)
    selected_count = int(np.sum(selected))
    total_count = int(selected.size)

    common = {
        "selected_windows": selected_count,
        "total_windows": total_count,
        "window": int(window),
        "tolerance": float(tolerance),
        "levels": int(levels),
        "shuffles": int(shuffles),
        "seed": int(seed),
        "ratio_denominator_floor_fraction": float(np.mean(floored)),
    }
    if selected_count == 0:
        return MIndexResult(
            value=float("nan"),
            valid=False,
            reason="no_textureless_window_satisfies_paper_tolerance",
            first_order_residual=float("nan"),
            delta_h=float("nan"),
            original_homogeneity=float("nan"),
            shuffled_homogeneity_mean=float("nan"),
            **common,
        )

    first_order = 0.5 * float(np.mean((r_enl + r_mean)[selected]))
    quantized = _quantize_equal_width(ratio, levels)
    original_h = _mean_horizontal_haralick_homogeneity(quantized, window)
    rng = np.random.default_rng(seed)
    flat = quantized.ravel()
    shuffled_values = np.empty(shuffles, dtype=np.float64)
    for index in range(shuffles):
        shuffled = rng.permutation(flat).reshape(quantized.shape)
        shuffled_values[index] = _mean_horizontal_haralick_homogeneity(
            shuffled, window
        )
    shuffled_mean = float(np.mean(shuffled_values))
    if original_h <= variance_eps:
        return MIndexResult(
            value=float("nan"),
            valid=False,
            reason="original_homogeneity_is_zero",
            first_order_residual=first_order,
            delta_h=float("nan"),
            original_homogeneity=original_h,
            shuffled_homogeneity_mean=shuffled_mean,
            **common,
        )
    delta_h = 100.0 * abs(original_h - shuffled_mean) / original_h
    return MIndexResult(
        value=first_order + delta_h,
        valid=True,
        reason="",
        first_order_residual=first_order,
        delta_h=delta_h,
        original_homogeneity=original_h,
        shuffled_homogeneity_mean=shuffled_mean,
        **common,
    )
