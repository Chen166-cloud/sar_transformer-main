"""Single metric implementation for training, synthetic test, and real SAR."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from scipy.ndimage import gaussian_filter, sobel

from numeric_domain import clip_intensity_01


DATA_RANGE = 1.0
SSIM_WINDOW = 11
SSIM_SIGMA = 1.5
DEFAULT_BORDER_CROP = 0


def _crop_border(image: np.ndarray, border: int) -> np.ndarray:
    if border < 0:
        raise ValueError("border must be non-negative")
    if border == 0:
        return image
    if min(image.shape) <= 2 * border:
        raise ValueError(f"Image {image.shape} is too small for border={border}")
    return image[border:-border, border:-border]


def psnr(
    prediction: np.ndarray,
    target: np.ndarray,
    data_range: float = DATA_RANGE,
    border: int = DEFAULT_BORDER_CROP,
) -> float:
    pred = _crop_border(clip_intensity_01(prediction), border).astype(np.float64)
    ref = _crop_border(clip_intensity_01(target), border).astype(np.float64)
    if pred.shape != ref.shape:
        raise ValueError(f"Shape mismatch: {pred.shape} vs {ref.shape}")
    mse = float(np.mean((pred - ref) ** 2))
    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10((data_range * data_range) / mse))


def ssim(
    prediction: np.ndarray,
    target: np.ndarray,
    data_range: float = DATA_RANGE,
    border: int = DEFAULT_BORDER_CROP,
) -> float:
    """Windowed SSIM with an 11x11 Gaussian window and valid support."""
    pred = _crop_border(clip_intensity_01(prediction), border).astype(np.float64)
    ref = _crop_border(clip_intensity_01(target), border).astype(np.float64)
    if pred.shape != ref.shape:
        raise ValueError(f"Shape mismatch: {pred.shape} vs {ref.shape}")
    if min(pred.shape) < SSIM_WINDOW:
        raise ValueError(f"SSIM requires images at least {SSIM_WINDOW}x{SSIM_WINDOW}")

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    truncate = ((SSIM_WINDOW - 1) / 2) / SSIM_SIGMA
    mu_x = gaussian_filter(pred, SSIM_SIGMA, mode="reflect", truncate=truncate)
    mu_y = gaussian_filter(ref, SSIM_SIGMA, mode="reflect", truncate=truncate)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y
    sigma_x2 = gaussian_filter(pred * pred, SSIM_SIGMA, mode="reflect", truncate=truncate) - mu_x2
    sigma_y2 = gaussian_filter(ref * ref, SSIM_SIGMA, mode="reflect", truncate=truncate) - mu_y2
    sigma_xy = gaussian_filter(pred * ref, SSIM_SIGMA, mode="reflect", truncate=truncate) - mu_xy
    numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    score = numerator / np.maximum(denominator, np.finfo(np.float64).eps)
    pad = SSIM_WINDOW // 2
    return float(np.mean(score[pad:-pad, pad:-pad]))


def batch_psnr_ssim(
    prediction: torch.Tensor,
    target: torch.Tensor,
    border: int = DEFAULT_BORDER_CROP,
) -> Tuple[float, float]:
    """Evaluate BCHW tensors with exactly the same NumPy metric functions."""
    pred = prediction.detach().float().cpu().numpy()
    ref = target.detach().float().cpu().numpy()
    if pred.shape != ref.shape or pred.ndim != 4 or pred.shape[1] != 1:
        raise ValueError(f"Expected matching Bx1xHxW tensors, got {pred.shape}/{ref.shape}")
    psnr_values = [psnr(p[0], r[0], border=border) for p, r in zip(pred, ref)]
    ssim_values = [ssim(p[0], r[0], border=border) for p, r in zip(pred, ref)]
    return float(np.mean(psnr_values)), float(np.mean(ssim_values))


Roi = Tuple[int, int, int, int]


def select_homogeneous_rois(
    noisy: np.ndarray,
    roi_size: int = 32,
    num_rois: int = 5,
    min_mean: float = 0.03,
) -> List[Roi]:
    """Select coordinates once from noisy input for use by every compared image."""
    image = clip_intensity_01(noisy)
    height, width = image.shape
    candidates = []
    for y in range(0, height - roi_size + 1, roi_size):
        for x in range(0, width - roi_size + 1, roi_size):
            roi = image[y : y + roi_size, x : x + roi_size]
            mean = float(np.mean(roi))
            if mean > min_mean:
                candidates.append((float(np.var(roi)), (x, y, roi_size, roi_size)))
    candidates.sort(key=lambda item: (item[0], item[1]))
    return [roi for _, roi in candidates[:num_rois]]


def enl(image: np.ndarray, rois: Sequence[Roi], eps: float = 1e-8) -> float:
    image = clip_intensity_01(image)
    if not rois:
        return 0.0
    values = []
    for x, y, width, height in rois:
        region = image[y : y + height, x : x + width]
        if region.shape != (height, width):
            raise ValueError(f"ROI {(x, y, width, height)} exceeds image {image.shape}")
        values.append(float(np.mean(region) ** 2 / (np.var(region) + eps)))
    return float(np.mean(values))


def _integral_image(array: np.ndarray) -> np.ndarray:
    height, width = array.shape
    result = np.zeros((height + 1, width + 1), dtype=np.float64)
    result[1:, 1:] = np.cumsum(np.cumsum(array, axis=0), axis=1)
    return result


def _box_sum_valid(array: np.ndarray, kh: int, kw: int) -> np.ndarray:
    integral = _integral_image(array)
    return (
        integral[kh:, kw:]
        - integral[:-kh, kw:]
        - integral[kh:, :-kw]
        + integral[:-kh, :-kw]
    )


def _local_mean_var_valid(array: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    area = window * window
    total = _box_sum_valid(array, window, window)
    total2 = _box_sum_valid(array * array, window, window)
    mean = total / area
    variance = np.maximum(total2 / area - mean * mean, 0.0)
    return mean, variance


def _quantize(array: np.ndarray, levels: int) -> np.ndarray:
    low = float(np.min(array))
    high = float(np.max(array))
    if high <= low:
        return np.zeros_like(array, dtype=np.int32)
    quantized = np.floor((array - low) / (high - low) * levels)
    return np.clip(quantized, 0, levels - 1).astype(np.int32)


def _local_homogeneity(q: np.ndarray, window: int, direction: str) -> float:
    if direction == "h":
        a, b, kh, kw = q[:, :-1], q[:, 1:], window, window - 1
    elif direction == "v":
        a, b, kh, kw = q[:-1, :], q[1:, :], window - 1, window
    else:
        raise ValueError("direction must be 'h' or 'v'")
    weights = 1.0 / (1.0 + (a - b) ** 2)
    return float(np.mean(_box_sum_valid(weights, kh, kw) / (kh * kw)))


def m_index(
    noisy: np.ndarray,
    prediction: np.ndarray,
    window: int = 25,
    tolerance: float = 0.03,
    levels: int = 8,
    shuffles: int = 100,
    seed: int = 42,
    direction: str = "h",
    eps: float = 1e-6,
) -> float:
    """Ratio-image M index, deterministic for a fixed shuffle seed."""
    noisy_01 = clip_intensity_01(noisy).astype(np.float64)
    pred_01 = clip_intensity_01(prediction).astype(np.float64)
    if noisy_01.shape != pred_01.shape or min(noisy_01.shape) < window:
        raise ValueError("M-index inputs must match and be at least window-sized")
    ratio = noisy_01 / np.maximum(pred_01, eps)

    noisy_mean, noisy_var = _local_mean_var_valid(noisy_01, window)
    ratio_mean, ratio_var = _local_mean_var_valid(ratio, window)
    noisy_enl = noisy_mean * noisy_mean / np.maximum(noisy_var, 1e-12)
    ratio_enl = ratio_mean * ratio_mean / np.maximum(ratio_var, 1e-12)
    r_enl = np.abs(noisy_enl - ratio_enl) / np.maximum(noisy_enl, 1e-12)
    r_mu = np.abs(1.0 - ratio_mean)
    finite = np.isfinite(r_enl) & np.isfinite(r_mu) & (noisy_enl > 0)
    selected = finite & (r_enl <= tolerance) & (r_mu <= tolerance)
    if not np.any(selected):
        combined = np.where(finite, r_enl + r_mu, np.inf)
        order = np.argsort(combined.ravel())
        order = order[np.isfinite(combined.ravel()[order])][:100]
        if not len(order):
            raise RuntimeError("No finite windows available for M-index")
        selected = np.zeros_like(combined, dtype=bool)
        selected.ravel()[order] = True
    first_order = 0.5 * float(np.mean((r_enl + r_mu)[selected]))

    quantized = _quantize(ratio, levels)
    observed = _local_homogeneity(quantized, window, direction)
    rng = np.random.default_rng(seed)
    flat = quantized.ravel()
    shuffled = [
        _local_homogeneity(rng.permutation(flat).reshape(quantized.shape), window, direction)
        for _ in range(shuffles)
    ]
    reference = float(np.mean(shuffled))
    delta_h = 100.0 * abs(observed - reference) / max(abs(observed), 1e-12)
    return float(first_order + delta_h)


def epi(noisy: np.ndarray, prediction: np.ndarray, eps: float = 1e-8) -> float:
    noisy_01 = clip_intensity_01(noisy)
    pred_01 = clip_intensity_01(prediction)
    edge_maps = []
    for image in (noisy_01, pred_01):
        gx = sobel(image, axis=1, mode="reflect")
        gy = sobel(image, axis=0, mode="reflect")
        edge_maps.append(np.sqrt(gx * gx + gy * gy))
    a = edge_maps[0] - float(np.mean(edge_maps[0]))
    b = edge_maps[1] - float(np.mean(edge_maps[1]))
    return float(np.sum(a * b) / (np.sqrt(np.sum(a * a) * np.sum(b * b)) + eps))


def real_sar_metrics(
    noisy: np.ndarray,
    prediction: np.ndarray,
    roi_size: int = 32,
    num_rois: int = 5,
    seed: int = 42,
) -> Dict[str, object]:
    rois = select_homogeneous_rois(noisy, roi_size=roi_size, num_rois=num_rois)
    return {
        "rois": [list(roi) for roi in rois],
        "noisy_enl": enl(noisy, rois),
        "prediction_enl": enl(prediction, rois),
        "m_index": m_index(noisy, prediction, seed=seed),
        "epi": epi(noisy, prediction),
    }
