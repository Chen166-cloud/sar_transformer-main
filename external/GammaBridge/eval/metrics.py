"""Evaluation metrics for the Gamma bridge.

Reference implementation: `SR_experiemts/compute_metrics.py` and
`compute_real_metrics.py` (Hu et al. 2025). All Hu-comparable metrics
operate on uint8 [0, 255] grayscale images and are single-image
properties of the denoised output (no explicit noisy-vs-denoised
comparison).

Full-reference (synthetic ground-truth), on uint8 [0, 255] with
per-image ``data_range = gt.max() - gt.min()`` (skimage convention):
    * PSNR — peak_signal_noise_ratio
    * SSIM — structural_similarity

Single-image (no reference, used on real SAR denoised outputs):
    * ENL   — (mean / std)^2
    * EPI   — Canny(u8, 100, 200) edge-pixel density
    * EPD_H — mean(|filter2D(f, [-1, 2, -1])|) / mean(f)
    * EPD_V — mean(|filter2D(f, [[-1],[2],[-1]])|) / mean(f)
    * SQI   — mean / (mean + std)
    * MOI   — mean / 255

Physics / ratio-image auxiliaries (used during model calibration,
kept for backward compatibility):
    * ratio mean, variance; KS test against Gamma(L_obs, 1/L_obs)
    * enl_homogeneous — patch-based P90 ENL (used for L̂ estimation)
"""

from __future__ import annotations
import math
import numpy as np
import cv2
from scipy import stats
from skimage.metrics import structural_similarity as ssim_fn
from skimage.metrics import peak_signal_noise_ratio as psnr_fn


def psnr(x_ref: np.ndarray, x_hat: np.ndarray, max_val: float | None = None) -> float:
    """PSNR matching Hu 2025 convention when max_val is None:
    data_range is taken from the ground truth's actual span
    (``gt.max() - gt.min()``). Pass ``max_val`` explicitly to override."""
    if max_val is None:
        dr = float(x_ref.max() - x_ref.min())
        if dr <= 0:
            return float("inf")
        return float(psnr_fn(x_ref, x_hat, data_range=dr))
    x_ref = np.clip(x_ref, 0.0, max_val)
    x_hat = np.clip(x_hat, 0.0, max_val)
    mse = float(np.mean((x_ref - x_hat) ** 2))
    if mse <= 0:
        return float("inf")
    return 10.0 * math.log10((max_val ** 2) / mse)


def ssim(x_ref: np.ndarray, x_hat: np.ndarray, data_range: float | None = None) -> float:
    """SSIM matching Hu 2025 convention when data_range is None:
    per-image data_range from ``x_ref.max() - x_ref.min()``."""
    if data_range is None:
        dr = float(x_ref.max() - x_ref.min())
    else:
        dr = float(data_range)
    if dr <= 0:
        return 1.0
    return float(ssim_fn(x_ref, x_hat, data_range=dr))


def ratio_stats(x_obs: np.ndarray, x_hat: np.ndarray, L_obs: float) -> dict:
    r = (x_obs / np.maximum(x_hat, 1e-6)).ravel()
    m = float(r.mean())
    v = float(r.var(ddof=1))
    theo_var = 1.0 / L_obs
    # KS against Gamma(L, 1/L).  We drop obvious outliers to avoid huge tail
    # contributions dominating a would-be pass; the theoretical Gamma has
    # infinite support so we keep a wide clip.
    r_c = r[(r > 1e-6) & (r < 1e3)]
    stat, p = stats.kstest(r_c, "gamma", args=(L_obs, 0.0, 1.0 / L_obs))
    return {
        "ratio_mean": m,
        "ratio_var": v,
        "ratio_var_theo": theo_var,
        "ratio_var_rel_err": abs(v - theo_var) / max(theo_var, 1e-8),
        "ratio_ks_stat": float(stat),
        "ratio_ks_p": float(p),
    }


def enl(patch: np.ndarray) -> float:
    """Equivalent Number of Looks estimator on a homogeneous patch."""
    m = float(patch.mean())
    s = float(patch.std(ddof=1))
    if s <= 0:
        return float("inf")
    return (m / s) ** 2


def enl_homogeneous(x: np.ndarray, patch_size: int = 32, stride: int = 16,
                    quantile: float = 0.90) -> float:
    """ENL from the top-decile most-homogeneous patches.

    Slides `patch_size`×`patch_size` windows at `stride`, computes
    (mean/std)^2 for each, returns the P90 quantile — patches with
    highest ENL are the flattest regions in the scene, matching the
    SAR-community convention of selecting a manually chosen "flat"
    region and reporting its ENL as an estimate of the effective number
    of looks."""
    H, W = x.shape[-2:]
    if H < patch_size or W < patch_size:
        return enl(x)
    enls = []
    for i in range(0, H - patch_size + 1, stride):
        row = x[i:i + patch_size]
        for j in range(0, W - patch_size + 1, stride):
            p = row[:, j:j + patch_size]
            m = float(p.mean())
            s = float(p.std(ddof=1))
            if m > 1e-3 and s > 1e-6:
                enls.append((m / s) ** 2)
    if not enls:
        return float("nan")
    return float(np.quantile(np.asarray(enls), quantile))


def enl_whole(img_u8: np.ndarray) -> float:
    """Hu 2025 ENL (paper Tables V–X): ``(mean/std)^2`` on uint8 grayscale.
    Reference: ``SR_experiemts/ENL.ipynb`` ``calculate_enl``. Image is
    read via ``skimage.io.imread(..., as_gray=True)`` (uint8 [0, 255])
    and cast to float64 before the reduction."""
    f = img_u8.astype(np.float64)
    m, s = float(np.mean(f)), float(np.std(f))
    return (m * m) / (s * s) if s > 0 else float("inf")


def epi_cv(img_u8: np.ndarray) -> float:
    """Hu 2025 EPI (paper Tables V–X): coefficient of variation
    ``std / mean`` on uint8 grayscale. Reference:
    ``SR_experiemts/ENL.ipynb`` ``calculate_epi``. This is the
    inverse of ENL^{1/2}, not a Canny edge density."""
    f = img_u8.astype(np.float64)
    m = float(np.mean(f))
    if m <= 0:
        return 0.0
    return float(np.std(f) / m)


def epd_diff(img_u8: np.ndarray, direction: str = "H") -> float:
    """Hu 2025 EPD-ROA (paper Tables V–X): mean absolute adjacent-
    pixel difference along a single axis, on uint8 grayscale float.
    Reference: ``SR_experiemts/ENL.ipynb`` ``calculate_epd_roa``.
    Note: this is NOT a ratio; it is the raw mean-gradient magnitude
    in the [0, 255] domain. Higher = more edges (denoised output
    still retains sharp gradients)."""
    f = img_u8.astype(np.float64)
    axis = 1 if direction == "H" else 0
    return float(np.mean(np.abs(np.diff(f, axis=axis))))


def sqi_secondmoment(img_u8: np.ndarray) -> float:
    """Legacy: second-moment ratio ``E[X^2] / E[X]^2`` (ENL.ipynb def).
    Value in [1, inf). Kept for internal diagnostics only; paper
    Tables V-X use the Sun 2011 form below."""
    f = img_u8.astype(np.float64)
    m = float(np.mean(f))
    if m <= 0:
        return 0.0
    return float(np.mean(f * f) / (m * m))


def sqi_sun(img_u8: np.ndarray) -> float:
    """SQI reported in Hu 2025 paper Tables V-X: ``mean / (mean + std)``
    on uint8 grayscale (Sun et al. 2011). Value in [0, 1]; higher =
    output is mean-dominant relative to residual noise. Reference
    (Hu code): ``compute_real_metrics.py`` ``compute_sqi``. Paper
    values align with this formula in scale and ordering, though
    small round-off differences of $\\pm 0.02$ exist versus the
    ENL.ipynb variant."""
    f = img_u8.astype(np.float64)
    m, s = float(np.mean(f)), float(np.std(f))
    return m / (m + s) if (m + s) > 0 else 0.0


def moi_varmean(img_u8: np.ndarray) -> float:
    """Hu 2025 MOI-style: ``Var[X] / E[X]`` on uint8 grayscale float.
    Reference: ``SR_experiemts/ENL.ipynb`` ``calculate_moi``. Distinct
    from the paper's "Mean" column, which is the mean intensity mapped
    to the [0, 100] convention (see ``mean_100_convention``)."""
    f = img_u8.astype(np.float64)
    m = float(np.mean(f))
    if m <= 0:
        return 0.0
    return float(np.var(f) / m)


def mean_100_convention(img_u8: np.ndarray) -> float:
    """Hu 2025 paper "Mean" column: uint8 mean rescaled to [0, 100],
    i.e. ``mean * 100 / 255``. Matches the noisy-input Mean headers
    in paper Tables V-X (e.g. Sentinel-1 MEAN=34.39)."""
    return float(np.mean(img_u8) * 100.0 / 255.0)


# ── Backward-compatibility wrappers: 2-argument (noisy, denoised)
#    signatures that many call sites in the codebase still use. These
#    all discard the noisy argument and defer to the Hu-style single-
#    image metric on the denoised output. Keep for API stability.

def epi(noisy: np.ndarray, denoised: np.ndarray) -> float:  # noqa: F811
    """Hu-style EPI on the denoised output (uint8-conversion inside)."""
    return epi_canny(_to_u8(denoised))


def sqi(noisy: np.ndarray, denoised: np.ndarray) -> float:  # noqa: F811
    """Hu-style SQI on the denoised output."""
    return sqi_hu(_to_u8(denoised))


def epd_roa(noisy: np.ndarray, denoised: np.ndarray) -> float:
    """Hu-style EPD (H+V averaged) on the denoised output."""
    u8 = _to_u8(denoised)
    return 0.5 * (epd_lap(u8, "H") + epd_lap(u8, "V"))


def epd_roa_directional(noisy: np.ndarray, denoised: np.ndarray,
                        direction: str = "H") -> float:
    return epd_lap(_to_u8(denoised), direction)


def _to_u8(x: np.ndarray) -> np.ndarray:
    """Convert float [0,1] to uint8 [0,255] if needed; leave uint8 alone."""
    if x.dtype == np.uint8:
        return x
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)
