"""Canonical numeric-domain transforms for the reproducible SAR protocol.

The paper-facing protocol uses normalized intensity throughout.  Legacy
amplitude conversion remains available only to reproduce historical runs.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch


INTENSITY_DOMAIN = "intensity_v1"
LEGACY_AMPLITUDE_DOMAIN = "legacy_amplitude_v0"
NUMERIC_DOMAIN_CHOICES = (INTENSITY_DOMAIN, LEGACY_AMPLITUDE_DOMAIN)
LOG_ALPHA = 10.0
REAL_LOW_PERCENTILE = 1.0
REAL_HIGH_PERCENTILE = 99.0


def validate_numeric_domain(domain: str) -> str:
    if domain not in NUMERIC_DOMAIN_CHOICES:
        raise ValueError(
            f"Unknown numeric domain {domain!r}; expected one of "
            f"{NUMERIC_DOMAIN_CHOICES}."
        )
    return domain


def as_single_channel_float32(array: np.ndarray) -> np.ndarray:
    """Return a finite HxW float32 array without changing its value range."""
    image = np.asarray(array, dtype=np.float32)
    while image.ndim > 2 and 1 in image.shape:
        image = np.squeeze(image, axis=image.shape.index(1))
    if image.ndim == 3:
        if image.shape[-1] == 3:
            image = (
                0.299 * image[..., 0]
                + 0.587 * image[..., 1]
                + 0.114 * image[..., 2]
            )
        elif image.shape[0] == 1:
            image = image[0]
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")
    if image.ndim != 2:
        raise ValueError(f"Expected a 2-D single-channel image, got {image.shape}")
    if not np.isfinite(image).all():
        raise ValueError("Image contains NaN or Inf values")
    return image.astype(np.float32, copy=False)


def clip_intensity_01(array: np.ndarray) -> np.ndarray:
    """Clip normalized intensity data to the network/metric range [0, 1]."""
    return np.clip(as_single_channel_float32(array), 0.0, 1.0).astype(np.float32)


def prepare_synthetic_pair(
    noisy: np.ndarray,
    clean: np.ndarray,
    domain: str = INTENSITY_DOMAIN,
) -> Tuple[np.ndarray, np.ndarray]:
    """Map an on-disk synthetic pair into the selected network domain."""
    validate_numeric_domain(domain)
    noisy_01 = clip_intensity_01(noisy)
    clean_01 = clip_intensity_01(clean)
    if domain == LEGACY_AMPLITUDE_DOMAIN:
        noisy_01 = np.sqrt(noisy_01)
        clean_01 = np.sqrt(clean_01)
    return noisy_01.astype(np.float32), clean_01.astype(np.float32)


def normalize_real_intensity(
    array: np.ndarray,
    low_percentile: float = REAL_LOW_PERCENTILE,
    high_percentile: float = REAL_HIGH_PERCENTILE,
) -> np.ndarray:
    """Apply the same robust intensity normalization for AMS and evaluation."""
    if not 0.0 <= low_percentile < high_percentile <= 100.0:
        raise ValueError("Expected 0 <= low_percentile < high_percentile <= 100")
    image = as_single_channel_float32(array)
    low = float(np.percentile(image, low_percentile))
    high = float(np.percentile(image, high_percentile))
    if high <= low:
        raise ValueError("Cannot normalize a constant or percentile-degenerate image")
    image = np.clip(image, low, high)
    image = (image - low) / (high - low)
    return np.clip(image, 0.0, 1.0).astype(np.float32)


def log_transform_01_torch(x: torch.Tensor, alpha: float = LOG_ALPHA) -> torch.Tensor:
    """Normalized log1p transform mapping [0, 1] exactly onto [0, 1]."""
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    x = torch.clamp(x, 0.0, 1.0)
    return torch.log1p(alpha * x) / math.log1p(alpha)


def inverse_log_transform_01_torch(
    log_x: torch.Tensor, alpha: float = LOG_ALPHA
) -> torch.Tensor:
    """Inverse of :func:`log_transform_01_torch` in normalized intensity."""
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    log_x = torch.clamp(log_x, 0.0, 1.0)
    return torch.expm1(log_x * math.log1p(alpha)) / alpha


def log_transform_01_numpy(array: np.ndarray, alpha: float = LOG_ALPHA) -> np.ndarray:
    image = clip_intensity_01(array)
    return (np.log1p(alpha * image) / math.log1p(alpha)).astype(np.float32)


def inverse_log_transform_01_numpy(
    array: np.ndarray, alpha: float = LOG_ALPHA
) -> np.ndarray:
    log_image = clip_intensity_01(array)
    return (np.expm1(log_image * math.log1p(alpha)) / alpha).astype(np.float32)


def intensity_to_display_amplitude(array: np.ndarray) -> np.ndarray:
    """Optional display-only mapping; never use this output for metrics."""
    return np.sqrt(clip_intensity_01(array)).astype(np.float32)
