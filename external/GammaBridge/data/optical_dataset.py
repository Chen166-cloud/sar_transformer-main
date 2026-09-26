"""Natural-image dataset for gamma-Bridge synthetic training.

Paper's setup (Sec. Experiments): training uses only BSDS500 train+val and
DIV2K train HR natural images -- 1,100 256x256 crops -- with synthetic
L_obs=1 Gamma corruption; no real SAR imagery is used.

* `OpticalTrainDataset` returns clean crops `x0`. The training loop
  applies Gamma corruption on the fly (fresh noise every step).
* `PairedEvalDataset` returns held-out (x0, x_obs) pairs with a fixed
  seed for deterministic PSNR / SSIM / ratio-statistic evaluation.

Grayscale, single channel, intensity in (0, 1]. Small floor keeps the
multiplicative model x = x_0 * N well-defined.
"""

from __future__ import annotations
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .download_bsds500 import ensure_bsds500

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.gamma_bridge import sample_gamma_noise  # noqa: E402


_IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def _load_gray(path: Path) -> np.ndarray:
    im = Image.open(path).convert("L")
    return np.asarray(im, dtype=np.float32) / 255.0


def _random_crop(img: np.ndarray, size: int, rng: random.Random) -> np.ndarray:
    H, W = img.shape
    if H < size or W < size:
        ph = max(0, size - H)
        pw = max(0, size - W)
        img = np.pad(img, ((0, ph), (0, pw)), mode="edge")
        H, W = img.shape
    y = rng.randint(0, H - size)
    x = rng.randint(0, W - size)
    return img[y : y + size, x : x + size]


def _prep(img: np.ndarray, eps: float = 0.05) -> torch.Tensor:
    """(H,W) in [0,1] -> (1,H,W) in [eps, 1] with a small floor so that
    the multiplicative Gamma channel is well-defined near zero."""
    img = np.clip(img, 0.0, 1.0)
    img = eps + (1.0 - eps) * img
    return torch.from_numpy(img).float().unsqueeze(0)


def _collect_images(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in _IMG_EXTS]


class OpticalTrainDataset(Dataset):
    """Random clean crops from BSDS500 train+val (+ optional extra pools such
    as DIV2K HR). One clean crop per index; the training loop samples fresh
    Gamma noise every step."""

    def __init__(
        self,
        crop_size: int = 256,
        split_seed: int = 0,
        length: int = 4000,
        extra_roots: Optional[list] = None,
    ):
        root = ensure_bsds500()
        pool = sorted(list((root / "train").glob("*.jpg")) + list((root / "val").glob("*.jpg")))
        if extra_roots:
            for r in extra_roots:
                rp = Path(r)
                if rp.exists():
                    pool.extend(sorted(_collect_images(rp)))
        rng = random.Random(split_seed)
        rng.shuffle(pool)
        self.pool = pool
        self.crop_size = int(crop_size)
        self.length = int(length)

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        rng = random.Random((idx + 1) * 2654435761 % 2**32)
        p = rng.choice(self.pool)
        img = _load_gray(p)
        crop = _random_crop(img, self.crop_size, rng)
        return {"x0": _prep(crop)}


class PairedEvalDataset(Dataset):
    """Held-out BSDS500 test crops with deterministic Gamma corruption at
    L_obs, used for PSNR / SSIM / ratio-statistic evaluation."""

    def __init__(self, crop_size: int = 256, L_obs: float = 1.0,
                 num_samples: int = 32, seed: int = 42):
        root = ensure_bsds500()
        self.paths = sorted((root / "test").glob("*.jpg"))
        self.crop_size = int(crop_size)
        self.L_obs = float(L_obs)
        self.num_samples = int(num_samples)
        self.seed = int(seed)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        rng = random.Random(self.seed + idx)
        path = self.paths[idx % len(self.paths)]
        crop = _random_crop(_load_gray(path), self.crop_size, rng)
        x0 = _prep(crop)
        L = torch.tensor([self.L_obs])
        dist = torch.distributions.Gamma(concentration=L.expand(x0.shape),
                                         rate=L.expand(x0.shape))
        state = torch.random.get_rng_state()
        torch.manual_seed(self.seed + idx * 7919)
        N = dist.sample()
        torch.random.set_rng_state(state)
        x_obs = (x0 * N).clamp_min(1e-6)
        return {"x0": x0, "x_obs": x_obs}
