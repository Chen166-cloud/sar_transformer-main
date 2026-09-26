"""Numerical sanity checks for the forward Gamma bridge kernel.

Verifies:
  1. Schedule endpoints: L(0)=L_max, L(1)=L_obs; monotone decreasing.
  2. Mean preservation: E[N_t] ≈ 1 => E[Y] = X.
  3. Variance law: Var[N_t] ≈ 1/L(t).
  4. Distribution shape: KS test on N samples vs. Gamma(L, 1/L) CDF.
  5. Gradient flow: d loss / d x0 exists and is finite.
  6. Embedding shape and finiteness.

Run: python tests/test_gamma_kernel.py
"""

from __future__ import annotations
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from scipy import stats

from models.gamma_bridge import (
    looks_schedule,
    sample_gamma_noise,
    gamma_corrupt,
    embed_looks,
)


def _approx(a, b, tol):
    return abs(float(a) - float(b)) <= tol


def test_schedule_endpoints():
    L_obs, L_max = 1.0, 1e4
    t = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    L = looks_schedule(t, L_obs=L_obs, L_max=L_max, mode="log")
    assert _approx(L[0].item(), L_max, tol=1.0), f"L(0)={L[0]} != {L_max}"
    assert _approx(L[-1].item(), L_obs, tol=1e-3), f"L(1)={L[-1]} != {L_obs}"
    # Monotone decreasing.
    diffs = (L[1:] - L[:-1]).tolist()
    assert all(d < 0 for d in diffs), f"L(t) not monotone dec: {L.tolist()}"
    print(f"[OK] schedule endpoints: L={L.tolist()}")


def test_mean_and_variance(L_obs=1.0, L_probe=(1.0, 4.0, 16.0, 100.0), n=200_000, tol=0.01):
    torch.manual_seed(0)
    for L_val in L_probe:
        L = torch.tensor([L_val])
        N = sample_gamma_noise(L, shape=(1, 1, n, 1), device="cpu")
        mean = N.mean().item()
        var = N.var(unbiased=True).item()
        theo_var = 1.0 / L_val
        assert abs(mean - 1.0) < tol, f"L={L_val}: mean={mean:.4f} != 1"
        rel_err = abs(var - theo_var) / theo_var
        assert rel_err < 0.05, f"L={L_val}: var={var:.5f} vs theo {theo_var:.5f} (rel_err={rel_err:.3f})"
        print(f"[OK] L={L_val:>6.1f}  mean={mean:.4f}  var={var:.5f}  theo_var={theo_var:.5f}")


def test_ks_gamma_marginal(L_probe=(1.0, 4.0, 16.0), n=50_000, p_thresh=1e-3):
    """KS test: samples from sample_gamma_noise vs Gamma(L, 1/L) CDF.

    scipy.stats.gamma param: shape a=L, scale=1/L (so mean=a*scale=1).
    We require p > p_thresh (very loose — with 50k samples KS is powerful).
    """
    torch.manual_seed(1)
    for L_val in L_probe:
        L = torch.tensor([L_val])
        N = sample_gamma_noise(L, shape=(n,), device="cpu").numpy()
        stat, p = stats.kstest(N, "gamma", args=(L_val, 0.0, 1.0 / L_val))
        assert p > p_thresh, f"L={L_val}: KS reject, p={p:.2e}, stat={stat:.4f}"
        print(f"[OK] L={L_val:>6.1f}  KS stat={stat:.4f}  p={p:.3f}")


def test_gamma_corrupt_shapes_and_grad():
    torch.manual_seed(2)
    B, C, H, W = 4, 1, 32, 32
    x0 = (torch.rand(B, C, H, W) + 0.1).requires_grad_(True)
    t = torch.tensor([0.1, 0.4, 0.7, 1.0])
    x_t, L_t = gamma_corrupt(x0, t, L_obs=1.0)
    assert x_t.shape == x0.shape
    assert L_t.shape == (B,)
    # Gradient flow through the noised image back to x0.
    loss = x_t.mean()
    loss.backward()
    g = x0.grad
    assert g is not None and torch.isfinite(g).all(), "gradient not finite"
    assert (g > 0).all(), "since N>0, d(mean x_t)/d(x0) should be positive"
    print(f"[OK] gamma_corrupt shapes/grad: x_t {tuple(x_t.shape)}  L_t {L_t.tolist()}")


def test_ratio_image_statistic(L_obs=4.0, H=256, W=256, tol_mean=0.01, tol_var=0.05):
    """End-to-end: given a fixed x0, y = x0 * N, ratio = y/x0 should be Gamma(L,1/L)."""
    torch.manual_seed(3)
    # Non-trivial "clean" image: smooth gradient + a couple of piecewise-constant patches.
    x0 = torch.linspace(0.2, 1.0, H).view(1, 1, H, 1).expand(1, 1, H, W).clone()
    x0[..., 64:128, 64:128] = 0.6
    x0[..., 150:200, 30:220] = 0.9
    t = torch.tensor([1.0])  # at t=1, L(t)=L_obs
    x_t, L_t = gamma_corrupt(x0, t, L_obs=L_obs)
    assert _approx(L_t.item(), L_obs, tol=1e-3)
    ratio = (x_t / x0).flatten().numpy()
    mean, var = float(ratio.mean()), float(ratio.var(ddof=1))
    assert abs(mean - 1.0) < tol_mean, f"ratio mean={mean:.4f}"
    theo_var = 1.0 / L_obs
    assert abs(var - theo_var) / theo_var < tol_var, f"ratio var={var:.4f} vs {theo_var:.4f}"
    # Also KS on ratio.
    stat, p = stats.kstest(ratio, "gamma", args=(L_obs, 0.0, 1.0 / L_obs))
    assert p > 1e-3, f"ratio KS p={p:.2e}"
    print(f"[OK] ratio-image stats: mean={mean:.4f} var={var:.5f} KS p={p:.3f}")


def test_embed_looks():
    L = torch.tensor([1.0, 10.0, 100.0, 10000.0])
    emb = embed_looks(L, dim=128)
    assert emb.shape == (4, 128)
    assert torch.isfinite(emb).all()
    # Distinct L should give distinct embeddings.
    for i in range(len(L)):
        for j in range(i + 1, len(L)):
            d = (emb[i] - emb[j]).norm().item()
            assert d > 1e-3, f"embeddings collided: L={L[i].item()} vs {L[j].item()}"
    print(f"[OK] embed_looks: shape {tuple(emb.shape)}, all distinct, finite")


def main():
    test_schedule_endpoints()
    test_mean_and_variance()
    test_ks_gamma_marginal()
    test_gamma_corrupt_shapes_and_grad()
    test_ratio_image_statistic()
    test_embed_looks()
    print("\nAll forward-kernel sanity checks passed.")


if __name__ == "__main__":
    main()
