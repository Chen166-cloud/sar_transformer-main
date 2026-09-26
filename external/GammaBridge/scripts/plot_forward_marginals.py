"""Visualise forward Gamma-bridge marginals on a synthetic clean image.

Produces figures/forward_marginals.png with two rows:
  Row 1: x_t at increasing t (t = 0, 0.25, 0.5, 0.75, 1.0), L(t) annotated.
  Row 2: histogram of ratio image (x_t / x_0) vs theoretical Gamma(L(t), 1/L(t)) pdf.

Run: python scripts/plot_forward_marginals.py
"""

from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import stats

from models.gamma_bridge import gamma_corrupt, looks_schedule


def make_synthetic_scene(H=256, W=256):
    """Piecewise-constant + gradient — mimics SAR-like intensity structure."""
    x = np.zeros((H, W), dtype=np.float32)
    x += np.linspace(0.2, 0.7, W)[None, :]
    x[40:100, 40:180] = 0.9
    x[130:200, 60:120] = 0.4
    x[150:220, 150:230] = 1.0
    return x


def main():
    L_obs = 1.0
    L_max = 1.0e4
    ts = [0.0, 0.25, 0.5, 0.75, 1.0]

    x0_np = make_synthetic_scene()
    x0 = torch.from_numpy(x0_np)[None, None]  # (1,1,H,W)

    fig, axes = plt.subplots(2, len(ts), figsize=(3 * len(ts), 6))
    for i, tval in enumerate(ts):
        t = torch.tensor([tval])
        x_t, L_t = gamma_corrupt(x0, t, L_obs=L_obs, L_max=L_max)
        L_val = float(L_t.item())
        x_t_np = x_t[0, 0].numpy()

        ax = axes[0, i]
        vmax = np.percentile(x_t_np, 99.5)
        ax.imshow(x_t_np, cmap="gray", vmin=0, vmax=vmax)
        ax.set_title(f"t={tval:.2f}   L(t)={L_val:.1f}", fontsize=10)
        ax.axis("off")

        # Ratio image and histogram vs theoretical Gamma(L_val, 1/L_val).
        ratio = (x_t_np / x0_np).ravel()
        ax = axes[1, i]
        ax.hist(ratio, bins=80, density=True, alpha=0.55, label="empirical")
        xs = np.linspace(max(ratio.min(), 1e-3), min(ratio.max(), 5), 400)
        pdf = stats.gamma.pdf(xs, a=L_val, scale=1.0 / L_val)
        ax.plot(xs, pdf, "r-", lw=1.5, label=f"Gamma(L={L_val:.1f})")
        ax.set_xlim(0, min(5.0, ratio.max()))
        ax.set_xlabel("ratio y/x")
        if i == 0:
            ax.set_ylabel("density")
        ax.legend(fontsize=8)

    fig.suptitle(
        f"Forward Gamma-bridge marginals  (L_obs={L_obs}, L_max={L_max:.0e})",
        fontsize=12,
    )
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures", "forward_marginals.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
