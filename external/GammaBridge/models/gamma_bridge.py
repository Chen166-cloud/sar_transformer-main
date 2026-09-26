"""Exact L-look Gamma diffusion bridge — forward kernel and scheduling.

Physics recap:
    Speckle model:   Y = X * N,   N ~ Gamma(shape=L, rate=L)  =>  E[N]=1, Var[N]=1/L
    Bridge forward:  x_t = X * N_t,  N_t ~ Gamma(L(t), L(t))
                     L(0)=L_max (~clean),  L(1)=L_obs (observed)
    Reverse target:  network predicts x_0 given (x_t, L(t))

All ops are torch, batched, differentiable w.r.t. x0. L(t) is treated as a physical
schedule (not a learned parameter). Sampling from Gamma with rsample() gives a
pathwise-differentiable estimator that we do NOT propagate through here — the noise
is sampled with no_grad w.r.t. its own shape parameter, but multiplication with x0
still lets gradients flow into x0 (which is what training needs).
"""

from __future__ import annotations
import math
import torch
from torch import Tensor


def looks_schedule(
    t: Tensor,
    L_obs: float,
    L_max: float = 1.0e4,
    mode: str = "log",
) -> Tensor:
    """Return L(t) with L(0)=L_max, L(1)=L_obs.

    Args:
        t:       time in [0, 1], any shape.
        L_obs:   observation-end looks (e.g. 1 for single-look SAR).
        L_max:   near-clean-end looks. Large but finite for numerical stability.
        mode:    'log' (log-linear interp on L, smoother variance decay) or 'linear'.
    """
    if mode == "log":
        log_lo = math.log(float(L_obs))
        log_hi = math.log(float(L_max))
        logL = (1.0 - t) * log_hi + t * log_lo
        return torch.exp(logL)
    elif mode == "linear":
        return (1.0 - t) * L_max + t * float(L_obs)
    else:
        raise ValueError(f"unknown schedule mode: {mode}")


def sample_gamma_noise(L: Tensor, shape: tuple[int, ...], device, dtype=torch.float32) -> Tensor:
    """Sample N ~ Gamma(shape=L, rate=L) with the requested tensor shape.

    L may be a scalar tensor, a (B,) tensor, or already broadcast-shaped. Broadcast
    to `shape` before sampling so each spatial location shares the same L for that
    example — which matches the physics (L is per-image at a given time).
    """
    L_b = L
    while L_b.ndim < len(shape):
        L_b = L_b.unsqueeze(-1)
    L_b = L_b.expand(shape).to(device=device, dtype=dtype)
    # torch.distributions.Gamma uses concentration=shape, rate=rate.
    # Detach: N's distribution parameter L(t) is a schedule, not a learnable.
    dist = torch.distributions.Gamma(concentration=L_b, rate=L_b)
    return dist.sample()


def gamma_corrupt(
    x0: Tensor,
    t: Tensor,
    L_obs: float,
    L_max: float = 1.0e4,
    schedule: str = "log",
    return_noise: bool = False,
):
    """Apply exact Gamma multiplicative noise at time t.

    Args:
        x0:    (B, C, H, W) clean/near-clean image, non-negative intensity.
        t:     (B,) times in [0,1].
        L_obs: observation-end looks.
        L_max: near-clean-end looks.
    Returns:
        x_t:   (B, C, H, W) noised image  (differentiable w.r.t. x0).
        L_t:   (B,) looks at time t.
        N_t:   (B, C, H, W) noise (only if return_noise=True).
    """
    if x0.ndim != 4:
        raise ValueError(f"x0 must be (B,C,H,W); got shape {tuple(x0.shape)}")
    if t.ndim != 1 or t.shape[0] != x0.shape[0]:
        raise ValueError(f"t must be (B,) matching batch; got {tuple(t.shape)}")
    L_t = looks_schedule(t, L_obs=L_obs, L_max=L_max, mode=schedule)  # (B,)
    N_t = sample_gamma_noise(L_t, x0.shape, device=x0.device, dtype=x0.dtype)
    x_t = x0 * N_t
    if return_noise:
        return x_t, L_t, N_t
    return x_t, L_t


def embed_looks(L: Tensor, dim: int = 128, L_ref: float = 1.0) -> Tensor:
    """Sinusoidal embedding of log(L / L_ref) — mirrors classic time embedding
    but on the log-looks axis so it stays well-scaled across L in [1, 1e4]."""
    logL = torch.log(L.float() / L_ref)
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=L.device, dtype=logL.dtype) / max(half - 1, 1)
    )
    args = logL[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1))
    return emb
