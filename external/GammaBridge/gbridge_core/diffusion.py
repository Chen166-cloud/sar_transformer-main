"""GammaBridge diffusion — API mirrors NVlabs/I2SB's `i2sb.diffusion.Diffusion`
so downstream training/sampling code can be swapped in with minimal changes.

Structure derived from (and named after) NVIDIA I2SB (NVIDIA Source Code License
for I2SB, non-commercial research). Original Gaussian bridge math is replaced
here with the exact L-look Gamma speckle marginal:

    x_t | x_0  =  x_0 * N_t,   N_t ~ Gamma(shape=L(step), rate=L(step))
    L(step):  step=0  -> L_max   (near-clean end)
              step=T-1 -> L_obs  (observed end)

Reverse uses the **closed-form Gamma-Lévy posterior**. Represent the forward as
a Gamma Lévy process X(L) ~ Gamma(L, 1) with x_t = x_0 * X(L_t)/L_t. Since
X(L_prev) = X(L_n) + Y with Y ~ Gamma(L_prev - L_n, 1) independent of X(L_n),
conditioning on x_n and an estimate x0_hat gives

    x_{prev} = alpha * x_n + (x0_hat / L_prev) * Y,   alpha = L_n / L_prev

whose deterministic limit (Y := E[Y] = L_prev - L_n) is the OT-ODE update
alpha * x_n + (1 - alpha) * x0_hat.
"""

from __future__ import annotations
import math
from functools import partial
from typing import Callable

import numpy as np
import torch
from tqdm import tqdm

from .util import unsqueeze_xdim

import sys, os
# So we can import the standalone forward kernel we already validated.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from models.gamma_bridge import looks_schedule, sample_gamma_noise  # noqa: E402


class GammaBridge:
    """Discretized L-look Gamma diffusion bridge.

    Args:
        num_timesteps: number of discrete steps T.
        L_obs:         observation-end looks (e.g. 1.0 single-look SAR).
        L_max:         near-clean-end looks (large; 1e4 is plenty).
        schedule:      'log' or 'linear' interpolation for L(step).
        device:        torch device.
    """

    def __init__(
        self,
        num_timesteps: int,
        L_obs: float,
        L_max: float = 1.0e4,
        schedule: str = "log",
        device: str | torch.device = "cpu",
    ):
        self.T = int(num_timesteps)
        self.L_obs = float(L_obs)
        self.L_max = float(L_max)
        self.schedule = schedule
        self.device = torch.device(device)

        # step goes 0..T-1; t in [0, 1] with t=0 <-> clean end, t=1 <-> obs end.
        steps = torch.arange(self.T, dtype=torch.float32)
        t = steps / max(self.T - 1, 1)
        L = looks_schedule(t, L_obs=self.L_obs, L_max=self.L_max, mode=schedule)

        self.t = t.to(self.device)                    # (T,)
        self.L = L.to(self.device)                    # (T,)
        self.log_L = torch.log(self.L)                # (T,) — used for network conditioning

    # --- convenience -----------------------------------------------------

    def L_at(self, step) -> torch.Tensor:
        """Vector L(step). `step` can be int, tensor of ints, or 0-d tensor."""
        if not torch.is_tensor(step):
            step = torch.as_tensor(step, device=self.device, dtype=torch.long)
        step = step.to(self.device).long()
        return self.L[step]

    def t_at(self, step) -> torch.Tensor:
        if not torch.is_tensor(step):
            step = torch.as_tensor(step, device=self.device, dtype=torch.long)
        step = step.to(self.device).long()
        return self.t[step]

    # --- forward (bridge marginal) --------------------------------------

    def q_sample(self, step: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        """Sample x_t ~ q(x_t | x_0) = x_0 * Gamma(L(step), L(step)).

        Args:
            step: (B,) long tensor of discrete step indices in [0, T-1].
            x0:   (B, C, H, W) clean/near-clean image (non-negative intensity).
        Returns:
            x_t:  (B, C, H, W) noised image  (differentiable w.r.t. x0).
        """
        assert step.ndim == 1 and step.shape[0] == x0.shape[0], (step.shape, x0.shape)
        L_step = self.L_at(step)                                  # (B,)
        N = sample_gamma_noise(L_step, shape=x0.shape, device=x0.device, dtype=x0.dtype)
        return x0 * N

    def q_sample_with_L(self, step: torch.Tensor, x0: torch.Tensor):
        """Same as q_sample but also returns L(step) for network conditioning."""
        L_step = self.L_at(step)
        N = sample_gamma_noise(L_step, shape=x0.shape, device=x0.device, dtype=x0.dtype)
        return x0 * N, L_step

    def q_sample_bridge(
        self, step: torch.Tensor, x0: torch.Tensor, x_obs: torch.Tensor
    ) -> torch.Tensor:
        """Sample x_t from the Gamma-Lévy BRIDGE q(x_t | x_0, x_obs).

        Derived from the Lévy decomposition X(L_t) = X(L_obs) + Y_t,
        Y_t ~ Gamma(L_t - L_obs, 1) independent of X(L_obs). Substituting
        X(L_obs) = L_obs * x_obs / x_0 gives

            x_t = (L_obs / L_t) * x_obs + (x_0 / L_t) * Y_t.

        Boundary conditions match by construction:
            L_t = L_obs (step = T-1):   x_t = x_obs
            L_t = L_max (step = 0):     x_t -> x_0 (Y_t/L_max concentrates on 1)

        And critically the (x_0-only) MARGINAL is preserved:
            x_t / x_0 = X(L_t) / L_t ~ Gamma(L_t, L_t)  as before,
        so any moment/marginal loss written against Gamma(L_t) still applies.

        This bridge forward matches the reverse-trajectory distribution used in
        multi-step inference — the point of aligning training with sampling.
        """
        assert step.ndim == 1 and step.shape[0] == x0.shape[0]
        assert x_obs.shape == x0.shape
        L_step = self.L_at(step).to(x0.dtype)                     # (B,)
        L_obs_t = torch.as_tensor(self.L_obs, device=x0.device, dtype=x0.dtype)
        xdim = x0.shape[1:]
        L_step_b = unsqueeze_xdim(L_step, xdim)
        dL = (L_step - L_obs_t).clamp_min(1e-6)                   # (B,)
        dL_b = unsqueeze_xdim(dL, xdim).expand_as(x0)
        Y = torch.distributions.Gamma(concentration=dL_b, rate=torch.ones_like(dL_b)).sample()
        return (L_obs_t / L_step_b) * x_obs + (x0 / L_step_b) * Y

    # --- reverse (endpoint-parameterisation) ----------------------------

    def p_posterior(
        self,
        nprev: int,
        n: int,
        x_n: torch.Tensor,
        x0_hat: torch.Tensor,
        ot_ode: bool = False,
        naive: bool = False,
    ) -> torch.Tensor:
        """One reverse step from step `n` (noisier, lower L) to step `nprev`
        (cleaner, higher L) given predicted x0_hat.

        Closed-form Gamma-Lévy posterior. Let X(L) be a Gamma Lévy process,
        x_t = x_0 * X(L_t)/L_t. Then X(L_prev) = X(L_n) + Y with
        Y ~ Gamma(L_prev - L_n, 1) independent, so with alpha = L_n / L_prev

            x_{nprev} = alpha * x_n + (x0_hat / L_prev) * Y.

        * ot_ode=True:   Y := E[Y] = L_prev - L_n
                         -> x_{nprev} = alpha * x_n + (1 - alpha) * x0_hat.
        * ot_ode=False:  Y ~ Gamma(L_prev - L_n, 1) sampled elementwise.

        Under an oracle x0_hat == x0, the identity X(L_n) + Y == X(L_prev)
        preserves the exact marginal q(x_{nprev} | x_0) at each step, so any
        number of reverse steps is consistent (no accumulation bias).

        For nprev == 0 the L_prev == L_max branch is deterministic (variance
        ~ 1/L_max ≈ 0) and we short-circuit to x0_hat.
        """
        assert nprev < n, f"{nprev=}, {n=}"
        if nprev == 0 or naive:
            return x0_hat
        B = x0_hat.shape[0]
        device, dtype = x0_hat.device, x0_hat.dtype
        step_n = torch.full((B,), n, device=device, dtype=torch.long)
        step_prev = torch.full((B,), nprev, device=device, dtype=torch.long)
        L_n = self.L_at(step_n).to(dtype)
        L_prev = self.L_at(step_prev).to(dtype)
        xdim = x_n.shape[1:]
        alpha = unsqueeze_xdim(L_n / L_prev, xdim)
        if ot_ode:
            return alpha * x_n + (1.0 - alpha) * x0_hat
        L_prev_b = unsqueeze_xdim(L_prev, xdim)
        dL = (L_prev - L_n).clamp_min(1e-6)
        dL_b = unsqueeze_xdim(dL, xdim).expand_as(x_n)
        Y = torch.distributions.Gamma(concentration=dL_b, rate=torch.ones_like(dL_b)).sample()
        return alpha * x_n + (x0_hat / L_prev_b) * Y

    # --- reverse sampling loop -----------------------------------------

    def ddpm_sampling(
        self,
        steps: list[int],
        pred_x0_fn: Callable[[torch.Tensor, int], torch.Tensor],
        x1: torch.Tensor,
        ot_ode: bool = False,
        log_steps: list[int] | None = None,
        verbose: bool = True,
        naive_posterior: bool = False,
    ):
        """Sample from step T-1 (observation) back to step steps[0].

        Args:
            steps: strictly increasing list of step indices ending at T-1
                   (== observation end). For a full clean denoise steps[0]=0;
                   for target-L output control steps[0] is the stopping step
                   whose L(step) matches the desired output look number.
            pred_x0_fn: callable(x_t, step_int) -> x0_hat.
            x1:    observation, i.e. x at step steps[-1].
            ot_ode: if True use deterministic (endpoint) reverse; else stochastic.
            log_steps: subset of steps at which to record intermediates.
        Returns:
            (xs, pred_x0s): stacked bwd trajectories, shape (B, K, C, H, W),
            with xs[:, 0] at step steps[0].
        """
        assert len(steps) >= 2, "need at least start + end step"
        assert steps[0] >= 0, f"steps[0] must be non-negative; got {steps[0]}"

        xt = x1.detach().to(self.device)
        xs, pred_x0s = [], []

        log_steps = log_steps or steps
        assert log_steps[0] == steps[0], \
            f"log_steps[0]={log_steps[0]} must match steps[0]={steps[0]}"

        rev = steps[::-1]  # walk from noisy end back
        pairs = list(zip(rev[1:], rev[:-1]))  # (prev, n)
        it = tqdm(pairs, desc="GammaBridge sampling", total=len(pairs)) if verbose else pairs
        for prev_step, step in it:
            assert prev_step < step
            x0_hat = pred_x0_fn(xt, step)
            xt = self.p_posterior(prev_step, step, xt, x0_hat, ot_ode=ot_ode,
                                  naive=naive_posterior)
            if prev_step in log_steps:
                pred_x0s.append(x0_hat.detach().cpu())
                xs.append(xt.detach().cpu())

        def stack_bwd(z):
            return torch.flip(torch.stack(z, dim=1), dims=(1,))

        return stack_bwd(xs), stack_bwd(pred_x0s)
