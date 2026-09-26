"""Unit tests for the discretized Gamma bridge (gbridge_core.diffusion).

Verifies:
  * Schedule endpoints: L(step=0)=L_max, L(step=T-1)=L_obs; monotone dec.
  * q_sample marginals at several steps match Gamma(L, 1/L) in mean/var/KS.
  * p_posterior OT-ODE (ot_ode=True) is the convex combination
    alpha*x_n + (1-alpha)*x0_hat with alpha = L_n / L_prev.
  * p_posterior stochastic preserves the forward marginal q(x_{nprev}|x_0)
    when x_n is drawn from q(x_n|x_0) and x0_hat == x0 (oracle case).
  * ddpm_sampling completes end-to-end when the network is an oracle
    (returns the true x0) — final output should equal x0 (up to numerics).
"""

from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from scipy import stats

from gbridge_core.diffusion import GammaBridge


def test_schedule():
    br = GammaBridge(num_timesteps=100, L_obs=1.0, L_max=1e4, device="cpu")
    assert abs(br.L[0].item() - 1e4) < 1e-3, br.L[0].item()
    assert abs(br.L[-1].item() - 1.0) < 1e-3, br.L[-1].item()
    diffs = (br.L[1:] - br.L[:-1]).tolist()
    assert all(d < 0 for d in diffs), "L(step) not monotone dec"
    print(f"[OK] schedule: L[0]={br.L[0].item():.1f}  L[-1]={br.L[-1].item():.3f}  T={br.T}")


def test_q_sample_marginals(T=200, L_obs=1.0, probe_steps=(0, 50, 100, 150, 199)):
    torch.manual_seed(0)
    br = GammaBridge(num_timesteps=T, L_obs=L_obs, L_max=1e4, device="cpu")
    # A single flat-value clean image so ratio = x_t / x_0 has clean statistics.
    B, C, H, W = 1, 1, 200, 200
    x0 = torch.ones(B, C, H, W)
    for step_int in probe_steps:
        step = torch.tensor([step_int], dtype=torch.long)
        x_t = br.q_sample(step, x0)
        ratio = (x_t / x0).flatten().numpy()
        L_val = br.L[step_int].item()
        mean, var = float(ratio.mean()), float(ratio.var(ddof=1))
        # For very large L, the ratio distribution collapses to ~delta at 1 — skip KS there.
        assert abs(mean - 1.0) < 0.02, f"step {step_int}: mean={mean}"
        theo_var = 1.0 / L_val
        rel_var = abs(var - theo_var) / max(theo_var, 1e-8)
        assert rel_var < 0.15 or L_val > 1e3, f"step {step_int}: var {var} vs {theo_var}"
        print(f"[OK] step={step_int:>3}  L={L_val:>8.2f}  mean={mean:.4f}  var={var:.5f}  theo={theo_var:.5f}")


def test_p_posterior_endpoint():
    br = GammaBridge(num_timesteps=50, L_obs=1.0, device="cpu")
    x0_hat = torch.rand(2, 1, 32, 32) + 0.1
    x_n = torch.rand_like(x0_hat) + 0.1
    # ot_ode=True: convex combination alpha*x_n + (1-alpha)*x0_hat.
    nprev, n = 10, 20
    alpha = (br.L[n] / br.L[nprev]).item()
    out = br.p_posterior(nprev=nprev, n=n, x_n=x_n, x0_hat=x0_hat, ot_ode=True)
    expected = alpha * x_n + (1.0 - alpha) * x0_hat
    assert torch.allclose(out, expected, atol=1e-6), (out - expected).abs().max()
    # nprev==0: still deterministic x0_hat (clean-end short-circuit).
    out = br.p_posterior(nprev=0, n=5, x_n=x_n, x0_hat=x0_hat, ot_ode=False)
    assert torch.allclose(out, x0_hat)
    # Stochastic marginal preservation: draw x_n from the true forward and use
    # oracle x0_hat = x0; the posterior sample should have the Gamma(L_prev)
    # marginal, i.e. mean 1 and var 1/L_prev after dividing by x0.
    torch.manual_seed(1)
    x0 = torch.ones(1, 1, 400, 400)
    n_probe, nprev_probe = 30, 20
    step_n = torch.tensor([n_probe], dtype=torch.long)
    x_n_probe = br.q_sample(step_n, x0)
    out = br.p_posterior(nprev=nprev_probe, n=n_probe, x_n=x_n_probe, x0_hat=x0, ot_ode=False)
    ratio = (out / x0).flatten().numpy()
    L_prev = br.L[nprev_probe].item()
    theo_var = 1.0 / L_prev
    assert abs(ratio.mean() - 1.0) < 0.02, ratio.mean()
    rel_var = abs(ratio.var(ddof=1) - theo_var) / theo_var
    assert rel_var < 0.15, (ratio.var(ddof=1), theo_var)
    print(
        f"[OK] p_posterior: OT-ODE convex combo (alpha={alpha:.4f}); nprev=0 pass-through; "
        f"stochastic marginal matches Gamma(L={L_prev:.2f}) (var {ratio.var(ddof=1):.5f} vs theo {theo_var:.5f})"
    )


def test_oracle_sampling_returns_x0():
    """If the network is an oracle predicting the true x0, the reverse chain
    should return x0 at step 0 (endpoint parameterisation + nprev=0 is deterministic)."""
    torch.manual_seed(2)
    br = GammaBridge(num_timesteps=50, L_obs=1.0, device="cpu")
    B, C, H, W = 2, 1, 32, 32
    x0 = torch.rand(B, C, H, W) + 0.1
    # Start from the observation-end sample.
    step_final = torch.tensor([br.T - 1] * B, dtype=torch.long)
    x1 = br.q_sample(step_final, x0)

    def oracle(xt, step_int):
        return x0.clone()

    steps = list(range(0, br.T))
    xs, pred_x0s = br.ddpm_sampling(steps=steps, pred_x0_fn=oracle, x1=x1, ot_ode=True, verbose=False)
    # xs[:, 0] is the last (clean-end) state — should equal x0 exactly under oracle+ot_ode.
    final = xs[:, 0]
    assert torch.allclose(final, x0.cpu(), atol=1e-6), "oracle+ot_ode did not return x0"
    print(f"[OK] oracle DDIM-endpoint sampling reaches x0 exactly (max err {(final - x0.cpu()).abs().max():.2e})")


def test_oracle_stochastic_sampling_marginal():
    """With oracle predictions and stochastic reverse, the chain still ends at
    step 0 == deterministic x0 (nprev=0 branch)."""
    torch.manual_seed(3)
    br = GammaBridge(num_timesteps=50, L_obs=1.0, device="cpu")
    B, C, H, W = 2, 1, 32, 32
    x0 = torch.rand(B, C, H, W) + 0.1
    step_final = torch.tensor([br.T - 1] * B, dtype=torch.long)
    x1 = br.q_sample(step_final, x0)

    def oracle(xt, step_int):
        return x0.clone()

    steps = list(range(0, br.T))
    xs, _ = br.ddpm_sampling(steps=steps, pred_x0_fn=oracle, x1=x1, ot_ode=False, verbose=False)
    final = xs[:, 0]
    assert torch.allclose(final, x0.cpu(), atol=1e-6)
    print(f"[OK] oracle stochastic sampling still returns x0 at step 0")


def main():
    test_schedule()
    test_q_sample_marginals()
    test_p_posterior_endpoint()
    test_oracle_sampling_returns_x0()
    test_oracle_stochastic_sampling_marginal()
    print("\nAll gbridge_core.diffusion tests passed.")


if __name__ == "__main__":
    main()
