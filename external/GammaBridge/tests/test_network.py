"""Smoke tests for GammaBridgeUNet on GPU."""

from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from gbridge_core.network import build_default_unet, count_params
from gbridge_core.diffusion import GammaBridge


def test_unet_shape_and_grad():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = build_default_unet(image_size=128, model_channels=64, channel_mult=(1, 2, 4)).to(device)
    print(f"[UNet] params = {count_params(net) / 1e6:.2f} M")

    x = torch.rand(4, 1, 128, 128, device=device) + 0.1
    ts = torch.rand(4, device=device) * 1000.0  # arbitrary scale
    log_L = torch.log(torch.tensor([1.0, 4.0, 16.0, 100.0], device=device))
    y = net(x, ts, log_L)
    assert y.shape == x.shape, y.shape

    loss = y.mean()
    loss.backward()
    n_grad = sum(1 for p in net.parameters() if p.grad is not None and torch.isfinite(p.grad).all())
    n_total = sum(1 for _ in net.parameters())
    assert n_grad == n_total, f"only {n_grad}/{n_total} grads finite"
    print(f"[OK] UNet fwd/bwd: out {tuple(y.shape)}, grads finite for all {n_total} params")


def test_unet_with_bridge_pipeline():
    """End-to-end: sample x_t from GammaBridge, feed to UNet with matching log_L."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    br = GammaBridge(num_timesteps=100, L_obs=1.0, device=device)
    net = build_default_unet(image_size=128, model_channels=64, channel_mult=(1, 2, 4)).to(device)

    x0 = torch.rand(4, 1, 128, 128, device=device) + 0.1
    step = torch.randint(0, br.T, (4,), device=device)
    x_t, L_t = br.q_sample_with_L(step, x0)
    log_L = torch.log(L_t)
    # Timestep index in [0, T-1]; pass as float for sinusoidal embedding.
    x0_hat = net(x_t, step.float(), log_L)
    l1 = (x0_hat - x0).abs().mean()
    l1.backward()
    print(f"[OK] bridge + UNet pipeline: L1={l1.item():.4f}")


def test_unet_variable_L_response():
    """Sanity: internal emb tensor should differ across log_L values.

    We can't directly check the network output because guided_diffusion's
    final conv is initialised via `zero_module` — so at init the output is
    identically zero regardless of input. Instead we verify the conditioning
    path itself: build the (time + log_L) embedding and confirm it varies
    with log_L. After training the zero_module conv will lift off from 0
    and outputs will respond to L; that's part of the training test.
    """
    from gbridge_core.network import logL_embedding
    from gbridge_core.gd.nn import timestep_embedding
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = build_default_unet(image_size=64, model_channels=64, channel_mult=(1, 2)).to(device)
    ts = torch.zeros(2, device=device)
    low = torch.log(torch.tensor([1.0, 1.0], device=device))
    hi = torch.log(torch.tensor([1000.0, 1000.0], device=device))
    with torch.no_grad():
        t_emb = net.time_embed(timestep_embedding(ts, net.model_channels))
        L_low = net.logL_embed(logL_embedding(low, net.model_channels))
        L_hi = net.logL_embed(logL_embedding(hi, net.model_channels))
        emb_low = t_emb + L_low
        emb_hi = t_emb + L_hi
    diff = (emb_low - emb_hi).abs().mean().item()
    assert diff > 1e-4, f"log_L had no effect on emb: diff={diff}"
    print(f"[OK] log_L conditioning drives embedding difference: mean|Δemb|={diff:.4f}")


def main():
    test_unet_shape_and_grad()
    test_unet_with_bridge_pipeline()
    test_unet_variable_L_response()
    print("\nAll network smoke tests passed.")


if __name__ == "__main__":
    main()
