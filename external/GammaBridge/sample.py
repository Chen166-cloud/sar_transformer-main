"""Sampling from a trained GammaBridge model, with optional target-L control.

Two modes:
  * `--target_L=1e4`  (default): full denoise, step 0 clean end.
  * `--target_L=8`:    stop at intermediate step whose L(step)=8, giving a
                       "controlled" partial denoising to 8-look output.

Steps:
  1. Load ckpt.
  2. For each observation from the paired eval set:
     - Multi-step DDIM-endpoint reverse using GammaBridge.ddpm_sampling
     - The `pred_x0_fn` wraps our log-residual parameterisation
  3. Save side-by-side (x0, x_obs, x0_hat) grid + per-sample tensors.
"""

from __future__ import annotations
import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from gbridge_core.diffusion import GammaBridge
from gbridge_core.network import build_default_unet
from data.optical_dataset import PairedEvalDataset


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="figures/samples")
    p.add_argument("--num_samples", type=int, default=8)
    p.add_argument("--num_steps", type=int, default=25, help="DDIM steps used for reverse")
    p.add_argument("--target_L", type=float, default=None,
                   help="if set, stop reverse at the discrete step whose L(step) is nearest target_L; "
                        "otherwise go to step 0 (full clean).")
    p.add_argument("--ot_ode", action="store_true", help="deterministic endpoint reverse")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_model(ckpt_path: Path, device, prefer_ema: bool = True):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ckpt["args"]
    in_ch = 2 if a.get("cond_x1") else 1
    G = build_default_unet(
        image_size=a["image_size"],
        in_channels=in_ch, out_channels=1,
        model_channels=a["model_channels"],
        channel_mult=(1, 2, 4),
    ).to(device)
    state = ckpt.get("G_ema") if (prefer_ema and ckpt.get("G_ema")) else ckpt["G"]
    G.load_state_dict(state, strict=False)
    G.eval()
    bridge = GammaBridge(num_timesteps=a["T"], L_obs=a["L_obs"], L_max=a["L_max"], device=device)
    return G, bridge, a


def predict_x0_fn(G, log_L_map, cond=None, log_residual_clip: float = 5.0):
    """Wrap the log-residual parameterisation from train_unpaired.py.

    `cond` is held constant across reverse steps (cond_x1 observation channel).
    """
    def fn(xt, step_int: int):
        step_t = torch.tensor([step_int] * xt.shape[0], device=xt.device, dtype=torch.long)
        log_L = log_L_map[step_int].expand(xt.shape[0])
        log_res = G(xt, step_t.float(), log_L, cond=cond)
        log_res = log_res.clamp(-log_residual_clip, log_residual_clip)
        return xt * torch.exp(log_res)
    return fn


def make_step_list(T: int, num_steps: int, target_step: int = 0):
    """Pick `num_steps` step indices between `target_step` and T-1 inclusive."""
    idx = np.linspace(target_step, T - 1, num=num_steps).round().astype(int)
    idx = sorted(set(int(i) for i in idx))
    if idx[0] != target_step:
        idx = [target_step] + idx
    if idx[-1] != T - 1:
        idx.append(T - 1)
    return sorted(set(idx))


def to_uint8_grid(imgs: list[np.ndarray]) -> np.ndarray:
    """Concatenate 2D arrays horizontally, scaling each to [0, 255]."""
    outs = []
    for a in imgs:
        v = a.copy()
        v = np.clip(v, 0.0, 1.0)
        outs.append((v * 255.0).astype(np.uint8))
    return np.concatenate(outs, axis=1)


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    G, bridge, ckpt_args = load_model(Path(args.ckpt), device)
    print(f"[load] step={ckpt_args.get('image_size')}  T={bridge.T}  L_obs={bridge.L_obs}", flush=True)

    # Target step: closest step index whose L(step) matches target_L.
    if args.target_L is None:
        target_step = 0
    else:
        diffs = (bridge.L.cpu().numpy() - args.target_L)
        target_step = int(np.argmin(np.abs(diffs)))
    L_at_target = bridge.L[target_step].item()
    print(f"[control] target_L={args.target_L}  ->  target_step={target_step}  L(step)={L_at_target:.2f}", flush=True)

    step_list = make_step_list(bridge.T, args.num_steps, target_step=target_step)
    print(f"[sampler] {len(step_list)} DDIM steps: {step_list[0]}..{step_list[-1]}", flush=True)

    ds = PairedEvalDataset(
        crop_size=ckpt_args["image_size"], L_obs=ckpt_args["L_obs"], num_samples=args.num_samples
    )
    x0s, x_obss = [], []
    for i in range(len(ds)):
        s = ds[i]
        x0s.append(s["x0"])
        x_obss.append(s["x_obs"])
    x0 = torch.stack(x0s).to(device)
    x_obs = torch.stack(x_obss).to(device)
    log_L_map = torch.log(bridge.L)

    use_cond = bool(ckpt_args.get("cond_x1"))
    fn = predict_x0_fn(G, log_L_map, cond=(x_obs if use_cond else None))
    with torch.no_grad():
        xs, pred_x0s = bridge.ddpm_sampling(
            steps=step_list, pred_x0_fn=fn, x1=x_obs, ot_ode=args.ot_ode, verbose=True,
        )
    # xs shape: (B, K, C, H, W).  We want the last state (target_step, K=0).
    x_final = xs[:, 0].to(device)  # already at target_step end
    print(f"[sample] x_final shape {tuple(x_final.shape)}", flush=True)

    # Save grid + metrics for each sample.
    from math import log10
    mse_full = ((x0 - x_final).clamp_min(0) ** 2).mean().item()
    psnr_full = 10.0 * log10(1.0 / max(mse_full, 1e-12))
    ratio = (x_obs / x_final.clamp_min(1e-6)).flatten().cpu().numpy()
    print(f"[metrics] PSNR(x0, x_final)={psnr_full:.3f} dB   ratio mean={ratio.mean():.4f}   var={ratio.var(ddof=1):.4f}", flush=True)

    n = min(x_final.shape[0], 8)
    strips = []
    for i in range(n):
        a = x0[i, 0].detach().cpu().numpy()
        b = x_obs[i, 0].detach().cpu().numpy()
        c = x_final[i, 0].detach().cpu().numpy()
        strips.append(to_uint8_grid([a, b, c]))
    grid = np.concatenate(strips, axis=0)
    grid_path = out / f"grid_targetL{args.target_L or 'clean'}_{args.num_steps}steps.png"
    Image.fromarray(grid).save(grid_path)
    print(f"[save] {grid_path}", flush=True)


if __name__ == "__main__":
    main()
