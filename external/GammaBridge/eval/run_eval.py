"""End-to-end evaluation of a Gamma-bridge checkpoint on the BSDS500 paired
eval set. Uses multi-step reverse (matching `sample.py`).

Prints a JSON-ish summary + writes CSV of per-sample metrics.
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from gbridge_core.diffusion import GammaBridge
from gbridge_core.network import build_default_unet
from data.optical_dataset import PairedEvalDataset
from eval.metrics import psnr, ssim, ratio_stats


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--num_samples", type=int, default=64)
    p.add_argument("--num_steps", type=int, default=25)
    p.add_argument("--ot_ode", action="store_true")
    p.add_argument("--target_L", type=float, default=None)
    p.add_argument("--out_csv", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda")
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
    bridge = GammaBridge(num_timesteps=a["T"], L_obs=a["L_obs"], L_max=a["L_max"],
                         schedule=a.get("schedule", "log"), device=device)
    return G, bridge, a


def make_pred_x0(G, bridge, cond=None, log_residual_clip=5.0, direct_pred=False):
    """Return a closure that predicts x0 from x_t at a given step."""
    log_L_map = torch.log(bridge.L)
    def fn(xt, step_int: int):
        step_t = torch.tensor([step_int] * xt.shape[0], device=xt.device, dtype=torch.long)
        log_L = log_L_map[step_int].expand(xt.shape[0])
        out = G(xt, step_t.float(), log_L, cond=cond)
        if direct_pred:
            return torch.relu(xt + out)
        log_res = out.clamp(-log_residual_clip, log_residual_clip)
        return xt * torch.exp(log_res)
    return fn


def make_step_list(T: int, num_steps: int, target_step: int = 0):
    idx = np.linspace(target_step, T - 1, num=num_steps).round().astype(int)
    return sorted(set(int(i) for i in idx))


def main():
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    G, bridge, ckpt_args = load_model(Path(args.ckpt), device)

    if args.target_L is None:
        target_step = 0
    else:
        target_step = int(np.argmin(np.abs(bridge.L.cpu().numpy() - args.target_L)))

    step_list = make_step_list(bridge.T, args.num_steps, target_step=target_step)
    ds = PairedEvalDataset(
        crop_size=ckpt_args["image_size"], L_obs=ckpt_args["L_obs"], num_samples=args.num_samples
    )
    use_cond = bool(ckpt_args.get("cond_x1"))

    all_metrics = []
    for i in range(len(ds)):
        s = ds[i]
        x0 = s["x0"].to(device).unsqueeze(0)
        x_obs = s["x_obs"].to(device).unsqueeze(0)
        cond = x_obs if use_cond else None
        fn = make_pred_x0(G, bridge, cond=cond, direct_pred=bool(ckpt_args.get("direct_pred")))
        with torch.no_grad():
            xs, _ = bridge.ddpm_sampling(
                steps=step_list, pred_x0_fn=fn, x1=x_obs, ot_ode=args.ot_ode, verbose=False,
                naive_posterior=bool(ckpt_args.get("naive_posterior")),
            )
        x_final = xs[:, 0].to(device)
        x0_np = x0[0, 0].cpu().numpy()
        x_obs_np = x_obs[0, 0].cpu().numpy()
        x_final_np = x_final[0, 0].cpu().numpy()
        row = {
            "psnr_denoised": psnr(x0_np, x_final_np),
            "psnr_noisy": psnr(x0_np, x_obs_np),
            "ssim_denoised": ssim(x0_np, x_final_np),
            "ssim_noisy": ssim(x0_np, x_obs_np),
        }
        row.update(ratio_stats(x_obs_np, x_final_np, L_obs=ckpt_args["L_obs"]))
        all_metrics.append(row)

    # Aggregate.
    agg = {k: float(np.mean([m[k] for m in all_metrics])) for k in all_metrics[0]}
    print("=== eval summary ===")
    print(json.dumps({"num_samples": len(all_metrics), **agg}, indent=2))

    if args.out_csv:
        import csv
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_metrics[0].keys()))
            w.writeheader()
            for m in all_metrics:
                w.writerow(m)
        print(f"[save] per-sample metrics -> {args.out_csv}")


if __name__ == "__main__":
    main()
