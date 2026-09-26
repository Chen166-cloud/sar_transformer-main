"""Feed an L=4 image to a model trained only on L=1.

Compare three strategies:
  (a) naive: treat L=4 as x_obs at step T-1 (network told log_L=log(1))
  (b) smart-start: treat L=4 as x_t at step nearest L=4 (~step 85)
      -- start reverse from there
  (c) reference: real L=1 x_obs on same clean image → normal full reverse

Rows = samples. Columns = x0, L=1 obs, L=4 obs, (a) naive, (b) smart-start,
       (c) L=1 baseline.
"""

from __future__ import annotations
import argparse
import sys
from math import log10
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from sample import load_model, predict_x0_fn
from data.optical_dataset import PairedEvalDataset


def to_u8(a): return (np.clip(a, 0, 1) * 255).astype(np.uint8)


def psnr(x, y):
    m = float(((x - y) ** 2).mean())
    return 10.0 * log10(1.0 / max(m, 1e-12))


def make_gamma_noise(shape, L, device):
    """x_obs = x0 * Gamma(L, L). Gamma(L,1)/L has mean 1, var 1/L."""
    return torch.distributions.Gamma(L, L).sample(shape).to(device)


def reverse(bridge, fn, x_start, start_step, nfe, cond=None):
    """Reverse from start_step down to step 0 using nfe hops."""
    idx = np.linspace(0, start_step, num=nfe + 1).round().astype(int)
    step_list = sorted(set(int(i) for i in idx))
    rev = step_list[::-1]
    xt = x_start.detach().clone()
    with torch.no_grad():
        for prev_step, step in zip(rev[1:], rev[:-1]):
            x0_hat = fn(xt, step)
            xt = bridge.p_posterior(prev_step, step, xt, x0_hat, ot_ode=True)
    return xt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--num_samples", type=int, default=6)
    p.add_argument("--L_in", type=float, default=4.0, help="look number of the alt input")
    p.add_argument("--nfe", type=int, default=5)
    p.add_argument("--out", default="figures/combo_v1_L1_L4input.png")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    G, bridge, ckpt_args = load_model(Path(args.ckpt), device)
    T = bridge.T
    L_arr = bridge.L.cpu().numpy()

    # Build L=1 dataset for consistent x0
    ds = PairedEvalDataset(crop_size=ckpt_args["image_size"], L_obs=ckpt_args["L_obs"],
                          num_samples=args.num_samples)
    x0s, x1s = [], []
    for i in range(len(ds)):
        s = ds[i]; x0s.append(s["x0"]); x1s.append(s["x_obs"])
    x0 = torch.stack(x0s).to(device)
    x_obs_L1 = torch.stack(x1s).to(device)  # ground-truth L=1 observation
    # Synthesize L=L_in observation on same x0 with fresh seed for the noise
    torch.manual_seed(args.seed + 1)
    speck = make_gamma_noise(x0.shape, args.L_in, device)
    x_obs_L4 = x0 * speck

    log_L_map = torch.log(bridge.L)
    use_cond = bool(ckpt_args.get("cond_x1"))

    # (a) naive: pretend L=4 is x_obs at step T-1, cond=L=4
    fn_naive = predict_x0_fn(G, log_L_map, cond=(x_obs_L4 if use_cond else None))
    x_naive = reverse(bridge, fn_naive, x_start=x_obs_L4, start_step=T - 1, nfe=args.nfe)

    # (b) smart-start: treat L=4 as intermediate step where L(step)~=L_in.
    smart_step = int(np.argmin(np.abs(L_arr - args.L_in)))
    L_smart = float(L_arr[smart_step])
    # cond: still needs a value; we don't have a real L=1 obs, so use the L=4 itself
    fn_smart = predict_x0_fn(G, log_L_map, cond=(x_obs_L4 if use_cond else None))
    x_smart = reverse(bridge, fn_smart, x_start=x_obs_L4, start_step=smart_step, nfe=args.nfe)

    # (c) reference: real L=1 obs, full reverse (the model's design case)
    fn_ref = predict_x0_fn(G, log_L_map, cond=(x_obs_L1 if use_cond else None))
    x_ref = reverse(bridge, fn_ref, x_start=x_obs_L1, start_step=T - 1, nfe=args.nfe)

    outs = {
        "L=1 obs": x_obs_L1,
        f"L={int(args.L_in)} obs": x_obs_L4,
        "(a) naive": x_naive,
        f"(b) smart-start step={smart_step} (L={L_smart:.1f})": x_smart,
        "(c) L=1 baseline": x_ref,
    }

    # Build grid
    N = x0.shape[0]; H = x0.shape[-2]; W = x0.shape[-1]
    col_labels = ["x0"] + list(outs.keys())
    ncols = len(col_labels)
    label_h = 28; row_h = H + 6
    total_h = label_h + N * row_h + 6
    total_w = ncols * (W + 6) + 6
    canvas = np.full((total_h, total_w, 3), 255, dtype=np.uint8)
    for i in range(N):
        y0 = label_h + i * row_h + 3
        strips = [to_u8(x0[i, 0].cpu().numpy())]
        for name in outs:
            strips.append(to_u8(outs[name][i, 0].cpu().numpy()))
        for j, s in enumerate(strips):
            xp = 3 + j * (W + 6)
            canvas[y0:y0 + H, xp:xp + W] = np.stack([s, s, s], -1)
    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    for j, name in enumerate(col_labels):
        draw.text((3 + j * (W + 6) + 4, 4), name, fill=(0, 0, 0), font=font)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    img.save(args.out)
    print(f"[save] {args.out}")

    # PSNR table
    x0_np = x0.cpu().numpy()
    print(f"\nL_in={args.L_in}  NFE={args.nfe}  seed={args.seed}")
    print("per-sample PSNR (vs x0):")
    header = "  idx | " + "  ".join(f"{k:>28}" for k in outs.keys())
    print(header)
    for i in range(N):
        row = [f"{psnr(x0_np[i,0], outs[k][i,0].cpu().numpy()):28.2f}" for k in outs]
        print(f"  {i:3d} | " + "  ".join(row))
    means = []
    for k in outs:
        m = np.mean([psnr(x0_np[i,0], outs[k][i,0].cpu().numpy()) for i in range(N)])
        means.append(f"{m:28.2f}")
    print("mean:")
    print("      | " + "  ".join(means))


if __name__ == "__main__":
    main()
