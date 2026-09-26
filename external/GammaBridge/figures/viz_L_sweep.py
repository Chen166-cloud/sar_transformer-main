"""Smart-start L-input sweep on an L=1-trained model.

For each L_in in {1, 2, 4, 8, 16, 32}:
  1. Synthesise x_obs = x0 * Gamma(L_in, L_in).
  2. Find bridge step whose L(step) is closest to L_in.
  3. Reverse from that step to 0 with `nfe` OT-ODE hops.
  4. Record PSNR, SSIM, ratio_mean, ratio_KS-p.

Outputs:
  * `<out_prefix>_grid.png`  visual grid: rows=samples, cols=[x0, denoised@each L_in]
  * `<out_prefix>_input_grid.png` visual grid of the raw x_obs at each L_in
  * `<out_prefix>_metrics.csv` per-sample metrics
  * stdout: summary table

Run:
    CUDA_VISIBLE_DEVICES=0 python figures/viz_L_sweep.py \
        --ckpt weights/combo_v1_L1_ckpt_0015000.pt \
        --num_samples 64 --num_grid 4 --nfe 5
"""

from __future__ import annotations
import argparse
import csv
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
from eval.metrics import psnr as psnr_np, ssim as ssim_np, ratio_stats


def to_u8(a): return (np.clip(a, 0, 1) * 255).astype(np.uint8)


def reverse_smart_start(bridge, fn, x_start, start_step, nfe):
    idx = np.linspace(0, start_step, num=nfe + 1).round().astype(int)
    step_list = sorted(set(int(i) for i in idx))
    rev = step_list[::-1]
    xt = x_start.detach().clone()
    with torch.no_grad():
        for prev_step, step in zip(rev[1:], rev[:-1]):
            x0_hat = fn(xt, step)
            xt = bridge.p_posterior(prev_step, step, xt, x0_hat, ot_ode=True)
    return xt


def make_speckle(shape, L, device):
    return torch.distributions.Gamma(L, L).sample(shape).to(device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--num_samples", type=int, default=64, help="samples for the metric average")
    p.add_argument("--num_grid", type=int, default=4, help="samples shown in the visual grid")
    p.add_argument("--L_ins", type=float, nargs="+", default=[1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
    p.add_argument("--nfe", type=int, default=5)
    p.add_argument("--out_prefix", default="figures/combo_v1_L1_L_sweep")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    G, bridge, ckpt_args = load_model(Path(args.ckpt), device)
    T = bridge.T
    L_arr = bridge.L.cpu().numpy()

    ds = PairedEvalDataset(
        crop_size=ckpt_args["image_size"], L_obs=ckpt_args["L_obs"],
        num_samples=args.num_samples,
    )
    x0_list = []
    for i in range(len(ds)):
        x0_list.append(ds[i]["x0"])
    x0 = torch.stack(x0_list).to(device)

    log_L_map = torch.log(bridge.L)
    use_cond = bool(ckpt_args.get("cond_x1"))

    per_L_metrics = {}     # L_in -> list of per-sample dicts
    per_L_outputs = {}     # L_in -> denoised tensor (all N)
    per_L_inputs = {}      # L_in -> x_obs tensor (all N)
    per_L_step = {}        # L_in -> smart_step used
    per_L_actual = {}      # L_in -> actual L(smart_step)

    BATCH = 4  # 256×256 UNet w/ attention is heavy; chunk the sweep
    for L_in in args.L_ins:
        smart_step = int(np.argmin(np.abs(L_arr - L_in)))
        per_L_step[L_in] = smart_step
        per_L_actual[L_in] = float(L_arr[smart_step])

        torch.manual_seed(args.seed + int(L_in * 100))
        speck = make_speckle(x0.shape, L_in, device)
        x_obs = x0 * speck

        x_out_chunks = []
        for lo in range(0, x0.shape[0], BATCH):
            hi = min(lo + BATCH, x0.shape[0])
            cond_chunk = x_obs[lo:hi] if use_cond else None
            fn = predict_x0_fn(G, log_L_map, cond=cond_chunk)
            if smart_step == 0:
                x_out_chunks.append(x_obs[lo:hi].clone())
            else:
                x_out_chunks.append(
                    reverse_smart_start(bridge, fn, x_obs[lo:hi], smart_step, args.nfe)
                )
        x_out = torch.cat(x_out_chunks, dim=0)

        per_L_inputs[L_in] = x_obs.detach()
        per_L_outputs[L_in] = x_out.detach()

        rows = []
        x0_np_all = x0.cpu().numpy()
        x_obs_np_all = x_obs.cpu().numpy()
        x_out_np_all = x_out.cpu().numpy()
        for i in range(x0.shape[0]):
            row = {
                "L_in": L_in,
                "smart_step": smart_step,
                "L_actual": per_L_actual[L_in],
                "psnr_noisy": psnr_np(x0_np_all[i, 0], x_obs_np_all[i, 0]),
                "psnr_denoised": psnr_np(x0_np_all[i, 0], x_out_np_all[i, 0]),
                "ssim_noisy": ssim_np(x0_np_all[i, 0], x_obs_np_all[i, 0]),
                "ssim_denoised": ssim_np(x0_np_all[i, 0], x_out_np_all[i, 0]),
            }
            row.update(ratio_stats(x_obs_np_all[i, 0], x_out_np_all[i, 0], L_obs=L_in))
            rows.append(row)
        per_L_metrics[L_in] = rows

    # ---- CSV ----
    csv_path = Path(args.out_prefix + "_metrics.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    all_rows = [r for L in args.L_ins for r in per_L_metrics[L]]
    fieldnames = list(all_rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    # ---- Summary table ----
    def mean(rows, key): return float(np.mean([r[key] for r in rows]))
    print(f"\nckpt={Path(args.ckpt).name}  NFE={args.nfe}  N={args.num_samples}  seed={args.seed}\n")
    header = (f"  {'L_in':>5}  {'step':>5}  {'L_used':>8}  "
              f"{'PSNR_in':>8}  {'PSNR_out':>9}  {'Δ':>6}  "
              f"{'SSIM_in':>8}  {'SSIM_out':>9}  "
              f"{'ratio_m':>8}  {'ratio_v':>8}  {'KS-p':>10}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for L_in in args.L_ins:
        r = per_L_metrics[L_in]
        pi = mean(r, "psnr_noisy")
        po = mean(r, "psnr_denoised")
        print(f"  {L_in:5.1f}  {per_L_step[L_in]:5d}  {per_L_actual[L_in]:8.2f}  "
              f"{pi:8.2f}  {po:9.2f}  {po-pi:+6.2f}  "
              f"{mean(r,'ssim_noisy'):8.3f}  {mean(r,'ssim_denoised'):9.3f}  "
              f"{mean(r,'ratio_mean'):8.4f}  {mean(r,'ratio_var'):8.4f}  "
              f"{mean(r,'ratio_ks_p'):10.3e}")
    print(f"\n[csv] {csv_path}")

    # ---- Visual grids ----
    N = min(args.num_grid, x0.shape[0])
    H = x0.shape[-2]; W = x0.shape[-1]
    label_h = 26; row_h = H + 6

    # Grid A: denoised outputs
    col_labels_a = ["x0 (ref)"] + [f"L_in={int(L)} → denoised" for L in args.L_ins]
    total_w_a = len(col_labels_a) * (W + 6) + 6
    total_h_a = label_h + N * row_h + 6
    canvas_a = np.full((total_h_a, total_w_a, 3), 255, dtype=np.uint8)
    for i in range(N):
        y0 = label_h + i * row_h + 3
        strips = [to_u8(x0[i, 0].cpu().numpy())]
        for L_in in args.L_ins:
            strips.append(to_u8(per_L_outputs[L_in][i, 0].cpu().numpy()))
        for j, s in enumerate(strips):
            xp = 3 + j * (W + 6)
            canvas_a[y0:y0 + H, xp:xp + W] = np.stack([s, s, s], -1)

    # Grid B: raw x_obs at each L_in
    col_labels_b = ["x0 (ref)"] + [f"L_in={int(L)} raw" for L in args.L_ins]
    canvas_b = np.full((total_h_a, total_w_a, 3), 255, dtype=np.uint8)
    for i in range(N):
        y0 = label_h + i * row_h + 3
        strips = [to_u8(x0[i, 0].cpu().numpy())]
        for L_in in args.L_ins:
            strips.append(to_u8(per_L_inputs[L_in][i, 0].cpu().numpy()))
        for j, s in enumerate(strips):
            xp = 3 + j * (W + 6)
            canvas_b[y0:y0 + H, xp:xp + W] = np.stack([s, s, s], -1)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except OSError:
        font = ImageFont.load_default()

    for canvas, cols, tag in [(canvas_a, col_labels_a, "grid"),
                              (canvas_b, col_labels_b, "input_grid")]:
        img = Image.fromarray(canvas)
        draw = ImageDraw.Draw(img)
        for j, name in enumerate(cols):
            draw.text((3 + j * (W + 6) + 4, 4), name, fill=(0, 0, 0), font=font)
        p = Path(args.out_prefix + f"_{tag}.png")
        img.save(p)
        print(f"[png] {p}")


if __name__ == "__main__":
    main()
