"""L-controllable intermediate output visualization.

For each sample, run reverse stopping at several target L values.
Columns: x0 | x_obs (L=1) | target_L=2 | 4 | 16 | 64 | 256 | 10000 (clean)

The bridge reverse goes from step T-1 (L=L_obs) back to a chosen target step
where L(step) is nearest the target_L. Uses 5-NFE OT-ODE (sweet spot).

Run:
    CUDA_VISIBLE_DEVICES=0 python figures/viz_target_L.py \
        --ckpt weights/combo_v1_L1_ckpt_0015000.pt \
        --num_samples 6 --out figures/combo_v1_L1_target_L.png
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from sample import load_model, predict_x0_fn, make_step_list
from data.optical_dataset import PairedEvalDataset


def to_u8(a: np.ndarray) -> np.ndarray:
    return (np.clip(a, 0, 1) * 255).astype(np.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--num_samples", type=int, default=6)
    p.add_argument("--target_Ls", type=float, nargs="+",
                   default=[2.0, 4.0, 16.0, 64.0, 256.0, 10000.0])
    p.add_argument("--nfe", type=int, default=5)
    p.add_argument("--out", default="figures/combo_v1_L1_target_L.png")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    G, bridge, ckpt_args = load_model(Path(args.ckpt), device)
    ds = PairedEvalDataset(
        crop_size=ckpt_args["image_size"], L_obs=ckpt_args["L_obs"],
        num_samples=args.num_samples,
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

    L_arr = bridge.L.cpu().numpy()
    outs = {}
    actual_L = {}
    for tL in args.target_Ls:
        target_step = int(np.argmin(np.abs(L_arr - tL)))
        actual_L[tL] = float(L_arr[target_step])
        # NFE+1 evenly spaced steps from target_step..T-1 inclusive
        idx = np.linspace(target_step, bridge.T - 1, num=args.nfe + 1).round().astype(int)
        step_list = sorted(set(int(i) for i in idx))
        rev = step_list[::-1]
        xt = x_obs.detach().clone()
        with torch.no_grad():
            for prev_step, step in zip(rev[1:], rev[:-1]):
                x0_hat = fn(xt, step)
                xt = bridge.p_posterior(prev_step, step, xt, x0_hat, ot_ode=True)
        outs[tL] = xt

    N = x0.shape[0]
    H = x0.shape[-2]
    W = x0.shape[-1]
    col_labels = ["x0 (ref)", "x_obs L=1"] + [f"L={int(round(actual_L[t]))}" for t in args.target_Ls]
    ncols = len(col_labels)

    label_h = 26
    row_h = H + 6
    total_h = label_h + N * row_h + 6
    total_w = ncols * (W + 6) + 6

    canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8) + 255

    for i in range(N):
        y0 = label_h + i * row_h + 3
        strips = [to_u8(x0[i, 0].cpu().numpy()),
                  to_u8(x_obs[i, 0].cpu().numpy())]
        for tL in args.target_Ls:
            strips.append(to_u8(outs[tL][i, 0].cpu().numpy()))
        for j, s in enumerate(strips):
            xp = 3 + j * (W + 6)
            canvas[y0:y0 + H, xp:xp + W] = np.stack([s, s, s], axis=-1)

    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
    except OSError:
        font = ImageFont.load_default()
    for j, name in enumerate(col_labels):
        xp = 3 + j * (W + 6)
        draw.text((xp + 4, 4), name, fill=(0, 0, 0), font=font)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    img.save(args.out)

    print(f"[save] {args.out}")
    print(f"ckpt={Path(args.ckpt).name}  NFE={args.nfe}  OT-ODE  seed={args.seed}")
    print("target_L (asked -> actual on schedule):")
    for tL in args.target_Ls:
        print(f"  {tL:>8.1f}  ->  {actual_L[tL]:8.2f}")
    print("ratio-image stats per column (mean, var vs. target Gamma(L,L)):")
    print(f"  col                 mean     var    var_theo(1/L)")
    x_obs_np = x_obs.cpu().numpy()
    for tL in args.target_Ls:
        out_np = outs[tL].cpu().numpy().clip(1e-6)
        r = (x_obs_np / out_np).flatten()
        L_eff = actual_L[tL]
        print(f"  L={int(round(L_eff)):>6}       {r.mean():8.4f}  {r.var(ddof=1):8.4f}  {1.0/L_eff:8.4f}")


if __name__ == "__main__":
    main()
