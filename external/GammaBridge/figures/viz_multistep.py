"""Side-by-side grid: x0 | x_obs | 1-step | 5-step | 25-step for one ckpt.

Run:
    CUDA_VISIBLE_DEVICES=0 python figures/viz_multistep.py \
        --ckpt weights/combo_v1_L1_ckpt_0015000.pt \
        --num_samples 8 --out figures/combo_v1_L1_multistep.png
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

from sample import load_model, predict_x0_fn, make_step_list
from data.optical_dataset import PairedEvalDataset


def to_u8(a: np.ndarray) -> np.ndarray:
    return (np.clip(a, 0, 1) * 255).astype(np.uint8)


def psnr(x, y):
    m = ((x - y) ** 2).mean()
    return 10.0 * log10(1.0 / max(float(m), 1e-12))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--num_samples", type=int, default=8)
    p.add_argument("--nfes", type=int, nargs="+", default=[1, 5, 25])
    p.add_argument("--out", default="figures/combo_v1_L1_multistep.png")
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

    preds = {}
    for nfe in args.nfes:
        step_list = make_step_list(bridge.T, num_steps=nfe + 1, target_step=0)
        with torch.no_grad():
            xs, _ = bridge.ddpm_sampling(
                steps=step_list, pred_x0_fn=fn, x1=x_obs, ot_ode=True, verbose=False,
            )
        preds[nfe] = xs[:, 0].to(device)

    N = x0.shape[0]
    H = x0.shape[-2]
    W = x0.shape[-1]
    cols = ["x0", "x_obs"] + [f"{n}-step" for n in args.nfes]
    ncols = len(cols)

    label_h = 24
    row_h = H + 6
    total_h = label_h + N * row_h + 6
    total_w = ncols * (W + 6) + 6

    canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8) + 255

    for i in range(N):
        y0 = label_h + i * row_h + 3
        strips = [to_u8(x0[i, 0].cpu().numpy()),
                  to_u8(x_obs[i, 0].cpu().numpy())]
        for nfe in args.nfes:
            strips.append(to_u8(preds[nfe][i, 0].cpu().numpy()))
        for j, s in enumerate(strips):
            x0p = 3 + j * (W + 6)
            canvas[y0:y0 + H, x0p:x0p + W] = np.stack([s, s, s], axis=-1)

    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    for j, name in enumerate(cols):
        x_c = 3 + j * (W + 6) + W // 2
        draw.text((x_c - 30, 4), name, fill=(0, 0, 0), font=font)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    img.save(args.out)

    print(f"[save] {args.out}")
    print(f"L_obs={ckpt_args['L_obs']:.1f}  ckpt={Path(args.ckpt).name}  seed={args.seed}")
    print("per-sample PSNR (dB):")
    header = "  idx | " + "  ".join([f"{c:>7}" for c in cols[2:]])
    print(header)
    for i in range(N):
        x0_np = x0[i, 0].cpu().numpy()
        vals = []
        for nfe in args.nfes:
            vals.append(f"{psnr(x0_np, preds[nfe][i, 0].cpu().numpy()):7.2f}")
        print(f"  {i:3d} | " + "  ".join(vals))
    print("mean:")
    means = []
    for nfe in args.nfes:
        m = np.mean([psnr(x0[i, 0].cpu().numpy(), preds[nfe][i, 0].cpu().numpy())
                     for i in range(N)])
        means.append(f"{m:7.2f}")
    print("      | " + "  ".join(means))


if __name__ == "__main__":
    main()
