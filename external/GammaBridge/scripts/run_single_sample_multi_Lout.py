"""Run γ-Bridge on a single SAR image at multiple target L_out stopping points.

Convention: L is equivalent number of looks (ENL). Higher L = less noise.
  - Input is at L(t*) ≈ L̂ (t* = smart-start step).
  - Denoising means walking backward from t* toward t=0 where L(0) = L_max.
  - A target L_out > L̂ means stopping partway (partial clean); L_out ≥ L_max
    (or the "full" flag) means the standard fully-clean reverse trajectory.
  - L_out ≤ L̂ is a no-op (would move forward, adding noise) — skipped.

Steps:
  1. Estimate L̂ via homogeneous-patch ENL (P90).
  2. For each requested L_out > L̂, partial-denoise from t*(L̂) → t*(L_out).
  3. Also run full clean (walk all the way to t=0) for reference.
  4. Post-hoc rescale each output to match input mean (real-SAR convention).
  5. Save individual PNGs + a labeled comparison strip + a metrics JSON.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from eval.run_external import (
    load_model, to_u8, find_target_step, pad_to_multiple,
    make_pred_x0, denoise_one, _tile_positions, _hann_window_2d,
)
from eval.metrics import (
    enl_whole, epi_cv, epd_diff, sqi_sun, mean_100_convention, enl_homogeneous,
)


TILE_SIZE = 512


def load_np(p: Path) -> np.ndarray:
    return np.asarray(Image.open(p).convert("L"), dtype=np.float32) / 255.0


def denoise_partial(G, bridge, x_np, L_in, L_out, num_steps, device, ot_ode=True):
    def _one_tile(tile):
        tile, (Hc, Wc) = pad_to_multiple(tile, 8)
        t_start = find_target_step(bridge, L_in)
        t_target = find_target_step(bridge, L_out)
        if t_target >= t_start:
            return tile[..., :Hc, :Wc]
        idx = sorted(set(int(i) for i in
                         np.linspace(t_target, t_start,
                                     num=max(num_steps, 2)).round().astype(int)))
        if idx[0] != t_target: idx = [t_target] + idx
        if idx[-1] != t_start: idx.append(t_start)
        steps = sorted(set(idx))
        fn = make_pred_x0(G, bridge, cond=tile)
        xt = tile.detach()
        rev = list(reversed(steps))
        for prev_step, step in zip(rev[1:], rev[:-1]):
            with torch.no_grad():
                x0_hat = fn(xt, step)
                xt = bridge.p_posterior(prev_step, step, xt, x0_hat, ot_ode=ot_ode)
        return xt[..., :Hc, :Wc]

    x = torch.from_numpy(x_np).float().unsqueeze(0).unsqueeze(0).to(device)
    _, _, H, W = x.shape
    if H <= TILE_SIZE and W <= TILE_SIZE:
        return _one_tile(x)[0, 0].cpu().numpy()

    stride = TILE_SIZE // 2
    out = torch.zeros_like(x)
    wsum = torch.zeros_like(x)
    hann = _hann_window_2d(TILE_SIZE, TILE_SIZE, device)
    for i in _tile_positions(H, TILE_SIZE, stride):
        for j in _tile_positions(W, TILE_SIZE, stride):
            tile = x[:, :, i:i + TILE_SIZE, j:j + TILE_SIZE]
            th = _one_tile(tile)
            out[:, :, i:i + TILE_SIZE, j:j + TILE_SIZE] += th * hann
            wsum[:, :, i:i + TILE_SIZE, j:j + TILE_SIZE] += hann
    return (out / wsum.clamp(min=1e-6))[0, 0].cpu().numpy()


def norm_mean(x_hat, x_obs):
    m, o = float(x_obs.mean()), float(x_hat.mean())
    return x_hat * (m / o) if o > 1e-6 else x_hat


def make_strip(images_u8, labels, label_h=20, margin=3):
    H, W = images_u8[0].shape
    n = len(images_u8)
    sw = n * (W + margin) - margin
    strip = Image.new("L", (sw, label_h + H), 230)
    draw = ImageDraw.Draw(strip)
    for k, (arr, lbl) in enumerate(zip(images_u8, labels)):
        x0 = k * (W + margin)
        draw.text((x0 + 2, 2), lbl, fill=0)
        strip.paste(Image.fromarray(arr), (x0, label_h))
    return strip


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, required=True,
                   help="Real SAR image (8-bit grayscale PNG/JPG).")
    p.add_argument("--out", type=Path, default=Path("figures/real_sar"),
                   help="Output directory for the per-L_out PNGs, comparison "
                        "strip and metrics.json.")
    p.add_argument("--ckpt", type=Path,
                   default=Path("weights/gbridge_L1.pt"))
    p.add_argument("--Louts", type=float, nargs="+",
                   default=[8.0, 16.0, 32.0, 64.0, 128.0],
                   help="Target output L (higher = cleaner). Only L_out > L̂ has effect; "
                        "the 'full clean' run is always appended in addition.")
    p.add_argument("--full_clean", action="store_true", default=True,
                   help="Also run the standard full reverse (walk to t=0).")
    p.add_argument("--num_steps_partial", type=int, default=10)
    p.add_argument("--num_steps_full", type=int, default=5)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"[load] ckpt = {args.ckpt}", flush=True)
    G, bridge, _ = load_model(args.ckpt, device)
    L_bounds = (float(bridge.L_obs), float(bridge.L_max))
    print(f"[cfg ] L_obs={L_bounds[0]}  L_max={L_bounds[1]}", flush=True)

    x_obs = load_np(args.src)
    L_hat = float(enl_homogeneous(x_obs))
    L_in = float(np.clip(L_hat, *L_bounds))
    print(f"[img ] {args.src.name}  H×W={x_obs.shape}  mean={x_obs.mean():.3f}", flush=True)
    print(f"[L̂  ] patch-ENL P90 = {L_hat:.2f}   (used as L_in = {L_in:.2f})", flush=True)

    # save clean copy of the noisy input
    Image.fromarray(to_u8(x_obs)).save(args.out / f"{args.src.stem}_noisy.png")

    images_u8, labels, all_metrics = [to_u8(x_obs)], [f"Noisy L̂≈{L_hat:.1f}"], []
    all_metrics.append({"stage": "noisy", "L_est_patchENL": L_hat,
                        "enl": enl_whole(to_u8(x_obs)),
                        "epi": epi_cv(to_u8(x_obs)),
                        "sqi": sqi_sun(to_u8(x_obs)),
                        "mean": mean_100_convention(to_u8(x_obs))})

    stages = [("partial", L) for L in args.Louts]
    if args.full_clean:
        stages.append(("full", None))

    for kind, L_out in stages:
        if kind == "partial":
            L_out_c = float(np.clip(L_out, *L_bounds))
            if L_out_c <= L_in + 0.1:
                print(f"[skip] L_out={L_out_c:.1f} ≤ L̂={L_in:.2f} — target is noisier "
                      f"than input, skipping.", flush=True)
                continue
            x_hat = denoise_partial(G, bridge, x_obs, L_in, L_out_c,
                                    args.num_steps_partial, device, ot_ode=True)
            tag = f"L{int(round(L_out_c)):03d}"
            label = f"L_out={int(round(L_out_c))}"
        else:
            x_hat = denoise_one(G, bridge, x_obs, L_in,
                                args.num_steps_full, device, ot_ode=True)
            tag = "full"
            label = "Full clean"
            L_out_c = float(L_bounds[1])
        x_hat = norm_mean(x_hat, x_obs)
        u8 = to_u8(x_hat)
        Image.fromarray(u8).save(args.out / f"{args.src.stem}_{tag}.png")

        m = {"stage": tag, "L_out_target": L_out_c,
             "enl":   enl_whole(u8),  "epi":   epi_cv(u8),
             "epd_h": epd_diff(u8, "H"), "epd_v": epd_diff(u8, "V"),
             "sqi":   sqi_sun(u8),  "mean":  mean_100_convention(u8)}
        all_metrics.append(m)
        images_u8.append(u8)
        labels.append(label)
        print(f"  {label:>12s}  ENL={m['enl']:6.2f}  EPI={m['epi']:.3f}  "
              f"EPD_H={m['epd_h']:6.2f}  SQI={m['sqi']:.3f}  Mean={m['mean']:.2f}",
              flush=True)

    strip = make_strip(images_u8, labels)
    strip.save(args.out / f"{args.src.stem}_compare.png")
    with open(args.out / "metrics.json", "w") as f:
        json.dump({"src": str(args.src), "L_est_patchENL": L_hat,
                   "results": all_metrics}, f, indent=2)
    print(f"\n[done] outputs -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
