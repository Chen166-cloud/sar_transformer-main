"""Combine `_input_grid.png` and `_grid.png` into a paired-column comparison.

Result layout (per row = one sample):
    x0 (ref) | L=1 raw | L=1 out | L=2 raw | L=2 out | ... | L=32 raw | L=32 out

Uses the same cell geometry as `viz_L_sweep.py` (label_h=26, W=H=256, pad=6).
Reads the two rendered PNGs and slices columns; no model re-run needed.

Run:
    python figures/combine_L_sweep.py \
        --prefix figures/combo_v1_L1_L_sweep \
        --L_ins 1 2 4 8 16 32
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def slice_columns(arr, n_cols, W=256, pad=6, left_pad=3, label_h=26):
    """Return a list of (H, W, 3) column strips excluding the label row."""
    strips = []
    for j in range(n_cols):
        x0 = left_pad + j * (W + pad)
        strips.append(arr[label_h:, x0:x0 + W])
    return strips


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prefix", default="figures/combo_v1_L1_L_sweep")
    p.add_argument("--L_ins", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    p.add_argument("--out", default=None,
                   help="output path; defaults to <prefix>_compare_grid.png")
    args = p.parse_args()

    inp = np.array(Image.open(args.prefix + "_input_grid.png").convert("RGB"))
    out = np.array(Image.open(args.prefix + "_grid.png").convert("RGB"))
    assert inp.shape == out.shape, f"shape mismatch {inp.shape} vs {out.shape}"

    W = 256; H = 256; pad = 6; left_pad = 3; label_h = 26
    n_L = len(args.L_ins)
    n_cols_src = 1 + n_L
    body_h = inp.shape[0] - label_h  # includes bottom pad
    N_body_rows = (body_h - 6) // (H + pad)

    inp_cols = slice_columns(inp, n_cols_src, W, pad, left_pad, label_h)
    out_cols = slice_columns(out, n_cols_src, W, pad, left_pad, label_h)

    # Interleave: [x0 (from output grid), raw_1, out_1, raw_2, out_2, ...]
    x0_col = out_cols[0]  # both grids share x0 in col 0; use output's copy
    new_cols = [x0_col]
    new_labels = ["x_0 (reference)"]
    for k, L in enumerate(args.L_ins, start=1):
        new_cols.append(inp_cols[k])
        new_cols.append(out_cols[k])
        new_labels.append(f"L_in={L}  raw")
        new_labels.append(f"L_in={L}  denoised")

    n_cols = len(new_cols)
    total_w = left_pad + n_cols * (W + pad) + 3
    total_h = label_h + N_body_rows * (H + pad) + 6
    canvas = np.full((total_h, total_w, 3), 255, dtype=np.uint8)

    for j, col in enumerate(new_cols):
        xp = left_pad + j * (W + pad)
        canvas[label_h:label_h + col.shape[0], xp:xp + W] = col

    # Draw a subtle vertical divider between each raw|denoised pair
    for k in range(len(args.L_ins)):
        pair_start = left_pad + (1 + 2 * k) * (W + pad)
        canvas[label_h:label_h + body_h, pair_start - 3] = 220  # left of raw
    # And a heavier divider between the reference and the first pair
    ref_end = left_pad + (W + pad) - 3
    canvas[label_h:label_h + body_h, ref_end:ref_end + 1] = 80

    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    for j, name in enumerate(new_labels):
        draw.text((left_pad + j * (W + pad) + 4, 4),
                  name, fill=(0, 0, 0), font=font)

    out_path = Path(args.out or (args.prefix + "_compare_grid.png"))
    img.save(out_path)
    print(f"[png] {out_path}  ({total_w}x{total_h})")


if __name__ == "__main__":
    main()
