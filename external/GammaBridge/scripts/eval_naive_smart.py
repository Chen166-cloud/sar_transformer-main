"""Naive vs smart-start sweep on the main ckpt.

For each L_in ∈ {1,2,4,8,16,32}:
  * Synthesise x_obs = x0 * Gamma(L_in, L_in) on the same 64 BSDS500 crops.
  * naive-start: reverse from step T-1 with NFE=5.
  * smart-start: reverse from step t* with L(t*)≈L_in, NFE=5.
Both use the main ckpt (L_obs=1 trained). Writes eval/results/naive_smart_start.csv.
"""

from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path
import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from sample import load_model, predict_x0_fn
from data.optical_dataset import PairedEvalDataset
from eval.metrics import psnr as psnr_np, ssim as ssim_np, ratio_stats


def reverse(bridge, fn, x_start, start_step, nfe):
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
    p.add_argument("--ckpt", default="weights/combo_v1_L1_ckpt_0015000.pt")
    p.add_argument("--num_samples", type=int, default=64)
    p.add_argument("--L_ins", type=float, nargs="+",
                   default=[1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
    p.add_argument("--nfe", type=int, default=5)
    p.add_argument("--out_csv", default="eval/results/naive_smart_start.csv")
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
    x0 = torch.stack([ds[i]["x0"] for i in range(len(ds))]).to(device)

    log_L_map = torch.log(bridge.L)
    use_cond = bool(ckpt_args.get("cond_x1"))

    rows = []
    BATCH = 4
    for L_in in args.L_ins:
        smart_step = int(np.argmin(np.abs(L_arr - L_in)))
        actual_L = float(L_arr[smart_step])
        L_t = torch.tensor(L_in, device=device, dtype=x0.dtype)
        noise_full = torch.distributions.Gamma(L_t, L_t).sample(x0.shape).to(device)
        x_obs = x0 * noise_full   # L_in-look observation

        # both start modes
        for mode, start_step in (("naive", T - 1), ("smart", smart_step)):
            outs = []
            for lo in range(0, x0.shape[0], BATCH):
                hi = min(lo + BATCH, x0.shape[0])
                cond = x_obs[lo:hi] if use_cond else None
                fn = predict_x0_fn(G, log_L_map, cond=cond)
                out = reverse(bridge, fn, x_obs[lo:hi], start_step, args.nfe)
                outs.append(out)
            out = torch.cat(outs, 0)

            x0_np = x0.cpu().numpy()
            xobs_np = x_obs.cpu().numpy()
            out_np = out.cpu().numpy()
            for i in range(x0.shape[0]):
                stats = ratio_stats(xobs_np[i, 0], out_np[i, 0], L_obs=L_in)
                rows.append({
                    "L_in": L_in,
                    "start_step": start_step,
                    "mode": mode,
                    "psnr_denoised": psnr_np(x0_np[i, 0], out_np[i, 0]),
                    "psnr_noisy": psnr_np(x0_np[i, 0], xobs_np[i, 0]),
                    "ssim_denoised": ssim_np(x0_np[i, 0], out_np[i, 0]),
                    "ssim_noisy": ssim_np(x0_np[i, 0], xobs_np[i, 0]),
                    **{k: float(v) for k, v in stats.items()},
                })
        print(f"[done] L_in={L_in}  smart_step={smart_step}  actual_L={actual_L:.3f}")

    # write CSV
    out_path = _ROOT / args.out_csv
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[save] {out_path}")

    # summary
    import statistics
    for L_in in args.L_ins:
        for mode in ("naive", "smart"):
            subset = [r for r in rows if r["L_in"] == L_in and r["mode"] == mode]
            m_psnr = statistics.mean(r["psnr_denoised"] for r in subset)
            m_ssim = statistics.mean(r["ssim_denoised"] for r in subset)
            print(f"  L_in={L_in:5.1f} {mode:6}  PSNR={m_psnr:6.2f}  SSIM={m_ssim:.3f}")


if __name__ == "__main__":
    main()
