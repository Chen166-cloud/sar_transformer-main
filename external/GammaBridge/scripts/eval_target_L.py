"""Target-L eval sweep: stop reverse at L(t*)~L_out and record ratio stats
including KS-p. Writes eval/results/target_L_sweep.csv.
"""
from __future__ import annotations
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


CKPT = _ROOT / "weights/combo_v1_L1_ckpt_0015000.pt"
L_OUTS = [2, 4, 16, 66, 266, 10000]
NFE = 6
NUM_SAMPLES = 64
SEED = 42
OUT = _ROOT / "eval/results/target_L_sweep.csv"


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G, bridge, ckpt_args = load_model(CKPT, device)
    T = bridge.T
    L_arr = bridge.L.cpu().numpy()
    log_L_map = torch.log(bridge.L)
    use_cond = bool(ckpt_args.get("cond_x1"))

    ds = PairedEvalDataset(
        crop_size=ckpt_args["image_size"], L_obs=ckpt_args["L_obs"],
        num_samples=NUM_SAMPLES,
    )
    x0 = torch.stack([ds[i]["x0"] for i in range(len(ds))]).to(device)
    xobs = torch.stack([ds[i]["x_obs"] for i in range(len(ds))]).to(device)

    rows = []
    BATCH = 4
    for L_out in L_OUTS:
        target_step = int(np.argmin(np.abs(L_arr - L_out)))
        actual_L = float(L_arr[target_step])
        idx = np.linspace(target_step, T - 1, num=NFE + 1).round().astype(int)
        step_list = sorted(set(int(i) for i in idx))
        rev = step_list[::-1]

        outs = []
        for lo in range(0, x0.shape[0], BATCH):
            hi = min(lo + BATCH, x0.shape[0])
            cond = xobs[lo:hi] if use_cond else None
            fn = predict_x0_fn(G, log_L_map, cond=cond)
            xt = xobs[lo:hi].detach().clone()
            with torch.no_grad():
                for prev_step, step in zip(rev[1:], rev[:-1]):
                    x0_hat = fn(xt, step)
                    xt = bridge.p_posterior(prev_step, step, xt, x0_hat, ot_ode=True)
            outs.append(xt)
        out = torch.cat(outs, 0)

        x0_np = x0.cpu().numpy()
        xobs_np = xobs.cpu().numpy()
        out_np = out.cpu().numpy()
        for i in range(x0.shape[0]):
            # ratio r = xobs / x_t*  (Beta-coupled ratio)
            stats = ratio_stats(xobs_np[i, 0], out_np[i, 0], L_obs=1.0)
            rows.append({
                "L_out": L_out,
                "actual_L": actual_L,
                "target_step": target_step,
                "psnr_denoised": psnr_np(x0_np[i, 0], out_np[i, 0]),
                "ssim_denoised": ssim_np(x0_np[i, 0], out_np[i, 0]),
                **{k: float(v) for k, v in stats.items()},
            })
        print(f"[done] L_out={L_out}  actual_L={actual_L:.3f}  step={target_step}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"[save] {OUT}")

    import statistics
    for L_out in L_OUTS:
        subset = [r for r in rows if r["L_out"] == L_out]
        m_mean = statistics.mean(r["ratio_mean"] for r in subset)
        m_var  = statistics.mean(r["ratio_var"] for r in subset)
        m_ksp  = statistics.mean(r["ratio_ks_p"] for r in subset)
        print(f"  L_out={L_out:>5}  rmean={m_mean:.3f}  rvar={m_var:.3f}  ksp_mean={m_ksp:.3g}")


if __name__ == "__main__":
    main()
