"""Zero-shot inference of a natural-image-trained γ-Bridge on external
SAR-like test tiles (synthetic multi-L and real SAR-viz PNG/JPG).

Pipeline per image:
  1. Load as grayscale intensity in [0, 1].
  2. Estimate effective look number L̂ via homogeneous-patch ENL
     (P90 quantile over 32² patches at stride 16) — see Section 3.8.
  3. Smart-start at t* = arg min_t |L(t) − L_smart|, where L_smart is
     * synthetic: the ENL{K} label parsed from filename (ground-truth L)
     * real:      the estimated L̂ (unless --real_L_in overrides)
  4. Reverse to step 0 (full denoise) with OT-ODE, 512² tiles + Hann
     overlap blending for images > 512 px.
  5. Report: PSNR/SSIM (if clean ref available), ratio-image stats,
     homogeneous-patch ENL of noisy and denoised, EPD-ROA.

Layout:
  data/external_test/synthetic/{Kodak24,Set12,McM}/<idx>_gamma_noise_ENL<L>.png
  data/external_test/real/{Sentinel-1,TerraSAR-X,Gaofen-3,miniSAR,FARAD_Ka,FARAD_X}/*.{png,jpg}
  data/external_test/clean/{Kodak24,Set12,McM}/*  (optional)
"""

from __future__ import annotations
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from gbridge_core.diffusion import GammaBridge
from gbridge_core.network import build_default_unet
from eval.metrics import (
    psnr, ssim, ratio_stats,
    enl, enl_homogeneous,
    # Hu 2025 metric set matching paper Tables V-X:
    enl_whole, epi_cv, epd_diff, sqi_sun, mean_100_convention,
)


def to_u8(x: np.ndarray) -> np.ndarray:
    """Convert float [0,1] to uint8 [0,255] for Hu-style metrics."""
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)


ENL_RE = re.compile(r"ENL(\d+)")


def load_gray_intensity(path: Path) -> np.ndarray:
    im = Image.open(path).convert("L")
    return np.asarray(im, dtype=np.float32) / 255.0


def pad_to_multiple(x: torch.Tensor, mult: int = 8):
    _, _, H, W = x.shape
    Hn = ((H + mult - 1) // mult) * mult
    Wn = ((W + mult - 1) // mult) * mult
    x = torch.nn.functional.pad(x, (0, Wn - W, 0, Hn - H), mode="reflect")
    return x, (H, W)


def find_target_step(bridge: GammaBridge, L_in: float) -> int:
    diffs = bridge.L.cpu().numpy() - float(L_in)
    return int(np.argmin(np.abs(diffs)))


def load_model(ckpt_path: Path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ckpt["args"]
    in_ch = 2 if a.get("cond_x1") else 1
    G = build_default_unet(
        image_size=a["image_size"],
        in_channels=in_ch, out_channels=1,
        model_channels=a["model_channels"],
        channel_mult=(1, 2, 4),
    ).to(device)
    state = ckpt.get("G_ema") if ckpt.get("G_ema") else ckpt["G"]
    G.load_state_dict(state, strict=False)
    G.eval()
    bridge = GammaBridge(num_timesteps=a["T"], L_obs=a["L_obs"], L_max=a["L_max"], device=device)
    return G, bridge, a


def make_pred_x0(G, bridge, cond=None, log_residual_clip=5.0):
    log_L_map = torch.log(bridge.L)
    def fn(xt, step_int: int):
        step_t = torch.tensor([step_int] * xt.shape[0], device=xt.device, dtype=torch.long)
        log_L = log_L_map[step_int].expand(xt.shape[0])
        log_res = G(xt, step_t.float(), log_L, cond=cond)
        log_res = log_res.clamp(-log_residual_clip, log_residual_clip)
        return xt * torch.exp(log_res)
    return fn


def make_smartstart_step_list(t_start: int, num_steps: int) -> list[int]:
    if t_start <= 0:
        return [0]
    idx = np.linspace(0, t_start, num=max(num_steps, 2)).round().astype(int)
    idx = sorted(set(int(i) for i in idx))
    if idx[0] != 0:
        idx = [0] + idx
    if idx[-1] != t_start:
        idx.append(t_start)
    return sorted(set(idx))


def _denoise_tile(G, bridge, x_tile: torch.Tensor, L_in: float,
                  num_steps: int, ot_ode: bool) -> torch.Tensor:
    x_padded, (H, W) = pad_to_multiple(x_tile, mult=8)
    t_start = find_target_step(bridge, L_in)
    if t_start == 0:
        return x_padded[..., :H, :W]
    steps = make_smartstart_step_list(t_start, num_steps)
    fn = make_pred_x0(G, bridge, cond=x_padded)
    with torch.no_grad():
        xs, _ = bridge.ddpm_sampling(steps=steps, pred_x0_fn=fn, x1=x_padded,
                                     ot_ode=ot_ode, verbose=False)
    x_final = xs[0, 0].to(x_padded.device)
    return x_final[..., :H, :W].unsqueeze(0)


def _hann_window_2d(H: int, W: int, device) -> torch.Tensor:
    wh = torch.hann_window(H, periodic=False, device=device) + 0.02
    ww = torch.hann_window(W, periodic=False, device=device) + 0.02
    return (wh[:, None] * ww[None, :])[None, None]


def _tile_positions(size: int, tile_size: int, stride: int) -> list[int]:
    if size <= tile_size:
        return [0]
    pos = list(range(0, size - tile_size + 1, stride))
    if pos[-1] + tile_size < size:
        pos.append(size - tile_size)
    return pos


def denoise_one(G, bridge, x_obs_np: np.ndarray, L_in: float,
                num_steps: int, device, ot_ode: bool = True,
                tile_size: int = 512) -> np.ndarray:
    """Tiled denoising with 50%-overlap + Hann-window blending. Removes the
    tile-boundary discontinuities that inflate whole-image variance and
    tank ENL on large images."""
    x = torch.from_numpy(x_obs_np).float().unsqueeze(0).unsqueeze(0).to(device)
    _, _, H, W = x.shape
    if H <= tile_size and W <= tile_size:
        x_hat = _denoise_tile(G, bridge, x, L_in, num_steps, ot_ode)
        return x_hat[0, 0].cpu().numpy()
    stride = tile_size // 2
    out = torch.zeros_like(x)
    wsum = torch.zeros_like(x)
    hann = _hann_window_2d(tile_size, tile_size, device)
    for i in _tile_positions(H, tile_size, stride):
        for j in _tile_positions(W, tile_size, stride):
            tile = x[:, :, i:i + tile_size, j:j + tile_size]
            tile_hat = _denoise_tile(G, bridge, tile, L_in, num_steps, ot_ode)
            out[:, :, i:i + tile_size, j:j + tile_size] += tile_hat * hann
            wsum[:, :, i:i + tile_size, j:j + tile_size] += hann
    out = out / wsum.clamp(min=1e-6)
    return out[0, 0].cpu().numpy()


def save_side_by_side(noisy: np.ndarray, denoised: np.ndarray, path: Path):
    a = (np.clip(noisy, 0, 1) * 255).astype(np.uint8)
    b = (np.clip(denoised, 0, 1) * 255).astype(np.uint8)
    grid = np.concatenate([a, b], axis=1)
    Image.fromarray(grid).save(path)


def save_pair(noisy: np.ndarray, denoised: np.ndarray, out_dir: Path,
              stem: str, save_individuals: bool = True):
    """Save noisy+denoised as a side-by-side PNG and individually so that
    each can be dropped into a paper figure independently."""
    out_dir.mkdir(parents=True, exist_ok=True)
    save_side_by_side(noisy, denoised, out_dir / f"{stem}_pair.png")
    if save_individuals:
        for arr, name in [(noisy, "noisy"), (denoised, "denoised")]:
            im = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
            Image.fromarray(im).save(out_dir / f"{stem}_{name}.png")


def parse_index(name: str):
    m = re.match(r"^(\d+)", name)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else None


def find_clean_ref(clean_root: Path, subset: str, idx: int):
    if subset == "Kodak24":
        p = clean_root / "Kodak24" / f"kodim{idx + 1:02d}.png"
        return p if p.exists() else None
    if subset == "Set12":
        p = clean_root / "Set12" / f"{idx:02d}.png"
        return p if p.exists() else None
    if subset == "McM":
        for pat in [f"{idx}.tif", f"{idx:02d}.tif", f"{idx}.png", f"{idx:02d}.png"]:
            p = clean_root / "McM" / pat
            if p.exists():
                return p
    return None


def load_ref_gray(path: Path) -> np.ndarray:
    return load_gray_intensity(path)


def match_resize_gray(clean: np.ndarray, target_shape) -> np.ndarray:
    if clean.shape == target_shape:
        return clean
    im = Image.fromarray((np.clip(clean, 0, 1) * 255).astype(np.uint8))
    im = im.resize((target_shape[1], target_shape[0]), Image.BICUBIC)
    return np.asarray(im, dtype=np.float32) / 255.0


def clamp_L(L_est: float, L_min: float, L_max: float) -> float:
    return float(np.clip(L_est, L_min, L_max))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="weights/combo_v1_L1_ckpt_0015000.pt")
    p.add_argument("--test_root", default="data/external_test")
    p.add_argument("--clean_root", default="data/external_test/clean")
    p.add_argument("--out_dir", default="eval/results/external_test",
                   help="Metrics + logs go here.")
    p.add_argument("--fig_dir", default="figures/external_test",
                   help="Per-image noisy+denoised PNGs (paper-ready). ")
    p.add_argument("--num_steps", type=int, default=5)
    p.add_argument("--stochastic", action="store_true")
    p.add_argument("--real_L_in", type=float, default=None,
                   help="Override auto-estimated L on real SAR (default: auto).")
    p.add_argument("--norm_mean_real", action="store_true", default=True,
                   help="Post-hoc rescale real-SAR outputs to match input mean; "
                        "standard SAR-despeckling practice, corrects the natural-"
                        "image-prior brightness drift.")
    p.add_argument("--no_norm_mean_real", dest="norm_mean_real", action="store_false")
    p.add_argument("--max_per_dir", type=int, default=None)
    p.add_argument("--only", default=None)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = _ROOT / ckpt_path
    G, bridge, ckpt_args = load_model(ckpt_path, device)
    ot_ode = not args.stochastic
    print(f"[load] T={bridge.T}  L_obs={bridge.L_obs}  L_max={bridge.L_max}  cond_x1={ckpt_args.get('cond_x1')}", flush=True)
    print(f"[cfg ] num_steps={args.num_steps}  ot_ode={ot_ode}  "
          f"real_L_in={'auto (patch-ENL)' if args.real_L_in is None else args.real_L_in}", flush=True)

    test_root = _ROOT / args.test_root if not Path(args.test_root).is_absolute() else Path(args.test_root)
    clean_root = _ROOT / args.clean_root if not Path(args.clean_root).is_absolute() else Path(args.clean_root)
    out_dir = _ROOT / args.out_dir if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = _ROOT / args.fig_dir if not Path(args.fig_dir).is_absolute() else Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    only = set(args.only.split(",")) if args.only else None
    all_metrics = []
    L_bounds = (float(bridge.L_obs), float(bridge.L_max))

    synth_subsets = ["Kodak24", "Set12", "McM"]
    real_subsets = ["Sentinel-1", "TerraSAR-X", "Gaofen-3", "miniSAR", "FARAD_Ka", "FARAD_X"]

    # --- Synthetic ---
    for subset in synth_subsets:
        if only and subset not in only: continue
        sdir = test_root / "synthetic" / subset
        if not sdir.exists(): continue
        files = sorted(list(sdir.glob("*.png")) + list(sdir.glob("*.jpg")))
        if args.max_per_dir: files = files[: args.max_per_dir]
        sub_out = fig_dir / "synthetic" / subset
        sub_out.mkdir(parents=True, exist_ok=True)
        print(f"\n[synthetic/{subset}] {len(files)} files", flush=True)
        for fp in files:
            m_enl = ENL_RE.search(fp.name)
            if not m_enl: continue
            L_gt = int(m_enl.group(1))
            idx = parse_index(fp.name)
            x_obs_np = load_gray_intensity(fp)
            L_est = enl_homogeneous(x_obs_np)
            L_smart = clamp_L(L_gt, *L_bounds)  # synthetic: use GT label
            x_hat = denoise_one(G, bridge, x_obs_np, L_smart, args.num_steps, device, ot_ode=ot_ode)
            save_pair(x_obs_np, x_hat, sub_out, fp.stem)
            u8_hat = to_u8(x_hat)
            u8_obs = to_u8(x_obs_np)
            m = {"subset": f"synthetic/{subset}", "file": fp.name,
                 "L_gt": L_gt, "L_est_patchENL": L_est, "L_smart": L_smart,
                 "H": x_obs_np.shape[0], "W": x_obs_np.shape[1],
                 "input_mean": float(x_obs_np.mean()), "output_mean": float(x_hat.mean()),
                 # Hu 2025 single-image metrics on denoised uint8, paper Table conv:
                 "enl":   enl_whole(u8_hat),
                 "epi":   epi_cv(u8_hat),
                 "epd_h": epd_diff(u8_hat, "H"),
                 "epd_v": epd_diff(u8_hat, "V"),
                 "sqi":   sqi_sun(u8_hat),
                 "mean":  mean_100_convention(u8_hat),
                 # patch-ENL kept for L̂ diagnostics only:
                 "enl_in_patch": enl_homogeneous(x_obs_np),
                 "enl_out_patch": enl_homogeneous(x_hat)}
            clean_p = find_clean_ref(clean_root, subset, idx) if idx is not None else None
            if clean_p is not None:
                # Hu convention: read GT as uint8 grayscale, compute PSNR/SSIM
                # with per-image data_range = gt.max() - gt.min().
                import cv2 as _cv2  # local alias
                clean_u8 = _cv2.imread(str(clean_p), _cv2.IMREAD_GRAYSCALE)
                if clean_u8 is not None:
                    if clean_u8.shape != u8_hat.shape:
                        clean_u8 = _cv2.resize(clean_u8, (u8_hat.shape[1], u8_hat.shape[0]))
                    dr = float(clean_u8.max() - clean_u8.min())
                    if dr > 0:
                        m["psnr_denoised"] = float(psnr(clean_u8, u8_hat, max_val=None))
                        m["psnr_noisy"]    = float(psnr(clean_u8, u8_obs, max_val=None))
                        m["ssim_denoised"] = float(ssim(clean_u8, u8_hat, data_range=None))
                        m["ssim_noisy"]    = float(ssim(clean_u8, u8_obs, data_range=None))
            try:
                m.update(ratio_stats(x_obs_np, x_hat, L_obs=L_gt))
            except Exception as e:
                m["ratio_error"] = str(e)
            all_metrics.append(m)
            psnr_str = f"{m.get('psnr_denoised', float('nan')):+6.2f}" if 'psnr_denoised' in m else "  n/a "
            print(f"  {fp.name:40s}  L_gt={L_gt:3d}  L_est={L_est:6.2f}  PSNR={psnr_str} dB", flush=True)

    # --- Real ---
    for subset in real_subsets:
        if only and subset not in only: continue
        sdir = test_root / "real" / subset
        if not sdir.exists(): continue
        files = sorted(list(sdir.glob("*.png")) + list(sdir.glob("*.jpg")))
        if args.max_per_dir: files = files[: args.max_per_dir]
        sub_out = fig_dir / "real" / subset
        sub_out.mkdir(parents=True, exist_ok=True)
        print(f"\n[real/{subset}] {len(files)} files", flush=True)
        for fp in files:
            x_obs_np = load_gray_intensity(fp)
            L_est = enl_homogeneous(x_obs_np)
            if args.real_L_in is not None:
                L_smart = clamp_L(args.real_L_in, *L_bounds)
                mode = "override"
            else:
                L_smart = clamp_L(L_est, *L_bounds)
                mode = "auto"
            x_hat = denoise_one(G, bridge, x_obs_np, L_smart, args.num_steps, device, ot_ode=ot_ode)
            if args.norm_mean_real:
                obs_mean = float(x_obs_np.mean())
                out_mean = float(x_hat.mean())
                if out_mean > 1e-6:
                    x_hat = x_hat * (obs_mean / out_mean)
            save_pair(x_obs_np, x_hat, sub_out, fp.stem)
            u8_hat = to_u8(x_hat)
            enl_in_p = enl_homogeneous(x_obs_np)
            enl_out_p = enl_homogeneous(x_hat)
            m = {"subset": f"real/{subset}", "file": fp.name,
                 "L_est_patchENL": L_est, "L_smart": L_smart, "L_mode": mode,
                 "H": x_obs_np.shape[0], "W": x_obs_np.shape[1],
                 "input_mean": float(x_obs_np.mean()), "output_mean": float(x_hat.mean()),
                 # Hu 2025 single-image metrics on denoised uint8, paper Table conv:
                 "enl":   enl_whole(u8_hat),
                 "epi":   epi_cv(u8_hat),
                 "epd_h": epd_diff(u8_hat, "H"),
                 "epd_v": epd_diff(u8_hat, "V"),
                 "sqi":   sqi_sun(u8_hat),
                 "mean":  mean_100_convention(u8_hat),
                 # patch-ENL kept for L̂ diagnostics only:
                 "enl_in_patch": enl_in_p, "enl_out_patch": enl_out_p,
                 "enl_ratio_patch": enl_out_p / max(enl_in_p, 1e-6)}
            try:
                m.update(ratio_stats(x_obs_np, x_hat, L_obs=L_smart))
            except Exception as e:
                m["ratio_error"] = str(e)
            all_metrics.append(m)
            print(f"  {fp.name:40s}  L̂={L_est:5.2f}  ({mode})  "
                  f"ENL={m['enl']:.2f}  EPI={m['epi']:.4f}  "
                  f"EPD_H={m['epd_h']:.3f}  SQI={m['sqi']:.3f}  Mean={m['mean']:.3f}", flush=True)

    # --- Summary ---
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    agg_keys_synth = ["psnr_denoised", "psnr_noisy", "ssim_denoised", "ssim_noisy",
                      "enl", "epi", "epd_h", "epd_v", "sqi", "mean",
                      "L_est_patchENL"]
    agg_keys_real = ["enl", "epi", "epd_h", "epd_v", "sqi", "mean",
                     "enl_in_patch", "enl_out_patch",
                     "L_est_patchENL", "L_smart"]
    print("\n=== SYNTHETIC per-subset averages (grouped by L_gt) ===")
    grouped = defaultdict(lambda: defaultdict(list))
    for m in all_metrics:
        if not m["subset"].startswith("synthetic/"): continue
        key = (m["subset"], m["L_gt"])
        for k in agg_keys_synth:
            if k in m: grouped[key][k].append(m[k])
    if grouped:
        hdr = ["subset", "L_gt", "N"] + agg_keys_synth
        print("  ".join(f"{h:>16}" for h in hdr))
        for (subset, L_gt), vals in sorted(grouped.items()):
            N = max((len(v) for v in vals.values()), default=0)
            row = [subset, str(L_gt), str(N)]
            for k in agg_keys_synth:
                v = vals.get(k, [])
                row.append(f"{float(np.mean(v)):.3f}" if v else "-")
            print("  ".join(f"{c:>16}" for c in row))

    print("\n=== REAL per-subset averages ===")
    grouped2 = defaultdict(lambda: defaultdict(list))
    for m in all_metrics:
        if not m["subset"].startswith("real/"): continue
        for k in agg_keys_real:
            if k in m: grouped2[m["subset"]][k].append(m[k])
    if grouped2:
        hdr = ["subset", "N"] + agg_keys_real
        print("  ".join(f"{h:>16}" for h in hdr))
        for subset, vals in sorted(grouped2.items()):
            N = max((len(v) for v in vals.values()), default=0)
            row = [subset, str(N)]
            for k in agg_keys_real:
                v = vals.get(k, [])
                row.append(f"{float(np.mean(v)):.3f}" if v else "-")
            print("  ".join(f"{c:>16}" for c in row))


if __name__ == "__main__":
    main()
