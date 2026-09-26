"""gamma-Bridge synthetic-training loop.

Paper setup (Sec. Experiments): train the L-conditioned UNet on natural
images (BSDS500 train+val + DIV2K train HR, 1,100 crops of 256x256) with
synthetic L_obs=1 Gamma corruption. No real SAR imagery, no adversarial
losses, no unpaired pipeline.

Building blocks:
  * gbridge_core.diffusion.GammaBridge  -- discrete forward Gamma marginals
                                            + closed-form Gamma-Levy posterior
  * gbridge_core.network.GammaBridgeUNet -- UNet + log-L conditioning
  * data.optical_dataset.OpticalTrainDataset -- clean natural-image crops

Losses at each iteration (per-image step sampled uniformly in [1, T-1]):
    L_rec   = lambda_rec   * L1(x0_hat, x0)
    L_ratio = lambda_ratio * moment-match( r = x_obs / x0_hat  ~  Gamma(L_obs, L_obs) )
    L_cons  = lambda_cons  * L1( G(x_mid, step_mid, x_obs), x0 )     # two-step consistency
Total: L = lambda_rec * L_rec + lambda_ratio * L_ratio + lambda_cons * L_cons
(Paper defaults: lambda_rec = 10, lambda_ratio = 1, lambda_cons = 5.)
"""

from __future__ import annotations
import argparse
import contextlib
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from gbridge_core.diffusion import GammaBridge
from gbridge_core.network import build_default_unet, count_params
from data.optical_dataset import OpticalTrainDataset, PairedEvalDataset


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, default="checkpoints/gbridge")
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--T", type=int, default=100)
    p.add_argument("--L_obs", type=float, default=1.0)
    p.add_argument("--L_max", type=float, default=1.0e4)
    p.add_argument("--model_channels", type=int, default=64)
    p.add_argument("--lambda_rec", type=float, default=10.0,
                   help="L1(x0_hat, x0) weight (paper: lambda_rec = 10)")
    p.add_argument("--lambda_ratio", type=float, default=1.0,
                   help="ratio moment-match weight (paper: lambda_ratio = 1)")
    p.add_argument("--lambda_consistency", type=float, default=5.0,
                   help="two-step consistency L1 weight (paper: lambda_cons = 5)")
    p.add_argument("--bridge_forward", action="store_true",
                   help="use Gamma-Levy bridge q(x_t | x_0, x_obs) forward. "
                        "Non-trivial only with --cond_x1 (the marginal is x_obs-invariant otherwise).")
    p.add_argument("--cond_x1", action="store_true",
                   help="condition the UNet on x_obs as a second input channel "
                        "(I2SB-style); in_channels becomes 2.")
    p.add_argument("--lr_G", type=float, default=1e-4)
    p.add_argument("--betas", type=float, nargs=2, default=(0.5, 0.999))
    p.add_argument("--total_steps", type=int, default=15_000)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--ckpt_every", type=int, default=2000)
    p.add_argument("--dataset_length", type=int, default=8000)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=1,
                   help="gradient accumulation micro-steps per optimizer update")
    p.add_argument("--extra_roots", type=str, nargs="*", default=None,
                   help="additional image root dirs to add to the training pool "
                        "(paper uses BSDS500 train+val + DIV2K HR train)")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--min_step", type=int, default=1,
                   help="lowest bridge step to sample (0 = clean, skip trivial)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ema_decay", type=float, default=0.999,
                   help="EMA decay for G's shadow weights used at eval/ckpt (0 = disabled)")
    p.add_argument("--resume", type=str, default=None,
                   help="warm-start G / G_ema from a ckpt (optimizer state NOT restored)")
    p.add_argument("--find_unused", action="store_true")
    # Ablation switches (paper Sec. Ablation studies)
    p.add_argument("--naive_posterior", action="store_true",
                   help="ablation: return x0_hat at every step (ignore x_n)")
    p.add_argument("--schedule", type=str, default="log", choices=["log", "linear"],
                   help="L(t) interpolation mode; paper uses 'log'")
    p.add_argument("--direct_pred", action="store_true",
                   help="ablation: predict x0 directly (relu(x_t + G)) instead of "
                        "log-residual (x_t * exp(G)).")
    return p.parse_args()


class EMAShadow:
    """Standard EMA on module parameters. Shadow is used at eval / ckpt time."""

    def __init__(self, module: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.enabled = decay > 0.0
        if not self.enabled:
            return
        self.shadow = {n: p.detach().clone() for n, p in module.state_dict().items()}

    @torch.no_grad()
    def update(self, module: torch.nn.Module):
        if not self.enabled:
            return
        sd = module.state_dict()
        for n, p in sd.items():
            if p.dtype.is_floating_point and n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[n] = p.detach().clone()

    def copy_to(self, module: torch.nn.Module):
        if self.enabled:
            module.load_state_dict(self.shadow, strict=False)

    def state_dict(self):
        return dict(self.shadow) if self.enabled else {}


def ddp_setup():
    if int(os.environ.get("WORLD_SIZE", 1)) > 1:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return dist.get_rank(), dist.get_world_size(), local_rank, True
    return 0, 1, 0, False


def ddp_cleanup(is_ddp: bool):
    if is_ddp and dist.is_initialized():
        dist.destroy_process_group()


def is_main(rank: int) -> bool:
    return rank == 0


def predict_x0(G, x_t, step, log_L, cond=None,
               log_residual_clip: float = 5.0, direct_pred: bool = False):
    """Log-residual head (paper Eq. log-residual): x0_hat = x_t * exp(G(...)).
    Ablation: direct_pred = True uses x0_hat = relu(x_t + G(...))."""
    out = G(x_t, step.float(), log_L, cond=cond)
    if direct_pred:
        return torch.relu(x_t + out)
    return x_t * torch.exp(out.clamp(-log_residual_clip, log_residual_clip))


def moment_match_gamma_loss(x_obs, x0_hat, L_obs: float):
    """Paper Eq. loss-ratio: r = x_obs / x0_hat should follow Gamma(L_obs, L_obs),
    with E[r]=1 and Var(r)=1/L_obs. Sample mean and variance pooled over the
    batch and spatial pixels."""
    eps = 1e-6
    r = x_obs / x0_hat.clamp_min(eps)
    m = r.mean()
    v = r.var(unbiased=False)
    target_v = torch.tensor(1.0 / float(L_obs), device=r.device, dtype=r.dtype)
    return (m - 1.0).pow(2) + (v - target_v).pow(2)


@torch.no_grad()
def eval_psnr(model, bridge, eval_ds, device, eval_chunk=4,
              cond_x1=False, direct_pred=False):
    """Quick 1-step DDIM-endpoint eval: PSNR + ratio stats on a held-out set."""
    model.eval()
    x0s = torch.stack([eval_ds[i]["x0"] for i in range(len(eval_ds))]).to(device)
    x_obss = torch.stack([eval_ds[i]["x_obs"] for i in range(len(eval_ds))]).to(device)
    parts = []
    for lo in range(0, x_obss.shape[0], eval_chunk):
        hi = min(lo + eval_chunk, x_obss.shape[0])
        x_obs = x_obss[lo:hi]
        step_final = torch.full((x_obs.shape[0],), bridge.T - 1, device=device, dtype=torch.long)
        log_L = torch.log(bridge.L_at(step_final))
        cond_eval = x_obs if cond_x1 else None
        parts.append(predict_x0(model, x_obs, step_final, log_L,
                                cond=cond_eval, direct_pred=direct_pred))
    x0_hat = torch.cat(parts, dim=0)
    mse = ((x0s - x0_hat).clamp_min(0) ** 2).mean().item()
    psnr = 10.0 * math.log10(1.0 / max(mse, 1e-12))
    ratio = (x_obss / x0_hat.clamp_min(1e-6)).flatten().cpu().numpy()
    model.train()
    return {
        "eval_psnr": psnr,
        "eval_ratio_mean": float(np.mean(ratio)),
        "eval_ratio_var": float(np.var(ratio, ddof=1)),
    }


def main():
    args = get_args()
    rank, world_size, local_rank, is_ddp = ddp_setup()
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.out_dir)
    if is_main(rank):
        out_dir.mkdir(parents=True, exist_ok=True)
        log_dir = _ROOT / "logs" / out_dir.name
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(str(log_dir))
        print(f"[env] rank={rank}/{world_size}  device={device}  ddp={is_ddp}", flush=True)
    else:
        writer = None

    # --- data ---
    train_ds = OpticalTrainDataset(
        crop_size=args.image_size, length=args.dataset_length,
        extra_roots=args.extra_roots,
    )
    if is_ddp:
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True,
        )
        train_dl = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    else:
        train_sampler = None
        train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    eval_ds = PairedEvalDataset(crop_size=args.image_size, L_obs=args.L_obs, num_samples=32)
    if is_main(rank):
        print(f"[data] train pool={len(train_ds.pool)}  extra_roots={args.extra_roots}", flush=True)
        print(f"[data] train iters/epoch={len(train_ds)//args.batch_size}  eval samples={len(eval_ds)}", flush=True)

    # --- bridge ---
    bridge = GammaBridge(num_timesteps=args.T, L_obs=args.L_obs, L_max=args.L_max,
                         schedule=args.schedule, device=device)

    # --- network ---
    in_ch = 2 if args.cond_x1 else 1
    G_core = build_default_unet(
        image_size=args.image_size, in_channels=in_ch, out_channels=1,
        model_channels=args.model_channels, channel_mult=(1, 2, 4),
    ).to(device)
    if is_main(rank):
        print(f"[net] G params = {count_params(G_core)/1e6:.2f}M", flush=True)

    G = DDP(G_core, device_ids=[local_rank], find_unused_parameters=args.find_unused,
            broadcast_buffers=False) if is_ddp else G_core

    opt_G = torch.optim.Adam((G.module if is_ddp else G).parameters(),
                             lr=args.lr_G, betas=tuple(args.betas))

    # --- resume ---
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        src_state = ck.get("G_ema") or ck["G"]
        missing, unexpected = G_core.load_state_dict(src_state, strict=False)
        if is_main(rank):
            print(f"[resume] loaded G from {args.resume} "
                  f"(used_G_ema={ck.get('G_ema') is not None}) "
                  f"missing={len(missing)} unexpected={len(unexpected)}", flush=True)
        if is_ddp:
            for p in G_core.parameters():
                dist.broadcast(p.data, src=0)

    # --- EMA ---
    ema = EMAShadow(G_core, decay=args.ema_decay)
    G_eval_copy = build_default_unet(
        image_size=args.image_size, in_channels=in_ch, out_channels=1,
        model_channels=args.model_channels, channel_mult=(1, 2, 4),
    ).to(device) if ema.enabled else None

    # --- train loop ---
    global_step = 0
    epoch = 0
    t0 = time.time()
    G.train()
    accum = max(1, int(args.grad_accum))

    def _new_iter():
        nonlocal epoch
        if is_ddp and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        epoch += 1
        return iter(train_dl)

    dl_iter = _new_iter()

    def _next_batch():
        nonlocal dl_iter
        try:
            return next(dl_iter)
        except StopIteration:
            dl_iter = _new_iter()
            return next(dl_iter)

    while global_step < args.total_steps:
        opt_G.zero_grad(set_to_none=True)
        acc = {"G": 0.0, "rec": 0.0, "ratio": 0.0, "cons": 0.0}

        for micro in range(accum):
            batch = _next_batch()
            x0 = batch["x0"].to(device, non_blocking=True)
            B = x0.size(0)

            # Per-image discrete step in [min_step, T-1].
            step = torch.randint(args.min_step, bridge.T, (B,), device=device)
            L_step = bridge.L_at(step)
            log_L = torch.log(L_step)

            # x_obs = single Gamma draw at step T-1 (observation endpoint).
            step_final = torch.full((B,), bridge.T - 1, device=device, dtype=torch.long)
            x_obs = bridge.q_sample(step_final, x0)

            if args.bridge_forward:
                x_t = bridge.q_sample_bridge(step, x0, x_obs)
            else:
                x_t = bridge.q_sample(step, x0)
            cond = x_obs if args.cond_x1 else None

            is_last = (micro == accum - 1)
            ctx_G = G.no_sync() if (is_ddp and not is_last) else contextlib.nullcontext()

            with ctx_G:
                x0_hat = predict_x0(G, x_t, step, log_L, cond=cond,
                                    direct_pred=args.direct_pred)
                loss_rec = F.l1_loss(x0_hat, x0) * args.lambda_rec
                loss_ratio = moment_match_gamma_loss(x_obs, x0_hat, args.L_obs) * args.lambda_ratio

                # ---- Two-step consistency (paper L_cons) ----
                if args.lambda_consistency > 0:
                    step_mid = (torch.rand(B, device=device) * step.float()).long()
                    L_step_mid = bridge.L_at(step_mid)
                    with torch.no_grad():
                        if args.naive_posterior:
                            x_mid = x0_hat.detach()
                        else:
                            alpha_c = (L_step / L_step_mid).view(-1, 1, 1, 1)
                            x_mid = alpha_c * x_t + (1.0 - alpha_c) * x0_hat.detach()
                    log_L_mid = torch.log(L_step_mid)
                    x0_hat_mid = predict_x0(G, x_mid, step_mid, log_L_mid,
                                            cond=cond, direct_pred=args.direct_pred)
                    loss_cons = F.l1_loss(x0_hat_mid, x0) * args.lambda_consistency
                else:
                    loss_cons = torch.zeros((), device=device)

                loss_G = loss_rec + loss_ratio + loss_cons
                (loss_G / accum).backward()

            acc["G"] += loss_G.item()
            acc["rec"] += loss_rec.item()
            acc["ratio"] += loss_ratio.item()
            acc["cons"] += loss_cons.item()

        opt_G.step()
        ema.update(G.module if is_ddp else G)

        for k_ in acc:
            acc[k_] /= accum
        global_step += 1

        if is_main(rank) and (global_step % args.log_every == 0 or global_step == 1):
            it_per_s = global_step / max(time.time() - t0, 1e-6)
            msg = (f"step {global_step:>6}  it/s {it_per_s:5.2f}  ws={world_size}  "
                   f"G {acc['G']:.3f}  rec {acc['rec']:.3f}  "
                   f"ratio {acc['ratio']:.3f}  cons {acc['cons']:.3f}")
            print(msg, flush=True)
            for k_ in acc:
                writer.add_scalar(f"loss/{k_}", acc[k_], global_step)

        if is_main(rank) and global_step % args.eval_every == 0:
            if ema.enabled:
                G_eval_copy.load_state_dict(ema.shadow, strict=False)
                eval_G = G_eval_copy
            else:
                eval_G = G.module if is_ddp else G
            metrics = eval_psnr(eval_G, bridge, eval_ds, device,
                                cond_x1=args.cond_x1, direct_pred=args.direct_pred)
            print("  [eval-ema] " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()),
                  flush=True)
            for k, v in metrics.items():
                writer.add_scalar(f"eval/{k}", v, global_step)

        if is_main(rank) and (global_step % args.ckpt_every == 0 or global_step == args.total_steps):
            ckpt = {
                "step": global_step,
                "G": (G.module if is_ddp else G).state_dict(),
                "G_ema": ema.state_dict(),
                "args": vars(args),
            }
            path = out_dir / f"ckpt_{global_step:07d}.pt"
            torch.save(ckpt, path)
            print(f"  [ckpt] saved {path}", flush=True)

    if is_main(rank):
        writer.close()
        print("done.", flush=True)
    ddp_cleanup(is_ddp)


if __name__ == "__main__":
    main()
