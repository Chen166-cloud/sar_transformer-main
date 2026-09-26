"""GammaBridgeUNet — I2SB / OpenAI guided-diffusion `UNetModel` subclass that
injects an additional continuous conditioning signal: log-looks L(t).

The base UNet already fuses a sinusoidal time embedding via `self.time_embed`.
We add a parallel sinusoidal-in-log-L branch (`self.logL_embed`), then feed
`emb + time_emb + logL_emb` down through the UNet.

By subclassing we keep the third-party UNet file (`gbridge_core/gd/unet.py`)
literally unmodified.
"""

from __future__ import annotations
import math
import torch as th
import torch.nn as nn

from .gd.unet import UNetModel
from .gd.nn import timestep_embedding, linear


def logL_embedding(log_L: th.Tensor, dim: int) -> th.Tensor:
    """Sinusoidal embedding of log-looks, mirroring the standard timestep
    embedding. `log_L` is a (B,) tensor of natural-log looks (any range)."""
    half = dim // 2
    freqs = th.exp(
        -math.log(10000.0)
        * th.arange(start=0, end=half, dtype=th.float32, device=log_L.device)
        / max(half - 1, 1)
    )
    args = log_L.float()[:, None] * freqs[None]
    emb = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = th.nn.functional.pad(emb, (0, 1))
    return emb


class GammaBridgeUNet(UNetModel):
    """UNetModel + a log-L conditioning branch fused into the time embedding.

    Forward signature: `(x, timesteps, log_L) -> x0_hat`  (no class label).
    """

    def __init__(self, *args, **kwargs):
        # Disable UNetModel's own class-conditional branch — we use continuous cond.
        kwargs.setdefault("num_classes", None)
        super().__init__(*args, **kwargs)

        time_embed_dim = self.model_channels * 4
        self.logL_embed = nn.Sequential(
            linear(self.model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, x: th.Tensor, timesteps: th.Tensor, log_L: th.Tensor,
                cond: th.Tensor | None = None) -> th.Tensor:
        assert timesteps.ndim == 1 and timesteps.shape[0] == x.shape[0]
        assert log_L.ndim == 1 and log_L.shape[0] == x.shape[0], \
            f"log_L shape {tuple(log_L.shape)} vs batch {x.shape[0]}"

        hs = []
        t_emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        L_emb = self.logL_embed(logL_embedding(log_L, self.model_channels))
        emb = t_emb + L_emb

        if cond is not None:
            assert cond.shape == x.shape, f"cond {cond.shape} vs x {x.shape}"
            x_in = th.cat([x, cond], dim=1)
        else:
            x_in = x
        h = x_in.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        return self.out(h)


def build_default_unet(
    image_size: int = 256,
    in_channels: int = 1,
    out_channels: int = 1,
    model_channels: int = 64,
    channel_mult=(1, 2, 4),
    num_res_blocks: int = 2,
    attention_resolutions=(),
    use_scale_shift_norm: bool = True,
    dropout: float = 0.0,
) -> GammaBridgeUNet:
    """Small UNet suitable for 256x256 single-channel SAR/optical patches on 3080."""
    return GammaBridgeUNet(
        image_size=image_size,
        in_channels=in_channels,
        model_channels=model_channels,
        out_channels=out_channels,
        num_res_blocks=num_res_blocks,
        attention_resolutions=attention_resolutions,
        channel_mult=channel_mult,
        dropout=dropout,
        use_scale_shift_norm=use_scale_shift_norm,
        num_classes=None,
    )


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
