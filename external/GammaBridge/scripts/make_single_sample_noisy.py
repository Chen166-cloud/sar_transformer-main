"""Generate Gamma-noised versions of one clean image at several L values."""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def load_gray01(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr)


def save_gray01(x: torch.Tensor, path: Path) -> None:
    arr = (x.clamp(0.0, 1.0).cpu().numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path,
                   default=Path("/home/x/u/xuranh/gamma_bridge/data/external_test/clean/Set12/09.png"))
    p.add_argument("--out", type=Path,
                   default=Path("/home/x/u/xuranh/gamma_bridge/data/single_sample_test"))
    p.add_argument("--Ls", type=int, nargs="+",
                   default=[1, 2, 4, 8, 16, 32, 64, 128, 256])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(args.seed)

    x = load_gray01(args.src)
    stem = args.src.stem
    save_gray01(x, args.out / f"{stem}_clean.png")

    for L in args.Ls:
        L_t = torch.tensor(float(L))
        n = torch._standard_gamma(L_t.expand(x.shape), generator=g) / L_t
        y = x * n
        save_gray01(y, args.out / f"{stem}_L{L:03d}.png")
        print(f"L={L:>3d}  mean={y.mean():.3f}  std={y.std():.3f}  -> {args.out / f'{stem}_L{L:03d}.png'}")


if __name__ == "__main__":
    main()
