"""Smoke tests for the BSDS500-backed natural-image dataset."""

from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from data.optical_dataset import OpticalTrainDataset, PairedEvalDataset


def test_train_dataset_shapes():
    ds = OpticalTrainDataset(crop_size=128, length=64)
    dl = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0)
    batch = next(iter(dl))
    assert batch["x0"].shape == (8, 1, 128, 128), batch["x0"].shape
    assert torch.isfinite(batch["x0"]).all()
    assert (batch["x0"] > 0).all(), "intensities must be positive for multiplicative Gamma model"
    print(f"[OK] train dataset: {batch['x0'].shape}")


def test_paired_eval_dataset_ratio_stats():
    ds = PairedEvalDataset(crop_size=128, L_obs=1.0, num_samples=8)
    dl = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
    for batch in dl:
        assert batch["x0"].shape == (4, 1, 128, 128)
        assert batch["x_obs"].shape == (4, 1, 128, 128)
        assert (batch["x0"] > 0).all() and (batch["x_obs"] > 0).all()
    # r = x_obs / x0 should be Gamma(L=1)
    b = next(iter(DataLoader(ds, batch_size=8, shuffle=False)))
    ratio = (b["x_obs"] / b["x0"]).flatten().numpy()
    m, v = float(ratio.mean()), float(ratio.var(ddof=1))
    assert abs(m - 1.0) < 0.05, f"ratio mean {m}"
    assert abs(v - 1.0) < 0.15, f"ratio var {v}"
    # Determinism across two constructions
    ds2 = PairedEvalDataset(crop_size=128, L_obs=1.0, num_samples=8)
    assert torch.allclose(ds[0]["x_obs"], ds2[0]["x_obs"]), "PairedEvalDataset must be deterministic"
    print(f"[OK] paired eval: 8 samples, ratio mean={m:.4f} var={v:.4f}, deterministic")


def main():
    test_train_dataset_shapes()
    test_paired_eval_dataset_ratio_stats()
    print("\nAll data tests passed.")


if __name__ == "__main__":
    main()
