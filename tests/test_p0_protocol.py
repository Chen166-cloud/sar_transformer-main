import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from numeric_domain import (
    inverse_log_transform_01_numpy,
    inverse_log_transform_01_torch,
    log_transform_01_numpy,
    log_transform_01_torch,
    normalize_real_intensity,
)
from resplit_real_sar_dataset import (
    assert_group_disjoint,
    grouped_split,
    parent_group_id,
)
from sar_metrics import enl, psnr, select_homogeneous_rois, ssim
from ablation_config import ABLATION_PRESETS, resolve_ablation_preset
from build_bsds500_split_manifest import source_split
from generate_nwpu_sar_dataset import stable_seed, synthesize_noisy


class NumericDomainTests(unittest.TestCase):
    def test_numpy_log_round_trip(self):
        values = np.linspace(0.0, 1.0, 4097, dtype=np.float32).reshape(1, -1)
        restored = inverse_log_transform_01_numpy(log_transform_01_numpy(values))
        self.assertLess(float(np.max(np.abs(restored - values))), 1e-6)

    def test_torch_log_round_trip(self):
        values = torch.linspace(0.0, 1.0, 4097).reshape(1, 1, 1, -1)
        restored = inverse_log_transform_01_torch(log_transform_01_torch(values))
        self.assertLess(float(torch.max(torch.abs(restored - values))), 1e-6)

    def test_real_normalization_is_bounded(self):
        image = np.arange(256, dtype=np.float32).reshape(16, 16)
        normalized = normalize_real_intensity(image)
        self.assertEqual(normalized.dtype, np.float32)
        self.assertEqual(float(normalized.min()), 0.0)
        self.assertEqual(float(normalized.max()), 1.0)


class GroupSplitTests(unittest.TestCase):
    def test_parent_id(self):
        self.assertEqual(parent_group_id("0_10240_y256_x0.mat"), "0_10240")

    def test_grouped_split_is_disjoint_and_complete(self):
        records = []
        for group in range(20):
            for y, x in ((0, 0), (0, 256), (256, 0), (256, 256)):
                records.append((f"train/{group}_0_y{y}_x{x}.mat", "train"))
        files, groups = grouped_split(records, seed=42, ratios=(8, 1, 1))
        assert_group_disjoint(groups)
        self.assertEqual(sum(map(len, files.values())), len(records))
        for split, paths in files.items():
            allowed = set(groups[split])
            self.assertTrue(all(parent_group_id(path) in allowed for path in paths))

    def test_overlap_assertion_fails(self):
        with self.assertRaises(AssertionError):
            assert_group_disjoint({"train": ["a"], "val": ["a"], "test": ["b"]})


class SyntheticSplitTests(unittest.TestCase):
    def test_official_bsds500_prefix_mapping(self):
        self.assertEqual(source_split("trn_100075.mat"), "train")
        self.assertEqual(source_split("val_101085.mat"), "val")
        self.assertEqual(source_split("tst_100007.mat"), "test")

    def test_unknown_prefix_fails(self):
        with self.assertRaises(ValueError):
            source_split("unknown_1.mat")

    def test_nwpu_synthesis_is_deterministic_and_bounded(self):
        clean = np.linspace(0.0, 1.0, 64 * 64, dtype=np.float32).reshape(64, 64)
        seed = stable_seed(20260904, "airplane/airplane_00000.jpg")
        first = synthesize_noisy(clean, np.random.default_rng(seed))
        second = synthesize_noisy(clean, np.random.default_rng(seed))
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.dtype, np.float32)
        self.assertGreaterEqual(float(first.min()), 0.0)
        self.assertLessEqual(float(first.max()), 1.0)


class MetricTests(unittest.TestCase):
    def test_identical_images(self):
        image = np.linspace(0, 1, 64 * 64, dtype=np.float32).reshape(64, 64)
        self.assertTrue(np.isinf(psnr(image, image)))
        self.assertAlmostEqual(ssim(image, image), 1.0, places=6)

    def test_known_psnr(self):
        target = np.zeros((32, 32), dtype=np.float32)
        prediction = np.full((32, 32), 0.1, dtype=np.float32)
        self.assertAlmostEqual(psnr(prediction, target), 20.0, places=5)

    def test_enl_reuses_coordinates(self):
        noisy = np.ones((64, 64), dtype=np.float32) * 0.5
        noisy[32:, 32:] += np.random.default_rng(42).normal(0, 0.1, (32, 32))
        rois = select_homogeneous_rois(noisy, roi_size=16, num_rois=2)
        prediction = noisy * 0.9
        self.assertEqual(len(rois), 2)
        self.assertGreater(enl(noisy, rois), 0)
        self.assertGreater(enl(prediction, rois), 0)


class AblationTests(unittest.TestCase):
    def test_each_preset_changes_exactly_one_factor(self):
        full = resolve_ablation_preset("full")
        self.assertTrue(all(full.values()))
        for name in ABLATION_PRESETS:
            preset = resolve_ablation_preset(name)
            if name == "full":
                continue
            changed = [key for key in full if preset[key] != full[key]]
            self.assertEqual(changed, [name.removeprefix("wout_")])


if __name__ == "__main__":
    unittest.main()
