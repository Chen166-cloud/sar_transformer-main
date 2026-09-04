from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.io import savemat

from evaluate_synthetic_reproducible import summarize_rows_by_l
from generate_nwpu_sar_global_l_dataset import (
    LOOKS,
    balanced_look_assignment,
    synthesize_noisy_global_l,
)
from numeric_domain import INTENSITY_DOMAIN
from utils import BSD_SAR
from synthetic_manifest import verify_paired_sar_manifest


class GlobalLProtocolTests(unittest.TestCase):
    def test_exact_source_content_leak_is_rejected(self):
        manifest = {
            "files": {"train": ["train/a.mat"], "val": ["val/b.mat"]},
            "global_L_by_file": {"train/a.mat": 1, "val/b.mat": 1},
            "source_files": {"train": ["a.jpg"], "val": ["b.jpg"]},
            "source_sha256": {"a.jpg": "same-content", "b.jpg": "same-content"},
        }
        with self.assertRaisesRegex(AssertionError, "content leaks"):
            verify_paired_sar_manifest(
                ".", manifest, verify_files=False, require_exact_partition=False
            )

    def test_validation_loader_preserves_float32_intensity(self):
        clean = np.linspace(0.0, 1.0, 64 * 64, dtype=np.float32).reshape(64, 64)
        noisy = np.clip(clean * np.float32(0.731), 0.0, 1.0)
        with tempfile.TemporaryDirectory() as temporary:
            savemat(Path(temporary) / "sample.mat", {"clean": clean, "noisy": noisy})
            noisy_tensor, clean_tensor, _ = BSD_SAR(
                temporary, (64, 64), False, INTENSITY_DOMAIN
            )[0]
            np.testing.assert_array_equal(noisy_tensor.numpy()[0], noisy)
            np.testing.assert_array_equal(clean_tensor.numpy()[0], clean)

    def test_balanced_assignment_is_deterministic_and_balanced(self):
        paths = [Path(f"image_{index}.jpg") for index in range(560)]
        first = balanced_look_assignment(paths, "airplane", 42)
        second = balanced_look_assignment(paths, "airplane", 42)
        self.assertEqual(first, second)
        self.assertEqual(
            {looks: list(first.values()).count(looks) for looks in LOOKS},
            {1: 140, 2: 140, 4: 140, 8: 140},
        )

    def test_global_l_synthesis_is_deterministic_and_bounded(self):
        clean = np.linspace(0.0, 1.0, 256 * 256, dtype=np.float32).reshape(256, 256)
        first = synthesize_noisy_global_l(clean, np.random.default_rng(123), 4)
        second = synthesize_noisy_global_l(clean, np.random.default_rng(123), 4)
        self.assertEqual(first.dtype, np.float32)
        np.testing.assert_array_equal(first, second)
        self.assertGreaterEqual(float(first.min()), 0.0)
        self.assertLessEqual(float(first.max()), 1.0)

    def test_per_l_summary_and_macro_average(self):
        rows = []
        for looks, value in zip(LOOKS, (10.0, 20.0, 30.0, 40.0)):
            rows.append(
                {
                    "file": f"L{looks}.mat",
                    "global_L": looks,
                    "noisy_psnr": value,
                    "noisy_ssim": value / 100.0,
                    "prediction_psnr": value + 1.0,
                    "prediction_ssim": value / 100.0 + 0.01,
                }
            )
        by_l, macro = summarize_rows_by_l(rows)
        self.assertEqual(list(by_l), ["L1", "L2", "L4", "L8"])
        self.assertEqual(macro["noisy_psnr_mean"], 25.0)
        self.assertEqual(macro["prediction_psnr_mean"], 26.0)


if __name__ == "__main__":
    unittest.main()
