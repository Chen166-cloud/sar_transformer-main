import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from scipy.io import savemat

import icsps2026_metrics as metrics
from evaluate_icsps2026_real import (
    PER_IMAGE_METRICS,
    _default_method_label,
    _validate_adaptation,
    main as evaluate_real_main,
)
from prepare_icsps2026_real import (
    alignment_audit,
    apply_fixed_mapping,
    canonical_pair_key,
    choose_dataset_mapping,
    parent_id_from_sample_name,
    main as prepare_main,
)


class RatioMetricTests(unittest.TestCase):
    def test_known_ratio_mean_bias(self):
        noisy = np.full((32, 32), 0.8, dtype=np.float32)
        prediction = np.full((32, 32), 0.4, dtype=np.float32)
        self.assertAlmostEqual(metrics.ratio_mean_bias(noisy, prediction), 1.0)

    def test_constant_ratio_has_zero_acf_sidelobe_energy(self):
        ratio = np.ones((32, 32), dtype=np.float32)
        self.assertEqual(metrics.ratio_acf_sidelobe_energy_from_ratio(ratio), 0.0)

    def test_structured_ratio_has_more_acf_than_white_noise(self):
        rng = np.random.default_rng(3)
        white = rng.normal(size=(64, 64))
        yy, xx = np.indices((64, 64))
        checkerboard = ((xx + yy) % 2).astype(np.float64)
        white_score = metrics.ratio_acf_sidelobe_energy_from_ratio(white)
        structured_score = metrics.ratio_acf_sidelobe_energy_from_ratio(checkerboard)
        self.assertGreater(structured_score, white_score * 10.0)

    def test_denominator_floor_fraction_is_reported(self):
        noisy = np.ones((2, 2), dtype=np.float32)
        prediction = np.asarray([[0.0, 0.5], [0.5, 0.0]], dtype=np.float32)
        _, fraction = metrics.intensity_ratio(noisy, prediction)
        self.assertEqual(fraction, 0.5)


class GradientAndENLTests(unittest.TestCase):
    def test_sobel_gc_is_affine_scale_invariant(self):
        image = np.random.default_rng(4).random((64, 64), dtype=np.float32)
        transformed = 0.2 * image + 0.3
        self.assertAlmostEqual(metrics.sobel_gradient_correlation(image, transformed), 1.0, places=12)

    def test_sobel_gc_is_not_exposed_as_rgpi(self):
        self.assertFalse(hasattr(metrics, "rgpi"))
        self.assertFalse(hasattr(metrics, "m_index"))
        self.assertIn("sobel_gc_noisy_output", PER_IMAGE_METRICS)
        self.assertNotIn("rgpi", PER_IMAGE_METRICS)

    def test_enl_roi_details_are_reconstructable(self):
        image = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        details = metrics.enl_roi_details(image, [(0, 0, 2, 2)])
        self.assertEqual(details[0]["roi_id"], "roi01")
        self.assertAlmostEqual(float(details[0]["mean"]), 2.5)
        self.assertAlmostEqual(float(details[0]["variance"]), 1.25)
        self.assertAlmostEqual(float(details[0]["enl"]), 5.0)

    def test_roi_selection_is_deterministic_and_noisy_only(self):
        image = np.random.default_rng(5).normal(0.5, 0.1, (64, 64)).astype(np.float32)
        image[:16, :16] = 0.5
        first = metrics.select_homogeneous_rois(image, roi_size=16, num_rois=2)
        second = metrics.select_homogeneous_rois(image, roi_size=16, num_rois=2)
        self.assertEqual(first, second)
        self.assertEqual((first[0]["x"], first[0]["y"]), (0, 0))


class ClusterBootstrapTests(unittest.TestCase):
    def test_parent_clusters_not_child_patches_are_resampled(self):
        rows = [
            {"parent_id": "a", "score": 1.0},
            {"parent_id": "a", "score": 3.0},
            {"parent_id": "b", "score": 10.0},
        ]
        result = metrics.parent_cluster_bootstrap(rows, "score", samples=200, seed=9)
        self.assertEqual(result["num_patches"], 3)
        self.assertEqual(result["num_parent_clusters"], 2)
        self.assertAlmostEqual(float(result["patch_mean"]), 14.0 / 3.0)
        self.assertAlmostEqual(float(result["parent_macro_mean"]), 6.0)
        repeated = metrics.parent_cluster_bootstrap(rows, "score", samples=200, seed=9)
        self.assertEqual(result, repeated)


class RealEvaluatorIdentityTests(unittest.TestCase):
    def test_default_labels_come_from_verified_identity(self):
        ours = {"method": "ours", "variant": "full"}
        ablation = {"method": "ours", "variant": "wout_fdr"}
        self.assertEqual(_default_method_label(ours, "base"), "DFNG-SARNet")
        self.assertEqual(_default_method_label(ours, "ams"), "DFNG-SARNet+AMS")
        self.assertEqual(
            _default_method_label(ablation, "base"),
            "DFNG-SARNet (wout_fdr)",
        )
        self.assertEqual(
            _default_method_label({"method": "transsar_v2", "variant": None}, "base"),
            "TransSARV2",
        )
        self.assertEqual(
            _default_method_label({"method": "sar_cam", "variant": None}, "base"),
            "SAR-CAM",
        )

    def test_adaptation_cannot_relabel_checkpoint(self):
        identity = {"method": "ours", "variant": "full"}
        _validate_adaptation("ams", identity, {"run_config": {"adaptation": "ams"}})
        with self.assertRaises(ValueError):
            _validate_adaptation("ams", identity, {"run_config": {}})
        with self.assertRaises(ValueError):
            _validate_adaptation(
                "ams",
                {"method": "transsar_v2", "variant": None},
                {"run_config": {"adaptation": "ams"}},
            )
        with self.assertRaises(ValueError):
            _validate_adaptation(
                "base", identity, {"run_config": {"adaptation": "ams"}}
            )


class PreparationProtocolTests(unittest.TestCase):
    def test_unambiguous_dataset_mapping(self):
        mapping = choose_dataset_mapping((0.0, 1.0), None, "auto", 0.0, 1.0)
        self.assertEqual(mapping["mode"], "unit_float")
        mapped = apply_fixed_mapping(np.asarray([[0.0, 0.5, 1.0]]), mapping)
        np.testing.assert_allclose(mapped, [[0.0, 0.5, 1.0]])

    def test_mixed_noisy_gt_scales_are_rejected(self):
        with self.assertRaises(ValueError):
            choose_dataset_mapping((0.0, 1.0), (0.0, 255.0), "auto", 0.0, 1.0)

    def test_pair_and_parent_keys(self):
        self.assertEqual(canonical_pair_key("Noisy/12_24_y0_x0_noisy.mat"), "12_24_y0_x0")
        self.assertEqual(parent_id_from_sample_name("12_24_y0_x256.mat"), "12_24")

    def test_identical_pair_passes_alignment_primitives(self):
        image = np.random.default_rng(6).random((64, 64), dtype=np.float32)
        result = alignment_audit(image, image)
        self.assertGreater(result["alignment_score"], 0.999)
        self.assertLess(abs(result["phase_shift_x"]), 0.1)
        self.assertLess(abs(result["phase_shift_y"]), 0.1)

    def test_prepare_script_mechanically_pairs_and_writes_fixed_mapping(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "real"
            noisy_root = root / "Noisy"
            gt_root = root / "GT16"
            output = Path(temporary) / "qc"
            noisy_root.mkdir(parents=True)
            gt_root.mkdir(parents=True)
            image = np.random.default_rng(7).uniform(0.2, 0.8, (64, 64)).astype(np.float32)
            savemat(noisy_root / "0_0_y0_x0.mat", {"noisy": image})
            savemat(gt_root / "0_0_y0_x0.mat", {"gt16": image})
            arguments = [
                "prepare_icsps2026_real.py",
                "--dataset-root", str(root),
                "--output-dir", str(output),
                "--gt16-root", str(gt_root),
                "--default-split", "val",
                "--expected-height", "64",
                "--expected-width", "64",
                "--roi-size", "16",
                "--num-rois", "2",
                "--roi-stride", "16",
            ]
            with mock.patch.object(sys, "argv", arguments):
                self.assertEqual(prepare_main(), 0)
            with (output / "real_qc_manifest.json").open(encoding="utf-8") as handle:
                manifest = json.load(handle)
            self.assertEqual(manifest["numeric_mapping"]["mode"], "unit_float")
            with (output / "real_qc_manifest.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["pair_rule"], "relative_key")
            self.assertEqual(rows[0]["valid_quasi_reference"], "True")

            evaluation = Path(temporary) / "noisy_evaluation"
            evaluation_arguments = [
                "evaluate_icsps2026_real.py",
                "--dataset-root", str(root),
                "--qc-manifest", str(output / "real_qc_manifest.json"),
                "--output-dir", str(evaluation),
                "--method", "noisy",
                "--adaptation", "base",
                "--split", "val",
                "--bootstrap-samples", "20",
                "--max-images", "1",
                "--nonformal-smoke",
            ]
            with mock.patch.object(sys, "argv", evaluation_arguments):
                self.assertEqual(evaluate_real_main(), 0)
            with (evaluation / "aggregate.json").open(encoding="utf-8") as handle:
                aggregate = json.load(handle)
            self.assertEqual(aggregate["method"], "noisy")
            self.assertFalse(aggregate["formal_run"])
            self.assertEqual(
                aggregate["method_label_source"], "built_in_noisy_baseline"
            )
            with (evaluation / "real_per_patch.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                evaluated_rows = list(csv.DictReader(handle))
            self.assertNotIn("inference_ms", evaluated_rows[0])

    def test_disable_gt16_prevents_named_directory_auto_discovery(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "real"
            noisy_root = root / "Noisy"
            gt_root = root / "GT16"
            output = Path(temporary) / "qc_no_gt16"
            noisy_root.mkdir(parents=True)
            gt_root.mkdir(parents=True)
            image = np.random.default_rng(8).uniform(0.2, 0.8, (64, 64)).astype(np.float32)
            savemat(noisy_root / "0_0_y0_x0.mat", {"noisy": image})
            savemat(gt_root / "0_0_y0_x0.mat", {"gt16": image})
            arguments = [
                "prepare_icsps2026_real.py",
                "--dataset-root", str(root),
                "--output-dir", str(output),
                "--default-split", "val",
                "--expected-height", "64",
                "--expected-width", "64",
                "--roi-size", "16",
                "--num-rois", "2",
                "--roi-stride", "16",
                "--disable-gt16",
            ]
            with mock.patch.object(sys, "argv", arguments):
                self.assertEqual(prepare_main(), 0)
            with (output / "real_qc_manifest.json").open(encoding="utf-8") as handle:
                manifest = json.load(handle)
            self.assertIsNone(manifest["gt16_root"])
            self.assertIs(manifest["gt16_disabled_by_protocol"], True)
            self.assertEqual(manifest["counts"]["gt16_files_discovered"], 0)
            with (output / "real_qc_manifest.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["gt16_path"], "")
            self.assertEqual(rows[0]["pair_exists"], "False")
            self.assertEqual(rows[0]["valid_quasi_reference"], "False")


if __name__ == "__main__":
    unittest.main()
