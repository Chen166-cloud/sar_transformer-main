"""Offline adapter-contract tests; no checkpoints, downloads, or GPU required.

Run with a Python containing NumPy, SciPy, Pillow and PyTorch. These tests check
numeric-domain handling, not despeckling quality or the paper benchmarks.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import savemat


ROOT = Path(__file__).resolve().parents[1]


def load_adapter(name: str):
    path = ROOT / "scripts" / name / "run_local.py"
    spec = importlib.util.spec_from_file_location(f"_test_adapter_{name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load adapter {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class LocalBaselineInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapters = {name: load_adapter(name) for name in ("cl_sar", "mulog_drunet")}

    def test_mat_and_npy_preserve_numeric_values(self):
        values = np.tile(np.array([[0.0, 0.25], [1.5, 9.0]], dtype=np.float32), (4, 4))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            savemat(root / "sample.mat", {"noisy": values})
            np.save(root / "sample.npy", values, allow_pickle=False)
            for name, adapter in self.adapters.items():
                for filename in ("sample.mat", "sample.npy"):
                    with self.subTest(adapter=name, file=filename):
                        actual, _ = adapter.load_array(root / filename, "noisy")
                        np.testing.assert_array_equal(actual, values)

    def test_missing_mat_field_is_not_silently_selected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wrong.mat"
            savemat(path, {"unknown": np.ones((4, 4))})
            for name, adapter in self.adapters.items():
                with self.subTest(adapter=name), self.assertRaises((ValueError, KeyError)):
                    adapter.load_array(path, "noisy")

    def test_invalid_arrays_are_rejected(self):
        invalid = {
            "complex": np.ones((8, 8), dtype=np.complex64),
            "color": np.ones((8, 8, 3)),
            "negative": np.tile(np.array([[0.0, 1.0], [-0.1, 2.0]]), (4, 4)),
            "nan": np.tile(np.array([[0.0, 1.0], [np.nan, 2.0]]), (4, 4)),
            "inf": np.tile(np.array([[0.0, 1.0], [np.inf, 2.0]]), (4, 4)),
            "empty": np.empty((0, 2)),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.npy"
            for case, values in invalid.items():
                np.save(path, values, allow_pickle=False)
                for name, adapter in self.adapters.items():
                    with self.subTest(adapter=name, case=case), self.assertRaises(ValueError):
                        adapter.load_array(path, "noisy")

    def test_color_raster_is_not_silently_converted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "color.png"
            Image.new("RGB", (4, 4), color=(25, 50, 75)).save(path)
            for name, adapter in self.adapters.items():
                with self.subTest(adapter=name), self.assertRaises(ValueError):
                    adapter.load_array(path, "noisy")

    def test_grayscale_integer_units_are_preserved(self):
        values = np.tile(np.array([[0, 32], [128, 255]], dtype=np.uint8), (4, 4))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gray.png"
            Image.fromarray(values).save(path)
            for name, adapter in self.adapters.items():
                with self.subTest(adapter=name):
                    actual, _ = adapter.load_array(path, "noisy")
                    np.testing.assert_array_equal(actual, values.astype(np.float32))


class CLSARNumericDomainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = load_adapter("cl_sar")

    def test_official_nonzero_minimum_and_absolute_value(self):
        intensity = np.array([[0.0, 1.0], [4.0, 9.0]], dtype=np.float32)
        expected = np.array([[0.5, 0.0], [0.5, 1.0]], dtype=np.float32)
        normalized, _ = self.adapter.prepare_input(intensity, "intensity")
        np.testing.assert_allclose(normalized, expected, atol=1e-7, rtol=0)

    def test_equivalent_amplitude_and_intensity_inputs(self):
        intensity = np.array([[0.0, 1.0], [4.0, 9.0]], dtype=np.float32)
        amplitude = np.sqrt(intensity)
        from_intensity, intensity_info = self.adapter.prepare_input(intensity, "intensity")
        from_amplitude, amplitude_info = self.adapter.prepare_input(amplitude, "amplitude")
        np.testing.assert_allclose(from_intensity, from_amplitude, atol=1e-7, rtol=0)
        raw = np.array([[-0.2, 0.25], [0.75, 1.2]], dtype=np.float32)
        restored_intensity = self.adapter.restore_output(raw.copy(), intensity_info)
        restored_amplitude = self.adapter.restore_output(raw.copy(), amplitude_info)
        np.testing.assert_allclose(restored_intensity, restored_amplitude ** 2, atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(restored_intensity, [[1.0, 2.25], [6.25, 9.0]], atol=1e-6)

    def test_explicit_scale_is_reversed(self):
        values = np.array([[0.0, 1.0], [4.0, 9.0]], dtype=np.float32)
        raw = np.array([[0.1, 0.25], [0.75, 0.9]], dtype=np.float32)
        for domain in ("intensity", "amplitude"):
            with self.subTest(domain=domain):
                original, original_info = self.adapter.prepare_input(values, domain, scale=1.0)
                scaled, scaled_info = self.adapter.prepare_input(values, domain, scale=9.0)
                np.testing.assert_allclose(original, scaled, atol=1e-6, rtol=1e-6)
                np.testing.assert_allclose(
                    self.adapter.restore_output(raw.copy(), original_info),
                    self.adapter.restore_output(raw.copy(), scaled_info), atol=1e-5, rtol=1e-6,
                )

    def test_degenerate_images_rejected_before_normalization(self):
        for values in (np.zeros((4, 4)), np.ones((4, 4)), np.array([[0.0, 1.0], [1.0, 0.0]])):
            with self.subTest(values=values.tolist()), self.assertRaises(ValueError):
                self.adapter.prepare_input(values, "intensity")

    def test_invalid_scales_and_domains_rejected(self):
        values = np.array([[0.0, 1.0], [4.0, 9.0]], dtype=np.float32)
        for scale in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                self.adapter.prepare_input(values, "intensity", scale=scale)
        with self.assertRaises(ValueError):
            self.adapter.prepare_input(values, "unknown")


class MuLoGNumericDomainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = load_adapter("mulog_drunet")

    def test_amplitude_is_squared_exactly_once(self):
        amplitude = np.tile(np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64), (4, 4))
        np.testing.assert_array_equal(
            self.adapter.to_intensity(amplitude, "amplitude", 1.0), amplitude ** 2,
        )
        np.testing.assert_array_equal(
            self.adapter.to_intensity(amplitude ** 2, "intensity", 1.0), amplitude ** 2,
        )

    def test_scale_and_domain_round_trip(self):
        values = np.tile(np.array([[0.0, 1.0], [4.0, 9.0]], dtype=np.float64), (4, 4))
        for domain in ("intensity", "amplitude"):
            for scale in (1.0, 9.0):
                with self.subTest(domain=domain, scale=scale):
                    intensity = self.adapter.to_intensity(values, domain, scale)
                    restored = self.adapter.from_intensity(intensity, domain, scale)
                    np.testing.assert_allclose(restored, values, atol=1e-12, rtol=1e-12)

    def test_invalid_scale_domain_and_all_zero_input(self):
        values = np.ones((8, 8), dtype=np.float64)
        for scale in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                self.adapter.to_intensity(values, "intensity", scale)
        with self.assertRaises(ValueError):
            self.adapter.to_intensity(values, "unknown", 1.0)
        with self.assertRaises(ValueError):
            self.adapter.to_intensity(np.zeros((8, 8)), "intensity", 1.0)

    def test_invalid_prediction_is_not_silently_clipped(self):
        for value in (-0.1, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.adapter.from_intensity(np.full((8, 8), value), "amplitude", 1.0)


class LocalBaselineCLITests(unittest.TestCase):
    def run_cli(self, adapter: str, *args: str):
        return subprocess.run(
            [sys.executable, str(ROOT / "scripts" / adapter / "run_local.py"), *args],
            capture_output=True, text=True, timeout=45, cwd=ROOT,
        )

    def test_help_is_available_without_loading_checkpoints(self):
        for name in ("cl_sar", "mulog_drunet"):
            with self.subTest(adapter=name):
                result = self.run_cli(name, "--help")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("--input-domain", result.stdout)

    def test_input_domain_is_mandatory(self):
        for name in ("cl_sar", "mulog_drunet"):
            with self.subTest(adapter=name):
                result = self.run_cli(name, "--input", "nonexistent.npy")
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("--input-domain", result.stderr)

    def test_mulog_looks_is_mandatory(self):
        result = self.run_cli("mulog_drunet", "--input", "nonexistent.npy", "--input-domain", "intensity")
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("--looks", result.stderr)

    def test_existing_outputs_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            marker = target / "keep.txt"
            marker.write_text("existing user result", encoding="utf-8")
            for name in ("cl_sar", "mulog_drunet"):
                extra = ["--looks", "1"] if name == "mulog_drunet" else []
                with self.subTest(adapter=name):
                    result = self.run_cli(name, "--input", "nonexistent.npy", "--input-domain",
                                          "intensity", "--output", str(target), *extra)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("empty", result.stderr.lower())
                    self.assertEqual(marker.read_text(encoding="utf-8"), "existing user result")


if __name__ == "__main__":
    unittest.main()
