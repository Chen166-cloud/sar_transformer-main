import gc
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch import nn

from ablation_config import (
    ABLATION_PRESETS,
    FORMAL_VARIANTS,
    legacy_preset_to_variant,
    resolve_ablation_preset,
    resolve_formal_variant,
)
from model_registry import (
    SAR_CAM_CONSTRUCTOR_KWARGS,
    SAR_CAM_PINNED_COMMIT,
    build_model,
    build_model_from_checkpoint_metadata,
    extract_checkpoint_model_metadata,
    validate_checkpoint_model_metadata,
    verify_sar_cam_checkout,
)
from transform_main import (
    BottleneckRefine,
    FFTRefineBlock,
    TransSARV2,
    TransSARV2_DualFreqNG_Bottle,
)


class AblationConfigurationTests(unittest.TestCase):
    def test_legacy_presets_are_unchanged(self):
        self.assertEqual(
            set(ABLATION_PRESETS),
            {
                "full",
                "wout_frequency",
                "wout_gate",
                "wout_fusion",
                "wout_msf",
                "wout_bottleneck",
            },
        )
        full = resolve_ablation_preset("full")
        self.assertTrue(all(full.values()))
        for name in ABLATION_PRESETS:
            resolved = resolve_ablation_preset(name)
            if name != "full":
                changed = [key for key in full if full[key] != resolved[key]]
                self.assertEqual(changed, [name.removeprefix("wout_")])

    def test_formal_variants_have_the_required_semantics(self):
        self.assertEqual(
            set(FORMAL_VARIANTS),
            {"full", "intensity_only", "log_only", "wout_all_fdr"},
        )
        intensity = resolve_formal_variant("intensity_only")
        self.assertEqual(intensity["representation"], "intensity")
        self.assertFalse(intensity["compensation"])
        log_only = resolve_formal_variant("log_only")
        self.assertEqual(log_only["representation"], "log")
        self.assertFalse(log_only["compensation"])
        no_fdr = resolve_formal_variant("wout_all_fdr")
        self.assertFalse(no_fdr["decoder_fdr"])
        self.assertFalse(no_fdr["bottleneck_fdr"])
        self.assertTrue(no_fdr["bottleneck_local"])

    def test_legacy_frequency_ablation_does_not_change_bottleneck(self):
        translated = legacy_preset_to_variant("wout_frequency")
        self.assertFalse(translated["decoder_fdr"])
        self.assertTrue(translated["bottleneck_fdr"])


class BottleneckTests(unittest.TestCase):
    def test_frequency_can_be_removed_without_removing_local_branch(self):
        block = BottleneckRefine(channels=8, use_frequency=False)
        self.assertIsNone(block.freq)
        self.assertIsInstance(block.local, nn.Sequential)
        self.assertIsInstance(block.fuse, nn.Sequential)
        sample = torch.rand(1, 8, 8, 8)
        result = block(sample)
        self.assertEqual(result.shape, sample.shape)
        self.assertTrue(torch.isfinite(result).all())


class FormalModelTests(unittest.TestCase):
    def tearDown(self):
        gc.collect()

    def test_full_and_log_only_model_structure(self):
        full = TransSARV2_DualFreqNG_Bottle(variant="full")
        self.assertEqual(full.representation, "log")
        self.assertIsNotNone(full.dual_fusion)
        self.assertIsInstance(full.log_branch.bottleneck_refine.freq, FFTRefineBlock)
        del full
        gc.collect()

        log_only = TransSARV2_DualFreqNG_Bottle(variant="log_only")
        self.assertEqual(log_only.representation, "log")
        self.assertIsNone(log_only.dual_fusion)

    def test_wout_all_fdr_keeps_local_bottleneck(self):
        model = TransSARV2_DualFreqNG_Bottle(variant="wout_all_fdr")
        bottleneck = model.log_branch.bottleneck_refine
        self.assertIsInstance(bottleneck, BottleneckRefine)
        self.assertIsNone(bottleneck.freq)
        self.assertIsInstance(bottleneck.local, nn.Sequential)
        decoder = model.log_branch.convproj
        for name in ("freq4", "freq3", "freq2", "freq1", "freq0"):
            self.assertIsInstance(getattr(decoder, name), nn.Identity)

    def test_historical_wout_frequency_retains_bottleneck_fdr(self):
        model = TransSARV2_DualFreqNG_Bottle(ablation="wout_frequency")
        self.assertEqual(
            model.ablation_config,
            resolve_ablation_preset("wout_frequency"),
        )
        self.assertIsInstance(model.log_branch.bottleneck_refine.freq, FFTRefineBlock)
        for name in ("freq4", "freq3", "freq2", "freq1", "freq0"):
            self.assertIsInstance(getattr(model.log_branch.convproj, name), nn.Identity)

    def test_intensity_only_bypasses_log_and_compensation(self):
        model = TransSARV2_DualFreqNG_Bottle(variant="intensity_only")
        self.assertEqual(model.representation, "intensity")
        self.assertIsNone(model.dual_fusion)
        # Replace the large restoration branch so this test isolates the outer
        # representation chain instead of performing a full CPU inference.
        model.log_branch = nn.Identity()
        sample = torch.rand(1, 1, 8, 8)
        with mock.patch(
            "transform_main.log_transform_01",
            side_effect=AssertionError("intensity-only called log transform"),
        ), mock.patch(
            "transform_main.inverse_log_transform_01",
            side_effect=AssertionError("intensity-only called inverse log transform"),
        ):
            output = model(sample)
        torch.testing.assert_close(output, sample)


class ModelRegistryTests(unittest.TestCase):
    def tearDown(self):
        gc.collect()

    def test_registry_builds_true_transsar_v2(self):
        model = build_model("transsar_v2")
        self.assertIsInstance(model, TransSARV2)
        self.assertNotIsInstance(model, TransSARV2_DualFreqNG_Bottle)
        self.assertEqual(model.registry_metadata["method"], "transsar_v2")

    def test_registry_builds_formal_ours_variant(self):
        model = build_model("ours", "wout_all_fdr")
        self.assertIsInstance(model, TransSARV2_DualFreqNG_Bottle)
        self.assertEqual(model.registry_metadata["variant"], "wout_all_fdr")
        self.assertIsNone(model.log_branch.bottleneck_refine.freq)

    def test_checkpoint_method_and_variant_are_strictly_checked(self):
        checkpoint = {
            "model_metadata": {
                "method": "ours",
                "class_name": "TransSARV2_DualFreqNG_Bottle",
                "variant": "log_only",
            }
        }
        validated = validate_checkpoint_model_metadata(
            checkpoint, "ours", "log_only"
        )
        self.assertEqual(validated["method"], "ours")
        with self.assertRaisesRegex(ValueError, "method mismatch"):
            validate_checkpoint_model_metadata(
                checkpoint, "transsar_v2"
            )
        with self.assertRaisesRegex(ValueError, "variant mismatch"):
            validate_checkpoint_model_metadata(checkpoint, "ours", "full")

    def test_historical_checkpoint_metadata_is_understood(self):
        checkpoint = {
            "run_config": {
                "model": "TransSARV2_DualFreqNG_Bottle",
                "ablation": "wout_fusion",
            }
        }
        metadata = extract_checkpoint_model_metadata(checkpoint)
        self.assertEqual(metadata["method"], "ours")
        self.assertEqual(metadata["variant"], "log_only")
        model = build_model_from_checkpoint_metadata(checkpoint)
        self.assertIsNone(model.dual_fusion)

    def test_raw_state_dict_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "no model identity"):
            extract_checkpoint_model_metadata({"weight": torch.ones(1)})

    def test_sar_cam_checkpoint_commit_is_checked(self):
        checkpoint = {
            "model_metadata": {
                "method": "sar_cam",
                "class_name": "SAR_CAM",
                "variant": None,
                "external_commit": "not-the-pinned-commit",
                "constructor_kwargs": SAR_CAM_CONSTRUCTOR_KWARGS,
            }
        }
        with self.assertRaisesRegex(ValueError, "commit mismatch"):
            validate_checkpoint_model_metadata(checkpoint, "sar_cam")

    def test_sar_cam_requires_a_git_checkout(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(FileNotFoundError, "not a Git checkout"):
                verify_sar_cam_checkout(temporary)

    def test_sar_cam_factory_is_loaded_from_external_checkout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "model_parts.py").write_text("# test stub\n", encoding="utf-8")
            (root / "model.py").write_text(
                "import torch\n"
                "def Model(scale, in_channels, channels, kernel_size, stride, dilation, bias):\n"
                "    assert (scale, in_channels, channels, kernel_size, stride, dilation, bias) == (2, 1, 128, 3, 1, 1, True)\n"
                "    return torch.nn.Conv2d(1, 1, 1)\n",
                encoding="utf-8",
            )
            with mock.patch(
                "model_registry.verify_sar_cam_checkout", return_value=root
            ):
                model = build_model("sar_cam", external_root=root)
        self.assertIsInstance(model, nn.Conv2d)
        self.assertEqual(model.registry_metadata["external_commit"], SAR_CAM_PINNED_COMMIT)


if __name__ == "__main__":
    unittest.main()
