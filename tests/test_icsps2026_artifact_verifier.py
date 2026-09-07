from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import torch

from experiment_protocol import atomic_json_dump, sha256_file
from finalize_icsps2026_sarbm3d_raw import (
    SARBM3D_DOMAIN_CONVERSION,
    SARBM3D_INPUT_DOMAIN,
    SARBM3D_OUTPUT_DOMAIN,
)
from verify_icsps2026_artifact import verify


def _pretest_credential() -> dict[str, object]:
    return {
        "ok": True,
        "protocol_id": "ICSPS26-FROZEN-v2",
        "real_use_gt16": False,
        "real_evaluation_scope": "no_reference_only",
        "real_valid_quasi_reference": 0,
        "strongest_non_ours_baseline": "transsar_v2",
        "registered_ucm_rows": 2,
        "registered_real_rows": 2,
        "test_unlock_utc": "2026-01-01T00:00:00Z",
        "pretest_seal": "/frozen/run/records/pretest_seal.json",
        **{
            field: "a" * 64
            for field in (
                "registration_sha256", "pretest_seal_sha256",
                "validation_selection_sha256", "figure_selection_sha256",
                "ucm_test_manifest_sha256", "real_qc_manifest_sha256",
                "real_qc_csv_sha256",
            )
        },
    }


class AtomicJsonTests(unittest.TestCase):
    def test_atomic_json_is_complete_and_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "marker.json"
            atomic_json_dump({"ok": True}, path)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"ok": True})
            self.assertFalse(any(path.parent.glob(".marker.json.*.tmp")))
            with self.assertRaises(FileExistsError):
                atomic_json_dump({"ok": False}, path)


class ArtifactVerifierTests(unittest.TestCase):
    def test_truncated_json_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            completion = Path(temporary) / "completion.json"
            completion.write_text('{"protocol_id":', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unreadable JSON artifact"):
                verify("supervised", completion)

    def test_supervised_and_ams_completions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            supervised_checkpoint = {
                "checkpoint_format": "icsps26-resumable-v1",
                "run_config": {
                    "method": "ours", "variant": "full", "seed": 42,
                },
                "model_metadata": {"method": "ours", "variant": "full"},
            }
            torch.save(supervised_checkpoint, root / "checkpoint_best.pth")
            torch.save(supervised_checkpoint, root / "checkpoint_last.pth")
            supervised = {
                "protocol_id": "ICSPS26-FROZEN-v2",
                "formal_run": True,
                "completed_updates": 100_000,
                "target_updates": 100_000,
                "best": {"global_step": 90_000},
                "training_synthesis_by_L": {
                    str(looks): {
                        "updates": 25_000,
                        "preclip_max_min": 1.0,
                        "preclip_max_max": 4.0,
                        "postclip_max_min": 1.0,
                        "postclip_max_max": 1.0,
                        "saturation_rate_mean": 0.01,
                        "saturation_rate_max": 0.1,
                    }
                    for looks in (1, 2, 4, 8)
                },
                "checkpoint_best_sha256": sha256_file(root / "checkpoint_best.pth"),
                "checkpoint_last_sha256": sha256_file(root / "checkpoint_last.pth"),
            }
            atomic_json_dump(supervised, root / "completion.json")
            self.assertEqual(
                verify(
                    "supervised", root / "completion.json",
                    expected_method="ours", expected_variant="full",
                    expected_seed=42, expected_adaptation="",
                )["updates"],
                100_000,
            )
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                verify(
                    "supervised", root / "completion.json",
                    expected_method="transsar_v2",
                )

            ams_root = root / "ams"
            ams_root.mkdir()
            selected = {"epoch": 4, "fixed_val_masked_loss": 0.01}
            ams_checkpoint = {
                "checkpoint_format": "icsps26-ams-v1",
                "run_config": {
                    "method": "ours", "variant": "full", "seed": 42,
                    "adaptation": "ams",
                    "selection": (
                        "lowest fixed-mask real-validation loss over adapted epochs 1-8"
                    ),
                },
                "model_metadata": {"method": "ours", "variant": "full"},
                "best": selected,
            }
            torch.save(
                {**ams_checkpoint, "epoch": selected["epoch"]},
                ams_root / "checkpoint_best.pth",
            )
            torch.save({**ams_checkpoint, "epoch": 8}, ams_root / "checkpoint_last.pth")
            ams = {
                "protocol_id": "ICSPS26-FROZEN-v2",
                "formal_run": True,
                "completed_epochs": 8,
                "best": selected,
                "checkpoint_best_sha256": sha256_file(ams_root / "checkpoint_best.pth"),
                "checkpoint_last_sha256": sha256_file(ams_root / "checkpoint_last.pth"),
            }
            atomic_json_dump(ams, ams_root / "completion.json")
            self.assertEqual(
                verify(
                    "ams", ams_root / "completion.json",
                    expected_method="ours", expected_variant="full",
                    expected_seed="42", expected_adaptation="ams",
                )["best_epoch"],
                4,
            )
            (ams_root / "checkpoint_best.pth").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                verify("ams", ams_root / "completion.json")

    def test_ucm_aggregate_and_per_image_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            per_image = root / "per_image.csv"
            with per_image.open("w", newline="", encoding="utf-8") as handle:
                fields = [
                    "pair_id", "source_id", "protocol_id", "dataset", "formal_run",
                    "method", "variant", "seed", "adaptation",
                    "class_name", "L", "psnr", "ssim",
                ]
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for index in range(8_400):
                    writer.writerow(
                        {
                            "pair_id": f"pair_{index:04d}",
                            "source_id": f"source_{index // 4:04d}",
                            "protocol_id": "ICSPS26-FROZEN-v2",
                            "dataset": "ucm_test",
                            "formal_run": True,
                            "method": "ours",
                            "variant": "full",
                            "seed": 42,
                            "adaptation": "",
                            "class_name": f"class_{(index // 4) // 100:02d}",
                            "L": (1, 2, 4, 8)[index % 4],
                            "psnr": 30.0,
                            "ssim": 0.9,
                        }
                    )
            aggregate = {
                "protocol_id": "ICSPS26-FROZEN-v2",
                "formal_run": True,
                "dataset": "ucm_test",
                "pairs": 8_400,
                "sources": 2_100,
                "method": "ours",
                "variant": "full",
                "seed": 42,
                "adaptation": None,
                "per_image_sha256": sha256_file(per_image),
                "manifest_sha256": "a" * 64,
                "pretest_gate": _pretest_credential(),
            }
            atomic_json_dump(aggregate, root / "aggregate.json")
            self.assertEqual(
                verify(
                    "ucm", root / "aggregate.json", expected_method="ours",
                    expected_variant="full", expected_seed=42, expected_adaptation="",
                )["pairs"],
                8_400,
            )
            with per_image.open("r", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

            legacy_fields = [*fields, "rpsd_error"]
            for row in rows:
                row["rpsd_error"] = "0.5"
            with per_image.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=legacy_fields)
                writer.writeheader()
                writer.writerows(rows)
            aggregate["per_image_sha256"] = sha256_file(per_image)
            (root / "aggregate.json").write_text(json.dumps(aggregate), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "retired custom RPSD"):
                verify("ucm", root / "aggregate.json")

            for row in rows:
                row.pop("rpsd_error")
            rows[0]["L"] = "2"
            with per_image.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            aggregate["per_image_sha256"] = sha256_file(per_image)
            (root / "aggregate.json").write_text(json.dumps(aggregate), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exactly one row per L"):
                verify("ucm", root / "aggregate.json")
            per_image.write_text("broken\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                verify("ucm", root / "aggregate.json")

    def test_lee_aggregate_uses_ucm_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            per_image = root / "per_image.csv"
            fields = [
                "pair_id", "source_id", "protocol_id", "dataset", "formal_run",
                "method", "variant", "seed", "adaptation",
                "has_valid_gt16",
                "class_name", "L", "psnr", "ssim",
            ]
            with per_image.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for index in range(8_400):
                    writer.writerow(
                        {
                            "pair_id": f"pair_{index:04d}",
                            "source_id": f"source_{index // 4:04d}",
                            "protocol_id": "ICSPS26-FROZEN-v2",
                            "dataset": "ucm_test",
                            "formal_run": True,
                            "method": "lee_mmse",
                            "variant": "window_7",
                            "seed": "",
                            "adaptation": "",
                            "class_name": f"class_{(index // 4) // 100:02d}",
                            "L": (1, 2, 4, 8)[index % 4],
                            "psnr": 30.0,
                            "ssim": 0.9,
                        }
                    )
            lee_aggregate = {
                "protocol_id": "ICSPS26-FROZEN-v2",
                "formal_run": True,
                "dataset": "ucm_test",
                "method": "lee_mmse",
                "variant": "window_7",
                "seed": None,
                "adaptation": None,
                "pairs": 8_400,
                "sources": 2_100,
                "per_image_sha256": sha256_file(per_image),
                "ucm_test_manifest_sha256": "a" * 64,
                "pretest_gate": _pretest_credential(),
            }
            atomic_json_dump(lee_aggregate, root / "aggregate.json")
            self.assertEqual(
                verify(
                    "ucm", root / "aggregate.json", expected_method="lee_mmse",
                    expected_seed="", expected_adaptation="",
                )["pairs"],
                8_400,
            )

    def test_real_aggregate_and_per_patch_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            per_patch = root / "real_per_patch.csv"
            fields = [
                "sample_id", "parent_id", "protocol_id", "split", "formal_run",
                "method", "variant", "seed", "adaptation",
                "has_valid_gt16",
                "ratio_mean_bias", "ratio_acf_sidelobe_energy", "sobel_gc_noisy_output",
                "qpsnr_gt16", "qssim_gt16",
            ]
            with per_patch.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for index in range(592):
                    writer.writerow(
                        {
                            "sample_id": f"sample_{index:03d}",
                            "parent_id": f"parent_{index // 4:03d}",
                            "protocol_id": "ICSPS26-FROZEN-v2",
                            "split": "test",
                            "formal_run": True,
                            "method": "ours",
                            "variant": "full",
                            "seed": 42,
                            "adaptation": "base",
                            "has_valid_gt16": False,
                            "ratio_mean_bias": 0.1,
                            "ratio_acf_sidelobe_energy": 0.2,
                            "sobel_gc_noisy_output": 0.8,
                            "qpsnr_gt16": "nan",
                            "qssim_gt16": "nan",
                        }
                    )
            aggregate = {
                "protocol_id": "ICSPS26-FROZEN-v2",
                "formal_run": True,
                "split": "test",
                "num_patches": 592,
                "num_parent_clusters": 148,
                "num_valid_gt16_pairs": 0,
                "method": "ours",
                "variant": "full",
                "seed": 42,
                "adaptation": "base",
                "qc_manifest_sha256": "a" * 64,
                "qc_csv_sha256": "a" * 64,
                "parent_cluster_bootstrap": {},
                "artifacts": {
                    "real_per_patch_csv": per_patch.name,
                    "real_per_patch_sha256": sha256_file(per_patch),
                },
                "pretest_gate": _pretest_credential(),
            }
            atomic_json_dump(aggregate, root / "aggregate.json")
            result = verify(
                "real", root / "aggregate.json", expected_method="ours",
                expected_variant="full", expected_seed=42, expected_adaptation="base",
            )
            self.assertEqual((result["patches"], result["parents"]), (592, 148))
            with per_patch.open("r", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

            rows[0]["qpsnr_gt16"] = "30.0"
            with per_patch.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            aggregate["artifacts"]["real_per_patch_sha256"] = sha256_file(per_patch)
            (root / "aggregate.json").write_text(json.dumps(aggregate), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "q-metric values"):
                verify("real", root / "aggregate.json")

            rows[0]["qpsnr_gt16"] = "nan"
            with per_patch.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            aggregate["artifacts"]["real_per_patch_sha256"] = sha256_file(per_patch)
            aggregate["num_valid_gt16_pairs"] = 1
            (root / "aggregate.json").write_text(json.dumps(aggregate), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "nonzero GT16-valid pairs"):
                verify("real", root / "aggregate.json")

            aggregate["num_valid_gt16_pairs"] = 0
            aggregate["parent_cluster_bootstrap"]["qssim_gt16"] = {}
            (root / "aggregate.json").write_text(json.dumps(aggregate), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "q-reference summaries"):
                verify("real", root / "aggregate.json")

            aggregate["parent_cluster_bootstrap"] = {}
            rows[0]["has_valid_gt16"] = "True"
            with per_patch.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            aggregate["artifacts"]["real_per_patch_sha256"] = sha256_file(per_patch)
            (root / "aggregate.json").write_text(json.dumps(aggregate), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no-GT16"):
                verify("real", root / "aggregate.json")

            rows[0]["has_valid_gt16"] = "False"
            rows[0]["sobel_gc_noisy_output"] = "nan"
            with per_patch.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            aggregate["artifacts"]["real_per_patch_sha256"] = sha256_file(per_patch)
            (root / "aggregate.json").write_text(
                json.dumps(aggregate), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "non-finite core metrics"):
                verify("real", root / "aggregate.json")

    def test_formal_sarbm_raw_chain(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            predictions = root / "predictions"
            predictions.mkdir()
            manifest = root / "prediction_manifest.csv"
            fields = [
                "protocol_id", "formal_run", "method", "variant", "pair_id", "L",
                "input_mat_path", "input_mat_sha256", "input_noisy_field",
                "prediction_path", "prediction_sha256", "height", "width",
                "inference_seconds", "raw_min", "raw_max",
                "prediction_amplitude_raw_min", "prediction_amplitude_raw_max",
                "input_domain", "output_domain", "domain_conversion", "job_index",
            ]
            prediction_hash = None
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for index in range(8_400):
                    pair_id = f"{index:024x}"
                    prediction = predictions / f"{pair_id}.mat"
                    prediction.write_bytes(b"validated-by-finalizer")
                    prediction_hash = prediction_hash or sha256_file(prediction)
                    writer.writerow(
                        {
                            "protocol_id": "ICSPS26-FROZEN-v2",
                            "formal_run": True,
                            "method": "sar_bm3d",
                            "variant": "v1.0",
                            "pair_id": pair_id,
                            "L": (1, 2, 4, 8)[index % 4],
                            "input_mat_path": f"fixed/{pair_id}.mat",
                            "input_mat_sha256": "a" * 64,
                            "input_noisy_field": "noisy",
                            "prediction_path": f"predictions/{pair_id}.mat",
                            "prediction_sha256": prediction_hash,
                            "height": 256,
                            "width": 256,
                            "inference_seconds": 0.25,
                            "raw_min": 0.0,
                            "raw_max": 1.0,
                            "prediction_amplitude_raw_min": 0.0,
                            "prediction_amplitude_raw_max": 1.0,
                            "input_domain": SARBM3D_INPUT_DOMAIN,
                            "output_domain": SARBM3D_OUTPUT_DOMAIN,
                            "domain_conversion": SARBM3D_DOMAIN_CONVERSION,
                            "job_index": index + 1,
                        }
                    )
            completion = {
                "protocol_id": "ICSPS26-FROZEN-v2",
                "formal_run": True,
                "method": "sar_bm3d",
                "variant": "v1.0",
                "seed": None,
                "adaptation": None,
                "completed_jobs": 8_400,
                "expected_jobs": 8_400,
                "sarbm3d_input_domain": SARBM3D_INPUT_DOMAIN,
                "scored_output_domain": SARBM3D_OUTPUT_DOMAIN,
                "domain_conversion": SARBM3D_DOMAIN_CONVERSION,
                "prediction_manifest": manifest.name,
                "prediction_manifest_sha256": sha256_file(manifest),
                "pair_manifest_sha256": "a" * 64,
                "pretest_gate": _pretest_credential(),
            }
            atomic_json_dump(completion, root / "completion.json")
            result = verify(
                "sarbm_raw", root / "completion.json",
                expected_method="sar_bm3d", expected_variant="v1.0",
                expected_seed="", expected_adaptation="",
            )
            self.assertEqual(result["predictions"], 8_400)


if __name__ == "__main__":
    unittest.main()
