from __future__ import annotations

import csv
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import torch

from experiment_protocol import sha256_file
from icsps2026_pretest import REAL_SPLIT_MANIFEST_SHA256
from icsps2026_protocol import stable_u64
from seal_icsps2026_pretest import seal
from verify_icsps2026_pretest_gate import verify


PROTOCOL_ID = "ICSPS26-FROZEN-v2"


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _synthesis_audit() -> dict[str, dict[str, float | int]]:
    return {
        str(looks): {
            "updates": 25_000,
            "preclip_max_min": 1.0,
            "preclip_max_max": 2.0,
            "postclip_max_min": 1.0,
            "postclip_max_max": 1.0,
            "saturation_rate_mean": 0.1,
            "saturation_rate_max": 0.2,
        }
        for looks in (1, 2, 4, 8)
    }


class PretestGateTests(unittest.TestCase):
    def _fixture(self, parent: Path) -> tuple[Path, Path, Path]:
        run = parent / "run"
        prep = parent / "prep"
        real = parent / "real_protocol"
        records = run / "records"
        records.mkdir(parents=True)
        prep.mkdir()
        real.mkdir()

        baseline_dir = run / "train" / "transsar_v2" / "seed42"
        baseline_dir.mkdir(parents=True)
        checkpoint = {
            "checkpoint_format": "icsps26-resumable-v1",
            "global_step": 100_000,
            "metrics": {"val_macro_mse": 0.01},
            "run_config": {
                "method": "transsar_v2", "variant": None, "seed": 42,
                "adaptation": None,
            },
            "model_metadata": {"method": "transsar_v2", "variant": None},
        }
        torch.save(checkpoint, baseline_dir / "checkpoint_best.pth")
        torch.save(checkpoint, baseline_dir / "checkpoint_last.pth")
        completion = {
            "protocol_id": PROTOCOL_ID,
            "formal_run": True,
            "completed_updates": 100_000,
            "target_updates": 100_000,
            "best": {"global_step": 100_000, "val_macro_mse": 0.01},
            "training_synthesis_by_L": _synthesis_audit(),
            "checkpoint_best_sha256": sha256_file(baseline_dir / "checkpoint_best.pth"),
            "checkpoint_last_sha256": sha256_file(baseline_dir / "checkpoint_last.pth"),
        }
        (baseline_dir / "completion.json").write_text(
            json.dumps(completion), encoding="utf-8"
        )
        _write_csv(
            baseline_dir / "step_metrics.csv",
            ["global_step", "val_macro_mse", "is_best"],
            [{"global_step": 100_000, "val_macro_mse": 0.01, "is_best": True}],
        )
        _write_csv(
            records / "validation_selection.csv",
            [
                "protocol_id", "run_id", "method", "variant", "seed", "global_step",
                "val_macro_mse", "val_macro_psnr", "val_macro_ssim",
                "learning_rate_before_scheduler", "learning_rate_after_scheduler",
                "is_best", "checkpoint_sha256", "formal_run", "selection_role", "notes",
            ],
            [{
                "protocol_id": PROTOCOL_ID,
                "run_id": "transsar_v2_seed42",
                "method": "transsar_v2",
                "variant": "",
                "seed": 42,
                "global_step": 100_000,
                "val_macro_mse": 0.01,
                "is_best": True,
                "checkpoint_sha256": sha256_file(baseline_dir / "checkpoint_best.pth"),
                "formal_run": True,
                "selection_role": "strongest_non_ours_baseline",
            }],
        )

        config = {
            "effective_protocol": {
                "protocol_id": PROTOCOL_ID,
                "artifact_protocol_id": PROTOCOL_ID,
                "mode": "formal",
                "looks": [1, 2, 4, 8],
                "ucm": {"expected_classes": 21, "expected_per_class": 100},
                "seeds": {"ucm_test": 20260905},
                "real": {
                    "use_gt16": False,
                    "evaluation_scope": "no_reference_only",
                    "forbidden_paper_metrics": [
                        "qpsnr_gt16", "qssim_gt16"
                    ],
                },
            }
        }
        (prep / "config_snapshot.json").write_text(json.dumps(config), encoding="utf-8")
        pair_fields = [
            "protocol_id", "dataset", "split", "pair_id", "source_id",
            "source_relative_path", "class_name", "selection_rank", "parent_split",
            "global_L", "rng_seed", "mat_path", "clean_path", "clean_field",
            "noisy_path", "noisy_field", "source_sha256", "source_pixel_sha256",
            "source_phash", "clean_sha256", "noisy_sha256", "mat_sha256",
            "preclip_max", "postclip_max", "saturation_rate", "original_height",
            "original_width", "processed_height", "processed_width", "size_policy",
            "size_changed", "numeric_domain",
        ]
        pair_rows: list[dict[str, object]] = []
        for class_index in range(21):
            class_name = f"class_{class_index:02d}"
            for rank in range(1, 101):
                source_id = f"source_{class_index:02d}_{rank:03d}"
                source_path = f"ucm/{class_name}/{source_id}.tif"
                for looks in (1, 2, 4, 8):
                    pair_id = f"pair_{class_index:02d}_{rank:03d}_L{looks}"
                    pair_path = f"fixed/{pair_id}.mat"
                    pair_rows.append({
                        "protocol_id": PROTOCOL_ID,
                        "dataset": "UCMerced_LandUse",
                        "split": "test",
                        "pair_id": pair_id,
                        "source_id": source_id,
                        "source_relative_path": source_path,
                        "class_name": class_name,
                        "selection_rank": rank,
                        "parent_split": "test",
                        "global_L": looks,
                        "rng_seed": stable_u64(20260905, source_id, looks),
                        "mat_path": pair_path,
                        "clean_path": pair_path,
                        "clean_field": "clean",
                        "noisy_path": pair_path,
                        "noisy_field": "noisy",
                        "source_sha256": "1" * 64,
                        "source_pixel_sha256": "2" * 64,
                        "source_phash": "3" * 16,
                        "clean_sha256": "4" * 64,
                        "noisy_sha256": "5" * 64,
                        "mat_sha256": "6" * 64,
                        "preclip_max": 1.0,
                        "postclip_max": 1.0,
                        "saturation_rate": 0.0,
                        "original_height": 256,
                        "original_width": 256,
                        "processed_height": 256,
                        "processed_width": 256,
                        "size_policy": "identity_if_256_else_bilinear_resize",
                        "size_changed": False,
                        "numeric_domain": "linear_normalized_intensity_0_1",
                    })
        _write_csv(prep / "ucm_test_manifest.csv", pair_fields, pair_rows)
        for name in (
            "source_manifest.csv", "train_schedule_seed42.csv",
            "nwpu_validation_manifest.csv", "cross_dataset_phash_audit.csv",
            "prepare_summary.json",
        ):
            (prep / name).write_text("fixture\n", encoding="utf-8")
        critical_names = (
            "config_snapshot.json", "source_manifest.csv", "train_schedule_seed42.csv",
            "nwpu_validation_manifest.csv", "ucm_test_manifest.csv",
            "cross_dataset_phash_audit.csv", "prepare_summary.json",
        )
        prepared_hashes = {
            "state": "complete",
            "protocol_id": PROTOCOL_ID,
            "critical_files": {name: sha256_file(prep / name) for name in critical_names},
        }
        (prep / "artifact_hashes.json").write_text(
            json.dumps(prepared_hashes), encoding="utf-8"
        )

        real_fields = [
            "sample_id", "parent_id", "split", "valid_no_reference",
            "valid_enl_rois", "noisy_shape", "gt16_path", "gt16_sha256",
            "pair_exists", "valid_quasi_reference",
        ]
        real_rows: list[dict[str, object]] = []
        offset = 0
        for split, parents in (("train", 1_175), ("val", 146), ("test", 148)):
            for parent_index in range(parents):
                parent_id = f"{split}_parent_{parent_index:04d}"
                for patch_index in range(4):
                    real_rows.append({
                        "sample_id": f"{split}_sample_{offset:04d}",
                        "parent_id": parent_id,
                        "split": split,
                        "valid_no_reference": True,
                        "valid_enl_rois": True,
                        "noisy_shape": "256x256",
                        "gt16_path": "",
                        "gt16_sha256": "",
                        "pair_exists": False,
                        "valid_quasi_reference": False,
                    })
                    offset += 1
        _write_csv(real / "real_qc.csv", real_fields, real_rows)
        real_manifest = {
            "schema_version": 1,
            "expected_shape": [256, 256],
            "split_manifest_sha256": REAL_SPLIT_MANIFEST_SHA256,
            "gt16_root": None,
            "gt16_disabled_by_protocol": True,
            "counts": {
                "gt16_files_discovered": 0,
                "gt16_files_matched_one_to_one": 0,
                "gt16_files_unmatched": 0,
            },
            "artifacts": {
                "qc_csv": "real_qc.csv",
                "qc_csv_sha256": sha256_file(real / "real_qc.csv"),
            },
        }
        (real / "real_qc_manifest.json").write_text(
            json.dumps(real_manifest), encoding="utf-8"
        )

        figure_fields = [
            "protocol_id", "figure", "panel", "dataset", "source_id", "class_name",
            "L", "parent_id", "sample_id", "region_type", "crop_x", "crop_y",
            "crop_width", "crop_height", "selection_rule",
            "selected_before_test_unlock", "notes",
        ]
        _write_csv(
            records / "figure_selection.csv",
            figure_fields,
            [
                {
                    "protocol_id": PROTOCOL_ID, "figure": "Fig4", "panel": "A",
                    "dataset": "UCM", "source_id": "source_00_001",
                    "class_name": "class_00", "L": 1, "crop_x": 0, "crop_y": 0,
                    "crop_width": 64, "crop_height": 64,
                    "selection_rule": "fixed before test",
                    "selected_before_test_unlock": True,
                },
                {
                    "protocol_id": PROTOCOL_ID, "figure": "Fig4", "panel": "B",
                    "dataset": "UCM", "source_id": "source_00_002",
                    "class_name": "class_00", "L": 4, "crop_x": 1, "crop_y": 2,
                    "crop_width": 64, "crop_height": 64,
                    "selection_rule": "fixed before test",
                    "selected_before_test_unlock": True,
                },
                {
                    "protocol_id": PROTOCOL_ID, "figure": "Fig5", "panel": "A",
                    "dataset": "RealSAR", "sample_id": "test_sample_5284",
                    "parent_id": "test_parent_0000", "region_type": "homogeneous",
                    "crop_x": 0, "crop_y": 0, "crop_width": 64, "crop_height": 64,
                    "selection_rule": "fixed before test",
                    "selected_before_test_unlock": True,
                },
                {
                    "protocol_id": PROTOCOL_ID, "figure": "Fig5", "panel": "B",
                    "dataset": "RealSAR", "sample_id": "test_sample_5288",
                    "parent_id": "test_parent_0001", "region_type": "structured",
                    "crop_x": 2, "crop_y": 3, "crop_width": 64, "crop_height": 64,
                    "selection_rule": "fixed before test",
                    "selected_before_test_unlock": True,
                },
            ],
        )
        return run, prep, real

    def test_valid_seal_and_tamper_detection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run, prep, real = self._fixture(Path(temporary))
            sealed = seal(run, prep, real)
            result = verify(run)
            self.assertTrue(result["ok"])
            self.assertEqual(result["pretest_seal_sha256"], sealed["pretest_seal_sha256"])
            validation = run / "records" / "validation_selection.csv"
            validation.write_text(validation.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "semantic binding mismatch"):
                verify(run)

    def test_blank_template_does_not_seal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run, prep, real = self._fixture(Path(temporary))
            _write_csv(
                run / "records" / "validation_selection.csv",
                ["method", "selection_role"],
                [{"method": "", "selection_role": ""}],
            )
            with self.assertRaisesRegex(ValueError, "exactly one canonical"):
                seal(run, prep, real)

    def test_nonexistent_ids_and_out_of_bounds_crops_are_rejected(self) -> None:
        for mutation, message in (("missing_source", "resolves to 0"), ("bad_crop", "exceeds")):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                run, prep, real = self._fixture(Path(temporary))
                figures = run / "records" / "figure_selection.csv"
                with figures.open(newline="", encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle))
                if mutation == "missing_source":
                    rows[0]["source_id"] = "does_not_exist"
                else:
                    rows[2]["crop_x"] = "250"
                _write_csv(figures, list(rows[0]), rows)
                with self.assertRaisesRegex(ValueError, message):
                    seal(run, prep, real)

    def test_missing_required_figure_stratum_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run, prep, real = self._fixture(Path(temporary))
            figures = run / "records" / "figure_selection.csv"
            with figures.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            rows = [row for row in rows if not (row["dataset"] == "UCM" and row["L"] == "4")]
            _write_csv(figures, list(rows[0]), rows)
            with self.assertRaisesRegex(ValueError, "L=1 and L=4"):
                seal(run, prep, real)

    def test_no_gt16_protocol_rejects_gt16_manifest_or_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run, prep, real = self._fixture(Path(temporary))
            manifest_path = real / "real_qc_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["gt16_root"] = "GT16"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no-GT16"):
                seal(run, prep, real)

        with tempfile.TemporaryDirectory() as temporary:
            run, prep, real = self._fixture(Path(temporary))
            qc_path = real / "real_qc.csv"
            with qc_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["gt16_path"] = "GT16/unexpected.mat"
            rows[0]["gt16_sha256"] = "a" * 64
            rows[0]["pair_exists"] = True
            rows[0]["valid_quasi_reference"] = True
            _write_csv(qc_path, list(rows[0]), rows)
            manifest_path = real / "real_qc_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["artifacts"]["qc_csv_sha256"] = sha256_file(qc_path)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "contain GT16 data"):
                seal(run, prep, real)

    def test_checkpoint_and_future_timestamp_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run, prep, real = self._fixture(Path(temporary))
            validation = run / "records" / "validation_selection.csv"
            with validation.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["checkpoint_sha256"] = "a" * 64
            _write_csv(validation, list(rows[0]), rows)
            with self.assertRaisesRegex(ValueError, "checkpoint SHA-256"):
                seal(run, prep, real)

        with tempfile.TemporaryDirectory() as temporary:
            run, prep, real = self._fixture(Path(temporary))
            seal(run, prep, real)
            timestamp = run / "records" / "test_unlock_utc.txt"
            future = (datetime.now(timezone.utc) + timedelta(days=2)).replace(
                microsecond=0
            ).isoformat().replace("+00:00", "Z")
            timestamp.write_text(future + "\n", encoding="utf-8")
            seal_path = run / "records" / "pretest_seal.json"
            payload = json.loads(seal_path.read_text(encoding="utf-8"))
            payload["sealed_at_utc"] = future
            payload["unlock_timestamp_sha256"] = sha256_file(timestamp)
            seal_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "future"):
                verify(run)

    def test_seal_is_one_shot_and_refuses_prior_test_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run, prep, real = self._fixture(Path(temporary))
            seal(run, prep, real)
            with self.assertRaisesRegex(FileExistsError, "one-shot"):
                seal(run, prep, real)
        with tempfile.TemporaryDirectory() as temporary:
            run, prep, real = self._fixture(Path(temporary))
            forbidden = run / "ucm" / "old" / "aggregate.json"
            forbidden.parent.mkdir(parents=True)
            forbidden.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "test-derived files"):
                seal(run, prep, real)


if __name__ == "__main__":
    unittest.main()
