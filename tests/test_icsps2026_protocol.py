from __future__ import annotations

import hashlib
import csv
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from experiment_protocol import require_finite_number, require_finite_tensor
from icsps2026_protocol import (
    ICSPSFixedPairDataset,
    ICSPSOnlineGammaDataset,
    apply_d4,
    build_train_schedule,
    load_clean_intensity,
    load_protocol_config,
    portable_relative_path,
    read_csv_records,
    synthesize_gamma_only,
    validate_train_schedule,
)
from prepare_icsps2026_data import prepare_data, verify_prepared_data
from train_icsps2026 import reconcile_logs_with_checkpoint
from train_icsps2026_ams import require_selected_ams_best


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "icsps2026_frozen_v2.json"


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _fake_train_records(classes: int = 45, sources_per_class: int = 20):
    rows = []
    for class_index in range(classes):
        class_name = f"class_{class_index:02d}"
        for source_index in range(sources_per_class):
            source_id = f"s{class_index:02d}_{source_index:02d}"
            rows.append(
                {
                    "protocol_id": "ICSPS26-FROZEN-v2",
                    "split": "train",
                    "class_name": class_name,
                    "source_id": source_id,
                    "source_relative_path": f"{class_name}/{source_index:03d}.png",
                    "selection_rank": source_index + 1,
                    "parent_split": "train",
                    "source_sha256": _digest(f"file:{source_id}"),
                    "source_pixel_sha256": _digest(f"pixels:{source_id}"),
                    "source_phash": _digest(f"phash:{source_id}")[:16],
                    "original_height": 256,
                    "original_width": 256,
                    "numeric_domain": "linear_normalized_intensity_0_1",
                }
            )
    return rows


def _unique_rgb(height: int, width: int, token: int) -> np.ndarray:
    yy, xx = np.indices((height, width), dtype=np.uint32)
    rgb = np.empty((height, width, 3), dtype=np.uint8)
    rgb[..., 0] = (xx + token * 17 + yy * 3) % 256
    rgb[..., 1] = (yy + token * 29 + xx * 5) % 256
    rgb[..., 2] = (xx * 7 + yy * 11 + token * 41) % 256
    return rgb


def _write_smoke_sources(root: Path) -> tuple[Path, Path]:
    nwpu = root / "nwpu"
    ucm = root / "ucm"
    for class_index in range(3):
        class_dir = nwpu / f"nwpu_{class_index:02d}"
        class_dir.mkdir(parents=True)
        for image_index in range(8):
            token = 1 + class_index * 20 + image_index
            Image.fromarray(_unique_rgb(256, 256, token)).save(
                class_dir / f"image_{image_index:03d}.png"
            )
    for class_index in range(2):
        class_dir = ucm / f"ucm_{class_index:02d}"
        class_dir.mkdir(parents=True)
        for image_index in range(3):
            token = 200 + class_index * 20 + image_index
            shape = (256, 256) if image_index == 0 else (240 + image_index, 270)
            Image.fromarray(_unique_rgb(*shape, token)).save(
                class_dir / f"image_{image_index:03d}.png"
            )
    return nwpu, ucm


class FrozenConfigurationTests(unittest.TestCase):
    def test_nonfinite_values_fail_fast(self) -> None:
        require_finite_tensor(torch.ones(1), "test tensor")
        self.assertEqual(require_finite_number(1.0, "test number"), 1.0)
        with self.assertRaises(FloatingPointError):
            require_finite_tensor(torch.tensor([float("nan")]), "prediction")
        with self.assertRaises(FloatingPointError):
            require_finite_number(float("inf"), "loss")

    def test_ams_run_without_selected_epoch_cannot_create_completion(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "completion.json was not created"):
            require_selected_ams_best(
                {"epoch": None, "fixed_val_masked_loss": float("inf")}
            )

    def test_formal_training_and_ams_values_are_frozen(self) -> None:
        config = load_protocol_config(CONFIG_PATH)
        training = config["training"]
        self.assertEqual(training["batch_size"], 1)
        self.assertEqual(training["optimizer_updates"], 100_000)
        self.assertEqual(training["validation_interval_updates"], 5_000)
        self.assertEqual(training["optimizer"]["name"], "Adam")
        self.assertEqual(training["optimizer"]["learning_rate"], 1e-3)
        self.assertEqual(training["optimizer"]["weight_decay"], 1e-5)
        self.assertEqual(training["loss"]["tv_definition"], "mean_total_variation")
        self.assertEqual(training["loss"]["lambda_tv"], 0.03)
        self.assertEqual(training["scheduler"]["factor"], 0.5)
        self.assertEqual(training["scheduler"]["patience_validation_events"], 4)
        self.assertEqual(training["scheduler"]["min_lr"], 1e-6)
        self.assertEqual(
            config["nwpu"]["canonical_master_manifest_sha256"],
            "748b7010bf4f4c29eac5c814e32ba0f360e49ae7e063a20e03981a231c75e9bb",
        )
        self.assertEqual(config["ams"]["epochs"], 8)
        self.assertEqual(config["ams"]["mask"]["ratio"], 0.2)
        self.assertEqual(config["ams"]["loss"]["lambda_tv"], 1e-3)
        self.assertIs(config["real"]["use_gt16"], False)
        self.assertEqual(config["real"]["evaluation_scope"], "no_reference_only")
        self.assertEqual(
            set(config["real"]["forbidden_paper_metrics"]),
            {"qpsnr_gt16", "qssim_gt16"},
        )
        self.assertEqual(
            config["ams"]["checkpoint_selection"],
            {
                "selection_metric": "fixed_real_validation_masked_loss",
                "selection_mode": "min",
                "candidate_epochs": "adapted_epochs_1_to_8",
            },
        )

    def test_smoke_overrides_budget_without_mutating_formal_config(self) -> None:
        smoke = load_protocol_config(CONFIG_PATH, smoke=True)
        formal = load_protocol_config(CONFIG_PATH, smoke=False)
        self.assertEqual(smoke["schedule"]["updates"], 48)
        self.assertEqual(smoke["training"]["optimizer_updates"], 48)
        self.assertEqual(smoke["training"]["validation_interval_updates"], 16)
        self.assertEqual(formal["schedule"]["updates"], 100_000)

    def test_resume_logs_are_reconciled_to_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metrics_path = root / "step_metrics.csv"
            with metrics_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["global_step", "loss"])
                writer.writeheader()
                writer.writerows(
                    [
                        {"global_step": 0, "loss": ""},
                        {"global_step": 5, "loss": 1.0},
                        {"global_step": 10, "loss": 0.5},
                    ]
                )
            progress_path = root / "train_progress.jsonl"
            progress_path.write_text(
                "".join(
                    json.dumps({"global_step": step}) + "\n"
                    for step in (2, 4, 6, 8)
                ),
                encoding="utf-8",
            )
            result = reconcile_logs_with_checkpoint(metrics_path, progress_path, 5)
            self.assertEqual(result, {"metrics_rows_removed": 1, "progress_rows_removed": 2})
            with metrics_path.open(newline="", encoding="utf-8") as handle:
                self.assertEqual(
                    [int(row["global_step"]) for row in csv.DictReader(handle)],
                    [0, 5],
                )
            self.assertEqual(
                [json.loads(line)["global_step"] for line in progress_path.read_text().splitlines()],
                [2, 4],
            )

    def test_portable_paths_reject_absolute_and_traversal(self) -> None:
        self.assertEqual(portable_relative_path("a/b/c.mat"), "a/b/c.mat")
        for unsafe in ("/tmp/a.mat", "../a.mat", "a/../../b.mat", "C:\\x.mat"):
            with self.assertRaises(ValueError, msg=unsafe):
                portable_relative_path(unsafe)


class FormalScheduleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.records = _fake_train_records()
        cls.schedule = build_train_schedule(
            cls.records,
            updates=100_000,
            looks=(1, 2, 4, 8),
            run_seed=42,
            synthesis_seed=20260904,
            protocol_id="ICSPS26-FROZEN-v2",
        )

    def test_exact_formal_marginals(self) -> None:
        audit = validate_train_schedule(
            self.schedule,
            self.records,
            updates=100_000,
            looks=(1, 2, 4, 8),
            run_seed=42,
            synthesis_seed=20260904,
            protocol_id="ICSPS26-FROZEN-v2",
        )
        self.assertEqual(audit["look_counts"], {1: 25_000, 2: 25_000, 4: 25_000, 8: 25_000})
        self.assertEqual(set(audit["d4_counts"].values()), {12_500})
        self.assertEqual((audit["source_count_min"], audit["source_count_max"]), (111, 112))
        self.assertEqual((audit["class_count_min"], audit["class_count_max"]), (2_222, 2_223))
        class_counts = Counter(row["class_name"] for row in self.schedule)
        source_counts = Counter(row["source_id"] for row in self.schedule)
        self.assertEqual(Counter(class_counts.values()), Counter({2_222: 35, 2_223: 10}))
        self.assertEqual(Counter(source_counts.values()), Counter({111: 800, 112: 100}))

    def test_schedule_is_byte_value_deterministic(self) -> None:
        repeated = build_train_schedule(
            self.records,
            updates=100_000,
            looks=(1, 2, 4, 8),
            run_seed=42,
            synthesis_seed=20260904,
            protocol_id="ICSPS26-FROZEN-v2",
        )
        self.assertEqual(self.schedule, repeated)


class SmokePreparationTests(unittest.TestCase):
    def test_end_to_end_smoke_prepare_replay_and_verify(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            nwpu_root, ucm_root = _write_smoke_sources(root)
            output = root / "prepared"
            summary = prepare_data(
                config_path=CONFIG_PATH,
                nwpu_root=nwpu_root,
                ucm_root=ucm_root,
                output_root=output,
                run_seed=42,
                smoke=True,
                mat_compression=False,
                quiet=True,
            )
            self.assertEqual(summary["source_manifest"]["train_sources"], 6)
            self.assertEqual(summary["source_manifest"]["validation_sources"], 3)
            self.assertEqual(summary["train_schedule"]["rows"], 48)
            self.assertEqual(summary["fixed_pairs"]["nwpu_validation"]["pairs"], 12)
            self.assertEqual(summary["fixed_pairs"]["ucm_test"]["pairs"], 24)
            self.assertEqual(
                {
                    key: value["pairs"]
                    for key, value in summary["fixed_pairs"]["nwpu_validation"][
                        "synthesis_by_L"
                    ].items()
                },
                {"1": 3, "2": 3, "4": 3, "8": 3},
            )
            self.assertEqual(
                {
                    key: value["pairs"]
                    for key, value in summary["fixed_pairs"]["ucm_test"][
                        "synthesis_by_L"
                    ].items()
                },
                {"1": 6, "2": 6, "4": 6, "8": 6},
            )
            for dataset in ("nwpu_validation", "ucm_test"):
                for values in summary["fixed_pairs"][dataset]["synthesis_by_L"].values():
                    self.assertLessEqual(
                        values["saturation_rate_min"], values["saturation_rate_mean"]
                    )
                    self.assertLessEqual(
                        values["saturation_rate_mean"], values["saturation_rate_max"]
                    )
            self.assertTrue((output / "cross_dataset_phash_audit.csv").is_file())
            self.assertFalse((root / ".prepared.INCOMPLETE").exists())

            source_rows = read_csv_records(output / "source_manifest.csv")
            schedule_rows = read_csv_records(output / "train_schedule_seed42.csv")
            nwpu_pairs = read_csv_records(output / "nwpu_validation_manifest.csv")
            ucm_pairs = read_csv_records(output / "ucm_test_manifest.csv")
            for row in source_rows:
                self.assertFalse(Path(row["source_relative_path"]).is_absolute())
            for row in nwpu_pairs + ucm_pairs:
                self.assertFalse(Path(row["mat_path"]).is_absolute())
                self.assertEqual(row["mat_path"], row["clean_path"])
                self.assertEqual(row["mat_path"], row["noisy_path"])
                self.assertTrue(row["selection_rank"])
                self.assertTrue(row["parent_split"])
            self.assertTrue(any(row["size_changed"] == "true" for row in ucm_pairs))
            self.assertEqual(
                {row["size_policy"] for row in ucm_pairs},
                {"identity_if_256_else_bilinear_resize"},
            )

            online = ICSPSOnlineGammaDataset(
                nwpu_root,
                source_rows,
                schedule_rows,
                protocol_id="ICSPS26-FROZEN-v2-SMOKE",
                verify_source_hashes=True,
            )
            item = online[0]
            source = {row["source_id"]: row for row in source_rows}[item["source_id"]]
            clean, _ = load_clean_intensity(
                nwpu_root / source["source_relative_path"],
                size_policy="require_exact_256x256_no_resampling",
            )
            expected_noisy, _ = synthesize_gamma_only(clean, item["L"], item["speckle_seed"])
            expected_clean = apply_d4(clean, item["d4_id"])
            expected_noisy = apply_d4(expected_noisy, item["d4_id"])
            torch.testing.assert_close(item["clean"], torch.from_numpy(expected_clean[None]))
            torch.testing.assert_close(item["noisy"], torch.from_numpy(expected_noisy[None]))
            torch.testing.assert_close(online[0]["noisy"], item["noisy"])
            collated = next(iter(DataLoader(online, batch_size=1, shuffle=False)))
            self.assertEqual(collated["speckle_seed"], [item["speckle_seed"]])
            self.assertEqual(int(collated["speckle_seed"][0]), int(schedule_rows[0]["speckle_seed"]))

            fixed = ICSPSFixedPairDataset(
                output,
                output / "nwpu_validation_manifest.csv",
                protocol_id="ICSPS26-FROZEN-v2-SMOKE",
                verify_hashes=True,
            )
            fixed_item = fixed[0]
            self.assertEqual(
                set(fixed_item),
                {"pair_id", "source_id", "class_name", "L", "noisy", "clean"},
            )
            self.assertEqual(fixed_item["clean"].shape, (1, 256, 256))
            self.assertEqual(fixed_item["noisy"].dtype, torch.float32)

            audit = verify_prepared_data(
                config_path=CONFIG_PATH,
                nwpu_root=nwpu_root,
                ucm_root=ucm_root,
                output_root=output,
                run_seed=42,
                smoke=True,
                verify_source_files=True,
                verify_pair_files=True,
            )
            self.assertEqual(audit["state"], "fully_verified")

            repeated_output = root / "prepared_repeat"
            repeated_summary = prepare_data(
                config_path=CONFIG_PATH,
                nwpu_root=nwpu_root,
                ucm_root=ucm_root,
                output_root=repeated_output,
                run_seed=42,
                smoke=True,
                mat_compression=False,
                quiet=True,
            )
            self.assertEqual(summary, repeated_summary)
            self.assertEqual(
                (output / "artifact_hashes.json").read_bytes(),
                (repeated_output / "artifact_hashes.json").read_bytes(),
            )
            with self.assertRaises(FileExistsError):
                prepare_data(
                    config_path=CONFIG_PATH,
                    nwpu_root=nwpu_root,
                    ucm_root=ucm_root,
                    output_root=output,
                    run_seed=42,
                    smoke=True,
                    quiet=True,
                )

            first_pair_path = output / nwpu_pairs[0]["mat_path"]
            first_pair_path.write_bytes(first_pair_path.read_bytes() + b"tampered")
            with self.assertRaisesRegex(AssertionError, "MAT SHA-256 mismatch"):
                fixed[0]

    def test_failed_prepare_never_publishes_final_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "failed_protocol"
            with self.assertRaises(FileNotFoundError):
                prepare_data(
                    config_path=CONFIG_PATH,
                    nwpu_root=root / "missing_nwpu",
                    ucm_root=root / "missing_ucm",
                    output_root=output,
                    run_seed=42,
                    smoke=True,
                    quiet=True,
                )
            self.assertFalse(output.exists())
            self.assertTrue((root / ".failed_protocol.INCOMPLETE").is_dir())


if __name__ == "__main__":
    unittest.main()
