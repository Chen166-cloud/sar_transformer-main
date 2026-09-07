from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from experiment_protocol import sha256_file
from export_icsps2026_real_figures import (
    build_completed_model,
    load_registered_noisy_sample,
    validate_real_qc_chain,
)


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class RealFigureQCChainTests(unittest.TestCase):
    def test_qc_chain_and_shared_registered_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "real_dataset"
            protocol = root / "protocol"
            dataset.mkdir()
            protocol.mkdir()
            source = dataset / "noisy.png"
            gradient = np.tile(np.arange(256, dtype=np.uint8), (256, 1))
            Image.fromarray(gradient).save(source)
            source_hash = sha256_file(source)

            qc_rows: list[dict[str, object]] = []
            for parent_index in range(148):
                for child in range(4):
                    qc_rows.append(
                        {
                            "sample_id": f"sample_{parent_index:03d}_{child}",
                            "parent_id": f"parent_{parent_index:03d}",
                            "split": "test",
                            "noisy_path": source.name,
                            "noisy_field": "image_file",
                            "noisy_shape": "256x256",
                            "noisy_dtype": "uint8",
                            "noisy_min": 0.0,
                            "noisy_max": 255.0,
                            "valid_no_reference": True,
                            "valid_enl_rois": True,
                            "noisy_sha256": source_hash,
                        }
                    )
            qc_path = protocol / "real_qc_manifest.csv"
            _write_csv(qc_path, list(qc_rows[0]), qc_rows)
            roi_path = protocol / "real_enl_roi_manifest.csv"
            alignment_path = protocol / "real_alignment_audit.csv"
            roi_path.write_text("sample_id,roi_id\n", encoding="utf-8")
            alignment_path.write_text("sample_id,alignment_pass\n", encoding="utf-8")
            manifest = {
                "schema_version": 1,
                "dataset_root_name": dataset.name,
                "expected_shape": [256, 256],
                "numeric_mapping": {
                    "mode": "uint8_255",
                    "offset": 0.0,
                    "scale": 255.0,
                    "per_image_percentile_normalization": False,
                },
                "artifacts": {
                    "qc_csv": qc_path.name,
                    "qc_csv_sha256": sha256_file(qc_path),
                    "roi_csv": roi_path.name,
                    "roi_csv_sha256": sha256_file(roi_path),
                    "alignment_csv": alignment_path.name,
                    "alignment_csv_sha256": sha256_file(alignment_path),
                },
            }
            manifest_path = protocol / "real_qc_manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            gate = {
                "real_qc_manifest_sha256": sha256_file(manifest_path),
                "real_qc_csv_sha256": sha256_file(qc_path),
            }
            figures = [
                {
                    "protocol_id": "ICSPS26-FROZEN-v2",
                    "figure": "Fig5",
                    "panel": panel,
                    "dataset": "RealSAR",
                    "sample_id": sample_id,
                    "parent_id": parent_id,
                    "region_type": region_type,
                    "crop_x": "16",
                    "crop_y": "24",
                    "crop_width": "64",
                    "crop_height": "64",
                    "selection_rule": "sealed before test",
                    "selected_before_test_unlock": "TRUE",
                }
                for panel, sample_id, parent_id, region_type in (
                    ("A", "sample_000_0", "parent_000", "homogeneous"),
                    ("B", "sample_001_0", "parent_001", "structured"),
                )
            ]
            loaded_manifest, returned_qc, _, matched = validate_real_qc_chain(
                dataset, protocol, gate, figures
            )
            self.assertEqual(returned_qc, qc_path)
            self.assertEqual(len(matched), 2)
            sample = load_registered_noisy_sample(
                dataset, loaded_manifest["numeric_mapping"], *matched[0]
            )
            self.assertEqual(sample["crop"], (16, 24, 64, 64))
            self.assertEqual(sample["noisy"].shape, (256, 256))
            self.assertAlmostEqual(float(sample["noisy"].max()), 1.0)

            qc_path.write_text(qc_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "artifact SHA-256 mismatch|sealed"):
                validate_real_qc_chain(dataset, protocol, gate, figures)

    def test_ams_checkpoint_must_match_completion_selected_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint_path = root / "checkpoint_best.pth"
            checkpoint = {
                "checkpoint_format": "icsps26-ams-v1",
                "state_dict": {},
                "epoch": 4,
                "run_config": {
                    "protocol_id": "ICSPS26-FROZEN-v2",
                    "artifact_protocol_id": "ICSPS26-FROZEN-v2",
                    "formal_run": True,
                    "numeric_domain": "intensity_v1",
                    "method": "ours",
                    "variant": "full",
                    "adaptation": "ams",
                    "seed": 42,
                    "epochs": 8,
                    "selection": (
                        "lowest fixed-mask real-validation loss over adapted epochs 1-8"
                    ),
                },
                "model_metadata": {"method": "ours", "variant": "full"},
            }
            torch.save(checkpoint, checkpoint_path)
            completion = {
                "protocol_id": "ICSPS26-FROZEN-v2",
                "formal_run": True,
                "completed_epochs": 8,
                "best": {"epoch": 5, "fixed_val_masked_loss": 0.01},
                "checkpoint_best_sha256": sha256_file(checkpoint_path),
            }
            (root / "completion.json").write_text(json.dumps(completion), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "completion_best_epoch"):
                build_completed_model(
                    "ours_full_ams",
                    checkpoint_path,
                    "ams",
                    torch.device("cpu"),
                    None,
                )


if __name__ == "__main__":
    unittest.main()
