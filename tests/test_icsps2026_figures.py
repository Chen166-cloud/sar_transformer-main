from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from experiment_protocol import sha256_file
from export_icsps2026_ucm_figures import (
    _checkpoint_model,
    select_registered_ucm_pairs,
    validate_crop,
)


class RegisteredFigureTests(unittest.TestCase):
    def test_selection_is_exact_and_requires_l1_l4(self) -> None:
        registrations = [
            {
                "protocol_id": "ICSPS26-FROZEN-v2",
                "dataset": "UCM",
                "source_id": "source-a",
                "class_name": "harbor",
                "L": str(looks),
                "crop_x": "4",
                "crop_y": "8",
                "crop_width": "16",
                "crop_height": "20",
                "selection_rule": "fixed before test",
                "selected_before_test_unlock": "TRUE",
            }
            for looks in (1, 4)
        ]
        manifest = [
            {
                "pair_id": f"pair-{looks}",
                "source_id": "source-a",
                "class_name": "harbor",
                "global_L": str(looks),
                "protocol_id": "ICSPS26-FROZEN-v2",
                "dataset": "UCMerced_LandUse",
                "processed_height": "256",
                "processed_width": "256",
            }
            for looks in (1, 2, 4, 8)
        ]
        selected = select_registered_ucm_pairs(registrations, manifest)
        self.assertEqual([row[1]["pair_id"] for row in selected], ["pair-1", "pair-4"])

        with self.assertRaisesRegex(ValueError, "L=1 and L=4"):
            select_registered_ucm_pairs(registrations[:1], manifest)

    def test_registered_crop_must_fit(self) -> None:
        registration = {
            "crop_x": "240",
            "crop_y": "240",
            "crop_width": "16",
            "crop_height": "16",
        }
        self.assertEqual(validate_crop(np.zeros((256, 256)), registration), (240, 240, 16, 16))
        registration["crop_width"] = "17"
        with self.assertRaisesRegex(ValueError, "exceeds"):
            validate_crop(np.zeros((256, 256)), registration)

    def test_checkpoint_must_be_completion_selected_best(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint_path = root / "checkpoint_best.pth"
            checkpoint = {
                "checkpoint_format": "icsps26-resumable-v1",
                "state_dict": {},
                "global_step": 50_000,
                "run_config": {
                    "protocol_id": "ICSPS26-FROZEN-v2",
                    "artifact_protocol_id": "ICSPS26-FROZEN-v2",
                    "formal_run": True,
                    "method": "transsar_v2",
                    "variant": None,
                    "seed": 42,
                    "target_updates": 100_000,
                    "numeric_domain": "intensity_v1",
                },
                "model_metadata": {"method": "transsar_v2", "variant": None},
            }
            torch.save(checkpoint, checkpoint_path)
            completion = {
                "protocol_id": "ICSPS26-FROZEN-v2",
                "formal_run": True,
                "completed_updates": 100_000,
                "target_updates": 100_000,
                "best": {"global_step": 45_000, "val_macro_mse": 0.01},
                "checkpoint_best_sha256": sha256_file(checkpoint_path),
            }
            (root / "completion.json").write_text(
                json.dumps(completion), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "completion_best_step"):
                _checkpoint_model("transsar_v2", checkpoint_path, torch.device("cpu"), None)


if __name__ == "__main__":
    unittest.main()
