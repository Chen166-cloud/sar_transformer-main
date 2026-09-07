from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import savemat

from experiment_protocol import sha256_file
from finalize_icsps2026_sarbm3d_raw import (
    SARBM3D_DOMAIN_CONVERSION,
    SARBM3D_INPUT_DOMAIN,
    SARBM3D_OUTPUT_DOMAIN,
    finalize_raw_predictions,
)


class SarBM3DRawFinalizerTests(unittest.TestCase):
    def _case(self, root: Path, *, bad_shape: bool = False) -> tuple[Path, Path, Path]:
        pair_root = root / "pairs"
        prediction_root = root / "raw"
        jobs_path = root / "ucm_jobs.csv"
        pair_root.mkdir()
        (prediction_root / "predictions").mkdir(parents=True)
        jobs = []
        for index, look in enumerate((1, 2), start=1):
            pair_id = f"pair_{index}"
            relative_input = f"fixed/{pair_id}.mat"
            input_path = pair_root / relative_input
            input_path.parent.mkdir(parents=True, exist_ok=True)
            noisy = np.full((4, 4), index / 10.0, dtype=np.float32)
            savemat(input_path, {"noisy": noisy, "global_L": look})
            relative_prediction = f"predictions/{pair_id}.mat"
            prediction = noisy[:3] if bad_shape and index == 2 else noisy
            savemat(
                prediction_root / relative_prediction,
                {
                    "prediction": prediction,
                    "pair_id": pair_id,
                    "looks": look,
                    "inference_seconds": 0.25,
                    "prediction_amplitude_min": float(np.min(prediction)),
                    "prediction_amplitude_max": float(np.max(prediction)),
                    "input_domain": SARBM3D_INPUT_DOMAIN,
                    "output_domain": SARBM3D_OUTPUT_DOMAIN,
                    "domain_conversion": SARBM3D_DOMAIN_CONVERSION,
                },
            )
            jobs.append(
                {
                    "pair_id": pair_id,
                    "mat_path": relative_input,
                    "noisy_field": "noisy",
                    "L": look,
                    "input_mat_sha256": sha256_file(input_path),
                    "output_relative_path": relative_prediction,
                }
            )
        with jobs_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(jobs[0]))
            writer.writeheader()
            writer.writerows(jobs)
        metadata = {
            "protocol_id": "ICSPS26-FROZEN-v2",
            "formal_run": False,
            "jobs": len(jobs),
            "job_manifest_sha256": sha256_file(jobs_path),
            "sarbm3d_input_domain": SARBM3D_INPUT_DOMAIN,
            "scored_output_domain": SARBM3D_OUTPUT_DOMAIN,
            "domain_conversion": SARBM3D_DOMAIN_CONVERSION,
        }
        jobs_path.with_suffix(".csv.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
        return jobs_path, pair_root, prediction_root

    def test_smoke_finalizer_publishes_manifest_then_completion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            jobs, pairs, predictions = self._case(Path(temporary))
            completion = finalize_raw_predictions(
                jobs, pairs, predictions, formal=False
            )
            self.assertFalse(completion["formal_run"])
            self.assertEqual(completion["completed_jobs"], 2)
            completion_path = predictions / "completion.json"
            manifest_path = predictions / "prediction_manifest.csv"
            self.assertTrue(completion_path.is_file())
            self.assertEqual(
                completion["prediction_manifest_sha256"], sha256_file(manifest_path)
            )
            with manifest_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["pair_id"] for row in rows], ["pair_1", "pair_2"])
            with self.assertRaises(FileExistsError):
                finalize_raw_predictions(jobs, pairs, predictions, formal=False)

    def test_invalid_prediction_never_publishes_completion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            jobs, pairs, predictions = self._case(Path(temporary), bad_shape=True)
            with self.assertRaisesRegex(ValueError, "shape mismatch"):
                finalize_raw_predictions(jobs, pairs, predictions, formal=False)
            self.assertFalse((predictions / "completion.json").exists())
            self.assertFalse((predictions / "prediction_manifest.csv").exists())

    def test_formal_finalizer_requires_8400_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            jobs, pairs, predictions = self._case(Path(temporary))
            with self.assertRaisesRegex(ValueError, "requires 8400 jobs"):
                finalize_raw_predictions(jobs, pairs, predictions, formal=True)
            self.assertFalse((predictions / "completion.json").exists())


if __name__ == "__main__":
    unittest.main()
