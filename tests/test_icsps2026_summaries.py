from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experiment_protocol import sha256_file
from summarize_icsps2026_multiseed import average_seed_rows
from summarize_icsps2026_real import main as summarize_real_main, read_and_validate
from summarize_icsps2026_synthetic import (
    parse_comparison,
    resolve_comparison_specs,
    validate_input_aggregate,
    validate_input_identity,
)


FIELDS = (
    "protocol_id", "method", "variant", "seed", "adaptation",
    "sample_id", "parent_id", "split", "formal_run", "has_valid_gt16",
    "ratio_mean_bias", "ratio_acf_sidelobe_energy", "sobel_gc_noisy_output",
    "qpsnr_gt16", "qssim_gt16",
)


def pretest_credential() -> dict[str, object]:
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


class MultiSeedSummaryTests(unittest.TestCase):
    def test_pair_average_preserves_frozen_identity(self) -> None:
        seed_rows = []
        for offset in (0.0, 1.0, 2.0):
            seed_rows.append(
                [{
                    "pair_id": "pair_1", "source_id": "source_1",
                    "class_name": "harbor", "L": "4", "psnr": 30.0 + offset,
                    "ssim": 0.8 + offset * 0.01,
                }]
            )
        averaged = average_seed_rows(seed_rows)
        self.assertEqual(len(averaged), 1)
        self.assertAlmostEqual(float(averaged[0]["psnr"]), 31.0)
        self.assertAlmostEqual(float(averaged[0]["ssim"]), 0.81)


def write_formal_real(
    root: Path, *, target: bool, method: str | None = None
) -> Path:
    root.mkdir()
    actual_method = method or ("ours" if target else "transsar_v2")
    variant = "full" if actual_method == "ours" else ""
    rows = []
    for index in range(592):
        valid_q = False
        rows.append(
            {
                "protocol_id": "ICSPS26-FROZEN-v2",
                "method": actual_method,
                "variant": variant,
                "seed": 42,
                "adaptation": "base",
                "sample_id": f"sample_{index:03d}",
                "parent_id": f"parent_{index // 4:03d}",
                "split": "test",
                "formal_run": True,
                "has_valid_gt16": valid_q,
                "ratio_mean_bias": 0.1 if target else 0.2,
                "ratio_acf_sidelobe_energy": 0.2 if target else 0.3,
                "sobel_gc_noisy_output": 0.6 if target else 0.5,
                "qpsnr_gt16": (31.0 if target else 30.0) if valid_q else "nan",
                "qssim_gt16": (0.91 if target else 0.90) if valid_q else "nan",
            }
        )
    csv_path = root / "real_per_patch.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    aggregate = {
        "protocol_id": "ICSPS26-FROZEN-v2",
        "formal_run": True,
        "split": "test",
        "num_patches": 592,
        "num_parent_clusters": 148,
        "num_valid_gt16_pairs": 0,
        "method": actual_method,
        "variant": variant or None,
        "seed": 42,
        "adaptation": "base",
        "qc_manifest_sha256": "a" * 64,
        "qc_csv_sha256": "a" * 64,
        "parent_cluster_bootstrap": {},
        "artifacts": {
            "real_per_patch_csv": csv_path.name,
            "real_per_patch_sha256": sha256_file(csv_path),
        },
        "pretest_gate": pretest_credential(),
    }
    (root / "aggregate.json").write_text(
        json.dumps(aggregate), encoding="utf-8"
    )
    return csv_path


class RealSummaryTests(unittest.TestCase):
    def test_paired_parent_bootstrap_and_no_gt16_scope(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = write_formal_real(root / "target", target=True)
            baseline = write_formal_real(root / "baseline", target=False)
            output = root / "summary"
            arguments = [
                "summarize_icsps2026_real.py",
                "--input", f"target={target}",
                "--input", f"baseline={baseline}",
                "--compare", "target:baseline",
                "--bootstrap-samples", "20",
                "--output-dir", str(output),
            ]
            with mock.patch.object(sys, "argv", arguments):
                self.assertEqual(summarize_real_main(), 0)
            with (output / "real_paired_parent_bootstrap.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 3)
            by_metric = {row["metric"]: row for row in rows}
            self.assertAlmostEqual(float(by_metric["ratio_mean_bias"]["advantage"]), 0.1)
            self.assertEqual(by_metric["ratio_mean_bias"]["advantage_definition"], "baseline_minus_target")
            self.assertAlmostEqual(float(by_metric["sobel_gc_noisy_output"]["advantage"]), 0.1)
            self.assertEqual(
                by_metric["sobel_gc_noisy_output"]["positive_favors"],
                "not_defined_diagnostic",
            )
            self.assertNotIn("qpsnr_gt16", by_metric)
            provenance = json.loads(
                (output / "summary_provenance.json").read_text(encoding="utf-8")
            )
            self.assertNotIn("positive_advantage_always_favors_target", provenance)
            self.assertEqual(
                provenance["inputs"]["target"]["verified_identity"]["method"], "ours"
            )
            self.assertEqual(provenance["real_evaluation_scope"], "no_reference_only")
            self.assertEqual(len(provenance["skipped_q_metrics"]), 2)

    def test_noisy_input_is_forbidden_in_real_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = write_formal_real(root / "target", target=True)
            noisy = write_formal_real(root / "noisy", target=False, method="noisy")
            arguments = [
                "summarize_icsps2026_real.py",
                "--input", f"target={target}",
                "--input", f"noisy={noisy}",
                "--compare", "target:noisy",
                "--bootstrap-samples", "20",
                "--output-dir", str(root / "summary"),
            ]
            with mock.patch.object(sys, "argv", arguments):
                with self.assertRaisesRegex(ValueError, "Noisy input cannot"):
                    summarize_real_main()

    def test_real_summary_rejects_mixed_row_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = write_formal_real(Path(temporary) / "result", target=True)
            with path.open("r", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["variant"] = "log_only"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            aggregate_path = path.parent / "aggregate.json"
            aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
            aggregate["artifacts"]["real_per_patch_sha256"] = sha256_file(path)
            aggregate_path.write_text(json.dumps(aggregate), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "mixed method/variant/seed/adaptation"):
                read_and_validate("label_only", path)

    def test_synthetic_comparison_parser(self) -> None:
        self.assertEqual(parse_comparison("full:transsar"), ("full", "transsar"))
        self.assertEqual(
            resolve_comparison_specs(
                [("full", "no_fdr"), ("full", "log_only")],
                "full", "transsar",
            ),
            [
                ("full", "no_fdr"),
                ("full", "log_only"),
                ("full", "transsar"),
            ],
        )

    def test_synthetic_identity_is_from_rows_not_key(self) -> None:
        rows = [
            {"method": "ours", "variant": "full", "seed": "42"},
            {"method": "ours", "variant": "full", "seed": "42"},
        ]
        self.assertEqual(
            validate_input_identity("misleading_key", rows),
            {"method": "ours", "variant": "full", "seed": 42},
        )
        rows[1]["variant"] = "log_only"
        with self.assertRaisesRegex(ValueError, "mixed method/variant/seed"):
            validate_input_identity("misleading_key", rows)

    def test_synthetic_summary_checks_aggregate_hash_and_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            per_image = root / "per_image.csv"
            per_image.write_text("method,variant,seed\nours,full,42\n", encoding="utf-8")
            aggregate = {
                "method": "ours",
                "variant": "full",
                "seed": 42,
                "per_image_sha256": sha256_file(per_image),
            }
            (root / "aggregate.json").write_text(json.dumps(aggregate), encoding="utf-8")
            identity = {"method": "ours", "variant": "full", "seed": 42}
            validate_input_aggregate(
                "label_only", per_image, identity, formal_required=False
            )
            per_image.write_text("method,variant,seed\nours,log_only,42\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                validate_input_aggregate(
                    "label_only", per_image, identity, formal_required=False
                )


if __name__ == "__main__":
    unittest.main()
