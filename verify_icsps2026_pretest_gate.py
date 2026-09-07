"""Read-only verification of the one-shot ICSPS 2026 pretest seal."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from experiment_protocol import sha256_file
from icsps2026_pretest import PROTOCOL_ID, build_binding


def _load_object(path: Path) -> dict[str, object]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Unreadable pretest seal {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Pretest seal must be a JSON object: {path}")
    return value


def _parse_utc_timestamp(text: str) -> datetime:
    if not text.endswith("Z"):
        raise ValueError("test_unlock_utc.txt must end in Z")
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError(f"Invalid UTC unlock timestamp: {text!r}") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError("test_unlock_utc.txt must contain an ISO-8601 UTC timestamp")
    if parsed > datetime.now(timezone.utc) + timedelta(minutes=5):
        raise ValueError("Pretest unlock timestamp is implausibly in the future")
    return parsed


def _declared_registration_hashes(path: Path) -> dict[Path, str]:
    declared: dict[Path, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            raise ValueError(f"Malformed pretest checksum line: {line!r}")
        digest, raw_path = parts
        resolved = Path(raw_path.lstrip("* ")).resolve()
        if resolved in declared:
            raise ValueError(f"Duplicate path in pretest checksum: {resolved}")
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(f"Non-canonical SHA-256 in pretest checksum: {digest!r}")
        declared[resolved] = digest
    return declared


def verify(run_root: str | Path) -> dict[str, object]:
    """Rebuild every semantic binding and fail closed on any mismatch."""

    root = Path(run_root).resolve()
    records = root / "records"
    validation = records / "validation_selection.csv"
    figures = records / "figure_selection.csv"
    checksum = records / "pretest_registration.sha256"
    timestamp = records / "test_unlock_utc.txt"
    seal_path = records / "pretest_seal.json"
    for path in (validation, figures, checksum, timestamp, seal_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing/non-empty pretest gate artifact: {path}")

    seal = _load_object(seal_path)
    expected_header = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "state": "sealed_before_test",
    }
    for field, expected in expected_header.items():
        if seal.get(field) != expected or type(seal.get(field)) is not type(expected):
            raise ValueError(
                f"Pretest seal {field} mismatch: expected={expected!r}, found={seal.get(field)!r}"
            )
    binding = seal.get("binding")
    if not isinstance(binding, dict):
        raise ValueError("Pretest seal lacks a binding object")
    if binding.get("run_root") != str(root):
        raise ValueError("Pretest seal is bound to a different RUN_ROOT")
    prep_text = binding.get("prep_root")
    real_text = binding.get("real_protocol_root")
    if not isinstance(prep_text, str) or not prep_text or not Path(prep_text).is_absolute():
        raise ValueError("Pretest seal has no absolute PREP_ROOT binding")
    if not isinstance(real_text, str) or not real_text or not Path(real_text).is_absolute():
        raise ValueError("Pretest seal has no absolute REAL_PROTOCOL_ROOT binding")

    rebuilt = build_binding(root, Path(prep_text), Path(real_text))
    if rebuilt != binding:
        raise ValueError("Pretest semantic binding mismatch; data, selections, or validation changed")

    declared = _declared_registration_hashes(checksum)
    expected_paths = {validation.resolve(), figures.resolve()}
    if set(declared) != expected_paths:
        raise ValueError(
            "Pretest checksum must cover exactly validation_selection.csv and "
            f"figure_selection.csv; found={sorted(map(str, declared))}"
        )
    expected_hashes = {
        validation.resolve(): str(binding["validation_selection_sha256"]),
        figures.resolve(): str(binding["figure_selection_sha256"]),
    }
    for path, expected in expected_hashes.items():
        if declared[path] != expected or sha256_file(path) != expected:
            raise ValueError(f"Pretest registration SHA-256 mismatch: {path}")
    if sha256_file(checksum) != seal.get("registration_checksum_sha256"):
        raise ValueError("Pretest registration checksum file does not match the seal")

    timestamp_text = timestamp.read_text(encoding="utf-8").strip()
    _parse_utc_timestamp(timestamp_text)
    if timestamp_text != seal.get("sealed_at_utc"):
        raise ValueError("Unlock timestamp does not match sealed_at_utc")
    if sha256_file(timestamp) != seal.get("unlock_timestamp_sha256"):
        raise ValueError("Unlock timestamp file does not match the seal")

    baseline = binding["strongest_non_ours_baseline"]
    assert isinstance(baseline, dict)  # established by build_binding
    return {
        "ok": True,
        "protocol_id": PROTOCOL_ID,
        "run_root": str(root),
        "strongest_non_ours_baseline": baseline["method"],
        "registered_ucm_rows": binding["registered_ucm_rows"],
        "registered_real_rows": binding["registered_real_rows"],
        "test_unlock_utc": timestamp_text,
        "registration_sha256": sha256_file(checksum),
        "pretest_seal": str(seal_path),
        "pretest_seal_sha256": sha256_file(seal_path),
        "validation_selection_sha256": binding["validation_selection_sha256"],
        "figure_selection_sha256": binding["figure_selection_sha256"],
        "ucm_test_manifest_sha256": binding["ucm_test_manifest_sha256"],
        "real_qc_manifest_sha256": binding["real_qc_manifest_sha256"],
        "real_qc_csv_sha256": binding["real_qc_csv_sha256"],
        "real_use_gt16": binding["real_use_gt16"],
        "real_evaluation_scope": binding["real_evaluation_scope"],
        "real_valid_quasi_reference": binding["real_valid_quasi_reference"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.run_root), indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
