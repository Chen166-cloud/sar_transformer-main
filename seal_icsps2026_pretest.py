"""Atomically seal test selections against frozen data and completed validation evidence."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from experiment_protocol import atomic_json_dump, sha256_file
from icsps2026_pretest import PROTOCOL_ID, build_binding


def _atomic_text(text: str, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite {destination}")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _assert_test_outputs_absent(run_root: Path) -> None:
    forbidden_roots = (
        run_root / "ucm",
        run_root / "real",
        run_root / "sarbm3d_jobs",
        run_root / "sarbm3d_raw_ucm",
        run_root / "paper_figures",
    )
    for root in forbidden_roots:
        if root.exists() and any(path.is_file() for path in root.rglob("*")):
            raise FileExistsError(
                f"Cannot create a pretest seal after test-derived files exist: {root}"
            )


def seal(run_root: str | Path, prep_root: str | Path, real_protocol_root: str | Path) -> dict:
    run = Path(run_root).resolve()
    prep = Path(prep_root).resolve()
    real = Path(real_protocol_root).resolve()
    records = run / "records"
    records.mkdir(parents=True, exist_ok=True)
    seal_path = records / "pretest_seal.json"
    checksum_path = records / "pretest_registration.sha256"
    timestamp_path = records / "test_unlock_utc.txt"
    for path in (seal_path, checksum_path, timestamp_path):
        if path.exists():
            raise FileExistsError(
                f"Pretest sealing is one-shot; refusing existing artifact: {path}"
            )

    _assert_test_outputs_absent(run)
    binding = build_binding(run, prep, real)
    validation_path = records / "validation_selection.csv"
    figure_path = records / "figure_selection.csv"
    checksum_text = (
        f"{binding['validation_selection_sha256']}  {validation_path.resolve()}\n"
        f"{binding['figure_selection_sha256']}  {figure_path.resolve()}\n"
    )
    sealed_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )
    created: list[Path] = []
    try:
        _atomic_text(checksum_text, checksum_path)
        created.append(checksum_path)
        _atomic_text(sealed_at + "\n", timestamp_path)
        created.append(timestamp_path)
        payload = {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "state": "sealed_before_test",
            "sealed_at_utc": sealed_at,
            "binding": binding,
            "registration_checksum_sha256": sha256_file(checksum_path),
            "unlock_timestamp_sha256": sha256_file(timestamp_path),
        }
        atomic_json_dump(payload, seal_path)
        created.append(seal_path)
    except Exception:
        # These exact files were absent above and were created only by this
        # invocation; remove a partial transaction so the user can correct the
        # inputs and retry the one-shot operation.
        for created_path in reversed(created):
            created_path.unlink(missing_ok=True)
        raise
    return {
        **payload,
        "pretest_seal": str(seal_path),
        "pretest_seal_sha256": sha256_file(seal_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--prep-root", required=True)
    parser.add_argument("--real-protocol-root", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            seal(args.run_root, args.prep_root, args.real_protocol_root),
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
