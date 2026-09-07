"""Add frozen run-seed schedules to one prepared ICSPS 2026 data root."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from icsps2026_protocol import (
    SCHEDULE_FIELDS,
    build_train_schedule,
    load_protocol_config,
    read_csv_records,
    sha256_file,
    validate_train_schedule,
    write_csv_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/icsps2026_frozen_v2.json")
    parser.add_argument("--prepared-root", required=True)
    parser.add_argument("--run-seeds", nargs="+", type=int, default=(42, 43, 44))
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_protocol_config(args.config, smoke=args.smoke)
    root = Path(args.prepared_root).resolve()
    source_path = root / "source_manifest.csv"
    sources = read_csv_records(source_path)
    train_sources = [row for row in sources if str(row["split"]) == "train"]
    allowed = {int(value) for value in config["schedule"]["run_seeds"]}
    results = []
    for run_seed in args.run_seeds:
        if not args.smoke and run_seed not in allowed:
            raise ValueError(f"Formal run seed must be one of {sorted(allowed)}")
        destination = root / f"train_schedule_seed{run_seed}.csv"
        if destination.exists():
            rows = read_csv_records(destination)
        else:
            rows = build_train_schedule(
                train_sources,
                updates=int(config["schedule"]["updates"]),
                looks=config["looks"],
                run_seed=run_seed,
                synthesis_seed=int(config["seeds"]["synthesis"]),
                protocol_id=str(config["artifact_protocol_id"]),
                d4_count=int(config["schedule"]["d4_count"]),
            )
            write_csv_records(destination, SCHEDULE_FIELDS, rows)
        audit = validate_train_schedule(
            rows,
            train_sources,
            updates=int(config["schedule"]["updates"]),
            looks=config["looks"],
            run_seed=run_seed,
            synthesis_seed=int(config["seeds"]["synthesis"]),
            protocol_id=str(config["artifact_protocol_id"]),
            d4_count=int(config["schedule"]["d4_count"]),
        )
        results.append(
            {
                "run_seed": run_seed,
                "path": destination.name,
                "sha256": sha256_file(destination),
                # Persist only stable facts so rerunning verification is
                # byte-value idempotent after the schedules already exist.
                "status": "verified",
                "audit": audit,
            }
        )
    manifest = {
        "protocol_id": config["artifact_protocol_id"],
        "source_manifest": source_path.name,
        "source_manifest_sha256": sha256_file(source_path),
        "schedules": results,
    }
    output = root / "additional_schedule_hashes.json"
    if output.exists():
        with open(output, "r", encoding="utf-8") as handle:
            if json.load(handle) != manifest:
                raise FileExistsError(
                    f"Existing {output} does not match regenerated schedule metadata"
                )
    else:
        with open(output, "x", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
