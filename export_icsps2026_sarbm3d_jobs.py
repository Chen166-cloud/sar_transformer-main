"""Export a portable job manifest for the authors' MATLAB SAR-BM3D package."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from finalize_icsps2026_sarbm3d_raw import (
    SARBM3D_DOMAIN_CONVERSION,
    SARBM3D_INPUT_DOMAIN,
    SARBM3D_OUTPUT_DOMAIN,
)
from icsps2026_protocol import read_csv_records, resolve_portable, sha256_file
from verify_icsps2026_pretest_gate import verify as verify_pretest_gate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-pair-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--run-root",
        help="Required pretest-registration root when exporting UCM jobs",
    )
    parser.add_argument("--nonformal-smoke", action="store_true")
    parser.add_argument("--max-pairs", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_pairs < 0:
        raise ValueError("--max-pairs must be non-negative")
    if args.max_pairs and not args.nonformal_smoke:
        raise ValueError("--max-pairs requires --nonformal-smoke")
    root = Path(args.fixed_pair_root).resolve()
    rows = read_csv_records(args.manifest)
    datasets = {row.get("dataset", "").strip() for row in rows}
    pretest_gate = None
    if "UCMerced_LandUse" in datasets:
        if datasets != {"UCMerced_LandUse"}:
            raise ValueError(f"SAR-BM3D job manifest mixes datasets: {sorted(datasets)}")
        if not args.run_root:
            raise ValueError("Every UCM SAR-BM3D export requires --run-root")
        pretest_gate = verify_pretest_gate(args.run_root)
        if sha256_file(args.manifest) != pretest_gate["ucm_test_manifest_sha256"]:
            raise ValueError("UCM manifest does not match the pretest-seal binding")
    if not args.nonformal_smoke and len(rows) not in (900, 8400):
        raise ValueError(f"Formal SAR-BM3D export expects 900 or 8400 pairs, found {len(rows)}")
    if args.max_pairs:
        rows = rows[: args.max_pairs]
    jobs = []
    for row in rows:
        mat_path = resolve_portable(root, row["mat_path"])
        if sha256_file(mat_path) != row["mat_sha256"]:
            raise AssertionError(f"Pair MAT hash mismatch: {mat_path}")
        jobs.append(
            {
                "pair_id": row["pair_id"],
                "mat_path": row["mat_path"],
                "noisy_field": row["noisy_field"],
                "L": int(row["global_L"]),
                "input_mat_sha256": row["mat_sha256"],
                "output_relative_path": f"predictions/{row['pair_id']}.mat",
            }
        )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    with open(output, "x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(jobs[0]))
        writer.writeheader()
        writer.writerows(jobs)
    metadata = {
        "protocol_id": "ICSPS26-FROZEN-v2",
        "formal_run": not args.nonformal_smoke,
        "jobs": len(jobs),
        "pair_manifest_sha256": sha256_file(args.manifest),
        "job_manifest_sha256": sha256_file(output),
        "sarbm3d_version": "1.0 (2013-07-31)",
        "look_mode": "oracle nominal L",
        "source_pair_domain": "linear_normalized_intensity_0_1",
        "sarbm3d_input_domain": SARBM3D_INPUT_DOMAIN,
        "scored_output_domain": SARBM3D_OUTPUT_DOMAIN,
        "domain_conversion": SARBM3D_DOMAIN_CONVERSION,
        "license": "GRIP-UNINA nonprofit-only closed license; user acceptance required",
        "pretest_gate": pretest_gate,
    }
    with open(output.with_suffix(output.suffix + ".json"), "x", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
