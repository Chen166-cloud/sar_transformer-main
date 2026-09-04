"""Audit the complete global-L dataset and persist reproducibility evidence."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import csv
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from synthetic_manifest import sha256_file, verify_paired_sar_manifest


def main():
    root = ROOT / "datasets" / "NWPU_RESISC45_SAR_global_L_v2"
    manifest = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
    plan = json.loads((root / "generation_plan.json").read_text(encoding="utf-8"))
    old = json.loads((ROOT / "datasets" / "NWPU_RESISC45_SAR_intensity_v1" / "global_L_v2_matched_split_manifest.json").read_text(encoding="utf-8"))
    records = plan["records"]
    assert len(records) == 50400
    assert len({r["output_relative"] for r in records}) == 50400
    assert len({r["sample_seed"] for r in records}) == 50400
    train = [r for r in records if r["split"] == "train"]
    val = [r for r in records if r["split"] == "val"]
    counts = Counter((r["class"], r["global_L"]) for r in train)
    assert len(counts) == 180 and set(counts.values()) == {140}
    validation_sources = defaultdict(set)
    for record in val:
        validation_sources[record["source_relative"]].add(record["global_L"])
    assert len(validation_sources) == 6300
    assert all(values == {1, 2, 4, 8} for values in validation_sources.values())
    for split in ("train", "val"):
        sources = {r["source_relative"] for r in records if r["split"] == split}
        assert sources == set(manifest["source_files"][split])
        assert sources == set(old["source_files"][split])
    assert not set(manifest["source_files"]["train"]) & set(manifest["source_files"]["val"])
    content_hashes = {split: {manifest["source_sha256"][source] for source in manifest["source_files"][split]} for split in ("train", "val")}
    assert not content_hashes["train"] & content_hashes["val"]
    assert manifest["source_sha256"] == old["source_sha256"]
    for record in records:
        path = record["output_relative"]
        assert manifest["global_L_by_file"][path] == record["global_L"]
        assert manifest["sample_seeds"][path] == record["sample_seed"]
    for split, filename in (("train", "train_l_assignment.csv"), ("val", "validation_l_sets.csv")):
        assert sha256_file(root / filename) == manifest["assignment_files"][filename]
        with (root / filename).open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        expected = [r for r in records if r["split"] == split]
        assert len(rows) == len(expected)
        for row, record in zip(rows, expected):
            assert all(str(record[key]) == value for key, value in row.items())
    assert sha256_file(ROOT / manifest["generator"]) == manifest["generator_sha256"]
    print("Plan, source split, L assignments and seeds verified; hashing all 50400 MAT files...", flush=True)
    result = verify_paired_sar_manifest(root, manifest, verify_files=False, require_exact_partition=False)
    # The generated manifest has already passed lexical path validation above.
    # os.walk avoids per-file Windows realpath resolution during a full audit.
    discovered = set()
    for directory, _, names in os.walk(root):
        for name in names:
            if name.lower().endswith(".mat"):
                discovered.add((Path(directory) / name).relative_to(root).as_posix())
    expected = set(manifest["file_sha256"])
    assert discovered == expected
    for index, relative in enumerate(sorted(expected), 1):
        assert sha256_file(root / relative) == manifest["file_sha256"][relative], relative
        if index % 2000 == 0 or index == len(expected):
            print(f"hashes_verified={index}/{len(expected)}", flush=True)
    result["hashes_checked"] = len(expected)
    result.update({
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_per_class_per_L": 140,
        "train_pairs_per_L": 6300,
        "validation_pairs_per_L": 6300,
        "validation_sources_each_have_all_four_L": True,
        "source_overlap": 0,
        "source_content_hash_overlap": 0,
        "source_split_identical_to_retained_mixed_L_dataset": True,
        "source_hashes_identical_to_retained_mixed_L_dataset": True,
        "assignment_tables_match_plan": True,
        "unique_sample_seeds": 50400,
        "manifest_sha256": sha256_file(root / "dataset_manifest.json"),
        "generator_sha256": manifest["generator_sha256"],
    })
    output = root / "verification_report.json"
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
