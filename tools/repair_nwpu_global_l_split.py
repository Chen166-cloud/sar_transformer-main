"""Deterministically repair exact-content leakage without deleting any data.

Unchanged MAT files are hard-linked into v2. Only swapped sources require new
noise realizations. Existing directories/manifests remain intact.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import copy
import csv
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parents[1]


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed(base, namespace):
    return int.from_bytes(hashlib.sha256(f"{base}:{namespace}".encode()).digest()[:8], "little")


def main():
    original_root = ROOT / "NWPU_RESISC45_SAR_global_L_v1"
    output_root = ROOT / "NWPU_RESISC45_SAR_global_L_v2"
    original_plan = json.loads((original_root / "generation_plan.json").read_text(encoding="utf-8"))
    plan = copy.deepcopy(original_plan)
    records = {r["output_relative"]: r for r in plan["records"]}
    source_records = defaultdict(list)
    hash_sources = defaultdict(set)
    for record in records.values():
        source_records[record["source_relative"]].append(record)
        hash_sources[record["source_sha256"]].add(record["source_relative"])
    split_for = {source: rows[0]["split"] for source, rows in source_records.items()}
    cross_hashes = sorted(
        value for value, sources in hash_sources.items()
        if {split_for[source] for source in sources} == {"train", "val"}
    )
    swaps = []
    changed_sources = set()
    for content_hash in cross_hashes:
        for moved_to_train in sorted(hash_sources[content_hash]):
            if split_for[moved_to_train] != "val":
                continue
            class_name = source_records[moved_to_train][0]["class"]
            candidates = [
                source for source, rows in source_records.items()
                if split_for[source] == "train" and rows[0]["class"] == class_name
                and len(hash_sources[rows[0]["source_sha256"]]) == 1
                and source not in changed_sources
            ]
            moved_to_val = min(candidates, key=lambda source: (seed(42, f"deduplicate:{content_hash}:{source}"), source))
            old_train = source_records[moved_to_val][0]
            looks = old_train["global_L"]
            for source in (moved_to_train, moved_to_val):
                for old_record in source_records[source]:
                    records.pop(old_record["output_relative"])
            def new_record(source, split, look):
                record = copy.deepcopy(source_records[source][0])
                record.update({
                    "split": split, "global_L": look,
                    "output_relative": f"{split}/L{look}/{class_name}/{Path(source).stem}.mat",
                    "sample_seed": seed(plan["synthesis_seed"], f"{split}:{source}:L{look}"),
                })
                records[record["output_relative"]] = record
            new_record(moved_to_train, "train", looks)
            for look in (1, 2, 4, 8):
                new_record(moved_to_val, "val", look)
            split_for[moved_to_train] = "train"
            split_for[moved_to_val] = "val"
            changed_sources.update((moved_to_train, moved_to_val))
            swaps.append({"duplicate_hash": content_hash, "to_train": moved_to_train, "to_val": moved_to_val, "transferred_training_L": looks})
    assert len(records) == 50400
    train_counts = Counter((r["class"], r["global_L"]) for r in records.values() if r["split"] == "train")
    assert len(train_counts) == 180 and set(train_counts.values()) == {140}
    assert all(len({split_for[s] for s in sources}) == 1 for sources in hash_sources.values())
    plan["records"] = sorted(records.values(), key=lambda r: (r["split"], r["class"], r["source_relative"], r["global_L"]))
    plan["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    plan["split"].update({
        "method": "per-class stratified 8:2 with deterministic exact-content-group repair",
        "content_hash_overlap": 0,
        "repair_seed": 42,
        "repair_swaps": swaps,
        "repair_script": "tools/repair_nwpu_global_l_split.py",
        "repair_script_sha256": file_hash(Path(__file__)),
        "initial_plan_sha256": file_hash(original_root / "generation_plan.json"),
    })
    plan["L_assignment"]["repair_policy"] = "Transfer the replaced training source's L to the incoming source; keep every class/L quota unchanged"
    plan["robustness_dataset"]["matched_split_manifest"] = "../NWPU_RESISC45_SAR_intensity_v1/global_L_v2_matched_split_manifest.json"
    output_root.mkdir(exist_ok=False)
    with (output_root / "generation_plan.json").open("x", encoding="utf-8") as handle:
        json.dump(plan, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    fields = ["split", "class", "source_relative", "output_relative", "global_L", "sample_seed"]
    for split, filename in (("train", "train_l_assignment.csv"), ("val", "validation_l_sets.csv")):
        with (output_root / filename).open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows({key: r[key] for key in fields} for r in plan["records"] if r["split"] == split)
    linked = 0
    for record in plan["records"]:
        if record["source_relative"] in changed_sources:
            continue
        source = original_root / record["output_relative"]
        destination = output_root / record["output_relative"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.link(source, destination)
        linked += 1
        if linked % 10000 == 0:
            print(f"hardlinked={linked}", flush=True)
    old_root = ROOT / "NWPU_RESISC45_SAR_intensity_v1"
    robustness = json.loads((old_root / "dataset_manifest.json").read_text(encoding="utf-8"))
    old_plan = json.loads((old_root / "generation_plan.json").read_text(encoding="utf-8"))
    robustness["files"] = {
        split: sorted(r["output_relative"] for r in old_plan["records"] if split_for[r["source_relative"]] == split)
        for split in ("train", "val")
    }
    robustness["source_files"] = {
        split: sorted(source for source, actual_split in split_for.items() if actual_split == split)
        for split in ("train", "val")
    }
    robustness["split"] = copy.deepcopy(plan["split"])
    robustness["role"] = "Retained mixed-L robustness-only data, matched to the content-disjoint global-L v2 split; disk paths unchanged"
    with (old_root / "global_L_v2_matched_split_manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(robustness, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"hardlinked": linked, "new_pairs_needed": 50400-linked, "swaps": swaps, "content_hash_overlap": 0}, indent=2), flush=True)


if __name__ == "__main__":
    main()
