"""Assemble the repaired manifest; the separate audit hashes every final file."""
from datetime import datetime, timezone
from pathlib import Path
import copy
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    root = ROOT / "NWPU_RESISC45_SAR_global_L_v2"
    donor_root = ROOT / "NWPU_RESISC45_SAR_global_L_v1"
    donor = json.loads((donor_root / "dataset_manifest.json").read_text(encoding="utf-8"))
    donor_plan = json.loads((donor_root / "generation_plan.json").read_text(encoding="utf-8"))
    donor_records = {r["output_relative"]: r for r in donor_plan["records"]}
    plan = json.loads((root / "generation_plan.json").read_text(encoding="utf-8"))
    manifest = copy.deepcopy(donor)
    manifest.update({
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": plan["split"],
        "L_assignment": plan["L_assignment"],
        "robustness_dataset": plan["robustness_dataset"],
        "assembly": {
            "script": "tools/finalize_nwpu_global_l_v2.py",
            "script_sha256": sha(Path(__file__)),
            "donor_manifest_sha256": sha(donor_root / "dataset_manifest.json"),
            "hardlinked_unchanged_pairs": 50390,
            "new_pairs": 10,
        },
    })
    manifest["files"] = {split: [r["output_relative"] for r in plan["records"] if r["split"] == split] for split in ("train", "val")}
    manifest["source_files"] = {split: sorted({r["source_relative"] for r in plan["records"] if r["split"] == split}) for split in ("train", "val")}
    manifest["global_L_by_file"] = {r["output_relative"]: r["global_L"] for r in plan["records"]}
    manifest["sample_seeds"] = {r["output_relative"]: r["sample_seed"] for r in plan["records"]}
    manifest["validation_sets"] = {f"L{looks}": {
        "global_L": looks, "count": 6300,
        "files": [r["output_relative"] for r in plan["records"] if r["split"] == "val" and r["global_L"] == looks],
    } for looks in (1, 2, 4, 8)}
    hashes = {}
    inherited = 0
    for record in plan["records"]:
        path = record["output_relative"]
        if donor_records.get(path) == record:
            hashes[path] = donor["file_sha256"][path]
            inherited += 1
        else:
            hashes[path] = sha(root / path)
    assert inherited == 50390
    manifest["file_sha256"] = hashes
    manifest["assignment_files"] = {name: sha(root / name) for name in ("train_l_assignment.csv", "validation_l_sets.csv")}
    with (root / "dataset_manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    print("Final v2 manifest assembled: 50390 unchanged pairs + 10 regenerated pairs.")


if __name__ == "__main__":
    main()
