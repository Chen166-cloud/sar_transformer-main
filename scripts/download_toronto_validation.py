"""Download and verify paired Toronto validation TIFFs from Mendeley Data v2."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from urllib.request import Request, urlopen


DATASET = "2xf5v5pwkr"
VERSION = 2
BASE = f"https://data.mendeley.com/public-api/datasets/{DATASET}/files"
DESTINATION = Path(r"E:\SAR_Data\Toronto_Paired_SAR\validation_full")
PREVIEW = Path(r"E:\SAR_Data\Toronto_Paired_SAR\validation_preview")
FOLDERS = {
    "Noisy_val": "b56730a7-6091-4b6e-9714-075862762917",
    "GTruth_val": "b5a1fb7e-d358-47be-a8e0-3708a8c94121",
}


def get_bytes(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "TorontoSARBenchmark/1.0"})
    with urlopen(request, timeout=45) as response:
        return response.read()


def list_folder(folder_id: str) -> list[dict[str, object]]:
    url = f"{BASE}?folder_id={folder_id}&version={VERSION}&%24start=0&%24limit=1000"
    entries = json.loads(get_bytes(url))
    if not isinstance(entries, list) or len(entries) != 100:
        raise ValueError(f"Expected 100 files in {folder_id}, found {len(entries)}")
    return entries


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download_one(kind: str, folder_id: str, entry: dict[str, object]) -> tuple[str, str, int]:
    name = str(entry["filename"])
    if not name.endswith(".tiff") or Path(name).name != name:
        raise ValueError(f"Unexpected filename: {name}")
    details = entry["content_details"]
    if not isinstance(details, dict):
        raise ValueError(f"No content details: {name}")
    expected = str(details["sha256_hash"]).lower()
    size = int(details["size"])
    destination = DESTINATION / kind / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size == size and sha256(destination) == expected:
        return kind, name, size
    preview = PREVIEW / folder_id / name
    if preview.exists() and preview.stat().st_size == size and sha256(preview) == expected:
        shutil.copy2(preview, destination)
        return kind, name, size
    url = str(details["download_url"])
    for attempt in range(4):
        try:
            payload = get_bytes(url)
            if len(payload) != size or hashlib.sha256(payload).hexdigest() != expected:
                raise ValueError(f"Size or SHA-256 mismatch: {kind}/{name}")
            temporary = destination.with_suffix(destination.suffix + ".part")
            temporary.write_bytes(payload)
            os.replace(temporary, destination)
            return kind, name, size
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def main() -> None:
    manifest: dict[str, object] = {
        "dataset": "SAR despeckling filters dataset",
        "version": VERSION,
        "doi": "10.17632/2xf5v5pwkr.2",
        "source_url": f"https://data.mendeley.com/datasets/{DATASET}/{VERSION}",
        "folders": {},
    }
    tasks: list[tuple[str, str, dict[str, object]]] = []
    for kind, folder_id in FOLDERS.items():
        entries = list_folder(folder_id)
        manifest["folders"][kind] = {
            "id": folder_id,
            "files": [
                {
                    "filename": item["filename"],
                    "id": item["id"],
                    "sha256": item["content_details"]["sha256_hash"],
                    "size": item["content_details"]["size"],
                    "download_url": item["content_details"]["download_url"],
                }
                for item in entries
            ],
        }
        tasks.extend((kind, folder_id, item) for item in entries)

    noisy_names = {entry["filename"] for entry in manifest["folders"]["Noisy_val"]["files"]}
    gt_names = {entry["filename"] for entry in manifest["folders"]["GTruth_val"]["files"]}
    if noisy_names != gt_names:
        raise ValueError("Noisy and reference validation filenames are not paired")

    count = 0
    total = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(download_one, *task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            kind, name, size = future.result()
            count += 1
            total += size
            if count % 20 == 0 or count == len(tasks):
                print(f"Verified {count}/{len(tasks)} files ({total / 1e6:.1f} MB): {kind}/{name}", flush=True)

    manifest_path = DESTINATION / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(manifest_path)


if __name__ == "__main__":
    main()
