"""Resume selected Umbra SICD downloads and record verified source identities.

The catalogue URLs stay HTTPS in the selection manifest.  A public HTTP
transport can be used for unreliable routes; S3 ETag and SHA-256 are checked
before a SICD is marked complete.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
import argparse
import json
import os
import time

from download_umbra_stability_experiment import (
    PINNED_TARGET_OBJECTS,
    download_object,
    filename_from_url,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SELECTIONS = ROOT / "output" / "umbra_scene_selection" / "selected_diverse_scenes_v2.json"
DEFAULT_DESTINATION = Path(r"E:\SAR_Data\Umbra\Stability_3Scenes_V2")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selections", type=Path, default=DEFAULT_SELECTIONS)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--site", action="append", help="exact site directory name; repeatable")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--transport", choices=("accelerate", "http", "https"), default="accelerate")
    args = parser.parse_args()
    if not 1 <= args.workers <= 24:
        parser.error("--workers must be in [1, 24]")

    selections = json.loads(args.selections.read_text(encoding="utf-8"))["selections"]
    if args.site:
        selected_sites = set(args.site)
        selections = [item for item in selections if item["site"] in selected_sites]
        if selected_sites - {item["site"] for item in selections}:
            parser.error(f"unknown site(s): {sorted(selected_sites - {item['site'] for item in selections})}")
    for selection in selections:
        site = str(selection["site"])
        source_url = str(selection["target"]["sicd"])
        if args.transport == "accelerate":
            parts = urlsplit(source_url)
            if parts.netloc != "umbra-open-data-catalog.s3.us-west-2.amazonaws.com":
                raise ValueError(f"unexpected provider host: {parts.netloc}")
            transfer_url = urlunsplit(
                ("https", "umbra-open-data-catalog.s3-accelerate.amazonaws.com", parts.path, parts.query, parts.fragment)
            )
        elif args.transport == "http":
            transfer_url = source_url.replace("https://", "http://", 1)
        else:
            transfer_url = source_url
        PINNED_TARGET_OBJECTS[filename_from_url(source_url)] = (
            int(selection["target"]["sicd_bytes"]),
            str(selection["target"]["sicd_etag"]),
        )
        site_root = args.destination / site
        target_root = site_root / "target"
        print(f"[{site}] {source_url}", flush=True)
        result = download_object(transfer_url, target_root, args.workers)
        record = {
            "site": site,
            "acquisition_id": selection["target"]["id"],
            "date": selection["target"]["date"],
            "provider_url": source_url,
            "transfer_url": result["url"],
            "file": result["file"],
            "bytes": result["bytes"],
            "etag": result["etag"],
            "etag_verified": result["etag_verified"],
            "sha256": result["sha256"],
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        site_root.mkdir(parents=True, exist_ok=True)
        output = site_root / "download_manifest.json"
        temporary = output.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(temporary, output)
        print(f"[{site}] verified {result['bytes']} bytes, SHA-256 {result['sha256']}", flush=True)


if __name__ == "__main__":
    main()
