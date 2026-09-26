"""Wait for three verified Umbra SICDs, then make 7x7 ROI contact sheets.

Each site is handled as soon as its target SICD appears in the download
manifest. Existing candidates.json files are left alone, so the watcher can
be restarted after an interruption.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SELECTIONS = ROOT / "output" / "umbra_scene_selection" / "selected_diverse_scenes_v2.json"
DEFAULT_DATA = Path(r"E:\SAR_Data\Umbra\Stability_3Scenes_V2")
DEFAULT_OUTPUT = ROOT / "output" / "umbra_stability_new_3scenes"
DEFAULT_PYTHON = ROOT / ".venv-cl-sar" / "Scripts" / "python.exe"
PREVIEW_SCRIPT = ROOT / "scripts" / "preview_umbra_stability_rois.py"


def verified_sicd(site_root: Path, target: dict[str, object]) -> Path | None:
    manifest_path = site_root / "download_manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        # The downloader writes this manifest after every product. It may be
        # temporarily incomplete while another process is updating it.
        return None

    filename = Path(urlsplit(str(target["sicd"])).path).name
    expected_file = site_root / "target" / filename
    expected_bytes = int(target["sicd_bytes"])
    expected_etag = str(target["sicd_etag"]).strip('"')
    # The V2 downloader writes one verified SICD record at the top level.
    # Also accept the older downloader's products[] manifest for reuse.
    products = manifest.get("products")
    records = products if isinstance(products, list) else [manifest]
    for product in records:
        if not isinstance(product, dict):
            continue
        if isinstance(products, list) and (
            product.get("role") != "target" or product.get("product") != "sicd"
        ):
            continue
        if product.get("acquisition_id") != target["id"]:
            continue
        if product.get("etag_verified") is not True:
            continue
        if int(product.get("bytes", -1)) != expected_bytes:
            continue
        if str(product.get("etag", "")).strip('"') != expected_etag:
            continue
        if Path(str(product.get("file", ""))).resolve() != expected_file.resolve():
            continue
        if expected_file.is_file() and expected_file.stat().st_size == expected_bytes:
            return expected_file
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selections", type=Path, default=DEFAULT_SELECTIONS)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--crop-size", type=int, default=1536)
    parser.add_argument("--thumbnail", type=int, default=256)
    parser.add_argument("--grid-rows", type=int, default=7)
    parser.add_argument("--grid-cols", type=int, default=7)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    if not args.python.is_file():
        parser.error(f"Python interpreter not found: {args.python}")
    if not PREVIEW_SCRIPT.is_file():
        parser.error(f"Preview script not found: {PREVIEW_SCRIPT}")
    if min(args.crop_size, args.thumbnail, args.grid_rows, args.grid_cols, args.poll_seconds) < 1:
        parser.error("crop, thumbnail, grid, and poll values must be positive")

    payload = json.loads(args.selections.read_text(encoding="utf-8"))
    scenes = payload["selections"]
    if len(scenes) != 3:
        parser.error(f"expected three selected scenes, got {len(scenes)}")
    completed: set[str] = set()
    last_waiting: set[str] = set()
    while len(completed) < len(scenes):
        waiting: set[str] = set()
        for selection in scenes:
            site = str(selection["site"])
            if site in completed:
                continue
            output = args.output_root / site / "roi_candidates"
            if (output / "candidates.json").is_file():
                print(f"[{site}] existing candidates.json; skip", flush=True)
                completed.add(site)
                continue
            source = verified_sicd(args.data_root / site, selection["target"])
            if source is None:
                waiting.add(site)
                continue
            command = [
                str(args.python), str(PREVIEW_SCRIPT),
                "--source", str(source),
                "--output", str(output),
                "--crop-size", str(args.crop_size),
                "--thumbnail", str(args.thumbnail),
                "--grid-rows", str(args.grid_rows),
                "--grid-cols", str(args.grid_cols),
            ]
            print(f"[{site}] verified SICD ready; generating contact sheet", flush=True)
            result = subprocess.run(command, check=False)
            if result.returncode == 0 and (output / "candidates.json").is_file():
                print(f"[{site}] complete: {output / 'contact_sheet.png'}", flush=True)
                completed.add(site)
            else:
                print(f"[{site}] preview failed (exit {result.returncode}); will retry", flush=True)
        if len(completed) == len(scenes):
            break
        if waiting != last_waiting:
            print(f"waiting for verified SICD: {', '.join(sorted(waiting)) or 'none'}", flush=True)
            last_waiting = waiting
        time.sleep(args.poll_seconds)
    print(f"All {len(scenes)} contact sheets ready under {args.output_root.resolve()}", flush=True)


if __name__ == "__main__":
    main()
