"""Inspect small overviews of Umbra GEC products using HTTP byte ranges.

This downloads only the TIFF directory and low-resolution overview tiles; the
SICD files used for experiments are selected and downloaded separately.
"""

from __future__ import annotations

from collections import OrderedDict
from io import RawIOBase
from pathlib import Path
import argparse
import json
import math
import urllib.request
import urllib.parse

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tifffile

Image.MAX_IMAGE_PIXELS = None


PLACES = {
    "Hamburg": (53.5, 9.9),
    "Rotterdam": (51.9, 4.3),
    "Antwerp": (51.3, 4.4),
    "Long_Beach": (33.7, -118.2),
    "Singapore": (1.3, 103.8),
    "Hong_Kong": (22.3, 114.1),
    "Dubai": (25.0, 55.1),
    "Melbourne": (-37.8, 144.9),
    "Sydney": (-33.9, 150.7),
}


class RemoteRangeFile(RawIOBase):
    def __init__(self, url: str, block_size: int = 512 * 1024):
        self.url = urllib.parse.quote(url, safe=":/?=&")
        self.block_size = block_size
        self.position = 0
        self.cache: OrderedDict[int, bytes] = OrderedDict()
        req = urllib.request.Request(self.url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=30) as response:
            self.length = int(response.headers["Content-Range"].split("/")[-1])

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            self.position = offset
        elif whence == 1:
            self.position += offset
        elif whence == 2:
            self.position = self.length + offset
        else:
            raise ValueError(whence)
        return self.position

    def _block(self, index: int) -> bytes:
        if index not in self.cache:
            start = index * self.block_size
            end = min(start + self.block_size, self.length) - 1
            print(f"range {start}-{end}", flush=True)
            req = urllib.request.Request(
                self.url, headers={"Range": f"bytes={start}-{end}"}
            )
            with urllib.request.urlopen(req, timeout=60) as response:
                result = response.read()
            if len(result) != end - start + 1:
                raise IOError(f"short HTTP range {start}-{end}: {len(result)}")
            self.cache[index] = result
            if len(self.cache) > 24:
                self.cache.popitem(last=False)
        self.cache.move_to_end(index)
        return self.cache[index]

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            raise IOError("unbounded remote TIFF read refused")
        size = max(0, min(size, self.length - self.position))
        parts = []
        remaining = size
        while remaining:
            index, offset = divmod(self.position, self.block_size)
            block = self._block(index)
            take = min(remaining, len(block) - offset)
            parts.append(block[offset : offset + take])
            self.position += take
            remaining -= take
        return b"".join(parts)


def midpoint(feature: dict) -> tuple[float, float]:
    ring = feature["geometry"]["coordinates"][0]
    points = ring[:-1] if ring[0] == ring[-1] else ring
    return sum(x[1] for x in points) / len(points), sum(x[0] for x in points) / len(points)


def candidates(features: list[dict], site: str, count: int) -> list[dict]:
    latitude, longitude = PLACES[site]
    rows = []
    for feature in features:
        prop = feature["properties"]
        if prop.get("provider") != "umbra" or "SICD" not in prop.get("products", {}):
            continue
        if prop.get("resolution") not in (0.25, 0.35):
            continue
        if prop.get("polarization") != "['VV']":
            continue
        lat, lon = midpoint(feature)
        if round(lat, 1) != latitude or round(lon, 1) != longitude:
            continue
        incidence = float(prop.get("incidence_angle") or 0)
        score = abs(incidence - 42) + abs(lat - latitude) * 5 + abs(lon - longitude) * 5
        rows.append((score, feature))
    rows.sort(key=lambda item: item[0])
    # Sample across geometry while keeping a reasonably midrange incidence.
    pool = rows[: max(count * 6, count)]
    pool.sort(key=lambda item: item[1]["properties"]["date"])
    if len(pool) <= count:
        return [item[1] for item in pool]
    indices = np.linspace(0, len(pool) - 1, count, dtype=int)
    return [pool[int(i)][1] for i in indices]


def preview(url: str) -> tuple[Image.Image, int, int]:
    remote = RemoteRangeFile(url)
    with tifffile.TiffFile(remote) as tif:
        matches = [
            (index, page)
            for index, page in enumerate(tif.pages)
            if min(page.shape) >= 500 and max(page.shape) <= 1600
        ]
        index, page = matches[-1] if matches else (len(tif.pages) - 1, tif.pages[-1])
        data = page.asarray()
    if data.ndim == 3:
        data = data[..., 0]
    lo, hi = np.percentile(data, (1, 99.7))
    data = np.rint(np.clip((data.astype(np.float32) - lo) / max(hi - lo, 1), 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(data), index, remote.length


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path("output/umbra_scene_selection/catalog_cache/scenes.geojson"))
    parser.add_argument("--output", type=Path, default=Path("output/umbra_scene_selection/new_visual_candidates"))
    parser.add_argument("--places", nargs="+", default=list(PLACES))
    parser.add_argument("--ids", nargs="+", default=None)
    parser.add_argument("--per-place", type=int, default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    features = json.loads(args.catalog.read_text(encoding="utf-8"))["features"]
    entries = []
    tiles = []
    selections = (
        [("ID_" + item[:8], next(f for f in features if f["properties"].get("id") == item)) for item in args.ids]
        if args.ids
        else [(site, feature) for site in args.places for feature in candidates(features, site, args.per_place)]
    )
    for site, feature in selections:
            prop = feature["properties"]
            try:
                image, page_index, gec_bytes = preview(prop["products"]["GEC"])
                image.thumbnail((500, 500), Image.Resampling.LANCZOS)
                canvas = Image.new("RGB", (520, 560), "white")
                canvas.paste(image.convert("RGB"), ((520 - image.width) // 2, 0))
                draw = ImageDraw.Draw(canvas)
                title = f"{site} | {prop['date']} | {prop['resolution']} m"
                draw.text((8, 510), title, fill="black")
                draw.text((8, 532), f"inc={prop['incidence_angle']} | {prop['orbit_state']} | {prop['id'][:12]}", fill="black")
                stem = f"{site}_{prop['date']}_{prop['id'][:8]}"
                image.save(args.output / f"{stem}.png")
                tiles.append(canvas)
                entry = {
                    "site": site,
                    "date": prop["date"],
                    "id": prop["id"],
                    "resolution_m": prop["resolution"],
                    "incidence_angle_deg": prop["incidence_angle"],
                    "orbit_state": prop["orbit_state"],
                    "look_dir": prop["look_dir"],
                    "gec_bytes": gec_bytes,
                    "preview_overview_page": page_index,
                    "gec": prop["products"]["GEC"],
                    "sicd": prop["products"]["SICD"],
                    "preview": str((args.output / f"{stem}.png").resolve()),
                }
                entries.append(entry)
                print(json.dumps({"site": site, "date": prop["date"], "id": prop["id"], "preview": entry["preview"]}), flush=True)
            except Exception as exc:
                print(f"FAILED {site} {prop['date']} {prop['id']}: {exc}", flush=True)
    if tiles:
        cols = 3
        sheet = Image.new("RGB", (cols * 520, math.ceil(len(tiles) / cols) * 560), "white")
        for i, tile in enumerate(tiles):
            sheet.paste(tile, ((i % cols) * 520, (i // cols) * 560))
        sheet.save(args.output / "contact_sheet.png")
    (args.output / "candidates.json").write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
