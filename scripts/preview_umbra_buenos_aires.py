"""Read a small subset of the official Umbra GEC COG, never the full SICD.

Persistent, ETag-pinned HTTP byte-range cache; the GEC remains remote. PNGs
are display-only and preserve the provider's uint8 mapping, if present.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.parse import quote
import argparse
import io
import json
import struct
import threading
import time
import numpy as np
import tifffile
from PIL import Image, TiffImagePlugin

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "output/umbra_buenos_aires_preview"
SCENE = "2025-01-31-14-10-46_UMBRA-08"
PREFIX = "sar-data/tasks/Buenos Aires, ARG/0049d051-c279-4b42-8cfe-58bf673683f0/" + SCENE + "/"
BASE = "https://umbra-open-data-catalog.s3.us-west-2.amazonaws.com/"
URL = BASE + quote(PREFIX + SCENE + "_GEC.tif", safe="/")
SIZE = 502352599
ETAG = '"f9d0afef5c091cb4eefb33b0d7f84c74-10"'
BLOCK = 1024 * 1024
LIMIT = 40 * 1024 * 1024


class RangeReader(io.RawIOBase):
    def __init__(self):
        self.pos = 0
        self.name = "remote_GEC.tif"
        self.cache = DEST / "range_cache_f9d0afef"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.requested = set()
        self.new_bytes = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        base = [0, self.pos, SIZE][whence]
        self.pos = max(0, base + offset)
        return self.pos

    def fetch(self, index):
        with self.lock:
            self.requested.add(index)
            if len(self.requested) * BLOCK > LIMIT:
                raise ValueError("Preview range budget exceeded; refusing a full download")
        path = self.cache / f"{index:06d}.bin"
        start = index * BLOCK
        end = min(start + BLOCK, SIZE) - 1
        if path.exists() and path.stat().st_size == end - start + 1:
            return
        for attempt in range(4):
            try:
                req = Request(URL, headers={"Range": f"bytes={start}-{end}", "If-Match": ETAG})
                with urlopen(req, timeout=30) as response:
                    if response.status != 206:
                        raise ValueError("Server did not return a partial response")
                    if response.headers.get("ETag") != ETAG:
                        raise ValueError("Remote object version changed")
                    if response.headers.get("Content-Range") != f"bytes {start}-{end}/{SIZE}":
                        raise ValueError("Unexpected byte range")
                    data = response.read()
                if len(data) != end - start + 1:
                    raise ValueError("Incomplete range")
                path.write_bytes(data)
                with self.lock:
                    self.new_bytes += len(data)
                return
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)

    def prefetch(self, ranges):
        indexes = set()
        for offset, size in ranges:
            if size:
                indexes.update(range(offset // BLOCK, (offset + size - 1) // BLOCK + 1))
        with ThreadPoolExecutor(max_workers=16) as pool:
            list(pool.map(self.fetch, sorted(indexes)))

    def read(self, n=-1):
        if n is None or n < 0:
            n = SIZE - self.pos
        n = min(n, SIZE - self.pos)
        if n <= 0:
            return b""
        if n > LIMIT:
            raise ValueError("Refusing a full-file read for a preview")
        start = self.pos
        end = start + n
        first, last = start // BLOCK, (end - 1) // BLOCK
        self.prefetch([(start, n)])
        chunks = []
        for index in range(first, last + 1):
            data = (self.cache / f"{index:06d}.bin").read_bytes()
            a = max(start - index * BLOCK, 0)
            b = min(end - index * BLOCK, len(data))
            chunks.append(data[a:b])
        self.pos = end
        return b"".join(chunks)


def decode_tile(page, payload):
    """Wrap one compressed tile as a tiny TIFF for Pillow's native libtiff.

    No pixel processing is performed. This avoids requiring imagecodecs while
    retaining the source compression, predictor, sample depth and photometric.
    """
    if page.dtype != np.uint8 or page.samplesperpixel != 1:
        raise ValueError("Only this scene's grayscale uint8 tiles are supported")
    directory = TiffImagePlugin.ImageFileDirectory_v2()
    values = {256: page.tilewidth, 257: page.tilelength, 258: 8,
              259: int(page.compression), 262: int(page.photometric),
              273: 0, 277: 1, 278: page.tilelength,
              279: len(payload), 317: int(page.predictor)}
    for key, value in values.items():
        directory[key] = value
    # Pillow relocates StripOffsets relative to the end of the IFD itself.
    tiny_tiff = b"II*\x00" + struct.pack("<I", 8) + directory.tobytes(8) + payload
    with Image.open(io.BytesIO(tiny_tiff)) as tile:
        return np.array(tile)


def decode_roi(page, reader, x, y, width=1024, height=1024):
    if not page.is_tiled or page.samplesperpixel != 1:
        raise ValueError("Expected a tiled single-channel GEC")
    if x < 0 or y < 0 or x + width > page.imagewidth or y + height > page.imagelength:
        raise ValueError("ROI outside raster")
    tw, th = page.tilewidth, page.tilelength
    cols = (page.imagewidth + tw - 1) // tw
    indexes = [row * cols + col
               for row in range(y // th, (y + height - 1) // th + 1)
               for col in range(x // tw, (x + width - 1) // tw + 1)]
    reader.prefetch([(page.dataoffsets[i], page.databytecounts[i]) for i in indexes])
    out = np.zeros((height, width), dtype=page.dtype)
    for i in indexes:
        reader.seek(page.dataoffsets[i])
        tile = decode_tile(page, reader.read(page.databytecounts[i]))
        tx, ty = (i % cols) * tw, (i // cols) * th
        left, top = max(x, tx), max(y, ty)
        right, bottom = min(x + width, tx + tw), min(y + height, ty + th)
        out[top - y:bottom - y, left - x:right - x] = tile[top - ty:bottom - ty, left - tx:right - tx]
    return out


def save_previews(arr, name):
    Image.fromarray(arr).save(DEST / (name + ".png"))
    # A fixed display gamma reveals dark urban structure without spatial
    # interpolation or any denoising. Keep the provider-mapped PNG as well.
    display = np.rint(np.sqrt(arr.astype(np.float32) / 255.0) * 255).astype(np.uint8)
    Image.fromarray(display).save(DEST / (name + "_display.png"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--roi", type=int, nargs=2, metavar=("X", "Y"))
    parser.add_argument("--name", default="urban_roi_1024")
    args = parser.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)
    reader = RangeReader()
    with tifffile.TiffFile(reader) as tf:
        pages = list(tf.pages)
        report = {"source": URL, "source_etag": ETAG, "full_gec_bytes": SIZE,
                  "sicd_downloaded": False, "scene": SCENE, "license": "CC BY 4.0",
                  "pages": [{"index": i, "shape": list(p.shape), "dtype": str(p.dtype),
                             "compression": p.compression.name, "tiled": p.is_tiled,
                             "segments": len(p.dataoffsets), "compressed_bytes": sum(p.databytecounts)}
                            for i, p in enumerate(pages)]}
        print(json.dumps(report, indent=2), flush=True)
        if args.render:
            eligible = [(i, p) for i, p in enumerate(pages) if max(p.shape) <= 2000]
            if not eligible:
                raise ValueError("No small overview exists")
            index, page = max(eligible, key=lambda pair: max(pair[1].shape))
            reader.prefetch(zip(page.dataoffsets, page.databytecounts))
            arr = decode_roi(page, reader, 0, 0, page.imagewidth, page.imagelength)
            if arr.dtype != np.uint8 or arr.ndim != 2:
                raise ValueError("Expected display-ready grayscale uint8")
            save_previews(arr, "overview")
            report["overview"] = {"page": index, "shape": list(arr.shape),
                                  "radiometric_transform": "none; provider mapping retained",
                                  "resampling": "provider embedded overview"}
        if args.roi:
            x, y = args.roi
            arr = decode_roi(pages[0], reader, x, y)
            if arr.dtype != np.uint8:
                raise ValueError("Expected display-ready grayscale uint8")
            save_previews(arr, args.name)
            report["roi"] = {"x": x, "y": y, "width": 1024, "height": 1024,
                             "resampled": False, "radiometric_transform": "none",
                             "file": args.name + ".png"}
        report["new_bytes_downloaded_this_run"] = reader.new_bytes
        report["total_cached_bytes"] = sum(p.stat().st_size for p in reader.cache.glob("*.bin"))
        report["original_data_modified"] = False
        report["despeckled"] = False
        report["display_png_transform"] = "Fixed gamma 0.5: round(sqrt(uint8_DN / 255) * 255). Display only."
        (DEST / (args.name + "_provenance.json")).write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Done. Cached bytes: {report['total_cached_bytes']}; output: {DEST}", flush=True)


if __name__ == "__main__":
    main()
