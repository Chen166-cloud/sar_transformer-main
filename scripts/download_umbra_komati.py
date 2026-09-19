"""Download one official Umbra open-data scene with resumable range requests.

Only the unmodified SICD, GEC and metadata products are downloaded. CPHD and
SIDD are intentionally omitted. A completed file is verified against the
official S3 ETag (8 MiB multipart MD5 for this specific published scene).
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.parse import quote
import hashlib
import json
import os
import time


ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "datasets" / "umbra_open" / "komati_20230802"
SCENE = "2023-08-02-19-35-16_UMBRA-05"
PREFIX = (
    "sar-data/tasks/Komati Power Station, S Africa/"
    "14d09467-68ec-4c03-87df-ea9b384d8b8c/" + SCENE + "/"
)
BASE = "https://umbra-open-data-catalog.s3.us-west-2.amazonaws.com/"
PRODUCTS = [
    ("_METADATA.json", 7415, "4568d2d5fc7506a59da07b541fb17136"),
    ("_GEC.tif", 81270892, "52c8dceafac2a41f856e9823c9b2fe20-10"),
    ("_SICD.nitf", 842145898, "ca1edef9b9986bf4640949d8c2822867-101"),
]
CHUNK = 1024 * 1024
ETAG_CHUNK = 8 * 1024 * 1024


def checksums(path, expected_etag):
    sha = hashlib.sha256()
    digests = []
    with path.open("rb") as stream:
        while block := stream.read(ETAG_CHUNK):
            sha.update(block)
            digests.append(hashlib.md5(block).digest())
    if "-" in expected_etag:
        etag = hashlib.md5(b"".join(digests)).hexdigest() + "-" + str(len(digests))
    else:
        etag = digests[0].hex()
    if etag != expected_etag:
        raise ValueError(f"Remote ETag mismatch for {path}: {etag}")
    return {"sha256": sha.hexdigest(), "s3_etag": etag, "etag_verified": True}


def download_product(suffix, size, etag):
    name = SCENE + suffix
    target = DEST / name
    url = BASE + quote(PREFIX + name, safe="/")
    if target.exists():
        if target.stat().st_size != size:
            raise ValueError(f"Existing destination has unexpected size: {target}")
        verified = checksums(target, etag)
        print(f"Verified existing {name}", flush=True)
        return {"file": name, "url": url, "bytes": size, **verified}
    parts_dir = DEST / ".download" / name
    parts_dir.mkdir(parents=True, exist_ok=True)
    count = (size + CHUNK - 1) // CHUNK

    def fetch(index):
        start = index * CHUNK
        end = min(size, start + CHUNK) - 1
        part = parts_dir / f"{index:05d}.part"
        if part.exists() and part.stat().st_size == end - start + 1:
            return part
        for attempt in range(5):
            try:
                request = Request(url, headers={
                    "Range": f"bytes={start}-{end}",
                    "User-Agent": "Umbra-open-data-research-download/1.0",
                })
                with urlopen(request, timeout=30) as response:
                    if response.status != 206:
                        raise ValueError(f"Server did not honor byte range: {response.status}")
                    if response.headers.get("Content-Range") != f"bytes {start}-{end}/{size}":
                        raise ValueError("Unexpected Content-Range")
                    data = response.read()
                if len(data) != end - start + 1:
                    raise ValueError("Incomplete range")
                part.write_bytes(data)
                return part
            except Exception as exc:
                print(f"  Retry range {index}, attempt {attempt + 1}: {exc}", flush=True)
                if attempt == 4:
                    raise
                time.sleep(min(2 ** attempt, 8))

    print(f"Downloading {name}: {size / 1e6:.1f} MB, {count} ranges", flush=True)
    with ThreadPoolExecutor(max_workers=48) as pool:
        futures = [pool.submit(fetch, i) for i in range(count)]
        for complete, future in enumerate(as_completed(futures), 1):
            future.result()
            if complete == count or complete % 20 == 0:
                print(f"  {name}: {complete}/{count} ranges", flush=True)
    staging = target.with_suffix(target.suffix + ".assembling")
    with staging.open("wb") as output:
        for i in range(count):
            output.write((parts_dir / f"{i:05d}.part").read_bytes())
    if staging.stat().st_size != size:
        raise ValueError("Assembled file has unexpected size")
    verified = checksums(staging, etag)
    os.replace(staging, target)
    # Only this downloader's exact range files are removed after verification.
    for i in range(count):
        (parts_dir / f"{i:05d}.part").unlink()
    parts_dir.rmdir()
    print(f"Verified {name}: SHA-256 {verified['sha256']}", flush=True)
    return {"file": name, "url": url, "bytes": size, **verified}


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    records = [download_product(*product) for product in PRODUCTS]
    manifest = {
        "scene": SCENE,
        "location": "Komati Power Station, South Africa",
        "source": "https://registry.opendata.aws/umbra-open-data/",
        "license": "CC BY 4.0",
        "downloaded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "products": records,
        "notes": "Provider originals. No despeckling, resampling or radiometric conversion applied.",
    }
    (DEST / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Complete: {DEST}", flush=True)


if __name__ == "__main__":
    main()
