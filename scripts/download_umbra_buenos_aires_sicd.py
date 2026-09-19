"""Download the official Umbra Buenos Aires SICD to E: with resume support.

The 3.51 GB provider original is split into HTTP byte ranges. Completed ranges
are retained on interruption. The assembled file is accepted only after its
size and official S3 multipart ETag have been verified; SHA-256 is recorded.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen
import argparse
import hashlib
import json
import os
import time


SCENE = "2025-01-31-14-10-46_UMBRA-08"
PREFIX = (
    "sar-data/tasks/Buenos Aires, ARG/"
    "0049d051-c279-4b42-8cfe-58bf673683f0/" + SCENE + "/"
)
BASE = "https://umbra-open-data-catalog.s3.us-west-2.amazonaws.com/"
SICD_NAME = SCENE + "_SICD.nitf"
SICD_SIZE = 3_512_703_554
SICD_ETAG = "2b5d7897bc9a410106e9d1d9f7fe7866-67"
STAC_NAME = SCENE + ".stac.v2.json"
STAC_SIZE = 8_691
STAC_ETAG = "4e6a022190e013d266fbea4dc900f663"
TRANSFER_CHUNK = 2 * 1024 * 1024
S3_ETAG_CHUNK = 50 * 1024 * 1024


def source_url(name):
    return BASE + quote(PREFIX + name, safe="/")


def validate_remote(url, expected_size, expected_etag):
    for attempt in range(6):
        try:
            request = Request(
                url,
                method="HEAD",
                headers={"User-Agent": "Umbra-research-download/1.0"},
            )
            with urlopen(request, timeout=45) as response:
                size = int(response.headers["Content-Length"])
                etag = response.headers["ETag"].strip('"')
            break
        except Exception as exc:
            if attempt == 5:
                raise
            delay = min(2 ** attempt, 12)
            print(f"Retry remote validation, attempt {attempt + 1}: {exc}", flush=True)
            time.sleep(delay)
    if (size, etag) != (expected_size, expected_etag):
        raise ValueError(f"Remote object changed: size={size}, ETag={etag}")


def checksums(path, expected_etag):
    sha256 = hashlib.sha256()
    etag_digests = []
    with path.open("rb") as stream:
        while block := stream.read(S3_ETAG_CHUNK):
            sha256.update(block)
            etag_digests.append(hashlib.md5(block).digest())
    etag = hashlib.md5(b"".join(etag_digests)).hexdigest() + "-" + str(len(etag_digests))
    if etag != expected_etag:
        raise ValueError(f"S3 ETag mismatch: computed {etag}, expected {expected_etag}")
    return sha256.hexdigest(), etag


def download_small(dest, name, expected_size, expected_etag):
    url = source_url(name)
    validate_remote(url, expected_size, expected_etag)
    target = dest / name
    if target.exists():
        data = target.read_bytes()
    else:
        with urlopen(Request(url, headers={"If-Match": f'"{expected_etag}"'}), timeout=30) as response:
            data = response.read()
        target.write_bytes(data)
    if len(data) != expected_size or hashlib.md5(data).hexdigest() != expected_etag:
        raise ValueError(f"Metadata validation failed: {target}")
    return {"file": name, "url": url, "bytes": expected_size,
            "md5": expected_etag, "etag_verified": True}


def download_sicd(dest, workers):
    url = source_url(SICD_NAME)
    validate_remote(url, SICD_SIZE, SICD_ETAG)
    target = dest / SICD_NAME
    if target.exists():
        if target.stat().st_size != SICD_SIZE:
            raise ValueError(f"Existing destination has unexpected size: {target}")
        sha256, etag = checksums(target, SICD_ETAG)
        print(f"Verified existing {target}", flush=True)
        return {"file": SICD_NAME, "url": url, "bytes": SICD_SIZE,
                "sha256": sha256, "s3_etag": etag, "etag_verified": True}

    parts = dest / ".download" / SICD_NAME
    parts.mkdir(parents=True, exist_ok=True)
    count = (SICD_SIZE + TRANSFER_CHUNK - 1) // TRANSFER_CHUNK

    def fetch(index):
        start = index * TRANSFER_CHUNK
        end = min(SICD_SIZE, start + TRANSFER_CHUNK) - 1
        expected = end - start + 1
        part = parts / f"{index:05d}.part"
        if part.exists() and part.stat().st_size == expected:
            return expected, False
        for attempt in range(6):
            try:
                request = Request(url, headers={
                    "Range": f"bytes={start}-{end}",
                    "If-Match": f'"{SICD_ETAG}"',
                    "User-Agent": "Umbra-research-download/1.0",
                })
                with urlopen(request, timeout=45) as response:
                    if response.status != 206:
                        raise ValueError(f"Range not honored: HTTP {response.status}")
                    if response.headers.get("ETag", "").strip('"') != SICD_ETAG:
                        raise ValueError("Remote object version changed")
                    if response.headers.get("Content-Range") != f"bytes {start}-{end}/{SICD_SIZE}":
                        raise ValueError("Unexpected Content-Range")
                    data = response.read()
                if len(data) != expected:
                    raise ValueError(f"Incomplete range: {len(data)} of {expected}")
                staging = part.with_suffix(".tmp")
                staging.write_bytes(data)
                os.replace(staging, part)
                return expected, True
            except Exception as exc:
                if attempt == 5:
                    raise
                print(f"Retry range {index}, attempt {attempt + 1}: {exc}", flush=True)
                time.sleep(min(2 ** attempt, 12))

    existing = sum(p.stat().st_size for p in parts.glob("*.part"))
    print(f"Destination: {target}", flush=True)
    print(f"Downloading {SICD_SIZE / 1e9:.3f} GB in {count} resumable ranges; "
          f"already present {existing / 1e6:.1f} MB", flush=True)
    complete_bytes = existing
    next_report = max(complete_bytes + 100 * 1024 * 1024, 100 * 1024 * 1024)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, i) for i in range(count)]
        for future in as_completed(futures):
            size, new = future.result()
            if new:
                complete_bytes += size
            if complete_bytes >= next_report or complete_bytes == SICD_SIZE:
                print(f"Progress: {complete_bytes / 1e9:.3f}/{SICD_SIZE / 1e9:.3f} GB "
                      f"({100 * complete_bytes / SICD_SIZE:.1f}%)", flush=True)
                next_report = complete_bytes + 100 * 1024 * 1024

    staging = target.with_suffix(target.suffix + ".assembling")
    print("All ranges present; assembling provider original...", flush=True)
    with staging.open("wb") as output:
        for index in range(count):
            part = parts / f"{index:05d}.part"
            expected = min(TRANSFER_CHUNK, SICD_SIZE - index * TRANSFER_CHUNK)
            if not part.exists() or part.stat().st_size != expected:
                raise ValueError(f"Missing or invalid range {index}")
            with part.open("rb") as stream:
                while block := stream.read(1024 * 1024):
                    output.write(block)
    if staging.stat().st_size != SICD_SIZE:
        raise ValueError("Assembled file has unexpected size")
    print("Computing SHA-256 and official multipart ETag...", flush=True)
    sha256, etag = checksums(staging, SICD_ETAG)
    os.replace(staging, target)
    for index in range(count):
        (parts / f"{index:05d}.part").unlink()
    parts.rmdir()
    try:
        parts.parent.rmdir()
    except OSError:
        pass
    print(f"Verified SICD. SHA-256: {sha256}", flush=True)
    return {"file": SICD_NAME, "url": url, "bytes": SICD_SIZE,
            "sha256": sha256, "s3_etag": etag, "etag_verified": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path,
                        default=Path(r"E:\SAR_Data\Umbra\Buenos_Aires_20250131"))
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    metadata = download_small(args.destination, STAC_NAME, STAC_SIZE, STAC_ETAG)
    sicd = download_sicd(args.destination, args.workers)
    manifest = {
        "scene": SCENE,
        "location": "Buenos Aires city center, Argentina",
        "source": "https://registry.opendata.aws/umbra-open-data/",
        "license": "CC BY 4.0",
        "downloaded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "products": [metadata, sicd],
        "notes": "Provider originals; no despeckling, resampling, radiometric conversion, or image editing applied.",
    }
    (args.destination / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Complete: {args.destination}", flush=True)


if __name__ == "__main__":
    main()
