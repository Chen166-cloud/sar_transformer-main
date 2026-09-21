"""Resumable downloader for the three Umbra stability-test scenes.

The scene inventory is read from ``output/umbra_scene_selection/selected_candidates.json``.
Provider originals are downloaded into E:\\SAR_Data\\Umbra\\Stability_3Scenes.
Every object is pinned by the size and ETag returned by the official S3 endpoint;
completed byte ranges survive interruption.  Target downloads include SICD, GEC,
and provider metadata.  Reference downloads intentionally use GEC plus metadata,
because the reference stack is geocoded/co-registered rather than fed to MERLIN.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote, urlsplit, urlunsplit
from urllib.request import Request, urlopen
import argparse
import hashlib
import json
import os
import re
import requests
import subprocess
import threading
import time


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SELECTIONS = ROOT / "output" / "umbra_scene_selection" / "selected_candidates.json"
DEFAULT_DESTINATION = Path(r"E:\SAR_Data\Umbra\Stability_3Scenes")
TRANSFER_CHUNK = 8 * 1024 * 1024
S3_MULTIPART_CHUNK = 50 * 1024 * 1024
SITE_DIRS = {
    "Busan Port": "01_Busan_Port",
    "Bangkok Suvarnabhumi Airport": "02_Bangkok_Suvarnabhumi_Airport",
    "Newark New York": "03_Newark_Port",
}
UMBRA_S3_HOST = "umbra-open-data-catalog.s3.us-west-2.amazonaws.com"
THREAD_STATE = threading.local()
PINNED_TARGET_OBJECTS = {
    "2025-05-21-12-38-53_UMBRA-07_SICD.nitf": (7_065_533_109, "895fca30567020fa1d5f0e91b0a8c393-135"),
    "2025-10-31-03-18-43_UMBRA-07_SICD.nitf": (12_522_080_481, "13ff187cf34c812b759a1f4694c5a466-239"),
    "2025-05-03-02-39-24_UMBRA-08_SICD.nitf": (6_638_993_630, "67cc2ef35fe38c1c6e4a402a9184e8e8-127"),
    "2025-05-21-12-38-53_UMBRA-07_GEC.tif": (486_856_188, "a29a8db7fe046b3594021e1fdc1bf920-10"),
    "2025-10-31-03-18-43_UMBRA-07_GEC.tif": (1_101_378_085, "af6fd4adfc5638fce55a2cab67d15dc0-22"),
    "2025-05-03-02-39-24_UMBRA-08_GEC.tif": (492_616_577, "1d3993c9ee0b98b188f3fb1792fad195-10"),
}


def bypass_local_proxy_for_umbra() -> None:
    """Use the public S3 endpoint directly; local HTTP proxies truncate large ranges."""
    for key in ("NO_PROXY", "no_proxy"):
        values = [item.strip() for item in os.environ.get(key, "").split(",") if item.strip()]
        if UMBRA_S3_HOST not in values:
            values.append(UMBRA_S3_HOST)
        os.environ[key] = ",".join(values)


def direct_session() -> requests.Session:
    session = getattr(THREAD_STATE, "session", None)
    if session is None:
        session = requests.Session()
        session.trust_env = os.environ.get("UMBRA_USE_PROXY") == "1"
        session.headers.update({"User-Agent": "Umbra-stability-experiment/1.0"})
        THREAD_STATE.session = session
    return session


def reset_session() -> None:
    session = getattr(THREAD_STATE, "session", None)
    if session is not None:
        session.close()
        del THREAD_STATE.session


def safe_url(raw: str) -> str:
    parts = urlsplit(raw)
    return urlunsplit((parts.scheme, parts.netloc, quote(parts.path, safe="/%"), parts.query, parts.fragment))


def request_with_retry(request: Request, timeout: int = 60):
    for attempt in range(8):
        try:
            return urlopen(request, timeout=timeout)
        except Exception as exc:
            if attempt == 7:
                raise
            delay = min(2 ** attempt, 20)
            print(f"retry {attempt + 1}/7 after {delay}s: {exc}", flush=True)
            time.sleep(delay)


def remote_identity(raw_url: str) -> dict[str, object]:
    url = safe_url(raw_url)
    for attempt in range(8):
        try:
            response = direct_session().head(url, timeout=(30, 60))
            if response.status_code != 200:
                raise RuntimeError(f"HEAD failed with HTTP {response.status_code}: {url}")
            return {
                "url": url,
                "bytes": int(response.headers["Content-Length"]),
                "etag": response.headers["ETag"].strip('"'),
                "last_modified": response.headers.get("Last-Modified"),
            }
        except Exception as exc:
            if attempt == 7:
                raise
            delay = min(2 ** attempt, 20)
            print(f"retry remote HEAD {attempt + 1}/7 after {delay}s: {exc}", flush=True)
            time.sleep(delay)
    raise AssertionError("unreachable")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_etag(path: Path, etag: str) -> bool:
    if "-" not in etag:
        digest = hashlib.md5()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest() == etag
    expected_parts = int(etag.rsplit("-", 1)[1])
    digests: list[bytes] = []
    with path.open("rb") as stream:
        while block := stream.read(S3_MULTIPART_CHUNK):
            digests.append(hashlib.md5(block).digest())
    computed = hashlib.md5(b"".join(digests)).hexdigest() + f"-{len(digests)}"
    return len(digests) == expected_parts and computed == etag


def filename_from_url(url: str) -> str:
    return urlsplit(url).path.rsplit("/", 1)[-1]


def cached_or_pinned_identity(raw_url: str, destination: Path) -> dict[str, object] | None:
    url = safe_url(raw_url)
    site_manifest = destination.parent / "download_manifest.json"
    if site_manifest.is_file():
        try:
            payload = json.loads(site_manifest.read_text(encoding="utf-8"))
            for item in payload.get("products", []):
                if safe_url(str(item.get("url", ""))) == url:
                    return {
                        "url": url,
                        "bytes": int(item["bytes"]),
                        "etag": str(item["etag"]),
                        "last_modified": item.get("last_modified"),
                    }
        except (OSError, ValueError, KeyError, TypeError):
            pass
    filename = filename_from_url(url)
    if filename in PINNED_TARGET_OBJECTS:
        size, etag = PINNED_TARGET_OBJECTS[filename]
        return {"url": url, "bytes": size, "etag": etag, "last_modified": None}
    return None


def download_object(raw_url: str, destination: Path, workers: int) -> dict[str, object]:
    identity = cached_or_pinned_identity(raw_url, destination) or remote_identity(raw_url)
    url, size, etag = str(identity["url"]), int(identity["bytes"]), str(identity["etag"])
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / filename_from_url(url)
    if target.exists():
        if target.stat().st_size != size:
            raise ValueError(f"existing file has the wrong size: {target}")
        if not verify_etag(target, etag):
            raise ValueError(f"existing file has the wrong ETag: {target}")
        print(f"verified existing: {target}", flush=True)
        return {**identity, "file": str(target), "sha256": sha256_file(target), "etag_verified": True}

    parts_dir = destination / ".download" / target.name
    parts_dir.mkdir(parents=True, exist_ok=True)
    count = (size + TRANSFER_CHUNK - 1) // TRANSFER_CHUNK

    def fetch(index: int) -> tuple[int, bool]:
        start = index * TRANSFER_CHUNK
        end = min(size, start + TRANSFER_CHUNK) - 1
        expected = end - start + 1
        part = parts_dir / f"{index:06d}.part"
        if part.exists() and part.stat().st_size == expected:
            return expected, False
        headers = {
            "Range": f"bytes={start}-{end}",
            "If-Match": f'"{etag}"',
            "User-Agent": "Umbra-stability-experiment/1.0",
        }
        request = Request(url, headers=headers)
        for attempt in range(8):
            try:
                temporary = part.with_suffix(".tmp")
                if attempt < 6:
                    response = direct_session().get(
                        url,
                        headers={"Range": headers["Range"], "If-Match": headers["If-Match"]},
                        timeout=(30, 180),
                    )
                    if response.status_code != 206:
                        raise RuntimeError(f"range request returned HTTP {response.status_code}")
                    if response.headers.get("ETag", "").strip('"') != etag:
                        raise RuntimeError("remote ETag changed")
                    if response.headers.get("Content-Range") != f"bytes {start}-{end}/{size}":
                        raise RuntimeError("unexpected Content-Range")
                    data = response.content
                    if len(data) != expected:
                        raise RuntimeError(f"short range: {len(data)} of {expected}")
                    temporary.write_bytes(data)
                else:
                    proxy_args = [] if os.environ.get("UMBRA_USE_PROXY") == "1" else ["--noproxy", "*"]
                    curl = subprocess.run(
                        [
                            "curl.exe", *proxy_args, "--fail", "--location", "--silent", "--show-error",
                            "--retry", "4", "--retry-all-errors", "--connect-timeout", "30",
                            "--max-time", "300", "--range", f"{start}-{end}",
                            "--header", f'If-Match: "{etag}"',
                            "--user-agent", "Umbra-stability-experiment/1.0",
                            "--output", str(temporary), url,
                        ],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        check=False,
                    )
                    if curl.returncode != 0:
                        raise RuntimeError(f"curl exit {curl.returncode}: {curl.stderr.strip()}")
                    if not temporary.exists() or temporary.stat().st_size != expected:
                        observed = temporary.stat().st_size if temporary.exists() else 0
                        raise RuntimeError(f"curl range size {observed}, expected {expected}")
                os.replace(temporary, part)
                return expected, True
            except Exception as exc:
                reset_session()
                if attempt == 7:
                    raise
                time.sleep(min(2 ** attempt, 20))
                print(f"retry part {index} ({attempt + 1}/7): {exc}", flush=True)
        raise AssertionError("unreachable")

    completed = sum(p.stat().st_size for p in parts_dir.glob("*.part"))
    print(
        f"download {target.name}: {size / 1e9:.3f} GB, {count} ranges, "
        f"resume={completed / 1e9:.3f} GB",
        flush=True,
    )
    report_step = max(256 * 1024 * 1024, size // 20)
    next_report = completed + report_step
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, index) for index in range(count)]
        for future in as_completed(futures):
            amount, is_new = future.result()
            if is_new:
                completed += amount
            if completed >= next_report or completed == size:
                print(f"progress {target.name}: {100 * completed / size:.1f}%", flush=True)
                next_report = completed + report_step

    assembling = target.with_suffix(target.suffix + ".assembling")
    with assembling.open("wb") as output:
        for index in range(count):
            part = parts_dir / f"{index:06d}.part"
            expected = min(TRANSFER_CHUNK, size - index * TRANSFER_CHUNK)
            if not part.exists() or part.stat().st_size != expected:
                raise RuntimeError(f"missing range {index} for {target.name}")
            with part.open("rb") as stream:
                for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                    output.write(block)
    if assembling.stat().st_size != size or not verify_etag(assembling, etag):
        raise RuntimeError(f"assembled object failed size/ETag verification: {target}")
    digest = sha256_file(assembling)
    os.replace(assembling, target)
    for part in parts_dir.glob("*.part"):
        part.unlink()
    parts_dir.rmdir()
    try:
        parts_dir.parent.rmdir()
    except OSError:
        pass
    print(f"verified: {target} sha256={digest}", flush=True)
    return {**identity, "file": str(target), "sha256": digest, "etag_verified": True}


def scene_name(selection: dict[str, object]) -> str:
    if selection["site"] not in SITE_DIRS:
        return re.sub(r"[^A-Za-z0-9._-]+", "_", str(selection["site"])).strip("_")
    return SITE_DIRS[str(selection["site"])]


def target_jobs(selection: dict[str, object]) -> list[tuple[str, str, dict[str, object]]]:
    target = selection["target"]
    return [
        ("target", "provider_metadata", target),
        ("target", "gec", target),
        ("target", "sicd", target),
    ]


def reference_jobs(selection: dict[str, object]) -> list[tuple[str, str, dict[str, object]]]:
    jobs: list[tuple[str, str, dict[str, object]]] = []
    for index, item in enumerate(selection["reference_pool"], start=1):
        jobs.append((f"reference_{index:02d}_{item['date']}", "provider_metadata", item))
        jobs.append((f"reference_{index:02d}_{item['date']}", "gec", item))
    return jobs


def main() -> None:
    if os.environ.get("UMBRA_USE_PROXY") != "1":
        bypass_local_proxy_for_umbra()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selections", type=Path, default=DEFAULT_SELECTIONS)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--scope", choices=("targets", "references-gec", "all"), default="targets")
    parser.add_argument("--site", action="append", help="exact site name; repeat to select multiple")
    parser.add_argument("--workers", type=int, default=48)
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 128:
        parser.error("--workers must be in [1, 128]")

    payload = json.loads(args.selections.read_text(encoding="utf-8"))
    selected = payload["selections"]
    if args.site:
        requested = set(args.site)
        selected = [item for item in selected if item["site"] in requested]
        missing = requested - {item["site"] for item in selected}
        if missing:
            parser.error(f"unknown site(s): {sorted(missing)}")

    args.destination.mkdir(parents=True, exist_ok=True)
    master = {
        "source_selection": str(args.selections.resolve()),
        "scope": args.scope,
        "license": "CC BY 4.0",
        "provider": "Umbra Open Data on AWS",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scenes": [],
    }
    for selection in selected:
        root = args.destination / scene_name(selection)
        root.mkdir(parents=True, exist_ok=True)
        jobs = []
        if args.scope in ("targets", "all"):
            jobs.extend(target_jobs(selection))
        if args.scope in ("references-gec", "all"):
            jobs.extend(reference_jobs(selection))
        scene_record = {
            "site": selection["site"],
            "scene_type": selection["scene_type"],
            "target_date": selection["target"]["date"],
            "common_bbox_lonlat": selection["common_bbox_lonlat"],
            "products": [],
        }
        for role, product, item in jobs:
            subdir = root / role
            print(f"[{selection['site']}] {role} {product}", flush=True)
            result = download_object(item[product], subdir, args.workers)
            result.update({"role": role, "product": product, "acquisition_id": item["id"], "date": item["date"]})
            scene_record["products"].append(result)
            manifest_path = root / "download_manifest.json"
            manifest_path.write_text(json.dumps(scene_record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        master["scenes"].append(scene_record)
        (args.destination / "download_manifest.json").write_text(
            json.dumps(master, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    print(f"complete: {args.destination}", flush=True)


if __name__ == "__main__":
    main()
