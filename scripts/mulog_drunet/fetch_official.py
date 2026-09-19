"""Fetch the author's source and generic model only, never the other networks.

The server ignores Range. Read the first ZIP member then close the response;
the partial ZIP is a prefix cache, not a validated complete Git blob/archive.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import struct
import time
import urllib.request
import zipfile
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "external" / "MuLoG-DRUNet"
COMMIT = "f468573f5bd4d7d30065380b8580269bcefbec19"
REPOSITORY = "https://gitlab.telecom-paris.fr/ring/mulog-drunet"
API = "https://gitlab.telecom-paris.fr/api/v4/projects/6784"
SOURCE_BLOB = "ee08e58163c967d99b4d3a131ba842043f155762"
SOURCE_SIZE = 26954
SOURCE_SHA256 = "74204053abd6c28b3b56bbf1205e5a60f8463f081499e9b5d70679d334534594"
MODEL_ARCHIVE_BLOB = "eff7e4c921bdd05caf875b126e7a8432c6c4d3e8"
MODEL_URL = f"{API}/repository/blobs/{MODEL_ARCHIVE_BLOB}/raw"
MODEL_MEMBER = "models/generic_model.pth"
MODEL_COMPRESSED_SIZE = 121302383
MODEL_SIZE = 130581503
MODEL_CRC32 = 1667635814
MODEL_PREFIX_SIZE = 121302437
# SHA-256 of the extracted first member after verification of the fixed
# official HTTPS blob endpoint, ZIP local header, DEFLATE stream, size and CRC32.
GENERIC_MODEL_SHA256 = "20bc285c8710214003f72bdf5e48004c1accb63547d4a304c89714c385dff124"


@contextlib.contextmanager
def download_lock():
    """Cross-process OS lock; released by the OS if setup exits or is killed."""
    ROOT.mkdir(parents=True, exist_ok=True)
    try:
        lock = (ROOT / ".download.lock").open("a+b")
    except PermissionError as exc:
        raise RuntimeError("Another MuLoG asset setup holds the download lock; wait for it to finish") from exc
    try:
        lock.seek(0, os.SEEK_END)
        if lock.tell() == 0:
            lock.write(b"\0")
            lock.flush()
        lock.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("Another MuLoG asset setup holds the download lock; wait for it to finish") from exc
        try:
            yield
        finally:
            lock.seek(0)
            if os.name == "nt":
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    finally:
        lock.close()


def digest(path: Path, algorithm: str = "sha256") -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def git_blob(path: Path) -> str:
    h = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def read_limited_response(response, stream, byte_count: int, label: str) -> None:
    """Do not consume any byte after the required prefix (including a probe)."""
    received = 0
    started = reported = time.monotonic()
    while received < byte_count:
        block = response.read(min(64 * 1024, byte_count - received))
        if not block:
            raise ValueError(f"Truncated {label}: {received}/{byte_count} bytes")
        stream.write(block)
        received += len(block)
        now = time.monotonic()
        if now - reported >= 15 or received == byte_count:
            print(f"{label}: {received / 1024**2:.1f}/{byte_count / 1024**2:.1f} MiB, "
                  f"{received / max(now - started, .001) / 1024:.1f} KiB/s", flush=True)
            reported = now


def ensure_source(allow_download: bool = True) -> dict:
    path = ROOT / "py_functions.zip"
    if not path.exists():
        if not allow_download:
            raise FileNotFoundError("Source archive missing; run --source-only first")
        partial = path.with_suffix(".zip.download")
        with urllib.request.urlopen(f"{API}/repository/blobs/{SOURCE_BLOB}/raw", timeout=60) as response, partial.open("wb") as stream:
            read_limited_response(response, stream, SOURCE_SIZE, path.name)
        if partial.stat().st_size != SOURCE_SIZE or digest(partial) != SOURCE_SHA256 or git_blob(partial) != SOURCE_BLOB:
            raise ValueError("Official source archive checksum mismatch")
        partial.rename(path)
    if path.stat().st_size != SOURCE_SIZE or digest(path) != SOURCE_SHA256 or git_blob(path) != SOURCE_BLOB:
        raise ValueError("Official source archive checksum mismatch")
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            target = (ROOT / member.filename).resolve()
            if not target.is_relative_to(ROOT.resolve()):
                raise ValueError("Unsafe source archive path")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            data = archive.read(member)
            if target.exists():
                if target.read_bytes() != data:
                    raise ValueError(f"Modified official source retained for inspection: {target}")
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
    return {"git_blob_sha1": SOURCE_BLOB, "size_bytes": SOURCE_SIZE, "sha256": SOURCE_SHA256}


def validate_member_header(stream) -> None:
    header = stream.read(30)
    if len(header) != 30:
        raise ValueError("Truncated ZIP local header")
    signature, version, flags, method, _, _, crc, compressed, size, name_size, extra_size = struct.unpack("<IHHHHHIIIHH", header)
    if (signature, version, flags, method, crc, compressed, size, name_size, extra_size) != (
            0x04034B50, 20, 0, 8, MODEL_CRC32, MODEL_COMPRESSED_SIZE, MODEL_SIZE, len(MODEL_MEMBER), 0):
        raise ValueError("Unexpected generic model ZIP member header")
    if stream.read(name_size) != MODEL_MEMBER.encode("ascii"):
        raise ValueError("First ZIP member is not the pinned generic model")


def extract_generic_from_prefix(prefix: Path) -> dict:
    """Offline extraction: fixed header, DEFLATE end, size, CRC, optional SHA pin.

    Reads only the first member from an existing partial download; no networking.
    CRC is transport-corruption detection, not an independent authenticity proof.
    """
    prefix = Path(prefix)
    if prefix.stat().st_size < MODEL_PREFIX_SIZE:
        raise ValueError(f"Prefix incomplete: {prefix.stat().st_size}/{MODEL_PREFIX_SIZE} bytes; existing progress retained")
    target = ROOT / MODEL_MEMBER
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".pth.extracting")
    checksum, crc, written = hashlib.sha256(), 0, 0
    created = False
    with prefix.open("rb") as source:
        validate_member_header(source)
        decompressor = zlib.decompressobj(-zlib.MAX_WBITS)
        remaining = MODEL_COMPRESSED_SIZE
        try:
            with temporary.open("xb") as output:
                created = True
                while remaining:
                    block = source.read(min(64 * 1024, remaining))
                    if not block:
                        raise ValueError("Truncated compressed member")
                    remaining -= len(block)
                    while block:
                        data = decompressor.decompress(block, 1024 * 1024)
                        block = decompressor.unconsumed_tail
                        written += len(data)
                        if written > MODEL_SIZE:
                            raise ValueError("Generic model expands beyond its pinned size")
                        output.write(data)
                        checksum.update(data)
                        crc = zlib.crc32(data, crc)
                if not decompressor.eof or decompressor.unused_data or written != MODEL_SIZE or crc != MODEL_CRC32:
                    raise ValueError("Generic model DEFLATE, size, or CRC32 verification failed")
                actual_sha = checksum.hexdigest()
                if GENERIC_MODEL_SHA256 is not None and actual_sha != GENERIC_MODEL_SHA256:
                    raise ValueError("Generic model SHA256 differs from pinned checkpoint")
            if target.exists():
                if digest(target) != actual_sha:
                    raise ValueError(f"Existing checkpoint differs; retained for inspection: {target}")
                temporary.unlink()
            else:
                temporary.rename(target)
        except BaseException:
            if created and temporary.exists():
                temporary.unlink()
            raise
    return {"download_url": MODEL_URL, "archive_git_blob_identifier_for_url": MODEL_ARCHIVE_BLOB,
            "complete_archive_git_blob_verified": False,
            "member": MODEL_MEMBER, "member_size_bytes": written, "member_crc32": crc,
            "member_crc32_verified": True, "checkpoint_sha256": actual_sha,
            "checkpoint_sha256_matches_pin": GENERIC_MODEL_SHA256 is not None,
            "prefix_cache": str(prefix.resolve()), "prefix_bytes_used": MODEL_PREFIX_SIZE,
            "prefix_is_complete_zip_archive": False}


def write_manifest(source: dict, model: dict) -> dict:
    manifest = {"repository": REPOSITORY, "commit": COMMIT,
                "archives": {"py_functions.zip": source}, "generic_model": model,
                "files": {str(path.relative_to(ROOT)).replace('\\', '/'): digest(path)
                          for folder in ("py_functions", "models")
                          for path in (ROOT / folder).rglob('*') if path.is_file() and '__pycache__' not in path.parts}}
    (ROOT / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def existing_model_record(target: Path) -> dict:
    """Reuse the final model without requiring any compressed prefix cache."""
    if target.stat().st_size != MODEL_SIZE:
        raise ValueError("Existing generic checkpoint size mismatch")
    checksum, crc = hashlib.sha256(), 0
    with target.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            checksum.update(block)
            crc = zlib.crc32(block, crc)
    actual_sha = checksum.hexdigest()
    if crc != MODEL_CRC32:
        raise ValueError("Existing generic checkpoint CRC32 mismatch")
    if GENERIC_MODEL_SHA256 is not None:
        if actual_sha != GENERIC_MODEL_SHA256:
            raise ValueError("Existing generic checkpoint SHA256 mismatch")
    else:
        manifest_path = ROOT / "source_manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError("Existing checkpoint has no source record; use --extract-prefix with its official cache")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        previous = manifest.get("generic_model", {})
        if (manifest.get("repository") != REPOSITORY or manifest.get("commit") != COMMIT
                or previous.get("download_url") != MODEL_URL or previous.get("member") != MODEL_MEMBER
                or previous.get("member_size_bytes") != MODEL_SIZE or previous.get("member_crc32") != MODEL_CRC32
                or previous.get("checkpoint_sha256") != actual_sha):
            raise ValueError("Existing generic checkpoint source record mismatch")
    return {"download_url": MODEL_URL, "archive_git_blob_identifier_for_url": MODEL_ARCHIVE_BLOB,
            "complete_archive_git_blob_verified": False, "member": MODEL_MEMBER,
            "member_size_bytes": MODEL_SIZE, "member_crc32": MODEL_CRC32, "member_crc32_verified": True,
            "checkpoint_sha256": actual_sha, "checkpoint_sha256_matches_pin": GENERIC_MODEL_SHA256 is not None}


def fetch_assets(extract_prefix: Path | None = None, source_only: bool = False) -> None:
    source = ensure_source(allow_download=extract_prefix is None)
    if source_only:
        print(json.dumps(source, indent=2), flush=True)
        return
    if extract_prefix is not None:
        prefix = extract_prefix
    else:
        target = ROOT / MODEL_MEMBER
        if target.is_file():
            model = existing_model_record(target)
            print(json.dumps(write_manifest(source, model), indent=2), flush=True)
            return
        candidates = (ROOT / "generic_model.zip-prefix", ROOT / "models.zip.download", ROOT / "models.zip")
        prefix = next((path for path in candidates if path.is_file() and path.stat().st_size >= MODEL_PREFIX_SIZE), None)
        if prefix is None:
            incomplete = [path for path in (*candidates, ROOT / "generic_model.zip-prefix.download") if path.exists()]
            if incomplete:
                raise RuntimeError(f"Incomplete prefix retained: {incomplete}. Do not start a duplicate download; "
                                   "wait for an active transfer or explicitly choose how to replace its cache.")
            partial = ROOT / "generic_model.zip-prefix.download"
            print(f"Downloading only generic member prefix: {MODEL_PREFIX_SIZE / 1024**2:.1f} MiB", flush=True)
            # No Range header: server ignores it. Closing at the exact member end
            # cancels the remaining three model transfers.
            with urllib.request.urlopen(MODEL_URL, timeout=60) as response, partial.open("xb") as stream:
                read_limited_response(response, stream, MODEL_PREFIX_SIZE, "generic model prefix")
            prefix = ROOT / "generic_model.zip-prefix"
            partial.rename(prefix)
    model = extract_generic_from_prefix(prefix)
    print(json.dumps(write_manifest(source, model), indent=2), flush=True)
    if GENERIC_MODEL_SHA256 is None:
        print("Initial member verified by fixed HTTPS provenance, ZIP size and CRC32; recorded SHA256 for local integrity. "
              "Independent pinned checkpoint SHA256 is pending; full archive Git hash was NOT verified.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--extract-prefix", type=Path, help="offline: reuse first member from this existing partial download")
    selection.add_argument("--source-only", action="store_true", help="fetch and verify only the tiny official source archive")
    args = parser.parse_args()
    with download_lock():
        fetch_assets(args.extract_prefix, args.source_only)


if __name__ == "__main__":
    main()
