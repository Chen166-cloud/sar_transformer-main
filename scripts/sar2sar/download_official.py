"""Download a pinned author-provided SAR2SAR inference snapshot (GPL-3.0)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request
import zipfile


COMMIT = "ca3c783333cbe072743f0b8060b5f168b9f0db11"
ARCHIVE_SHA256 = "70d03e0b664c45b9ddc4a31d67990c550cc51feb0aee68baef3a31c45f423bd3"
PROJECT = "https://gitlab.telecom-paris.fr/api/v4/projects/3344"
ROOT = Path(__file__).resolve().parents[2]
DESTINATION = ROOT / "external" / "SAR2SAR"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(relative: str, expected: str | None = None) -> Path:
    destination = DESTINATION / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = f"{PROJECT}/repository/files/{urllib.parse.quote(relative, safe='')}/raw?ref={COMMIT}"
    if not destination.exists():
        temporary = destination.with_name(destination.name + ".part")
        print(f"Downloading official {relative}", flush=True)
        with urllib.request.urlopen(url, timeout=600) as response, temporary.open("wb") as out:
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
        if expected and sha256(temporary) != expected:
            raise RuntimeError(f"SHA-256 mismatch for {relative}; partial file retained")
        temporary.replace(destination)
    if expected and sha256(destination) != expected:
        raise RuntimeError(f"SHA-256 mismatch for {destination}")
    return destination


def main() -> None:
    paths = [download("network_weights/SAR2SAR-test.zip", ARCHIVE_SHA256)]
    paths.extend(download(name) for name in ("README.md", "LICENSE", "SAR2SAR_Single_Look_test.ipynb"))
    with zipfile.ZipFile(paths[0]) as archive:
        root = DESTINATION.resolve()
        for member in archive.infolist():
            destination = (root / member.filename).resolve()
            if not destination.is_relative_to(root):
                raise RuntimeError(f"Unsafe archive path: {member.filename}")
            if member.is_dir():
                continue
            content = archive.read(member)
            if destination.exists():
                if destination.read_bytes() != content:
                    raise RuntimeError(f"Existing author file was modified: {destination}")
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(content)
            paths.append(destination)
    provenance = {
        "implementation": "Authors' SAR2SAR Single-Look Sentinel-1 inference package",
        "source_repository": "https://gitlab.telecom-paris.fr/ring/sar2sar",
        "source_commit": COMMIT,
        "snapshot_type": "commit-pinned GitLab API files, not a full repository clone",
        "paper_doi": "10.1109/JSTARS.2021.3071864",
        "license": "GPL-3.0",
        "archive_sha256": ARCHIVE_SHA256,
        "files_sha256": {p.relative_to(DESTINATION).as_posix(): sha256(p) for p in paths},
    }
    (DESTINATION / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"Verified official SAR2SAR snapshot: {DESTINATION}")


if __name__ == "__main__":
    main()
