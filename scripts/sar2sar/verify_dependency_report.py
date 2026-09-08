"""Verify mirror wheel hashes against PyPI's own release metadata."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "external/SAR2SAR/dependency-install-report.json"


def verify(item):
    name, version = item["metadata"]["name"], item["metadata"]["version"]
    url = item["download_info"]["url"]
    filename = urllib.parse.unquote(url.rsplit("/", 1)[-1])
    actual = item["download_info"]["archive_info"]["hashes"]["sha256"]
    metadata_url = f"https://pypi.org/pypi/{name}/{version}/json"
    with urllib.request.urlopen(metadata_url, timeout=60) as response:
        metadata = json.load(response)
    candidates = [entry for entry in metadata["urls"] if entry["filename"] == filename]
    if len(candidates) != 1 or candidates[0]["digests"]["sha256"] != actual:
        raise RuntimeError(f"Official PyPI hash does not match {filename}")
    return {"name": name, "version": version, "wheel": filename, "sha256": actual,
            "official_metadata_url": metadata_url, "download_url": url}


def main():
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    with ThreadPoolExecutor(max_workers=6) as pool:
        verified = sorted(pool.map(verify, report["install"]), key=lambda x: x["name"].lower())
    target = REPORT.with_name("dependency-verification.json")
    target.write_text(json.dumps({"all_wheels_match_official_pypi_sha256": True,
                                  "packages": verified}, indent=2) + "\n", encoding="utf-8")
    lock = ROOT / "scripts/sar2sar/requirements-win-py312.lock"
    lock.write_text("# Windows x64 / Python 3.12; wheel hashes verified with official PyPI metadata.\n" +
                    "\n".join(f"{v['name']}=={v['version']} --hash=sha256:{v['sha256']}" for v in verified) + "\n",
                    encoding="utf-8")
    print(f"Verified {len(verified)} wheels against official PyPI SHA-256 metadata.")


if __name__ == "__main__":
    main()
