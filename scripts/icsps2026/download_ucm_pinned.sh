#!/usr/bin/env bash
set -euo pipefail

destination_root=${1:-}
if [[ -z "$destination_root" ]]; then
  echo "Usage: $0 /path/on/data-disk" >&2
  exit 2
fi
case "$destination_root" in
  /|/SET|/SET/*)
    echo "Destination is unsafe or still a placeholder: $destination_root" >&2
    exit 1
    ;;
esac
mkdir -p "$destination_root"
destination_root=$(cd "$destination_root" && pwd)
archive="$destination_root/UCMerced_LandUse.zip"
url="https://hf.co/datasets/torchgeo/ucmerced/resolve/d0af6e2eeea2322af86078068bd83337148a2149/UCMerced_LandUse.zip"
expected_md5="5b7ec56793786b6dc8a908e8854ac0e4"

if [[ ! -f "$archive" ]]; then
  partial="$archive.part"
  curl --fail --location --retry 8 --retry-all-errors --continue-at - \
    "$url" -o "$partial"
  partial_md5=$(md5sum "$partial" | awk '{print $1}')
  if [[ "$partial_md5" != "$expected_md5" ]]; then
    echo "Downloaded UCM archive MD5 mismatch: expected $expected_md5, found $partial_md5" >&2
    echo "The resumable partial file was retained for inspection: $partial" >&2
    exit 1
  fi
  mv "$partial" "$archive"
fi
actual_md5=$(md5sum "$archive" | awk '{print $1}')
if [[ "$actual_md5" != "$expected_md5" ]]; then
  echo "UCM archive MD5 mismatch: expected $expected_md5, found $actual_md5" >&2
  echo "Move the invalid archive aside before retrying; it was not overwritten." >&2
  exit 1
fi
if [[ ! -d "$destination_root/UCMerced_LandUse" ]]; then
  extraction_root=$(mktemp -d "$destination_root/.ucm_extract.XXXXXX")
  cleanup() {
    rm -rf -- "$extraction_root"
  }
  trap cleanup EXIT
  unzip -q "$archive" -d "$extraction_root"
  if [[ ! -d "$extraction_root/UCMerced_LandUse/Images" ]]; then
    echo "UCM archive does not contain UCMerced_LandUse/Images" >&2
    exit 1
  fi
  mv "$extraction_root/UCMerced_LandUse" "$destination_root/UCMerced_LandUse"
  trap - EXIT
  rmdir "$extraction_root"
fi
image_count=$(find "$destination_root/UCMerced_LandUse/Images" -type f \
  \( -iname '*.tif' -o -iname '*.tiff' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) \
  | wc -l | tr -d ' ')
if [[ "$image_count" -ne 2100 ]]; then
  echo "Expected 2100 UCM images, found $image_count; refusing formal use." >&2
  exit 1
fi
sha256sum "$archive" > "$archive.sha256"
echo "UCM archive verified with the checksum published by TorchGeo: $actual_md5"
echo "Images: $destination_root/UCMerced_LandUse/Images"
