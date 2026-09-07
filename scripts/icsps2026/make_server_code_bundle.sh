#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
destination=${1:-}
if [[ -z "$destination" ]]; then
  echo "Usage: $0 /absolute/path/sar_icsps26_code.tar.gz" >&2
  exit 2
fi
destination_parent=$(cd "$(dirname "$destination")" && pwd)
destination="$destination_parent/$(basename "$destination")"
case "$destination" in
  "$project_root"/*)
    echo "Destination must be outside the project tree: $destination" >&2
    exit 1
    ;;
esac
if [[ -e "$destination" ]]; then
  echo "Refusing to overwrite: $destination" >&2
  exit 1
fi

project_parent=$(dirname "$project_root")
project_name=$(basename "$project_root")
tar -czf "$destination" \
  --exclude="$project_name/.git" \
  --exclude="$project_name/datasets" \
  --exclude="$project_name/examples" \
  --exclude="$project_name/tmp" \
  --exclude="$project_name/experiments_icsps26" \
  --exclude="$project_name/results" \
  --exclude="$project_name/checkpoints" \
  --exclude="$project_name/**/__pycache__" \
  --exclude="$project_name/__pycache__" \
  --exclude="$project_name/**/*.pyc" \
  --exclude="$project_name/**/*.docx" \
  --exclude="$project_name/*.docx" \
  --exclude="$project_name/**/.DS_Store" \
  --exclude="$project_name/.DS_Store" \
  -C "$project_parent" "$project_name"

destination_name=$(basename "$destination")
if command -v sha256sum >/dev/null 2>&1; then
  (cd "$destination_parent" && sha256sum "$destination_name") > "$destination.sha256"
else
  (cd "$destination_parent" && shasum -a 256 "$destination_name") > "$destination.sha256"
fi
ls -lh "$destination" "$destination.sha256"
echo "The bundle intentionally excludes datasets, examples, checkpoints, results, .git objects, caches, DOCX files, and tmp artifacts."
echo "Transfer the two frozen split manifests and the three raw inputs (NWPU, UCM, RealSAR Noisy) separately; do not transfer GT16."
