#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 --external-root PATH [--with-sar-cam] [--with-sarbm3d --accept-sarbm3d-nonprofit-license]"
}

external_root=""
with_sar_cam=0
with_sarbm3d=0
accept_sarbm3d=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --external-root)
      if [[ $# -lt 2 ]]; then
        echo "--external-root requires a path" >&2
        exit 2
      fi
      external_root="$2"
      shift 2
      ;;
    --with-sarbm3d)
      with_sarbm3d=1
      shift
      ;;
    --with-sar-cam)
      with_sar_cam=1
      shift
      ;;
    --accept-sarbm3d-nonprofit-license)
      accept_sarbm3d=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$external_root" ]]; then
  usage >&2
  exit 2
fi
case "$external_root" in
  /|/SET|/SET/*)
    echo "External root is unsafe or still a placeholder: $external_root" >&2
    exit 1
    ;;
esac

mkdir -p "$external_root"
external_root=$(cd "$external_root" && pwd)

if [[ "$with_sar_cam" -eq 1 ]]; then
  sar_cam_dir="$external_root/SAR-CAM"
  sar_cam_commit="ea5ee3bed00ab22735a7c87518fe5388c2d6c49a"
  if [[ -e "$sar_cam_dir" && ! -d "$sar_cam_dir/.git" ]]; then
    echo "Refusing to replace non-Git path: $sar_cam_dir" >&2
    exit 1
  fi
  if [[ ! -d "$sar_cam_dir/.git" ]]; then
    git clone https://github.com/JK-the-Ko/SAR-CAM.git "$sar_cam_dir"
  fi
  git -C "$sar_cam_dir" fetch --tags origin
  git -C "$sar_cam_dir" checkout --detach "$sar_cam_commit"
  if [[ -n "$(git -C "$sar_cam_dir" status --porcelain --untracked-files=no)" ]]; then
    echo "Pinned SAR-CAM checkout has modified tracked files; refusing formal use." >&2
    exit 1
  fi
  printf '%s  %s\n' "$sar_cam_commit" "$sar_cam_dir" > "$external_root/SAR-CAM.PINNED_COMMIT.txt"
fi

if [[ "$with_sarbm3d" -eq 1 ]]; then
  if [[ "$accept_sarbm3d" -ne 1 ]]; then
    echo "SAR-BM3D is nonprofit-only closed software." >&2
    echo "Read https://www.grip.unina.it/download/LICENSE_CLOSED.txt and rerun with" >&2
    echo "--accept-sarbm3d-nonprofit-license only if your use complies." >&2
    exit 1
  fi
  sarbm3d_url="https://www.grip.unina.it/download/prog/SAR-BM3D/version_1.0/SARBM3D_v10_linux64.tar.gz"
  sarbm3d_archive="$external_root/SARBM3D_v10_linux64.tar.gz"
  if [[ ! -f "$sarbm3d_archive" ]]; then
    partial="$sarbm3d_archive.part"
    curl --fail --location --retry 8 --retry-all-errors --continue-at - \
      "$sarbm3d_url" -o "$partial"
    if ! gzip -t "$partial"; then
      echo "Incomplete/corrupt SAR-BM3D partial retained at $partial" >&2
      echo "Retry to resume it, or move it aside before a clean download." >&2
      exit 1
    fi
    mv "$partial" "$sarbm3d_archive"
  fi
  if ! gzip -t "$sarbm3d_archive"; then
    echo "Existing SAR-BM3D archive is corrupt: $sarbm3d_archive" >&2
    echo "Move it aside before retrying; it was not overwritten." >&2
    exit 1
  fi
  python - "$sarbm3d_archive" <<'PY'
import sys
import tarfile
from pathlib import PurePosixPath

with tarfile.open(sys.argv[1], "r:gz") as archive:
    for member in archive.getmembers():
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts or member.issym() or member.islnk():
            raise SystemExit(f"Unsafe path or link in SAR-BM3D archive: {member.name}")
PY
  if [[ -e "$external_root/SARBM3D_v10_linux64" ]]; then
    echo "SAR-BM3D destination already exists; leaving it unchanged." >&2
  else
    extraction_root=$(mktemp -d "$external_root/.sarbm3d_extract.XXXXXX")
    cleanup() {
      rm -rf -- "$extraction_root"
    }
    trap cleanup EXIT
    tar --no-same-owner --no-same-permissions -xzf "$sarbm3d_archive" -C "$extraction_root"
    if ! find "$extraction_root" -type f -name 'SARBM3D_v10.m' -print -quit | grep -q .; then
      echo "Downloaded archive does not contain SARBM3D_v10.m" >&2
      exit 1
    fi
    mv "$extraction_root" "$external_root/SARBM3D_v10_linux64"
    trap - EXIT
  fi
  sha256sum "$sarbm3d_archive" > "$external_root/SARBM3D_v10_linux64.sha256"
fi

if [[ "$with_sar_cam" -eq 1 ]]; then
  echo "SAR_CAM_ROOT=$sar_cam_dir"
fi
if [[ "$with_sarbm3d" -eq 1 ]]; then
  echo "SARBM3D_ROOT=$external_root/SARBM3D_v10_linux64"
  echo "SARBM3D_ARCHIVE=$sarbm3d_archive"
fi
echo "External sources are ready at $external_root"
