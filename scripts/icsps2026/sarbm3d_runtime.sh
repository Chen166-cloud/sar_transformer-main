#!/usr/bin/env bash

# Source-only helper for the legacy GRIP-UNINA SAR-BM3D Linux package.
# It does not modify the MATLAB installation; bundled OpenCV 2.1 libraries are
# exposed only to the current shell/MATLAB child process.
sarbm3d_prepare_runtime() {
  local package_root=$1
  local probe_log=$2
  local opencv_dirs=()
  local function_files=()
  local install_files=()
  local mex_files=()
  local mex_name
  local mex_path
  local ldd_output

  if ! command -v ldd >/dev/null 2>&1; then
    echo "ldd is required to diagnose the legacy SAR-BM3D MEX runtime." >&2
    return 1
  fi
  mapfile -t opencv_dirs < <(
    find "$package_root" -type d -path '*/lib_opencv210/glnxa64' -print
  )
  mapfile -t function_files < <(
    find "$package_root" -type f -name 'SARBM3D_v10.m' -print
  )
  mapfile -t install_files < <(
    find "$package_root" -type f -name 'install_opencv.m' -print
  )
  if [[ "${#opencv_dirs[@]}" -ne 1 ]]; then
    echo "Expected exactly one bundled lib_opencv210/glnxa64 directory; found ${#opencv_dirs[@]}." >&2
    return 1
  fi
  if [[ "${#function_files[@]}" -ne 1 || "${#install_files[@]}" -ne 1 ]]; then
    echo "Expected exactly one SARBM3D_v10.m and install_opencv.m in the official package." >&2
    return 1
  fi

  export SARBM3D_FUNCTION_FILE=${function_files[0]}
  export SARBM3D_OPENCV_LIB_DIR=${opencv_dirs[0]}
  export LD_LIBRARY_PATH="$SARBM3D_OPENCV_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  mkdir -p "$(dirname "$probe_log")"
  {
    echo "SARBM3D function: $SARBM3D_FUNCTION_FILE"
    echo "Bundled OpenCV loader path: $SARBM3D_OPENCV_LIB_DIR"
    echo "install_opencv.m present (not executed): ${install_files[0]}"
  } >> "$probe_log"

  for mex_name in SARBM3D_step1.mexa64 SARBM3D_step2.mexa64 removezeros.mexa64; do
    mapfile -t mex_files < <(find "$package_root" -type f -name "$mex_name" -print)
    if [[ "${#mex_files[@]}" -ne 1 ]]; then
      echo "Expected exactly one $mex_name; found ${#mex_files[@]}." >&2
      return 1
    fi
    mex_path=${mex_files[0]}
    if ! ldd_output=$(ldd "$mex_path" 2>&1); then
      echo "$ldd_output" >> "$probe_log"
      echo "ldd failed for $mex_path; inspect $probe_log" >&2
      return 1
    fi
    {
      echo "ldd $mex_path"
      echo "$ldd_output"
    } >> "$probe_log"
    # libmex/libmx are injected by the version-specific MATLAB launcher and
    # commonly appear unresolved in a parent-shell ldd.  Only the package's
    # bundled OpenCV 2.1 dependencies are actionable here; MATLAB smoke below
    # is the authoritative test for the remaining runtime.
    if grep -Eq 'lib(cv|cxcore)\.so[^ ]* => not found' <<< "$ldd_output"; then
      echo "Bundled OpenCV 2.1 is unresolved for SAR-BM3D; inspect $probe_log" >&2
      return 1
    fi
  done
}
