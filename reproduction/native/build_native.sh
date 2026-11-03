#!/usr/bin/env bash
set -euo pipefail
NATIVE_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
ANYDEX_PY="${ANYDEX_PY:-/home/wangrenpeng/boya-inhand-wm/.venv/bin/python}"
"$ANYDEX_PY" "$NATIVE_REPO/reproduction/native/prepare_sources.py"
"$ANYDEX_PY" "$NATIVE_REPO/reproduction/native/acquire_cuda.py"
"$ANYDEX_PY" "$NATIVE_REPO/reproduction/native/write_env.py"
source "$NATIVE_REPO/outputs/native_deps/env.sh"
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
NATIVE_LOG_DIR="$ANYDEX_NATIVE/logs/build_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$NATIVE_LOG_DIR"
(
 cd "$ANYDEX_NATIVE/build_src/MinkowskiEngine"
 "$ANYDEX_PY" -B setup.py build_ext --inplace --force_cuda \
  --blas=openblas --blas_include_dirs="$ANYDEX_NATIVE/openblas/include" \
  --blas_library_dirs="$ANYDEX_NATIVE/openblas/lib"
) > "$NATIVE_LOG_DIR/minkowski.log" 2>&1
(
 cd "$ANYDEX_NATIVE/build_src/pointnet2"
 "$ANYDEX_PY" -B setup.py build_ext \
  --build-lib "$ANYDEX_NATIVE/pointnet2/lib" \
  --build-temp "$ANYDEX_NATIVE/pointnet2/temp"
) > "$NATIVE_LOG_DIR/pointnet2.log" 2>&1
"$ANYDEX_PY" -B "$NATIVE_REPO/reproduction/native/validate_native_ops.py"
