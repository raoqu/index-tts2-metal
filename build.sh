#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-"$ROOT_DIR/build"}"
BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"

if [[ -z "${JOBS:-}" ]]; then
  if command -v sysctl >/dev/null 2>&1; then
    JOBS="$(sysctl -n hw.ncpu 2>/dev/null || true)"
  fi
  if [[ -z "$JOBS" ]]; then
    JOBS="4"
  else
    JOBS="${JOBS%%[!0-9]*}"
    if [[ -z "$JOBS" ]]; then
      JOBS="4"
    fi
  fi
fi

cmake -S "$ROOT_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
cmake --build "$BUILD_DIR" --target mtts --parallel "$JOBS"

echo "Built: $BUILD_DIR/mtts"
