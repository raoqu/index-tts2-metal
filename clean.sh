#!/usr/bin/env bash
# Remove build artifacts produced by build.sh.
#
# Deletes the CMake build directory (which also contains the vendored
# dependency build outputs: sentencepiece-build/, openfst-build/, _deps/, and
# the mtts binary). The vendored SOURCES under third_party/ are NOT touched.
#
# Honors the same BUILD_DIR override as build.sh:
#   BUILD_DIR=/tmp/foo ./clean.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-"$ROOT_DIR/build"}"

if [[ -d "$BUILD_DIR" ]]; then
  rm -rf "$BUILD_DIR"
  echo "Removed: $BUILD_DIR"
else
  echo "Nothing to clean: $BUILD_DIR does not exist"
fi
