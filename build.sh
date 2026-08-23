#!/bin/bash
# Fast build script for simple-nn
# Usage: ./build.sh [run]

set -e

BUILD_DIR="dist/build"
OUTPUT="$BUILD_DIR/simple-nn"

mkdir -p "$BUILD_DIR"

echo "Building simple-nn..."
ghc -O0 -j -isrc -o "$OUTPUT" src/Main.hs

if [ "$1" = "run" ]; then
    echo "Running..."
    "$OUTPUT"
fi
