#!/bin/bash
# Fast build script for simple-nn
# Usage: ./build.sh [run]

set -e

BUILD_DIR="dist/build"
OUTPUT="$BUILD_DIR/simple-nn"

mkdir -p "$BUILD_DIR"

echo "Building simple-nn..."
ghc -O2 -j -o "$OUTPUT" src/nn.hs

if [ "$1" = "run" ]; then
    echo "Running..."
    "$OUTPUT"
fi
