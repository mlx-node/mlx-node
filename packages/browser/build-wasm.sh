#!/bin/bash
# Build the MLX WASM module for browser use.
#
# Prerequisites:
#   - WASI-SDK at /opt/wasi-sdk (or set WASI_SDK_PATH)
#   - Rust with wasm32-wasip1-threads target: rustup target add wasm32-wasip1-threads
#   - wasm-opt (Binaryen): brew install binaryen
#   - @napi-rs/cli: installed via yarn
#
# Usage: ./build-wasm.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
WASI_SDK_PATH="${WASI_SDK_PATH:-/opt/wasi-sdk}"
OUTPUT_DIR="$SCRIPT_DIR/dist"

echo "=== MLX Browser WASM Build ==="
echo "Root: $ROOT_DIR"
echo "WASI-SDK: $WASI_SDK_PATH"
echo ""

# Step 1: Build with NAPI-RS targeting wasm32-wasip1-threads
echo "--- Step 1: Building WASM via NAPI-RS ---"
cd "$ROOT_DIR"
WASI_SDK_PATH="$WASI_SDK_PATH" \
  npx @napi-rs/cli build \
    --manifest-path crates/mlx-core/Cargo.toml \
    --target wasm32-wasip1-threads \
    --no-default-features \
    --features browser \
    --release \
    --output-dir "$OUTPUT_DIR" \
    --js-binding mlx-core.wasi-browser.js \
    --dts mlx-core.d.ts

echo ""

# Step 2: Post-process with asyncify
echo "--- Step 2: Applying asyncify ---"
WASM_FILE="$OUTPUT_DIR/mlx-core.wasm32-wasi.wasm"
if [ -f "$WASM_FILE" ]; then
  wasm-opt \
    --asyncify \
    --pass-arg=asyncify-imports@env.mlx_webgpu_poll \
    --enable-threads \
    --enable-bulk-memory \
    --enable-mutable-globals \
    "$WASM_FILE" \
    -o "$WASM_FILE"
  echo "Asyncify applied to $WASM_FILE"
else
  echo "WARNING: WASM file not found at $WASM_FILE"
  echo "The NAPI-RS build may have used a different output filename."
  echo "Check $OUTPUT_DIR for .wasm files."
fi

echo ""
echo "=== Build Complete ==="
echo "Output: $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR"/*.wasm 2>/dev/null || echo "(no .wasm files found)"
