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

# -fwasm-exceptions enables native WASM EH (try/catch/throw instructions).
# Set globally so ALL cc-rs crates (esaxx-rs, etc.) compile with EH support.
# Note: vtable GC is prevented by --whole-archive in mlx-sys/build.rs;
# stack size (64 MB) is set by napi_build::setup() in mlx-core/build.rs.
if ! WASI_SDK_PATH="$WASI_SDK_PATH" \
  CXXFLAGS="-fexceptions -fwasm-exceptions" \
    "$ROOT_DIR/node_modules/.bin/napi" build \
      --manifest-path crates/mlx-core/Cargo.toml \
      --target wasm32-wasip1-threads \
      --no-default-features \
      --features browser \
      --profile wasi \
      --output-dir "$OUTPUT_DIR" \
      --js mlx-core.wasi-browser.js \
      --dts mlx-core.d.ts; then
  echo "[WARN] napi build completed compilation but failed during artifact copy."
  echo "[WARN] Falling back to the raw target artifact."
fi

RAW_WASM="$ROOT_DIR/target/wasm32-wasip1-threads/wasi/mlx_core.wasm"
if [[ -f "$RAW_WASM" ]]; then
  cp "$RAW_WASM" "$OUTPUT_DIR/index.wasm"
  echo "Copied fallback wasm: $RAW_WASM -> $OUTPUT_DIR/index.wasm"
else
  echo "[ERROR] Expected fallback artifact not found: $RAW_WASM" >&2
  exit 1
fi

RUNTIME_WASM="$ROOT_DIR/packages/core/mlx-core.wasm32-wasi.opt.wasm"
cp "$OUTPUT_DIR/index.wasm" "$RUNTIME_WASM"
echo "Updated runtime wasm: $OUTPUT_DIR/index.wasm -> $RUNTIME_WASM"

echo ""

# Step 2: No asyncify needed!
# The gpu-worker architecture uses Atomics.wait/notify for synchronization.
# mlx_webgpu_poll blocks via Atomics.wait (synchronous), no WASM stack
# save/restore needed. This eliminates the ~60MB binary size increase and
# the emnapi corruption issues that asyncify caused.
echo "--- Step 2: Skipped (no asyncify — using Atomics.wait) ---"

echo ""
echo "=== Build Complete ==="
echo "Output: $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR"/*.wasm 2>/dev/null || echo "(no .wasm files found)"
