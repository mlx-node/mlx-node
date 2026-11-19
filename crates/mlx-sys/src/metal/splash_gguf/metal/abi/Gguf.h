// Adapted from Splash 1.6.1 (https://github.com/incoai/splash) d8af1054aab0f97de21d043e01aa98af3754f10e,
// runtime/metal/abi/Gguf.h. Apache-2.0; see THIRD_PARTY_NOTICES. Changed: only the staged decode tile's geometry and
// parameters kept (no prefill, register, int8, rotation, float or embedding parts).
#pragma once

// GGUF quantized projection parameters and tile geometry shared by host dispatch code and the staged decode
// kernels of kernels/shared/gguf_linear.metal.
#include "metal/abi/QuantFormat.h"

// A group is 32 inputs (the image's quantization unit); a plane tile is QUANT_TILE_ROWS rows of the weight image,
// a column tile the GGUF_TILE_COLUMNS output columns of one threadgroup (grid.x counts column tiles).
#define GGUF_TILE_COLUMNS 64u
// Decode tile of the staged kernels: two simdgroups, each staging its own GGUF_STAGED_COLUMNS columns
// GGUF_STAGED_STEP inputs at a time over 8, 16 or 32 rows.
#define GGUF_STAGED_COLUMNS 32u
#define GGUF_STAGED_STEP 32u
#define GGUF_STAGED_THREADS (GGUF_TILE_COLUMNS / GGUF_STAGED_COLUMNS * 32u)
// Decode tile of a gate/up pair of one format in one pass: one simdgroup staging the same GGUF_STAGED_COLUMNS
// columns of both tensors.
#define GGUF_GATE_UP_THREADS 32u

// One tensor per dispatch, over the dispatch's tiles (grid.x) and K partitions (grid.y).
struct GgufDecodeParams {
  uint32_t input_size;  // K
  uint32_t splits;      // K partitions; 1 = no cross-threadgroup reduction
  uint32_t out_stride;  // columns of a destination row
  uint32_t out_offset;  // first destination column of the tensor
  uint32_t spread;      // staged tile: 1 = each column tile starts its K walk at its own step, 0 = lockstep
};
static_assert(sizeof(GgufDecodeParams) == 20, "GGUF decode parameters are 20 bytes on both sides");

// Decode of a fused projection: up to three column segments of any formats in one dispatch, tiles in segment
// order.
struct GgufDecodeFusedParams {
  uint32_t input_size;  // K
  uint32_t splits;      // K partitions of every segment
  uint32_t out_stride;  // columns of a destination row
  uint32_t cols[3];     // columns per segment; 0 past the last
  uint32_t fmt[3];      // GGUF_FMT_* per segment
  uint32_t offset[3];   // first destination column per segment
  uint32_t spread;      // as GgufDecodeParams', each segment's tiles from their index in it
};
static_assert(sizeof(GgufDecodeFusedParams) == 52, "GGUF fused decode parameters are 52 bytes on both sides");

// The test-only decode of planes to half (gguf_dequant): rows x K halves, row-major.
struct GgufDequantParams {
  uint32_t rows;        // rows of the tensor (a multiple of QUANT_TILE_ROWS)
  uint32_t input_size;  // K
  uint32_t fmt;         // GGUF_FMT_*
};
static_assert(sizeof(GgufDequantParams) == 12, "GGUF dequant parameters are 12 bytes on both sides");
