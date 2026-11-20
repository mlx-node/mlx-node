#pragma once

// Shared C++/Metal dispatch ABI. Each segment keeps the standalone native
// projection's split count and its own contiguous output.
// Segment formats; the values are the FFI's per-segment codes
// (crates/mlx-core/src/models/quantized_linear.rs).
enum MixedProjectionFormat {
  kMixedQ4K = 4,
  kMixedQ5K = 5,
  kMixedQ6K = 6,
  // Q8_0 as MLX affine: 8-bit codes, group 32, F32 scales and biases.
  kMixedA8G32 = 8,
  kMixedIQ4XS = 20,
  kMixedIQ4NL = 21,
};
struct MixedProjectionParams {
  unsigned int input_size;
  unsigned int segments;
  unsigned int spread;
  unsigned int columns[3];
  unsigned int format[3];
  unsigned int splits[3];
  unsigned int group_end[3];
  unsigned int partial_offset[3];
  unsigned int counter_offset[3];
  // kMixedA8G32 only: x rows per threadgroup (qmv_wide's vecs_per_tg).
  unsigned int vectors[3];
};
