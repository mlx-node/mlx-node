// Per-row symmetric int8 KV quantization (Splash's target KV format), for
// fast::metal_kernel JIT. One simdgroup per (batch, head, token) row of 256:
// each lane owns 8 elements, the row's |max| is one simd_max, and
//   scale = max / 127,  q = clamp(rint(x * 127 / max), -127, 127)
// (zero rows quantize to 0 with scale 0), exactly as Splash's
// paged_store_row.h. Rejected verify rows are overwritten by the next
// cycle's store, so nothing is ever un-quantized.
//
// Bound names (fast::metal_kernel generates the signature):
//   x        : const device bfloat16_t*  [B, H, N, 256], any strides with a
//              contiguous last axis (x_strides is supplied by MLX)
//   rows     : const constant int32_t&   B * H * N
//   heads    : const constant int32_t&   H
//   tokens   : const constant int32_t&   N
//   q        : device int8_t*            [B, H, N, 256] row-major
//   s        : device float*             [B, H, N] row-major
//
// Grid: 256 threads per threadgroup (8 simdgroups = 8 rows), ceil(rows / 8)
// threadgroups along y.
R"(
  constexpr int D = 256;
  constexpr int PER_LANE = D / 32;
  const int row = int(threadgroup_position_in_grid.y) * 8 +
      int(simdgroup_index_in_threadgroup);
  if (row >= rows) {
    return;
  }
  const int lane = int(thread_index_in_simdgroup);
  const int n = row % tokens;
  const int bh = row / tokens;
  const int h = bh % heads;
  const int b = bh / heads;
  const device bfloat16_t* src = x + b * x_strides[0] + h * x_strides[1] +
      n * x_strides[2] + lane * PER_LANE;
  float v[PER_LANE];
  float local_max = 0.0f;
  for (int j = 0; j < PER_LANE; ++j) {
    v[j] = float(src[j]);
    local_max = max(local_max, fabs(v[j]));
  }
  const float maximum = simd_max(local_max);
  const float scale = maximum == 0.0f ? 0.0f : maximum / 127.0f;
  device int8_t* dst = q + size_t(row) * D + lane * PER_LANE;
  for (int j = 0; j < PER_LANE; ++j) {
    const int quantized = maximum == 0.0f
        ? 0
        : clamp(int(rint(v[j] * 127.0f / maximum)), -127, 127);
    dst[j] = int8_t(quantized);
  }
  if (lane == 0) {
    s[row] = scale;
  }
)"
