// Per-row int8 KV rows x fp32 scale -> BF16, for fast::metal_kernel JIT. The
// inverse of kv_int8_quantize_rows; prefill chunks after the first read the
// cached int8 prefix through it (one pass over the prefix per layer per
// chunk). One simdgroup per row, 8 elements per lane.
//
// Bound names:
//   q        : const device int8_t*     [B, H, N, 256], any strides with a
//              contiguous last axis (q_strides is supplied by MLX)
//   s        : const device float*      [B, H, N], any strides (s_strides)
//   rows     : const constant int32_t&  B * H * N
//   heads    : const constant int32_t&  H
//   tokens   : const constant int32_t&  N
//   out      : device bfloat16_t*       [B, H, N, 256] row-major
//
// Grid: 256 threads per threadgroup (8 rows), ceil(rows / 8) threadgroups
// along y.
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
  const device int8_t* src = q + b * q_strides[0] + h * q_strides[1] +
      n * q_strides[2] + lane * PER_LANE;
  const float scale = s[b * s_strides[0] + h * s_strides[1] + n * s_strides[2]];
  device bfloat16_t* dst = out + size_t(row) * D + lane * PER_LANE;
  for (int j = 0; j < PER_LANE; ++j) {
    dst[j] = bfloat16_t(float(src[j]) * scale);
  }
)"
