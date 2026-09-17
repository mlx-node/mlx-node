// Fused per-token (per-row) dynamic symmetric int8 activation quant for
// fast::metal_kernel JIT. Replaces the lazy MLX op chain
//   astype(x,f32) -> abs -> max(axis=1) -> maximum(.,eps) -> /127 -> round ->
//   clip(-127,127) -> astype(int8)
// with ONE kernel pass-pair, so x is never materialized to a [M,K] f32 copy and
// the row reduction + quantize happen in-place over the bf16 input.
//
// fast::metal_kernel auto-generates the [[kernel]] signature. Bound names:
//   x   : const device bfloat16_t*  [M,K] row-major activations
//   M,K : const constant int32_t&   scalars
//   x_i8: device int8_t*            [M,K] row-major output (quantized)
//   s_x : device float*            [M] (== [M,1]) per-row act scale
// Attributes injected because referenced below:
//   thread_position_in_threadgroup, threadgroup_position_in_grid,
//   threads_per_threadgroup, simdgroup_index_in_threadgroup,
//   thread_index_in_simdgroup.
//
// Numerics MUST be bit-identical to the lazy chain:
//   absmax    = max_k |x[m,k]|                 (computed in f32)
//   s_x[m]    = max(absmax, 1e-12f) / 127.0f
//   x_i8[m,k] = (int8_t) clamp(rint(x[m,k]/s_x[m]), -127, 127)
// MLX `round` is `metal::rint` (round-half-to-even), so `rint` here matches it
// exactly; the (int8_t) cast of an already-integral, clamped value is exact.
//
// One threadgroup per row m (grid.y = M), 256 threads per group. Two strided
// global passes over x[m,:] — pass 1 reduces to the row absmax, pass 2 reads x
// again and writes x_i8. The whole row is NOT cached in threadgroup memory
// (K can reach 17408 -> 34KB bf16 > the 32KB tg limit); strided two-pass over
// global is correct for any K.
R"(
  // 256 threads = 8 simdgroups of 32 (Apple simd width is 32).
  constexpr uint TG = 256u;
  constexpr uint SIMD_W = 32u;
  constexpr uint NSIMD = TG / SIMD_W;  // 8

  const uint m = threadgroup_position_in_grid.y;   // row index
  const uint tid = thread_position_in_threadgroup.x;  // 0..255
  const uint lane = thread_index_in_simdgroup;        // 0..31
  const uint sg = simdgroup_index_in_threadgroup;     // 0..7
  const int Ki = K;
  const uint row_off = (uint)m * (uint)Ki;

  threadgroup float tg_partials[NSIMD];

  // ---- Pass 1: per-row absmax in f32 (strided over global x) ----
  float local_absmax = 0.0f;
  for (int k = (int)tid; k < Ki; k += (int)TG) {
    float v = (float)x[row_off + (uint)k];
    local_absmax = max(local_absmax, fabs(v));
  }
  // Reduce within each simdgroup, then across simdgroups via threadgroup mem.
  float sg_absmax = simd_max(local_absmax);
  if (lane == 0) {
    tg_partials[sg] = sg_absmax;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // First simdgroup folds the 8 partials. NSIMD (8) <= SIMD_W (32), so one
  // simd_max over the lane-loaded partials yields the full-row absmax.
  float row_absmax = 0.0f;
  if (sg == 0) {
    float p = (lane < NSIMD) ? tg_partials[lane] : 0.0f;
    row_absmax = simd_max(p);
    if (lane == 0) {
      // Broadcast the row absmax to all threads via slot 0.
      tg_partials[0] = row_absmax;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  row_absmax = tg_partials[0];

  // s_x[m] = max(absmax, 1e-12f) / 127.0f  (matches the lazy chain exactly).
  float sx = max(row_absmax, 1e-12f) / 127.0f;
  if (tid == 0) {
    s_x[m] = sx;
  }

  // ---- Pass 2: quantize (re-read x, strided) ----
  for (int k = (int)tid; k < Ki; k += (int)TG) {
    float v = (float)x[row_off + (uint)k];
    // NOTE: the lazy chain computes x / s_x (a divide), not x * (1/s_x). To stay
    // bit-identical we MUST divide, since 1/sx then * is not equal in f32.
    float q = rint(v / sx);
    q = clamp(q, -127.0f, 127.0f);
    x_i8[row_off + (uint)k] = (int8_t)q;
  }
)"
