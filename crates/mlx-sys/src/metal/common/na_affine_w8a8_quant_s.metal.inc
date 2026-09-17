// Affine-W8A8 fused per-token activation quant + per-group act-sum S, for
// fast::metal_kernel JIT. Extends na_int8_quant.metal.inc (symmetric per-token
// int8 quant) with a THIRD output: S[m,g] = sum_{k in group g} x_q[m,k], the
// n-independent activation group-sum the affine bias term needs:
//   y[m,n] += s_x[m] * bias_adj[n,g] * S[m,g].
//
// Bound names (fast::metal_kernel auto-generates the [[kernel]] signature):
//   x    : const device bfloat16_t*  [M,K]    row-major activations
//   M,K,G: const constant int32_t&            scalars (G = group_size, K % G==0)
//   x_i8 : device int8_t*            [M,K]     quantized int8 activations (x_q)
//   s_x  : device float*            [M] (==[M,1]) per-row act scale
//   S    : device int32_t*          [M, K/G]   per-(row, group) sum of x_q
//
// The per-token int8 quant (absmax / s_x / round / clip) is BIT-IDENTICAL to
// na_int8_quant.metal.inc — same arithmetic, same two-pass-over-global layout.
// Pass 2 additionally accumulates, per K-group, the int32 group-sum of x_q via a
// simdgroup+threadgroup reduction, written to S[m,:].
//
// One threadgroup per row m (grid.y = M), 256 threads (8 simdgroups of 32).
R"(
  constexpr uint TG = 256u;
  constexpr uint SIMD_W = 32u;
  constexpr uint NSIMD = TG / SIMD_W;  // 8

  const uint m = threadgroup_position_in_grid.y;   // row index
  const uint tid = thread_position_in_threadgroup.x;  // 0..255
  const uint lane = thread_index_in_simdgroup;        // 0..31
  const uint sg = simdgroup_index_in_threadgroup;     // 0..7
  const int Ki = K;
  const int Gi = G;
  const int n_groups = Ki / Gi;
  const uint row_off = (uint)m * (uint)Ki;

  threadgroup float tg_partials[NSIMD];

  // ---- Pass 1: per-row absmax in f32 (strided over global x) ----
  float local_absmax = 0.0f;
  for (int k = (int)tid; k < Ki; k += (int)TG) {
    float v = (float)x[row_off + (uint)k];
    local_absmax = max(local_absmax, fabs(v));
  }
  float sg_absmax = simd_max(local_absmax);
  if (lane == 0) {
    tg_partials[sg] = sg_absmax;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float row_absmax = 0.0f;
  if (sg == 0) {
    float p = (lane < NSIMD) ? tg_partials[lane] : 0.0f;
    row_absmax = simd_max(p);
    if (lane == 0) {
      tg_partials[0] = row_absmax;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  row_absmax = tg_partials[0];

  // s_x[m] = max(absmax, 1e-12f) / 127.0f  (bit-identical to the symmetric quant).
  float sx = max(row_absmax, 1e-12f) / 127.0f;
  if (tid == 0) {
    s_x[m] = sx;
  }

  // ---- Pass 2: quantize x_q AND accumulate per-group S[m,g] ----
  // One group at a time: every thread strides its share of the group's K, the
  // simdgroups reduce-sum the int8 values, slot-0 of each simdgroup writes to
  // tg, simdgroup 0 folds the 8 partials and writes S[m,g]. Reusing the SAME
  // tg_partials slots is safe — a barrier separates groups.
  threadgroup int tg_isum[NSIMD];
  const uint sb_row = (uint)m * (uint)n_groups;
  for (int g = 0; g < n_groups; ++g) {
    const int k0 = g * Gi;
    int local_sum = 0;
    for (int kk = (int)tid; kk < Gi; kk += (int)TG) {
      const int k = k0 + kk;
      float v = (float)x[row_off + (uint)k];
      // NOTE: divide (not * reciprocal) to stay bit-identical to the lazy chain.
      float q = rint(v / sx);
      q = clamp(q, -127.0f, 127.0f);
      int qi = (int)q;
      x_i8[row_off + (uint)k] = (int8_t)qi;
      local_sum += qi;
    }
    int sg_sum = simd_sum(local_sum);
    if (lane == 0) {
      tg_isum[sg] = sg_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
      int p = (lane < NSIMD) ? tg_isum[lane] : 0;
      int group_sum = simd_sum(p);
      if (lane == 0) {
        S[sb_row + (uint)g] = group_sum;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
)"
