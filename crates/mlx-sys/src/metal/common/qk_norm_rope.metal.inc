// Fused per-head RMSNorm + partial RoPE for the Q and K projections of one
// attention layer, in one dispatch (Splash `attention_qkv.metal` does norm +
// rope + staging in one kernel). Bit-identical to MLX's
// `fast::rms_norm` followed by `fast::rope(traditional=false)` on the
// `[B, H, T, D]` transpose: the RMS part mirrors `rms_single_row` (N_READS=4
// contiguous chunks, the same two `simd_sum` folds, `precise::rsqrt`, the
// BF16 `w * T(x * inv)` product), the rotation part mirrors `rope_impl`
// (`exp2(-d * log2(base))`, `fast::cos/sin`, the same product order), and
// the normed row is rounded to T before it is rotated, as the two-op chain
// rounds it in memory. Dimensions `>= ROT` are copied, as the partial-rope
// `copy_gpu` does.
//
// Bound names (fast::metal_kernel generates the signature):
//   q, k      : const device T*     [B, T, HQ, D] / [B, T, HK, D], any strides
//               with a contiguous last axis (q_strides / k_strides supplied)
//   wq, wk    : const device T*     [D] contiguous RMSNorm weights
//   offsets   : int32 [B] (or a scalar) position offset per batch row
//   eps, base, scale : const constant float&  (base = log2(theta))
//   heads_q, heads_k, tokens : const constant int32_t&
//   q_out, k_out : device T*       [B, HQ, T, D] / [B, HK, T, D] row-major
//
// Template: T (bfloat16_t / half / float), D (head dim, D % 128 == 0, one
// thread per 4 elements), ROT (rotated dims, even, <= D).
// Grid: (D / 4, B * (HQ + HK) * T, 1); threadgroup (D / 4, 1, 1).
R"(
  {
    constexpr int N_READS = 4;
    constexpr int SIMD_SIZE = 32;
    constexpr int HALF = ROT / 2;
    const uint row = thread_position_in_grid.y;
    const uint lid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sgid = simdgroup_index_in_threadgroup;

    const int HQ = heads_q;
    const int HK = heads_k;
    const int TOK = tokens;
    const int t = int(row) % TOK;
    const int bh = int(row) / TOK;
    const int h = bh % (HQ + HK);
    const int b = bh / (HQ + HK);
    const bool is_q = h < HQ;
    const int hh = is_q ? h : h - HQ;

    const device T* x = is_q
        ? q + b * q_strides[0] + t * q_strides[1] + hh * q_strides[2]
        : k + b * k_strides[0] + t * k_strides[1] + hh * k_strides[2];
    const device T* w = is_q ? wq : wk;
    device T* out = is_q
        ? q_out + (size_t(b) * HQ + hh) * size_t(TOK) * D + size_t(t) * D
        : k_out + (size_t(b) * HK + hh) * size_t(TOK) * D + size_t(t) * D;
    x += lid * N_READS;

    threadgroup float local_inv_mean[1];
    threadgroup float local_sums[SIMD_SIZE];
    threadgroup T normed[D];

    // rms_single_row, full-chunk branch (D % N_READS == 0).
    float acc = 0;
    float thread_x[N_READS];
    for (int i = 0; i < N_READS; i++) {
      thread_x[i] = x[i];
      acc += thread_x[i] * thread_x[i];
    }
    acc = simd_sum(acc);
    if (sgid == 0) {
      local_sums[lane] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
      local_sums[sgid] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0) {
      acc = simd_sum(local_sums[lane]);
      if (lane == 0) {
        local_inv_mean[0] = metal::precise::rsqrt(acc / uint(D) + eps);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = 0; i < N_READS; i++) {
      const uint d = lid * N_READS + i;
      normed[d] = w[d] * static_cast<T>(thread_x[i] * local_inv_mean[0]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // rope_impl, forward, non-traditional: pair (p, p + HALF), p < HALF.
    const float L = scale * static_cast<float>(uint(t) + offsets[b]);
    for (int i = 0; i < N_READS; i++) {
      const uint d = lid * N_READS + i;
      T v = normed[d];
      if (d < uint(ROT)) {
        const uint p = d < uint(HALF) ? d : d - uint(HALF);
        const float dd = static_cast<float>(p) / static_cast<float>(HALF);
        const float inv_freq = metal::exp2(-dd * base);
        const float theta = L * inv_freq;
        const float costheta = metal::fast::cos(theta);
        const float sintheta = metal::fast::sin(theta);
        const float x1 = static_cast<float>(normed[p]);
        const float x2 = static_cast<float>(normed[p + HALF]);
        float r;
        if (d < uint(HALF)) {
          r = x1 * costheta - x2 * sintheta;
        } else {
          r = x1 * sintheta + x2 * costheta;
        }
        v = static_cast<T>(r);
      }
      out[d] = v;
    }
  }
)"
