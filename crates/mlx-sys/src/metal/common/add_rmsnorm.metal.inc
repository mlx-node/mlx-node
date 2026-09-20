// Fused residual-add + RMSNorm: h = x + res, normed = rms_norm(h) * w.
// Emits BOTH outputs so the residual stream stays exact: h is rounded to T
// before the sum-of-squares, matching a separate Add -> fast::rms_norm pair
// bit-for-bit (the element mapping below mirrors rms_looped: N_READS-wide
// contiguous chunks striding by lsize * N_READS).
R"(
  {
    constexpr int N_READS = 4;
    constexpr int SIMD_SIZE = 32;
    const uint gid = thread_position_in_grid.y;
    const uint lid = thread_position_in_threadgroup.x;
    const uint lsize = threads_per_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sgid = simdgroup_index_in_threadgroup;

    threadgroup float local_inv[1];
    threadgroup float sums[SIMD_SIZE];

    const uint base = gid * AXIS;
    float acc = 0.0f;
    for (uint r = lid * N_READS; r < AXIS; r += lsize * N_READS) {
      for (int i = 0; i < N_READS; ++i) {
        uint c = r + i;
        if (c < AXIS) {
          float hv = float(x[base + c]) + float(res[base + c]);
          T ht = T(hv);
          h_out[base + c] = ht;
          acc += float(ht) * float(ht);
        }
      }
    }
    acc = simd_sum(acc);
    if (sgid == 0) {
      sums[lane] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
      sums[sgid] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0) {
      acc = simd_sum(sums[lane]);
      if (lane == 0) {
        local_inv[0] = metal::precise::rsqrt(acc / float(AXIS) + eps);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = local_inv[0];
    for (uint r = lid * N_READS; r < AXIS; r += lsize * N_READS) {
      for (int i = 0; i < N_READS; ++i) {
        uint c = r + i;
        if (c < AXIS) {
          normed[base + c] = w[c] * T(float(h_out[base + c]) * inv);
        }
      }
    }
  }
)"
