R"METAL(
// Short-row grouped Q8 decode helpers (K = 192 half-warp, K = 512 full-warp).
template <
    int M_TILE,
    bool SYM_F16,
    bool WITH_ROPE,
    typename ScalePtr,
    typename BiasPtr,
    typename PosPtr,
    typename InvFreqPtr>
inline void hy4_q8_grouped_k192_impl(
    const device float* x,
    const device uint32_t* w,
    ScalePtr scales,
    BiasPtr biases,
    PosPtr pos,
    InvFreqPtr inv_freq,
    device float* out,
    int M,
    int G,
    int N,
    int rope_dim,
    size_t x_s0,
    size_t x_s1,
    uint tgp_y,
    uint tgp_z,
    uint sg,
    uint lane) {
  constexpr int NUM_SG = 4;
  constexpr int ROWS_PER_SG = 4;
  constexpr int ROWS_PER_TG = NUM_SG * ROWS_PER_SG;
  constexpr int N_F4 = 48;
  constexpr int N_GRP = 6;

  const uint half_id = lane >> 4;
  const uint hlane = lane & 15u;
  const int out_row0 = int(tgp_y) * ROWS_PER_TG + int(sg) * ROWS_PER_SG +
                       int(half_id) * 2;

  const int n_mtiles = (M + M_TILE - 1) / M_TILE;
  const int g = int(tgp_z) / n_mtiles;
  const int row0 = (int(tgp_z) % n_mtiles) * M_TILE;
  if (row0 >= M || g >= G) return;

  const int out_n = WITH_ROPE ? (N + rope_dim) : N;
  const size_t x_s0_f4 = x_s0 >> 2;
  const size_t x_s1_f4 = x_s1 >> 2;

  if (out_row0 < N) {
    const int r0 = out_row0;
    const int r1 = metal::min(out_row0 + 1, N - 1);
    const size_t wrow0 = size_t(g) * size_t(N) + size_t(r0);
    const size_t wrow1 = size_t(g) * size_t(N) + size_t(r1);

    const int k0 = int(hlane) * 3;
    const int g0 = k0 >> 3;
    const int g1 = (k0 + 1) >> 3;
    const int g2 = (k0 + 2) >> 3;

    const device uint32_t* w_ptr0 = w + wrow0 * size_t(N_F4) + size_t(k0);
    const device uint32_t* w_ptr1 = w + wrow1 * size_t(N_F4) + size_t(k0);
    const auto s_row0 = scales + wrow0 * size_t(N_GRP);
    const auto s_row1 = scales + wrow1 * size_t(N_GRP);

    const uint32_t u0_0 = w_ptr0[0];
    const uint32_t u0_1 = w_ptr0[1];
    const uint32_t u0_2 = w_ptr0[2];
    const float s0_0 = float(s_row0[g0]);
    const float s0_1 = float(s_row0[g1]);
    const float s0_2 = float(s_row0[g2]);

    const uint32_t u1_0 = w_ptr1[0];
    const uint32_t u1_1 = w_ptr1[1];
    const uint32_t u1_2 = w_ptr1[2];
    const float s1_0 = float(s_row1[g0]);
    const float s1_1 = float(s_row1[g1]);
    const float s1_2 = float(s_row1[g2]);

    const device float4* x_base = reinterpret_cast<const device float4*>(x);
    float acc[M_TILE][2];

    if constexpr (SYM_F16) {
      const float4 w0_0 = float4(as_type<char4>(u0_0 ^ 0x80808080u)) * s0_0;
      const float4 w0_1 = float4(as_type<char4>(u0_1 ^ 0x80808080u)) * s0_1;
      const float4 w0_2 = float4(as_type<char4>(u0_2 ^ 0x80808080u)) * s0_2;

      const float4 w1_0 = float4(as_type<char4>(u1_0 ^ 0x80808080u)) * s1_0;
      const float4 w1_1 = float4(as_type<char4>(u1_1 ^ 0x80808080u)) * s1_1;
      const float4 w1_2 = float4(as_type<char4>(u1_2 ^ 0x80808080u)) * s1_2;

#pragma unroll
      for (int r = 0; r < M_TILE; ++r) {
        const int mr = metal::min(row0 + r, M - 1);
        const device float4* xp =
            x_base + size_t(mr) * x_s0_f4 + size_t(g) * x_s1_f4 + size_t(k0);
        const float4 x0 = xp[0];
        const float4 x1 = xp[1];
        const float4 x2 = xp[2];
        acc[r][0] = metal::dot(x0, w0_0) + metal::dot(x1, w0_1) + metal::dot(x2, w0_2);
        acc[r][1] = metal::dot(x0, w1_0) + metal::dot(x1, w1_1) + metal::dot(x2, w1_2);
      }
    } else {
      const auto b_row0 = biases + wrow0 * size_t(N_GRP);
      const auto b_row1 = biases + wrow1 * size_t(N_GRP);
      const float b0_0 = float(b_row0[g0]);
      const float b0_1 = float(b_row0[g1]);
      const float b0_2 = float(b_row0[g2]);
      const float b1_0 = float(b_row1[g0]);
      const float b1_1 = float(b_row1[g1]);
      const float b1_2 = float(b_row1[g2]);

      const float4 w0_0 = float4(as_type<uchar4>(u0_0)) * s0_0 + b0_0;
      const float4 w0_1 = float4(as_type<uchar4>(u0_1)) * s0_1 + b0_1;
      const float4 w0_2 = float4(as_type<uchar4>(u0_2)) * s0_2 + b0_2;

      const float4 w1_0 = float4(as_type<uchar4>(u1_0)) * s1_0 + b1_0;
      const float4 w1_1 = float4(as_type<uchar4>(u1_1)) * s1_1 + b1_1;
      const float4 w1_2 = float4(as_type<uchar4>(u1_2)) * s1_2 + b1_2;

#pragma unroll
      for (int r = 0; r < M_TILE; ++r) {
        const int mr = metal::min(row0 + r, M - 1);
        const device float4* xp =
            x_base + size_t(mr) * x_s0_f4 + size_t(g) * x_s1_f4 + size_t(k0);
        const float4 x0 = xp[0];
        const float4 x1 = xp[1];
        const float4 x2 = xp[2];
        acc[r][0] = metal::dot(x0, w0_0) + metal::dot(x1, w0_1) + metal::dot(x2, w0_2);
        acc[r][1] = metal::dot(x0, w1_0) + metal::dot(x1, w1_1) + metal::dot(x2, w1_2);
      }
    }

#pragma unroll
    for (int r = 0; r < M_TILE; ++r) {
      float v0 = acc[r][0];
      float v1 = acc[r][1];
      v0 += simd_shuffle_xor(v0, 8);
      v1 += simd_shuffle_xor(v1, 8);
      v0 += simd_shuffle_xor(v0, 4);
      v1 += simd_shuffle_xor(v1, 4);
      v0 += simd_shuffle_xor(v0, 2);
      v1 += simd_shuffle_xor(v1, 2);
      v0 += simd_shuffle_xor(v0, 1);
      v1 += simd_shuffle_xor(v1, 1);
      const int mr = row0 + r;
      if (hlane == 0 && mr < M) {
        const size_t obase =
            (size_t(mr) * size_t(G) + size_t(g)) * size_t(out_n);
        out[obase + size_t(out_row0)] = v0;
        if (out_row0 + 1 < N) {
          out[obase + size_t(out_row0 + 1)] = v1;
        }
      }
    }
  }

  if constexpr (WITH_ROPE) {
    if (tgp_y == 0u) {
      const int half_rd = rope_dim >> 1;
      for (int r = int(sg); r < M_TILE; r += NUM_SG) {
        const int mr = row0 + r;
        if (mr < M) {
          const float pos_m = float(pos[mr]);
          const size_t x_pe_base = size_t(mr) * x_s0 + size_t(g) * x_s1 + 192u;
          const size_t o_pe_base =
              (size_t(mr) * size_t(G) + size_t(g)) * size_t(out_n) + size_t(N);
          for (int p = int(lane); p < half_rd; p += 32) {
            const float theta = pos_m * inv_freq[p];
            const float c = metal::precise::cos(theta);
            const float s = metal::precise::sin(theta);
            const float x0 = x[x_pe_base + size_t(2 * p)];
            const float x1 = x[x_pe_base + size_t(2 * p + 1)];
            out[o_pe_base + size_t(2 * p)] = x0 * c - x1 * s;
            out[o_pe_base + size_t(2 * p + 1)] = x0 * s + x1 * c;
          }
        }
      }
    }
  }
}

template <int M_TILE, bool SYM_F16, bool GATED, typename ScalePtr, typename BiasPtr, typename GatePtr>
inline void hy4_q8_grouped_k512_impl(
    const device float* x,
    const device uint32_t* w,
    ScalePtr scales,
    BiasPtr biases,
    GatePtr gate,
    device float* out,
    int M,
    int G,
    int N,
    size_t g_s0,
    size_t g_s1,
    uint tgp_y,
    uint tgp_z,
    uint sg,
    uint lane) {
  constexpr int NUM_SG = 4;
  constexpr int ROWS_PER_SG = 2;
  constexpr int ROWS_PER_TG = NUM_SG * ROWS_PER_SG;
  constexpr int N_U4 = 32;
  constexpr int N_F4 = 128;
  constexpr int N_GRP = 16;

  const int out_row0 = int(tgp_y) * ROWS_PER_TG + int(sg) * ROWS_PER_SG;
  const int n_mtiles = (M + M_TILE - 1) / M_TILE;
  const int g = int(tgp_z) / n_mtiles;
  const int row0 = (int(tgp_z) % n_mtiles) * M_TILE;
  if (out_row0 >= N || row0 >= M || g >= G) return;

  const int r0 = out_row0;
  const int r1 = metal::min(out_row0 + 1, N - 1);
  const size_t wrow0 = size_t(g) * size_t(N) + size_t(r0);
  const size_t wrow1 = size_t(g) * size_t(N) + size_t(r1);
  const int grp = int(lane >> 1);
  const int f4_idx = int(lane) * 4;

  const device uint4* w_ptr0 =
      reinterpret_cast<const device uint4*>(w) + wrow0 * size_t(N_U4);
  const device uint4* w_ptr1 =
      reinterpret_cast<const device uint4*>(w) + wrow1 * size_t(N_U4);
  const auto s_row0 = scales + wrow0 * size_t(N_GRP);
  const auto s_row1 = scales + wrow1 * size_t(N_GRP);

  const uint4 w0_u = w_ptr0[lane];
  const float sc0 = float(s_row0[grp]);
  const uint4 w1_u = w_ptr1[lane];
  const float sc1 = float(s_row1[grp]);

  const device float4* x_base = reinterpret_cast<const device float4*>(x);
  float acc[M_TILE][2];

  if constexpr (SYM_F16) {
    // Int8 weights are exact in FP16: holding both rows as half4 uses 16 32-bit
    // regs (half of float4) while sharing each x0..x3 load across both rows.
    const half4 w00 = half4(as_type<char4>(w0_u.x ^ 0x80808080u));
    const half4 w01 = half4(as_type<char4>(w0_u.y ^ 0x80808080u));
    const half4 w02 = half4(as_type<char4>(w0_u.z ^ 0x80808080u));
    const half4 w03 = half4(as_type<char4>(w0_u.w ^ 0x80808080u));
    const half4 w10 = half4(as_type<char4>(w1_u.x ^ 0x80808080u));
    const half4 w11 = half4(as_type<char4>(w1_u.y ^ 0x80808080u));
    const half4 w12 = half4(as_type<char4>(w1_u.z ^ 0x80808080u));
    const half4 w13 = half4(as_type<char4>(w1_u.w ^ 0x80808080u));
#pragma unroll
    for (int r = 0; r < M_TILE; ++r) {
      const int mr = metal::min(row0 + r, M - 1);
      const device float4* xp =
          x_base + (size_t(mr) * size_t(G) + size_t(g)) * size_t(N_F4) +
          size_t(f4_idx);
      const float4 x0 = xp[0];
      const float4 x1 = xp[1];
      const float4 x2 = xp[2];
      const float4 x3 = xp[3];
      const float d0 =
          metal::dot(x0, float4(w00)) + metal::dot(x1, float4(w01)) +
          metal::dot(x2, float4(w02)) + metal::dot(x3, float4(w03));
      const float d1 =
          metal::dot(x0, float4(w10)) + metal::dot(x1, float4(w11)) +
          metal::dot(x2, float4(w12)) + metal::dot(x3, float4(w13));
      acc[r][0] = d0 * sc0;
      acc[r][1] = d1 * sc1;
    }
  } else {
    const float bi0 = float(biases[wrow0 * size_t(N_GRP) + size_t(grp)]);
    const float bi1 = float(biases[wrow1 * size_t(N_GRP) + size_t(grp)]);
    const half4 w00 = half4(as_type<uchar4>(w0_u.x));
    const half4 w01 = half4(as_type<uchar4>(w0_u.y));
    const half4 w02 = half4(as_type<uchar4>(w0_u.z));
    const half4 w03 = half4(as_type<uchar4>(w0_u.w));
    const half4 w10 = half4(as_type<uchar4>(w1_u.x));
    const half4 w11 = half4(as_type<uchar4>(w1_u.y));
    const half4 w12 = half4(as_type<uchar4>(w1_u.z));
    const half4 w13 = half4(as_type<uchar4>(w1_u.w));
#pragma unroll
    for (int r = 0; r < M_TILE; ++r) {
      const int mr = metal::min(row0 + r, M - 1);
      const device float4* xp =
          x_base + (size_t(mr) * size_t(G) + size_t(g)) * size_t(N_F4) +
          size_t(f4_idx);
      const float4 x0 = xp[0];
      const float4 x1 = xp[1];
      const float4 x2 = xp[2];
      const float4 x3 = xp[3];
      const float4 xs = (x0 + x1) + (x2 + x3);
      const float xsum = (xs.x + xs.y) + (xs.z + xs.w);
      const float d0 =
          metal::dot(x0, float4(w00)) + metal::dot(x1, float4(w01)) +
          metal::dot(x2, float4(w02)) + metal::dot(x3, float4(w03));
      const float d1 =
          metal::dot(x0, float4(w10)) + metal::dot(x1, float4(w11)) +
          metal::dot(x2, float4(w12)) + metal::dot(x3, float4(w13));
      acc[r][0] = metal::fma(d0, sc0, xsum * bi0);
      acc[r][1] = metal::fma(d1, sc1, xsum * bi1);
    }
  }

#pragma unroll
  for (int r = 0; r < M_TILE; ++r) {
    float v0 = simd_sum(acc[r][0]);
    float v1 = simd_sum(acc[r][1]);
    const int mr = row0 + r;
    if (lane == 0 && mr < M) {
      const size_t obase = (size_t(mr) * size_t(G) + size_t(g)) * size_t(N);
      const size_t gbase = size_t(mr) * g_s0 + size_t(g) * g_s1;
      if constexpr (GATED) {
        const float g0 = gate[gbase + size_t(out_row0)];
        v0 *= 1.0f / (1.0f + metal::exp(-g0));
      }
      out[obase + size_t(out_row0)] = v0;
      if (out_row0 + 1 < N) {
        if constexpr (GATED) {
          const float g1 = gate[gbase + size_t(out_row0 + 1)];
          v1 *= 1.0f / (1.0f + metal::exp(-g1));
        }
        out[obase + size_t(out_row0 + 1)] = v1;
      }
    }
  }
}
)METAL"