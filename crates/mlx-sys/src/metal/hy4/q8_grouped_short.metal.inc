R"METAL(
// Short-row grouped SymQ8 decode helpers (K = 192 half-warp, K = 512 full-warp).
template <int M_TILE, typename ScalePtr>
inline void hy4_q8_grouped_k192_impl(
    const device float* x,
    const device uint32_t* w,
    ScalePtr scales,
    device float* out,
    int M,
    int G,
    int N,
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
  if (out_row0 >= N || row0 >= M || g >= G) return;

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

  const float4 w0_0 = float4(as_type<char4>(u0_0 ^ 0x80808080u)) * s0_0;
  const float4 w0_1 = float4(as_type<char4>(u0_1 ^ 0x80808080u)) * s0_1;
  const float4 w0_2 = float4(as_type<char4>(u0_2 ^ 0x80808080u)) * s0_2;

  const float4 w1_0 = float4(as_type<char4>(u1_0 ^ 0x80808080u)) * s1_0;
  const float4 w1_1 = float4(as_type<char4>(u1_1 ^ 0x80808080u)) * s1_1;
  const float4 w1_2 = float4(as_type<char4>(u1_2 ^ 0x80808080u)) * s1_2;

  const device float4* x_base = reinterpret_cast<const device float4*>(x);
  float acc[M_TILE][2];
#pragma unroll
  for (int r = 0; r < M_TILE; ++r) {
    const int mr = metal::min(row0 + r, M - 1);
    const device float4* xp =
        x_base + (size_t(mr) * size_t(G) + size_t(g)) * size_t(N_F4) +
        size_t(k0);
    const float4 x0 = xp[0];
    const float4 x1 = xp[1];
    const float4 x2 = xp[2];
    acc[r][0] = metal::dot(x0, w0_0) + metal::dot(x1, w0_1) + metal::dot(x2, w0_2);
    acc[r][1] = metal::dot(x0, w1_0) + metal::dot(x1, w1_1) + metal::dot(x2, w1_2);
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
      const size_t obase = (size_t(mr) * size_t(G) + size_t(g)) * size_t(N);
      out[obase + size_t(out_row0)] = v0;
      if (out_row0 + 1 < N) {
        out[obase + size_t(out_row0 + 1)] = v1;
      }
    }
  }
}

template <int M_TILE, typename ScalePtr>
inline void hy4_q8_grouped_k512_impl(
    const device float* x,
    const device uint32_t* w,
    ScalePtr scales,
    device float* out,
    int M,
    int G,
    int N,
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

  const float4 q0_0 = float4(as_type<char4>(w0_u.x ^ 0x80808080u));
  const float4 q0_1 = float4(as_type<char4>(w0_u.y ^ 0x80808080u));
  const float4 q0_2 = float4(as_type<char4>(w0_u.z ^ 0x80808080u));
  const float4 q0_3 = float4(as_type<char4>(w0_u.w ^ 0x80808080u));

  const float4 q1_0 = float4(as_type<char4>(w1_u.x ^ 0x80808080u));
  const float4 q1_1 = float4(as_type<char4>(w1_u.y ^ 0x80808080u));
  const float4 q1_2 = float4(as_type<char4>(w1_u.z ^ 0x80808080u));
  const float4 q1_3 = float4(as_type<char4>(w1_u.w ^ 0x80808080u));

  const device float4* x_base = reinterpret_cast<const device float4*>(x);
  float acc[M_TILE][2];
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
    const float d0 = metal::dot(x0, q0_0) + metal::dot(x1, q0_1) +
                     metal::dot(x2, q0_2) + metal::dot(x3, q0_3);
    const float d1 = metal::dot(x0, q1_0) + metal::dot(x1, q1_1) +
                     metal::dot(x2, q1_2) + metal::dot(x3, q1_3);
    acc[r][0] = d0 * sc0;
    acc[r][1] = d1 * sc1;
  }

#pragma unroll
  for (int r = 0; r < M_TILE; ++r) {
    const float v0 = simd_sum(acc[r][0]);
    const float v1 = simd_sum(acc[r][1]);
    const int mr = row0 + r;
    if (lane == 0 && mr < M) {
      const size_t obase = (size_t(mr) * size_t(G) + size_t(g)) * size_t(N);
      out[obase + size_t(out_row0)] = v0;
      if (out_row0 + 1 < N) {
        out[obase + size_t(out_row0 + 1)] = v1;
      }
    }
  }
}
)METAL"
