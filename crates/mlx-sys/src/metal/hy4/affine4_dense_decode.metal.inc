R"METAL(
inline float4 hy4_aff4_dense_prescale4(float4 v) {
  return float4(
      v[0],
      v[1] * (1.0f / 16.0f),
      v[2] * (1.0f / 256.0f),
      v[3] * (1.0f / 4096.0f));
}

inline float hy4_aff4_dense_dot4(float4 xs, uint16_t p4) {
  float a = xs[0] * float(p4 & 0x000Fu);
  a = metal::fma(xs[1], float(p4 & 0x00F0u), a);
  a = metal::fma(xs[2], float(p4 & 0x0F00u), a);
  a = metal::fma(xs[3], float(p4 & 0xF000u), a);
  return a;
}

inline float hy4_aff4_dense_dot16(
    uint2 w2,
    float4 x0s,
    float4 x1s,
    float4 x2s,
    float4 x3s) {
  const float d0 = hy4_aff4_dense_dot4(x0s, uint16_t(w2.x & 0xFFFFu));
  const float d1 = hy4_aff4_dense_dot4(x1s, uint16_t(w2.x >> 16));
  const float d2 = hy4_aff4_dense_dot4(x2s, uint16_t(w2.y & 0xFFFFu));
  const float d3 = hy4_aff4_dense_dot4(x3s, uint16_t(w2.y >> 16));
  return (d0 + d1) + (d2 + d3);
}

inline float hy4_aff4_dense_sum16(float4 x0, float4 x1, float4 x2, float4 x3) {
  const float4 xs = (x0 + x1) + (x2 + x3);
  return (xs.x + xs.y) + (xs.z + xs.w);
}

template <
    int M_TILE,
    int ROWS_PER_SG,
    bool FAST_K,
    typename ScalePtr,
    typename BiasPtr>
inline void hy4_affine4_dense_decode_impl(
    const device float* x,
    const device uint32_t* w,
    ScalePtr scales,
    BiasPtr biases,
    device float* out,
    int M,
    int N,
    int K,
    uint tg_y,
    uint tg_z,
    uint sg,
    uint lane) {
  constexpr int NUM_SG = 4;
  constexpr int ROWS_PER_TG = NUM_SG * ROWS_PER_SG;

  const int out_row0 = int(tg_y) * ROWS_PER_TG + int(sg) * ROWS_PER_SG;
  const int row0 = int(tg_z) * M_TILE;
  if (out_row0 >= N || row0 >= M) return;

  const int n_groups = K >> 6;
  const int n_u2 = K >> 4;
  const int n_f4 = K >> 2;
  const device uint2* w_base = reinterpret_cast<const device uint2*>(w);
  const device float4* x_base = reinterpret_cast<const device float4*>(x);

  size_t x_off[M_TILE];
#pragma unroll
  for (int r = 0; r < M_TILE; ++r) {
    const int mr = metal::min(row0 + r, M - 1);
    x_off[r] = size_t(mr) * size_t(n_f4);
  }

  float acc[M_TILE][ROWS_PER_SG];
#pragma unroll
  for (int r = 0; r < M_TILE; ++r) {
#pragma unroll
    for (int o = 0; o < ROWS_PER_SG; ++o) {
      acc[r][o] = 0.0f;
    }
  }

  const int k_main = FAST_K ? K : (K & ~511);

  if constexpr (M_TILE == 1) {
    for (int base = 0; base < k_main; base += 512) {
      const int u2_idx = (base >> 4) + int(lane);
      const int grp = (base >> 6) + int(lane >> 2);
      const size_t f4_idx = size_t((base >> 2) + int(lane) * 4);
      const device float4* xp = x_base + x_off[0] + f4_idx;
      const float4 x0 = xp[0];
      const float4 x1 = xp[1];
      const float4 x2 = xp[2];
      const float4 x3 = xp[3];
      const float xsum = hy4_aff4_dense_sum16(x0, x1, x2, x3);
      const float4 x0s = hy4_aff4_dense_prescale4(x0);
      const float4 x1s = hy4_aff4_dense_prescale4(x1);
      const float4 x2s = hy4_aff4_dense_prescale4(x2);
      const float4 x3s = hy4_aff4_dense_prescale4(x3);

      if constexpr (ROWS_PER_SG == 4) {
        const size_t wr0 = size_t(metal::min(out_row0 + 0, N - 1));
        const size_t wr1 = size_t(metal::min(out_row0 + 1, N - 1));
        const size_t wr2 = size_t(metal::min(out_row0 + 2, N - 1));
        const size_t wr3 = size_t(metal::min(out_row0 + 3, N - 1));
        const uint2 wu0 = w_base[wr0 * size_t(n_u2) + size_t(u2_idx)];
        const float sc0 = float(scales[wr0 * size_t(n_groups) + size_t(grp)]);
        const float bi0 = float(biases[wr0 * size_t(n_groups) + size_t(grp)]);
        const uint2 wu1 = w_base[wr1 * size_t(n_u2) + size_t(u2_idx)];
        const float sc1 = float(scales[wr1 * size_t(n_groups) + size_t(grp)]);
        const float bi1 = float(biases[wr1 * size_t(n_groups) + size_t(grp)]);
        const uint2 wu2 = w_base[wr2 * size_t(n_u2) + size_t(u2_idx)];
        const float sc2 = float(scales[wr2 * size_t(n_groups) + size_t(grp)]);
        const float bi2 = float(biases[wr2 * size_t(n_groups) + size_t(grp)]);
        const uint2 wu3 = w_base[wr3 * size_t(n_u2) + size_t(u2_idx)];
        const float sc3 = float(scales[wr3 * size_t(n_groups) + size_t(grp)]);
        const float bi3 = float(biases[wr3 * size_t(n_groups) + size_t(grp)]);

        const float d0 = hy4_aff4_dense_dot16(wu0, x0s, x1s, x2s, x3s);
        const float d1 = hy4_aff4_dense_dot16(wu1, x0s, x1s, x2s, x3s);
        const float d2 = hy4_aff4_dense_dot16(wu2, x0s, x1s, x2s, x3s);
        const float d3 = hy4_aff4_dense_dot16(wu3, x0s, x1s, x2s, x3s);
        acc[0][0] = metal::fma(sc0, d0, metal::fma(bi0, xsum, acc[0][0]));
        acc[0][1] = metal::fma(sc1, d1, metal::fma(bi1, xsum, acc[0][1]));
        acc[0][2] = metal::fma(sc2, d2, metal::fma(bi2, xsum, acc[0][2]));
        acc[0][3] = metal::fma(sc3, d3, metal::fma(bi3, xsum, acc[0][3]));
      } else {
#pragma unroll
        for (int o = 0; o < ROWS_PER_SG; o += 2) {
          const size_t wr0 = size_t(metal::min(out_row0 + o, N - 1));
          const size_t wr1 = size_t(metal::min(out_row0 + o + 1, N - 1));
          const uint2 wu0 = w_base[wr0 * size_t(n_u2) + size_t(u2_idx)];
          const float sc0 = float(scales[wr0 * size_t(n_groups) + size_t(grp)]);
          const float bi0 = float(biases[wr0 * size_t(n_groups) + size_t(grp)]);
          const uint2 wu1 = w_base[wr1 * size_t(n_u2) + size_t(u2_idx)];
          const float sc1 = float(scales[wr1 * size_t(n_groups) + size_t(grp)]);
          const float bi1 = float(biases[wr1 * size_t(n_groups) + size_t(grp)]);
          const float d0 = hy4_aff4_dense_dot16(wu0, x0s, x1s, x2s, x3s);
          const float d1 = hy4_aff4_dense_dot16(wu1, x0s, x1s, x2s, x3s);
          acc[0][o] = metal::fma(sc0, d0, metal::fma(bi0, xsum, acc[0][o]));
          acc[0][o + 1] = metal::fma(sc1, d1, metal::fma(bi1, xsum, acc[0][o + 1]));
        }
      }
    }
  } else {
    for (int base = 0; base < k_main; base += 512) {
      const int u2_idx = (base >> 4) + int(lane);
      const int grp = (base >> 6) + int(lane >> 2);
      const size_t f4_idx = size_t((base >> 2) + int(lane) * 4);
#pragma unroll
      for (int o = 0; o < ROWS_PER_SG; o += 2) {
        const size_t wr0 = size_t(metal::min(out_row0 + o, N - 1));
        const size_t wr1 = size_t(metal::min(out_row0 + o + 1, N - 1));
        const uint2 wu0 = w_base[wr0 * size_t(n_u2) + size_t(u2_idx)];
        const float sc0 = float(scales[wr0 * size_t(n_groups) + size_t(grp)]);
        const float bi0 = float(biases[wr0 * size_t(n_groups) + size_t(grp)]);
        const uint2 wu1 = w_base[wr1 * size_t(n_u2) + size_t(u2_idx)];
        const float sc1 = float(scales[wr1 * size_t(n_groups) + size_t(grp)]);
        const float bi1 = float(biases[wr1 * size_t(n_groups) + size_t(grp)]);
#pragma unroll
        for (int r = 0; r < M_TILE; ++r) {
          const device float4* xp = x_base + x_off[r] + f4_idx;
          const float4 x0 = xp[0];
          const float4 x1 = xp[1];
          const float4 x2 = xp[2];
          const float4 x3 = xp[3];
          const float xsum = hy4_aff4_dense_sum16(x0, x1, x2, x3);
          const float4 x0s = hy4_aff4_dense_prescale4(x0);
          const float4 x1s = hy4_aff4_dense_prescale4(x1);
          const float4 x2s = hy4_aff4_dense_prescale4(x2);
          const float4 x3s = hy4_aff4_dense_prescale4(x3);
          const float d0 = hy4_aff4_dense_dot16(wu0, x0s, x1s, x2s, x3s);
          const float d1 = hy4_aff4_dense_dot16(wu1, x0s, x1s, x2s, x3s);
          acc[r][o] = metal::fma(sc0, d0, metal::fma(bi0, xsum, acc[r][o]));
          acc[r][o + 1] = metal::fma(sc1, d1, metal::fma(bi1, xsum, acc[r][o + 1]));
        }
      }
    }
  }

  if constexpr (!FAST_K) {
    const int col = k_main + int(lane) * 16;
    if (col < K) {
      const int u2_idx = col >> 4;
      const int grp = col >> 6;
      const size_t f4_idx = size_t(col >> 2);
#pragma unroll
      for (int o = 0; o < ROWS_PER_SG; o += 2) {
        const size_t wr0 = size_t(metal::min(out_row0 + o, N - 1));
        const size_t wr1 = size_t(metal::min(out_row0 + o + 1, N - 1));
        const uint2 wu0 = w_base[wr0 * size_t(n_u2) + size_t(u2_idx)];
        const float sc0 = float(scales[wr0 * size_t(n_groups) + size_t(grp)]);
        const float bi0 = float(biases[wr0 * size_t(n_groups) + size_t(grp)]);
        const uint2 wu1 = w_base[wr1 * size_t(n_u2) + size_t(u2_idx)];
        const float sc1 = float(scales[wr1 * size_t(n_groups) + size_t(grp)]);
        const float bi1 = float(biases[wr1 * size_t(n_groups) + size_t(grp)]);
#pragma unroll
        for (int r = 0; r < M_TILE; ++r) {
          const device float4* xp = x_base + x_off[r] + f4_idx;
          const float4 x0 = xp[0];
          const float4 x1 = xp[1];
          const float4 x2 = xp[2];
          const float4 x3 = xp[3];
          const float xsum = hy4_aff4_dense_sum16(x0, x1, x2, x3);
          const float4 x0s = hy4_aff4_dense_prescale4(x0);
          const float4 x1s = hy4_aff4_dense_prescale4(x1);
          const float4 x2s = hy4_aff4_dense_prescale4(x2);
          const float4 x3s = hy4_aff4_dense_prescale4(x3);
          const float d0 = hy4_aff4_dense_dot16(wu0, x0s, x1s, x2s, x3s);
          const float d1 = hy4_aff4_dense_dot16(wu1, x0s, x1s, x2s, x3s);
          acc[r][o] = metal::fma(sc0, d0, metal::fma(bi0, xsum, acc[r][o]));
          acc[r][o + 1] = metal::fma(sc1, d1, metal::fma(bi1, xsum, acc[r][o + 1]));
        }
      }
    }
  }

#pragma unroll
  for (int r = 0; r < M_TILE; ++r) {
    const int mr = row0 + r;
    const size_t obase = size_t(mr) * size_t(N);
#pragma unroll
    for (int o = 0; o < ROWS_PER_SG; ++o) {
      const float v = simd_sum(acc[r][o]);
      if (lane == 0u && mr < M && out_row0 + o < N) {
        out[obase + size_t(out_row0 + o)] = v;
      }
    }
  }
}
)METAL"
