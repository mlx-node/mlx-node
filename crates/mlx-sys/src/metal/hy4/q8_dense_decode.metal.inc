R"METAL(
constexpr int NUM_SG = 4;
constexpr int ROWS_PER_TG = NUM_SG * ROWS_PER_SG;
constexpr int K_AXIS = GROUPED ? 2 : 1;

const int M = x_shape[0];
const int G = GROUPED ? x_shape[1] : 1;
const int K = x_shape[K_AXIS];
const int N = w_shape[0] / G;

const uint lane = thread_index_in_simdgroup;
const uint sg = simdgroup_index_in_threadgroup;
const int tg_row0 = int(threadgroup_position_in_grid.y) * ROWS_PER_TG;
const int out_row0 = tg_row0 + int(sg) * ROWS_PER_SG;

const int n_mtiles = (M + M_TILE - 1) / M_TILE;
const int z = int(threadgroup_position_in_grid.z);
const int g = GROUPED ? (z / n_mtiles) : 0;
const int row0 = (GROUPED ? (z % n_mtiles) : z) * M_TILE;
if (out_row0 >= N || row0 >= M || g >= G) return;

const int n_groups = K >> 5;
const int n_u4 = K >> 4;
const int n_f4 = K >> 2;
const size_t g_wbase = size_t(g) * size_t(N);
const device uint4* w_base = reinterpret_cast<const device uint4*>(w);
const device float4* x_base = reinterpret_cast<const device float4*>(x);

uint x_off[M_TILE];
#pragma unroll
for (int r = 0; r < M_TILE; ++r) {
  const int mr = metal::min(row0 + r, M - 1);
  x_off[r] = uint((size_t(mr) * size_t(G) + size_t(g)) * size_t(n_f4));
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
    const int u4_idx = (base >> 4) + int(lane);
    const int grp = (base >> 5) + int(lane >> 1);
    const uint f4_idx = uint((base >> 2) + int(lane) * 4);
    const device float4* xp = x_base + size_t(x_off[0] + f4_idx);
    const float4 x0 = xp[0];
    const float4 x1 = xp[1];
    const float4 x2 = xp[2];
    const float4 x3 = xp[3];
    float xsum = 0.0f;
    if constexpr (!SYM_F16) {
      const float4 xs = (x0 + x1) + (x2 + x3);
      xsum = (xs.x + xs.y) + (xs.z + xs.w);
    }
    if constexpr (ROWS_PER_SG == 4) {
      // Issue all 4 row loads upfront so 64B/lane of weights stay in flight.
      const size_t wr0 = g_wbase + size_t(metal::min(out_row0 + 0, N - 1));
      const size_t wr1 = g_wbase + size_t(metal::min(out_row0 + 1, N - 1));
      const size_t wr2 = g_wbase + size_t(metal::min(out_row0 + 2, N - 1));
      const size_t wr3 = g_wbase + size_t(metal::min(out_row0 + 3, N - 1));
      const uint4 wu0 = w_base[wr0 * size_t(n_u4) + size_t(u4_idx)];
      const float sc0 = float(scales[wr0 * size_t(n_groups) + size_t(grp)]);
      const uint4 wu1 = w_base[wr1 * size_t(n_u4) + size_t(u4_idx)];
      const float sc1 = float(scales[wr1 * size_t(n_groups) + size_t(grp)]);
      const uint4 wu2 = w_base[wr2 * size_t(n_u4) + size_t(u4_idx)];
      const float sc2 = float(scales[wr2 * size_t(n_groups) + size_t(grp)]);
      const uint4 wu3 = w_base[wr3 * size_t(n_u4) + size_t(u4_idx)];
      const float sc3 = float(scales[wr3 * size_t(n_groups) + size_t(grp)]);
      if constexpr (SYM_F16) {
        const float d0 =
            metal::dot(x0, float4(as_type<char4>(wu0.x ^ 0x80808080u))) +
            metal::dot(x1, float4(as_type<char4>(wu0.y ^ 0x80808080u))) +
            metal::dot(x2, float4(as_type<char4>(wu0.z ^ 0x80808080u))) +
            metal::dot(x3, float4(as_type<char4>(wu0.w ^ 0x80808080u)));
        const float d1 =
            metal::dot(x0, float4(as_type<char4>(wu1.x ^ 0x80808080u))) +
            metal::dot(x1, float4(as_type<char4>(wu1.y ^ 0x80808080u))) +
            metal::dot(x2, float4(as_type<char4>(wu1.z ^ 0x80808080u))) +
            metal::dot(x3, float4(as_type<char4>(wu1.w ^ 0x80808080u)));
        const float d2 =
            metal::dot(x0, float4(as_type<char4>(wu2.x ^ 0x80808080u))) +
            metal::dot(x1, float4(as_type<char4>(wu2.y ^ 0x80808080u))) +
            metal::dot(x2, float4(as_type<char4>(wu2.z ^ 0x80808080u))) +
            metal::dot(x3, float4(as_type<char4>(wu2.w ^ 0x80808080u)));
        const float d3 =
            metal::dot(x0, float4(as_type<char4>(wu3.x ^ 0x80808080u))) +
            metal::dot(x1, float4(as_type<char4>(wu3.y ^ 0x80808080u))) +
            metal::dot(x2, float4(as_type<char4>(wu3.z ^ 0x80808080u))) +
            metal::dot(x3, float4(as_type<char4>(wu3.w ^ 0x80808080u)));
        acc[0][0] = metal::fma(d0, sc0, acc[0][0]);
        acc[0][1] = metal::fma(d1, sc1, acc[0][1]);
        acc[0][2] = metal::fma(d2, sc2, acc[0][2]);
        acc[0][3] = metal::fma(d3, sc3, acc[0][3]);
      } else {
        const float bi0 = float(biases[wr0 * size_t(n_groups) + size_t(grp)]);
        const float bi1 = float(biases[wr1 * size_t(n_groups) + size_t(grp)]);
        const float bi2 = float(biases[wr2 * size_t(n_groups) + size_t(grp)]);
        const float bi3 = float(biases[wr3 * size_t(n_groups) + size_t(grp)]);
        const float d0 =
            metal::dot(x0, float4(as_type<uchar4>(wu0.x))) +
            metal::dot(x1, float4(as_type<uchar4>(wu0.y))) +
            metal::dot(x2, float4(as_type<uchar4>(wu0.z))) +
            metal::dot(x3, float4(as_type<uchar4>(wu0.w)));
        const float d1 =
            metal::dot(x0, float4(as_type<uchar4>(wu1.x))) +
            metal::dot(x1, float4(as_type<uchar4>(wu1.y))) +
            metal::dot(x2, float4(as_type<uchar4>(wu1.z))) +
            metal::dot(x3, float4(as_type<uchar4>(wu1.w)));
        const float d2 =
            metal::dot(x0, float4(as_type<uchar4>(wu2.x))) +
            metal::dot(x1, float4(as_type<uchar4>(wu2.y))) +
            metal::dot(x2, float4(as_type<uchar4>(wu2.z))) +
            metal::dot(x3, float4(as_type<uchar4>(wu2.w)));
        const float d3 =
            metal::dot(x0, float4(as_type<uchar4>(wu3.x))) +
            metal::dot(x1, float4(as_type<uchar4>(wu3.y))) +
            metal::dot(x2, float4(as_type<uchar4>(wu3.z))) +
            metal::dot(x3, float4(as_type<uchar4>(wu3.w)));
        acc[0][0] = metal::fma(d0, sc0, metal::fma(xsum, bi0, acc[0][0]));
        acc[0][1] = metal::fma(d1, sc1, metal::fma(xsum, bi1, acc[0][1]));
        acc[0][2] = metal::fma(d2, sc2, metal::fma(xsum, bi2, acc[0][2]));
        acc[0][3] = metal::fma(d3, sc3, metal::fma(xsum, bi3, acc[0][3]));
      }
    } else {
#pragma unroll
      for (int o = 0; o < ROWS_PER_SG; o += 2) {
        const size_t wr0 = g_wbase + size_t(metal::min(out_row0 + o, N - 1));
        const size_t wr1 = g_wbase + size_t(metal::min(out_row0 + o + 1, N - 1));
        const uint4 wu0 = w_base[wr0 * size_t(n_u4) + size_t(u4_idx)];
        const float sc0 = float(scales[wr0 * size_t(n_groups) + size_t(grp)]);
        const uint4 wu1 = w_base[wr1 * size_t(n_u4) + size_t(u4_idx)];
        const float sc1 = float(scales[wr1 * size_t(n_groups) + size_t(grp)]);
        if constexpr (SYM_F16) {
          const float d0 =
              metal::dot(x0, float4(as_type<char4>(wu0.x ^ 0x80808080u))) +
              metal::dot(x1, float4(as_type<char4>(wu0.y ^ 0x80808080u))) +
              metal::dot(x2, float4(as_type<char4>(wu0.z ^ 0x80808080u))) +
              metal::dot(x3, float4(as_type<char4>(wu0.w ^ 0x80808080u)));
          const float d1 =
              metal::dot(x0, float4(as_type<char4>(wu1.x ^ 0x80808080u))) +
              metal::dot(x1, float4(as_type<char4>(wu1.y ^ 0x80808080u))) +
              metal::dot(x2, float4(as_type<char4>(wu1.z ^ 0x80808080u))) +
              metal::dot(x3, float4(as_type<char4>(wu1.w ^ 0x80808080u)));
          acc[0][o] = metal::fma(d0, sc0, acc[0][o]);
          acc[0][o + 1] = metal::fma(d1, sc1, acc[0][o + 1]);
        } else {
          const float bi0 = float(biases[wr0 * size_t(n_groups) + size_t(grp)]);
          const float bi1 = float(biases[wr1 * size_t(n_groups) + size_t(grp)]);
          const float d0 =
              metal::dot(x0, float4(as_type<uchar4>(wu0.x))) +
              metal::dot(x1, float4(as_type<uchar4>(wu0.y))) +
              metal::dot(x2, float4(as_type<uchar4>(wu0.z))) +
              metal::dot(x3, float4(as_type<uchar4>(wu0.w)));
          const float d1 =
              metal::dot(x0, float4(as_type<uchar4>(wu1.x))) +
              metal::dot(x1, float4(as_type<uchar4>(wu1.y))) +
              metal::dot(x2, float4(as_type<uchar4>(wu1.z))) +
              metal::dot(x3, float4(as_type<uchar4>(wu1.w)));
          acc[0][o] = metal::fma(d0, sc0, metal::fma(xsum, bi0, acc[0][o]));
          acc[0][o + 1] = metal::fma(d1, sc1, metal::fma(xsum, bi1, acc[0][o + 1]));
        }
      }
    }
  }
} else {
  // Multi-row path: int8/uint8 integers are exact in FP16, so unpacking both rows
  // into half4 holds 32 weights in 16 32-bit regs (zero spill), runs integer
  // unpack once outside `for r`, AND shares every x0..x3 load across both rows.
  for (int base = 0; base < k_main; base += 512) {
    const int u4_idx = (base >> 4) + int(lane);
    const int grp = (base >> 5) + int(lane >> 1);
    const uint f4_idx = uint((base >> 2) + int(lane) * 4);
#pragma unroll
    for (int o = 0; o < ROWS_PER_SG; o += 2) {
      const size_t wr0 = g_wbase + size_t(metal::min(out_row0 + o, N - 1));
      const size_t wr1 = g_wbase + size_t(metal::min(out_row0 + o + 1, N - 1));
      const uint4 wu0 = w_base[wr0 * size_t(n_u4) + size_t(u4_idx)];
      const float sc0 = float(scales[wr0 * size_t(n_groups) + size_t(grp)]);
      const uint4 wu1 = w_base[wr1 * size_t(n_u4) + size_t(u4_idx)];
      const float sc1 = float(scales[wr1 * size_t(n_groups) + size_t(grp)]);

      if constexpr (SYM_F16) {
        const half4 w00 = half4(as_type<char4>(wu0.x ^ 0x80808080u));
        const half4 w01 = half4(as_type<char4>(wu0.y ^ 0x80808080u));
        const half4 w02 = half4(as_type<char4>(wu0.z ^ 0x80808080u));
        const half4 w03 = half4(as_type<char4>(wu0.w ^ 0x80808080u));
        const half4 w10 = half4(as_type<char4>(wu1.x ^ 0x80808080u));
        const half4 w11 = half4(as_type<char4>(wu1.y ^ 0x80808080u));
        const half4 w12 = half4(as_type<char4>(wu1.z ^ 0x80808080u));
        const half4 w13 = half4(as_type<char4>(wu1.w ^ 0x80808080u));
#pragma unroll
        for (int r = 0; r < M_TILE; ++r) {
          const device float4* xp = x_base + size_t(x_off[r] + f4_idx);
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
          acc[r][o] = metal::fma(d0, sc0, acc[r][o]);
          acc[r][o + 1] = metal::fma(d1, sc1, acc[r][o + 1]);
        }
      } else {
        const float bi0 = float(biases[wr0 * size_t(n_groups) + size_t(grp)]);
        const float bi1 = float(biases[wr1 * size_t(n_groups) + size_t(grp)]);
        const half4 w00 = half4(as_type<uchar4>(wu0.x));
        const half4 w01 = half4(as_type<uchar4>(wu0.y));
        const half4 w02 = half4(as_type<uchar4>(wu0.z));
        const half4 w03 = half4(as_type<uchar4>(wu0.w));
        const half4 w10 = half4(as_type<uchar4>(wu1.x));
        const half4 w11 = half4(as_type<uchar4>(wu1.y));
        const half4 w12 = half4(as_type<uchar4>(wu1.z));
        const half4 w13 = half4(as_type<uchar4>(wu1.w));
#pragma unroll
        for (int r = 0; r < M_TILE; ++r) {
          const device float4* xp = x_base + size_t(x_off[r] + f4_idx);
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
          acc[r][o] = metal::fma(d0, sc0, metal::fma(xsum, bi0, acc[r][o]));
          acc[r][o + 1] = metal::fma(d1, sc1, metal::fma(xsum, bi1, acc[r][o + 1]));
        }
      }
    }
  }
}

if constexpr (!FAST_K) {
  const int col = k_main + int(lane) * 16;
  if (col < K) {
    const int u4_idx = col >> 4;
    const int grp = col >> 5;
    const uint f4_idx = uint(col >> 2);
#pragma unroll
    for (int o = 0; o < ROWS_PER_SG; o += 2) {
      const size_t wr0 = g_wbase + size_t(metal::min(out_row0 + o, N - 1));
      const size_t wr1 = g_wbase + size_t(metal::min(out_row0 + o + 1, N - 1));
      const uint4 wu0 = w_base[wr0 * size_t(n_u4) + size_t(u4_idx)];
      const float sc0 = float(scales[wr0 * size_t(n_groups) + size_t(grp)]);
      const uint4 wu1 = w_base[wr1 * size_t(n_u4) + size_t(u4_idx)];
      const float sc1 = float(scales[wr1 * size_t(n_groups) + size_t(grp)]);

      if constexpr (SYM_F16) {
        const half4 w00 = half4(as_type<char4>(wu0.x ^ 0x80808080u));
        const half4 w01 = half4(as_type<char4>(wu0.y ^ 0x80808080u));
        const half4 w02 = half4(as_type<char4>(wu0.z ^ 0x80808080u));
        const half4 w03 = half4(as_type<char4>(wu0.w ^ 0x80808080u));
        const half4 w10 = half4(as_type<char4>(wu1.x ^ 0x80808080u));
        const half4 w11 = half4(as_type<char4>(wu1.y ^ 0x80808080u));
        const half4 w12 = half4(as_type<char4>(wu1.z ^ 0x80808080u));
        const half4 w13 = half4(as_type<char4>(wu1.w ^ 0x80808080u));
#pragma unroll
        for (int r = 0; r < M_TILE; ++r) {
          const device float4* xp = x_base + size_t(x_off[r] + f4_idx);
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
          acc[r][o] = metal::fma(d0, sc0, acc[r][o]);
          acc[r][o + 1] = metal::fma(d1, sc1, acc[r][o + 1]);
        }
      } else {
        const float bi0 = float(biases[wr0 * size_t(n_groups) + size_t(grp)]);
        const float bi1 = float(biases[wr1 * size_t(n_groups) + size_t(grp)]);
        const half4 w00 = half4(as_type<uchar4>(wu0.x));
        const half4 w01 = half4(as_type<uchar4>(wu0.y));
        const half4 w02 = half4(as_type<uchar4>(wu0.z));
        const half4 w03 = half4(as_type<uchar4>(wu0.w));
        const half4 w10 = half4(as_type<uchar4>(wu1.x));
        const half4 w11 = half4(as_type<uchar4>(wu1.y));
        const half4 w12 = half4(as_type<uchar4>(wu1.z));
        const half4 w13 = half4(as_type<uchar4>(wu1.w));
#pragma unroll
        for (int r = 0; r < M_TILE; ++r) {
          const device float4* xp = x_base + size_t(x_off[r] + f4_idx);
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
          acc[r][o] = metal::fma(d0, sc0, metal::fma(xsum, bi0, acc[r][o]));
          acc[r][o + 1] = metal::fma(d1, sc1, metal::fma(xsum, bi1, acc[r][o + 1]));
        }
      }
    }
  }
}

#pragma unroll
for (int r = 0; r < M_TILE; ++r) {
  const int mr = row0 + r;
  const size_t obase = (size_t(mr) * size_t(G) + size_t(g)) * size_t(N);
#pragma unroll
  for (int o = 0; o < ROWS_PER_SG; ++o) {
    const float v = simd_sum(acc[r][o]);
    if (lane == 0 && mr < M && out_row0 + o < N) {
      out[obase + size_t(out_row0 + o)] = v;
    }
  }
}
)METAL"