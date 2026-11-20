R"METAL(
constexpr int K_LANES = 8;
constexpr int ROWS_PER_SG = 32 / K_LANES;
constexpr int NUM_SG = 2;
constexpr int ROWS_PER_TG = ROWS_PER_SG * NUM_SG;
constexpr int K_AXIS = GROUPED ? 2 : 1;

const int M = x_shape[0];
const int G = GROUPED ? x_shape[1] : 1;
const int K = x_shape[K_AXIS];
const int N = w_shape[0] / G;

const uint lane = thread_index_in_simdgroup;
const uint k_lane = lane & 7u;
const uint sg_row = lane >> 3;
const int out_row = int(threadgroup_position_in_grid.y) * ROWS_PER_TG +
    int(simdgroup_index_in_threadgroup) * ROWS_PER_SG + int(sg_row);

const int n_rtiles = (M + R_TILE - 1) / R_TILE;
const int z = int(threadgroup_position_in_grid.z);
const int g = GROUPED ? (z / n_rtiles) : 0;
const int row0 = (GROUPED ? (z % n_rtiles) : z) * R_TILE;
if (row0 >= M || g >= G) return;

// Clamp out-of-bounds lanes to a valid row so simd_shuffle_down stays active.
const int row = metal::min(out_row, N - 1);
const size_t wrow = size_t(g) * size_t(N) + size_t(row);
const int n_groups = K >> 5;
const int n_f4 = K >> 2;

const device uchar4* w4 =
    reinterpret_cast<const device uchar4*>(w) + wrow * size_t(n_f4);
const auto s_row = scales + wrow * size_t(n_groups);

const device float4* x_row0 = reinterpret_cast<const device float4*>(x) +
    (size_t(row0) * size_t(G) + size_t(g)) * size_t(n_f4);
const size_t x_row_stride = size_t(G) * size_t(n_f4);
const int max_r = M - 1 - row0;

float acc[R_TILE];
#pragma unroll
for (int r = 0; r < R_TILE; ++r) {
  acc[r] = 0.0f;
}

for (int grp = int(k_lane); grp < n_groups; grp += K_LANES) {
  const float sc = float(s_row[grp]);
  float bi;
  if constexpr (SYM_F16) {
    bi = -128.0f * sc;
  } else {
    bi = float(biases[wrow * size_t(n_groups) + size_t(grp)]);
  }
  const int f4_base = grp * 8;
#pragma unroll
  for (int sub = 0; sub < 4; ++sub) {
    const int idx = f4_base + sub * 2;
    const float4 w0 = metal::fma(float4(w4[idx]), float4(sc), float4(bi));
    const float4 w1 = metal::fma(float4(w4[idx + 1]), float4(sc), float4(bi));
    const device float4* xr = x_row0 + size_t(idx);
#pragma unroll
    for (int r = 0; r < R_TILE; ++r) {
      acc[r] += metal::dot(xr[0], w0) + metal::dot(xr[1], w1);
      if (r < max_r) {
        xr += x_row_stride;
      }
    }
  }
}

#pragma unroll
for (int r = 0; r < R_TILE; ++r) {
  acc[r] += simd_shuffle_down(acc[r], 4);
  acc[r] += simd_shuffle_down(acc[r], 2);
  acc[r] += simd_shuffle_down(acc[r], 1);
}

if (k_lane == 0 && out_row < N) {
#pragma unroll
  for (int r = 0; r < R_TILE; ++r) {
    const int m_idx = row0 + r;
    if (m_idx < M) {
      out[(size_t(m_idx) * size_t(G) + size_t(g)) * size_t(N) +
          size_t(out_row)] = acc[r];
    }
  }
}
)METAL"
