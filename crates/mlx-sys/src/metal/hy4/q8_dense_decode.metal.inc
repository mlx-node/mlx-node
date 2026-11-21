R"METAL(
constexpr int NUM_SG = 4;
constexpr int ROWS_PER_SG = 2;
constexpr int ROWS_PER_TG = NUM_SG * ROWS_PER_SG;
constexpr int K_AXIS = GROUPED ? 2 : 1;

const int M = x_shape[0];
const int G = GROUPED ? x_shape[1] : 1;
const int K = x_shape[K_AXIS];
const int N = w_shape[0] / G;

const uint lane = thread_index_in_simdgroup;
const uint sg = simdgroup_index_in_threadgroup;
const uint tid = sg * 32u + lane;
const int tg_row0 = int(threadgroup_position_in_grid.y) * ROWS_PER_TG;
const int out_row0 = tg_row0 + int(sg) * ROWS_PER_SG;

const int n_mtiles = (M + M_TILE - 1) / M_TILE;
const int z = int(threadgroup_position_in_grid.z);
const int g = GROUPED ? (z / n_mtiles) : 0;
const int row0 = (GROUPED ? (z % n_mtiles) : z) * M_TILE;
// Threadgroup-uniform early exit so all 4 simdgroups hit barriers together.
if (tg_row0 >= N || row0 >= M || g >= G) return;

const int r0 = metal::min(out_row0, N - 1);
const int r1 = metal::min(out_row0 + 1, N - 1);
const size_t wrow0 = size_t(g) * size_t(N) + size_t(r0);
const size_t wrow1 = size_t(g) * size_t(N) + size_t(r1);
const int n_groups = K >> 5;
const int n_u4 = K >> 4;
const int n_f4 = K >> 2;

const device uint4* w_ptr0 =
    reinterpret_cast<const device uint4*>(w) + wrow0 * size_t(n_u4);
const device uint4* w_ptr1 =
    reinterpret_cast<const device uint4*>(w) + wrow1 * size_t(n_u4);
const auto s_row0 = scales + wrow0 * size_t(n_groups);
const auto s_row1 = scales + wrow1 * size_t(n_groups);

const device float4* x_base = reinterpret_cast<const device float4*>(x);
uint x_off[M_TILE];
#pragma unroll
for (int r = 0; r < M_TILE; ++r) {
  const int mr = metal::min(row0 + r, M - 1);
  x_off[r] = uint((size_t(mr) * size_t(G) + size_t(g)) * size_t(n_f4));
}

float acc[M_TILE][2];
#pragma unroll
for (int r = 0; r < M_TILE; ++r) {
  acc[r][0] = 0.0f;
  acc[r][1] = 0.0f;
}

const int k_main = FAST_K ? K : (K & ~511);

if constexpr (USE_TG_X) {
  // Planar [M_TILE][16][32] layout gives stride-1 float per lane (zero bank conflicts).
  threadgroup float x_tg[M_TILE][16][32];
  const uint x_j = (tid & 3u) * 4u;
  const uint x_l = tid >> 2;

  for (int base = 0; base < k_main; base += 512) {
    const uint f4_tid = uint((base >> 2) + int(tid));
#pragma unroll
    for (int r = 0; r < M_TILE; ++r) {
      const float4 xv = x_base[size_t(x_off[r] + f4_tid)];
      x_tg[r][x_j + 0u][x_l] = xv.x;
      x_tg[r][x_j + 1u][x_l] = xv.y;
      x_tg[r][x_j + 2u][x_l] = xv.z;
      x_tg[r][x_j + 3u][x_l] = xv.w;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int u4_idx = (base >> 4) + int(lane);
    const int grp = (base >> 5) + int(lane >> 1);
    const uint4 wu0 = w_ptr0[u4_idx];
    const float sc0 = float(s_row0[grp]);
    const uint4 wu1 = w_ptr1[u4_idx];
    const float sc1 = float(s_row1[grp]);

    if constexpr (SYM_F16) {
      {
        const float4 q0 = float4(as_type<char4>(wu0.x ^ 0x80808080u));
        const float4 q1 = float4(as_type<char4>(wu0.y ^ 0x80808080u));
        const float4 q2 = float4(as_type<char4>(wu0.z ^ 0x80808080u));
        const float4 q3 = float4(as_type<char4>(wu0.w ^ 0x80808080u));
#pragma unroll
        for (int r = 0; r < M_TILE; ++r) {
          const float4 x0 = float4(x_tg[r][0][lane], x_tg[r][1][lane], x_tg[r][2][lane], x_tg[r][3][lane]);
          const float4 x1 = float4(x_tg[r][4][lane], x_tg[r][5][lane], x_tg[r][6][lane], x_tg[r][7][lane]);
          const float4 x2 = float4(x_tg[r][8][lane], x_tg[r][9][lane], x_tg[r][10][lane], x_tg[r][11][lane]);
          const float4 x3 = float4(x_tg[r][12][lane], x_tg[r][13][lane], x_tg[r][14][lane], x_tg[r][15][lane]);
          const float d = metal::dot(x0, q0) + metal::dot(x1, q1) +
                          metal::dot(x2, q2) + metal::dot(x3, q3);
          acc[r][0] = metal::fma(d, sc0, acc[r][0]);
        }
      }
      {
        const float4 q0 = float4(as_type<char4>(wu1.x ^ 0x80808080u));
        const float4 q1 = float4(as_type<char4>(wu1.y ^ 0x80808080u));
        const float4 q2 = float4(as_type<char4>(wu1.z ^ 0x80808080u));
        const float4 q3 = float4(as_type<char4>(wu1.w ^ 0x80808080u));
#pragma unroll
        for (int r = 0; r < M_TILE; ++r) {
          const float4 x0 = float4(x_tg[r][0][lane], x_tg[r][1][lane], x_tg[r][2][lane], x_tg[r][3][lane]);
          const float4 x1 = float4(x_tg[r][4][lane], x_tg[r][5][lane], x_tg[r][6][lane], x_tg[r][7][lane]);
          const float4 x2 = float4(x_tg[r][8][lane], x_tg[r][9][lane], x_tg[r][10][lane], x_tg[r][11][lane]);
          const float4 x3 = float4(x_tg[r][12][lane], x_tg[r][13][lane], x_tg[r][14][lane], x_tg[r][15][lane]);
          const float d = metal::dot(x0, q0) + metal::dot(x1, q1) +
                          metal::dot(x2, q2) + metal::dot(x3, q3);
          acc[r][1] = metal::fma(d, sc1, acc[r][1]);
        }
      }
    } else {
      const float bi0 = float(biases[wrow0 * size_t(n_groups) + size_t(grp)]);
      const float bi1 = float(biases[wrow1 * size_t(n_groups) + size_t(grp)]);
      {
        const float4 q0 = float4(as_type<uchar4>(wu0.x));
        const float4 q1 = float4(as_type<uchar4>(wu0.y));
        const float4 q2 = float4(as_type<uchar4>(wu0.z));
        const float4 q3 = float4(as_type<uchar4>(wu0.w));
#pragma unroll
        for (int r = 0; r < M_TILE; ++r) {
          const float4 x0 = float4(x_tg[r][0][lane], x_tg[r][1][lane], x_tg[r][2][lane], x_tg[r][3][lane]);
          const float4 x1 = float4(x_tg[r][4][lane], x_tg[r][5][lane], x_tg[r][6][lane], x_tg[r][7][lane]);
          const float4 x2 = float4(x_tg[r][8][lane], x_tg[r][9][lane], x_tg[r][10][lane], x_tg[r][11][lane]);
          const float4 x3 = float4(x_tg[r][12][lane], x_tg[r][13][lane], x_tg[r][14][lane], x_tg[r][15][lane]);
          // Exact binary-tree reduction order matches the non-TG path bit-for-bit.
          const float4 xs = (x0 + x1) + (x2 + x3);
          const float xsum = (xs.x + xs.y) + (xs.z + xs.w);
          const float d = metal::dot(x0, q0) + metal::dot(x1, q1) +
                          metal::dot(x2, q2) + metal::dot(x3, q3);
          acc[r][0] = metal::fma(d, sc0, metal::fma(xsum, bi0, acc[r][0]));
        }
      }
      {
        const float4 q0 = float4(as_type<uchar4>(wu1.x));
        const float4 q1 = float4(as_type<uchar4>(wu1.y));
        const float4 q2 = float4(as_type<uchar4>(wu1.z));
        const float4 q3 = float4(as_type<uchar4>(wu1.w));
#pragma unroll
        for (int r = 0; r < M_TILE; ++r) {
          const float4 x0 = float4(x_tg[r][0][lane], x_tg[r][1][lane], x_tg[r][2][lane], x_tg[r][3][lane]);
          const float4 x1 = float4(x_tg[r][4][lane], x_tg[r][5][lane], x_tg[r][6][lane], x_tg[r][7][lane]);
          const float4 x2 = float4(x_tg[r][8][lane], x_tg[r][9][lane], x_tg[r][10][lane], x_tg[r][11][lane]);
          const float4 x3 = float4(x_tg[r][12][lane], x_tg[r][13][lane], x_tg[r][14][lane], x_tg[r][15][lane]);
          const float4 xs = (x0 + x1) + (x2 + x3);
          const float xsum = (xs.x + xs.y) + (xs.z + xs.w);
          const float d = metal::dot(x0, q0) + metal::dot(x1, q1) +
                          metal::dot(x2, q2) + metal::dot(x3, q3);
          acc[r][1] = metal::fma(d, sc1, metal::fma(xsum, bi1, acc[r][1]));
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
} else {
  for (int base = 0; base < k_main; base += 512) {
    const int u4_idx = (base >> 4) + int(lane);
    const int grp = (base >> 5) + int(lane >> 1);
    const uint f4_idx = uint((base >> 2) + int(lane) * 4);

    const uint4 wu0 = w_ptr0[u4_idx];
    const float sc0 = float(s_row0[grp]);
    const uint4 wu1 = w_ptr1[u4_idx];
    const float sc1 = float(s_row1[grp]);

    if constexpr (SYM_F16) {
      const float4 q0_0 = float4(as_type<char4>(wu0.x ^ 0x80808080u));
      const float4 q0_1 = float4(as_type<char4>(wu0.y ^ 0x80808080u));
      const float4 q0_2 = float4(as_type<char4>(wu0.z ^ 0x80808080u));
      const float4 q0_3 = float4(as_type<char4>(wu0.w ^ 0x80808080u));
      const float4 q1_0 = float4(as_type<char4>(wu1.x ^ 0x80808080u));
      const float4 q1_1 = float4(as_type<char4>(wu1.y ^ 0x80808080u));
      const float4 q1_2 = float4(as_type<char4>(wu1.z ^ 0x80808080u));
      const float4 q1_3 = float4(as_type<char4>(wu1.w ^ 0x80808080u));
#pragma unroll
      for (int r = 0; r < M_TILE; ++r) {
        const device float4* xp = x_base + size_t(x_off[r] + f4_idx);
        const float4 x0 = xp[0];
        const float4 x1 = xp[1];
        const float4 x2 = xp[2];
        const float4 x3 = xp[3];
        const float d0 = metal::dot(x0, q0_0) + metal::dot(x1, q0_1) +
                         metal::dot(x2, q0_2) + metal::dot(x3, q0_3);
        const float d1 = metal::dot(x0, q1_0) + metal::dot(x1, q1_1) +
                         metal::dot(x2, q1_2) + metal::dot(x3, q1_3);
        acc[r][0] = metal::fma(d0, sc0, acc[r][0]);
        acc[r][1] = metal::fma(d1, sc1, acc[r][1]);
      }
    } else {
      const float bi0 = float(biases[wrow0 * size_t(n_groups) + size_t(grp)]);
      const float bi1 = float(biases[wrow1 * size_t(n_groups) + size_t(grp)]);
      const float4 q0_0 = float4(as_type<uchar4>(wu0.x));
      const float4 q0_1 = float4(as_type<uchar4>(wu0.y));
      const float4 q0_2 = float4(as_type<uchar4>(wu0.z));
      const float4 q0_3 = float4(as_type<uchar4>(wu0.w));
      const float4 q1_0 = float4(as_type<uchar4>(wu1.x));
      const float4 q1_1 = float4(as_type<uchar4>(wu1.y));
      const float4 q1_2 = float4(as_type<uchar4>(wu1.z));
      const float4 q1_3 = float4(as_type<uchar4>(wu1.w));
#pragma unroll
      for (int r = 0; r < M_TILE; ++r) {
        const device float4* xp = x_base + size_t(x_off[r] + f4_idx);
        const float4 x0 = xp[0];
        const float4 x1 = xp[1];
        const float4 x2 = xp[2];
        const float4 x3 = xp[3];
        const float4 xs = (x0 + x1) + (x2 + x3);
        const float xsum = (xs.x + xs.y) + (xs.z + xs.w);
        const float d0 = metal::dot(x0, q0_0) + metal::dot(x1, q0_1) +
                         metal::dot(x2, q0_2) + metal::dot(x3, q0_3);
        const float d1 = metal::dot(x0, q1_0) + metal::dot(x1, q1_1) +
                         metal::dot(x2, q1_2) + metal::dot(x3, q1_3);
        acc[r][0] = metal::fma(d0, sc0, metal::fma(xsum, bi0, acc[r][0]));
        acc[r][1] = metal::fma(d1, sc1, metal::fma(xsum, bi1, acc[r][1]));
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
    const uint4 wu0 = w_ptr0[u4_idx];
    const float sc0 = float(s_row0[grp]);
    const uint4 wu1 = w_ptr1[u4_idx];
    const float sc1 = float(s_row1[grp]);

    if constexpr (SYM_F16) {
      const float4 q0_0 = float4(as_type<char4>(wu0.x ^ 0x80808080u));
      const float4 q0_1 = float4(as_type<char4>(wu0.y ^ 0x80808080u));
      const float4 q0_2 = float4(as_type<char4>(wu0.z ^ 0x80808080u));
      const float4 q0_3 = float4(as_type<char4>(wu0.w ^ 0x80808080u));
      const float4 q1_0 = float4(as_type<char4>(wu1.x ^ 0x80808080u));
      const float4 q1_1 = float4(as_type<char4>(wu1.y ^ 0x80808080u));
      const float4 q1_2 = float4(as_type<char4>(wu1.z ^ 0x80808080u));
      const float4 q1_3 = float4(as_type<char4>(wu1.w ^ 0x80808080u));
#pragma unroll
      for (int r = 0; r < M_TILE; ++r) {
        const device float4* xp = x_base + size_t(x_off[r] + f4_idx);
        const float4 x0 = xp[0];
        const float4 x1 = xp[1];
        const float4 x2 = xp[2];
        const float4 x3 = xp[3];
        const float d0 = metal::dot(x0, q0_0) + metal::dot(x1, q0_1) +
                         metal::dot(x2, q0_2) + metal::dot(x3, q0_3);
        const float d1 = metal::dot(x0, q1_0) + metal::dot(x1, q1_1) +
                         metal::dot(x2, q1_2) + metal::dot(x3, q1_3);
        acc[r][0] = metal::fma(d0, sc0, acc[r][0]);
        acc[r][1] = metal::fma(d1, sc1, acc[r][1]);
      }
    } else {
      const float bi0 = float(biases[wrow0 * size_t(n_groups) + size_t(grp)]);
      const float bi1 = float(biases[wrow1 * size_t(n_groups) + size_t(grp)]);
      const float4 q0_0 = float4(as_type<uchar4>(wu0.x));
      const float4 q0_1 = float4(as_type<uchar4>(wu0.y));
      const float4 q0_2 = float4(as_type<uchar4>(wu0.z));
      const float4 q0_3 = float4(as_type<uchar4>(wu0.w));
      const float4 q1_0 = float4(as_type<uchar4>(wu1.x));
      const float4 q1_1 = float4(as_type<uchar4>(wu1.y));
      const float4 q1_2 = float4(as_type<uchar4>(wu1.z));
      const float4 q1_3 = float4(as_type<uchar4>(wu1.w));
#pragma unroll
      for (int r = 0; r < M_TILE; ++r) {
        const device float4* xp = x_base + size_t(x_off[r] + f4_idx);
        const float4 x0 = xp[0];
        const float4 x1 = xp[1];
        const float4 x2 = xp[2];
        const float4 x3 = xp[3];
        const float4 xs = (x0 + x1) + (x2 + x3);
        const float xsum = (xs.x + xs.y) + (xs.z + xs.w);
        const float d0 = metal::dot(x0, q0_0) + metal::dot(x1, q0_1) +
                         metal::dot(x2, q0_2) + metal::dot(x3, q0_3);
        const float d1 = metal::dot(x0, q1_0) + metal::dot(x1, q1_1) +
                         metal::dot(x2, q1_2) + metal::dot(x3, q1_3);
        acc[r][0] = metal::fma(d0, sc0, metal::fma(xsum, bi0, acc[r][0]));
        acc[r][1] = metal::fma(d1, sc1, metal::fma(xsum, bi1, acc[r][1]));
      }
    }
  }
}

if (out_row0 < N) {
#pragma unroll
  for (int r = 0; r < M_TILE; ++r) {
    const int mr = row0 + r;
    const size_t obase = (size_t(mr) * size_t(G) + size_t(g)) * size_t(N);
    const float v0 = simd_sum(acc[r][0]);
    const float v1 = simd_sum(acc[r][1]);
    if (lane == 0 && mr < M) {
      out[obase + size_t(out_row0)] = v0;
      if (out_row0 + 1 < N) {
        out[obase + size_t(out_row0 + 1)] = v1;
      }
    }
  }
}
)METAL"