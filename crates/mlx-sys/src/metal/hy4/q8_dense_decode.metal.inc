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
const int out_row0 = int(threadgroup_position_in_grid.y) * ROWS_PER_TG +
    int(sg) * ROWS_PER_SG;

const int n_mtiles = (M + M_TILE - 1) / M_TILE;
const int z = int(threadgroup_position_in_grid.z);
const int g = GROUPED ? (z / n_mtiles) : 0;
const int row0 = (GROUPED ? (z % n_mtiles) : z) * M_TILE;
if (out_row0 >= N || row0 >= M || g >= G) return;

const int r0 = out_row0;
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
size_t x_off[M_TILE];
#pragma unroll
for (int r = 0; r < M_TILE; ++r) {
  const int mr = metal::min(row0 + r, M - 1);
  x_off[r] = (size_t(mr) * size_t(G) + size_t(g)) * size_t(n_f4);
}

float acc[M_TILE][ROWS_PER_SG];
#pragma unroll
for (int r = 0; r < M_TILE; ++r) {
  acc[r][0] = 0.0f;
  acc[r][1] = 0.0f;
}

const int k_main = FAST_K ? K : (K & ~511);
for (int base = 0; base < k_main; base += 512) {
  const int u4_idx = (base >> 4) + int(lane);
  const int grp = (base >> 5) + int(lane >> 1);
  const int f4_idx = (base >> 2) + int(lane) * 4;

  const uint4 w0_u = w_ptr0[u4_idx];
  const float sc0 = float(s_row0[grp]);
  const uint4 w1_u = w_ptr1[u4_idx];
  const float sc1 = float(s_row1[grp]);

  if constexpr (SYM_F16) {
    const float4 q0_0 = float4(as_type<char4>(w0_u.x ^ 0x80808080u));
    const float4 q0_1 = float4(as_type<char4>(w0_u.y ^ 0x80808080u));
    const float4 q0_2 = float4(as_type<char4>(w0_u.z ^ 0x80808080u));
    const float4 q0_3 = float4(as_type<char4>(w0_u.w ^ 0x80808080u));
    const float4 q1_0 = float4(as_type<char4>(w1_u.x ^ 0x80808080u));
    const float4 q1_1 = float4(as_type<char4>(w1_u.y ^ 0x80808080u));
    const float4 q1_2 = float4(as_type<char4>(w1_u.z ^ 0x80808080u));
    const float4 q1_3 = float4(as_type<char4>(w1_u.w ^ 0x80808080u));
#pragma unroll
    for (int r = 0; r < M_TILE; ++r) {
      const device float4* xp = x_base + x_off[r] + size_t(f4_idx);
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
    const float4 q0_0 = float4(as_type<uchar4>(w0_u.x));
    const float4 q0_1 = float4(as_type<uchar4>(w0_u.y));
    const float4 q0_2 = float4(as_type<uchar4>(w0_u.z));
    const float4 q0_3 = float4(as_type<uchar4>(w0_u.w));
    const float4 q1_0 = float4(as_type<uchar4>(w1_u.x));
    const float4 q1_1 = float4(as_type<uchar4>(w1_u.y));
    const float4 q1_2 = float4(as_type<uchar4>(w1_u.z));
    const float4 q1_3 = float4(as_type<uchar4>(w1_u.w));
#pragma unroll
    for (int r = 0; r < M_TILE; ++r) {
      const device float4* xp = x_base + x_off[r] + size_t(f4_idx);
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

if constexpr (!FAST_K) {
  const int col = k_main + int(lane) * 16;
  if (col < K) {
    const int u4_idx = col >> 4;
    const int grp = col >> 5;
    const int f4_idx = col >> 2;

    const uint4 w0_u = w_ptr0[u4_idx];
    const float sc0 = float(s_row0[grp]);
    const uint4 w1_u = w_ptr1[u4_idx];
    const float sc1 = float(s_row1[grp]);

    if constexpr (SYM_F16) {
      const float4 q0_0 = float4(as_type<char4>(w0_u.x ^ 0x80808080u));
      const float4 q0_1 = float4(as_type<char4>(w0_u.y ^ 0x80808080u));
      const float4 q0_2 = float4(as_type<char4>(w0_u.z ^ 0x80808080u));
      const float4 q0_3 = float4(as_type<char4>(w0_u.w ^ 0x80808080u));
      const float4 q1_0 = float4(as_type<char4>(w1_u.x ^ 0x80808080u));
      const float4 q1_1 = float4(as_type<char4>(w1_u.y ^ 0x80808080u));
      const float4 q1_2 = float4(as_type<char4>(w1_u.z ^ 0x80808080u));
      const float4 q1_3 = float4(as_type<char4>(w1_u.w ^ 0x80808080u));
#pragma unroll
      for (int r = 0; r < M_TILE; ++r) {
        const device float4* xp = x_base + x_off[r] + size_t(f4_idx);
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
      const float4 q0_0 = float4(as_type<uchar4>(w0_u.x));
      const float4 q0_1 = float4(as_type<uchar4>(w0_u.y));
      const float4 q0_2 = float4(as_type<uchar4>(w0_u.z));
      const float4 q0_3 = float4(as_type<uchar4>(w0_u.w));
      const float4 q1_0 = float4(as_type<uchar4>(w1_u.x));
      const float4 q1_1 = float4(as_type<uchar4>(w1_u.y));
      const float4 q1_2 = float4(as_type<uchar4>(w1_u.z));
      const float4 q1_3 = float4(as_type<uchar4>(w1_u.w));
#pragma unroll
      for (int r = 0; r < M_TILE; ++r) {
        const device float4* xp = x_base + x_off[r] + size_t(f4_idx);
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
)METAL"
