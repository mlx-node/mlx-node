R"METAL(
constexpr int NUM_SG = 4;
constexpr int ROWS_PER_SG = 2;
constexpr int ROWS_PER_TG = NUM_SG * ROWS_PER_SG;

const int M = routed_latent_shape[0];
const int K = routed_latent_shape[1];
const int N = w_up_shape[0];

const uint lane = thread_index_in_simdgroup;
const uint sg = simdgroup_index_in_threadgroup;
const int out_row0 = int(threadgroup_position_in_grid.y) * ROWS_PER_TG +
    int(sg) * ROWS_PER_SG;
const int row0 = int(threadgroup_position_in_grid.z) * M_TILE;
if (out_row0 >= N || row0 >= M) return;

const int r0 = out_row0;
const int r1 = metal::min(out_row0 + 1, N - 1);
const int n_groups = K >> 5;
const int n_u4 = K >> 4;
const int n_f4 = K >> 2;
const float lim = limit[0];

const device uint4* wup0 =
    reinterpret_cast<const device uint4*>(w_up) + size_t(r0) * size_t(n_u4);
const device uint4* wup1 =
    reinterpret_cast<const device uint4*>(w_up) + size_t(r1) * size_t(n_u4);
const auto sup0 = s_up + size_t(r0) * size_t(n_groups);
const auto sup1 = s_up + size_t(r1) * size_t(n_groups);

const device uint4* wsd0 =
    reinterpret_cast<const device uint4*>(w_sd) + size_t(r0) * size_t(n_u4);
const device uint4* wsd1 =
    reinterpret_cast<const device uint4*>(w_sd) + size_t(r1) * size_t(n_u4);
const auto ssd0 = s_sd + size_t(r0) * size_t(n_groups);
const auto ssd1 = s_sd + size_t(r1) * size_t(n_groups);

const device float4* rl_base =
    reinterpret_cast<const device float4*>(routed_latent);
const device float4* sgu_base =
    reinterpret_cast<const device float4*>(shared_gu);

size_t rl_off[M_TILE];
size_t sgu_off[M_TILE];
#pragma unroll
for (int r = 0; r < M_TILE; ++r) {
  const int mr = metal::min(row0 + r, M - 1);
  rl_off[r] = size_t(mr) * size_t(n_f4);
  sgu_off[r] = size_t(mr) * size_t(2 * n_f4);
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

  {
    const uint4 w0_u = wup0[u4_idx];
    const float sc0 = float(sup0[grp]);
    const uint4 w1_u = wup1[u4_idx];
    const float sc1 = float(sup1[grp]);
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
      const device float4* xp = rl_base + rl_off[r] + size_t(f4_idx);
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
  }

  {
    const uint4 w0_u = wsd0[u4_idx];
    const float sc0 = float(ssd0[grp]);
    const uint4 w1_u = wsd1[u4_idx];
    const float sc1 = float(ssd1[grp]);
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
      const device float4* gp = sgu_base + sgu_off[r] + size_t(f4_idx);
      const device float4* up = gp + size_t(n_f4);
      const float4 s0 = hy4_dual_swiglu4<CLAMP>(gp[0], up[0], lim);
      const float4 s1 = hy4_dual_swiglu4<CLAMP>(gp[1], up[1], lim);
      const float4 s2 = hy4_dual_swiglu4<CLAMP>(gp[2], up[2], lim);
      const float4 s3 = hy4_dual_swiglu4<CLAMP>(gp[3], up[3], lim);
      const float d0 = metal::dot(s0, q0_0) + metal::dot(s1, q0_1) +
                       metal::dot(s2, q0_2) + metal::dot(s3, q0_3);
      const float d1 = metal::dot(s0, q1_0) + metal::dot(s1, q1_1) +
                       metal::dot(s2, q1_2) + metal::dot(s3, q1_3);
      acc[r][0] = metal::fma(d0, sc0, acc[r][0]);
      acc[r][1] = metal::fma(d1, sc1, acc[r][1]);
    }
  }
}

if constexpr (!FAST_K) {
  const int col = k_main + int(lane) * 16;
  if (col < K) {
    const int u4_idx = col >> 4;
    const int grp = col >> 5;
    const int f4_idx = col >> 2;

    {
      const uint4 w0_u = wup0[u4_idx];
      const float sc0 = float(sup0[grp]);
      const uint4 w1_u = wup1[u4_idx];
      const float sc1 = float(sup1[grp]);
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
        const device float4* xp = rl_base + rl_off[r] + size_t(f4_idx);
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
    }

    {
      const uint4 w0_u = wsd0[u4_idx];
      const float sc0 = float(ssd0[grp]);
      const uint4 w1_u = wsd1[u4_idx];
      const float sc1 = float(ssd1[grp]);
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
        const device float4* gp = sgu_base + sgu_off[r] + size_t(f4_idx);
        const device float4* up = gp + size_t(n_f4);
        const float4 s0 = hy4_dual_swiglu4<CLAMP>(gp[0], up[0], lim);
        const float4 s1 = hy4_dual_swiglu4<CLAMP>(gp[1], up[1], lim);
        const float4 s2 = hy4_dual_swiglu4<CLAMP>(gp[2], up[2], lim);
        const float4 s3 = hy4_dual_swiglu4<CLAMP>(gp[3], up[3], lim);
        const float d0 = metal::dot(s0, q0_0) + metal::dot(s1, q0_1) +
                         metal::dot(s2, q0_2) + metal::dot(s3, q0_3);
        const float d1 = metal::dot(s0, q1_0) + metal::dot(s1, q1_1) +
                         metal::dot(s2, q1_2) + metal::dot(s3, q1_3);
        acc[r][0] = metal::fma(d0, sc0, acc[r][0]);
        acc[r][1] = metal::fma(d1, sc1, acc[r][1]);
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
    const size_t obase = size_t(mr) * size_t(N);
    out[obase + size_t(out_row0)] = v0;
    if (out_row0 + 1 < N) {
      out[obase + size_t(out_row0 + 1)] = v1;
    }
  }
}
)METAL"
