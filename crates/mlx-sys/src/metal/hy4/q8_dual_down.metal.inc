R"METAL(
constexpr int NUM_SG = 4;
constexpr int ROWS_PER_TG = NUM_SG * ROWS_PER_SG;

const int M = routed_latent_shape[0];
const int K = routed_latent_shape[1];
const int N = w_up_shape[0];

const uint lane = thread_index_in_simdgroup;
const uint sg = simdgroup_index_in_threadgroup;
const int tg_row0 = int(threadgroup_position_in_grid.y) * ROWS_PER_TG;
const int out_row0 = tg_row0 + int(sg) * ROWS_PER_SG;
const int row0 = int(threadgroup_position_in_grid.z) * M_TILE;
// Threadgroup-uniform early exit so all 4 simdgroups hit barriers together.
if (tg_row0 >= N || row0 >= M) return;

const int n_groups = K >> 5;
const int n_u4 = K >> 4;
const int n_f4 = K >> 2;
const float lim = limit[0];

const device uint4* wup_base = reinterpret_cast<const device uint4*>(w_up);
const device uint4* wsd_base = reinterpret_cast<const device uint4*>(w_sd);
const device float4* rl_base =
    reinterpret_cast<const device float4*>(routed_latent);
const device float4* sgu_base =
    reinterpret_cast<const device float4*>(shared_gu);

int r_clamped[ROWS_PER_SG];
#pragma unroll
for (int o = 0; o < ROWS_PER_SG; ++o) {
  r_clamped[o] = metal::min(out_row0 + o, N - 1);
}

const size_t rl_s0_f4 = size_t(routed_latent_strides[0]) >> 2;
const size_t sgu_s0_f4 = size_t(shared_gu_strides[0]) >> 2;
uint rl_off[M_TILE];
uint sgu_off[M_TILE];
#pragma unroll
for (int r = 0; r < M_TILE; ++r) {
  const int mr = metal::min(row0 + r, M - 1);
  rl_off[r] = uint(size_t(mr) * rl_s0_f4);
  sgu_off[r] = uint(size_t(mr) * sgu_s0_f4);
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

if constexpr (!USE_TG) {
  const int r0 = r_clamped[0];
  const int r1 = r_clamped[1];
  const device uint4* wup0 = wup_base + size_t(r0) * size_t(n_u4);
  const device uint4* wup1 = wup_base + size_t(r1) * size_t(n_u4);
  const auto sup0 = s_up + size_t(r0) * size_t(n_groups);
  const auto sup1 = s_up + size_t(r1) * size_t(n_groups);
  const device uint4* wsd0 = wsd_base + size_t(r0) * size_t(n_u4);
  const device uint4* wsd1 = wsd_base + size_t(r1) * size_t(n_u4);
  const auto ssd0 = s_sd + size_t(r0) * size_t(n_groups);
  const auto ssd1 = s_sd + size_t(r1) * size_t(n_groups);

  for (int base = 0; base < k_main; base += 512) {
    const int u4_idx = (base >> 4) + int(lane);
    const int grp = (base >> 5) + int(lane >> 1);
    const uint f4_idx = uint((base >> 2) + int(lane) * 4);

    const uint4 u0 = wup0[u4_idx];
    const float su0 = float(sup0[grp]);
    const uint4 u1 = wup1[u4_idx];
    const float su1 = float(sup1[grp]);
    const uint4 d0_u = wsd0[u4_idx];
    const float sd0 = float(ssd0[grp]);
    const uint4 d1_u = wsd1[u4_idx];
    const float sd1 = float(ssd1[grp]);

    if constexpr (M_TILE == 1) {
      const device float4* xp = rl_base + size_t(rl_off[0] + f4_idx);
      const float4 x0 = xp[0];
      const float4 x1 = xp[1];
      const float4 x2 = xp[2];
      const float4 x3 = xp[3];
      const float du0 =
          metal::dot(x0, float4(as_type<char4>(u0.x ^ 0x80808080u))) +
          metal::dot(x1, float4(as_type<char4>(u0.y ^ 0x80808080u))) +
          metal::dot(x2, float4(as_type<char4>(u0.z ^ 0x80808080u))) +
          metal::dot(x3, float4(as_type<char4>(u0.w ^ 0x80808080u)));
      const float du1 =
          metal::dot(x0, float4(as_type<char4>(u1.x ^ 0x80808080u))) +
          metal::dot(x1, float4(as_type<char4>(u1.y ^ 0x80808080u))) +
          metal::dot(x2, float4(as_type<char4>(u1.z ^ 0x80808080u))) +
          metal::dot(x3, float4(as_type<char4>(u1.w ^ 0x80808080u)));
      acc[0][0] = metal::fma(du0, su0, acc[0][0]);
      acc[0][1] = metal::fma(du1, su1, acc[0][1]);

      const device float4* gp = sgu_base + size_t(sgu_off[0] + f4_idx);
      const device float4* up = gp + size_t(n_f4);
      const float4 s0 = hy4_dual_swiglu4<CLAMP>(gp[0], up[0], lim);
      const float4 s1 = hy4_dual_swiglu4<CLAMP>(gp[1], up[1], lim);
      const float4 s2 = hy4_dual_swiglu4<CLAMP>(gp[2], up[2], lim);
      const float4 s3 = hy4_dual_swiglu4<CLAMP>(gp[3], up[3], lim);
      const float ds0 =
          metal::dot(s0, float4(as_type<char4>(d0_u.x ^ 0x80808080u))) +
          metal::dot(s1, float4(as_type<char4>(d0_u.y ^ 0x80808080u))) +
          metal::dot(s2, float4(as_type<char4>(d0_u.z ^ 0x80808080u))) +
          metal::dot(s3, float4(as_type<char4>(d0_u.w ^ 0x80808080u)));
      const float ds1 =
          metal::dot(s0, float4(as_type<char4>(d1_u.x ^ 0x80808080u))) +
          metal::dot(s1, float4(as_type<char4>(d1_u.y ^ 0x80808080u))) +
          metal::dot(s2, float4(as_type<char4>(d1_u.z ^ 0x80808080u))) +
          metal::dot(s3, float4(as_type<char4>(d1_u.w ^ 0x80808080u)));
      acc[0][0] = metal::fma(ds0, sd0, acc[0][0]);
      acc[0][1] = metal::fma(ds1, sd1, acc[0][1]);
    } else {
      {
        const float4 q0 = float4(as_type<char4>(u0.x ^ 0x80808080u));
        const float4 q1 = float4(as_type<char4>(u0.y ^ 0x80808080u));
        const float4 q2 = float4(as_type<char4>(u0.z ^ 0x80808080u));
        const float4 q3 = float4(as_type<char4>(u0.w ^ 0x80808080u));
#pragma unroll
        for (int r = 0; r < M_TILE; ++r) {
          const device float4* xp = rl_base + size_t(rl_off[r] + f4_idx);
          const float d = metal::dot(xp[0], q0) + metal::dot(xp[1], q1) +
                          metal::dot(xp[2], q2) + metal::dot(xp[3], q3);
          acc[r][0] = metal::fma(d, su0, acc[r][0]);
        }
      }
      {
        const float4 q0 = float4(as_type<char4>(u1.x ^ 0x80808080u));
        const float4 q1 = float4(as_type<char4>(u1.y ^ 0x80808080u));
        const float4 q2 = float4(as_type<char4>(u1.z ^ 0x80808080u));
        const float4 q3 = float4(as_type<char4>(u1.w ^ 0x80808080u));
#pragma unroll
        for (int r = 0; r < M_TILE; ++r) {
          const device float4* xp = rl_base + size_t(rl_off[r] + f4_idx);
          const float d = metal::dot(xp[0], q0) + metal::dot(xp[1], q1) +
                          metal::dot(xp[2], q2) + metal::dot(xp[3], q3);
          acc[r][1] = metal::fma(d, su1, acc[r][1]);
        }
      }
      {
        const float4 q0_0 = float4(as_type<char4>(d0_u.x ^ 0x80808080u));
        const float4 q0_1 = float4(as_type<char4>(d0_u.y ^ 0x80808080u));
        const float4 q0_2 = float4(as_type<char4>(d0_u.z ^ 0x80808080u));
        const float4 q0_3 = float4(as_type<char4>(d0_u.w ^ 0x80808080u));
        const float4 q1_0 = float4(as_type<char4>(d1_u.x ^ 0x80808080u));
        const float4 q1_1 = float4(as_type<char4>(d1_u.y ^ 0x80808080u));
        const float4 q1_2 = float4(as_type<char4>(d1_u.z ^ 0x80808080u));
        const float4 q1_3 = float4(as_type<char4>(d1_u.w ^ 0x80808080u));
#pragma unroll
        for (int r = 0; r < M_TILE; ++r) {
          const device float4* gp = sgu_base + size_t(sgu_off[r] + f4_idx);
          const device float4* up = gp + size_t(n_f4);
          const float4 s0 = hy4_dual_swiglu4<CLAMP>(gp[0], up[0], lim);
          const float4 s1 = hy4_dual_swiglu4<CLAMP>(gp[1], up[1], lim);
          const float4 s2 = hy4_dual_swiglu4<CLAMP>(gp[2], up[2], lim);
          const float4 s3 = hy4_dual_swiglu4<CLAMP>(gp[3], up[3], lim);
          const float d0 = metal::dot(s0, q0_0) + metal::dot(s1, q0_1) +
                           metal::dot(s2, q0_2) + metal::dot(s3, q0_3);
          const float d1 = metal::dot(s0, q1_0) + metal::dot(s1, q1_1) +
                           metal::dot(s2, q1_2) + metal::dot(s3, q1_3);
          acc[r][0] = metal::fma(d0, sd0, acc[r][0]);
          acc[r][1] = metal::fma(d1, sd1, acc[r][1]);
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
      const uint4 u0 = wup0[u4_idx];
      const float su0 = float(sup0[grp]);
      const uint4 u1 = wup1[u4_idx];
      const float su1 = float(sup1[grp]);
      const uint4 d0_u = wsd0[u4_idx];
      const float sd0 = float(ssd0[grp]);
      const uint4 d1_u = wsd1[u4_idx];
      const float sd1 = float(ssd1[grp]);
      {
        const float4 q0 = float4(as_type<char4>(u0.x ^ 0x80808080u));
        const float4 q1 = float4(as_type<char4>(u0.y ^ 0x80808080u));
        const float4 q2 = float4(as_type<char4>(u0.z ^ 0x80808080u));
        const float4 q3 = float4(as_type<char4>(u0.w ^ 0x80808080u));
#pragma unroll
        for (int r = 0; r < M_TILE; ++r) {
          const device float4* xp = rl_base + size_t(rl_off[r] + f4_idx);
          const float d = metal::dot(xp[0], q0) + metal::dot(xp[1], q1) +
                          metal::dot(xp[2], q2) + metal::dot(xp[3], q3);
          acc[r][0] = metal::fma(d, su0, acc[r][0]);
        }
      }
      {
        const float4 q0 = float4(as_type<char4>(u1.x ^ 0x80808080u));
        const float4 q1 = float4(as_type<char4>(u1.y ^ 0x80808080u));
        const float4 q2 = float4(as_type<char4>(u1.z ^ 0x80808080u));
        const float4 q3 = float4(as_type<char4>(u1.w ^ 0x80808080u));
#pragma unroll
        for (int r = 0; r < M_TILE; ++r) {
          const device float4* xp = rl_base + size_t(rl_off[r] + f4_idx);
          const float d = metal::dot(xp[0], q0) + metal::dot(xp[1], q1) +
                          metal::dot(xp[2], q2) + metal::dot(xp[3], q3);
          acc[r][1] = metal::fma(d, su1, acc[r][1]);
        }
      }
      {
        const float4 q0_0 = float4(as_type<char4>(d0_u.x ^ 0x80808080u));
        const float4 q0_1 = float4(as_type<char4>(d0_u.y ^ 0x80808080u));
        const float4 q0_2 = float4(as_type<char4>(d0_u.z ^ 0x80808080u));
        const float4 q0_3 = float4(as_type<char4>(d0_u.w ^ 0x80808080u));
        const float4 q1_0 = float4(as_type<char4>(d1_u.x ^ 0x80808080u));
        const float4 q1_1 = float4(as_type<char4>(d1_u.y ^ 0x80808080u));
        const float4 q1_2 = float4(as_type<char4>(d1_u.z ^ 0x80808080u));
        const float4 q1_3 = float4(as_type<char4>(d1_u.w ^ 0x80808080u));
#pragma unroll
        for (int r = 0; r < M_TILE; ++r) {
          const device float4* gp = sgu_base + size_t(sgu_off[r] + f4_idx);
          const device float4* up = gp + size_t(n_f4);
          const float4 s0 = hy4_dual_swiglu4<CLAMP>(gp[0], up[0], lim);
          const float4 s1 = hy4_dual_swiglu4<CLAMP>(gp[1], up[1], lim);
          const float4 s2 = hy4_dual_swiglu4<CLAMP>(gp[2], up[2], lim);
          const float4 s3 = hy4_dual_swiglu4<CLAMP>(gp[3], up[3], lim);
          const float d0 = metal::dot(s0, q0_0) + metal::dot(s1, q0_1) +
                           metal::dot(s2, q0_2) + metal::dot(s3, q0_3);
          const float d1 = metal::dot(s0, q1_0) + metal::dot(s1, q1_1) +
                           metal::dot(s2, q1_2) + metal::dot(s3, q1_3);
          acc[r][0] = metal::fma(d0, sd0, acc[r][0]);
          acc[r][1] = metal::fma(d1, sd1, acc[r][1]);
        }
      }
    }
  }
} else {
  // Cooperative SwiGLU staging for M_TILE >= 4: 4 SGs compute 512 elements once per TG.
  threadgroup float s_act[M_TILE][16][32];

  for (int base = 0; base < k_main; base += 512) {
    const int u4_idx = (base >> 4) + int(lane);
    const int grp = (base >> 5) + int(lane >> 1);
    const uint f4_idx = uint((base >> 2) + int(lane) * 4);

#pragma unroll
    for (int r = 0; r < M_TILE; ++r) {
      const uint elem_f4 = sgu_off[r] + f4_idx + sg;
      const float4 sv = hy4_dual_swiglu4<CLAMP>(
          sgu_base[elem_f4], sgu_base[elem_f4 + uint(n_f4)], lim);
      const int p = int(sg) * 4;
      s_act[r][p + 0][lane] = sv.x;
      s_act[r][p + 1][lane] = sv.y;
      s_act[r][p + 2][lane] = sv.z;
      s_act[r][p + 3][lane] = sv.w;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

#pragma unroll
    for (int o = 0; o < ROWS_PER_SG; ++o) {
      const size_t ro = size_t(r_clamped[o]);
      const uint4 u = wup_base[ro * size_t(n_u4) + size_t(u4_idx)];
      const float su = float(s_up[ro * size_t(n_groups) + size_t(grp)]);
      const uint4 d = wsd_base[ro * size_t(n_u4) + size_t(u4_idx)];
      const float sd = float(s_sd[ro * size_t(n_groups) + size_t(grp)]);
      {
        const float4 q0 = float4(as_type<char4>(u.x ^ 0x80808080u));
        const float4 q1 = float4(as_type<char4>(u.y ^ 0x80808080u));
        const float4 q2 = float4(as_type<char4>(u.z ^ 0x80808080u));
        const float4 q3 = float4(as_type<char4>(u.w ^ 0x80808080u));
#pragma unroll
        for (int r = 0; r < M_TILE; ++r) {
          const device float4* xp = rl_base + size_t(rl_off[r] + f4_idx);
          const float du = metal::dot(xp[0], q0) + metal::dot(xp[1], q1) +
                           metal::dot(xp[2], q2) + metal::dot(xp[3], q3);
          acc[r][o] = metal::fma(du, su, acc[r][o]);
        }
      }
      {
        const float4 q0 = float4(as_type<char4>(d.x ^ 0x80808080u));
        const float4 q1 = float4(as_type<char4>(d.y ^ 0x80808080u));
        const float4 q2 = float4(as_type<char4>(d.z ^ 0x80808080u));
        const float4 q3 = float4(as_type<char4>(d.w ^ 0x80808080u));
#pragma unroll
        for (int r = 0; r < M_TILE; ++r) {
          const float4 s0 = float4(s_act[r][0][lane], s_act[r][1][lane],
                                   s_act[r][2][lane], s_act[r][3][lane]);
          const float4 s1 = float4(s_act[r][4][lane], s_act[r][5][lane],
                                   s_act[r][6][lane], s_act[r][7][lane]);
          const float4 s2 = float4(s_act[r][8][lane], s_act[r][9][lane],
                                   s_act[r][10][lane], s_act[r][11][lane]);
          const float4 s3 = float4(s_act[r][12][lane], s_act[r][13][lane],
                                   s_act[r][14][lane], s_act[r][15][lane]);
          const float ds = metal::dot(s0, q0) + metal::dot(s1, q1) +
                           metal::dot(s2, q2) + metal::dot(s3, q3);
          acc[r][o] = metal::fma(ds, sd, acc[r][o]);
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if constexpr (!FAST_K) {
    const int col = k_main + int(lane) * 16;
    const bool active = (col < K);
    const uint f4_idx = uint(col >> 2);
#pragma unroll
    for (int r = 0; r < M_TILE; ++r) {
      float4 sv = float4(0.0f);
      if (active) {
        const uint elem_f4 = sgu_off[r] + f4_idx + sg;
        sv = hy4_dual_swiglu4<CLAMP>(
            sgu_base[elem_f4], sgu_base[elem_f4 + uint(n_f4)], lim);
      }
      const int p = int(sg) * 4;
      s_act[r][p + 0][lane] = sv.x;
      s_act[r][p + 1][lane] = sv.y;
      s_act[r][p + 2][lane] = sv.z;
      s_act[r][p + 3][lane] = sv.w;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (active) {
      const int u4_idx = col >> 4;
      const int grp = col >> 5;
#pragma unroll
      for (int o = 0; o < ROWS_PER_SG; ++o) {
        const size_t ro = size_t(r_clamped[o]);
        const uint4 u = wup_base[ro * size_t(n_u4) + size_t(u4_idx)];
        const float su = float(s_up[ro * size_t(n_groups) + size_t(grp)]);
        const uint4 d = wsd_base[ro * size_t(n_u4) + size_t(u4_idx)];
        const float sd = float(s_sd[ro * size_t(n_groups) + size_t(grp)]);
        {
          const float4 q0 = float4(as_type<char4>(u.x ^ 0x80808080u));
          const float4 q1 = float4(as_type<char4>(u.y ^ 0x80808080u));
          const float4 q2 = float4(as_type<char4>(u.z ^ 0x80808080u));
          const float4 q3 = float4(as_type<char4>(u.w ^ 0x80808080u));
#pragma unroll
          for (int r = 0; r < M_TILE; ++r) {
            const device float4* xp = rl_base + size_t(rl_off[r] + f4_idx);
            const float du = metal::dot(xp[0], q0) + metal::dot(xp[1], q1) +
                             metal::dot(xp[2], q2) + metal::dot(xp[3], q3);
            acc[r][o] = metal::fma(du, su, acc[r][o]);
          }
        }
        {
          const float4 q0 = float4(as_type<char4>(d.x ^ 0x80808080u));
          const float4 q1 = float4(as_type<char4>(d.y ^ 0x80808080u));
          const float4 q2 = float4(as_type<char4>(d.z ^ 0x80808080u));
          const float4 q3 = float4(as_type<char4>(d.w ^ 0x80808080u));
#pragma unroll
          for (int r = 0; r < M_TILE; ++r) {
            const float4 s0 = float4(s_act[r][0][lane], s_act[r][1][lane],
                                     s_act[r][2][lane], s_act[r][3][lane]);
            const float4 s1 = float4(s_act[r][4][lane], s_act[r][5][lane],
                                     s_act[r][6][lane], s_act[r][7][lane]);
            const float4 s2 = float4(s_act[r][8][lane], s_act[r][9][lane],
                                     s_act[r][10][lane], s_act[r][11][lane]);
            const float4 s3 = float4(s_act[r][12][lane], s_act[r][13][lane],
                                     s_act[r][14][lane], s_act[r][15][lane]);
            const float ds = metal::dot(s0, q0) + metal::dot(s1, q1) +
                             metal::dot(s2, q2) + metal::dot(s3, q3);
            acc[r][o] = metal::fma(ds, sd, acc[r][o]);
          }
        }
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
    if (lane == 0 && mr < M && out_row0 + o < N) {
      out[obase + size_t(out_row0 + o)] = v;
    }
  }
}
)METAL"