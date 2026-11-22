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
  for (int base = 0; base < k_main; base += 512) {
    const int u4_idx = (base >> 4) + int(lane);
    const int grp = (base >> 5) + int(lane >> 1);
    const uint f4_idx = uint((base >> 2) + int(lane) * 4);

    if constexpr (M_TILE == 1) {
      const device float4* xp = rl_base + size_t(rl_off[0] + f4_idx);
      const float4 x0 = xp[0];
      const float4 x1 = xp[1];
      const float4 x2 = xp[2];
      const float4 x3 = xp[3];
      const device float4* gp = sgu_base + size_t(sgu_off[0] + f4_idx);
      const device float4* up = gp + size_t(n_f4);
      const float4 s0 = hy4_dual_swiglu4<CLAMP>(gp[0], up[0], lim);
      const float4 s1 = hy4_dual_swiglu4<CLAMP>(gp[1], up[1], lim);
      const float4 s2 = hy4_dual_swiglu4<CLAMP>(gp[2], up[2], lim);
      const float4 s3 = hy4_dual_swiglu4<CLAMP>(gp[3], up[3], lim);
#pragma unroll
      for (int o = 0; o < ROWS_PER_SG; o += 2) {
        const size_t r0 = size_t(metal::min(out_row0 + o, N - 1));
        const size_t r1 = size_t(metal::min(out_row0 + o + 1, N - 1));
        const uint4 u0 = wup_base[r0 * size_t(n_u4) + size_t(u4_idx)];
        const float su0 = float(s_up[r0 * size_t(n_groups) + size_t(grp)]);
        const uint4 u1 = wup_base[r1 * size_t(n_u4) + size_t(u4_idx)];
        const float su1 = float(s_up[r1 * size_t(n_groups) + size_t(grp)]);
        const uint4 d0_u = wsd_base[r0 * size_t(n_u4) + size_t(u4_idx)];
        const float sd0 = float(s_sd[r0 * size_t(n_groups) + size_t(grp)]);
        const uint4 d1_u = wsd_base[r1 * size_t(n_u4) + size_t(u4_idx)];
        const float sd1 = float(s_sd[r1 * size_t(n_groups) + size_t(grp)]);
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
        acc[0][o] = metal::fma(du0, su0, acc[0][o]);
        acc[0][o + 1] = metal::fma(du1, su1, acc[0][o + 1]);
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
        acc[0][o] = metal::fma(ds0, sd0, acc[0][o]);
        acc[0][o + 1] = metal::fma(ds1, sd1, acc[0][o + 1]);
      }
    } else {
#pragma unroll
      for (int o = 0; o < ROWS_PER_SG; o += 2) {
        const size_t r0 = size_t(metal::min(out_row0 + o, N - 1));
        const size_t r1 = size_t(metal::min(out_row0 + o + 1, N - 1));
        const uint4 u0 = wup_base[r0 * size_t(n_u4) + size_t(u4_idx)];
        const float su0 = float(s_up[r0 * size_t(n_groups) + size_t(grp)]);
        const uint4 u1 = wup_base[r1 * size_t(n_u4) + size_t(u4_idx)];
        const float su1 = float(s_up[r1 * size_t(n_groups) + size_t(grp)]);
        const uint4 d0_u = wsd_base[r0 * size_t(n_u4) + size_t(u4_idx)];
        const float sd0 = float(s_sd[r0 * size_t(n_groups) + size_t(grp)]);
        const uint4 d1_u = wsd_base[r1 * size_t(n_u4) + size_t(u4_idx)];
        const float sd1 = float(s_sd[r1 * size_t(n_groups) + size_t(grp)]);
#pragma unroll
        for (int r = 0; r < M_TILE; ++r) {
          const device float4* xp = rl_base + size_t(rl_off[r] + f4_idx);
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
          acc[r][o] = metal::fma(du0, su0, acc[r][o]);
          acc[r][o + 1] = metal::fma(du1, su1, acc[r][o + 1]);

          const device float4* gp = sgu_base + size_t(sgu_off[r] + f4_idx);
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
          acc[r][o] = metal::fma(ds0, sd0, acc[r][o]);
          acc[r][o + 1] = metal::fma(ds1, sd1, acc[r][o + 1]);
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
        const size_t r0 = size_t(metal::min(out_row0 + o, N - 1));
        const size_t r1 = size_t(metal::min(out_row0 + o + 1, N - 1));
        const uint4 u0 = wup_base[r0 * size_t(n_u4) + size_t(u4_idx)];
        const float su0 = float(s_up[r0 * size_t(n_groups) + size_t(grp)]);
        const uint4 u1 = wup_base[r1 * size_t(n_u4) + size_t(u4_idx)];
        const float su1 = float(s_up[r1 * size_t(n_groups) + size_t(grp)]);
        const uint4 d0_u = wsd_base[r0 * size_t(n_u4) + size_t(u4_idx)];
        const float sd0 = float(s_sd[r0 * size_t(n_groups) + size_t(grp)]);
        const uint4 d1_u = wsd_base[r1 * size_t(n_u4) + size_t(u4_idx)];
        const float sd1 = float(s_sd[r1 * size_t(n_groups) + size_t(grp)]);
#pragma unroll
        for (int r = 0; r < M_TILE; ++r) {
          const device float4* xp = rl_base + size_t(rl_off[r] + f4_idx);
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
          acc[r][o] = metal::fma(du0, su0, acc[r][o]);
          acc[r][o + 1] = metal::fma(du1, su1, acc[r][o + 1]);

          const device float4* gp = sgu_base + size_t(sgu_off[r] + f4_idx);
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
          acc[r][o] = metal::fma(ds0, sd0, acc[r][o]);
          acc[r][o + 1] = metal::fma(ds1, sd1, acc[r][o + 1]);
        }
      }
    }
  }
} else {
  // Cooperative float4 SwiGLU staging: 4 SGs compute 512 elements once per TG.
  // Int8 weights are exact in half4 (16 32-bit regs for 2 rows), so rl and s_act
  // load once per r and accumulate both rows o and o+1 without register spill.
  threadgroup float4 s_act[M_TILE][4][32];

  for (int base = 0; base < k_main; base += 512) {
    const int u4_idx = (base >> 4) + int(lane);
    const int grp = (base >> 5) + int(lane >> 1);
    const uint f4_idx = uint((base >> 2) + int(lane) * 4);

#pragma unroll
    for (int r = 0; r < M_TILE; ++r) {
      const uint elem_f4 = sgu_off[r] + f4_idx + sg;
      s_act[r][sg][lane] = hy4_dual_swiglu4<CLAMP>(
          sgu_base[elem_f4], sgu_base[elem_f4 + uint(n_f4)], lim);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

#pragma unroll
    for (int o = 0; o < ROWS_PER_SG; o += 2) {
      const size_t r0 = size_t(metal::min(out_row0 + o, N - 1));
      const size_t r1 = size_t(metal::min(out_row0 + o + 1, N - 1));
      {
        const uint4 u0 = wup_base[r0 * size_t(n_u4) + size_t(u4_idx)];
        const float su0 = float(s_up[r0 * size_t(n_groups) + size_t(grp)]);
        const uint4 u1 = wup_base[r1 * size_t(n_u4) + size_t(u4_idx)];
        const float su1 = float(s_up[r1 * size_t(n_groups) + size_t(grp)]);
        const half4 w00 = half4(as_type<char4>(u0.x ^ 0x80808080u));
        const half4 w01 = half4(as_type<char4>(u0.y ^ 0x80808080u));
        const half4 w02 = half4(as_type<char4>(u0.z ^ 0x80808080u));
        const half4 w03 = half4(as_type<char4>(u0.w ^ 0x80808080u));
        const half4 w10 = half4(as_type<char4>(u1.x ^ 0x80808080u));
        const half4 w11 = half4(as_type<char4>(u1.y ^ 0x80808080u));
        const half4 w12 = half4(as_type<char4>(u1.z ^ 0x80808080u));
        const half4 w13 = half4(as_type<char4>(u1.w ^ 0x80808080u));
#pragma unroll
        for (int r = 0; r < M_TILE; ++r) {
          const device float4* xp = rl_base + size_t(rl_off[r] + f4_idx);
          const float4 x0 = xp[0];
          const float4 x1 = xp[1];
          const float4 x2 = xp[2];
          const float4 x3 = xp[3];
          const float du0 =
              metal::dot(x0, float4(w00)) + metal::dot(x1, float4(w01)) +
              metal::dot(x2, float4(w02)) + metal::dot(x3, float4(w03));
          const float du1 =
              metal::dot(x0, float4(w10)) + metal::dot(x1, float4(w11)) +
              metal::dot(x2, float4(w12)) + metal::dot(x3, float4(w13));
          acc[r][o] = metal::fma(du0, su0, acc[r][o]);
          acc[r][o + 1] = metal::fma(du1, su1, acc[r][o + 1]);
        }
      }
      {
        const uint4 d0_u = wsd_base[r0 * size_t(n_u4) + size_t(u4_idx)];
        const float sd0 = float(s_sd[r0 * size_t(n_groups) + size_t(grp)]);
        const uint4 d1_u = wsd_base[r1 * size_t(n_u4) + size_t(u4_idx)];
        const float sd1 = float(s_sd[r1 * size_t(n_groups) + size_t(grp)]);
        const half4 w00 = half4(as_type<char4>(d0_u.x ^ 0x80808080u));
        const half4 w01 = half4(as_type<char4>(d0_u.y ^ 0x80808080u));
        const half4 w02 = half4(as_type<char4>(d0_u.z ^ 0x80808080u));
        const half4 w03 = half4(as_type<char4>(d0_u.w ^ 0x80808080u));
        const half4 w10 = half4(as_type<char4>(d1_u.x ^ 0x80808080u));
        const half4 w11 = half4(as_type<char4>(d1_u.y ^ 0x80808080u));
        const half4 w12 = half4(as_type<char4>(d1_u.z ^ 0x80808080u));
        const half4 w13 = half4(as_type<char4>(d1_u.w ^ 0x80808080u));
#pragma unroll
        for (int r = 0; r < M_TILE; ++r) {
          const float4 s0 = s_act[r][0][lane];
          const float4 s1 = s_act[r][1][lane];
          const float4 s2 = s_act[r][2][lane];
          const float4 s3 = s_act[r][3][lane];
          const float ds0 =
              metal::dot(s0, float4(w00)) + metal::dot(s1, float4(w01)) +
              metal::dot(s2, float4(w02)) + metal::dot(s3, float4(w03));
          const float ds1 =
              metal::dot(s0, float4(w10)) + metal::dot(s1, float4(w11)) +
              metal::dot(s2, float4(w12)) + metal::dot(s3, float4(w13));
          acc[r][o] = metal::fma(ds0, sd0, acc[r][o]);
          acc[r][o + 1] = metal::fma(ds1, sd1, acc[r][o + 1]);
        }
      }
    }
    if (base + 512 < k_main || !FAST_K) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
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
      s_act[r][sg][lane] = sv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (active) {
      const int u4_idx = col >> 4;
      const int grp = col >> 5;
#pragma unroll
      for (int o = 0; o < ROWS_PER_SG; o += 2) {
        const size_t r0 = size_t(metal::min(out_row0 + o, N - 1));
        const size_t r1 = size_t(metal::min(out_row0 + o + 1, N - 1));
        {
          const uint4 u0 = wup_base[r0 * size_t(n_u4) + size_t(u4_idx)];
          const float su0 = float(s_up[r0 * size_t(n_groups) + size_t(grp)]);
          const uint4 u1 = wup_base[r1 * size_t(n_u4) + size_t(u4_idx)];
          const float su1 = float(s_up[r1 * size_t(n_groups) + size_t(grp)]);
          const half4 w00 = half4(as_type<char4>(u0.x ^ 0x80808080u));
          const half4 w01 = half4(as_type<char4>(u0.y ^ 0x80808080u));
          const half4 w02 = half4(as_type<char4>(u0.z ^ 0x80808080u));
          const half4 w03 = half4(as_type<char4>(u0.w ^ 0x80808080u));
          const half4 w10 = half4(as_type<char4>(u1.x ^ 0x80808080u));
          const half4 w11 = half4(as_type<char4>(u1.y ^ 0x80808080u));
          const half4 w12 = half4(as_type<char4>(u1.z ^ 0x80808080u));
          const half4 w13 = half4(as_type<char4>(u1.w ^ 0x80808080u));
#pragma unroll
          for (int r = 0; r < M_TILE; ++r) {
            const device float4* xp = rl_base + size_t(rl_off[r] + f4_idx);
            const float4 x0 = xp[0];
            const float4 x1 = xp[1];
            const float4 x2 = xp[2];
            const float4 x3 = xp[3];
            const float du0 =
                metal::dot(x0, float4(w00)) + metal::dot(x1, float4(w01)) +
                metal::dot(x2, float4(w02)) + metal::dot(x3, float4(w03));
            const float du1 =
                metal::dot(x0, float4(w10)) + metal::dot(x1, float4(w11)) +
                metal::dot(x2, float4(w12)) + metal::dot(x3, float4(w13));
            acc[r][o] = metal::fma(du0, su0, acc[r][o]);
            acc[r][o + 1] = metal::fma(du1, su1, acc[r][o + 1]);
          }
        }
        {
          const uint4 d0_u = wsd_base[r0 * size_t(n_u4) + size_t(u4_idx)];
          const float sd0 = float(s_sd[r0 * size_t(n_groups) + size_t(grp)]);
          const uint4 d1_u = wsd_base[r1 * size_t(n_u4) + size_t(u4_idx)];
          const float sd1 = float(s_sd[r1 * size_t(n_groups) + size_t(grp)]);
          const half4 w00 = half4(as_type<char4>(d0_u.x ^ 0x80808080u));
          const half4 w01 = half4(as_type<char4>(d0_u.y ^ 0x80808080u));
          const half4 w02 = half4(as_type<char4>(d0_u.z ^ 0x80808080u));
          const half4 w03 = half4(as_type<char4>(d0_u.w ^ 0x80808080u));
          const half4 w10 = half4(as_type<char4>(d1_u.x ^ 0x80808080u));
          const half4 w11 = half4(as_type<char4>(d1_u.y ^ 0x80808080u));
          const half4 w12 = half4(as_type<char4>(d1_u.z ^ 0x80808080u));
          const half4 w13 = half4(as_type<char4>(d1_u.w ^ 0x80808080u));
#pragma unroll
          for (int r = 0; r < M_TILE; ++r) {
            const float4 s0 = s_act[r][0][lane];
            const float4 s1 = s_act[r][1][lane];
            const float4 s2 = s_act[r][2][lane];
            const float4 s3 = s_act[r][3][lane];
            const float ds0 =
                metal::dot(s0, float4(w00)) + metal::dot(s1, float4(w01)) +
                metal::dot(s2, float4(w02)) + metal::dot(s3, float4(w03));
            const float ds1 =
                metal::dot(s0, float4(w10)) + metal::dot(s1, float4(w11)) +
                metal::dot(s2, float4(w12)) + metal::dot(s3, float4(w13));
            acc[r][o] = metal::fma(ds0, sd0, acc[r][o]);
            acc[r][o + 1] = metal::fma(ds1, sd1, acc[r][o + 1]);
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