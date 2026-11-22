R"cuda(
#include <cuda_bf16.h>
#include <cuda_fp16.h>

// Fused CUDA MoE Decode/Verify kernels (M in 1..=16) for Qwen3.8-Flash-Next:
//   E = 512 routed experts, TOP_K = 10, H = 2560, D_FF = 640
//   Routed gate/up: RowMajor Q4_K (mode=0) or Q5_K (mode=1), [E*640, 2560*b/32]
//   Routed down:    RowMajor affine 5-bit (Q5_1) or 8-bit (Q8_0) g32 f16 s+b, [E*2560, 640*b/32]
//   Shared expert:  RowMajor affine 8-bit (Q8_0) g32 f16 s+b

__device__ __forceinline__ uint4 q4m_ld_cs_u4(const uint4* ptr) {
  uint4 v;
  asm("ld.global.cs.v4.u32 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
      : "l"(ptr));
  return v;
}

__device__ __forceinline__ int q4m_dp4a(unsigned int a, unsigned int b, int c) {
  int d;
  asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
}

__device__ __forceinline__ void q4m_cp_async16(void* smem_ptr, const void* glob_ptr) {
  unsigned int smem_addr = static_cast<unsigned int>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "cp.async.cg.shared.global [%0], [%1], 16;\n"
      :: "r"(smem_addr), "l"(glob_ptr) : "memory");
}

__device__ __forceinline__ void q4m_cp_commit() {
  asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template <int N_WAIT>
__device__ __forceinline__ void q4m_cp_wait() {
  asm volatile("cp.async.wait_group %0;\n" :: "n"(N_WAIT) : "memory");
}

__device__ __forceinline__ void q4m_mma_s8(
    unsigned int a0, unsigned int a1, unsigned int a2, unsigned int a3,
    unsigned int b0, unsigned int b1,
    int& c0, int& c1, int& c2, int& c3) {
  asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
      : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void q4m_mma_bf16(
    float& d0, float& d1, float& d2, float& d3,
    unsigned int a0, unsigned int a1, unsigned int a2, unsigned int a3,
    unsigned int b0, unsigned int b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

template <int mode>
__device__ __forceinline__ unsigned int q4m_unpack4_kq(unsigned int u) {
  if constexpr (mode == 0) {
    return (u & 0x0000000fu) | ((u & 0x000000f0u) << 4) |
        ((u & 0x00000f00u) << 8) | ((u & 0x0000f000u) << 12);
  } else {
    return (u & 0x0000001fu) | ((u & 0x000003e0u) << 3) |
        ((u & 0x00007c00u) << 6) | ((u & 0x000f8000u) << 9);
  }
}

template <int mode>
__device__ __forceinline__ void q4m_load_pair_kq(
    const unsigned int* seg, int t, unsigned int& out0, unsigned int& out1) {
  if constexpr (mode == 0) {
    const unsigned short* seg16 = reinterpret_cast<const unsigned short*>(seg);
    out0 = q4m_unpack4_kq<0>(seg16[t]);
    out1 = q4m_unpack4_kq<0>(seg16[4 + t]);
  } else {
    const int bit0 = t * 20;
    const int w0 = bit0 >> 5;
    const int sh0 = bit0 & 31;
    unsigned int u0 = seg[w0];
    if (sh0 > 12) u0 = __funnelshift_r(u0, seg[w0 + 1], sh0);
    else u0 >>= sh0;

    const int bit1 = 80 + t * 20;
    const int w1 = bit1 >> 5;
    const int sh1 = bit1 & 31;
    unsigned int u1 = seg[w1];
    if (sh1 > 12) u1 = __funnelshift_r(u1, seg[w1 + 1], sh1);
    else u1 >>= sh1;

    out0 = q4m_unpack4_kq<1>(u0);
    out1 = q4m_unpack4_kq<1>(u1);
  }
}

__device__ __forceinline__ unsigned int q4m_pack_bf16x2(float f0, float f1) {
  __nv_bfloat162 v = __floats2bfloat162_rn(f0, f1);
  return *reinterpret_cast<const unsigned int*>(&v);
}

__device__ __forceinline__ float q4m_sum_bf16x8(const uint4& v) {
  const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&v);
  const float2 f0 = __bfloat1622float2(p[0]);
  const float2 f1 = __bfloat1622float2(p[1]);
  const float2 f2 = __bfloat1622float2(p[2]);
  const float2 f3 = __bfloat1622float2(p[3]);
  return ((f0.x + f0.y) + (f1.x + f1.y)) + ((f2.x + f2.y) + (f3.x + f3.y));
}

__device__ __forceinline__ float q4m_silu(float x) {
  return x / (1.0f + expf(-x));
}

__device__ __forceinline__ float q4m_sigmoid(float x) {
  return 1.0f / (1.0f + expf(-x));
}

// ---------------------------------------------------------------------------
// 1. Fused Single-Launch GPU Expert Grouping + Q8_1 Activation Quantization
//    blockIdx.x == 0: deterministic sorted expert grouping (M<=16, P<=160, E<=512)
//    blockIdx.x >= 1: per-token Q8_1 quantization (8 threads per 32-elem group)
//    Grid: 1 + ceil(M * (K/32) * 8 / 256), Block: 256. Zero host sync!
// ---------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(256) void qwen4_moe_plan_and_quantize_few(
    const int* __restrict__ expert_ids,
    const nv_bfloat16* __restrict__ x,
    int* __restrict__ unique_experts,
    int* __restrict__ bucket_offsets,
    int* __restrict__ pair_indices,
    int* __restrict__ num_unique,
    signed char* __restrict__ xq,
    float* __restrict__ x_scales,
    float* __restrict__ x_sums,
    int M,
    int P,
    int E,
    int K) {
  if (blockIdx.x == 0) {
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    __shared__ int s_cnt[512];
    __shared__ int s_start[512];
    __shared__ int s_ids[256];
    __shared__ int s_warp_c[8];
    __shared__ int s_warp_u[8];

    s_cnt[tid] = 0;
    s_cnt[tid + 256] = 0;
    if (tid < P) {
      s_ids[tid] = expert_ids[tid];
    }
    __syncthreads();

    if (tid < P) {
      const int e = s_ids[tid];
      if (static_cast<unsigned int>(e) < static_cast<unsigned int>(E)) {
        atomicAdd(&s_cnt[e], 1);
      }
    }
    __syncthreads();

    const int e0 = tid * 2;
    const int e1 = e0 + 1;
    const int c0 = s_cnt[e0];
    const int c1 = s_cnt[e1];
    const int u0 = c0 > 0 ? 1 : 0;
    const int u1 = c1 > 0 ? 1 : 0;

    int cp = c0 + c1;
    int up = u0 + u1;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
      int nc = __shfl_up_sync(0xffffffffu, cp, off);
      int nu = __shfl_up_sync(0xffffffffu, up, off);
      if (lane >= off) {
        cp += nc;
        up += nu;
      }
    }
    if (lane == 31) {
      s_warp_c[warp] = cp;
      s_warp_u[warp] = up;
    }
    __syncthreads();

    if (warp == 0) {
      int wc = lane < 8 ? s_warp_c[lane] : 0;
      int wu = lane < 8 ? s_warp_u[lane] : 0;
#pragma unroll
      for (int off = 1; off < 8; off <<= 1) {
        int nc = __shfl_up_sync(0xffffffffu, wc, off);
        int nu = __shfl_up_sync(0xffffffffu, wu, off);
        if (lane >= off) {
          wc += nc;
          wu += nu;
        }
      }
      if (lane < 8) {
        s_warp_c[lane] = wc - s_warp_c[lane];
        s_warp_u[lane] = wu - s_warp_u[lane];
      }
    }
    __syncthreads();

    const int base_c = s_warp_c[warp] + (cp - (c0 + c1));
    const int base_u = s_warp_u[warp] + (up - (u0 + u1));
    const int st0 = base_c;
    const int st1 = base_c + c0;
    const int rk0 = base_u;
    const int rk1 = base_u + u0;

    s_start[e0] = st0;
    s_start[e1] = st1;
    if (c0 > 0) {
      unique_experts[rk0] = e0;
      bucket_offsets[rk0] = st0;
    }
    if (c1 > 0) {
      unique_experts[rk1] = e1;
      bucket_offsets[rk1] = st1;
    }
    if (tid == 255) {
      const int tot_u = rk1 + u1;
      const int tot_c = st1 + c1;
      num_unique[0] = tot_u;
      bucket_offsets[tot_u] = tot_c;
    }
    __syncthreads();

    if (tid < P) {
      const int e = s_ids[tid];
      if (static_cast<unsigned int>(e) < static_cast<unsigned int>(E)) {
        int rank = 0;
        for (int j = 0; j < tid; ++j) {
          rank += (s_ids[j] == e) ? 1 : 0;
        }
        pair_indices[s_start[e] + rank] = tid;
      }
    }
    return;
  }

  // blockIdx.x >= 1: quantize X[M, K] into per-token g32 Q8_1
  const int groups_per_row = K >> 5;
  const int total_groups = M * groups_per_row;
  const int gtid = (blockIdx.x - 1) * blockDim.x + threadIdx.x;
  const int grp = gtid >> 3;
  const int sub = gtid & 7;
  if (grp >= total_groups) return;

  const int row = grp / groups_per_row;
  const int s = grp - row * groups_per_row;
  const uint2 raw = *reinterpret_cast<const uint2*>(
      x + static_cast<size_t>(row) * K + s * 32 + sub * 4);
  const __nv_bfloat162* pairs = reinterpret_cast<const __nv_bfloat162*>(&raw);
  const float2 f0 = __bfloat1622float2(pairs[0]);
  const float2 f1 = __bfloat1622float2(pairs[1]);
  float v[4] = {f0.x, f0.y, f1.x, f1.y};
  float amax = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
#pragma unroll
  for (int off = 1; off < 8; off <<= 1) {
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off, 8));
  }
  const float inv = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
  int q[4];
  int qsum = 0;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    q[j] = (amax > 0.0f) ? static_cast<int>(rintf(v[j] * inv)) : 0;
    qsum += q[j];
  }
#pragma unroll
  for (int off = 1; off < 8; off <<= 1) {
    qsum += __shfl_xor_sync(0xffffffffu, qsum, off, 8);
  }
  *reinterpret_cast<unsigned int*>(xq + static_cast<size_t>(grp) * 32 + sub * 4) =
      (static_cast<unsigned int>(q[0] & 0xff)) |
      (static_cast<unsigned int>(q[1] & 0xff) << 8) |
      (static_cast<unsigned int>(q[2] & 0xff) << 16) |
      (static_cast<unsigned int>(q[3] & 0xff) << 24);
  if (sub == 0) {
    const float d = (amax > 0.0f) ? (amax * (1.0f / 127.0f)) : 0.0f;
    x_scales[grp] = d;
    x_sums[grp] = d * static_cast<float>(qsum);
  }
}
)cuda"
R"cuda(
// ---------------------------------------------------------------------------
// 2A. Fused Grouped Q4_K gate + up + SwiGLU (47 of 48 layers!)
//     100% CONTIGUOUS full-row 20 KB cp.async.cg streaming (zero row-stride skips!)
//     with zero-padding XOR bank swizzle (c ^ (r & 7)) + contiguous scales/biases cp.async.cg!
//     5 warps (160 threads) per CTA, processes TILES_PER_CTA (2) consecutive 16-row tiles.
//     2-stage SMEM = 47,360 B (< 48 KB static SMEM, 2 CTAs/SM = 10 warps/SM).
//     Each warp owns 2 super-blocks (5 * 2 = 10 exact!).
//     Grid: dim3(N / (16 * TILES_PER_CTA), max_unique), Block: dim3(32, 5).
// ---------------------------------------------------------------------------
template <int TILES_PER_CTA, bool ALLOW_HI>
__global__ __launch_bounds__(160, 2) void qwen4_moe_gate_up_swiglu_q4k_few(
    const unsigned int* __restrict__ wg,
    const unsigned char* __restrict__ sg,
    const unsigned short* __restrict__ bg,
    const unsigned int* __restrict__ wu,
    const unsigned char* __restrict__ su,
    const unsigned short* __restrict__ bu,
    const signed char* __restrict__ xq,
    const float* __restrict__ x_scales,
    const float* __restrict__ x_sums,
    const int* __restrict__ unique_experts,
    const int* __restrict__ bucket_offsets,
    const int* __restrict__ pair_indices,
    const int* __restrict__ num_unique,
    nv_bfloat16* __restrict__ mid,
    int N,
    int K,
    int top_k) {
  const int u_idx = blockIdx.y;
  if (u_idx >= num_unique[0]) return;
  const int expert = unique_experts[u_idx];
  const int first = bucket_offsets[u_idx];
  const int cnt = bucket_offsets[u_idx + 1] - first;
  if (cnt <= 0) return;

  const int base_row = blockIdx.x * (16 * TILES_PER_CTA);
  if (base_row >= N) return;

  const int lane = threadIdx.x;
  const int warp = threadIdx.y; // 0..4
  const int tid = warp * 32 + lane;
  const int g = lane >> 2;      // 0..7
  const int t = lane & 3;       // 0..3

  constexpr int UNITS = 10;
  constexpr int GROUPS = 80;
  constexpr int ROW_V4 = 80;    // 10 * 8 uint4s = 1280 B/row

  struct StageSmem {
    alignas(16) uint4 w[16][ROW_V4];       // 20,480 B (XOR swizzle: c ^ (r & 7))
    alignas(16) uint4 sc[16][UNITS];       //  2,560 B (160 uint4s)
    alignas(16) unsigned int bi[16][UNITS];//    640 B (40 uint4s; half2 [d, dmin])
  };
  __shared__ StageSmem smem[2];            // 47,360 B!
  __shared__ int s_tok[16];
  __shared__ int s_pair[16];

  if (tid < cnt) {
    int p = pair_indices[first + tid];
    s_pair[tid] = p;
    s_tok[tid] = p / top_k;
  } else if (tid < 16) {
    s_pair[tid] = 0;
    s_tok[tid] = -1;
  }

  const size_t exp_base = static_cast<size_t>(expert) * N;
  auto load_mat = [&](int st, const unsigned int* wb, const unsigned char* sb,
                      const unsigned short* bb, int row0) {
    const size_t r_off = exp_base + row0;
    const uint4* w_src = reinterpret_cast<const uint4*>(wb) + r_off * ROW_V4;
    // 16 * 80 = 1280 uint4s = EXACTLY 8 sweeps of 160 threads, 100% CONTIGUOUS DRAM!
#pragma unroll
    for (int iter = 0; iter < 8; ++iter) {
      const int idx = tid + iter * 160;
      const int r = idx / ROW_V4;
      const int c = idx - r * ROW_V4;
      q4m_cp_async16(&smem[st].w[r][c ^ (r & 7)], w_src + idx);
    }
    // 160 uint4s of scales = EXACTLY 1 uint4 per thread, 100% CONTIGUOUS DRAM!
    const uint4* sc_src = reinterpret_cast<const uint4*>(sb) + r_off * UNITS;
    q4m_cp_async16(reinterpret_cast<uint4*>(smem[st].sc) + tid, sc_src + tid);
    // 40 uint4s of biases (16 * 10 * 4 B = 640 B), 100% CONTIGUOUS DRAM!
    if (tid < 40) {
      const uint4* bi_src = reinterpret_cast<const uint4*>(bb) + (r_off * UNITS) / 4;
      q4m_cp_async16(reinterpret_cast<uint4*>(smem[st].bi) + tid, bi_src + tid);
    }
    q4m_cp_commit();
  };

  load_mat(0, wg, sg, bg, base_row);
  __syncthreads();

  const int tok_g_lo = s_tok[g];
  const int tok_c0_lo = s_tok[2 * t];
  const int tok_c1_lo = s_tok[2 * t + 1];
  const int tok_g_hi = ALLOW_HI ? s_tok[8 + g] : -1;
  const int tok_c0_hi = ALLOW_HI ? s_tok[8 + 2 * t] : -1;
  const int tok_c1_hi = ALLOW_HI ? s_tok[8 + 2 * t + 1] : -1;
  const bool do_hi = ALLOW_HI && (cnt > 8);

  const int total_steps = TILES_PER_CTA * 2; // (gate0, up0, gate1, up1, ...)
  float g_tot_lo[4], g_tot_hi[4];

#pragma unroll 1
  for (int step = 0; step < total_steps; ++step) {
    const int ti = step >> 1;
    const bool is_up = (step & 1) != 0;
    const int row0 = base_row + ti * 16;
    if (row0 >= N) break;

    const int cur = step & 1;
    const int nxt = cur ^ 1;
    if (step + 1 < total_steps) {
      const int n_ti = (step + 1) >> 1;
      const bool n_up = ((step + 1) & 1) != 0;
      const int n_row0 = base_row + n_ti * 16;
      if (n_row0 < N) {
        load_mat(nxt, n_up ? wu : wg, n_up ? su : sg, n_up ? bu : bg, n_row0);
        q4m_cp_wait<1>();
      } else {
        q4m_cp_wait<0>();
      }
    } else {
      q4m_cp_wait<0>();
    }
    __syncthreads();

    float tot_lo[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float tot_hi[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    // Each of 5 warps processes 2 super-blocks: warp*2 and warp*2+1
#pragma unroll 1
    for (int ku = 0; ku < 2; ++ku) {
      const int unit = warp * 2 + ku;
      const uint4 sc0 = smem[cur].sc[g][unit];
      const uint4 sc1 = smem[cur].sc[g + 8][unit];
      float dot_lo[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      float dot_hi[4] = {0.0f, 0.0f, 0.0f, 0.0f};

#pragma unroll 1
      for (int s = 0; s < 8; ++s) {
        // Unswizzle: sub-block s is 1 uint4 at c = unit * 8 + s
        const unsigned int* seg0 = reinterpret_cast<const unsigned int*>(
            &smem[cur].w[g][unit * 8 + (s ^ g)]);
        const unsigned int* seg1 = reinterpret_cast<const unsigned int*>(
            &smem[cur].w[g + 8][unit * 8 + (s ^ g)]);
        unsigned int a0, a1, a2, a3;
        q4m_load_pair_kq<0>(seg0, t, a0, a2);
        q4m_load_pair_kq<0>(seg1, t, a1, a3);

        const unsigned int sw0 = (s < 2) ? sc0.x : (s < 4 ? sc0.y : (s < 6 ? sc0.z : sc0.w));
        const unsigned int sw1 = (s < 2) ? sc1.x : (s < 4 ? sc1.y : (s < 6 ? sc1.z : sc1.w));
        const int sh = (s & 1) * 16;
        const int ws0 = (sw0 >> sh) & 0xff;
        const int ws1 = (sw1 >> sh) & 0xff;
        const int grp_k = unit * 8 + s;

        unsigned int b0 = 0u, b1 = 0u;
        if (tok_g_lo >= 0) {
          const unsigned int* xp = reinterpret_cast<const unsigned int*>(
              xq + (static_cast<size_t>(tok_g_lo) * GROUPS + grp_k) * 32);
          b0 = xp[t];
          b1 = xp[4 + t];
        }
        int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
        q4m_mma_s8(a0, a1, a2, a3, b0, b1, c0, c1, c2, c3);
        const float ad0 = (tok_c0_lo >= 0) ? x_scales[tok_c0_lo * GROUPS + grp_k] : 0.0f;
        const float ad1 = (tok_c1_lo >= 0) ? x_scales[tok_c1_lo * GROUPS + grp_k] : 0.0f;
        dot_lo[0] = fmaf(ad0, static_cast<float>(c0 * ws0), dot_lo[0]);
        dot_lo[1] = fmaf(ad1, static_cast<float>(c1 * ws0), dot_lo[1]);
        dot_lo[2] = fmaf(ad0, static_cast<float>(c2 * ws1), dot_lo[2]);
        dot_lo[3] = fmaf(ad1, static_cast<float>(c3 * ws1), dot_lo[3]);

        if constexpr (ALLOW_HI) {
          if (do_hi) {
            unsigned int hb0 = 0u, hb1 = 0u;
            if (tok_g_hi >= 0) {
              const unsigned int* xp = reinterpret_cast<const unsigned int*>(
                  xq + (static_cast<size_t>(tok_g_hi) * GROUPS + grp_k) * 32);
              hb0 = xp[t];
              hb1 = xp[4 + t];
            }
            int h0 = 0, h1 = 0, h2 = 0, h3 = 0;
            q4m_mma_s8(a0, a1, a2, a3, hb0, hb1, h0, h1, h2, h3);
            const float had0 = (tok_c0_hi >= 0) ? x_scales[tok_c0_hi * GROUPS + grp_k] : 0.0f;
            const float had1 = (tok_c1_hi >= 0) ? x_scales[tok_c1_hi * GROUPS + grp_k] : 0.0f;
            dot_hi[0] = fmaf(had0, static_cast<float>(h0 * ws0), dot_hi[0]);
            dot_hi[1] = fmaf(had1, static_cast<float>(h1 * ws0), dot_hi[1]);
            dot_hi[2] = fmaf(had0, static_cast<float>(h2 * ws1), dot_hi[2]);
            dot_hi[3] = fmaf(had1, static_cast<float>(h3 * ws1), dot_hi[3]);
          }
        }
      }

      float min_lo[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      float min_hi[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll 1
      for (int s = 0; s < 8; ++s) {
        const unsigned int sw0 = (s < 2) ? sc0.x : (s < 4 ? sc0.y : (s < 6 ? sc0.z : sc0.w));
        const unsigned int sw1 = (s < 2) ? sc1.x : (s < 4 ? sc1.y : (s < 6 ? sc1.z : sc1.w));
        const int sh = (s & 1) * 16 + 8;
        const float wm0 = static_cast<float>((sw0 >> sh) & 0xff);
        const float wm1 = static_cast<float>((sw1 >> sh) & 0xff);
        const int grp_k = unit * 8 + s;
        const float as0 = (tok_c0_lo >= 0) ? x_sums[tok_c0_lo * GROUPS + grp_k] : 0.0f;
        const float as1 = (tok_c1_lo >= 0) ? x_sums[tok_c1_lo * GROUPS + grp_k] : 0.0f;
        min_lo[0] = fmaf(as0, wm0, min_lo[0]);
        min_lo[1] = fmaf(as1, wm0, min_lo[1]);
        min_lo[2] = fmaf(as0, wm1, min_lo[2]);
        min_lo[3] = fmaf(as1, wm1, min_lo[3]);
        if constexpr (ALLOW_HI) {
          if (do_hi) {
            const float has0 = (tok_c0_hi >= 0) ? x_sums[tok_c0_hi * GROUPS + grp_k] : 0.0f;
            const float has1 = (tok_c1_hi >= 0) ? x_sums[tok_c1_hi * GROUPS + grp_k] : 0.0f;
            min_hi[0] = fmaf(has0, wm0, min_hi[0]);
            min_hi[1] = fmaf(has1, wm0, min_hi[1]);
            min_hi[2] = fmaf(has0, wm1, min_hi[2]);
            min_hi[3] = fmaf(has1, wm1, min_hi[3]);
          }
        }
      }

      const unsigned int b0_pair = smem[cur].bi[g][unit];
      const unsigned int b1_pair = smem[cur].bi[g + 8][unit];
      const float2 dm0 = __half22float2(*reinterpret_cast<const __half2*>(&b0_pair));
      const float2 dm1 = __half22float2(*reinterpret_cast<const __half2*>(&b1_pair));

      tot_lo[0] += dm0.x * dot_lo[0] - dm0.y * min_lo[0];
      tot_lo[1] += dm0.x * dot_lo[1] - dm0.y * min_lo[1];
      tot_lo[2] += dm1.x * dot_lo[2] - dm1.y * min_lo[2];
      tot_lo[3] += dm1.x * dot_lo[3] - dm1.y * min_lo[3];
      if constexpr (ALLOW_HI) {
        if (do_hi) {
          tot_hi[0] += dm0.x * dot_hi[0] - dm0.y * min_hi[0];
          tot_hi[1] += dm0.x * dot_hi[1] - dm0.y * min_hi[1];
          tot_hi[2] += dm1.x * dot_hi[2] - dm1.y * min_hi[2];
          tot_hi[3] += dm1.x * dot_hi[3] - dm1.y * min_hi[3];
        }
      }
    }

    if (!is_up) {
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        g_tot_lo[q] = tot_lo[q];
        if constexpr (ALLOW_HI) g_tot_hi[q] = tot_hi[q];
      }
      __syncthreads();
    } else {
      // Reuse smem[cur].w (already consumed!) for warp reduction of gate + up!
      __syncthreads();
      float* red_g = reinterpret_cast<float*>(smem[cur].w);
      float* red_u = red_g + 5 * 16 * 16;
      red_g[(warp * 16 + 2 * t) * 16 + g]         = g_tot_lo[0];
      red_g[(warp * 16 + 2 * t + 1) * 16 + g]     = g_tot_lo[1];
      red_g[(warp * 16 + 2 * t) * 16 + g + 8]     = g_tot_lo[2];
      red_g[(warp * 16 + 2 * t + 1) * 16 + g + 8] = g_tot_lo[3];
      red_u[(warp * 16 + 2 * t) * 16 + g]         = tot_lo[0];
      red_u[(warp * 16 + 2 * t + 1) * 16 + g]     = tot_lo[1];
      red_u[(warp * 16 + 2 * t) * 16 + g + 8]     = tot_lo[2];
      red_u[(warp * 16 + 2 * t + 1) * 16 + g + 8] = tot_lo[3];
      if constexpr (ALLOW_HI) {
        if (do_hi) {
          red_g[(warp * 16 + 8 + 2 * t) * 16 + g]         = g_tot_hi[0];
          red_g[(warp * 16 + 8 + 2 * t + 1) * 16 + g]     = g_tot_hi[1];
          red_g[(warp * 16 + 8 + 2 * t) * 16 + g + 8]     = g_tot_hi[2];
          red_g[(warp * 16 + 8 + 2 * t + 1) * 16 + g + 8] = g_tot_hi[3];
          red_u[(warp * 16 + 8 + 2 * t) * 16 + g]         = tot_hi[0];
          red_u[(warp * 16 + 8 + 2 * t + 1) * 16 + g]     = tot_hi[1];
          red_u[(warp * 16 + 8 + 2 * t) * 16 + g + 8]     = tot_hi[2];
          red_u[(warp * 16 + 8 + 2 * t + 1) * 16 + g + 8] = tot_hi[3];
        }
      }
      __syncthreads();
      const int total_out = cnt * 16;
      for (int idx = tid; idx < total_out; idx += 160) {
        const int c = idx >> 4;
        const int r = idx & 15;
        float gv = 0.0f, uv = 0.0f;
#pragma unroll
        for (int ww = 0; ww < 5; ++ww) {
          gv += red_g[(ww * 16 + c) * 16 + r];
          uv += red_u[(ww * 16 + c) * 16 + r];
        }
        mid[static_cast<size_t>(s_pair[c]) * N + row0 + r] =
            __float2bfloat16(q4m_silu(gv) * uv);
      }
      __syncthreads();
    }
  }
}
)cuda"
R"cuda(
// ---------------------------------------------------------------------------
// 2B. Fused Grouped Q5_K gate + up + SwiGLU (1 of 48 layers; also supports Q4_K)
//     5 warps (160 threads) per CTA, 16 output rows per CTA.
//     2-stage cp.async.cg pipeline: 5 consecutive super-blocks per half (800 B/row Q5_K)
//     + contiguous scales/biases loads.
//     Grid: dim3(N/16, max_unique), Block: dim3(32, 5).
// ---------------------------------------------------------------------------
template <int mode, bool ALLOW_HI>
__global__ __launch_bounds__(160, 3) void qwen4_moe_gate_up_swiglu_few(
    const unsigned int* __restrict__ wg,
    const unsigned char* __restrict__ sg,
    const unsigned short* __restrict__ bg,
    const unsigned int* __restrict__ wu,
    const unsigned char* __restrict__ su,
    const unsigned short* __restrict__ bu,
    const signed char* __restrict__ xq,
    const float* __restrict__ x_scales,
    const float* __restrict__ x_sums,
    const int* __restrict__ unique_experts,
    const int* __restrict__ bucket_offsets,
    const int* __restrict__ pair_indices,
    const int* __restrict__ num_unique,
    nv_bfloat16* __restrict__ mid,
    int N,
    int K,
    int top_k) {
  const int u_idx = blockIdx.y;
  if (u_idx >= num_unique[0]) return;
  const int expert = unique_experts[u_idx];
  const int first = bucket_offsets[u_idx];
  const int cnt = bucket_offsets[u_idx + 1] - first;
  if (cnt <= 0) return;

  const int row0 = blockIdx.x * 16;
  if (row0 >= N) return;

  const int lane = threadIdx.x;
  const int warp = threadIdx.y; // 0..4
  const int tid = warp * 32 + lane;
  const int g = lane >> 2;
  const int t = lane & 3;

  constexpr int BITS = (mode == 1) ? 5 : 4;
  constexpr int V4_U = BITS * 2;
  constexpr int HALF_U = 5;
  constexpr int HALF_V4 = HALF_U * V4_U;
  constexpr int PAD_V4 = HALF_V4 + 1;

  const int units = K >> 8;
  const int groups = K >> 5;
  const size_t row_w_stride_u4 = static_cast<size_t>(units) * V4_U;
  const size_t exp_row0 = static_cast<size_t>(expert) * N + row0;

  union SharedBuf {
    uint4 w[2][16][PAD_V4];
    struct {
      float gate[5][16][16];
      float up[5][16][16];
    } red;
  };
  __shared__ SharedBuf smem;
  __shared__ int s_tok[16];
  __shared__ int s_pair[16];

  if (tid < cnt) {
    int p = pair_indices[first + tid];
    s_pair[tid] = p;
    s_tok[tid] = p / top_k;
  } else if (tid < 16) {
    s_pair[tid] = 0;
    s_tok[tid] = -1;
  }

  auto load_half = [&](int st, const unsigned int* w_bank, int half_idx) {
    const uint4* w_u4 = reinterpret_cast<const uint4*>(w_bank) +
        exp_row0 * row_w_stride_u4 + static_cast<size_t>(half_idx) * HALF_V4;
    constexpr int TOTAL = 16 * HALF_V4;
#pragma unroll
    for (int idx = tid; idx < TOTAL; idx += 160) {
      const int r = idx / HALF_V4;
      const int c = idx - r * HALF_V4;
      q4m_cp_async16(&smem.w[st][r][c], w_u4 + static_cast<size_t>(r) * row_w_stride_u4 + c);
    }
    q4m_cp_commit();
  };

  load_half(0, wg, 0);
  __syncthreads();

  const int tok_g_lo = s_tok[g];
  const int tok_c0_lo = s_tok[2 * t];
  const int tok_c1_lo = s_tok[2 * t + 1];
  const int tok_g_hi = ALLOW_HI ? s_tok[8 + g] : -1;
  const int tok_c0_hi = ALLOW_HI ? s_tok[8 + 2 * t] : -1;
  const int tok_c1_hi = ALLOW_HI ? s_tok[8 + 2 * t + 1] : -1;
  const bool do_hi = ALLOW_HI && (cnt > 8);

  float g_tot_lo[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float g_tot_hi[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float u_tot_lo[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float u_tot_hi[4] = {0.0f, 0.0f, 0.0f, 0.0f};

#pragma unroll 1
  for (int stage = 0; stage < 4; ++stage) {
    const int cur = stage & 1;
    const int nxt = cur ^ 1;
    if (stage == 0) {
      load_half(nxt, wg, 1);
      q4m_cp_wait<1>();
    } else if (stage == 1) {
      load_half(nxt, wu, 0);
      q4m_cp_wait<1>();
    } else if (stage == 2) {
      load_half(nxt, wu, 1);
      q4m_cp_wait<1>();
    } else {
      q4m_cp_wait<0>();
    }
    __syncthreads();

    const bool is_up = (stage >= 2);
    const int half_idx = stage & 1;
    const int unit = half_idx * HALF_U + warp;
    const unsigned char* sc_bank = is_up ? su : sg;
    const unsigned short* bi_bank = is_up ? bu : bg;

    const size_t r0_idx = exp_row0 + g;
    const size_t r1_idx = exp_row0 + g + 8;
    const uint4 sc0 = *reinterpret_cast<const uint4*>(sc_bank + (r0_idx * units + unit) * 16);
    const uint4 sc1 = *reinterpret_cast<const uint4*>(sc_bank + (r1_idx * units + unit) * 16);

    const unsigned int* w_r0 = reinterpret_cast<const unsigned int*>(&smem.w[cur][g][warp * V4_U]);
    const unsigned int* w_r1 = reinterpret_cast<const unsigned int*>(&smem.w[cur][g + 8][warp * V4_U]);

    float dot_lo[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float dot_hi[4] = {0.0f, 0.0f, 0.0f, 0.0f};

#pragma unroll 1
    for (int s = 0; s < 8; ++s) {
      unsigned int a0, a1, a2, a3;
      q4m_load_pair_kq<mode>(w_r0 + s * BITS, t, a0, a2);
      q4m_load_pair_kq<mode>(w_r1 + s * BITS, t, a1, a3);

      const unsigned int sw0 = (s < 2) ? sc0.x : (s < 4 ? sc0.y : (s < 6 ? sc0.z : sc0.w));
      const unsigned int sw1 = (s < 2) ? sc1.x : (s < 4 ? sc1.y : (s < 6 ? sc1.z : sc1.w));
      const int sh = (s & 1) * 16;
      const int ws0 = (sw0 >> sh) & 0xff;
      const int ws1 = (sw1 >> sh) & 0xff;
      const int grp_k = unit * 8 + s;

      unsigned int b0 = 0u, b1 = 0u;
      if (tok_g_lo >= 0) {
        const unsigned int* xp = reinterpret_cast<const unsigned int*>(
            xq + (static_cast<size_t>(tok_g_lo) * groups + grp_k) * 32);
        b0 = xp[t];
        b1 = xp[4 + t];
      }
      int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
      q4m_mma_s8(a0, a1, a2, a3, b0, b1, c0, c1, c2, c3);
      const float ad0 = (tok_c0_lo >= 0) ? x_scales[tok_c0_lo * groups + grp_k] : 0.0f;
      const float ad1 = (tok_c1_lo >= 0) ? x_scales[tok_c1_lo * groups + grp_k] : 0.0f;
      dot_lo[0] = fmaf(ad0, static_cast<float>(c0 * ws0), dot_lo[0]);
      dot_lo[1] = fmaf(ad1, static_cast<float>(c1 * ws0), dot_lo[1]);
      dot_lo[2] = fmaf(ad0, static_cast<float>(c2 * ws1), dot_lo[2]);
      dot_lo[3] = fmaf(ad1, static_cast<float>(c3 * ws1), dot_lo[3]);

      if constexpr (ALLOW_HI) {
        if (do_hi) {
          unsigned int hb0 = 0u, hb1 = 0u;
          if (tok_g_hi >= 0) {
            const unsigned int* xp = reinterpret_cast<const unsigned int*>(
                xq + (static_cast<size_t>(tok_g_hi) * groups + grp_k) * 32);
            hb0 = xp[t];
            hb1 = xp[4 + t];
          }
          int h0 = 0, h1 = 0, h2 = 0, h3 = 0;
          q4m_mma_s8(a0, a1, a2, a3, hb0, hb1, h0, h1, h2, h3);
          const float had0 = (tok_c0_hi >= 0) ? x_scales[tok_c0_hi * groups + grp_k] : 0.0f;
          const float had1 = (tok_c1_hi >= 0) ? x_scales[tok_c1_hi * groups + grp_k] : 0.0f;
          dot_hi[0] = fmaf(had0, static_cast<float>(h0 * ws0), dot_hi[0]);
          dot_hi[1] = fmaf(had1, static_cast<float>(h1 * ws0), dot_hi[1]);
          dot_hi[2] = fmaf(had0, static_cast<float>(h2 * ws1), dot_hi[2]);
          dot_hi[3] = fmaf(had1, static_cast<float>(h3 * ws1), dot_hi[3]);
        }
      }
    }

    float min_lo[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float min_hi[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll 1
    for (int s = 0; s < 8; ++s) {
      const unsigned int sw0 = (s < 2) ? sc0.x : (s < 4 ? sc0.y : (s < 6 ? sc0.z : sc0.w));
      const unsigned int sw1 = (s < 2) ? sc1.x : (s < 4 ? sc1.y : (s < 6 ? sc1.z : sc1.w));
      const int sh = (s & 1) * 16 + 8;
      const float wm0 = static_cast<float>((sw0 >> sh) & 0xff);
      const float wm1 = static_cast<float>((sw1 >> sh) & 0xff);
      const int grp_k = unit * 8 + s;
      const float as0 = (tok_c0_lo >= 0) ? x_sums[tok_c0_lo * groups + grp_k] : 0.0f;
      const float as1 = (tok_c1_lo >= 0) ? x_sums[tok_c1_lo * groups + grp_k] : 0.0f;
      min_lo[0] = fmaf(as0, wm0, min_lo[0]);
      min_lo[1] = fmaf(as1, wm0, min_lo[1]);
      min_lo[2] = fmaf(as0, wm1, min_lo[2]);
      min_lo[3] = fmaf(as1, wm1, min_lo[3]);
      if constexpr (ALLOW_HI) {
        if (do_hi) {
          const float has0 = (tok_c0_hi >= 0) ? x_sums[tok_c0_hi * groups + grp_k] : 0.0f;
          const float has1 = (tok_c1_hi >= 0) ? x_sums[tok_c1_hi * groups + grp_k] : 0.0f;
          min_hi[0] = fmaf(has0, wm0, min_hi[0]);
          min_hi[1] = fmaf(has1, wm0, min_hi[1]);
          min_hi[2] = fmaf(has0, wm1, min_hi[2]);
          min_hi[3] = fmaf(has1, wm1, min_hi[3]);
        }
      }
    }

    const size_t bi0 = (r0_idx * units + unit) * 2;
    const size_t bi1 = (r1_idx * units + unit) * 2;
    const float d0 = __half2float(__ushort_as_half(bi_bank[bi0]));
    const float dm0 = __half2float(__ushort_as_half(bi_bank[bi0 + 1]));
    const float d1 = __half2float(__ushort_as_half(bi_bank[bi1]));
    const float dm1 = __half2float(__ushort_as_half(bi_bank[bi1 + 1]));

    float* t_lo = is_up ? u_tot_lo : g_tot_lo;
    t_lo[0] += d0 * dot_lo[0] - dm0 * min_lo[0];
    t_lo[1] += d0 * dot_lo[1] - dm0 * min_lo[1];
    t_lo[2] += d1 * dot_lo[2] - dm1 * min_lo[2];
    t_lo[3] += d1 * dot_lo[3] - dm1 * min_lo[3];
    if constexpr (ALLOW_HI) {
      if (do_hi) {
        float* t_hi = is_up ? u_tot_hi : g_tot_hi;
        t_hi[0] += d0 * dot_hi[0] - dm0 * min_hi[0];
        t_hi[1] += d0 * dot_hi[1] - dm0 * min_hi[1];
        t_hi[2] += d1 * dot_hi[2] - dm1 * min_hi[2];
        t_hi[3] += d1 * dot_hi[3] - dm1 * min_hi[3];
      }
    }
    __syncthreads();
  }

  smem.red.gate[warp][2 * t][g] = g_tot_lo[0];
  smem.red.gate[warp][2 * t + 1][g] = g_tot_lo[1];
  smem.red.gate[warp][2 * t][g + 8] = g_tot_lo[2];
  smem.red.gate[warp][2 * t + 1][g + 8] = g_tot_lo[3];
  smem.red.up[warp][2 * t][g] = u_tot_lo[0];
  smem.red.up[warp][2 * t + 1][g] = u_tot_lo[1];
  smem.red.up[warp][2 * t][g + 8] = u_tot_lo[2];
  smem.red.up[warp][2 * t + 1][g + 8] = u_tot_lo[3];
  if constexpr (ALLOW_HI) {
    if (do_hi) {
      smem.red.gate[warp][8 + 2 * t][g] = g_tot_hi[0];
      smem.red.gate[warp][8 + 2 * t + 1][g] = g_tot_hi[1];
      smem.red.gate[warp][8 + 2 * t][g + 8] = g_tot_hi[2];
      smem.red.gate[warp][8 + 2 * t + 1][g + 8] = g_tot_hi[3];
      smem.red.up[warp][8 + 2 * t][g] = u_tot_hi[0];
      smem.red.up[warp][8 + 2 * t + 1][g] = u_tot_hi[1];
      smem.red.up[warp][8 + 2 * t][g + 8] = u_tot_hi[2];
      smem.red.up[warp][8 + 2 * t + 1][g + 8] = u_tot_hi[3];
    }
  }
  __syncthreads();

  const int total_out = cnt * 16;
  for (int idx = tid; idx < total_out; idx += 160) {
    const int c = idx >> 4;
    const int r = idx & 15;
    float gv = 0.0f, uv = 0.0f;
#pragma unroll
    for (int ww = 0; ww < 5; ++ww) {
      gv += smem.red.gate[ww][c][r];
      uv += smem.red.up[ww][c][r];
    }
    mid[static_cast<size_t>(s_pair[c]) * N + row0 + r] =
        __float2bfloat16(q4m_silu(gv) * uv);
  }
}
)cuda"
R"cuda(
// ---------------------------------------------------------------------------
// 2C. Ultra-High-Occupancy M=1 dp4a Streaming Kernel for Q4_K gate + up + SwiGLU
//     0 bytes SMEM, ~32 regs/thread => 48 warps/SM occupancy!
//     8 threads (1/4 warp) own 1 output row; each thread reads 1 full 32-elem sub-block
//     (1 uint4 = 16 B) => 8 threads read 1 full 128 B cache line per super-block via
//     ld.global.cs.v4.u32! Interleaves gate+up to reuse Q8 activation loads in registers.
//     Grid: dim3(N / 16, top_k), Block: 128 (4 warps x 4 rows/warp = 16 rows/CTA).
// ---------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(128, 8) void qwen4_moe_gate_up_swiglu_q4k_m1(
    const unsigned int* __restrict__ wg,
    const unsigned char* __restrict__ sg,
    const unsigned short* __restrict__ bg,
    const unsigned int* __restrict__ wu,
    const unsigned char* __restrict__ su,
    const unsigned short* __restrict__ bu,
    const signed char* __restrict__ xq,
    const float* __restrict__ x_scales,
    const float* __restrict__ x_sums,
    const int* __restrict__ expert_ids,
    nv_bfloat16* __restrict__ mid,
    int N,
    int K) {
  const int slot = blockIdx.y;
  const int expert = expert_ids[slot];
  const int row = blockIdx.x * 16 + (threadIdx.x >> 3);
  const int sub = threadIdx.x & 7; // sub-block 0..7 inside each 256-elem super-block
  if (row >= N) return;

  const int units = K >> 8; // 10
  const size_t r_idx = static_cast<size_t>(expert) * N + row;
  const uint4* wg_row = reinterpret_cast<const uint4*>(wg) + r_idx * (units * 8);
  const uint4* wu_row = reinterpret_cast<const uint4*>(wu) + r_idx * (units * 8);
  const unsigned short* sg_row = reinterpret_cast<const unsigned short*>(sg) + r_idx * (units * 8);
  const unsigned short* su_row = reinterpret_cast<const unsigned short*>(su) + r_idx * (units * 8);
  const unsigned int* bg_row = reinterpret_cast<const unsigned int*>(bg) + r_idx * units;
  const unsigned int* bu_row = reinterpret_cast<const unsigned int*>(bu) + r_idx * units;

  float g_acc = 0.0f;
  float u_acc = 0.0f;

#pragma unroll 1
  for (int u = 0; u < units; ++u) {
    const int grp = u * 8 + sub;
    const uint4 xv0 = reinterpret_cast<const uint4*>(xq)[grp * 2 + 0];
    const uint4 xv1 = reinterpret_cast<const uint4*>(xq)[grp * 2 + 1];
    const float xsc = x_scales[grp];
    const float xsm = x_sums[grp];

    const uint4 gw = q4m_ld_cs_u4(wg_row + grp);
    const uint4 uw = q4m_ld_cs_u4(wu_row + grp);

    auto dp4a_q4 = [&](const uint4& wv) -> int {
      int d = 0;
      d = q4m_dp4a(q4m_unpack4_kq<0>(wv.x & 0xFFFFu), xv0.x, d);
      d = q4m_dp4a(q4m_unpack4_kq<0>(wv.x >> 16),     xv0.y, d);
      d = q4m_dp4a(q4m_unpack4_kq<0>(wv.y & 0xFFFFu), xv0.z, d);
      d = q4m_dp4a(q4m_unpack4_kq<0>(wv.y >> 16),     xv0.w, d);
      d = q4m_dp4a(q4m_unpack4_kq<0>(wv.z & 0xFFFFu), xv1.x, d);
      d = q4m_dp4a(q4m_unpack4_kq<0>(wv.z >> 16),     xv1.y, d);
      d = q4m_dp4a(q4m_unpack4_kq<0>(wv.w & 0xFFFFu), xv1.z, d);
      d = q4m_dp4a(q4m_unpack4_kq<0>(wv.w >> 16),     xv1.w, d);
      return d;
    };

    const int gd = dp4a_q4(gw);
    const int ud = dp4a_q4(uw);

    const unsigned short gsc_m = sg_row[grp];
    const unsigned short usc_m = su_row[grp];
    const int gws = gsc_m & 0xff;
    const int gwm = gsc_m >> 8;
    const int uws = usc_m & 0xff;
    const int uwm = usc_m >> 8;

    const unsigned int gb_pair = bg_row[u];
    const unsigned int ub_pair = bu_row[u];
    const float2 gdm = __half22float2(*reinterpret_cast<const __half2*>(&gb_pair));
    const float2 udm = __half22float2(*reinterpret_cast<const __half2*>(&ub_pair));

    g_acc = fmaf(gdm.x * xsc, static_cast<float>(gd * gws),
                 fmaf(-gdm.y * xsm, static_cast<float>(gwm), g_acc));
    u_acc = fmaf(udm.x * xsc, static_cast<float>(ud * uws),
                 fmaf(-udm.y * xsm, static_cast<float>(uwm), u_acc));
  }

#pragma unroll
  for (int off = 4; off > 0; off >>= 1) {
    g_acc += __shfl_xor_sync(0xffffffffu, g_acc, off, 8);
    u_acc += __shfl_xor_sync(0xffffffffu, u_acc, off, 8);
  }
  if (sub == 0) {
    mid[static_cast<size_t>(slot) * N + row] = __float2bfloat16(q4m_silu(g_acc) * u_acc);
  }
}
)cuda"
R"cuda(
// ---------------------------------------------------------------------------
// 3. Fused Grouped Down Projection (Q5_1 bits=5, Q8_0 bits=8, g32 f16 s+b)
//    N = 2560, K = 640 (20 groups of 32).
//    5 warps (160 threads) per CTA, processes TILES_PER_CTA consecutive 16-row
//    tiles with 2-stage double-buffered 100% contiguous cp.async.cg!
//    Grid: dim3(N / (16 * TILES_PER_CTA), max_unique), Block: dim3(32, 5).
// ---------------------------------------------------------------------------
template <int BITS>
__device__ __forceinline__ void q4m_unpack8_affine_bf16(
    const unsigned int* seg, int t, unsigned int out_bf16x2[4]) {
  if constexpr (BITS == 5) {
    const int sh = t * 8;
    const unsigned int lo32 = (t == 0) ? seg[0] : __funnelshift_r(seg[t], seg[t + 1], sh);
    const unsigned int hi8 = seg[t + 1] >> sh;
    out_bf16x2[0] = q4m_pack_bf16x2(float(lo32 & 31u), float((lo32 >> 5) & 31u));
    out_bf16x2[1] = q4m_pack_bf16x2(float((lo32 >> 10) & 31u), float((lo32 >> 15) & 31u));
    out_bf16x2[2] = q4m_pack_bf16x2(float((lo32 >> 20) & 31u), float((lo32 >> 25) & 31u));
    out_bf16x2[3] = q4m_pack_bf16x2(
        float(((lo32 >> 30) | ((hi8 & 7u) << 2)) & 31u), float((hi8 >> 3) & 31u));
  } else {
    const unsigned int u0 = seg[2 * t];
    const unsigned int u1 = seg[2 * t + 1];
    out_bf16x2[0] = q4m_pack_bf16x2(float(u0 & 0xffu), float((u0 >> 8) & 0xffu));
    out_bf16x2[1] = q4m_pack_bf16x2(float((u0 >> 16) & 0xffu), float(u0 >> 24));
    out_bf16x2[2] = q4m_pack_bf16x2(float(u1 & 0xffu), float((u1 >> 8) & 0xffu));
    out_bf16x2[3] = q4m_pack_bf16x2(float((u1 >> 16) & 0xffu), float(u1 >> 24));
  }
}

template <int BITS, int TILES_PER_CTA, bool ALLOW_HI, bool ATOMIC_COMBINE>
__global__ __launch_bounds__(160, ALLOW_HI ? 2 : 3) void qwen4_moe_down_few(
    const unsigned int* __restrict__ wd,
    const unsigned short* __restrict__ sd,
    const unsigned short* __restrict__ bd,
    const nv_bfloat16* __restrict__ mid,
    const float* __restrict__ scores_f32,
    const int* __restrict__ unique_experts,
    const int* __restrict__ bucket_offsets,
    const int* __restrict__ pair_indices,
    const int* __restrict__ num_unique,
    void* __restrict__ out_ptr,
    int N,
    int K,
    int top_k) {
  const int u_idx = blockIdx.y;
  if (u_idx >= num_unique[0]) return;
  const int expert = unique_experts[u_idx];
  const int first = bucket_offsets[u_idx];
  const int cnt = bucket_offsets[u_idx + 1] - first;
  if (cnt <= 0) return;

  const int base_row = blockIdx.x * (16 * TILES_PER_CTA);
  if (base_row >= N) return;

  const int lane = threadIdx.x;
  const int warp = threadIdx.y; // 0..4
  const int tid = warp * 32 + lane;
  const int g = lane >> 2;
  const int t = lane & 3;

  constexpr int GROUPS = 20;
  constexpr int ROW_U4 = (BITS == 5) ? 25 : 40;
  constexpr int PAD_U4 = (BITS == 5) ? 25 : 41;
  constexpr int TILE_W_U4 = 16 * ROW_U4;
  constexpr int TILE_SB_U4 = 40;

  struct StageSmem {
    alignas(16) uint4 w[16][PAD_U4];
    alignas(16) unsigned short sc[16][20];
    alignas(16) unsigned short bi[16][20];
  };
  __shared__ StageSmem smem[2];
  __shared__ float s_red[5][16][17];
  __shared__ int s_pair[16];
  __shared__ float s_score[16];

  if (tid < cnt) {
    int p = pair_indices[first + tid];
    s_pair[tid] = p;
    if constexpr (ATOMIC_COMBINE) s_score[tid] = scores_f32[p];
  } else if (tid < 16) {
    s_pair[tid] = -1;
    if constexpr (ATOMIC_COMBINE) s_score[tid] = 0.0f;
  }

  const size_t exp_base = static_cast<size_t>(expert) * N;
  const uint4* wd_u4 = reinterpret_cast<const uint4*>(wd);
  const uint4* sd_u4 = reinterpret_cast<const uint4*>(sd);
  const uint4* bd_u4 = reinterpret_cast<const uint4*>(bd);

  auto load_tile = [&](int st, int row0) {
    const size_t r_off = exp_base + row0;
    const uint4* w_src = wd_u4 + r_off * ROW_U4;
    if constexpr (BITS == 5) {
      uint4* w_dst = &smem[st].w[0][0];
      for (int i = tid; i < TILE_W_U4; i += 160) {
        q4m_cp_async16(w_dst + i, w_src + i);
      }
    } else {
#pragma unroll
      for (int iter = 0; iter < 4; ++iter) {
        const int i = tid + iter * 160;
        const int r = i / 40;
        const int c = i - r * 40;
        q4m_cp_async16(&smem[st].w[r][c], w_src + i);
      }
    }
    const uint4* s_src = sd_u4 + (r_off * GROUPS) / 8;
    const uint4* b_src = bd_u4 + (r_off * GROUPS) / 8;
    if (tid < TILE_SB_U4) {
      q4m_cp_async16(reinterpret_cast<uint4*>(smem[st].sc) + tid, s_src + tid);
    } else if (tid < 2 * TILE_SB_U4) {
      const int k = tid - TILE_SB_U4;
      q4m_cp_async16(reinterpret_cast<uint4*>(smem[st].bi) + k, b_src + k);
    }
    q4m_cp_commit();
  };

  load_tile(0, base_row);
  __syncthreads();

  const int pair_g_lo = s_pair[g];
  const int pair_g_hi = ALLOW_HI ? s_pair[8 + g] : -1;
  const bool do_hi = ALLOW_HI && (cnt > 8);

  uint4 x_lo[4];
  float xs_lo_c0[4], xs_lo_c1[4];
  uint4 x_hi[ALLOW_HI ? 4 : 1];
  float xs_hi_c0[ALLOW_HI ? 4 : 1], xs_hi_c1[ALLOW_HI ? 4 : 1];

  const size_t mid_stride_u4 = static_cast<size_t>(K) >> 3;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const int grp = warp * 4 + k;
    x_lo[k] = (pair_g_lo >= 0) ? reinterpret_cast<const uint4*>(mid)[
        static_cast<size_t>(pair_g_lo) * mid_stride_u4 + grp * 4 + t] : make_uint4(0u, 0u, 0u, 0u);
    float my_s = (pair_g_lo >= 0) ? q4m_sum_bf16x8(x_lo[k]) : 0.0f;
    my_s += __shfl_xor_sync(0xffffffffu, my_s, 1);
    my_s += __shfl_xor_sync(0xffffffffu, my_s, 2);
    xs_lo_c0[k] = __shfl_sync(0xffffffffu, my_s, (2 * t) * 4);
    xs_lo_c1[k] = __shfl_sync(0xffffffffu, my_s, (2 * t + 1) * 4);

    if constexpr (ALLOW_HI) {
      if (do_hi) {
        x_hi[k] = (pair_g_hi >= 0) ? reinterpret_cast<const uint4*>(mid)[
            static_cast<size_t>(pair_g_hi) * mid_stride_u4 + grp * 4 + t] : make_uint4(0u, 0u, 0u, 0u);
        float my_sh = (pair_g_hi >= 0) ? q4m_sum_bf16x8(x_hi[k]) : 0.0f;
        my_sh += __shfl_xor_sync(0xffffffffu, my_sh, 1);
        my_sh += __shfl_xor_sync(0xffffffffu, my_sh, 2);
        xs_hi_c0[k] = __shfl_sync(0xffffffffu, my_sh, (2 * t) * 4);
        xs_hi_c1[k] = __shfl_sync(0xffffffffu, my_sh, (2 * t + 1) * 4);
      }
    }
  }

#pragma unroll 1
  for (int ti = 0; ti < TILES_PER_CTA; ++ti) {
    const int row0 = base_row + ti * 16;
    if (row0 >= N) break;
    const int cur = ti & 1;
    const int nxt = cur ^ 1;
    const bool has_next = (ti + 1 < TILES_PER_CTA) && (row0 + 16 < N);
    if (has_next) {
      load_tile(nxt, row0 + 16);
      q4m_cp_wait<1>();
    } else {
      q4m_cp_wait<0>();
    }
    __syncthreads();

    float tot_lo[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float tot_hi[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const unsigned int* w_r0 = reinterpret_cast<const unsigned int*>(smem[cur].w[g]);
    const unsigned int* w_r1 = reinterpret_cast<const unsigned int*>(smem[cur].w[g + 8]);

#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const int grp = warp * 4 + k;
      unsigned int wb0[4], wb1[4];
      q4m_unpack8_affine_bf16<BITS>(w_r0 + grp * BITS, t, wb0);
      q4m_unpack8_affine_bf16<BITS>(w_r1 + grp * BITS, t, wb1);

      float d0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      float d1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      q4m_mma_bf16(d0[0], d0[1], d0[2], d0[3], wb0[0], wb1[0], wb0[1], wb1[1], x_lo[k].x, x_lo[k].y);
      q4m_mma_bf16(d0[0], d0[1], d0[2], d0[3], wb0[2], wb1[2], wb0[3], wb1[3], x_lo[k].z, x_lo[k].w);
      if constexpr (ALLOW_HI) {
        if (do_hi) {
          q4m_mma_bf16(d1[0], d1[1], d1[2], d1[3], wb0[0], wb1[0], wb0[1], wb1[1], x_hi[k].x, x_hi[k].y);
          q4m_mma_bf16(d1[0], d1[1], d1[2], d1[3], wb0[2], wb1[2], wb0[3], wb1[3], x_hi[k].z, x_hi[k].w);
        }
      }

      const float sc0 = __half2float(__ushort_as_half(smem[cur].sc[g][grp]));
      const float bi0 = __half2float(__ushort_as_half(smem[cur].bi[g][grp]));
      const float sc1 = __half2float(__ushort_as_half(smem[cur].sc[g + 8][grp]));
      const float bi1 = __half2float(__ushort_as_half(smem[cur].bi[g + 8][grp]));

      tot_lo[0] = fmaf(sc0, d0[0], fmaf(bi0, xs_lo_c0[k], tot_lo[0]));
      tot_lo[1] = fmaf(sc0, d0[1], fmaf(bi0, xs_lo_c1[k], tot_lo[1]));
      tot_lo[2] = fmaf(sc1, d0[2], fmaf(bi1, xs_lo_c0[k], tot_lo[2]));
      tot_lo[3] = fmaf(sc1, d0[3], fmaf(bi1, xs_lo_c1[k], tot_lo[3]));
      if constexpr (ALLOW_HI) {
        if (do_hi) {
          tot_hi[0] = fmaf(sc0, d1[0], fmaf(bi0, xs_hi_c0[k], tot_hi[0]));
          tot_hi[1] = fmaf(sc0, d1[1], fmaf(bi0, xs_hi_c1[k], tot_hi[1]));
          tot_hi[2] = fmaf(sc1, d1[2], fmaf(bi1, xs_hi_c0[k], tot_hi[2]));
          tot_hi[3] = fmaf(sc1, d1[3], fmaf(bi1, xs_hi_c1[k], tot_hi[3]));
        }
      }
    }

    s_red[warp][2 * t][g] = tot_lo[0];
    s_red[warp][2 * t + 1][g] = tot_lo[1];
    s_red[warp][2 * t][g + 8] = tot_lo[2];
    s_red[warp][2 * t + 1][g + 8] = tot_lo[3];
    if constexpr (ALLOW_HI) {
      if (do_hi) {
        s_red[warp][8 + 2 * t][g] = tot_hi[0];
        s_red[warp][8 + 2 * t + 1][g] = tot_hi[1];
        s_red[warp][8 + 2 * t][g + 8] = tot_hi[2];
        s_red[warp][8 + 2 * t + 1][g + 8] = tot_hi[3];
      }
    }
    __syncthreads();

    const int total_out = cnt * 16;
    for (int idx = tid; idx < total_out; idx += 160) {
      const int c = idx >> 4;
      const int r = idx & 15;
      float val = 0.0f;
#pragma unroll
      for (int ww = 0; ww < 5; ++ww) val += s_red[ww][c][r];
      const int pair = s_pair[c];
      if constexpr (ATOMIC_COMBINE) {
        atomicAdd(reinterpret_cast<float*>(out_ptr) +
                  static_cast<size_t>(pair / top_k) * N + row0 + r, val * s_score[c]);
      } else {
        reinterpret_cast<nv_bfloat16*>(out_ptr)[static_cast<size_t>(pair) * N + row0 + r] =
            __float2bfloat16(val);
      }
    }
    __syncthreads();
  }
}
)cuda"
R"cuda(
// ---------------------------------------------------------------------------
// 4. Deterministic L2-resident Weighted Combine (+ optional Shared Expert Add)
//    Grid: dim3((N/8 + 255)/256, M), Block: 256.
// ---------------------------------------------------------------------------
template <bool ADD_SHARED>
__global__ __launch_bounds__(256) void qwen4_moe_combine_few(
    const nv_bfloat16* __restrict__ down_pairs,
    const float* __restrict__ scores_f32,
    const nv_bfloat16* __restrict__ shared_out,
    const nv_bfloat16* __restrict__ shared_gate,
    nv_bfloat16* __restrict__ out,
    int M,
    int N,
    int top_k) {
  const int tok = blockIdx.y;
  const int col8 = blockIdx.x * blockDim.x + threadIdx.x;
  const int n8 = N >> 3;
  if (tok >= M || col8 >= n8) return;

  float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  const uint4* dp_u4 = reinterpret_cast<const uint4*>(down_pairs);
  const int base_pair = tok * top_k;
#pragma unroll 1
  for (int slot = 0; slot < top_k; ++slot) {
    const float sc = scores_f32[base_pair + slot];
    const uint4 raw = dp_u4[static_cast<size_t>(base_pair + slot) * n8 + col8];
    const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      float2 f = __bfloat1622float2(p[k]);
      acc[2 * k]     = fmaf(f.x, sc, acc[2 * k]);
      acc[2 * k + 1] = fmaf(f.y, sc, acc[2 * k + 1]);
    }
  }
  if constexpr (ADD_SHARED) {
    const float sg = q4m_sigmoid(__bfloat162float(shared_gate[tok]));
    const uint4 sraw = reinterpret_cast<const uint4*>(shared_out)[static_cast<size_t>(tok) * n8 + col8];
    const __nv_bfloat162* sp = reinterpret_cast<const __nv_bfloat162*>(&sraw);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      float2 f = __bfloat1622float2(sp[k]);
      acc[2 * k]     = fmaf(f.x, sg, acc[2 * k]);
      acc[2 * k + 1] = fmaf(f.y, sg, acc[2 * k + 1]);
    }
  }
  uint4 out_v;
  __nv_bfloat162* op = reinterpret_cast<__nv_bfloat162*>(&out_v);
#pragma unroll
  for (int k = 0; k < 4; ++k) op[k] = __floats2bfloat162_rn(acc[2 * k], acc[2 * k + 1]);
  reinterpret_cast<uint4*>(out)[static_cast<size_t>(tok) * n8 + col8] = out_v;
}
)cuda"
R"cuda(
// ---------------------------------------------------------------------------
// 5. Fused Shared Expert Q8_0 gate + up + SwiGLU (M in 1..=16, N=640, K=2560)
//    Grid: dim3(N / 16), Block: dim3(32, 4).
// ---------------------------------------------------------------------------
template <bool ALLOW_HI>
__global__ __launch_bounds__(128, 4) void qwen4_shared_gate_up_swiglu_few(
    const unsigned int* __restrict__ wg,
    const unsigned short* __restrict__ sg,
    const unsigned short* __restrict__ bg,
    const unsigned int* __restrict__ wu,
    const unsigned short* __restrict__ su,
    const unsigned short* __restrict__ bu,
    const nv_bfloat16* __restrict__ x,
    nv_bfloat16* __restrict__ shared_mid,
    int M,
    int N,
    int K) {
  const int row0 = blockIdx.x * 16;
  if (row0 >= N) return;

  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const int tid = warp * 32 + lane;
  const int g = lane >> 2;
  const int t = lane & 3;
  const bool do_hi = ALLOW_HI && (M > 8);

  const int groups = K >> 5;
  const int num_chunks = groups >> 3;
  const size_t w_stride_u4 = static_cast<size_t>(K) >> 4;
  const size_t x_stride_u4 = static_cast<size_t>(K) >> 3;

  struct Stage {
    alignas(16) uint4 w[16][16];
    alignas(16) unsigned short sc[16][8];
    alignas(16) unsigned short bi[16][8];
  };
  union SharedBuf {
    Stage pipe[2];
    struct {
      float gate[4][16][17];
      float up[4][16][17];
    } red;
  };
  __shared__ SharedBuf smem;

  const int ld_r = tid >> 3;
  const int ld_c0 = (tid & 7) * 2;
  auto load_chunk = [&](int st, const unsigned int* wb, const unsigned short* sb,
                        const unsigned short* bb, int chunk) {
    const uint4* wp = reinterpret_cast<const uint4*>(wb) +
        static_cast<size_t>(row0 + ld_r) * w_stride_u4 + chunk * 16 + ld_c0;
    q4m_cp_async16(&smem.pipe[st].w[ld_r][ld_c0 ^ ld_r], wp);
    q4m_cp_async16(&smem.pipe[st].w[ld_r][(ld_c0 + 1) ^ ld_r], wp + 1);
    const int sb_r = tid & 15;
    const int sb_h = tid >> 4;
    if (sb_h < 2) {
      const unsigned short* sp = (sb_h == 0 ? sb : bb) +
          static_cast<size_t>(row0 + sb_r) * groups + chunk * 8;
      q4m_cp_async16(sb_h == 0 ? smem.pipe[st].sc[sb_r] : smem.pipe[st].bi[sb_r], sp);
    }
    q4m_cp_commit();
  };

  float g_tot_lo[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float g_tot_hi[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float u_tot_lo[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float u_tot_hi[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  load_chunk(0, wg, sg, bg, 0);
  const int total_steps = num_chunks * 2;
  for (int step = 0; step < total_steps; ++step) {
    const int cur = step & 1;
    const int nxt = cur ^ 1;
    if (step + 1 < total_steps) {
      const int ns = step + 1;
      const bool n_up = (ns >= num_chunks);
      load_chunk(nxt, n_up ? wu : wg, n_up ? su : sg, n_up ? bu : bg,
                 n_up ? (ns - num_chunks) : ns);
      q4m_cp_wait<1>();
    } else {
      q4m_cp_wait<0>();
    }
    __syncthreads();

    const bool is_up = (step >= num_chunks);
    const int chunk = is_up ? (step - num_chunks) : step;
    float* t_lo = is_up ? u_tot_lo : g_tot_lo;
    float* t_hi = is_up ? u_tot_hi : g_tot_hi;

#pragma unroll
    for (int kg = 0; kg < 2; ++kg) {
      const int lg = warp * 2 + kg;
      const int grp = chunk * 8 + lg;
      const int col_u4 = lg * 2 + (t >> 1);
      const int half_u2 = t & 1;
      const uint2 w0 = reinterpret_cast<const uint2*>(
          &smem.pipe[cur].w[g][col_u4 ^ g])[half_u2];
      const uint2 w1 = reinterpret_cast<const uint2*>(
          &smem.pipe[cur].w[g + 8][col_u4 ^ (g + 8)])[half_u2];

      uint4 xv0 = (g < M) ? reinterpret_cast<const uint4*>(x)[
          static_cast<size_t>(g) * x_stride_u4 + grp * 4 + t] : make_uint4(0u, 0u, 0u, 0u);
      float my_s0 = (g < M) ? q4m_sum_bf16x8(xv0) : 0.0f;
      my_s0 += __shfl_xor_sync(0xffffffffu, my_s0, 1);
      my_s0 += __shfl_xor_sync(0xffffffffu, my_s0, 2);
      const float xs00 = __shfl_sync(0xffffffffu, my_s0, (2 * t) * 4);
      const float xs01 = __shfl_sync(0xffffffffu, my_s0, (2 * t + 1) * 4);

      uint4 xv1 = make_uint4(0u, 0u, 0u, 0u);
      float xs10 = 0.0f, xs11 = 0.0f;
      if constexpr (ALLOW_HI) {
        if (do_hi) {
          if (g + 8 < M) {
            xv1 = reinterpret_cast<const uint4*>(x)[
                static_cast<size_t>(g + 8) * x_stride_u4 + grp * 4 + t];
          }
          float my_s1 = (g + 8 < M) ? q4m_sum_bf16x8(xv1) : 0.0f;
          my_s1 += __shfl_xor_sync(0xffffffffu, my_s1, 1);
          my_s1 += __shfl_xor_sync(0xffffffffu, my_s1, 2);
          xs10 = __shfl_sync(0xffffffffu, my_s1, (2 * t) * 4);
          xs11 = __shfl_sync(0xffffffffu, my_s1, (2 * t + 1) * 4);
        }
      }

      unsigned int wb0[4], wb1[4];
      wb0[0] = q4m_pack_bf16x2(float(w0.x & 0xffu), float((w0.x >> 8) & 0xffu));
      wb0[1] = q4m_pack_bf16x2(float((w0.x >> 16) & 0xffu), float(w0.x >> 24));
      wb0[2] = q4m_pack_bf16x2(float(w0.y & 0xffu), float((w0.y >> 8) & 0xffu));
      wb0[3] = q4m_pack_bf16x2(float((w0.y >> 16) & 0xffu), float(w0.y >> 24));
      wb1[0] = q4m_pack_bf16x2(float(w1.x & 0xffu), float((w1.x >> 8) & 0xffu));
      wb1[1] = q4m_pack_bf16x2(float((w1.x >> 16) & 0xffu), float(w1.x >> 24));
      wb1[2] = q4m_pack_bf16x2(float(w1.y & 0xffu), float((w1.y >> 8) & 0xffu));
      wb1[3] = q4m_pack_bf16x2(float((w1.y >> 16) & 0xffu), float(w1.y >> 24));

      float d0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      float d1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      q4m_mma_bf16(d0[0], d0[1], d0[2], d0[3], wb0[0], wb1[0], wb0[1], wb1[1], xv0.x, xv0.y);
      q4m_mma_bf16(d0[0], d0[1], d0[2], d0[3], wb0[2], wb1[2], wb0[3], wb1[3], xv0.z, xv0.w);
      if constexpr (ALLOW_HI) {
        if (do_hi) {
          q4m_mma_bf16(d1[0], d1[1], d1[2], d1[3], wb0[0], wb1[0], wb0[1], wb1[1], xv1.x, xv1.y);
          q4m_mma_bf16(d1[0], d1[1], d1[2], d1[3], wb0[2], wb1[2], wb0[3], wb1[3], xv1.z, xv1.w);
        }
      }

      const float sc0 = __half2float(__ushort_as_half(smem.pipe[cur].sc[g][lg]));
      const float bi0 = __half2float(__ushort_as_half(smem.pipe[cur].bi[g][lg]));
      const float sc1 = __half2float(__ushort_as_half(smem.pipe[cur].sc[g + 8][lg]));
      const float bi1 = __half2float(__ushort_as_half(smem.pipe[cur].bi[g + 8][lg]));

      t_lo[0] = fmaf(sc0, d0[0], fmaf(bi0, xs00, t_lo[0]));
      t_lo[1] = fmaf(sc0, d0[1], fmaf(bi0, xs01, t_lo[1]));
      t_lo[2] = fmaf(sc1, d0[2], fmaf(bi1, xs00, t_lo[2]));
      t_lo[3] = fmaf(sc1, d0[3], fmaf(bi1, xs01, t_lo[3]));
      if constexpr (ALLOW_HI) {
        if (do_hi) {
          t_hi[0] = fmaf(sc0, d1[0], fmaf(bi0, xs10, t_hi[0]));
          t_hi[1] = fmaf(sc0, d1[1], fmaf(bi0, xs11, t_hi[1]));
          t_hi[2] = fmaf(sc1, d1[2], fmaf(bi1, xs10, t_hi[2]));
          t_hi[3] = fmaf(sc1, d1[3], fmaf(bi1, xs11, t_hi[3]));
        }
      }
    }
    __syncthreads();
  }

  smem.red.gate[warp][2 * t][g] = g_tot_lo[0];
  smem.red.gate[warp][2 * t + 1][g] = g_tot_lo[1];
  smem.red.gate[warp][2 * t][g + 8] = g_tot_lo[2];
  smem.red.gate[warp][2 * t + 1][g + 8] = g_tot_lo[3];
  smem.red.up[warp][2 * t][g] = u_tot_lo[0];
  smem.red.up[warp][2 * t + 1][g] = u_tot_lo[1];
  smem.red.up[warp][2 * t][g + 8] = u_tot_lo[2];
  smem.red.up[warp][2 * t + 1][g + 8] = u_tot_lo[3];
  if constexpr (ALLOW_HI) {
    if (do_hi) {
      smem.red.gate[warp][8 + 2 * t][g] = g_tot_hi[0];
      smem.red.gate[warp][8 + 2 * t + 1][g] = g_tot_hi[1];
      smem.red.gate[warp][8 + 2 * t][g + 8] = g_tot_hi[2];
      smem.red.gate[warp][8 + 2 * t + 1][g + 8] = g_tot_hi[3];
      smem.red.up[warp][8 + 2 * t][g] = u_tot_hi[0];
      smem.red.up[warp][8 + 2 * t + 1][g] = u_tot_hi[1];
      smem.red.up[warp][8 + 2 * t][g + 8] = u_tot_hi[2];
      smem.red.up[warp][8 + 2 * t + 1][g + 8] = u_tot_hi[3];
    }
  }
  __syncthreads();

  for (int idx = tid; idx < M * 16; idx += 128) {
    const int m = idx >> 4;
    const int r = idx & 15;
    float gv = 0.0f, uv = 0.0f;
#pragma unroll
    for (int ww = 0; ww < 4; ++ww) {
      gv += smem.red.gate[ww][m][r];
      uv += smem.red.up[ww][m][r];
    }
    shared_mid[static_cast<size_t>(m) * N + row0 + r] = __float2bfloat16(q4m_silu(gv) * uv);
  }
}
)cuda"
R"cuda(
// ---------------------------------------------------------------------------
// 6. Fused Shared Q8_0 Down + Routed Weighted Combine (1 launch!)
//    Grid: dim3(N / (16 * TILES_PER_CTA)), Block: dim3(32, 5).
// ---------------------------------------------------------------------------
template <int TILES_PER_CTA, bool ALLOW_HI>
__global__ __launch_bounds__(160, ALLOW_HI ? 2 : 3) void qwen4_shared_down_combine_few(
    const unsigned int* __restrict__ swd,
    const unsigned short* __restrict__ ssd,
    const unsigned short* __restrict__ sbd,
    const nv_bfloat16* __restrict__ shared_mid,
    const nv_bfloat16* __restrict__ shared_gate,
    const nv_bfloat16* __restrict__ down_pairs,
    const float* __restrict__ scores_f32,
    nv_bfloat16* __restrict__ out,
    int M,
    int N,
    int K,
    int top_k) {
  const int base_row = blockIdx.x * (16 * TILES_PER_CTA);
  if (base_row >= N) return;

  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const int tid = warp * 32 + lane;
  const int g = lane >> 2;
  const int t = lane & 3;
  const bool do_hi = ALLOW_HI && (M > 8);

  constexpr int GROUPS = 20;
  constexpr int ROW_U4 = 40;
  constexpr int PAD_U4 = 41;
  constexpr int TILE_SB_U4 = 40;

  struct StageSmem {
    alignas(16) uint4 w[16][PAD_U4];
    alignas(16) unsigned short sc[16][20];
    alignas(16) unsigned short bi[16][20];
  };
  __shared__ StageSmem smem[2];
  __shared__ float s_red[5][16][17];
  __shared__ float s_sg[16];
  if (tid < M) {
    s_sg[tid] = q4m_sigmoid(__bfloat162float(shared_gate[tid]));
  }

  const uint4* wd_u4 = reinterpret_cast<const uint4*>(swd);
  const uint4* sd_u4 = reinterpret_cast<const uint4*>(ssd);
  const uint4* bd_u4 = reinterpret_cast<const uint4*>(sbd);

  auto load_tile = [&](int st, int row0) {
    const uint4* w_src = wd_u4 + static_cast<size_t>(row0) * ROW_U4;
#pragma unroll
    for (int iter = 0; iter < 4; ++iter) {
      const int i = tid + iter * 160;
      const int r = i / 40;
      const int c = i - r * 40;
      q4m_cp_async16(&smem[st].w[r][c], w_src + i);
    }
    const uint4* s_src = sd_u4 + (static_cast<size_t>(row0) * GROUPS) / 8;
    const uint4* b_src = bd_u4 + (static_cast<size_t>(row0) * GROUPS) / 8;
    if (tid < TILE_SB_U4) {
      q4m_cp_async16(reinterpret_cast<uint4*>(smem[st].sc) + tid, s_src + tid);
    } else if (tid < 2 * TILE_SB_U4) {
      const int k = tid - TILE_SB_U4;
      q4m_cp_async16(reinterpret_cast<uint4*>(smem[st].bi) + k, b_src + k);
    }
    q4m_cp_commit();
  };

  load_tile(0, base_row);
  __syncthreads();

  uint4 x_lo[4];
  float xs_lo_c0[4], xs_lo_c1[4];
  uint4 x_hi[ALLOW_HI ? 4 : 1];
  float xs_hi_c0[ALLOW_HI ? 4 : 1], xs_hi_c1[ALLOW_HI ? 4 : 1];
  const size_t mid_stride_u4 = static_cast<size_t>(K) >> 3;

#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const int grp = warp * 4 + k;
    x_lo[k] = (g < M) ? reinterpret_cast<const uint4*>(shared_mid)[
        static_cast<size_t>(g) * mid_stride_u4 + grp * 4 + t] : make_uint4(0u, 0u, 0u, 0u);
    float my_s = (g < M) ? q4m_sum_bf16x8(x_lo[k]) : 0.0f;
    my_s += __shfl_xor_sync(0xffffffffu, my_s, 1);
    my_s += __shfl_xor_sync(0xffffffffu, my_s, 2);
    xs_lo_c0[k] = __shfl_sync(0xffffffffu, my_s, (2 * t) * 4);
    xs_lo_c1[k] = __shfl_sync(0xffffffffu, my_s, (2 * t + 1) * 4);
    if constexpr (ALLOW_HI) {
      if (do_hi) {
        x_hi[k] = (g + 8 < M) ? reinterpret_cast<const uint4*>(shared_mid)[
            static_cast<size_t>(g + 8) * mid_stride_u4 + grp * 4 + t] : make_uint4(0u, 0u, 0u, 0u);
        float my_sh = (g + 8 < M) ? q4m_sum_bf16x8(x_hi[k]) : 0.0f;
        my_sh += __shfl_xor_sync(0xffffffffu, my_sh, 1);
        my_sh += __shfl_xor_sync(0xffffffffu, my_sh, 2);
        xs_hi_c0[k] = __shfl_sync(0xffffffffu, my_sh, (2 * t) * 4);
        xs_hi_c1[k] = __shfl_sync(0xffffffffu, my_sh, (2 * t + 1) * 4);
      }
    }
  }

#pragma unroll 1
  for (int ti = 0; ti < TILES_PER_CTA; ++ti) {
    const int row0 = base_row + ti * 16;
    if (row0 >= N) break;
    const int cur = ti & 1;
    const int nxt = cur ^ 1;
    const bool has_next = (ti + 1 < TILES_PER_CTA) && (row0 + 16 < N);
    if (has_next) {
      load_tile(nxt, row0 + 16);
      q4m_cp_wait<1>();
    } else {
      q4m_cp_wait<0>();
    }
    __syncthreads();

    float tot_lo[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float tot_hi[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const unsigned int* w_r0 = reinterpret_cast<const unsigned int*>(smem[cur].w[g]);
    const unsigned int* w_r1 = reinterpret_cast<const unsigned int*>(smem[cur].w[g + 8]);

#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const int grp = warp * 4 + k;
      unsigned int wb0[4], wb1[4];
      q4m_unpack8_affine_bf16<8>(w_r0 + grp * 8, t, wb0);
      q4m_unpack8_affine_bf16<8>(w_r1 + grp * 8, t, wb1);

      float d0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      float d1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      q4m_mma_bf16(d0[0], d0[1], d0[2], d0[3], wb0[0], wb1[0], wb0[1], wb1[1], x_lo[k].x, x_lo[k].y);
      q4m_mma_bf16(d0[0], d0[1], d0[2], d0[3], wb0[2], wb1[2], wb0[3], wb1[3], x_lo[k].z, x_lo[k].w);
      if constexpr (ALLOW_HI) {
        if (do_hi) {
          q4m_mma_bf16(d1[0], d1[1], d1[2], d1[3], wb0[0], wb1[0], wb0[1], wb1[1], x_hi[k].x, x_hi[k].y);
          q4m_mma_bf16(d1[0], d1[1], d1[2], d1[3], wb0[2], wb1[2], wb0[3], wb1[3], x_hi[k].z, x_hi[k].w);
        }
      }
      const float sc0 = __half2float(__ushort_as_half(smem[cur].sc[g][grp]));
      const float bi0 = __half2float(__ushort_as_half(smem[cur].bi[g][grp]));
      const float sc1 = __half2float(__ushort_as_half(smem[cur].sc[g + 8][grp]));
      const float bi1 = __half2float(__ushort_as_half(smem[cur].bi[g + 8][grp]));

      tot_lo[0] = fmaf(sc0, d0[0], fmaf(bi0, xs_lo_c0[k], tot_lo[0]));
      tot_lo[1] = fmaf(sc0, d0[1], fmaf(bi0, xs_lo_c1[k], tot_lo[1]));
      tot_lo[2] = fmaf(sc1, d0[2], fmaf(bi1, xs_lo_c0[k], tot_lo[2]));
      tot_lo[3] = fmaf(sc1, d0[3], fmaf(bi1, xs_lo_c1[k], tot_lo[3]));
      if constexpr (ALLOW_HI) {
        if (do_hi) {
          tot_hi[0] = fmaf(sc0, d1[0], fmaf(bi0, xs_hi_c0[k], tot_hi[0]));
          tot_hi[1] = fmaf(sc0, d1[1], fmaf(bi0, xs_hi_c1[k], tot_hi[1]));
          tot_hi[2] = fmaf(sc1, d1[2], fmaf(bi1, xs_hi_c0[k], tot_hi[2]));
          tot_hi[3] = fmaf(sc1, d1[3], fmaf(bi1, xs_hi_c1[k], tot_hi[3]));
        }
      }
    }

    s_red[warp][2 * t][g] = tot_lo[0];
    s_red[warp][2 * t + 1][g] = tot_lo[1];
    s_red[warp][2 * t][g + 8] = tot_lo[2];
    s_red[warp][2 * t + 1][g + 8] = tot_lo[3];
    if constexpr (ALLOW_HI) {
      if (do_hi) {
        s_red[warp][8 + 2 * t][g] = tot_hi[0];
        s_red[warp][8 + 2 * t + 1][g] = tot_hi[1];
        s_red[warp][8 + 2 * t][g + 8] = tot_hi[2];
        s_red[warp][8 + 2 * t + 1][g + 8] = tot_hi[3];
      }
    }
    __syncthreads();

    for (int idx = tid; idx < M * 16; idx += 160) {
      const int m = idx >> 4;
      const int r = idx & 15;
      float sv = 0.0f;
#pragma unroll
      for (int ww = 0; ww < 5; ++ww) sv += s_red[ww][m][r];
      float acc = sv * s_sg[m];
      const int base_pair = m * top_k;
      const int col = row0 + r;
#pragma unroll
      for (int slot = 0; slot < 10; ++slot) {
        if (slot < top_k) {
          float dv = __bfloat162float(down_pairs[static_cast<size_t>(base_pair + slot) * N + col]);
          acc = fmaf(dv, scores_f32[base_pair + slot], acc);
        }
      }
      out[static_cast<size_t>(m) * N + col] = __float2bfloat16(acc);
    }
    __syncthreads();
  }
}
)cuda"
R"cuda(
// ---------------------------------------------------------------------------
// 7. Ultra-High-Occupancy M=1 Streaming Kernels (0 MMA, ld.global.cs.v4.u32)
//    Replaces 28KB-SMEM m16n8k16 MMA (87.5% idle on M=1) and 40-CTA shared
//    kernels with 48-warp/SM streaming kernels saturating LPDDR5X!
// ---------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(256) void qwen4_moe_quantize_x_m1(
    const nv_bfloat16* __restrict__ x,
    signed char* __restrict__ xq,
    float* __restrict__ x_scales,
    float* __restrict__ x_sums,
    int K) {
  const int groups_per_row = K >> 5;
  const int gtid = blockIdx.x * blockDim.x + threadIdx.x;
  const int grp = gtid >> 3;
  const int sub = gtid & 7;
  if (grp >= groups_per_row) return;

  const uint2 raw = *reinterpret_cast<const uint2*>(x + grp * 32 + sub * 4);
  const __nv_bfloat162* pairs = reinterpret_cast<const __nv_bfloat162*>(&raw);
  const float2 f0 = __bfloat1622float2(pairs[0]);
  const float2 f1 = __bfloat1622float2(pairs[1]);
  float v[4] = {f0.x, f0.y, f1.x, f1.y};
  float amax = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
#pragma unroll
  for (int off = 1; off < 8; off <<= 1) {
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off, 8));
  }
  const float inv = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
  int q[4];
  int qsum = 0;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    q[j] = (amax > 0.0f) ? static_cast<int>(rintf(v[j] * inv)) : 0;
    qsum += q[j];
  }
#pragma unroll
  for (int off = 1; off < 8; off <<= 1) {
    qsum += __shfl_xor_sync(0xffffffffu, qsum, off, 8);
  }
  *reinterpret_cast<unsigned int*>(xq + grp * 32 + sub * 4) =
      (static_cast<unsigned int>(q[0] & 0xff)) |
      (static_cast<unsigned int>(q[1] & 0xff) << 8) |
      (static_cast<unsigned int>(q[2] & 0xff) << 16) |
      (static_cast<unsigned int>(q[3] & 0xff) << 24);
  if (sub == 0) {
    const float d = (amax > 0.0f) ? (amax * (1.0f / 127.0f)) : 0.0f;
    x_scales[grp] = d;
    x_sums[grp] = d * static_cast<float>(qsum);
  }
}

__device__ __forceinline__ float q4m_dot_q8_u4_sx(
    const uint4& wv, const float* __restrict__ sx, float& out_xs) {
  float d = 0.0f;
  float s = 0.0f;
  unsigned int uu;
  uu = wv.x;
  d = fmaf(sx[0], static_cast<float>(uu & 0xFFu),
      fmaf(sx[1], static_cast<float>((uu >> 8) & 0xFFu),
      fmaf(sx[2], static_cast<float>((uu >> 16) & 0xFFu),
      fmaf(sx[3], static_cast<float>(uu >> 24), d))));
  s += (sx[0] + sx[1]) + (sx[2] + sx[3]);
  uu = wv.y;
  d = fmaf(sx[4], static_cast<float>(uu & 0xFFu),
      fmaf(sx[5], static_cast<float>((uu >> 8) & 0xFFu),
      fmaf(sx[6], static_cast<float>((uu >> 16) & 0xFFu),
      fmaf(sx[7], static_cast<float>(uu >> 24), d))));
  s += (sx[4] + sx[5]) + (sx[6] + sx[7]);
  uu = wv.z;
  d = fmaf(sx[8], static_cast<float>(uu & 0xFFu),
      fmaf(sx[9], static_cast<float>((uu >> 8) & 0xFFu),
      fmaf(sx[10], static_cast<float>((uu >> 16) & 0xFFu),
      fmaf(sx[11], static_cast<float>(uu >> 24), d))));
  s += (sx[8] + sx[9]) + (sx[10] + sx[11]);
  uu = wv.w;
  d = fmaf(sx[12], static_cast<float>(uu & 0xFFu),
      fmaf(sx[13], static_cast<float>((uu >> 8) & 0xFFu),
      fmaf(sx[14], static_cast<float>((uu >> 16) & 0xFFu),
      fmaf(sx[15], static_cast<float>(uu >> 24), d))));
  s += (sx[12] + sx[13]) + (sx[14] + sx[15]);
  out_xs = s;
  return d;
}

// M=1 Routed Down Projection (Q5_1 bits=5 or Q8_0 bits=8, N=2560, K=640).
// BITS == 5: Grid dim3(N/16, top_k), Block dim3(32, 4). 4 rows/warp, 9 KB SMEM.
// BITS == 8: Grid dim3(N/4, top_k),  Block dim3(32, 4). 1 row/warp,  2.6 KB SMEM.
template <int BITS>
__global__ __launch_bounds__(128, 6) void qwen4_moe_down_m1(
    const unsigned int* __restrict__ wd,
    const unsigned short* __restrict__ sd,
    const unsigned short* __restrict__ bd,
    const nv_bfloat16* __restrict__ mid,
    const int* __restrict__ expert_ids,
    nv_bfloat16* __restrict__ down_pairs,
    int N,
    int K) {
  const int slot = blockIdx.y;
  const int expert = expert_ids[slot];
  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const int tid = warp * 32 + lane;

  __shared__ float s_x[640];
  if (tid < 80) {
    const uint4 xv = reinterpret_cast<const uint4*>(
        mid + static_cast<size_t>(slot) * 640)[tid];
    const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&xv);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float2 f = __bfloat1622float2(p[j]);
      s_x[tid * 8 + j * 2] = f.x;
      s_x[tid * 8 + j * 2 + 1] = f.y;
    }
  }
  __syncthreads();

  if constexpr (BITS == 5) {
    __shared__ float s_xsum[20];
    if (tid < 20) {
      float s = 0.0f;
      const float* xp = s_x + tid * 32;
#pragma unroll
      for (int e = 0; e < 32; ++e) s += xp[e];
      s_xsum[tid] = s;
    }
    __syncthreads();

    const int row0 = blockIdx.x * 16 + warp * 4;
    if (row0 >= N) return;

    __shared__ uint4 s_w[4][100];
    const size_t base_r = static_cast<size_t>(expert) * N + row0;
    const uint4* w_src = reinterpret_cast<const uint4*>(wd) + base_r * 25;
#pragma unroll
    for (int it = 0; it < 3; ++it) {
      s_w[warp][it * 32 + lane] = q4m_ld_cs_u4(w_src + it * 32 + lane);
    }
    if (lane < 4) {
      s_w[warp][96 + lane] = q4m_ld_cs_u4(w_src + 96 + lane);
    }
    __syncwarp();

#pragma unroll 1
    for (int r = 0; r < 4; ++r) {
      const int row = row0 + r;
      if (row >= N) break;
      float val = 0.0f;
      if (lane < 20) {
        const unsigned int* g_u32 =
            reinterpret_cast<const unsigned int*>(&s_w[warp][r * 25]) + lane * 5;
        const float* xp = s_x + lane * 32;
        float d = 0.0f;
        const unsigned int u0 = g_u32[0];
        const unsigned int u1 = g_u32[1];
        const unsigned int u2 = g_u32[2];
        const unsigned int u3 = g_u32[3];
        const unsigned int u4 = g_u32[4];
        d = fmaf(xp[0],  float(u0 & 31u), d);
        d = fmaf(xp[1],  float((u0 >> 5) & 31u), d);
        d = fmaf(xp[2],  float((u0 >> 10) & 31u), d);
        d = fmaf(xp[3],  float((u0 >> 15) & 31u), d);
        d = fmaf(xp[4],  float((u0 >> 20) & 31u), d);
        d = fmaf(xp[5],  float((u0 >> 25) & 31u), d);
        d = fmaf(xp[6],  float(((u0 >> 30) | ((u1 & 7u) << 2)) & 31u), d);
        d = fmaf(xp[7],  float((u1 >> 3) & 31u), d);
        d = fmaf(xp[8],  float((u1 >> 8) & 31u), d);
        d = fmaf(xp[9],  float((u1 >> 13) & 31u), d);
        d = fmaf(xp[10], float((u1 >> 18) & 31u), d);
        d = fmaf(xp[11], float((u1 >> 23) & 31u), d);
        d = fmaf(xp[12], float(((u1 >> 28) | ((u2 & 1u) << 4)) & 31u), d);
        d = fmaf(xp[13], float((u2 >> 1) & 31u), d);
        d = fmaf(xp[14], float((u2 >> 6) & 31u), d);
        d = fmaf(xp[15], float((u2 >> 11) & 31u), d);
        d = fmaf(xp[16], float((u2 >> 16) & 31u), d);
        d = fmaf(xp[17], float((u2 >> 21) & 31u), d);
        d = fmaf(xp[18], float((u2 >> 26) & 31u), d);
        d = fmaf(xp[19], float(((u2 >> 31) | ((u3 & 15u) << 1)) & 31u), d);
        d = fmaf(xp[20], float((u3 >> 4) & 31u), d);
        d = fmaf(xp[21], float((u3 >> 9) & 31u), d);
        d = fmaf(xp[22], float((u3 >> 14) & 31u), d);
        d = fmaf(xp[23], float((u3 >> 19) & 31u), d);
        d = fmaf(xp[24], float((u3 >> 24) & 31u), d);
        d = fmaf(xp[25], float(((u3 >> 29) | ((u4 & 3u) << 3)) & 31u), d);
        d = fmaf(xp[26], float((u4 >> 2) & 31u), d);
        d = fmaf(xp[27], float((u4 >> 7) & 31u), d);
        d = fmaf(xp[28], float((u4 >> 12) & 31u), d);
        d = fmaf(xp[29], float((u4 >> 17) & 31u), d);
        d = fmaf(xp[30], float((u4 >> 22) & 31u), d);
        d = fmaf(xp[31], float((u4 >> 27) & 31u), d);
        const size_t sb_idx = (static_cast<size_t>(expert) * N + row) * 20 + lane;
        const float sc = __half2float(__ushort_as_half(sd[sb_idx]));
        const float bi = __half2float(__ushort_as_half(bd[sb_idx]));
        val = fmaf(sc, d, bi * s_xsum[lane]);
      }
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) {
        val += __shfl_xor_sync(0xffffffffu, val, off);
      }
      if (lane == 0) {
        down_pairs[static_cast<size_t>(slot) * N + row] = __float2bfloat16(val);
      }
    }
  } else {
    const int row = blockIdx.x * 4 + warp;
    if (row >= N) return;
    const size_t r_idx = static_cast<size_t>(expert) * N + row;
    const uint4* w_row = reinterpret_cast<const uint4*>(wd) + r_idx * 40;
    const unsigned short* sc_row = sd + r_idx * 20;
    const unsigned short* bi_row = bd + r_idx * 20;

    float val = 0.0f;
    {
      const uint4 wv = q4m_ld_cs_u4(w_row + lane);
      float xs = 0.0f;
      const float d = q4m_dot_q8_u4_sx(wv, s_x + lane * 16, xs);
      const int g = lane >> 1;
      const float sc = __half2float(__ushort_as_half(sc_row[g]));
      const float bi = __half2float(__ushort_as_half(bi_row[g]));
      val = fmaf(sc, d, bi * xs);
    }
    if (lane < 8) {
      const int u = 32 + lane;
      const uint4 wv = q4m_ld_cs_u4(w_row + u);
      float xs = 0.0f;
      const float d = q4m_dot_q8_u4_sx(wv, s_x + u * 16, xs);
      const int g = u >> 1;
      const float sc = __half2float(__ushort_as_half(sc_row[g]));
      const float bi = __half2float(__ushort_as_half(bi_row[g]));
      val = fmaf(sc, d, fmaf(bi, xs, val));
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      val += __shfl_xor_sync(0xffffffffu, val, off);
    }
    if (lane == 0) {
      down_pairs[static_cast<size_t>(slot) * N + row] = __float2bfloat16(val);
    }
  }
}
)cuda"
R"cuda(
// M=1 Shared Expert Q8_0 gate + up + SwiGLU (N=640, K=2560, 0 SMEM).
// Grid: dim3(N), Block: dim3(32, 4). 640 CTAs saturating all SMs!
extern "C" __global__ __launch_bounds__(128, 6) void qwen4_shared_gate_up_swiglu_m1(
    const unsigned int* __restrict__ wg,
    const unsigned short* __restrict__ sg,
    const unsigned short* __restrict__ bg,
    const unsigned int* __restrict__ wu,
    const unsigned short* __restrict__ su,
    const unsigned short* __restrict__ bu,
    const nv_bfloat16* __restrict__ x,
    nv_bfloat16* __restrict__ shared_mid,
    int N,
    int K) {
  const int row = blockIdx.x;
  if (row >= N) return;
  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const int tid = warp * 32 + lane;
  const int num_u4 = K >> 4;  // 160
  const int groups = K >> 5;  // 80

  const uint4* wg_row = reinterpret_cast<const uint4*>(wg) + static_cast<size_t>(row) * num_u4;
  const uint4* wu_row = reinterpret_cast<const uint4*>(wu) + static_cast<size_t>(row) * num_u4;
  const unsigned short* sg_row = sg + static_cast<size_t>(row) * groups;
  const unsigned short* bg_row = bg + static_cast<size_t>(row) * groups;
  const unsigned short* su_row = su + static_cast<size_t>(row) * groups;
  const unsigned short* bu_row = bu + static_cast<size_t>(row) * groups;
  const uint4* x_u4 = reinterpret_cast<const uint4*>(x);

  float g_acc = 0.0f;
  float u_acc = 0.0f;

  auto accum_u = [&](int u) {
    const uint4 gw = q4m_ld_cs_u4(wg_row + u);
    const uint4 uw = q4m_ld_cs_u4(wu_row + u);
    const uint4 x0 = x_u4[u * 2];
    const uint4 x1 = x_u4[u * 2 + 1];
    const __nv_bfloat162* p0 = reinterpret_cast<const __nv_bfloat162*>(&x0);
    const __nv_bfloat162* p1 = reinterpret_cast<const __nv_bfloat162*>(&x1);
    float xf[16];
    float xs = 0.0f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float2 f = __bfloat1622float2(p0[j]);
      xf[2 * j] = f.x;
      xf[2 * j + 1] = f.y;
      xs += f.x + f.y;
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float2 f = __bfloat1622float2(p1[j]);
      xf[8 + 2 * j] = f.x;
      xf[8 + 2 * j + 1] = f.y;
      xs += f.x + f.y;
    }
    float dummy_xs = 0.0f;
    const float gd = q4m_dot_q8_u4_sx(gw, xf, dummy_xs);
    const float ud = q4m_dot_q8_u4_sx(uw, xf, dummy_xs);
    const int grp = u >> 1;
    const float gsc = __half2float(__ushort_as_half(sg_row[grp]));
    const float gbi = __half2float(__ushort_as_half(bg_row[grp]));
    const float usc = __half2float(__ushort_as_half(su_row[grp]));
    const float ubi = __half2float(__ushort_as_half(bu_row[grp]));
    g_acc = fmaf(gsc, gd, fmaf(gbi, xs, g_acc));
    u_acc = fmaf(usc, ud, fmaf(ubi, xs, u_acc));
  };

  if (tid < num_u4) accum_u(tid);
  if (tid + 128 < num_u4) accum_u(tid + 128);

#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    g_acc += __shfl_xor_sync(0xffffffffu, g_acc, off);
    u_acc += __shfl_xor_sync(0xffffffffu, u_acc, off);
  }
  __shared__ float s_g[4];
  __shared__ float s_u[4];
  if (lane == 0) {
    s_g[warp] = g_acc;
    s_u[warp] = u_acc;
  }
  __syncthreads();
  if (tid == 0) {
    const float gv = (s_g[0] + s_g[1]) + (s_g[2] + s_g[3]);
    const float uv = (s_u[0] + s_u[1]) + (s_u[2] + s_u[3]);
    shared_mid[row] = __float2bfloat16(q4m_silu(gv) * uv);
  }
}

// M=1 Shared Q8_0 Down + Routed Weighted Combine (N=2560, K=640).
// Grid: dim3(N/4), Block: dim3(32, 4). 640 CTAs saturating all SMs!
extern "C" __global__ __launch_bounds__(128, 6) void qwen4_shared_down_combine_m1(
    const unsigned int* __restrict__ swd,
    const unsigned short* __restrict__ ssd,
    const unsigned short* __restrict__ sbd,
    const nv_bfloat16* __restrict__ shared_mid,
    const nv_bfloat16* __restrict__ shared_gate,
    const nv_bfloat16* __restrict__ down_pairs,
    const float* __restrict__ scores_f32,
    nv_bfloat16* __restrict__ out,
    int N,
    int K,
    int top_k) {
  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const int tid = warp * 32 + lane;
  const int row = blockIdx.x * 4 + warp;

  __shared__ float s_x[640];
  __shared__ float s_sg;
  __shared__ float s_scores[16];
  if (tid < 80) {
    const uint4 xv = reinterpret_cast<const uint4*>(shared_mid)[tid];
    const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&xv);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float2 f = __bfloat1622float2(p[j]);
      s_x[tid * 8 + j * 2] = f.x;
      s_x[tid * 8 + j * 2 + 1] = f.y;
    }
  }
  if (tid == 80) {
    s_sg = q4m_sigmoid(__bfloat162float(shared_gate[0]));
  }
  if (tid < top_k) {
    s_scores[tid] = scores_f32[tid];
  }
  __syncthreads();
  if (row >= N) return;

  const uint4* w_row = reinterpret_cast<const uint4*>(swd) + static_cast<size_t>(row) * 40;
  const unsigned short* sc_row = ssd + static_cast<size_t>(row) * 20;
  const unsigned short* bi_row = sbd + static_cast<size_t>(row) * 20;

  float sv = 0.0f;
  {
    const uint4 wv = q4m_ld_cs_u4(w_row + lane);
    float xs = 0.0f;
    const float d = q4m_dot_q8_u4_sx(wv, s_x + lane * 16, xs);
    const int g = lane >> 1;
    const float sc = __half2float(__ushort_as_half(sc_row[g]));
    const float bi = __half2float(__ushort_as_half(bi_row[g]));
    sv = fmaf(sc, d, bi * xs);
  }
  if (lane < 8) {
    const int u = 32 + lane;
    const uint4 wv = q4m_ld_cs_u4(w_row + u);
    float xs = 0.0f;
    const float d = q4m_dot_q8_u4_sx(wv, s_x + u * 16, xs);
    const int g = u >> 1;
    const float sc = __half2float(__ushort_as_half(sc_row[g]));
    const float bi = __half2float(__ushort_as_half(bi_row[g]));
    sv = fmaf(sc, d, fmaf(bi, xs, sv));
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    sv += __shfl_xor_sync(0xffffffffu, sv, off);
  }
  if (lane == 0) {
    float acc = sv * s_sg;
#pragma unroll
    for (int slot = 0; slot < 10; ++slot) {
      if (slot < top_k) {
        const float dv = __bfloat162float(down_pairs[static_cast<size_t>(slot) * N + row]);
        acc = fmaf(dv, s_scores[slot], acc);
      }
    }
    out[row] = __float2bfloat16(acc);
  }
}
)cuda"
