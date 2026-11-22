R"cuda(
// ============================================================================
// Qwen3.8-Flash-Next (qwen4_exp) Weight-Stationary MoE Prefill Kernels (v2)
// Target: RTX Spark N1X / GB10 (sm_121a, 48 SMs, 24 MB L2, LPDDR5X-9400)
//
// Upgrades over v1 (BM=32):
//   1. 2-bucket LPT hot/cold route_sort_v2:
//      - Hot experts (cnt > 32) tiled at BM=64 (scheduled first for L2 reuse!)
//      - Cold experts (cnt <= 32) tiled at BM=32 (zero wasted syncs/occupancy!)
//      - Emits num_tiles[0] = total_tiles, num_tiles[1] = num_hot_tiles on GPU.
//   2. GU-v5 (qwen4_moe_prefill_gate_up_swiglu_bm64):
//      - 4-warp BM=64, BN=64, K=128 100%-overlapped double-buffered (wg/wu/xq/meta)
//      - 30.0 KB (Q4_K) / 32.0 KB (Q5_K) SMEM => 3 CTAs/SM (144 CTAs in flight)!
//      - Coalesced 128-bit uint4 mid epilogue, RZ mma0, 0 B register spills!
//   3. Upgraded BM=32 cold gate_up_swiglu with RZ mma0 + coalesced uint4 mid!
//   4. N4 BM=64 Down Projection (ZERO cross-warp K-reduction, SMEM-staged mid!):
//      - Q5_1: BK=128 3-CTA/SM (29.7 KB) & BK=64 4-CTA/SM (22.5 KB), 0 B spills!
//      - Q8_0: BK=64 8-warp 3-CTA/SM (25.6 KB, 77 regs), 0 B spills (2.18x vs v1!)
//   5. Combine V2: exact 320 threads (2560 cols/CTA), SMEM-cached inv_perm/scores!
// ============================================================================

#include <cuda_bf16.h>
#include <cuda_fp16.h>

__device__ __forceinline__ float q4p_silu(float x) {
  return x / (1.0f + expf(-x));
}

__device__ __forceinline__ float q4p_sigmoid(float x) {
  return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ void q4p_cp_async16(void* smem_ptr, const void* glob_ptr) {
  unsigned int addr = static_cast<unsigned int>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(addr), "l"(glob_ptr) : "memory");
}

__device__ __forceinline__ void q4p_cp_async16_ca(void* smem_ptr, const void* glob_ptr) {
  unsigned int addr = static_cast<unsigned int>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(addr), "l"(glob_ptr) : "memory");
}

__device__ __forceinline__ void q4p_st_cs_u4(uint4* ptr, uint4 v) {
  asm volatile("st.global.cs.v4.u32 [%0], {%1, %2, %3, %4};\n"
               :: "l"(ptr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}

__device__ __forceinline__ void q4p_cp_async8(void* smem_ptr, const void* glob_ptr) {
  unsigned int addr = static_cast<unsigned int>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" :: "r"(addr), "l"(glob_ptr) : "memory");
}

__device__ __forceinline__ void q4p_cp_async4(void* smem_ptr, const void* glob_ptr) {
  unsigned int addr = static_cast<unsigned int>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(addr), "l"(glob_ptr) : "memory");
}

__device__ __forceinline__ void q4p_cp_commit() {
  asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template <int N>
__device__ __forceinline__ void q4p_cp_wait() {
  asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
}

__device__ __forceinline__ void q4p_mma_s8_z(
    unsigned int a0, unsigned int a1, unsigned int a2, unsigned int a3,
    unsigned int b0, unsigned int b1,
    int& d0, int& d1, int& d2, int& d3) {
  int z = 0;
  asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %10, %10, %10};"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(z));
}

__device__ __forceinline__ void q4p_mma_bf16_z(
    float& d0, float& d1, float& d2, float& d3,
    unsigned int a0, unsigned int a1, unsigned int a2, unsigned int a3,
    unsigned int b0, unsigned int b1) {
  float z = 0.0f;
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %10, %10, %10};"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(z));
}

__device__ __forceinline__ void q4p_mma_bf16(
    float& d0, float& d1, float& d2, float& d3,
    unsigned int a0, unsigned int a1, unsigned int a2, unsigned int a3,
    unsigned int b0, unsigned int b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ unsigned int q4p_pack_bf16x2(float a, float b) {
  __nv_bfloat162 h = __floats2bfloat162_rn(a, b);
  return *reinterpret_cast<const unsigned int*>(&h);
}

__device__ __forceinline__ float q4p_sum_bf16x8(uint4 v) {
  const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&v);
  float2 f0 = __bfloat1622float2(p[0]);
  float2 f1 = __bfloat1622float2(p[1]);
  float2 f2 = __bfloat1622float2(p[2]);
  float2 f3 = __bfloat1622float2(p[3]);
  return ((f0.x + f0.y) + (f1.x + f1.y)) + ((f2.x + f2.y) + (f3.x + f3.y));
}

__device__ __forceinline__ int q4p_min_dot8(const uint4& sc, const uint4& qsu4) {
  const short* q = reinterpret_cast<const short*>(&qsu4);
  int sum = static_cast<int>(q[0]) * static_cast<int>((sc.x >> 8) & 0xffu) +
      static_cast<int>(q[1]) * static_cast<int>(sc.x >> 24);
  sum += static_cast<int>(q[2]) * static_cast<int>((sc.y >> 8) & 0xffu) +
      static_cast<int>(q[3]) * static_cast<int>(sc.y >> 24);
  sum += static_cast<int>(q[4]) * static_cast<int>((sc.z >> 8) & 0xffu) +
      static_cast<int>(q[5]) * static_cast<int>(sc.z >> 24);
  sum += static_cast<int>(q[6]) * static_cast<int>((sc.w >> 8) & 0xffu) +
      static_cast<int>(q[7]) * static_cast<int>(sc.w >> 24);
  return sum;
}

template <int mode>
__device__ __forceinline__ unsigned int q4p_unpack4_kq(unsigned int u) {
  if constexpr (mode == 0) {
    return (u & 0x0000000fu) | ((u & 0x000000f0u) << 4) |
        ((u & 0x00000f00u) << 8) | ((u & 0x0000f000u) << 12);
  } else {
    return (u & 0x0000001fu) | ((u & 0x000003e0u) << 3) |
        ((u & 0x00007c00u) << 6) | ((u & 0x000f8000u) << 9);
  }
}

template <int mode>
__device__ __forceinline__ void q4p_load_pair_kq(
    const unsigned int* seg, int t, unsigned int& out0, unsigned int& out1) {
  if constexpr (mode == 0) {
    const unsigned short* seg16 = reinterpret_cast<const unsigned short*>(seg);
    out0 = q4p_unpack4_kq<0>(seg16[t]);
    out1 = q4p_unpack4_kq<0>(seg16[4 + t]);
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

    out0 = q4p_unpack4_kq<1>(u0);
    out1 = q4p_unpack4_kq<1>(u1);
  }
}

template <int BITS>
__device__ __forceinline__ void q4p_unpack8_affine_bf16(
    const unsigned int* seg, int t, unsigned int out_bf16x2[4]) {
  if constexpr (BITS == 5) {
    const int sh = t * 8;
    const unsigned int lo32 = (t == 0) ? seg[0] : __funnelshift_r(seg[t], seg[t + 1], sh);
    const unsigned int hi8 = seg[t + 1] >> sh;
    out_bf16x2[0] = q4p_pack_bf16x2(float(lo32 & 31u), float((lo32 >> 5) & 31u));
    out_bf16x2[1] = q4p_pack_bf16x2(float((lo32 >> 10) & 31u), float((lo32 >> 15) & 31u));
    out_bf16x2[2] = q4p_pack_bf16x2(float((lo32 >> 20) & 31u), float((lo32 >> 25) & 31u));
    out_bf16x2[3] = q4p_pack_bf16x2(
        float(((lo32 >> 30) | ((hi8 & 7u) << 2)) & 31u), float((hi8 >> 3) & 31u));
  } else {
    const unsigned int u0 = seg[2 * t];
    const unsigned int u1 = seg[2 * t + 1];
    out_bf16x2[0] = q4p_pack_bf16x2(float(u0 & 0xffu), float((u0 >> 8) & 0xffu));
    out_bf16x2[1] = q4p_pack_bf16x2(float((u0 >> 16) & 0xffu), float(u0 >> 24));
    out_bf16x2[2] = q4p_pack_bf16x2(float(u1 & 0xffu), float((u1 >> 8) & 0xffu));
    out_bf16x2[3] = q4p_pack_bf16x2(float((u1 >> 16) & 0xffu), float(u1 >> 24));
  }
}

)cuda"
R"cuda(
// ---------------------------------------------------------------------------
// 1. 2-Bucket LPT Route Sort (v2)
//    - Hot experts (cnt > cold_max, e.g., 32) are tiled at BM_HOT=64 first
//      (contiguous per expert for 24 MB L2 weight residency!).
//    - Cold experts (cnt <= cold_max) are tiled at BM_COLD=32 (1 tile/expert).
//    - num_tiles[0] = total_tiles, num_tiles[1] = num_hot_tiles (when cold_max > 0).
//    - Grid: 1, Block: 512 (E <= 512).
// ---------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(512) void qwen4_moe_prefill_route_sort_v2(
    const int* __restrict__ expert_ids,
    int* __restrict__ sorted_pairs,
    int* __restrict__ inv_perm,
    int* __restrict__ tile_expert,
    int* __restrict__ tile_begin,
    int* __restrict__ tile_end,
    int* __restrict__ num_tiles,
    int P, int E, int BM_hot, int cold_max) {
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  __shared__ int s_cnt[512];
  __shared__ int s_cursor[512];
  __shared__ int s_warp_c[16];
  __shared__ int s_warp_th[16];
  __shared__ int s_warp_tc[16];
  __shared__ int s_tot_hot;

  s_cnt[tid] = 0;
  __syncthreads();
  for (int i = tid; i < P; i += 512) {
    const int e = expert_ids[i];
    if (static_cast<unsigned int>(e) < static_cast<unsigned int>(E)) atomicAdd(&s_cnt[e], 1);
  }
  __syncthreads();

  const int c = (tid < E) ? s_cnt[tid] : 0;
  const bool is_hot = (c > cold_max);
  const int nt_h = (c > 0 && is_hot) ? ((c + BM_hot - 1) / BM_hot) : 0;
  const int nt_c = (c > 0 && !is_hot) ? 1 : 0;

  int scan_c = c, scan_th = nt_h, scan_tc = nt_c;
#pragma unroll
  for (int off = 1; off < 32; off <<= 1) {
    int nc = __shfl_up_sync(0xffffffffu, scan_c, off);
    int nh = __shfl_up_sync(0xffffffffu, scan_th, off);
    int n_c = __shfl_up_sync(0xffffffffu, scan_tc, off);
    if (lane >= off) { scan_c += nc; scan_th += nh; scan_tc += n_c; }
  }
  if (lane == 31) { s_warp_c[warp] = scan_c; s_warp_th[warp] = scan_th; s_warp_tc[warp] = scan_tc; }
  __syncthreads();

  if (warp == 0) {
    int wc = (lane < 16) ? s_warp_c[lane] : 0;
    int wh = (lane < 16) ? s_warp_th[lane] : 0;
    int w_c = (lane < 16) ? s_warp_tc[lane] : 0;
#pragma unroll
    for (int off = 1; off < 16; off <<= 1) {
      int nc = __shfl_up_sync(0xffffffffu, wc, off);
      int nh = __shfl_up_sync(0xffffffffu, wh, off);
      int n_c = __shfl_up_sync(0xffffffffu, w_c, off);
      if (lane >= off) { wc += nc; wh += nh; w_c += n_c; }
    }
    if (lane == 15) {
      s_tot_hot = wh;
      num_tiles[0] = wh + w_c;
      if (cold_max > 0) num_tiles[1] = wh;
    }
    if (lane < 16) {
      s_warp_c[lane] = wc - s_warp_c[lane];
      s_warp_th[lane] = wh - s_warp_th[lane];
      s_warp_tc[lane] = w_c - s_warp_tc[lane];
    }
  }
  __syncthreads();

  const int row_start = s_warp_c[warp] + (scan_c - c);
  s_cursor[tid] = row_start;
  if (tid < E && c > 0) {
    if (is_hot) {
      const int ts = s_warp_th[warp] + (scan_th - nt_h);
      for (int k = 0; k < nt_h; ++k) {
        const int b = row_start + k * BM_hot;
        const int en = (b + BM_hot < row_start + c) ? (b + BM_hot) : (row_start + c);
        tile_expert[ts + k] = tid;
        tile_begin[ts + k] = b;
        tile_end[ts + k] = en;
      }
    } else {
      const int ts = s_tot_hot + s_warp_tc[warp] + (scan_tc - nt_c);
      tile_expert[ts] = tid;
      tile_begin[ts] = row_start;
      tile_end[ts] = row_start + c;
    }
  }
  __syncthreads();

  for (int i = tid; i < P; i += 512) {
    const int e = expert_ids[i];
    if (static_cast<unsigned int>(e) < static_cast<unsigned int>(E)) {
      const int dst = atomicAdd(&s_cursor[e], 1);
      sorted_pairs[dst] = i;
      inv_perm[i] = dst;
    }
  }
}

// ---------------------------------------------------------------------------
// 2. Activation Quantizer (BF16 -> int8, GS=256 superblocks + 8x int16 qsums)
//    Grid: ceil((M * (K/256)) / 8), Block: 256 (1 warp = 256 elems).
// ---------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(256) void qwen4_moe_prefill_quantize_x(
    const nv_bfloat16* __restrict__ x,
    signed char* __restrict__ xq,
    float* __restrict__ x_scales,
    short* __restrict__ x_qsums,
    int M, int K) {
  const int units_per_row = K >> 8;
  const int total_units = M * units_per_row;
  const int warp_id = static_cast<int>((static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x) >> 5);
  const int lane = threadIdx.x & 31;
  if (warp_id >= total_units) return;

  const uint4 raw = reinterpret_cast<const uint4*>(x)[static_cast<size_t>(warp_id) * 32 + lane];
  const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&raw);
  float v[8], amax = 0.0f;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    float2 f = __bfloat1622float2(p[k]);
    v[2 * k] = f.x; v[2 * k + 1] = f.y;
    amax = fmaxf(amax, fmaxf(fabsf(f.x), fabsf(f.y)));
  }
#pragma unroll
  for (int off = 1; off < 32; off <<= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
  const bool valid = (amax > 0.0f) && isfinite(amax);
  const float inv = valid ? (127.0f / amax) : 0.0f;
  int q[8], lsum = 0;
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    q[k] = valid ? static_cast<int>(rintf(v[k] * inv)) : 0;
    lsum += q[k];
  }
  uint2 pk;
  pk.x = (static_cast<unsigned int>(q[0]) & 0xffu) | ((static_cast<unsigned int>(q[1]) & 0xffu) << 8) |
         ((static_cast<unsigned int>(q[2]) & 0xffu) << 16) | ((static_cast<unsigned int>(q[3]) & 0xffu) << 24);
  pk.y = (static_cast<unsigned int>(q[4]) & 0xffu) | ((static_cast<unsigned int>(q[5]) & 0xffu) << 8) |
         ((static_cast<unsigned int>(q[6]) & 0xffu) << 16) | ((static_cast<unsigned int>(q[7]) & 0xffu) << 24);
  reinterpret_cast<uint2*>(xq)[static_cast<size_t>(warp_id) * 32 + lane] = pk;
  if (lane == 0) x_scales[warp_id] = valid ? (amax * (1.0f / 127.0f)) : 0.0f;
  lsum += __shfl_xor_sync(0xffffffffu, lsum, 1, 4);
  lsum += __shfl_xor_sync(0xffffffffu, lsum, 2, 4);
  if ((lane & 3) == 0) x_qsums[static_cast<size_t>(warp_id) * 8 + (lane >> 2)] = static_cast<short>(lsum);
}
)cuda"
R"cuda(
// ---------------------------------------------------------------------------
// 3A. Hot Weight-Stationary Gate + Up + SwiGLU (GU-v5: BM=64, BN=64, 4w, K=128)
//     - 100%-overlapped double-buffered wg/wu/xq/meta via cp.async!
//     - 30.0 KB (Q4_K) / 32.0 KB (Q5_K) SMEM => 3 CTAs/SM (144 CTAs in flight)!
//     - Coalesced 128-bit uint4 mid epilogue, 0 B register spills!
//     - Processes tiles [0 .. limit), where limit = num_tiles[USE_HOT_LIMIT ? 1 : 0].
//     Grid: dim3(N / 64, max_tiles), Block: dim3(32, 4).
// ---------------------------------------------------------------------------
template <int mode, bool USE_HOT_LIMIT>
__global__ __launch_bounds__(128, 3) void qwen4_moe_prefill_gate_up_swiglu_bm64(
    const unsigned int* __restrict__ wg,
    const unsigned char* __restrict__ sg,
    const unsigned short* __restrict__ bg,
    const unsigned int* __restrict__ wu,
    const unsigned char* __restrict__ su,
    const unsigned short* __restrict__ bu,
    const signed char* __restrict__ xq,
    const float* __restrict__ x_scales,
    const short* __restrict__ x_qsums,
    const int* __restrict__ sorted_pairs,
    const int* __restrict__ tile_expert,
    const int* __restrict__ tile_begin,
    const int* __restrict__ tile_end,
    const int* __restrict__ num_tiles,
    signed char* __restrict__ mid_q,
    float* __restrict__ mid_sc,
    short* __restrict__ mid_qsum,
    int N, int K, int top_k) {
  const int tile_idx = blockIdx.y;
  const int tile_limit = USE_HOT_LIMIT ? num_tiles[1] : num_tiles[0];
  if (tile_idx >= tile_limit) return;
  const int expert = tile_expert[tile_idx];
  const int begin = tile_begin[tile_idx];
  const int end = tile_end[tile_idx];
  const int cnt = end - begin;
  if (cnt <= 0) return;
  const int base_row = blockIdx.x * 64;
  if (base_row >= N) return;

  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const int tid = warp * 32 + lane;
  const int g = lane >> 2;
  const int t = lane & 3;

  constexpr int UNITS = 10;
  constexpr int V4_U = (mode == 0) ? 8 : 10;
  constexpr int V4_H = V4_U / 2;
  constexpr int ROW_V4 = UNITS * V4_U;

  union SharedGu64 {
    struct {
      alignas(16) uint4 w[2][64][V4_H];  //  8,192 B (Q4_K) / 10,240 B (Q5_K)
      alignas(16) uint4 xq[2][64][8];    // 16,384 B (swizzled c ^ (r & 7))
      alignas(16) uint4 scg[2][64];      //  2,048 B
      alignas(16) uint4 scu[2][64];      //  2,048 B
      alignas(16) unsigned int big[64];  //    256 B (cp.async during hf==0!)
      alignas(16) unsigned int biu[64];  //    256 B
      alignas(16) uint4 xqs[64];         //  1,024 B
      alignas(16) float xsc[64];         //    256 B
      alignas(16) int tok[64];           //    256 B
    } pipe;                              // 30,720 B (Q4_K) / 32,768 B (Q5_K) => 3 CTAs/SM!
    alignas(16) float s_out[16][64];
  };
  __shared__ SharedGu64 sh;

  if (tid < 64) sh.pipe.tok[tid] = (tid < cnt) ? (sorted_pairs[begin + tid] / top_k) : -1;
  __syncthreads();

  const size_t exp_row0 = static_cast<size_t>(expert) * N + base_row;
  const uint4* wg_u4 = reinterpret_cast<const uint4*>(wg) + exp_row0 * ROW_V4;
  const uint4* wu_u4 = reinterpret_cast<const uint4*>(wu) + exp_row0 * ROW_V4;
  const uint4* xq_u4 = reinterpret_cast<const uint4*>(xq);
  const uint4* xqs_u4 = reinterpret_cast<const uint4*>(x_qsums);
  const int n_slices = (cnt + 7) >> 3;
  const int load_x_rows = n_slices * 8;

  auto load_w_half = [&](int wst, const uint4* w_base_exp, int hstep) {
    const int unit = hstep >> 1;
    const int hf = hstep & 1;
    const int col_off = unit * V4_U + hf * V4_H;
    if constexpr (mode == 0) {
#pragma unroll
      for (int iter = 0; iter < 2; ++iter) {
        const int idx = tid + iter * 128;
        const int r = idx >> 2;
        const int c = idx & 3;
        q4p_cp_async16(&sh.pipe.w[wst][r][c ^ (r & 3)], w_base_exp + static_cast<size_t>(r) * ROW_V4 + col_off + c);
      }
    } else {
      for (int idx = tid; idx < 320; idx += 128) {
        const int r = idx / 5;
        const int c = idx - r * 5;
        q4p_cp_async16(&sh.pipe.w[wst][r][c], w_base_exp + static_cast<size_t>(r) * ROW_V4 + col_off + c);
      }
    }
  };

  auto load_x_half = [&](int xst, int hstep) {
    const int unit = hstep >> 1;
    const int hf = hstep & 1;
    const int total_x = load_x_rows * 8;
    for (int idx = tid; idx < total_x; idx += 128) {
      const int r = idx >> 3;
      const int c = idx & 7;
      const int tok = sh.pipe.tok[r];
      uint4* dst = &sh.pipe.xq[xst][r][c ^ (r & 7)];
      if (tok >= 0) q4p_cp_async16_ca(dst, xq_u4 + (static_cast<size_t>(tok) * UNITS + unit) * 16 + hf * 8 + c);
      else *dst = make_uint4(0u, 0u, 0u, 0u);
    }
  };

  auto load_sc_unit = [&](int ub, int unit) {
    if (tid < 64) q4p_cp_async16(&sh.pipe.scg[ub][tid], reinterpret_cast<const uint4*>(sg) + (exp_row0 + tid) * UNITS + unit);
    else          q4p_cp_async16(&sh.pipe.scu[ub][tid - 64], reinterpret_cast<const uint4*>(su) + (exp_row0 + (tid - 64)) * UNITS + unit);
  };

  auto load_xmeta_bi_async = [&](int unit) {
    if (tid < 64) {
      q4p_cp_async4(&sh.pipe.big[tid], reinterpret_cast<const unsigned int*>(bg) + (exp_row0 + tid) * UNITS + unit);
      q4p_cp_async4(&sh.pipe.biu[tid], reinterpret_cast<const unsigned int*>(bu) + (exp_row0 + tid) * UNITS + unit);
    } else {
      const int r = tid - 64;
      if (r < load_x_rows) {
        const int tok = sh.pipe.tok[r];
        if (tok >= 0) {
          q4p_cp_async4(&sh.pipe.xsc[r], x_scales + static_cast<size_t>(tok) * UNITS + unit);
          q4p_cp_async16_ca(&sh.pipe.xqs[r], xqs_u4 + static_cast<size_t>(tok) * UNITS + unit);
        } else {
          sh.pipe.xsc[r] = 0.0f;
          sh.pipe.xqs[r] = make_uint4(0u, 0u, 0u, 0u);
        }
      }
    }
  };

  load_w_half(0, wg_u4, 0);
  load_x_half(0, 0);
  load_sc_unit(0, 0);
  q4p_cp_commit();
  q4p_cp_wait<0>();
  __syncthreads();

  const int r0 = warp * 16 + g;
  const int r1 = r0 + 8;

  float g_tot[8][4], u_tot[8][4];
#pragma unroll
  for (int ms = 0; ms < 8; ++ms) {
    g_tot[ms][0] = g_tot[ms][1] = g_tot[ms][2] = g_tot[ms][3] = 0.0f;
    u_tot[ms][0] = u_tot[ms][1] = u_tot[ms][2] = u_tot[ms][3] = 0.0f;
  }

  auto accum_half = [&](int wst, int xst, int ub, int hf, bool is_up, int idot[8][4]) {
    const uint4 sc0 = is_up ? sh.pipe.scu[ub][r0] : sh.pipe.scg[ub][r0];
    const uint4 sc1 = is_up ? sh.pipe.scu[ub][r1] : sh.pipe.scg[ub][r1];
#pragma unroll 1
    for (int ls = 0; ls < 4; ++ls) {
      const int s = hf * 4 + ls;
      unsigned int a0, a1, a2, a3;
      if constexpr (mode == 0) {
        q4p_load_pair_kq<0>(reinterpret_cast<const unsigned int*>(&sh.pipe.w[wst][r0][ls ^ (r0 & 3)]), t, a0, a2);
        q4p_load_pair_kq<0>(reinterpret_cast<const unsigned int*>(&sh.pipe.w[wst][r1][ls ^ (r1 & 3)]), t, a1, a3);
      } else {
        q4p_load_pair_kq<1>(reinterpret_cast<const unsigned int*>(sh.pipe.w[wst][r0]) + ls * 5, t, a0, a2);
        q4p_load_pair_kq<1>(reinterpret_cast<const unsigned int*>(sh.pipe.w[wst][r1]) + ls * 5, t, a1, a3);
      }
      const unsigned int sw0 = (s < 2) ? sc0.x : (s < 4 ? sc0.y : (s < 6 ? sc0.z : sc0.w));
      const unsigned int sw1 = (s < 2) ? sc1.x : (s < 4 ? sc1.y : (s < 6 ? sc1.z : sc1.w));
      const int shft = (s & 1) * 16;
      const int ws0 = static_cast<int>((sw0 >> shft) & 0xffu);
      const int ws1 = static_cast<int>((sw1 >> shft) & 0xffu);
      const int col0 = (2 * ls) ^ g;
      const int col1 = (2 * ls + 1) ^ g;
#pragma unroll
      for (int ms = 0; ms < 8; ++ms) {
        if (ms < n_slices) {
          const int row_x = ms * 8 + g;
          const unsigned int b0 = reinterpret_cast<const unsigned int*>(&sh.pipe.xq[xst][row_x][col0])[t];
          const unsigned int b1 = reinterpret_cast<const unsigned int*>(&sh.pipe.xq[xst][row_x][col1])[t];
          int c0, c1, c2, c3;
          q4p_mma_s8_z(a0, a1, a2, a3, b0, b1, c0, c1, c2, c3);
          idot[ms][0] += c0 * ws0; idot[ms][1] += c1 * ws0;
          idot[ms][2] += c2 * ws1; idot[ms][3] += c3 * ws1;
        }
      }
    }
  };

  auto fold_unit = [&](int ub, bool is_up, const int idot[8][4], float tot[8][4]) {
    const uint4 sc0 = is_up ? sh.pipe.scu[ub][r0] : sh.pipe.scg[ub][r0];
    const uint4 sc1 = is_up ? sh.pipe.scu[ub][r1] : sh.pipe.scg[ub][r1];
    const unsigned int b0p = is_up ? sh.pipe.biu[r0] : sh.pipe.big[r0];
    const unsigned int b1p = is_up ? sh.pipe.biu[r1] : sh.pipe.big[r1];
    const float2 dm0 = __half22float2(*reinterpret_cast<const __half2*>(&b0p));
    const float2 dm1 = __half22float2(*reinterpret_cast<const __half2*>(&b1p));
#pragma unroll
    for (int ms = 0; ms < 8; ++ms) {
      if (ms < n_slices) {
        const int c0_loc = ms * 8 + 2 * t;
        const int c1_loc = c0_loc + 1;
        const float ad0 = sh.pipe.xsc[c0_loc];
        const float ad1 = sh.pipe.xsc[c1_loc];
        const uint4 qs0 = sh.pipe.xqs[c0_loc];
        const uint4 qs1 = sh.pipe.xqs[c1_loc];
        tot[ms][0] = fmaf(ad0, fmaf(dm0.x, float(idot[ms][0]), -dm0.y * float(q4p_min_dot8(sc0, qs0))), tot[ms][0]);
        tot[ms][1] = fmaf(ad1, fmaf(dm0.x, float(idot[ms][1]), -dm0.y * float(q4p_min_dot8(sc0, qs1))), tot[ms][1]);
        tot[ms][2] = fmaf(ad0, fmaf(dm1.x, float(idot[ms][2]), -dm1.y * float(q4p_min_dot8(sc1, qs0))), tot[ms][2]);
        tot[ms][3] = fmaf(ad1, fmaf(dm1.x, float(idot[ms][3]), -dm1.y * float(q4p_min_dot8(sc1, qs1))), tot[ms][3]);
      }
    }
  };

)cuda"
R"cuda(
#pragma unroll 1
  for (int unit = 0; unit < UNITS; ++unit) {
    const int ub = unit & 1;
    int ig[8][4], iu[8][4];
#pragma unroll
    for (int ms = 0; ms < 8; ++ms) {
      ig[ms][0] = ig[ms][1] = ig[ms][2] = ig[ms][3] = 0;
      iu[ms][0] = iu[ms][1] = iu[ms][2] = iu[ms][3] = 0;
    }

    // Half 0 (K=0..127)
    {
      const int hstep = unit * 2;
      load_w_half(1, wu_u4, hstep);
      q4p_cp_commit();
      load_x_half(1, hstep + 1);
      load_xmeta_bi_async(unit);
      q4p_cp_commit();

      accum_half(0, 0, ub, 0, false, ig);
      q4p_cp_wait<1>();
      __syncthreads();

      load_w_half(0, wg_u4, hstep + 1);
      q4p_cp_commit();

      accum_half(1, 0, ub, 0, true, iu);
      q4p_cp_wait<0>();
      __syncthreads();
    }
    // Half 1 (K=128..255)
    {
      const int hstep = unit * 2 + 1;
      const bool has_nxt = (unit + 1 < UNITS);
      load_w_half(1, wu_u4, hstep);
      q4p_cp_commit();
      if (has_nxt) {
        load_x_half(0, hstep + 1);
        load_sc_unit(ub ^ 1, unit + 1);
        q4p_cp_commit();
      }

      accum_half(0, 1, ub, 1, false, ig);
      fold_unit(ub, false, ig, g_tot);

      if (has_nxt) q4p_cp_wait<1>();
      else         q4p_cp_wait<0>();
      __syncthreads();

      if (has_nxt) {
        load_w_half(0, wg_u4, hstep + 1);
        q4p_cp_commit();
      }

      accum_half(1, 1, ub, 1, true, iu);
      fold_unit(ub, true, iu, u_tot);

      if (has_nxt) q4p_cp_wait<0>();
      __syncthreads();
    }
  }

  const int n_passes = (cnt + 15) >> 4;
  const int g64 = base_row >> 6;
  const int g32 = base_row >> 5;
  const int r_in = tid >> 3;
  const int sub = tid & 7;
#pragma unroll
  for (int pass = 0; pass < 4; ++pass) {
    if (pass < n_passes) {
      const int s0 = pass * 2;
      const int s1 = s0 + 1;
      const int wc = warp * 16;
      sh.s_out[2 * t][wc + g]         = q4p_silu(g_tot[s0][0]) * u_tot[s0][0];
      sh.s_out[2 * t + 1][wc + g]     = q4p_silu(g_tot[s0][1]) * u_tot[s0][1];
      sh.s_out[2 * t][wc + g + 8]     = q4p_silu(g_tot[s0][2]) * u_tot[s0][2];
      sh.s_out[2 * t + 1][wc + g + 8] = q4p_silu(g_tot[s0][3]) * u_tot[s0][3];
      if (s1 < n_slices) {
        sh.s_out[8 + 2 * t][wc + g]         = q4p_silu(g_tot[s1][0]) * u_tot[s1][0];
        sh.s_out[8 + 2 * t + 1][wc + g]     = q4p_silu(g_tot[s1][1]) * u_tot[s1][1];
        sh.s_out[8 + 2 * t][wc + g + 8]     = q4p_silu(g_tot[s1][2]) * u_tot[s1][2];
        sh.s_out[8 + 2 * t + 1][wc + g + 8] = q4p_silu(g_tot[s1][3]) * u_tot[s1][3];
      }
      __syncthreads();
      const int r_loc = pass * 16 + r_in;
      const bool active_r = (r_loc < cnt);
      const float4 f0 = active_r
          ? *reinterpret_cast<const float4*>(&sh.s_out[r_in][sub * 8])
          : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      const float4 f1 = active_r
          ? *reinterpret_cast<const float4*>(&sh.s_out[r_in][sub * 8 + 4])
          : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      float amax = fmaxf(fmaxf(fabsf(f0.x), fabsf(f0.y)), fmaxf(fabsf(f0.z), fabsf(f0.w)));
      amax = fmaxf(amax, fmaxf(fmaxf(fabsf(f1.x), fabsf(f1.y)), fmaxf(fabsf(f1.z), fabsf(f1.w))));
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1, 8));
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2, 8));
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 4, 8));
      const bool valid = active_r && (amax > 0.0f) && isfinite(amax);
      const float inv = valid ? (127.0f / amax) : 0.0f;
      const int q0 = valid ? static_cast<int>(rintf(f0.x * inv)) : 0;
      const int q1 = valid ? static_cast<int>(rintf(f0.y * inv)) : 0;
      const int q2 = valid ? static_cast<int>(rintf(f0.z * inv)) : 0;
      const int q3 = valid ? static_cast<int>(rintf(f0.w * inv)) : 0;
      const int q4 = valid ? static_cast<int>(rintf(f1.x * inv)) : 0;
      const int q5 = valid ? static_cast<int>(rintf(f1.y * inv)) : 0;
      const int q6 = valid ? static_cast<int>(rintf(f1.z * inv)) : 0;
      const int q7 = valid ? static_cast<int>(rintf(f1.w * inv)) : 0;
      int lsum = (q0 + q1) + (q2 + q3) + (q4 + q5) + (q6 + q7);
      lsum += __shfl_xor_sync(0xffffffffu, lsum, 1, 4);
      lsum += __shfl_xor_sync(0xffffffffu, lsum, 2, 4);
      if (active_r) {
        uint2 pk;
        pk.x = (static_cast<unsigned int>(q0) & 0xffu) | ((static_cast<unsigned int>(q1) & 0xffu) << 8) |
               ((static_cast<unsigned int>(q2) & 0xffu) << 16) | ((static_cast<unsigned int>(q3) & 0xffu) << 24);
        pk.y = (static_cast<unsigned int>(q4) & 0xffu) | ((static_cast<unsigned int>(q5) & 0xffu) << 8) |
               ((static_cast<unsigned int>(q6) & 0xffu) << 16) | ((static_cast<unsigned int>(q7) & 0xffu) << 24);
        const size_t grow = static_cast<size_t>(begin + r_loc);
        reinterpret_cast<uint2*>(mid_q + grow * N + base_row)[sub] = pk;
        if ((sub & 3) == 0) {
          mid_qsum[grow * (N >> 5) + g32 + (sub >> 2)] = static_cast<short>(lsum);
        }
        if (sub == 0) {
          mid_sc[grow * (N >> 6) + g64] = valid ? (amax * (1.0f / 127.0f)) : 0.0f;
        }
      }
      __syncthreads();
    }
  }
}
)cuda"
R"cuda(
// ---------------------------------------------------------------------------
// 3B. Cold Expert Gate + Up + SwiGLU (BM=32, BN=64, 4w, K=256 single-shot)
//     - Optimal for cold experts (cnt <= 32): 20 barriers, 27.9 KB SMEM (3 CTAs/SM).
//     - Upgraded with RZ mma0 + coalesced 128-bit uint4 mid epilogue!
//     - Processes tiles [offset .. num_tiles[0]), offset = USE_COLD_OFFSET ? num_tiles[1] : 0.
//     Grid: dim3(N / 64, max_cold_tiles), Block: dim3(32, 4).
// ---------------------------------------------------------------------------
template <int mode, bool USE_COLD_OFFSET>
__global__ __launch_bounds__(128, mode == 0 ? 3 : 2) void qwen4_moe_prefill_gate_up_swiglu_bm32(
    const unsigned int* __restrict__ wg,
    const unsigned char* __restrict__ sg,
    const unsigned short* __restrict__ bg,
    const unsigned int* __restrict__ wu,
    const unsigned char* __restrict__ su,
    const unsigned short* __restrict__ bu,
    const signed char* __restrict__ xq,
    const float* __restrict__ x_scales,
    const short* __restrict__ x_qsums,
    const int* __restrict__ sorted_pairs,
    const int* __restrict__ tile_expert,
    const int* __restrict__ tile_begin,
    const int* __restrict__ tile_end,
    const int* __restrict__ num_tiles,
    signed char* __restrict__ mid_q,
    float* __restrict__ mid_sc,
    short* __restrict__ mid_qsum,
    int N, int K, int top_k) {
  const int tile_off = USE_COLD_OFFSET ? num_tiles[1] : 0;
  const int tile_idx = tile_off + blockIdx.y;
  if (tile_idx >= num_tiles[0]) return;
  const int expert = tile_expert[tile_idx];
  const int begin = tile_begin[tile_idx];
  const int end = tile_end[tile_idx];
  const int cnt = end - begin;
  if (cnt <= 0) return;
  const int base_row = blockIdx.x * 64;
  if (base_row >= N) return;

  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const int tid = warp * 32 + lane;
  const int g = lane >> 2;
  const int t = lane & 3;

  constexpr int UNITS = 10;
  constexpr int V4_U = (mode == 0) ? 8 : 10;
  constexpr int PAD_V4 = (mode == 0) ? 8 : 11;
  constexpr int ROW_V4 = UNITS * V4_U;

  struct WeightStage {
    alignas(16) uint4 w[64][PAD_V4];
    alignas(16) uint4 sc[64];
    alignas(16) unsigned int bi[64];
  };
  union SharedGu32 {
    struct {
      WeightStage wst[2];
      alignas(16) uint4 s_xq[32][16];
      alignas(16) uint4 s_xqs[32];
      alignas(16) float s_xsc[32];
      alignas(16) int s_tok[32];
    } pipe;
    alignas(16) float s_out[16][64];
  };
  __shared__ SharedGu32 sh;

  if (tid < 32) sh.pipe.s_tok[tid] = (tid < cnt) ? (sorted_pairs[begin + tid] / top_k) : -1;
  __syncthreads();

  const size_t exp_row0 = static_cast<size_t>(expert) * N + base_row;
  const uint4* xq_u4 = reinterpret_cast<const uint4*>(xq);
  const uint4* xqs_u4 = reinterpret_cast<const uint4*>(x_qsums);
  const int n_slices = (cnt + 7) >> 3;
  const int load_x_rows = n_slices * 8;

  auto load_w = [&](int st, const unsigned int* wb, const unsigned char* sb,
                    const unsigned short* bb, int unit) {
    const uint4* w_base = reinterpret_cast<const uint4*>(wb) + exp_row0 * ROW_V4 + unit * V4_U;
    if constexpr (mode == 0) {
#pragma unroll
      for (int iter = 0; iter < 4; ++iter) {
        const int idx = tid + iter * 128;
        const int r = idx >> 3;
        const int c = idx & 7;
        q4p_cp_async16(&sh.pipe.wst[st].w[r][c ^ (r & 7)], w_base + static_cast<size_t>(r) * ROW_V4 + c);
      }
    } else {
#pragma unroll
      for (int iter = 0; iter < 5; ++iter) {
        const int idx = tid + iter * 128;
        const int r = idx / 10;
        const int c = idx - r * 10;
        q4p_cp_async16(&sh.pipe.wst[st].w[r][c], w_base + static_cast<size_t>(r) * ROW_V4 + c);
      }
    }
    if (tid < 64) {
      q4p_cp_async16(&sh.pipe.wst[st].sc[tid], reinterpret_cast<const uint4*>(sb) + (exp_row0 + tid) * UNITS + unit);
    } else {
      const int r = tid - 64;
      sh.pipe.wst[st].bi[r] = reinterpret_cast<const unsigned int*>(bb)[(exp_row0 + r) * UNITS + unit];
    }
    q4p_cp_commit();
  };

  auto load_x_unit = [&](int unit) {
    const int total_u4 = load_x_rows * 16;
    for (int idx = tid; idx < total_u4; idx += 128) {
      const int r = idx >> 4;
      const int c = idx & 15;
      const int tok = sh.pipe.s_tok[r];
      uint4* dst = &sh.pipe.s_xq[r][c ^ (r & 7)];
      if (tok >= 0) q4p_cp_async16_ca(dst, xq_u4 + (static_cast<size_t>(tok) * UNITS + unit) * 16 + c);
      else *dst = make_uint4(0u, 0u, 0u, 0u);
    }
    if (tid < load_x_rows) {
      const int tok = sh.pipe.s_tok[tid];
      if (tok >= 0) {
        sh.pipe.s_xsc[tid] = x_scales[static_cast<size_t>(tok) * UNITS + unit];
        sh.pipe.s_xqs[tid] = xqs_u4[static_cast<size_t>(tok) * UNITS + unit];
      } else {
        sh.pipe.s_xsc[tid] = 0.0f;
        sh.pipe.s_xqs[tid] = make_uint4(0u, 0u, 0u, 0u);
      }
    }
    q4p_cp_commit();
  };

  load_w(0, wg, sg, bg, 0);
  load_x_unit(0);
  q4p_cp_wait<0>();
  __syncthreads();

  const int r0 = warp * 16 + g;
  const int r1 = r0 + 8;
  float g_tot[4][4] = {{0.0f}};
  float u_tot[4][4] = {{0.0f}};

  auto compute_stage = [&](int st, float tot[4][4]) {
    const uint4 sc0 = sh.pipe.wst[st].sc[r0];
    const uint4 sc1 = sh.pipe.wst[st].sc[r1];
    int idot[4][4] = {{0}};
#pragma unroll 1
    for (int s = 0; s < 8; ++s) {
      unsigned int a0, a1, a2, a3;
      if constexpr (mode == 0) {
        q4p_load_pair_kq<0>(reinterpret_cast<const unsigned int*>(&sh.pipe.wst[st].w[r0][s ^ g]), t, a0, a2);
        q4p_load_pair_kq<0>(reinterpret_cast<const unsigned int*>(&sh.pipe.wst[st].w[r1][s ^ g]), t, a1, a3);
      } else {
        q4p_load_pair_kq<1>(reinterpret_cast<const unsigned int*>(sh.pipe.wst[st].w[r0]) + s * 5, t, a0, a2);
        q4p_load_pair_kq<1>(reinterpret_cast<const unsigned int*>(sh.pipe.wst[st].w[r1]) + s * 5, t, a1, a3);
      }
      const unsigned int sw0 = (s < 2) ? sc0.x : (s < 4 ? sc0.y : (s < 6 ? sc0.z : sc0.w));
      const unsigned int sw1 = (s < 2) ? sc1.x : (s < 4 ? sc1.y : (s < 6 ? sc1.z : sc1.w));
      const int shft = (s & 1) * 16;
      const int ws0 = static_cast<int>((sw0 >> shft) & 0xffu);
      const int ws1 = static_cast<int>((sw1 >> shft) & 0xffu);
      const int col0 = (2 * s) ^ g;
      const int col1 = (2 * s + 1) ^ g;
#pragma unroll
      for (int ms = 0; ms < 4; ++ms) {
        if (ms < n_slices) {
          const int row_x = ms * 8 + g;
          const unsigned int b0 = reinterpret_cast<const unsigned int*>(&sh.pipe.s_xq[row_x][col0])[t];
          const unsigned int b1 = reinterpret_cast<const unsigned int*>(&sh.pipe.s_xq[row_x][col1])[t];
          int c0, c1, c2, c3;
          q4p_mma_s8_z(a0, a1, a2, a3, b0, b1, c0, c1, c2, c3);
          idot[ms][0] += c0 * ws0; idot[ms][1] += c1 * ws0;
          idot[ms][2] += c2 * ws1; idot[ms][3] += c3 * ws1;
        }
      }
    }
    const unsigned int b0_pair = sh.pipe.wst[st].bi[r0];
    const unsigned int b1_pair = sh.pipe.wst[st].bi[r1];
    const float2 dm0 = __half22float2(*reinterpret_cast<const __half2*>(&b0_pair));
    const float2 dm1 = __half22float2(*reinterpret_cast<const __half2*>(&b1_pair));
#pragma unroll
    for (int ms = 0; ms < 4; ++ms) {
      if (ms < n_slices) {
        const int c0_loc = ms * 8 + 2 * t;
        const int c1_loc = c0_loc + 1;
        const float ad0 = sh.pipe.s_xsc[c0_loc];
        const float ad1 = sh.pipe.s_xsc[c1_loc];
        const uint4 qs0 = sh.pipe.s_xqs[c0_loc];
        const uint4 qs1 = sh.pipe.s_xqs[c1_loc];
        tot[ms][0] = fmaf(ad0, fmaf(dm0.x, float(idot[ms][0]), -dm0.y * float(q4p_min_dot8(sc0, qs0))), tot[ms][0]);
        tot[ms][1] = fmaf(ad1, fmaf(dm0.x, float(idot[ms][1]), -dm0.y * float(q4p_min_dot8(sc0, qs1))), tot[ms][1]);
        tot[ms][2] = fmaf(ad0, fmaf(dm1.x, float(idot[ms][2]), -dm1.y * float(q4p_min_dot8(sc1, qs0))), tot[ms][2]);
        tot[ms][3] = fmaf(ad1, fmaf(dm1.x, float(idot[ms][3]), -dm1.y * float(q4p_min_dot8(sc1, qs1))), tot[ms][3]);
      }
    }
  };

)cuda"
R"cuda(
#pragma unroll 1
  for (int unit = 0; unit < UNITS; ++unit) {
    load_w(1, wu, su, bu, unit);
    compute_stage(0, g_tot);
    q4p_cp_wait<0>();
    __syncthreads();
    if (unit + 1 < UNITS) load_w(0, wg, sg, bg, unit + 1);
    compute_stage(1, u_tot);
    __syncthreads();
    if (unit + 1 < UNITS) {
      load_x_unit(unit + 1);
      q4p_cp_wait<0>();
      __syncthreads();
    }
  }

  const int n_passes = (cnt > 16) ? 2 : 1;
  const int g64 = base_row >> 6;
  const int g32 = base_row >> 5;
  const int r_in = tid >> 3;
  const int sub = tid & 7;
#pragma unroll
  for (int pass = 0; pass < 2; ++pass) {
    if (pass < n_passes) {
      const int s0 = pass * 2;
      const int s1 = s0 + 1;
      const int wc = warp * 16;
      sh.s_out[2 * t][wc + g]         = q4p_silu(g_tot[s0][0]) * u_tot[s0][0];
      sh.s_out[2 * t + 1][wc + g]     = q4p_silu(g_tot[s0][1]) * u_tot[s0][1];
      sh.s_out[2 * t][wc + g + 8]     = q4p_silu(g_tot[s0][2]) * u_tot[s0][2];
      sh.s_out[2 * t + 1][wc + g + 8] = q4p_silu(g_tot[s0][3]) * u_tot[s0][3];
      if (s1 < n_slices) {
        sh.s_out[8 + 2 * t][wc + g]         = q4p_silu(g_tot[s1][0]) * u_tot[s1][0];
        sh.s_out[8 + 2 * t + 1][wc + g]     = q4p_silu(g_tot[s1][1]) * u_tot[s1][1];
        sh.s_out[8 + 2 * t][wc + g + 8]     = q4p_silu(g_tot[s1][2]) * u_tot[s1][2];
        sh.s_out[8 + 2 * t + 1][wc + g + 8] = q4p_silu(g_tot[s1][3]) * u_tot[s1][3];
      }
      __syncthreads();
      const int r_loc = pass * 16 + r_in;
      const bool active_r = (r_loc < cnt);
      const float4 f0 = active_r
          ? *reinterpret_cast<const float4*>(&sh.s_out[r_in][sub * 8])
          : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      const float4 f1 = active_r
          ? *reinterpret_cast<const float4*>(&sh.s_out[r_in][sub * 8 + 4])
          : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      float amax = fmaxf(fmaxf(fabsf(f0.x), fabsf(f0.y)), fmaxf(fabsf(f0.z), fabsf(f0.w)));
      amax = fmaxf(amax, fmaxf(fmaxf(fabsf(f1.x), fabsf(f1.y)), fmaxf(fabsf(f1.z), fabsf(f1.w))));
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1, 8));
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2, 8));
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 4, 8));
      const bool valid = active_r && (amax > 0.0f) && isfinite(amax);
      const float inv = valid ? (127.0f / amax) : 0.0f;
      const int q0 = valid ? static_cast<int>(rintf(f0.x * inv)) : 0;
      const int q1 = valid ? static_cast<int>(rintf(f0.y * inv)) : 0;
      const int q2 = valid ? static_cast<int>(rintf(f0.z * inv)) : 0;
      const int q3 = valid ? static_cast<int>(rintf(f0.w * inv)) : 0;
      const int q4 = valid ? static_cast<int>(rintf(f1.x * inv)) : 0;
      const int q5 = valid ? static_cast<int>(rintf(f1.y * inv)) : 0;
      const int q6 = valid ? static_cast<int>(rintf(f1.z * inv)) : 0;
      const int q7 = valid ? static_cast<int>(rintf(f1.w * inv)) : 0;
      int lsum = (q0 + q1) + (q2 + q3) + (q4 + q5) + (q6 + q7);
      lsum += __shfl_xor_sync(0xffffffffu, lsum, 1, 4);
      lsum += __shfl_xor_sync(0xffffffffu, lsum, 2, 4);
      if (active_r) {
        uint2 pk;
        pk.x = (static_cast<unsigned int>(q0) & 0xffu) | ((static_cast<unsigned int>(q1) & 0xffu) << 8) |
               ((static_cast<unsigned int>(q2) & 0xffu) << 16) | ((static_cast<unsigned int>(q3) & 0xffu) << 24);
        pk.y = (static_cast<unsigned int>(q4) & 0xffu) | ((static_cast<unsigned int>(q5) & 0xffu) << 8) |
               ((static_cast<unsigned int>(q6) & 0xffu) << 16) | ((static_cast<unsigned int>(q7) & 0xffu) << 24);
        const size_t grow = static_cast<size_t>(begin + r_loc);
        reinterpret_cast<uint2*>(mid_q + grow * N + base_row)[sub] = pk;
        if ((sub & 3) == 0) {
          mid_qsum[grow * (N >> 5) + g32 + (sub >> 2)] = static_cast<short>(lsum);
        }
        if (sub == 0) {
          mid_sc[grow * (N >> 6) + g64] = valid ? (amax * (1.0f / 127.0f)) : 0.0f;
        }
      }
      __syncthreads();
    }
  }
}
)cuda"
R"cuda(
// ---------------------------------------------------------------------------
// 4. INT8 Tensor-Core BM=64 Down Projection (v3: Q5_1 & Q8_0, BK=128, 2-stage!)
//    - Uses GS=64 int8 mid_q + mid_sc + GS=32 mid_qsum fused in gate_up epilogue!
//    - 100%-overlapped 2-stage cp.async of BOTH weights AND activations + meta!
//    - Q5_1 (BITS=5): 30.7 KB SMEM => 3 CTAs/SM (144 CTAs in flight!), 6-op bit unpack!
//    - Q8_0 (BITS=8): 36.8 KB SMEM => 2 CTAs/SM, 1-op XOR sign flip + general bias!
//    - 2x Tensor-Core throughput (m16n8k32.s8 vs m16n8k16.bf16), 0 B register spills!
//    Grid: dim3(N / 64, max_tiles), Block: dim3(32, 4).
// ---------------------------------------------------------------------------
template <int BITS>
__global__ __launch_bounds__(128, BITS == 5 ? 3 : 2) void qwen4_moe_prefill_down_bm64_i8(
    const unsigned int* __restrict__ wd,
    const unsigned short* __restrict__ sd,
    const unsigned short* __restrict__ bd,
    const signed char* __restrict__ mid_q,
    const float* __restrict__ mid_sc,
    const short* __restrict__ mid_qsum,
    const int* __restrict__ tile_expert,
    const int* __restrict__ tile_begin,
    const int* __restrict__ tile_end,
    const int* __restrict__ num_tiles,
    nv_bfloat16* __restrict__ down_out,
    int N, int K) {
  const int tile_idx = blockIdx.y;
  if (tile_idx >= num_tiles[0]) return;
  const int expert = tile_expert[tile_idx];
  const int begin = tile_begin[tile_idx];
  const int end = tile_end[tile_idx];
  const int cnt = end - begin;
  if (cnt <= 0) return;
  const int base_row = blockIdx.x * 64;
  if (base_row >= N) return;

  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const int tid = warp * 32 + lane;
  const int g = lane >> 2;
  const int t = lane & 3;

  constexpr int GROUPS = 20;
  constexpr int ROW_U4 = (BITS == 5) ? 25 : 40;
  constexpr int W_STG_U4 = (BITS == 5) ? 5 : 8;

  struct StageDnI8 {
    alignas(16) uint4 x[64][8];            //  8,192 B (swizzled c ^ (r & 7))
    alignas(16) uint4 w[64][W_STG_U4];     //  5,120 B (Q5_1) / 8,192 B (Q8_0)
    alignas(16) unsigned short sc[64][4];  //    512 B
    alignas(16) unsigned short bi[64][4];  //    512 B
    alignas(16) float xsc[64][2];          //    512 B
    alignas(16) short xqs[64][4];          //    512 B
  };
  union SharedDnI8 {
    StageDnI8 st[2];
    alignas(16) nv_bfloat16 s_out[4][16][16];
  };
  __shared__ SharedDnI8 sh;

  const size_t exp_row0 = static_cast<size_t>(expert) * N + base_row;
  const uint4* wd_u4 = reinterpret_cast<const uint4*>(wd) + exp_row0 * ROW_U4;
  const uint4* mid_u4 = reinterpret_cast<const uint4*>(mid_q);
  const size_t mid_stride_u4 = static_cast<size_t>(K) >> 4;
  const int n_slices = (cnt + 7) >> 3;
  const int load_x_rows = n_slices * 8;

  auto load_stage = [&](int s_idx, int step) {
    const uint4* w_base = wd_u4 + step * W_STG_U4;
    if constexpr (BITS == 5) {
      for (int i = tid; i < 320; i += 128) {
        const int r = (i * 52429) >> 18;
        const int c = i - r * 5;
        q4p_cp_async16(&sh.st[s_idx].w[r][c], w_base + static_cast<size_t>(r) * ROW_U4 + c);
      }
    } else {
#pragma unroll
      for (int iter = 0; iter < 4; ++iter) {
        const int i = tid + iter * 128;
        const int r = i >> 3;
        const int c = i & 7;
        q4p_cp_async16(&sh.st[s_idx].w[r][c ^ (r & 7)], w_base + static_cast<size_t>(r) * ROW_U4 + c);
      }
    }
    const int x_off = step * 8;
    const int total_x = load_x_rows * 8;
    for (int i = tid; i < total_x; i += 128) {
      const int r = i >> 3;
      const int c = i & 7;
      uint4* dst = &sh.st[s_idx].x[r][c ^ (r & 7)];
      if (r < cnt) q4p_cp_async16_ca(dst, mid_u4 + static_cast<size_t>(begin + r) * mid_stride_u4 + x_off + c);
      else *dst = make_uint4(0u, 0u, 0u, 0u);
    }
    if (tid < 64) {
      const size_t sb_off = (exp_row0 + tid) * GROUPS + step * 4;
      q4p_cp_async8(&sh.st[s_idx].sc[tid][0], sd + sb_off);
      q4p_cp_async8(&sh.st[s_idx].bi[tid][0], bd + sb_off);
    } else {
      const int r = tid - 64;
      if (r < load_x_rows) {
        if (r < cnt) {
          const size_t grow = static_cast<size_t>(begin + r);
          q4p_cp_async8(&sh.st[s_idx].xsc[r][0], mid_sc + grow * 10 + step * 2);
          q4p_cp_async8(&sh.st[s_idx].xqs[r][0], mid_qsum + grow * 20 + step * 4);
        } else {
          sh.st[s_idx].xsc[r][0] = 0.0f;
          sh.st[s_idx].xsc[r][1] = 0.0f;
          *reinterpret_cast<uint2*>(&sh.st[s_idx].xqs[r][0]) = make_uint2(0u, 0u);
        }
      }
    }
    q4p_cp_commit();
  };

  load_stage(0, 0);

  float tot[8][4];
#pragma unroll
  for (int ms = 0; ms < 8; ++ms) tot[ms][0] = tot[ms][1] = tot[ms][2] = tot[ms][3] = 0.0f;

  const int r0 = warp * 16 + g;
  const int r1 = r0 + 8;

#pragma unroll 1
  for (int step = 0; step < 5; ++step) {
    const int cur = step & 1;
    const int nxt = cur ^ 1;
    if (step + 1 < 5) {
      load_stage(nxt, step + 1);
      q4p_cp_wait<1>();
    } else {
      q4p_cp_wait<0>();
    }
    __syncthreads();

    const unsigned int* w_r0 = reinterpret_cast<const unsigned int*>(sh.st[cur].w[r0]);
    const unsigned int* w_r1 = reinterpret_cast<const unsigned int*>(sh.st[cur].w[r1]);

#pragma unroll 1
    for (int k4 = 0; k4 < 4; ++k4) {
      unsigned int a0, a1, a2, a3;
      if constexpr (BITS == 5) {
        q4p_load_pair_kq<1>(w_r0 + k4 * 5, t, a0, a2);
        q4p_load_pair_kq<1>(w_r1 + k4 * 5, t, a1, a3);
      } else {
        const int c0_0 = (2 * k4) ^ (r0 & 7);
        const int c1_0 = (2 * k4 + 1) ^ (r0 & 7);
        const int c0_1 = (2 * k4) ^ (r1 & 7);
        const int c1_1 = (2 * k4 + 1) ^ (r1 & 7);
        a0 = reinterpret_cast<const unsigned int*>(&sh.st[cur].w[r0][c0_0])[t] ^ 0x80808080u;
        a2 = reinterpret_cast<const unsigned int*>(&sh.st[cur].w[r0][c1_0])[t] ^ 0x80808080u;
        a1 = reinterpret_cast<const unsigned int*>(&sh.st[cur].w[r1][c0_1])[t] ^ 0x80808080u;
        a3 = reinterpret_cast<const unsigned int*>(&sh.st[cur].w[r1][c1_1])[t] ^ 0x80808080u;
      }
      const float sc0 = __half2float(__ushort_as_half(sh.st[cur].sc[r0][k4]));
      const float bi0_raw = __half2float(__ushort_as_half(sh.st[cur].bi[r0][k4]));
      const float bi0 = (BITS == 8) ? fmaf(128.0f, sc0, bi0_raw) : bi0_raw;
      const float sc1 = __half2float(__ushort_as_half(sh.st[cur].sc[r1][k4]));
      const float bi1_raw = __half2float(__ushort_as_half(sh.st[cur].bi[r1][k4]));
      const float bi1 = (BITS == 8) ? fmaf(128.0f, sc1, bi1_raw) : bi1_raw;

      const int col0 = (2 * k4) ^ g;
      const int col1 = (2 * k4 + 1) ^ g;
      const int h64 = k4 >> 1;
#pragma unroll
      for (int ms = 0; ms < 8; ++ms) {
        if (ms < n_slices) {
          const int row_x = ms * 8 + g;
          const unsigned int b0 = reinterpret_cast<const unsigned int*>(&sh.st[cur].x[row_x][col0])[t];
          const unsigned int b1 = reinterpret_cast<const unsigned int*>(&sh.st[cur].x[row_x][col1])[t];
          int c0, c1, c2, c3;
          q4p_mma_s8_z(a0, a1, a2, a3, b0, b1, c0, c1, c2, c3);
          const int c0_loc = ms * 8 + 2 * t;
          const int c1_loc = c0_loc + 1;
          const float xs0 = sh.st[cur].xsc[c0_loc][h64];
          const float xs1 = sh.st[cur].xsc[c1_loc][h64];
          const float q0 = static_cast<float>(sh.st[cur].xqs[c0_loc][k4]);
          const float q1 = static_cast<float>(sh.st[cur].xqs[c1_loc][k4]);
          tot[ms][0] = fmaf(xs0, fmaf(sc0, static_cast<float>(c0), bi0 * q0), tot[ms][0]);
          tot[ms][1] = fmaf(xs1, fmaf(sc0, static_cast<float>(c1), bi0 * q1), tot[ms][1]);
          tot[ms][2] = fmaf(xs0, fmaf(sc1, static_cast<float>(c2), bi1 * q0), tot[ms][2]);
          tot[ms][3] = fmaf(xs1, fmaf(sc1, static_cast<float>(c3), bi1 * q1), tot[ms][3]);
        }
      }
    }
    __syncthreads();
  }

  const int w_col0 = base_row + warp * 16;
  const int n_passes = (cnt + 15) >> 4;
#pragma unroll
  for (int pass = 0; pass < 4; ++pass) {
    if (pass < n_passes) {
      const int s0 = pass * 2;
      const int s1 = s0 + 1;
      sh.s_out[warp][2 * t][g]         = __float2bfloat16(tot[s0][0]);
      sh.s_out[warp][2 * t + 1][g]     = __float2bfloat16(tot[s0][1]);
      sh.s_out[warp][2 * t][g + 8]     = __float2bfloat16(tot[s0][2]);
      sh.s_out[warp][2 * t + 1][g + 8] = __float2bfloat16(tot[s0][3]);
      if (s1 < n_slices) {
        sh.s_out[warp][8 + 2 * t][g]         = __float2bfloat16(tot[s1][0]);
        sh.s_out[warp][8 + 2 * t + 1][g]     = __float2bfloat16(tot[s1][1]);
        sh.s_out[warp][8 + 2 * t][g + 8]     = __float2bfloat16(tot[s1][2]);
        sh.s_out[warp][8 + 2 * t + 1][g + 8] = __float2bfloat16(tot[s1][3]);
      }
      __syncwarp();
      const int r_in = lane >> 1;
      const int half = lane & 1;
      const int r_loc = pass * 16 + r_in;
      if (r_loc < cnt) {
        q4p_st_cs_u4(
            reinterpret_cast<uint4*>(down_out + static_cast<size_t>(begin + r_loc) * N + w_col0) + half,
            reinterpret_cast<const uint4*>(sh.s_out[warp][r_in])[half]);
      }
      __syncwarp();
    }
  }
}

// ---------------------------------------------------------------------------
// 5. Weighted Combine V2 (Exact 320 threads = 2560 cols/CTA, SMEM-cached meta)
//    Grid: dim3(1, M), Block: 320 (or dim3((N/8 + 319)/320, M), 320 for general N).
// ---------------------------------------------------------------------------
template <bool ADD_SHARED>
__global__ __launch_bounds__(320, 4) void qwen4_moe_prefill_combine_v2(
    const nv_bfloat16* __restrict__ down_out,
    const float* __restrict__ scores_f32,
    const int* __restrict__ inv_perm,
    const nv_bfloat16* __restrict__ shared_out,
    const nv_bfloat16* __restrict__ shared_gate,
    nv_bfloat16* __restrict__ out,
    int M, int N, int top_k) {
  const int tok = blockIdx.y;
  const int col8 = blockIdx.x * 320 + threadIdx.x;
  const int n8 = N >> 3;
  if (tok >= M) return;

  __shared__ int s_srow[16];
  __shared__ float s_sc[16];
  if (threadIdx.x < top_k) {
    const int bp = tok * top_k + threadIdx.x;
    s_srow[threadIdx.x] = inv_perm[bp];
    s_sc[threadIdx.x] = scores_f32[bp];
  }
  __syncthreads();
  if (col8 >= n8) return;

  float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  const uint4* dn_u4 = reinterpret_cast<const uint4*>(down_out);
#pragma unroll 1
  for (int slot = 0; slot < top_k; ++slot) {
    const uint4 raw = dn_u4[static_cast<size_t>(s_srow[slot]) * n8 + col8];
    const float sc = s_sc[slot];
    const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      float2 f = __bfloat1622float2(p[k]);
      acc[2 * k]     = fmaf(f.x, sc, acc[2 * k]);
      acc[2 * k + 1] = fmaf(f.y, sc, acc[2 * k + 1]);
    }
  }
  if constexpr (ADD_SHARED) {
    const float sg = q4p_sigmoid(__bfloat162float(shared_gate[tok]));
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
