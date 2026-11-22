R"cuda(
#include <cuda_bf16.h>
#include <cuda_fp16.h>

// One eight-thread group quantizes 32 values of one activation row. The packed
// activation stores Q8 codes, then FP32 scales and Q8 sums for each 32-value
// group. Eight rows are padded so one INT8 MMA tile covers every M in 1..8.
extern "C" __global__ void mlx_node_kquant_quantize_few(
    const nv_bfloat16* x, unsigned char* act, int M, int K) {
  const int units = K / 256;
  const int groups = units * 64;
  const int group = (blockIdx.x * blockDim.x + threadIdx.x) / 8;
  const int sub = threadIdx.x % 8;
  if (group >= groups) {
    return;
  }
  const int unit = group / 64;
  const int remainder = group % 64;
  const int row = remainder / 8;
  const int s = remainder % 8;
  float values[4];
  float amax = 0.0f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int col = unit * 256 + s * 32 + sub * 4 + j;
    values[j] = row < M ? __bfloat162float(x[(size_t)row * K + col]) : 0.0f;
    amax = fmaxf(amax, fabsf(values[j]));
  }
#pragma unroll
  for (int shift = 1; shift < 8; shift <<= 1) {
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, shift, 8));
  }
  const float inverse = amax > 0.0f ? 127.0f / amax : 0.0f;
  int q[4];
  int qsum = 0;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    q[j] = amax > 0.0f ? static_cast<int>(roundf(values[j] * inverse)) : 0;
    qsum += q[j];
  }
#pragma unroll
  for (int shift = 1; shift < 8; shift <<= 1) {
    qsum += __shfl_xor_sync(0xffffffffu, qsum, shift, 8);
  }
  *reinterpret_cast<unsigned int*>(act + group * 32 + sub * 4) =
      static_cast<unsigned int>(q[0] & 0xff) |
      (static_cast<unsigned int>(q[1] & 0xff) << 8) |
      (static_cast<unsigned int>(q[2] & 0xff) << 16) |
      (static_cast<unsigned int>(q[3] & 0xff) << 24);
  float* scales = reinterpret_cast<float*>(act + units * 2048);
  float* sums = scales + groups;
  const float d = amax > 0.0f ? 1.0f / inverse : 0.0f;
  if (sub == 0) {
    scales[group] = d;
    sums[group] = d * static_cast<float>(qsum);
  }
}

// Sixteen-row variant for M in 9..16: packs two 8-row halves (rows 0..7 at
// act[0..units*2560), rows 8..15 at act[units*2560..2*units*2560)) using the
// exact same layout per half as mlx_node_kquant_quantize_few.
extern "C" __global__ void mlx_node_kquant_quantize_few_16(
    const nv_bfloat16* x, unsigned char* act, int M, int K) {
  const int units = K / 256;
  const int groups_per_half = units * 64;
  const int total_groups = groups_per_half * 2;
  const int group_all = (blockIdx.x * blockDim.x + threadIdx.x) / 8;
  const int sub = threadIdx.x % 8;
  if (group_all >= total_groups) {
    return;
  }
  const int half = group_all >= groups_per_half ? 1 : 0;
  const int group = half ? (group_all - groups_per_half) : group_all;
  const int unit = group / 64;
  const int remainder = group % 64;
  const int row = half * 8 + remainder / 8;
  const int s = remainder % 8;
  float values[4];
  float amax = 0.0f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int col = unit * 256 + s * 32 + sub * 4 + j;
    values[j] = row < M ? __bfloat162float(x[(size_t)row * K + col]) : 0.0f;
    amax = fmaxf(amax, fabsf(values[j]));
  }
#pragma unroll
  for (int shift = 1; shift < 8; shift <<= 1) {
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, shift, 8));
  }
  const float inverse = amax > 0.0f ? 127.0f / amax : 0.0f;
  int q[4];
  int qsum = 0;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    q[j] = amax > 0.0f ? static_cast<int>(roundf(values[j] * inverse)) : 0;
    qsum += q[j];
  }
#pragma unroll
  for (int shift = 1; shift < 8; shift <<= 1) {
    qsum += __shfl_xor_sync(0xffffffffu, qsum, shift, 8);
  }
  unsigned char* act_half = act + (size_t)half * units * 2560;
  *reinterpret_cast<unsigned int*>(act_half + group * 32 + sub * 4) =
      static_cast<unsigned int>(q[0] & 0xff) |
      (static_cast<unsigned int>(q[1] & 0xff) << 8) |
      (static_cast<unsigned int>(q[2] & 0xff) << 16) |
      (static_cast<unsigned int>(q[3] & 0xff) << 24);
  float* scales = reinterpret_cast<float*>(act_half + units * 2048);
  float* sums = scales + groups_per_half;
  const float d = amax > 0.0f ? 1.0f / inverse : 0.0f;
  if (sub == 0) {
    scales[group] = d;
    sums[group] = d * static_cast<float>(qsum);
  }
}

template <int mode>
__device__ __forceinline__ unsigned int kqf_unpack_four(unsigned int u) {
  if constexpr (mode == 0) {
    return (u & 0x0000000fu) | ((u & 0x000000f0u) << 4) |
        ((u & 0x00000f00u) << 8) | ((u & 0x0000f000u) << 12);
  } else if constexpr (mode == 3 || mode == 5) {
    // Decode four IQ4_NL/IQ4_XS indices with byte permutes.
    const unsigned int low = __byte_perm(0xbfad9881u, 0xf6eaddcfu, u);
    const unsigned int high = __byte_perm(0x26190d01u, 0x71594535u, u);
    return __byte_perm(low, high, 0x3210u | ((u & 0x8888u) >> 1));
  } else if constexpr (mode == 1) {
    return (u & 0x0000001fu) | ((u & 0x000003e0u) << 3) |
        ((u & 0x00007c00u) << 6) | ((u & 0x000f8000u) << 9);
  } else if constexpr (mode == 2) {
    const unsigned int v = (u & 0x0000003fu) | ((u & 0x00000fc0u) << 2) |
        ((u & 0x0003f000u) << 4) | ((u & 0x00fc0000u) << 6);
    const unsigned int neg = (~v) & 0x20202020u;
    return (v & 0x1f1f1f1fu) | neg | (neg << 1) | (neg << 2);
  } else {
    const unsigned int v = (u & 0x00000007u) | ((u & 0x00000038u) << 5) |
        ((u & 0x000001c0u) << 10) | ((u & 0x00000e00u) << 15);
    const unsigned int neg = (~v) & 0x04040404u;
    return (v & 0x03030303u) | (neg * 0x3fu);
  }
}

template <int mode>
__device__ __forceinline__ void kqf_load_pair(
    const unsigned int* seg, int t, unsigned int& out0, unsigned int& out1) {
  if constexpr (mode == 0 || mode == 3 || mode == 5) {
    const unsigned short* seg16 = reinterpret_cast<const unsigned short*>(seg);
    out0 = kqf_unpack_four<mode>(seg16[t]);
    out1 = kqf_unpack_four<mode>(seg16[4 + t]);
  } else if constexpr (mode == 1) {
    // Q5_K: 5 uint32 words (160 bits) per 32-weight segment.
    const int bit0 = t * 20;
    const int w0 = bit0 >> 5;
    const int sh0 = bit0 & 31;
    unsigned int u0 = seg[w0];
    if (sh0 > 12) {
      u0 = __funnelshift_r(u0, seg[w0 + 1], sh0);
    } else {
      u0 >>= sh0;
    }

    const int bit1 = 80 + t * 20;
    const int w1 = bit1 >> 5;
    const int sh1 = bit1 & 31;
    unsigned int u1 = seg[w1];
    if (sh1 > 12) {
      u1 = __funnelshift_r(u1, seg[w1 + 1], sh1);
    } else {
      u1 >>= sh1;
    }
    out0 = kqf_unpack_four<1>(u0);
    out1 = kqf_unpack_four<1>(u1);
  }
}

template <int mode>
__device__ __forceinline__ unsigned int kqf_load_k16(
    const unsigned int* seg, int kh, int t) {
  if constexpr (mode == 2) {
    // Q6_K: 6 uint32 words (192 bits) per 32-weight segment.
    const int bit = kh * 96 + t * 24;
    const int wi = bit >> 5;
    const int sh = bit & 31;
    unsigned int u = seg[wi];
    if (sh > 8) {
      u = __funnelshift_r(u, seg[wi + 1], sh);
    } else {
      u >>= sh;
    }
    return kqf_unpack_four<2>(u);
  } else {
    // Q3_K: 3 uint32 words (96 bits) per 32-weight segment.
    const int bit = kh * 48 + t * 12;
    const int wi = bit >> 5;
    const int sh = bit & 31;
    unsigned int u = seg[wi];
    if (sh > 20) {
      u = __funnelshift_r(u, seg[wi + 1], sh);
    } else {
      u >>= sh;
    }
    return kqf_unpack_four<4>(u);
  }
}

__device__ __forceinline__ void kqf_cp_async16(void* smem_ptr, const void* glob_ptr) {
  unsigned int smem_addr = static_cast<unsigned int>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "cp.async.cg.shared.global [%0], [%1], 16;\n"
      :: "r"(smem_addr), "l"(glob_ptr) : "memory");
}

__device__ __forceinline__ void kqf_cp_commit() {
  asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template <int N_WAIT>
__device__ __forceinline__ void kqf_cp_wait() {
  asm volatile("cp.async.wait_group %0;\n" :: "n"(N_WAIT) : "memory");
}

__device__ __forceinline__ void kqf_mma_k16(
    const unsigned int a0, const unsigned int a1, const unsigned int b,
    int& c0, int& c1, int& c2, int& c3) {
  asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
      "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
      : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
      : "r"(a0), "r"(a1), "r"(b));
}

__device__ __forceinline__ void kqf_mma(
    const unsigned int a0, const unsigned int a1, const unsigned int a2,
    const unsigned int a3, const unsigned int b0, const unsigned int b1,
    int& c0, int& c1, int& c2, int& c3) {
  asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
      : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
)cuda"
R"cuda(
// Warps split 256-value units. Each warp evaluates a 16-row by 8-row (M16=false)
// or 16-row by 16-row (M16=true, reusing unpacked weight registers across both
// 8-row activation halves with zero extra weight DRAM traffic) INT8 MMA tile.
// For tiled=true (Tiled64), weights stream into 2-stage double-buffered SMEM
// via coalesced 128-bit L2-bypassing cp.async.cg.shared.global DMA transfers.
// Modes: 0=Q4_K, 1=Q5_K, 2=Q6_K, 3=IQ4_XS, 4=Q3_K, 5=IQ4_NL.
template <int mode, bool tiled = false, int NUM_WARPS = 4, bool M16 = false>
__global__ __launch_bounds__(NUM_WARPS * 32, NUM_WARPS >= 12 ? 1 : 2)
void mlx_node_kquant_matmul_few(
    const unsigned int* __restrict__ w,
    const unsigned char* __restrict__ scales,
    const unsigned short* __restrict__ biases,
    const unsigned char* __restrict__ act,
    nv_bfloat16* __restrict__ out, int M, int N, int K) {
  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const int t = lane % 4;
  const int g = lane / 4;
  const int row0 = blockIdx.x * 16;
  const int rows[2] = {min(row0 + g, N - 1), min(row0 + g + 8, N - 1)};
  const int units = K / 256;
  const int units32 = K / 32;
  constexpr int bits = mode == 1 ? 5 : (mode == 2 ? 6 : (mode == 4 ? 3 : 4));
  constexpr int V4 = bits * 2; // 16-byte vectors per 256-weight unit (8, 10, 12, 6)
  constexpr int SEG_V4 = bits * 4; // 16-byte vectors per 16-row Tiled64 segment (12..24)
  constexpr int sc_unit_bytes = mode == 3 ? 8 : 16;
  constexpr int b_unit_entries = (mode == 0 || mode == 1) ? 2 : 1;
  constexpr int M_COLS = M16 ? 16 : 8;

  // Shared memory:
  // - tiled=true double-buffers one 16-row segment via cp.async.cg (<= 6 KB for 8 warps).
  // - tiled=false pads row-major tile stride to V4+1 to avoid 8-way bank conflicts.
  union SharedBuf {
    uint4 t64_seg[2][NUM_WARPS][tiled ? SEG_V4 : 1];
    uint4 w_tile[NUM_WARPS][tiled ? 1 : 16][tiled ? 1 : (V4 + 1)];
    float partials[NUM_WARPS * 16 * M_COLS];
  };
  __shared__ SharedBuf smem;

  const int tile_id = row0 / 64;
  const int base_in_tile = row0 % 64;
  const int tile_ids[2] = {rows[0] / 64, rows[1] / 64};
  const int in_tile[2] = {rows[0] % 64, rows[1] % 64};
  const float* act_scales = reinterpret_cast<const float*>(act + units * 2048);
  const float* act_sums = act_scales + units * 64;
  const int act_half_u32 = units * 640; // 2560 bytes / 4
  const size_t row_w_stride_u4 = (size_t)units * V4;
  const uint4* w_u4 = reinterpret_cast<const uint4*>(w);

  float total_lo[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float total_hi[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  for (int unit = warp; unit < units; unit += NUM_WARPS) {
    if constexpr (!tiled) {
      // Coalesced 128-bit streaming load of 16 rows for this unit into shared memory.
      const size_t unit_off_u4 = (size_t)unit * V4;
      if constexpr (V4 == 8) {
        const int r_in = lane >> 3;
        const int c_in = lane & 7;
#pragma unroll
        for (int rb = 0; rb < 16; rb += 4) {
          const int r = rb + r_in;
          const int rr = min(row0 + r, N - 1);
          smem.w_tile[warp][r][c_in] =
              w_u4[(size_t)rr * row_w_stride_u4 + unit_off_u4 + c_in];
        }
      } else {
        constexpr int total_v4 = 16 * V4;
#pragma unroll
        for (int idx = lane; idx < total_v4; idx += 32) {
          const int r = idx / V4;
          const int c = idx % V4;
          const int rr = min(row0 + r, N - 1);
          smem.w_tile[warp][r][c] =
              w_u4[(size_t)rr * row_w_stride_u4 + unit_off_u4 + c];
        }
      }
      __syncwarp();
    }

    float dot_lo[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float dot_hi[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    uint4 sc_vec[2];
    uint2 sc_vec8[2];
    if constexpr (mode != 5) {
      const unsigned char* sc_base[2] = {
          scales + (tiled
              ? (((size_t)tile_ids[0] * units + unit) * 64 + in_tile[0]) * sc_unit_bytes
              : ((size_t)rows[0] * units + unit) * sc_unit_bytes),
          scales + (tiled
              ? (((size_t)tile_ids[1] * units + unit) * 64 + in_tile[1]) * sc_unit_bytes
              : ((size_t)rows[1] * units + unit) * sc_unit_bytes)};
      if constexpr (sc_unit_bytes == 16) {
        sc_vec[0] = *reinterpret_cast<const uint4*>(sc_base[0]);
        sc_vec[1] = *reinterpret_cast<const uint4*>(sc_base[1]);
      } else {
        sc_vec8[0] = *reinterpret_cast<const uint2*>(sc_base[0]);
        sc_vec8[1] = *reinterpret_cast<const uint2*>(sc_base[1]);
      }
    }

    const unsigned int* w_base[2] = {
        tiled ? nullptr : reinterpret_cast<const unsigned int*>(smem.w_tile[warp][g]),
        tiled ? nullptr : reinterpret_cast<const unsigned int*>(smem.w_tile[warp][g + 8])};

    if constexpr (tiled) {
      // Prime cp.async stage 0 for segment s = 0.
      if (lane < SEG_V4) {
        const uint4* seg_u4 = reinterpret_cast<const uint4*>(
            w + (((size_t)tile_id * units32 + unit * 8) * 64 + base_in_tile) * bits);
        kqf_cp_async16(&smem.t64_seg[0][warp][lane], seg_u4 + lane);
      }
      kqf_cp_commit();
    }

#pragma unroll
    for (int s = 0; s < 8; ++s) {
      const unsigned int* seg0;
      const unsigned int* seg1;
      if constexpr (tiled) {
        const int cur = s & 1;
        const int nxt = cur ^ 1;
        if (s + 1 < 8) {
          if (lane < SEG_V4) {
            const uint4* nxt_u4 = reinterpret_cast<const uint4*>(
                w + (((size_t)tile_id * units32 + unit * 8 + s + 1) * 64 + base_in_tile) * bits);
            kqf_cp_async16(&smem.t64_seg[nxt][warp][lane], nxt_u4 + lane);
          }
          kqf_cp_commit();
          kqf_cp_wait<1>();
        } else {
          kqf_cp_wait<0>();
        }
        __syncwarp();
        const unsigned int* seg_smem =
            reinterpret_cast<const unsigned int*>(smem.t64_seg[cur][warp]);
        seg0 = seg_smem + g * bits;
        seg1 = seg_smem + (g + 8) * bits;
      } else {
        seg0 = w_base[0] + s * bits;
        seg1 = w_base[1] + s * bits;
      }
      const unsigned int* q32_lo =
          reinterpret_cast<const unsigned int*>(act + (unit * 64 + g * 8 + s) * 32);
      const int sc_idx0 = unit * 64 + (2 * t) * 8 + s;
      const int sc_idx1 = unit * 64 + (2 * t + 1) * 8 + s;
      if constexpr (mode == 2 || mode == 4) {
        const unsigned int sc_w0 = (s < 2) ? sc_vec[0].x : (s < 4 ? sc_vec[0].y : (s < 6 ? sc_vec[0].z : sc_vec[0].w));
        const unsigned int sc_w1 = (s < 2) ? sc_vec[1].x : (s < 4 ? sc_vec[1].y : (s < 6 ? sc_vec[1].z : sc_vec[1].w));
        const int shift_base = (s & 1) * 16;
#pragma unroll
        for (int kh = 0; kh < 2; ++kh) {
          const unsigned int a0 = kqf_load_k16<mode>(seg0, kh, t);
          const unsigned int a1 = kqf_load_k16<mode>(seg1, kh, t);
          const int ws0 = static_cast<signed char>((sc_w0 >> (shift_base + kh * 8)) & 0xff);
          const int ws1 = static_cast<signed char>((sc_w1 >> (shift_base + kh * 8)) & 0xff);
          {
            int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
            kqf_mma_k16(a0, a1, q32_lo[kh * 4 + t], c0, c1, c2, c3);
            const float ad0 = act_scales[sc_idx0];
            const float ad1 = act_scales[sc_idx1];
            dot_lo[0] += ad0 * static_cast<float>(c0 * ws0);
            dot_lo[1] += ad1 * static_cast<float>(c1 * ws0);
            dot_lo[2] += ad0 * static_cast<float>(c2 * ws1);
            dot_lo[3] += ad1 * static_cast<float>(c3 * ws1);
          }
          if constexpr (M16) {
            int h0 = 0, h1 = 0, h2 = 0, h3 = 0;
            kqf_mma_k16(a0, a1, q32_lo[act_half_u32 + kh * 4 + t], h0, h1, h2, h3);
            const float ad0 = act_scales[sc_idx0 + act_half_u32];
            const float ad1 = act_scales[sc_idx1 + act_half_u32];
            dot_hi[0] += ad0 * static_cast<float>(h0 * ws0);
            dot_hi[1] += ad1 * static_cast<float>(h1 * ws0);
            dot_hi[2] += ad0 * static_cast<float>(h2 * ws1);
            dot_hi[3] += ad1 * static_cast<float>(h3 * ws1);
          }
        }
      } else {
        unsigned int a0, a1, a2, a3;
        kqf_load_pair<mode>(seg0, t, a0, a2);
        kqf_load_pair<mode>(seg1, t, a1, a3);
        int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
        kqf_mma(a0, a1, a2, a3, q32_lo[t], q32_lo[4 + t], c0, c1, c2, c3);
        const float ad0_lo = act_scales[sc_idx0];
        const float ad1_lo = act_scales[sc_idx1];
        if constexpr (mode == 5) {
          const size_t idx0 = tiled
              ? (((size_t)tile_ids[0] * units32 + unit * 8 + s) * 64 + in_tile[0])
              : ((size_t)rows[0] * units32 + unit * 8 + s);
          const size_t idx1 = tiled
              ? (((size_t)tile_ids[1] * units32 + unit * 8 + s) * 64 + in_tile[1])
              : ((size_t)rows[1] * units32 + unit * 8 + s);
          const float ds0 = __half2float(__ushort_as_half(biases[idx0])) *
              static_cast<float>(static_cast<signed char>(scales[idx0]));
          const float ds1 = __half2float(__ushort_as_half(biases[idx1])) *
              static_cast<float>(static_cast<signed char>(scales[idx1]));
          total_lo[0] += ad0_lo * ds0 * static_cast<float>(c0);
          total_lo[1] += ad1_lo * ds0 * static_cast<float>(c1);
          total_lo[2] += ad0_lo * ds1 * static_cast<float>(c2);
          total_lo[3] += ad1_lo * ds1 * static_cast<float>(c3);
          if constexpr (M16) {
            const unsigned int* q32_hi = q32_lo + act_half_u32;
            int h0 = 0, h1 = 0, h2 = 0, h3 = 0;
            kqf_mma(a0, a1, a2, a3, q32_hi[t], q32_hi[4 + t], h0, h1, h2, h3);
            const float ad0_hi = act_scales[sc_idx0 + act_half_u32];
            const float ad1_hi = act_scales[sc_idx1 + act_half_u32];
            total_hi[0] += ad0_hi * ds0 * static_cast<float>(h0);
            total_hi[1] += ad1_hi * ds0 * static_cast<float>(h1);
            total_hi[2] += ad0_hi * ds1 * static_cast<float>(h2);
            total_hi[3] += ad1_hi * ds1 * static_cast<float>(h3);
          }
        } else if constexpr (mode == 3) {
          const unsigned int sw0 = s < 4 ? sc_vec8[0].x : sc_vec8[0].y;
          const unsigned int sw1 = s < 4 ? sc_vec8[1].x : sc_vec8[1].y;
          const int sh = (s & 3) * 8;
          const int ws0 = static_cast<signed char>((sw0 >> sh) & 0xff);
          const int ws1 = static_cast<signed char>((sw1 >> sh) & 0xff);
          dot_lo[0] += ad0_lo * static_cast<float>(c0 * ws0);
          dot_lo[1] += ad1_lo * static_cast<float>(c1 * ws0);
          dot_lo[2] += ad0_lo * static_cast<float>(c2 * ws1);
          dot_lo[3] += ad1_lo * static_cast<float>(c3 * ws1);
          if constexpr (M16) {
            const unsigned int* q32_hi = q32_lo + act_half_u32;
            int h0 = 0, h1 = 0, h2 = 0, h3 = 0;
            kqf_mma(a0, a1, a2, a3, q32_hi[t], q32_hi[4 + t], h0, h1, h2, h3);
            const float ad0_hi = act_scales[sc_idx0 + act_half_u32];
            const float ad1_hi = act_scales[sc_idx1 + act_half_u32];
            dot_hi[0] += ad0_hi * static_cast<float>(h0 * ws0);
            dot_hi[1] += ad1_hi * static_cast<float>(h1 * ws0);
            dot_hi[2] += ad0_hi * static_cast<float>(h2 * ws1);
            dot_hi[3] += ad1_hi * static_cast<float>(h3 * ws1);
          }
        } else {
          // Q4_K / Q5_K: compute dot products here; min_sum is deferred after the
          // MMA loop to free 8 live float registers and eliminate M16 spills.
          const unsigned int sw0 = (s < 2) ? sc_vec[0].x : (s < 4 ? sc_vec[0].y : (s < 6 ? sc_vec[0].z : sc_vec[0].w));
          const unsigned int sw1 = (s < 2) ? sc_vec[1].x : (s < 4 ? sc_vec[1].y : (s < 6 ? sc_vec[1].z : sc_vec[1].w));
          const int sh = (s & 1) * 16;
          const int ws0 = (sw0 >> sh) & 0xff;
          const int ws1 = (sw1 >> sh) & 0xff;
          dot_lo[0] += ad0_lo * static_cast<float>(c0 * ws0);
          dot_lo[1] += ad1_lo * static_cast<float>(c1 * ws0);
          dot_lo[2] += ad0_lo * static_cast<float>(c2 * ws1);
          dot_lo[3] += ad1_lo * static_cast<float>(c3 * ws1);
          if constexpr (M16) {
            const unsigned int* q32_hi = q32_lo + act_half_u32;
            int h0 = 0, h1 = 0, h2 = 0, h3 = 0;
            kqf_mma(a0, a1, a2, a3, q32_hi[t], q32_hi[4 + t], h0, h1, h2, h3);
            const float ad0_hi = act_scales[sc_idx0 + act_half_u32];
            const float ad1_hi = act_scales[sc_idx1 + act_half_u32];
            dot_hi[0] += ad0_hi * static_cast<float>(h0 * ws0);
            dot_hi[1] += ad1_hi * static_cast<float>(h1 * ws0);
            dot_hi[2] += ad0_hi * static_cast<float>(h2 * ws1);
            dot_hi[3] += ad1_hi * static_cast<float>(h3 * ws1);
          }
        }
      }
    }
)cuda"
R"cuda(
    if constexpr (mode != 5) {
      float min_sum_lo[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      float min_sum_hi[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      if constexpr (mode == 0 || mode == 1) {
#pragma unroll
        for (int s = 0; s < 8; ++s) {
          const int sc_idx0 = unit * 64 + (2 * t) * 8 + s;
          const int sc_idx1 = unit * 64 + (2 * t + 1) * 8 + s;
          const unsigned int sw0 = (s < 2) ? sc_vec[0].x : (s < 4 ? sc_vec[0].y : (s < 6 ? sc_vec[0].z : sc_vec[0].w));
          const unsigned int sw1 = (s < 2) ? sc_vec[1].x : (s < 4 ? sc_vec[1].y : (s < 6 ? sc_vec[1].z : sc_vec[1].w));
          const int sh = (s & 1) * 16 + 8;
          const float wm0 = static_cast<float>((sw0 >> sh) & 0xff);
          const float wm1 = static_cast<float>((sw1 >> sh) & 0xff);
          const float as0_lo = act_sums[sc_idx0];
          const float as1_lo = act_sums[sc_idx1];
          min_sum_lo[0] += as0_lo * wm0;
          min_sum_lo[1] += as1_lo * wm0;
          min_sum_lo[2] += as0_lo * wm1;
          min_sum_lo[3] += as1_lo * wm1;
          if constexpr (M16) {
            const float as0_hi = act_sums[sc_idx0 + act_half_u32];
            const float as1_hi = act_sums[sc_idx1 + act_half_u32];
            min_sum_hi[0] += as0_hi * wm0;
            min_sum_hi[1] += as1_hi * wm0;
            min_sum_hi[2] += as0_hi * wm1;
            min_sum_hi[3] += as1_hi * wm1;
          }
        }
      }
      for (int r = 0; r < 2; ++r) {
        const size_t bias_index = tiled
            ? (((size_t)tile_ids[r] * units + unit) * 64 + in_tile[r]) * b_unit_entries
            : ((size_t)rows[r] * units + unit) * b_unit_entries;
        const float d = __half2float(__ushort_as_half(biases[bias_index]));
        if constexpr (mode == 0 || mode == 1) {
          const float dmin = __half2float(__ushort_as_half(biases[bias_index + 1]));
          total_lo[r * 2 + 0] += d * dot_lo[r * 2 + 0] - dmin * min_sum_lo[r * 2 + 0];
          total_lo[r * 2 + 1] += d * dot_lo[r * 2 + 1] - dmin * min_sum_lo[r * 2 + 1];
          if constexpr (M16) {
            total_hi[r * 2 + 0] += d * dot_hi[r * 2 + 0] - dmin * min_sum_hi[r * 2 + 0];
            total_hi[r * 2 + 1] += d * dot_hi[r * 2 + 1] - dmin * min_sum_hi[r * 2 + 1];
          }
        } else {
          total_lo[r * 2 + 0] += d * dot_lo[r * 2 + 0];
          total_lo[r * 2 + 1] += d * dot_lo[r * 2 + 1];
          if constexpr (M16) {
            total_hi[r * 2 + 0] += d * dot_hi[r * 2 + 0];
            total_hi[r * 2 + 1] += d * dot_hi[r * 2 + 1];
          }
        }
      }
    }
  }

  __syncthreads();
  const int columns[2] = {2 * t, 2 * t + 1};
  for (int r = 0; r < 2; ++r) {
    const int row = g + r * 8;
    for (int half = 0; half < 2; ++half) {
      smem.partials[(warp * 16 + row) * M_COLS + columns[half]] = total_lo[r * 2 + half];
      if constexpr (M16) {
        smem.partials[(warp * 16 + row) * M_COLS + 8 + columns[half]] = total_hi[r * 2 + half];
      }
    }
  }
  __syncthreads();
  const int flat = warp * 32 + lane;
  constexpr int total_out = 16 * M_COLS;
#pragma unroll
  for (int idx = flat; idx < total_out; idx += NUM_WARPS * 32) {
    // Coalesced output stores: 16 adjacent threads write contiguous rows of row0..row0+15.
    const int column = idx >> 4;
    const int row = idx & 15;
    if (row0 + row < N && column < M) {
      float value = 0.0f;
#pragma unroll
      for (int ww = 0; ww < NUM_WARPS; ++ww) {
        value += smem.partials[(ww * 16 + row) * M_COLS + column];
      }
      out[(size_t)column * N + row0 + row] = __float2bfloat16(value);
    }
  }
}
)cuda"