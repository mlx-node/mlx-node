R"cuda(
#include <cuda_bf16.h>
#include <cuda_fp16.h>

// ============================================================================
// Inkling-2-Mini Core CUDA Kernels (sm_80+ / sm_121a, 0 B register spills)
//   1. ink2_block_attn_res_norm_cuda  (Block AttnRes + optional sublayer RMSNorm)
//   2. ink2_loop_{down_fused,up48_mma,up_blend_f32}_cuda + ink2_rms_norm_f32_to_bf16_cuda
//   3. ink2_kv_conv_prep_store_cuda   (Q/R L2-clip + K/V ShortConv + KV cache store)
//   4. ink2_sconv_k4_residual_cuda    (Static k=4 ShortConv + optional residual)
//   5. ink2_small_bf16_proj_cuda      (High-occupancy small-N BF16 projection)
//   6. ink2_commit_short_conv_tapes_cuda (Single-launch batched ShortConv tape commit)
//   7. ink2_embed_rmsnorm_cuda        (Fused BF16/SymQ8/AffineQ8 embed + F32 RMSNorm)
// ============================================================================

__device__ __forceinline__ uint4 ink2_ld_cs_u4(const uint4* ptr) {
  uint4 v;
  asm("ld.global.cs.v4.u32 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
      : "l"(ptr));
  return v;
}

__device__ __forceinline__ void ink2_mma_m16n8k16_bf16(
    float& d0, float& d1, float& d2, float& d3,
    unsigned int a0, unsigned int a1, unsigned int a2, unsigned int a3,
    unsigned int b0, unsigned int b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

template <int SB_DTYPE>
__device__ __forceinline__ float ink2_load_sb1(const void* ptr, size_t idx) {
  if constexpr (SB_DTYPE == 0) {
    return reinterpret_cast<const float*>(ptr)[idx];
  } else if constexpr (SB_DTYPE == 1) {
    return __bfloat162float(reinterpret_cast<const nv_bfloat16*>(ptr)[idx]);
  } else {
    return __half2float(reinterpret_cast<const __half*>(ptr)[idx]);
  }
}

__device__ __forceinline__ float ink2_warp_sum(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    v += __shfl_xor_sync(0xffffffffu, v, off);
  }
  return v;
}

__device__ __forceinline__ float ink2_warp_max(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
  }
  return v;
}

__device__ __forceinline__ float ink2_load_f1(
    const char* ptr, size_t idx, bool is_f32) {
  if (is_f32) {
    return reinterpret_cast<const float*>(ptr)[idx];
  }
  return __bfloat162float(reinterpret_cast<const nv_bfloat16*>(ptr)[idx]);
}

__device__ __forceinline__ float4 ink2_load_f4(
    const char* ptr, size_t idx4, bool is_f32) {
  if (is_f32) {
    return reinterpret_cast<const float4*>(ptr)[idx4];
  }
  const uint2 raw = reinterpret_cast<const uint2*>(ptr)[idx4];
  const float2 lo =
      __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.x));
  const float2 hi =
      __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.y));
  return make_float4(lo.x, lo.y, hi.x, hi.y);
}

template <bool W_BF16>
__device__ __forceinline__ float4 ink2_load_w4(const void* ptr, size_t idx4) {
  if constexpr (W_BF16) {
    const uint2 raw = reinterpret_cast<const uint2*>(ptr)[idx4];
    const float2 lo =
        __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.x));
    const float2 hi =
        __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.y));
    return make_float4(lo.x, lo.y, hi.x, hi.y);
  } else {
    return reinterpret_cast<const float4*>(ptr)[idx4];
  }
}

__device__ __forceinline__ void ink2_store_bf16x4(
    nv_bfloat16* out, size_t idx4, float4 v) {
  uint2 raw;
  const __nv_bfloat162 lo = __floats2bfloat162_rn(v.x, v.y);
  const __nv_bfloat162 hi = __floats2bfloat162_rn(v.z, v.w);
  raw.x = *reinterpret_cast<const unsigned int*>(&lo);
  raw.y = *reinterpret_cast<const unsigned int*>(&hi);
  reinterpret_cast<uint2*>(out)[idx4] = raw;
}

__device__ __forceinline__ const char* ink2_pick_src(
    int n,
    const char* s0, const char* s1, const char* s2, const char* s3,
    const char* s4, const char* s5, const char* s6, const char* s7) {
  switch (n) {
    case 0: return s0;
    case 1: return s1;
    case 2: return s2;
    case 3: return s3;
    case 4: return s4;
    case 5: return s5;
    case 6: return s6;
    default: return s7;
  }
}

// ============================================================================
// 1. Block AttnRes over N in 1..=8 sources + optional fused sublayer RMSNorm
// Grid: dim3(T, Q, 1), Block: dim3(256, 1, 1)
// ============================================================================
template <bool HAS_NORM, bool VEC4>
__global__ __launch_bounds__(256, 4) void ink2_block_attn_res_norm_cuda(
    const char* __restrict__ s0,
    const char* __restrict__ s1,
    const char* __restrict__ s2,
    const char* __restrict__ s3,
    const char* __restrict__ s4,
    const char* __restrict__ s5,
    const char* __restrict__ s6,
    const char* __restrict__ s7,
    const float* __restrict__ queries,
    const float* __restrict__ norm_weights,
    void* __restrict__ out,
    int T,
    int D,
    int N,
    unsigned int src_f32_mask,
    int norm_w_bcast,
    float eps) {
  const int t = static_cast<int>(blockIdx.x);
  const int q = static_cast<int>(blockIdx.y);
  if (t >= T) {
    return;
  }
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;

  __shared__ float s_sq[8][8];
  __shared__ float s_dot[8][8];
  __shared__ float s_inv_rms[8];
  __shared__ float s_p[8];

  if constexpr (VEC4) {
    const int D4 = D >> 2;
    const size_t row_off4 = static_cast<size_t>(t) * static_cast<size_t>(D4);
    const float4* q4 =
        reinterpret_cast<const float4*>(queries) +
        static_cast<size_t>(q) * static_cast<size_t>(D4);

    for (int n = 0; n < N; ++n) {
      const char* ptr = ink2_pick_src(n, s0, s1, s2, s3, s4, s5, s6, s7);
      const bool is_f32 = ((src_f32_mask >> n) & 1u) != 0u;
      float sq = 0.0f;
      float dot = 0.0f;
      for (int i4 = tid; i4 < D4; i4 += 256) {
        const float4 v = ink2_load_f4(ptr, row_off4 + i4, is_f32);
        sq += (v.x * v.x + v.y * v.y) + (v.z * v.z + v.w * v.w);
        if (N > 1) {
          const float4 qv = q4[i4];
          dot += (qv.x * v.x + qv.y * v.y) + (qv.z * v.z + qv.w * v.w);
        }
      }
      sq = ink2_warp_sum(sq);
      if (N > 1) {
        dot = ink2_warp_sum(dot);
      }
      if (lane == 0) {
        s_sq[n][warp] = sq;
        if (N > 1) {
          s_dot[n][warp] = dot;
        }
      }
    }
    __syncthreads();

    if (warp == 0 && lane == 0) {
      const float inv_d = 1.0f / static_cast<float>(D);
      if (N == 1) {
        float total_sq = 0.0f;
#pragma unroll
        for (int w = 0; w < 8; ++w) {
          total_sq += s_sq[0][w];
        }
        s_inv_rms[0] = rsqrtf(total_sq * inv_d + eps);
        s_p[0] = 1.0f;
      } else {
        float max_s = -__int_as_float(0x7f800000);
        for (int n = 0; n < N; ++n) {
          float total_sq = 0.0f;
          float total_dot = 0.0f;
#pragma unroll
          for (int w = 0; w < 8; ++w) {
            total_sq += s_sq[n][w];
            total_dot += s_dot[n][w];
          }
          const float inv_r = rsqrtf(total_sq * inv_d + eps);
          s_inv_rms[n] = inv_r;
          const float sc = total_dot * inv_r;
          s_p[n] = sc;
          max_s = fmaxf(max_s, sc);
        }
        float denom = 0.0f;
        for (int n = 0; n < N; ++n) {
          const float e = expf(s_p[n] - max_s);
          s_p[n] = e;
          denom += e;
        }
        const float inv_denom = 1.0f / denom;
        for (int n = 0; n < N; ++n) {
          s_p[n] *= inv_denom;
        }
      }
    }
    __syncthreads();

    const size_t out_row_off4 =
        (static_cast<size_t>(q) * static_cast<size_t>(T) + static_cast<size_t>(t)) *
        static_cast<size_t>(D4);

    if constexpr (!HAS_NORM) {
      float4* out4 = reinterpret_cast<float4*>(out) + out_row_off4;
      for (int i4 = tid; i4 < D4; i4 += 256) {
        float4 y = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (int n = 0; n < N; ++n) {
          const char* ptr = ink2_pick_src(n, s0, s1, s2, s3, s4, s5, s6, s7);
          const bool is_f32 = ((src_f32_mask >> n) & 1u) != 0u;
          const float4 v = ink2_load_f4(ptr, row_off4 + i4, is_f32);
          const float ir = s_inv_rms[n];
          const float pr = s_p[n];
          y.x += pr * (v.x * ir);
          y.y += pr * (v.y * ir);
          y.z += pr * (v.z * ir);
          y.w += pr * (v.w * ir);
        }
        out4[i4] = y;
      }
      return;
    }

    // HAS_NORM == true: cache up to 3 float4s/thread (covers D <= 3072, incl. D=2304)
)cuda"
R"cuda(
    float4 y0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 y1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 y2 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    const bool fits_regs = (D4 <= 768);
    float y_sq = 0.0f;

    int step = 0;
    for (int i4 = tid; i4 < D4; i4 += 256, ++step) {
      float4 y = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      for (int n = 0; n < N; ++n) {
        const char* ptr = ink2_pick_src(n, s0, s1, s2, s3, s4, s5, s6, s7);
        const bool is_f32 = ((src_f32_mask >> n) & 1u) != 0u;
        const float4 v = ink2_load_f4(ptr, row_off4 + i4, is_f32);
        const float ir = s_inv_rms[n];
        const float pr = s_p[n];
        y.x += pr * (v.x * ir);
        y.y += pr * (v.y * ir);
        y.z += pr * (v.z * ir);
        y.w += pr * (v.w * ir);
      }
      y_sq += (y.x * y.x + y.y * y.y) + (y.z * y.z + y.w * y.w);
      if (fits_regs) {
        if (step == 0) y0 = y;
        else if (step == 1) y1 = y;
        else y2 = y;
      }
    }

    y_sq = ink2_warp_sum(y_sq);
    if (lane == 0) {
      s_sq[0][warp] = y_sq;
    }
    __syncthreads();
    if (warp == 0 && lane == 0) {
      float total = 0.0f;
#pragma unroll
      for (int w = 0; w < 8; ++w) {
        total += s_sq[0][w];
      }
      s_sq[1][0] = rsqrtf(total / static_cast<float>(D) + eps);
    }
    __syncthreads();
    const float inv_rms_y = s_sq[1][0];

    const float4* w4 = reinterpret_cast<const float4*>(norm_weights) +
        (norm_w_bcast ? 0 : static_cast<size_t>(q) * static_cast<size_t>(D4));
    nv_bfloat16* out_bf = reinterpret_cast<nv_bfloat16*>(out);

    step = 0;
    for (int i4 = tid; i4 < D4; i4 += 256, ++step) {
      float4 y;
      if (fits_regs) {
        y = (step == 0) ? y0 : ((step == 1) ? y1 : y2);
      } else {
        y = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (int n = 0; n < N; ++n) {
          const char* ptr = ink2_pick_src(n, s0, s1, s2, s3, s4, s5, s6, s7);
          const bool is_f32 = ((src_f32_mask >> n) & 1u) != 0u;
          const float4 v = ink2_load_f4(ptr, row_off4 + i4, is_f32);
          const float ir = s_inv_rms[n];
          const float pr = s_p[n];
          y.x += pr * (v.x * ir);
          y.y += pr * (v.y * ir);
          y.z += pr * (v.z * ir);
          y.w += pr * (v.w * ir);
        }
      }
      const float4 wv = w4[i4];
      const float4 res = make_float4(
          (y.x * inv_rms_y) * wv.x,
          (y.y * inv_rms_y) * wv.y,
          (y.z * inv_rms_y) * wv.z,
          (y.w * inv_rms_y) * wv.w);
      ink2_store_bf16x4(out_bf, out_row_off4 + i4, res);
    }
  } else {
    // General scalar path (D % 4 != 0)
    const size_t row_off = static_cast<size_t>(t) * static_cast<size_t>(D);
    const float* q_row = queries + static_cast<size_t>(q) * static_cast<size_t>(D);

    for (int n = 0; n < N; ++n) {
      const char* ptr = ink2_pick_src(n, s0, s1, s2, s3, s4, s5, s6, s7);
      const bool is_f32 = ((src_f32_mask >> n) & 1u) != 0u;
      float sq = 0.0f;
      float dot = 0.0f;
      for (int i = tid; i < D; i += 256) {
        const float v = ink2_load_f1(ptr, row_off + i, is_f32);
        sq += v * v;
        if (N > 1) {
          dot += q_row[i] * v;
        }
      }
      sq = ink2_warp_sum(sq);
      if (N > 1) {
        dot = ink2_warp_sum(dot);
      }
      if (lane == 0) {
        s_sq[n][warp] = sq;
        if (N > 1) {
          s_dot[n][warp] = dot;
        }
      }
    }
    __syncthreads();

    if (warp == 0 && lane == 0) {
      const float inv_d = 1.0f / static_cast<float>(D);
      if (N == 1) {
        float total_sq = 0.0f;
#pragma unroll
        for (int w = 0; w < 8; ++w) {
          total_sq += s_sq[0][w];
        }
        s_inv_rms[0] = rsqrtf(total_sq * inv_d + eps);
        s_p[0] = 1.0f;
      } else {
        float max_s = -__int_as_float(0x7f800000);
        for (int n = 0; n < N; ++n) {
          float total_sq = 0.0f;
          float total_dot = 0.0f;
#pragma unroll
          for (int w = 0; w < 8; ++w) {
            total_sq += s_sq[n][w];
            total_dot += s_dot[n][w];
          }
          const float inv_r = rsqrtf(total_sq * inv_d + eps);
          s_inv_rms[n] = inv_r;
          const float sc = total_dot * inv_r;
          s_p[n] = sc;
          max_s = fmaxf(max_s, sc);
        }
        float denom = 0.0f;
        for (int n = 0; n < N; ++n) {
          const float e = expf(s_p[n] - max_s);
          s_p[n] = e;
          denom += e;
        }
        const float inv_denom = 1.0f / denom;
        for (int n = 0; n < N; ++n) {
          s_p[n] *= inv_denom;
        }
      }
    }
    __syncthreads();

    const size_t out_row_off =
        (static_cast<size_t>(q) * static_cast<size_t>(T) + static_cast<size_t>(t)) *
        static_cast<size_t>(D);

    if constexpr (!HAS_NORM) {
      float* out_f = reinterpret_cast<float*>(out) + out_row_off;
      for (int i = tid; i < D; i += 256) {
        float y = 0.0f;
        for (int n = 0; n < N; ++n) {
          const char* ptr = ink2_pick_src(n, s0, s1, s2, s3, s4, s5, s6, s7);
          const bool is_f32 = ((src_f32_mask >> n) & 1u) != 0u;
          y += s_p[n] * (ink2_load_f1(ptr, row_off + i, is_f32) * s_inv_rms[n]);
        }
        out_f[i] = y;
      }
      return;
    }

    float y_sq = 0.0f;
    for (int i = tid; i < D; i += 256) {
      float y = 0.0f;
      for (int n = 0; n < N; ++n) {
        const char* ptr = ink2_pick_src(n, s0, s1, s2, s3, s4, s5, s6, s7);
        const bool is_f32 = ((src_f32_mask >> n) & 1u) != 0u;
        y += s_p[n] * (ink2_load_f1(ptr, row_off + i, is_f32) * s_inv_rms[n]);
      }
      y_sq += y * y;
    }
    y_sq = ink2_warp_sum(y_sq);
    if (lane == 0) {
      s_sq[0][warp] = y_sq;
    }
    __syncthreads();
    if (warp == 0 && lane == 0) {
      float total = 0.0f;
#pragma unroll
      for (int w = 0; w < 8; ++w) {
        total += s_sq[0][w];
      }
      s_sq[1][0] = rsqrtf(total / static_cast<float>(D) + eps);
    }
    __syncthreads();
    const float inv_rms_y = s_sq[1][0];
    const float* w_row = norm_weights +
        (norm_w_bcast ? 0 : static_cast<size_t>(q) * static_cast<size_t>(D));
    nv_bfloat16* out_bf = reinterpret_cast<nv_bfloat16*>(out) + out_row_off;
    for (int i = tid; i < D; i += 256) {
      float y = 0.0f;
      for (int n = 0; n < N; ++n) {
        const char* ptr = ink2_pick_src(n, s0, s1, s2, s3, s4, s5, s6, s7);
        const bool is_f32 = ((src_f32_mask >> n) & 1u) != 0u;
        y += s_p[n] * (ink2_load_f1(ptr, row_off + i, is_f32) * s_inv_rms[n]);
      }
      out_bf[i] = __float2bfloat16_rn((y * inv_rms_y) * w_row[i]);
    }
  }
}
)cuda"
R"cuda(
// ============================================================================
// 2. Fused LoopController + Sublayer RMSNorm (G=1 or G=3, rank=64, D%4==0)
//    3-stage GPU pipeline inside one MLX Primitive (eliminates inv_rms launch):
//      2a. ink2_loop_down_fused_cuda:      [2, G, T, 64] FP32 h_part (inv_rms fused in-loop!)
//      2b. ink2_loop_up48_mma_cuda:        [G, T, D] FP32 blended (2-term BF16 TC MMA, 48 ch/CTA)
//          or ink2_loop_up_blend_f32_cuda: fallback for FP32 weights / SM < 8.0
//      2c. ink2_rms_norm_f32_to_bf16_cuda: [G, T, D] BF16 out
// ============================================================================

// Step 2a: Grid dim3(128, G, (T + TILE_T - 1) / TILE_T), Block dim3(NWARPS * 32, 1, 1)
//   blockIdx.x = r_idx * 2 + half_id (half_id=0 -> dest, half_id=1 -> msg).
//   Computes sum_sq(dest/msg) in the exact same loop over D -> ZERO extra DRAM reads!
template <bool W_BF16, bool VEC8, int TILE_T, int NWARPS>
__global__ __launch_bounds__(NWARPS * 32, NWARPS == 3 ? 8 : 4) void ink2_loop_down_fused_cuda(
    const char* __restrict__ dest,
    const char* __restrict__ msg,
    const void* __restrict__ down_w,
    float* __restrict__ h_part,
    int T,
    int D,
    int G,
    int dest_bcast,
    int msg_bcast,
    int dest_is_f32,
    int msg_is_f32,
    float eps) {
  const int r_idx = static_cast<int>(blockIdx.x) >> 1;
  const int half_id = static_cast<int>(blockIdx.x) & 1;
  const int g = static_cast<int>(blockIdx.y);
  const int row0 = static_cast<int>(blockIdx.z) * TILE_T;
  if (r_idx >= 64 || row0 >= T) {
    return;
  }
  const int n_rows = (T - row0 < TILE_T) ? (T - row0) : TILE_T;
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;

  const char* in_ptr = (half_id == 0) ? dest : msg;
  const bool in_bcast = (half_id == 0) ? (dest_bcast != 0) : (msg_bcast != 0);
  const bool in_f32 = (half_id == 0) ? (dest_is_f32 != 0) : (msg_is_f32 != 0);
  const size_t in_row0 =
      in_bcast ? static_cast<size_t>(row0)
               : static_cast<size_t>(g) * static_cast<size_t>(T) + static_cast<size_t>(row0);

  float acc[TILE_T];
  float sq[TILE_T];
#pragma unroll
  for (int r = 0; r < TILE_T; ++r) {
    acc[r] = 0.0f;
    sq[r] = 0.0f;
  }

  if constexpr (W_BF16 && VEC8) {
    const int D8 = D >> 3;
    const size_t w_base8 =
        ((static_cast<size_t>(g) * 64 + static_cast<size_t>(r_idx)) * 2 +
         static_cast<size_t>(half_id)) *
        static_cast<size_t>(D8);
    const uint4* w_u4 = reinterpret_cast<const uint4*>(down_w) + w_base8;
    if (in_f32) {
      const size_t stride4 = static_cast<size_t>(D >> 2);
      const float4* in_f4 = reinterpret_cast<const float4*>(in_ptr) + in_row0 * stride4;
      for (int i8 = tid; i8 < D8; i8 += NWARPS * 32) {
        const uint4 wraw = ink2_ld_cs_u4(w_u4 + i8);
        const __nv_bfloat162* wp = reinterpret_cast<const __nv_bfloat162*>(&wraw);
        const float2 w0 = __bfloat1622float2(wp[0]);
        const float2 w1 = __bfloat1622float2(wp[1]);
        const float2 w2 = __bfloat1622float2(wp[2]);
        const float2 w3 = __bfloat1622float2(wp[3]);
        const size_t col4 = static_cast<size_t>(i8) * 2;
#pragma unroll
        for (int r = 0; r < TILE_T; ++r) {
          if (r < n_rows) {
            const float4 x0 = in_f4[static_cast<size_t>(r) * stride4 + col4];
            const float4 x1 = in_f4[static_cast<size_t>(r) * stride4 + col4 + 1];
            acc[r] = fmaf(w0.x, x0.x, fmaf(w0.y, x0.y, fmaf(w1.x, x0.z, fmaf(w1.y, x0.w, acc[r]))));
            acc[r] = fmaf(w2.x, x1.x, fmaf(w2.y, x1.y, fmaf(w3.x, x1.z, fmaf(w3.y, x1.w, acc[r]))));
            sq[r] = fmaf(x0.x, x0.x, fmaf(x0.y, x0.y, fmaf(x0.z, x0.z, fmaf(x0.w, x0.w, sq[r]))));
            sq[r] = fmaf(x1.x, x1.x, fmaf(x1.y, x1.y, fmaf(x1.z, x1.z, fmaf(x1.w, x1.w, sq[r]))));
          }
        }
      }
    } else {
      const size_t stride8 = static_cast<size_t>(D8);
      const uint4* in_b8 = reinterpret_cast<const uint4*>(in_ptr) + in_row0 * stride8;
      for (int i8 = tid; i8 < D8; i8 += NWARPS * 32) {
        const uint4 wraw = ink2_ld_cs_u4(w_u4 + i8);
        const __nv_bfloat162* wp = reinterpret_cast<const __nv_bfloat162*>(&wraw);
        const float2 w0 = __bfloat1622float2(wp[0]);
        const float2 w1 = __bfloat1622float2(wp[1]);
        const float2 w2 = __bfloat1622float2(wp[2]);
        const float2 w3 = __bfloat1622float2(wp[3]);
#pragma unroll
        for (int r = 0; r < TILE_T; ++r) {
          if (r < n_rows) {
            const uint4 xraw = in_b8[static_cast<size_t>(r) * stride8 + i8];
            const __nv_bfloat162* xp = reinterpret_cast<const __nv_bfloat162*>(&xraw);
            const float2 x0 = __bfloat1622float2(xp[0]);
            const float2 x1 = __bfloat1622float2(xp[1]);
            const float2 x2 = __bfloat1622float2(xp[2]);
            const float2 x3 = __bfloat1622float2(xp[3]);
            acc[r] = fmaf(w0.x, x0.x, fmaf(w0.y, x0.y, fmaf(w1.x, x1.x, fmaf(w1.y, x1.y, acc[r]))));
            acc[r] = fmaf(w2.x, x2.x, fmaf(w2.y, x2.y, fmaf(w3.x, x3.x, fmaf(w3.y, x3.y, acc[r]))));
            sq[r] = fmaf(x0.x, x0.x, fmaf(x0.y, x0.y, fmaf(x1.x, x1.x, fmaf(x1.y, x1.y, sq[r]))));
            sq[r] = fmaf(x2.x, x2.x, fmaf(x2.y, x2.y, fmaf(x3.x, x3.x, fmaf(x3.y, x3.y, sq[r]))));
          }
        }
      }
    }
  } else {
    const int D4 = D >> 2;
    const size_t w_base4 =
        ((static_cast<size_t>(g) * 64 + static_cast<size_t>(r_idx)) * 2 +
         static_cast<size_t>(half_id)) *
        static_cast<size_t>(D4);
    const size_t in_base4 = in_row0 * static_cast<size_t>(D4);
    for (int i4 = tid; i4 < D4; i4 += NWARPS * 32) {
      const float4 wv = ink2_load_w4<W_BF16>(down_w, w_base4 + i4);
#pragma unroll
      for (int r = 0; r < TILE_T; ++r) {
        if (r < n_rows) {
          const float4 xv =
              ink2_load_f4(in_ptr, in_base4 + static_cast<size_t>(r) * D4 + i4, in_f32);
          acc[r] = fmaf(wv.x, xv.x, fmaf(wv.y, xv.y, fmaf(wv.z, xv.z, fmaf(wv.w, xv.w, acc[r]))));
          sq[r] = fmaf(xv.x, xv.x, fmaf(xv.y, xv.y, fmaf(xv.z, xv.z, fmaf(xv.w, xv.w, sq[r]))));
        }
      }
    }
  }

  __shared__ float s_acc[NWARPS][TILE_T];
  __shared__ float s_sq[NWARPS][TILE_T];
#pragma unroll
  for (int r = 0; r < TILE_T; ++r) {
    if (r < n_rows) {
      const float va = ink2_warp_sum(acc[r]);
      const float vs = ink2_warp_sum(sq[r]);
      if (lane == 0) {
        s_acc[warp][r] = va;
        s_sq[warp][r] = vs;
      }
    }
  }
  __syncthreads();

  if (tid < n_rows) {
    float ac = 0.0f;
    float sq_tot = 0.0f;
#pragma unroll
    for (int ww = 0; ww < NWARPS; ++ww) {
      ac += s_acc[ww][tid];
      sq_tot += s_sq[ww][tid];
    }
    const float inv_r = rsqrtf(sq_tot / static_cast<float>(D) + eps);
    const size_t rf =
        (static_cast<size_t>(half_id) * static_cast<size_t>(G) + static_cast<size_t>(g)) *
            static_cast<size_t>(T) +
        static_cast<size_t>(row0 + tid);
    h_part[rf * 64 + r_idx] = ac * inv_r;
  }
}
)cuda"
R"cuda(
// Step 2b (fast BF16 Tensor Core path, sm_80+): 48 channels/CTA (exact 1.0 wave for G=1,
// 3.0 waves for G=3 at D=2304!), 2-term BF16 split of h (15-bit mantissa -> FP32 parity,
// 100% batch-invariant across all T). Grid: dim3((D + 47) / 48, G, (T + 7) / 8), Block: 192.
__global__ __launch_bounds__(192, 4) void ink2_loop_up48_mma_cuda(
    const float* __restrict__ h_part,
    const nv_bfloat16* __restrict__ gain_forget_w,
    const char* __restrict__ dest,
    const char* __restrict__ msg,
    float* __restrict__ blended,
    int T,
    int D,
    int G,
    int dest_bcast,
    int msg_bcast,
    int dest_is_f32,
    int msg_is_f32) {
  constexpr int CH = 48;
  constexpr int NTHREADS = 192;
  const int d_base = static_cast<int>(blockIdx.x) * CH;
  const int g = static_cast<int>(blockIdx.y);
  const int row0 = static_cast<int>(blockIdx.z) * 8;
  if (d_base >= D || row0 >= T) {
    return;
  }
  const int n_rows = (T - row0 < 8) ? (T - row0) : 8;
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;

  __shared__ nv_bfloat16 s_h_hi[8][64];
  __shared__ nv_bfloat16 s_h_lo[8][64];
  union WOrOut {
    uint4 w[2][CH][8];
    struct {
      float g[8][CH + 1];
      float f[8][CH + 1];
    } out;
  };
  __shared__ WOrOut sh;

  // Combine h_part[0] + h_part[1] into exact 2-term BF16 (hi + residual lo) in SMEM.
  const int total_h4 = n_rows * 16;
  for (int idx4 = tid; idx4 < total_h4; idx4 += NTHREADS) {
    const int r = idx4 >> 4;
    const int c4 = idx4 & 15;
    const size_t rf0 = static_cast<size_t>(g) * static_cast<size_t>(T) + static_cast<size_t>(row0 + r);
    const size_t rf1 =
        (static_cast<size_t>(G) + static_cast<size_t>(g)) * static_cast<size_t>(T) +
        static_cast<size_t>(row0 + r);
    const float4 hv0 = reinterpret_cast<const float4*>(h_part + rf0 * 64)[c4];
    const float4 hv1 = reinterpret_cast<const float4*>(h_part + rf1 * 64)[c4];
    const float4 h = make_float4(hv0.x + hv1.x, hv0.y + hv1.y, hv0.z + hv1.z, hv0.w + hv1.w);
    ink2_store_bf16x4(s_h_hi[r], c4, h);
    const uint2 raw = reinterpret_cast<const uint2*>(s_h_hi[r])[c4];
    const float2 lo2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.x));
    const float2 hi2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.y));
    ink2_store_bf16x4(
        s_h_lo[r], c4, make_float4(h.x - lo2.x, h.y - lo2.y, h.z - hi2.x, h.w - hi2.y));
  }
  for (int idx4 = total_h4 + tid; idx4 < 8 * 16; idx4 += NTHREADS) {
    reinterpret_cast<uint2*>(s_h_hi[0])[idx4] = make_uint2(0, 0);
    reinterpret_cast<uint2*>(s_h_lo[0])[idx4] = make_uint2(0, 0);
  }

  // Coalesced 128-bit ld.global.cs.v4.u32 load of [2][48][8] uint4 weights into XOR-swizzled SMEM.
  const uint4* gf_u4 = reinterpret_cast<const uint4*>(gain_forget_w);
#pragma unroll
  for (int idx = tid; idx < 2 * CH * 8; idx += NTHREADS) {
    const int u_col = idx & 7;
    const int row_flat = idx >> 3;
    const int half_id = (row_flat >= CH) ? 1 : 0;
    const int lc = row_flat - half_id * CH;
    const int col = (d_base + lc < D) ? (d_base + lc) : (D - 1);
    const size_t w_off =
        ((static_cast<size_t>(g) * 2 + static_cast<size_t>(half_id)) * static_cast<size_t>(D) +
         static_cast<size_t>(col)) *
            8 +
        u_col;
    sh.w[half_id][lc][u_col ^ (lc & 7)] = ink2_ld_cs_u4(gf_u4 + w_off);
  }
  __syncthreads();

  const int half_id = (warp >= 3) ? 1 : 0;
  const int c_off = (warp - half_id * 3) * 16;
  const int g_lane = lane >> 2;
  const int t_lane = lane & 3;
  const int r0 = c_off + g_lane;
  const int r1 = c_off + g_lane + 8;

  float d0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  const unsigned int* hb_hi = reinterpret_cast<const unsigned int*>(s_h_hi[g_lane]);
  const unsigned int* hb_lo = reinterpret_cast<const unsigned int*>(s_h_lo[g_lane]);
#pragma unroll
  for (int ks = 0; ks < 4; ++ks) {
    const int u0 = ks * 2;
    const int u1 = ks * 2 + 1;
    const unsigned int a0 =
        reinterpret_cast<const unsigned int*>(&sh.w[half_id][r0][u0 ^ (r0 & 7)])[t_lane];
    const unsigned int a2 =
        reinterpret_cast<const unsigned int*>(&sh.w[half_id][r0][u1 ^ (r0 & 7)])[t_lane];
    const unsigned int a1 =
        reinterpret_cast<const unsigned int*>(&sh.w[half_id][r1][u0 ^ (r1 & 7)])[t_lane];
    const unsigned int a3 =
        reinterpret_cast<const unsigned int*>(&sh.w[half_id][r1][u1 ^ (r1 & 7)])[t_lane];
    const int k_u32 = ks * 8;
    ink2_mma_m16n8k16_bf16(
        d0[0], d0[1], d0[2], d0[3], a0, a1, a2, a3, hb_hi[k_u32 + t_lane],
        hb_hi[k_u32 + 4 + t_lane]);
    ink2_mma_m16n8k16_bf16(
        d0[0], d0[1], d0[2], d0[3], a0, a1, a2, a3, hb_lo[k_u32 + t_lane],
        hb_lo[k_u32 + 4 + t_lane]);
  }
  __syncthreads();

  float (*dst)[CH + 1] = (half_id == 0) ? sh.out.g : sh.out.f;
  if (2 * t_lane < n_rows) {
    dst[2 * t_lane][r0] = d0[0];
    dst[2 * t_lane][r1] = d0[2];
  }
  if (2 * t_lane + 1 < n_rows) {
    dst[2 * t_lane + 1][r0] = d0[1];
    dst[2 * t_lane + 1][r1] = d0[3];
  }
  __syncthreads();

  const bool d_f32 = (dest_is_f32 != 0);
  const bool m_f32 = (msg_is_f32 != 0);
  const size_t d_stream_off =
      dest_bcast ? 0 : static_cast<size_t>(g) * static_cast<size_t>(T) * static_cast<size_t>(D);
  const size_t m_stream_off =
      msg_bcast ? 0 : static_cast<size_t>(g) * static_cast<size_t>(T) * static_cast<size_t>(D);
  const size_t b_stream_off =
      static_cast<size_t>(g) * static_cast<size_t>(T) * static_cast<size_t>(D);

  const int total_out = n_rows * CH;
  for (int idx = tid; idx < total_out; idx += NTHREADS) {
    const int r = idx / CH;
    const int lc = idx - r * CH;
    const int col = d_base + lc;
    if (col < D) {
      const float g_val = tanhf(fminf(fmaxf(sh.out.g[r][lc], -20.0f), 20.0f));
      const float f_val = tanhf(fminf(fmaxf(sh.out.f[r][lc], -20.0f), 20.0f));
      const size_t row_off = static_cast<size_t>(row0 + r) * static_cast<size_t>(D) + col;
      const float dv = ink2_load_f1(dest, d_stream_off + row_off, d_f32);
      const float mv = ink2_load_f1(msg, m_stream_off + row_off, m_f32);
      const float kept = __fmul_rn(1.0f - f_val, dv);
      const float added = __fmul_rn(g_val, mv);
      blended[b_stream_off + row_off] = __fadd_rn(kept, added);
    }
  }
}

// Step 2b fallback (FP32 weights or SM < 8.0): Grid dim3((D + 31) / 32, G, (T + 15) / 16), Block 128
template <bool W_BF16>
__global__ __launch_bounds__(128, 4) void ink2_loop_up_blend_f32_cuda(
    const float* __restrict__ h_part,
    const void* __restrict__ gain_forget_w,
    const char* __restrict__ dest,
    const char* __restrict__ msg,
    float* __restrict__ blended,
    int T,
    int D,
    int G,
    int dest_bcast,
    int msg_bcast,
    int dest_is_f32,
    int msg_is_f32) {
  const int d_base = static_cast<int>(blockIdx.x) * 32;
  const int g = static_cast<int>(blockIdx.y);
  const int row0 = static_cast<int>(blockIdx.z) * 16;
  if (d_base >= D || row0 >= T) {
    return;
  }
  const int n_rows = (T - row0 < 16) ? (T - row0) : 16;
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;

  __shared__ float s_h[16][64];
  __shared__ float s_g[16][33];
  __shared__ float s_f[16][33];

  const int total_h4 = n_rows * 16;
  for (int idx4 = tid; idx4 < total_h4; idx4 += 128) {
    const int hr = idx4 >> 4;
    const int hc4 = idx4 & 15;
    const size_t rf0 = static_cast<size_t>(g) * static_cast<size_t>(T) + static_cast<size_t>(row0 + hr);
    const size_t rf1 =
        (static_cast<size_t>(G) + static_cast<size_t>(g)) * static_cast<size_t>(T) +
        static_cast<size_t>(row0 + hr);
    const float4 v0 = reinterpret_cast<const float4*>(h_part + rf0 * 64)[hc4];
    const float4 v1 = reinterpret_cast<const float4*>(h_part + rf1 * 64)[hc4];
    reinterpret_cast<float4*>(s_h[hr])[hc4] =
        make_float4(v0.x + v1.x, v0.y + v1.y, v0.z + v1.z, v0.w + v1.w);
  }
  __syncthreads();

  const int half_id = lane >> 4;  // 0 = gain, 1 = forget
  const int sub = lane & 15;      // 0..15 (16 float4s = 64 rank elems)
  const size_t w_half_base4 =
      (static_cast<size_t>(g) * 2 + static_cast<size_t>(half_id)) *
      static_cast<size_t>(D) * 16;

  const int c_base = d_base + warp * 8;
  for (int c = 0; c < 8; ++c) {
    const int col = c_base + c;
    if (col >= D) {
      break;
    }
    const float4 wv =
        ink2_load_w4<W_BF16>(gain_forget_w, w_half_base4 + static_cast<size_t>(col) * 16 + sub);
    for (int r = 0; r < n_rows; ++r) {
      const float4 hv = reinterpret_cast<const float4*>(s_h[r])[sub];
      float p = (wv.x * hv.x + wv.y * hv.y) + (wv.z * hv.z + wv.w * hv.w);
      p += __shfl_xor_sync(0xffffffffu, p, 8);
      p += __shfl_xor_sync(0xffffffffu, p, 4);
      p += __shfl_xor_sync(0xffffffffu, p, 2);
      p += __shfl_xor_sync(0xffffffffu, p, 1);
      if (sub == 0) {
        if (half_id == 0) {
          s_g[r][warp * 8 + c] = p;
        } else {
          s_f[r][warp * 8 + c] = p;
        }
      }
    }
  }
  __syncthreads();

  const int lc = tid & 31;
  const int col = d_base + lc;
  if (col < D) {
    const bool d_f32 = (dest_is_f32 != 0);
    const bool m_f32 = (msg_is_f32 != 0);
    const size_t d_stream_off =
        dest_bcast ? 0 : static_cast<size_t>(g) * static_cast<size_t>(T) * static_cast<size_t>(D);
    const size_t m_stream_off =
        msg_bcast ? 0 : static_cast<size_t>(g) * static_cast<size_t>(T) * static_cast<size_t>(D);
    const size_t b_stream_off =
        static_cast<size_t>(g) * static_cast<size_t>(T) * static_cast<size_t>(D);

    for (int r = tid >> 5; r < n_rows; r += 4) {
      const float g_val = tanhf(fminf(fmaxf(s_g[r][lc], -20.0f), 20.0f));
      const float f_val = tanhf(fminf(fmaxf(s_f[r][lc], -20.0f), 20.0f));
      const size_t row_off = static_cast<size_t>(row0 + r) * static_cast<size_t>(D) + col;
      const float dv = ink2_load_f1(dest, d_stream_off + row_off, d_f32);
      const float mv = ink2_load_f1(msg, m_stream_off + row_off, m_f32);
      const float kept = __fmul_rn(1.0f - f_val, dv);
      const float added = __fmul_rn(g_val, mv);
      blended[b_stream_off + row_off] = __fadd_rn(kept, added);
    }
  }
}

// Step 2c: Grid dim3(T, G, 1), Block dim3(256, 1, 1)
__global__ __launch_bounds__(256, 4) void ink2_rms_norm_f32_to_bf16_cuda(
    const float* __restrict__ blended,
    const float* __restrict__ norm_w,
    nv_bfloat16* __restrict__ out,
    int T,
    int D,
    int norm_w_bcast,
    float eps) {
  const int t = static_cast<int>(blockIdx.x);
  const int g = static_cast<int>(blockIdx.y);
  if (t >= T) {
    return;
  }
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int D4 = D >> 2;
  const size_t row_off4 =
      (static_cast<size_t>(g) * static_cast<size_t>(T) + static_cast<size_t>(t)) *
      static_cast<size_t>(D4);
  const float4* in4 = reinterpret_cast<const float4*>(blended) + row_off4;

  float4 b0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  float4 b1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  float4 b2 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  const bool fits_regs = (D4 <= 768);
  float sq = 0.0f;
  int step = 0;
  for (int i4 = tid; i4 < D4; i4 += 256, ++step) {
    const float4 v = in4[i4];
    sq += (v.x * v.x + v.y * v.y) + (v.z * v.z + v.w * v.w);
    if (fits_regs) {
      if (step == 0) b0 = v;
      else if (step == 1) b1 = v;
      else b2 = v;
    }
  }
  sq = ink2_warp_sum(sq);

  __shared__ float s_sq[8];
  if (lane == 0) {
    s_sq[warp] = sq;
  }
  __syncthreads();
  if (warp == 0 && lane == 0) {
    float total = 0.0f;
#pragma unroll
    for (int w = 0; w < 8; ++w) {
      total += s_sq[w];
    }
    s_sq[0] = rsqrtf(total / static_cast<float>(D) + eps);
  }
  __syncthreads();
  const float inv_rms = s_sq[0];

  const float4* w4 = reinterpret_cast<const float4*>(norm_w) +
      (norm_w_bcast ? 0 : static_cast<size_t>(g) * static_cast<size_t>(D4));
  step = 0;
  for (int i4 = tid; i4 < D4; i4 += 256, ++step) {
    const float4 v = fits_regs ? ((step == 0) ? b0 : ((step == 1) ? b1 : b2)) : in4[i4];
    const float4 wv = w4[i4];
    const float4 res = make_float4(
        (v.x * inv_rms) * wv.x,
        (v.y * inv_rms) * wv.y,
        (v.z * inv_rms) * wv.z,
        (v.w * inv_rms) * wv.w);
    ink2_store_bf16x4(out, row_off4 + i4, res);
  }
}
)cuda"
R"cuda(
// ============================================================================
// 3. Fused Q/R L2-clip + K/V ShortConv + K L2-clip + V RMSNorm + Global tau
//    + In-Place KV Cache Store (BF16 or INT8, linear flat or SWA ring, linear or tree)
// Grid: dim3(T, 40, 1), Block: dim3(32, 1, 1)
//   blockIdx.y in  0..23 -> Q + R head h = blockIdx.y
//   blockIdx.y in 24..31 -> K head h_kv = blockIdx.y - 24
//   blockIdx.y in 32..39 -> V head h_kv = blockIdx.y - 32
// ============================================================================
template <bool IS_INT8>
__global__ __launch_bounds__(32, 8) void ink2_kv_conv_prep_store_cuda(
    const nv_bfloat16* __restrict__ q_raw,
    const nv_bfloat16* __restrict__ r_raw,
    const nv_bfloat16* __restrict__ k_raw,
    const nv_bfloat16* __restrict__ v_raw,
    const float* __restrict__ q_clip,
    const float* __restrict__ rel_clip,
    const float* __restrict__ k_conv_w,
    const float* __restrict__ v_conv_w,
    const char* __restrict__ dyn_k,
    const char* __restrict__ dyn_v,
    const nv_bfloat16* __restrict__ k_hist_in,
    const nv_bfloat16* __restrict__ v_hist_in,
    const int* __restrict__ conv_taps,
    const int* __restrict__ depths,
    nv_bfloat16* __restrict__ out_q,
    float* __restrict__ out_r,
    nv_bfloat16* __restrict__ out_k_hist,
    nv_bfloat16* __restrict__ out_v_hist,
    void* __restrict__ kv_k_cache,
    void* __restrict__ kv_v_cache,
    float* __restrict__ kv_k_scales,
    float* __restrict__ kv_v_scales,
    int T,
    int start_pos,
    int write_offset,
    int ring_cap,
    float log_n_floor,
    float log_alpha,
    unsigned int flags,
    long long q_s0,
    long long r_s0,
    long long k_s0,
    long long v_s0,
    long long kc_sh,
    long long kc_st,
    long long vc_sh,
    long long vc_st,
    long long ks_sh,
    long long ks_st,
    long long vs_sh,
    long long vs_st) {
  // flags: bit0=is_dyn, bit1=dyn_k_f32, bit2=dyn_v_f32, bit3=has_taps,
  //        bit4=has_depths, bit5=is_global, bit6=update_hist
  const int t = static_cast<int>(blockIdx.x);
  const int by = static_cast<int>(blockIdx.y);
  if (t >= T || by >= 40) {
    return;
  }
  const int lane = static_cast<int>(threadIdx.x);

  constexpr float kQkClip = 22.627416997969522f;
  constexpr float kRelClip = 8.0f;
  constexpr float kEps = 1e-6f;
  constexpr float kStaticScale = 0.44721359549995793f;
  constexpr float kDynScale = 0.3333333333333333f;

  if (by < 24) {
    // --- Q + R head h = by ---
    const int h = by;
    const bool is_global = ((flags >> 5) & 1u) != 0u;
    const bool has_depths = ((flags >> 4) & 1u) != 0u;
    float tau = 1.0f;
    if (is_global && log_alpha != 0.0f && log_n_floor > 0.0f) {
      const int pos = start_pos + (has_depths ? depths[t] : t);
      const float ratio = (static_cast<float>(pos) + 1.0f) / log_n_floor;
      if (ratio > 1.0f) {
        tau = 1.0f + log_alpha * logf(ratio);
      }
    }

    const size_t q_in_base = static_cast<size_t>(t) * static_cast<size_t>(q_s0) + static_cast<size_t>(h) * 128;
    const size_t qc_base = static_cast<size_t>(h) * 128;
    const float u0 = __bfloat162float(q_raw[q_in_base + lane]) * q_clip[qc_base + lane];
    const float u1 = __bfloat162float(q_raw[q_in_base + lane + 32]) * q_clip[qc_base + lane + 32];
    const float u2 = __bfloat162float(q_raw[q_in_base + lane + 64]) * q_clip[qc_base + lane + 64];
    const float u3 = __bfloat162float(q_raw[q_in_base + lane + 96]) * q_clip[qc_base + lane + 96];

    const float q_sq = ink2_warp_sum((u0 * u0 + u1 * u1) + (u2 * u2 + u3 * u3));
    const float q_fac = fminf(kQkClip / (sqrtf(q_sq) + kEps), 1.0f);

    const size_t q_out_base =
        (static_cast<size_t>(h) * static_cast<size_t>(T) + static_cast<size_t>(t)) * 128;
    nv_bfloat16 qb0 = __float2bfloat16_rn(u0 * q_fac);
    nv_bfloat16 qb1 = __float2bfloat16_rn(u1 * q_fac);
    nv_bfloat16 qb2 = __float2bfloat16_rn(u2 * q_fac);
    nv_bfloat16 qb3 = __float2bfloat16_rn(u3 * q_fac);
    if (is_global) {
      qb0 = __float2bfloat16_rn(__bfloat162float(qb0) * tau);
      qb1 = __float2bfloat16_rn(__bfloat162float(qb1) * tau);
      qb2 = __float2bfloat16_rn(__bfloat162float(qb2) * tau);
      qb3 = __float2bfloat16_rn(__bfloat162float(qb3) * tau);
    }
    out_q[q_out_base + lane] = qb0;
    out_q[q_out_base + lane + 32] = qb1;
    out_q[q_out_base + lane + 64] = qb2;
    out_q[q_out_base + lane + 96] = qb3;

    // R head (d_rel = 16 -> lanes 0..15)
    float ru = 0.0f;
    if (lane < 16) {
      const size_t r_in_idx =
          static_cast<size_t>(t) * static_cast<size_t>(r_s0) + static_cast<size_t>(h) * 16 + lane;
      ru = __bfloat162float(r_raw[r_in_idx]) * rel_clip[static_cast<size_t>(h) * 16 + lane];
    }
    const float r_sq = ink2_warp_sum(ru * ru);
    const float r_fac = fminf(kRelClip / (sqrtf(r_sq) + kEps), 1.0f);
    if (lane < 16) {
      float rc = __bfloat162float(__float2bfloat16_rn(ru * r_fac));
      if (is_global) {
        rc *= tau;
      }
      out_r[(static_cast<size_t>(t) * 24 + static_cast<size_t>(h)) * 16 + lane] = rc;
    }
    return;
  }

  // --- K (by in 24..31) or V (by in 32..39) ---
  const bool is_v = (by >= 32);
  const int h_kv = by - (is_v ? 32 : 24);
  const bool is_dyn = ((flags >> 0) & 1u) != 0u;
  const bool dyn_f32 = is_v ? (((flags >> 2) & 1u) != 0u) : (((flags >> 1) & 1u) != 0u);
  const bool has_taps = ((flags >> 3) & 1u) != 0u;
  const bool update_hist = ((flags >> 6) & 1u) != 0u;

  const nv_bfloat16* x_raw = is_v ? v_raw : k_raw;
  const long long x_s0 = is_v ? v_s0 : k_s0;
  const float* conv_w = is_v ? v_conv_w : k_conv_w;
  const nv_bfloat16* hist_in = is_v ? v_hist_in : k_hist_in;
  nv_bfloat16* hist_out = is_v ? out_v_hist : out_k_hist;

  int t0, t1, t2, t3;
  if (has_taps) {
    const int4 taps = reinterpret_cast<const int4*>(conv_taps)[t];
    t0 = taps.x;
    t1 = taps.y;
    t2 = taps.z;
    t3 = taps.w;
  } else {
    t0 = t - 3;
    t1 = t - 2;
    t2 = t - 1;
    t3 = t;
  }

  float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
  float conv_scale = kStaticScale;
  if (is_dyn) {
    const char* dyn_ptr = is_v ? dyn_v : dyn_k;
    const size_t d_base = (static_cast<size_t>(t) * 8 + static_cast<size_t>(h_kv)) * 4;
    d0 = ink2_load_f1(dyn_ptr, d_base + 0, dyn_f32);
    d1 = ink2_load_f1(dyn_ptr, d_base + 1, dyn_f32);
    d2 = ink2_load_f1(dyn_ptr, d_base + 2, dyn_f32);
    d3 = ink2_load_f1(dyn_ptr, d_base + 3, dyn_f32);
    conv_scale = kDynScale;
  }

  auto load_tap = [&](int idx, int ch) -> float {
    if (idx < 0) {
      const int hrow = 3 + idx;
      return (hrow >= 0 && hrow < 3)
          ? __bfloat162float(hist_in[static_cast<size_t>(hrow) * 1024 + ch])
          : 0.0f;
    }
    return (idx < T)
        ? __bfloat162float(x_raw[static_cast<size_t>(idx) * static_cast<size_t>(x_s0) + ch])
        : 0.0f;
  };

  const bool write_hist = update_hist && (t == T - 1);
  float cv[4];
  float sum_sq = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int d = lane + i * 32;
    const int ch = h_kv * 128 + d;
    const float x3 = load_tap(t0, ch);
    const float x2 = load_tap(t1, ch);
    const float x1 = load_tap(t2, ch);
    const float x0 = load_tap(t3, ch);
    if (write_hist) {
      hist_out[0 * 1024 + ch] = __float2bfloat16_rn(x2);
      hist_out[1 * 1024 + ch] = __float2bfloat16_rn(x1);
      hist_out[2 * 1024 + ch] = __float2bfloat16_rn(x0);
    }
    const float4 wf = reinterpret_cast<const float4*>(conv_w)[ch];
    const float w3 = wf.x + d3;
    const float w2 = wf.y + d2;
    const float w1 = wf.z + d1;
    const float w0 = wf.w + d0;
    float acc = __fadd_rn(x0, __fmul_rn(w3, x3));
    acc = __fadd_rn(acc, __fmul_rn(w2, x2));
    acc = __fadd_rn(acc, __fmul_rn(w1, x1));
    acc = __fadd_rn(acc, __fmul_rn(w0, x0));
    const float cval = __bfloat162float(__float2bfloat16_rn(acc * conv_scale));
    cv[i] = cval;
    sum_sq += cval * cval;
  }

  // Avoid write race if caller passes T > ring_cap into a circular SWA ring
  if (ring_cap > 0 && t < T - ring_cap) {
    return;
  }

  sum_sq = ink2_warp_sum(sum_sq);
  const float factor = is_v
      ? rsqrtf(sum_sq * (1.0f / 128.0f) + kEps)
      : fminf(kQkClip / (sqrtf(sum_sq) + kEps), 1.0f);

  const int slot = (ring_cap > 0) ? ((write_offset + t) % ring_cap) : (write_offset + t);
  const long long c_sh = is_v ? vc_sh : kc_sh;
  const long long c_st = is_v ? vc_st : kc_st;
  const size_t cache_row =
      static_cast<size_t>(h_kv) * static_cast<size_t>(c_sh) +
      static_cast<size_t>(slot) * static_cast<size_t>(c_st);

  if constexpr (!IS_INT8) {
    nv_bfloat16* dst = reinterpret_cast<nv_bfloat16*>(is_v ? kv_v_cache : kv_k_cache);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      dst[cache_row + lane + i * 32] = __float2bfloat16_rn(cv[i] * factor);
    }
  } else {
    float qv[4];
    float local_max = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float val = __bfloat162float(__float2bfloat16_rn(cv[i] * factor));
      qv[i] = val;
      local_max = fmaxf(local_max, fabsf(val));
    }
    const float amax = ink2_warp_max(local_max);
    const float q_scale = (amax > 0.0f) ? (amax / 127.0f) : 0.0f;
    const float inv_scale = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
    if (lane == 0) {
      float* s_dst = is_v ? kv_v_scales : kv_k_scales;
      const long long s_sh = is_v ? vs_sh : ks_sh;
      const long long s_st = is_v ? vs_st : ks_st;
      s_dst[static_cast<size_t>(h_kv) * static_cast<size_t>(s_sh) +
            static_cast<size_t>(slot) * static_cast<size_t>(s_st)] = q_scale;
    }
    signed char* dst = reinterpret_cast<signed char*>(is_v ? kv_v_cache : kv_k_cache);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int q = (amax > 0.0f)
          ? max(-127, min(127, static_cast<int>(rintf(qv[i] * inv_scale))))
          : 0;
      dst[cache_row + lane + i * 32] = static_cast<signed char>(q);
    }
  }
}

)cuda"
R"cuda(
// ============================================================================
// 4. Static ShortConv k=4 + optional fused FP32 residual add (linear OR tree)
// Grid: dim3((C + 255) / 256, min(T, 64), 1), Block: dim3(256, 1, 1)
// ============================================================================
__global__ __launch_bounds__(256, 4) void ink2_sconv_k4_residual_cuda(
    const nv_bfloat16* __restrict__ x,
    const float* __restrict__ weight,
    const nv_bfloat16* __restrict__ hist_in,
    const char* __restrict__ residual,
    const int* __restrict__ conv_taps,
    nv_bfloat16* __restrict__ out_y,
    nv_bfloat16* __restrict__ out_hist,
    int T,
    int C,
    long long x_s0,
    long long res_s0,
    unsigned int flags) {
  // flags: bit0=has_res, bit1=res_is_f32, bit2=has_taps, bit3=update_hist
  const int ch = static_cast<int>(blockIdx.x) * 256 + static_cast<int>(threadIdx.x);
  if (ch >= C) {
    return;
  }
  const bool has_res = ((flags >> 0) & 1u) != 0u;
  const bool res_is_f32 = ((flags >> 1) & 1u) != 0u;
  const bool has_taps = ((flags >> 2) & 1u) != 0u;
  const bool update_hist = ((flags >> 3) & 1u) != 0u;

  constexpr float kStaticScale = 0.44721359549995793f;
  const float4 wf = reinterpret_cast<const float4*>(weight)[ch];
  const float w3 = wf.x;
  const float w2 = wf.y;
  const float w1 = wf.z;
  const float w0 = wf.w;

  auto load_tap = [&](int idx) -> float {
    if (idx < 0) {
      const int hrow = 3 + idx;
      return (hrow >= 0 && hrow < 3)
          ? __bfloat162float(hist_in[static_cast<size_t>(hrow) * static_cast<size_t>(C) + ch])
          : 0.0f;
    }
    return (idx < T)
        ? __bfloat162float(x[static_cast<size_t>(idx) * static_cast<size_t>(x_s0) + ch])
        : 0.0f;
  };

  for (int t = static_cast<int>(blockIdx.y); t < T; t += static_cast<int>(gridDim.y)) {
    int t0, t1, t2, t3;
    if (has_taps) {
      const int4 taps = reinterpret_cast<const int4*>(conv_taps)[t];
      t0 = taps.x;
      t1 = taps.y;
      t2 = taps.z;
      t3 = taps.w;
    } else {
      t0 = t - 3;
      t1 = t - 2;
      t2 = t - 1;
      t3 = t;
    }

    const float x3 = load_tap(t0);
    const float x2 = load_tap(t1);
    const float x1 = load_tap(t2);
    const float x0 = load_tap(t3);

    float acc = __fadd_rn(x0, __fmul_rn(w3, x3));
    acc = __fadd_rn(acc, __fmul_rn(w2, x2));
    acc = __fadd_rn(acc, __fmul_rn(w1, x1));
    acc = __fadd_rn(acc, __fmul_rn(w0, x0));
    const nv_bfloat16 conv_bf = __float2bfloat16_rn(acc * kStaticScale);

    const size_t out_idx = static_cast<size_t>(t) * static_cast<size_t>(C) + ch;
    if (has_res) {
      const size_t res_idx = static_cast<size_t>(t) * static_cast<size_t>(res_s0) + ch;
      const float rval = ink2_load_f1(residual, res_idx, res_is_f32);
      out_y[out_idx] = __float2bfloat16_rn(__fadd_rn(rval, __bfloat162float(conv_bf)));
    } else {
      out_y[out_idx] = conv_bf;
    }

    if (update_hist && t == T - 1) {
      out_hist[static_cast<size_t>(0) * static_cast<size_t>(C) + ch] = __float2bfloat16_rn(x2);
      out_hist[static_cast<size_t>(1) * static_cast<size_t>(C) + ch] = __float2bfloat16_rn(x1);
      out_hist[static_cast<size_t>(2) * static_cast<size_t>(C) + ch] = __float2bfloat16_rn(x0);
    }
  }
}
)cuda"
R"cuda(
// ============================================================================
// 5. High-Occupancy Small-N BF16 Projection (N in 1..=128, M in 1..=16, K%8==0)
//    Fixes sink_proj [24, 2304] and kv_conv_proj [32, 2304]: 1 CTA per output
//    row with cache-streaming 128-bit weight loads and exact FP32 accumulation.
//    Grid: dim3(N, 1, 1), Block: dim3(32, NUM_WARPS, 1).
// ============================================================================
template <int M_CAP, int NUM_WARPS>
__global__ __launch_bounds__(NUM_WARPS * 32, M_CAP <= 8 ? 4 : 2) void ink2_small_bf16_proj_cuda(
    const nv_bfloat16* __restrict__ x,
    const nv_bfloat16* __restrict__ weight,
    nv_bfloat16* __restrict__ out,
    int M,
    int N,
    int K) {
  const int n = static_cast<int>(blockIdx.x);
  if (n >= N) {
    return;
  }
  const int lane = static_cast<int>(threadIdx.x);
  const int warp = static_cast<int>(threadIdx.y);
  const int kv_end = K >> 3;
  const uint4* w_row = reinterpret_cast<const uint4*>(weight + static_cast<size_t>(n) * K);

  float acc[M_CAP];
#pragma unroll
  for (int m = 0; m < M_CAP; ++m) {
    acc[m] = 0.0f;
  }

  for (int kv = warp * 32 + lane; kv < kv_end; kv += NUM_WARPS * 32) {
    const uint4 wv = ink2_ld_cs_u4(w_row + kv);
    const __nv_bfloat162* wb = reinterpret_cast<const __nv_bfloat162*>(&wv);
    const float2 w0 = __bfloat1622float2(wb[0]);
    const float2 w1 = __bfloat1622float2(wb[1]);
    const float2 w2 = __bfloat1622float2(wb[2]);
    const float2 w3 = __bfloat1622float2(wb[3]);
#pragma unroll
    for (int m = 0; m < M_CAP; ++m) {
      if (m < M) {
        const uint4 xv = reinterpret_cast<const uint4*>(x + static_cast<size_t>(m) * K)[kv];
        const __nv_bfloat162* xb = reinterpret_cast<const __nv_bfloat162*>(&xv);
        const float2 x0 = __bfloat1622float2(xb[0]);
        const float2 x1 = __bfloat1622float2(xb[1]);
        const float2 x2 = __bfloat1622float2(xb[2]);
        const float2 x3 = __bfloat1622float2(xb[3]);
        acc[m] = fmaf(x0.x, w0.x, fmaf(x0.y, w0.y, fmaf(x1.x, w1.x, fmaf(x1.y, w1.y, acc[m]))));
        acc[m] = fmaf(x2.x, w2.x, fmaf(x2.y, w2.y, fmaf(x3.x, w3.x, fmaf(x3.y, w3.y, acc[m]))));
      }
    }
  }

  __shared__ float s_acc[NUM_WARPS][M_CAP];
#pragma unroll
  for (int m = 0; m < M_CAP; ++m) {
    if (m < M) {
      const float v = ink2_warp_sum(acc[m]);
      if (lane == 0) {
        s_acc[warp][m] = v;
      }
    }
  }
  __syncthreads();

  if (warp == 0 && lane < M) {
    float sum = 0.0f;
#pragma unroll
    for (int ww = 0; ww < NUM_WARPS; ++ww) {
      sum += s_acc[ww][lane];
    }
    out[static_cast<size_t>(lane) * N + n] = __float2bfloat16_rn(sum);
  }
}

// ============================================================================
// 6. Batched ShortConv Rollback Tape Commit (up to 96 streams per launch)
//    Handles strided rows_in slices directly -> ZERO copy_g_nd, ZERO gathers.
//    Grid: dim3(n_chunk, (C + 255) / 256, 1), Block: dim3(256, 1, 1).
// ============================================================================
struct Ink2CommitTapeBatch {
  unsigned long long hists_in[96];
  unsigned long long rows_in[96];
  unsigned long long hists_out[96];
  long long hist_strides[96];
  long long row_strides[96];
};

__global__ __launch_bounds__(256, 4) void ink2_commit_short_conv_tapes_cuda(
    Ink2CommitTapeBatch batch,
    const int* __restrict__ accepted_nodes,
    int n_chunk,
    int C,
    int T,
    int keep,
    int has_nodes) {
  const int s = static_cast<int>(blockIdx.x);
  const int c = static_cast<int>(blockIdx.y) * 256 + static_cast<int>(threadIdx.x);
  if (s >= n_chunk || c >= C) {
    return;
  }
  const nv_bfloat16* hist = reinterpret_cast<const nv_bfloat16*>(batch.hists_in[s]);
  const nv_bfloat16* rows = reinterpret_cast<const nv_bfloat16*>(batch.rows_in[s]);
  nv_bfloat16* out = reinterpret_cast<nv_bfloat16*>(batch.hists_out[s]);
  const long long hs = batch.hist_strides[s];
  const long long rs = batch.row_strides[s];

#pragma unroll
  for (int r = 0; r < 3; ++r) {
    const int src = keep - 3 + r;
    nv_bfloat16 val;
    if (src < 0) {
      const int hrow = 3 + src;
      val = (hrow >= 0 && hrow < 3)
          ? hist[static_cast<size_t>(hrow) * static_cast<size_t>(hs) + c]
          : __float2bfloat16_rn(0.0f);
    } else {
      const int node = has_nodes ? accepted_nodes[src] : src;
      val = (node >= 0 && node < T)
          ? rows[static_cast<size_t>(node) * static_cast<size_t>(rs) + c]
          : __float2bfloat16_rn(0.0f);
    }
    out[static_cast<size_t>(r) * static_cast<size_t>(C) + c] = val;
  }
}

// ============================================================================
// 7. Fused Token Embedding Lookup + Precise FP32 RMSNorm -> BF16
//    MODE: 0 = BF16 dense [V, D], 1 = SymQ8 [V, D/4] u32, 2 = AffineQ8 [V, D/4] u32
//    SB_DTYPE: 0 = F32, 1 = BF16, 2 = F16 (ignored for MODE == 0)
//    Grid: dim3(T, 1, 1), Block: dim3(256, 1, 1).
// ============================================================================
template <int MODE, int SB_DTYPE>
__global__ __launch_bounds__(256, 4) void ink2_embed_rmsnorm_cuda(
    const int* __restrict__ token_ids,
    const void* __restrict__ weight,
    const void* __restrict__ scales,
    const void* __restrict__ biases,
    const float* __restrict__ norm_weight,
    nv_bfloat16* __restrict__ out,
    int T,
    int V,
    int D,
    float eps) {
  const int t = static_cast<int>(blockIdx.x);
  if (t >= T) {
    return;
  }
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int tok = token_ids[t];
  const bool valid_tok =
      (static_cast<unsigned int>(tok) < static_cast<unsigned int>(V));

  __shared__ float s_sq[8];

  if constexpr (MODE == 0) {
    if ((D & 3) == 0) {
      const int D4 = D >> 2;
      const size_t row_off4 = static_cast<size_t>(t) * static_cast<size_t>(D4);
      const size_t w_off4 =
          valid_tok ? static_cast<size_t>(tok) * static_cast<size_t>(D4) : 0;
      const char* w_bytes = reinterpret_cast<const char*>(weight);

      float4 y0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      float4 y1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      float4 y2 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      const bool fits_regs = (D4 <= 768);
      float sq = 0.0f;
      int step = 0;
      for (int i4 = tid; i4 < D4; i4 += 256, ++step) {
        const float4 v = valid_tok
            ? ink2_load_f4(w_bytes, w_off4 + i4, false)
            : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        sq += (v.x * v.x + v.y * v.y) + (v.z * v.z + v.w * v.w);
        if (fits_regs) {
          if (step == 0) y0 = v;
          else if (step == 1) y1 = v;
          else y2 = v;
        }
      }
      sq = ink2_warp_sum(sq);
      if (lane == 0) {
        s_sq[warp] = sq;
      }
      __syncthreads();
      if (warp == 0 && lane == 0) {
        float tot = 0.0f;
#pragma unroll
        for (int w = 0; w < 8; ++w) {
          tot += s_sq[w];
        }
        s_sq[0] = rsqrtf(tot / static_cast<float>(D) + eps);
      }
      __syncthreads();
      const float inv_rms = s_sq[0];
      const float4* nw4 = reinterpret_cast<const float4*>(norm_weight);
      step = 0;
      for (int i4 = tid; i4 < D4; i4 += 256, ++step) {
        const float4 v = fits_regs
            ? ((step == 0) ? y0 : ((step == 1) ? y1 : y2))
            : (valid_tok ? ink2_load_f4(w_bytes, w_off4 + i4, false)
                         : make_float4(0.0f, 0.0f, 0.0f, 0.0f));
        const float4 nw = nw4[i4];
        ink2_store_bf16x4(
            out, row_off4 + i4,
            make_float4(
                (v.x * inv_rms) * nw.x,
                (v.y * inv_rms) * nw.y,
                (v.z * inv_rms) * nw.z,
                (v.w * inv_rms) * nw.w));
      }
    } else {
      const nv_bfloat16* w_bf = reinterpret_cast<const nv_bfloat16*>(weight);
      const size_t w_off = valid_tok ? static_cast<size_t>(tok) * static_cast<size_t>(D) : 0;
      const size_t out_off = static_cast<size_t>(t) * static_cast<size_t>(D);
      float sq = 0.0f;
      for (int d = tid; d < D; d += 256) {
        const float v = valid_tok ? __bfloat162float(w_bf[w_off + d]) : 0.0f;
        sq += v * v;
      }
      sq = ink2_warp_sum(sq);
      if (lane == 0) {
        s_sq[warp] = sq;
      }
      __syncthreads();
      if (warp == 0 && lane == 0) {
        float tot = 0.0f;
#pragma unroll
        for (int w = 0; w < 8; ++w) {
          tot += s_sq[w];
        }
        s_sq[0] = rsqrtf(tot / static_cast<float>(D) + eps);
      }
      __syncthreads();
      const float inv_rms = s_sq[0];
      for (int d = tid; d < D; d += 256) {
        const float v = valid_tok ? __bfloat162float(w_bf[w_off + d]) : 0.0f;
        out[out_off + d] = __float2bfloat16_rn((v * inv_rms) * norm_weight[d]);
      }
    }
    return;
  }

  // Q8_0 (MODE == 1: SymQ8, MODE == 2: AffineQ8), D % 32 == 0
  const int D4 = D >> 2;
  const int groups = D >> 5;
  const size_t row_off4 = static_cast<size_t>(t) * static_cast<size_t>(D4);
  const size_t w_off4 =
      valid_tok ? static_cast<size_t>(tok) * static_cast<size_t>(D4) : 0;
  const size_t sb_off =
      valid_tok ? static_cast<size_t>(tok) * static_cast<size_t>(groups) : 0;
  const unsigned int* w_u32 = reinterpret_cast<const unsigned int*>(weight);

  auto decode4 = [&](int i4) -> float4 {
    if (!valid_tok) {
      return make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    const unsigned int w = w_u32[w_off4 + i4];
    const int grp = i4 >> 3;
    const float sc = ink2_load_sb1<SB_DTYPE>(scales, sb_off + grp);
    if constexpr (MODE == 1) {
      const unsigned int sw = w ^ 0x80808080u;
      return make_float4(
          static_cast<float>(static_cast<signed char>(sw & 0xFFu)) * sc,
          static_cast<float>(static_cast<signed char>((sw >> 8) & 0xFFu)) * sc,
          static_cast<float>(static_cast<signed char>((sw >> 16) & 0xFFu)) * sc,
          static_cast<float>(static_cast<signed char>(sw >> 24)) * sc);
    } else {
      const float bi = ink2_load_sb1<SB_DTYPE>(biases, sb_off + grp);
      return make_float4(
          fmaf(static_cast<float>(w & 0xFFu), sc, bi),
          fmaf(static_cast<float>((w >> 8) & 0xFFu), sc, bi),
          fmaf(static_cast<float>((w >> 16) & 0xFFu), sc, bi),
          fmaf(static_cast<float>(w >> 24), sc, bi));
    }
  };

  float4 y0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  float4 y1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  float4 y2 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  const bool fits_regs = (D4 <= 768);
  float sq = 0.0f;
  int step = 0;
  for (int i4 = tid; i4 < D4; i4 += 256, ++step) {
    const float4 v = decode4(i4);
    sq += (v.x * v.x + v.y * v.y) + (v.z * v.z + v.w * v.w);
    if (fits_regs) {
      if (step == 0) y0 = v;
      else if (step == 1) y1 = v;
      else y2 = v;
    }
  }
  sq = ink2_warp_sum(sq);
  if (lane == 0) {
    s_sq[warp] = sq;
  }
  __syncthreads();
  if (warp == 0 && lane == 0) {
    float tot = 0.0f;
#pragma unroll
    for (int w = 0; w < 8; ++w) {
      tot += s_sq[w];
    }
    s_sq[0] = rsqrtf(tot / static_cast<float>(D) + eps);
  }
  __syncthreads();
  const float inv_rms = s_sq[0];
  const float4* nw4 = reinterpret_cast<const float4*>(norm_weight);
  step = 0;
  for (int i4 = tid; i4 < D4; i4 += 256, ++step) {
    const float4 v = fits_regs ? ((step == 0) ? y0 : ((step == 1) ? y1 : y2)) : decode4(i4);
    const float4 nw = nw4[i4];
    ink2_store_bf16x4(
        out, row_off4 + i4,
        make_float4(
            (v.x * inv_rms) * nw.x,
            (v.y * inv_rms) * nw.y,
            (v.z * inv_rms) * nw.z,
            (v.w * inv_rms) * nw.w));
  }
}
)cuda"
R"cuda(
// ============================================================================
// 8. Batched In-Place KV Cache Compaction (up to 96 caches per launch)
//    Stages all K <= 16 source rows in thread registers before writing any
//    destination slot -> alias/overlap-free, zero temp alloc, zero HtoD copy.
//    Grid: dim3(n_chunk, H_kv, 1), Block: dim3(32 or 128, 1, 1).
// ============================================================================
struct Ink2CompactKvBatch {
  unsigned long long ptrs[96];
  long long stride_h[96];
  long long stride_t[96];
  int write_offsets[96];
  int caps[96];
  unsigned int d_and_bytes[96];
};

__device__ __forceinline__ unsigned int ink2_kv_load_elem(
    unsigned long long row_ptr, int d, int elem_bytes) {
  if (elem_bytes == 2) {
    return static_cast<unsigned int>(
        reinterpret_cast<const unsigned short*>(row_ptr)[d]);
  }
  if (elem_bytes == 1) {
    return static_cast<unsigned int>(
        reinterpret_cast<const unsigned char*>(row_ptr)[d]);
  }
  return reinterpret_cast<const unsigned int*>(row_ptr)[d];
}

__device__ __forceinline__ void ink2_kv_store_elem(
    unsigned long long row_ptr, int d, int elem_bytes, unsigned int val) {
  if (elem_bytes == 2) {
    reinterpret_cast<unsigned short*>(row_ptr)[d] =
        static_cast<unsigned short>(val);
  } else if (elem_bytes == 1) {
    reinterpret_cast<unsigned char*>(row_ptr)[d] =
        static_cast<unsigned char>(val);
  } else {
    reinterpret_cast<unsigned int*>(row_ptr)[d] = val;
  }
}

__global__ __launch_bounds__(128, 4) void ink2_compact_kv_batch_cuda(
    Ink2CompactKvBatch batch,
    const int* __restrict__ accepted_nodes,
    int n_chunk,
    int H_kv,
    int K) {
  const int c_idx = static_cast<int>(blockIdx.x);
  const int h_kv = static_cast<int>(blockIdx.y);
  if (c_idx >= n_chunk || h_kv >= H_kv) {
    return;
  }
  const unsigned int db = batch.d_and_bytes[c_idx];
  const int D = static_cast<int>(db & 0xFFFFu);
  const int elem_bytes = static_cast<int>(db >> 16);
  const int base = batch.write_offsets[c_idx];
  const int cap = batch.caps[c_idx];
  const long long head_off = static_cast<long long>(h_kv) * batch.stride_h[c_idx];
  const long long st = batch.stride_t[c_idx];
  const unsigned long long base_ptr = batch.ptrs[c_idx];

  __shared__ unsigned long long s_src_ptr[16];
  __shared__ unsigned long long s_dst_ptr[16];
  const int tid = static_cast<int>(threadIdx.x);
  if (tid < K) {
    const int node = accepted_nodes[tid];
    const int src_slot = base + node;
    const int dst_slot = base + tid;
    const bool src_ok = (src_slot >= 0 && src_slot < cap);
    const bool dst_ok = (dst_slot != src_slot && dst_slot >= 0 && dst_slot < cap);
    s_src_ptr[tid] = src_ok
        ? (base_ptr +
           static_cast<unsigned long long>(
               (head_off + static_cast<long long>(src_slot) * st) * elem_bytes))
        : 0ull;
    s_dst_ptr[tid] = dst_ok
        ? (base_ptr +
           static_cast<unsigned long long>(
               (head_off + static_cast<long long>(dst_slot) * st) * elem_bytes))
        : 0ull;
  }
  __syncthreads();

  for (int d = tid; d < D; d += static_cast<int>(blockDim.x)) {
    unsigned int vals[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      if (i < K) {
        const unsigned long long sp = s_src_ptr[i];
        vals[i] = (sp != 0ull) ? ink2_kv_load_elem(sp, d, elem_bytes) : 0u;
      }
    }
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      if (i < K) {
        const unsigned long long dp = s_dst_ptr[i];
        if (dp != 0ull) {
          ink2_kv_store_elem(dp, d, elem_bytes, vals[i]);
        }
      }
    }
  }
}
)cuda"
R"cuda(
// ============================================================================
// 9. LoopCtrl Prefill Helpers (G=1 or G=3, Grid dim3(G, min(T, 65535), 1))
//    Adjacent blockIdx.x = g keeps broadcast SWA dest[t] hot in L2 across G=3.
//    9a. ink2_loop_pair_rms_norm_cuda:     [G, T, D] x 2 -> [G, T, 2*D] (BF16 or F32)
//    9b. ink2_loop_blend_rms_norm_cuda:    dest, msg, gf[2*D], norm_w -> [G, T, D] BF16
//    9c. ink2_loop_up_blend_rms_norm_cuda: fused BF16 TC up-proj + blend + RMSNorm
// ============================================================================

template <bool OUT_BF16, bool VEC4>
__global__ __launch_bounds__(256, 3) void ink2_loop_pair_rms_norm_cuda(
    const char* __restrict__ dest,
    const char* __restrict__ msg,
    void* __restrict__ out,
    int T,
    int D,
    int dest_bcast,
    int msg_bcast,
    int dest_is_f32,
    int msg_is_f32,
    float eps) {
  const int g = static_cast<int>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const bool d_f32 = (dest_is_f32 != 0);
  const bool m_f32 = (msg_is_f32 != 0);

  __shared__ float s_sq_d[8];
  __shared__ float s_sq_m[8];

  for (int t = static_cast<int>(blockIdx.y); t < T; t += static_cast<int>(gridDim.y)) {
    const size_t d_row =
        dest_bcast ? static_cast<size_t>(t)
                   : static_cast<size_t>(g) * static_cast<size_t>(T) + static_cast<size_t>(t);
    const size_t m_row =
        msg_bcast ? static_cast<size_t>(t)
                  : static_cast<size_t>(g) * static_cast<size_t>(T) + static_cast<size_t>(t);
    const size_t out_row =
        static_cast<size_t>(g) * static_cast<size_t>(T) + static_cast<size_t>(t);
    const char* d_ptr =
        dest + d_row * static_cast<size_t>(D) * (d_f32 ? 4u : 2u);
    const char* m_ptr =
        msg + m_row * static_cast<size_t>(D) * (m_f32 ? 4u : 2u);

    if constexpr (VEC4) {
      const int D4 = D >> 2;
      const size_t out_off4 = out_row * static_cast<size_t>(D4) * 2;
      const bool fits_regs = (D4 <= 768);  // covers D <= 3072 (incl. D=2304) in 1 DRAM pass

      float4 d0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      float4 d1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      float4 d2 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      float4 m0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      float4 m1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      float4 m2 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      float sq_d = 0.0f;
      float sq_m = 0.0f;

      int step = 0;
      for (int i4 = tid; i4 < D4; i4 += 256, ++step) {
        const float4 dv = ink2_load_f4(d_ptr, static_cast<size_t>(i4), d_f32);
        const float4 mv = ink2_load_f4(m_ptr, static_cast<size_t>(i4), m_f32);
        sq_d += (dv.x * dv.x + dv.y * dv.y) + (dv.z * dv.z + dv.w * dv.w);
        sq_m += (mv.x * mv.x + mv.y * mv.y) + (mv.z * mv.z + mv.w * mv.w);
        if (fits_regs) {
          if (step == 0) {
            d0 = dv;
            m0 = mv;
          } else if (step == 1) {
            d1 = dv;
            m1 = mv;
          } else {
            d2 = dv;
            m2 = mv;
          }
        }
      }
      sq_d = ink2_warp_sum(sq_d);
      sq_m = ink2_warp_sum(sq_m);
      if (lane == 0) {
        s_sq_d[warp] = sq_d;
        s_sq_m[warp] = sq_m;
      }
      __syncthreads();
      if (warp == 0 && lane == 0) {
        float tot_d = 0.0f;
        float tot_m = 0.0f;
#pragma unroll
        for (int w = 0; w < 8; ++w) {
          tot_d += s_sq_d[w];
          tot_m += s_sq_m[w];
        }
        const float inv_dim = 1.0f / static_cast<float>(D);
        s_sq_d[0] = rsqrtf(tot_d * inv_dim + eps);
        s_sq_m[0] = rsqrtf(tot_m * inv_dim + eps);
      }
      __syncthreads();
      const float inv_d = s_sq_d[0];
      const float inv_m = s_sq_m[0];

      step = 0;
      for (int i4 = tid; i4 < D4; i4 += 256, ++step) {
        const float4 dv = fits_regs
            ? ((step == 0) ? d0 : ((step == 1) ? d1 : d2))
            : ink2_load_f4(d_ptr, static_cast<size_t>(i4), d_f32);
        const float4 mv = fits_regs
            ? ((step == 0) ? m0 : ((step == 1) ? m1 : m2))
            : ink2_load_f4(m_ptr, static_cast<size_t>(i4), m_f32);
        const float4 dn = make_float4(dv.x * inv_d, dv.y * inv_d, dv.z * inv_d, dv.w * inv_d);
        const float4 mn = make_float4(mv.x * inv_m, mv.y * inv_m, mv.z * inv_m, mv.w * inv_m);
        if constexpr (OUT_BF16) {
          nv_bfloat16* out_bf = reinterpret_cast<nv_bfloat16*>(out);
          ink2_store_bf16x4(out_bf, out_off4 + i4, dn);
          ink2_store_bf16x4(out_bf, out_off4 + D4 + i4, mn);
        } else {
          float4* out_f4 = reinterpret_cast<float4*>(out);
          out_f4[out_off4 + i4] = dn;
          out_f4[out_off4 + D4 + i4] = mn;
        }
      }
    } else {
      const size_t out_off = out_row * static_cast<size_t>(D) * 2;
      float sq_d = 0.0f;
      float sq_m = 0.0f;
      for (int i = tid; i < D; i += 256) {
        const float dv = ink2_load_f1(d_ptr, static_cast<size_t>(i), d_f32);
        const float mv = ink2_load_f1(m_ptr, static_cast<size_t>(i), m_f32);
        sq_d += dv * dv;
        sq_m += mv * mv;
      }
      sq_d = ink2_warp_sum(sq_d);
      sq_m = ink2_warp_sum(sq_m);
      if (lane == 0) {
        s_sq_d[warp] = sq_d;
        s_sq_m[warp] = sq_m;
      }
      __syncthreads();
      if (warp == 0 && lane == 0) {
        float tot_d = 0.0f;
        float tot_m = 0.0f;
#pragma unroll
        for (int w = 0; w < 8; ++w) {
          tot_d += s_sq_d[w];
          tot_m += s_sq_m[w];
        }
        const float inv_dim = 1.0f / static_cast<float>(D);
        s_sq_d[0] = rsqrtf(tot_d * inv_dim + eps);
        s_sq_m[0] = rsqrtf(tot_m * inv_dim + eps);
      }
      __syncthreads();
      const float inv_d = s_sq_d[0];
      const float inv_m = s_sq_m[0];
      for (int i = tid; i < D; i += 256) {
        const float dn = ink2_load_f1(d_ptr, static_cast<size_t>(i), d_f32) * inv_d;
        const float mn = ink2_load_f1(m_ptr, static_cast<size_t>(i), m_f32) * inv_m;
        if constexpr (OUT_BF16) {
          nv_bfloat16* out_bf = reinterpret_cast<nv_bfloat16*>(out);
          out_bf[out_off + i] = __float2bfloat16_rn(dn);
          out_bf[out_off + D + i] = __float2bfloat16_rn(mn);
        } else {
          float* out_f = reinterpret_cast<float*>(out);
          out_f[out_off + i] = dn;
          out_f[out_off + D + i] = mn;
        }
      }
    }
    if (t + static_cast<int>(gridDim.y) < T) {
      __syncthreads();
    }
  }
}

__device__ __forceinline__ float ink2_loop_blend_elem(
    float dv, float mv, float gain_raw, float forget_raw) {
  const float g_val = tanhf(fminf(fmaxf(gain_raw, -20.0f), 20.0f));
  const float f_val = tanhf(fminf(fmaxf(forget_raw, -20.0f), 20.0f));
  const float kept = __fmul_rn(1.0f - f_val, dv);
  const float added = __fmul_rn(g_val, mv);
  return __fadd_rn(kept, added);
}

template <bool VEC4>
__global__ __launch_bounds__(256, 3) void ink2_loop_blend_rms_norm_cuda(
    const char* __restrict__ dest,
    const char* __restrict__ msg,
    const char* __restrict__ gf,
    const float* __restrict__ norm_w,
    nv_bfloat16* __restrict__ out,
    int T,
    int D,
    int dest_bcast,
    int msg_bcast,
    int norm_w_bcast,
    int dest_is_f32,
    int msg_is_f32,
    int gf_is_f32,
    float eps) {
  const int g = static_cast<int>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const bool d_f32 = (dest_is_f32 != 0);
  const bool m_f32 = (msg_is_f32 != 0);
  const bool g_f32 = (gf_is_f32 != 0);

  __shared__ float s_sq[8];

  for (int t = static_cast<int>(blockIdx.y); t < T; t += static_cast<int>(gridDim.y)) {
    const size_t d_row =
        dest_bcast ? static_cast<size_t>(t)
                   : static_cast<size_t>(g) * static_cast<size_t>(T) + static_cast<size_t>(t);
    const size_t m_row =
        msg_bcast ? static_cast<size_t>(t)
                  : static_cast<size_t>(g) * static_cast<size_t>(T) + static_cast<size_t>(t);
    const size_t out_row =
        static_cast<size_t>(g) * static_cast<size_t>(T) + static_cast<size_t>(t);
    const char* d_ptr =
        dest + d_row * static_cast<size_t>(D) * (d_f32 ? 4u : 2u);
    const char* m_ptr =
        msg + m_row * static_cast<size_t>(D) * (m_f32 ? 4u : 2u);
    const char* gf_ptr =
        gf + out_row * static_cast<size_t>(D) * 2 * (g_f32 ? 4u : 2u);

    if constexpr (VEC4) {
      const int D4 = D >> 2;
      const size_t out_off4 = out_row * static_cast<size_t>(D4);
      const bool fits_regs = (D4 <= 1024);  // covers D <= 4096 (incl. D=2304) in registers

      float4 b0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      float4 b1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      float4 b2 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      float4 b3 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      float sq = 0.0f;

      int step = 0;
      for (int i4 = tid; i4 < D4; i4 += 256, ++step) {
        const float4 dv = ink2_load_f4(d_ptr, static_cast<size_t>(i4), d_f32);
        const float4 mv = ink2_load_f4(m_ptr, static_cast<size_t>(i4), m_f32);
        const float4 gv = ink2_load_f4(gf_ptr, static_cast<size_t>(i4), g_f32);
        const float4 fv = ink2_load_f4(gf_ptr, static_cast<size_t>(D4 + i4), g_f32);
        const float4 b = make_float4(
            ink2_loop_blend_elem(dv.x, mv.x, gv.x, fv.x),
            ink2_loop_blend_elem(dv.y, mv.y, gv.y, fv.y),
            ink2_loop_blend_elem(dv.z, mv.z, gv.z, fv.z),
            ink2_loop_blend_elem(dv.w, mv.w, gv.w, fv.w));
        sq += (b.x * b.x + b.y * b.y) + (b.z * b.z + b.w * b.w);
        if (fits_regs) {
          if (step == 0) b0 = b;
          else if (step == 1) b1 = b;
          else if (step == 2) b2 = b;
          else b3 = b;
        }
      }
      sq = ink2_warp_sum(sq);
      if (lane == 0) {
        s_sq[warp] = sq;
      }
      __syncthreads();
      if (warp == 0 && lane == 0) {
        float total = 0.0f;
#pragma unroll
        for (int w = 0; w < 8; ++w) {
          total += s_sq[w];
        }
        s_sq[0] = rsqrtf(total / static_cast<float>(D) + eps);
      }
      __syncthreads();
      const float inv_rms = s_sq[0];

      const float4* w4 = reinterpret_cast<const float4*>(norm_w) +
          (norm_w_bcast ? 0 : static_cast<size_t>(g) * static_cast<size_t>(D4));
      step = 0;
      for (int i4 = tid; i4 < D4; i4 += 256, ++step) {
        float4 b;
        if (fits_regs) {
          b = (step == 0) ? b0 : ((step == 1) ? b1 : ((step == 2) ? b2 : b3));
        } else {
          const float4 dv = ink2_load_f4(d_ptr, static_cast<size_t>(i4), d_f32);
          const float4 mv = ink2_load_f4(m_ptr, static_cast<size_t>(i4), m_f32);
          const float4 gv = ink2_load_f4(gf_ptr, static_cast<size_t>(i4), g_f32);
          const float4 fv = ink2_load_f4(gf_ptr, static_cast<size_t>(D4 + i4), g_f32);
          b = make_float4(
              ink2_loop_blend_elem(dv.x, mv.x, gv.x, fv.x),
              ink2_loop_blend_elem(dv.y, mv.y, gv.y, fv.y),
              ink2_loop_blend_elem(dv.z, mv.z, gv.z, fv.z),
              ink2_loop_blend_elem(dv.w, mv.w, gv.w, fv.w));
        }
        const float4 wv = w4[i4];
        const float4 res = make_float4(
            (b.x * inv_rms) * wv.x,
            (b.y * inv_rms) * wv.y,
            (b.z * inv_rms) * wv.z,
            (b.w * inv_rms) * wv.w);
        ink2_store_bf16x4(out, out_off4 + i4, res);
      }
    } else {
      const size_t out_off = out_row * static_cast<size_t>(D);
      float sq = 0.0f;
      for (int i = tid; i < D; i += 256) {
        const float dv = ink2_load_f1(d_ptr, static_cast<size_t>(i), d_f32);
        const float mv = ink2_load_f1(m_ptr, static_cast<size_t>(i), m_f32);
        const float gv = ink2_load_f1(gf_ptr, static_cast<size_t>(i), g_f32);
        const float fv = ink2_load_f1(gf_ptr, static_cast<size_t>(D + i), g_f32);
        const float b = ink2_loop_blend_elem(dv, mv, gv, fv);
        sq += b * b;
      }
      sq = ink2_warp_sum(sq);
      if (lane == 0) {
        s_sq[warp] = sq;
      }
      __syncthreads();
      if (warp == 0 && lane == 0) {
        float total = 0.0f;
#pragma unroll
        for (int w = 0; w < 8; ++w) {
          total += s_sq[w];
        }
        s_sq[0] = rsqrtf(total / static_cast<float>(D) + eps);
      }
      __syncthreads();
      const float inv_rms = s_sq[0];
      const float* w_row = norm_w +
          (norm_w_bcast ? 0 : static_cast<size_t>(g) * static_cast<size_t>(D));
      for (int i = tid; i < D; i += 256) {
        const float dv = ink2_load_f1(d_ptr, static_cast<size_t>(i), d_f32);
        const float mv = ink2_load_f1(m_ptr, static_cast<size_t>(i), m_f32);
        const float gv = ink2_load_f1(gf_ptr, static_cast<size_t>(i), g_f32);
        const float fv = ink2_load_f1(gf_ptr, static_cast<size_t>(D + i), g_f32);
        const float b = ink2_loop_blend_elem(dv, mv, gv, fv);
        out[out_off + i] = __float2bfloat16_rn((b * inv_rms) * w_row[i]);
      }
    }
    if (t + static_cast<int>(gridDim.y) < T) {
      __syncthreads();
    }
  }
}
)cuda"
R"cuda(
// ============================================================================
// 9c. Fused LoopCtrl Up-Proj (BF16 TC MMA) + Tanh Gate + Exact No-FMA Blend
//     + Sublayer RMSNorm -> BF16 (G=1 or G=3, rank=64, D <= 2304, D % 16 == 0)
//     Eliminates [G, T, 2*D] gf DRAM materialization and [G, T, D] F32 blended.
//     Grid: dim3(G, (T + 7) / 8, 1), Block: dim3(32, 8, 1)
//     Dynamic SMEM: s_blend[8][D + 4] + s_h[8][72] + s_sq[8][8] + s_inv[8]
// ============================================================================
__global__ __launch_bounds__(256, 1) void ink2_loop_up_blend_rms_norm_cuda(
    const char* __restrict__ dest,
    const char* __restrict__ msg,
    const char* __restrict__ h,
    const nv_bfloat16* __restrict__ gf_w,
    const float* __restrict__ norm_w,
    nv_bfloat16* __restrict__ out,
    int T,
    int D,
    int dest_bcast,
    int msg_bcast,
    int norm_w_bcast,
    int dest_is_f32,
    int msg_is_f32,
    int h_is_f32,
    float eps) {
  const int g = static_cast<int>(blockIdx.x);
  const int t0 = static_cast<int>(blockIdx.y) * 8;
  if (t0 >= T) {
    return;
  }
  const int n_rows = (T - t0 < 8) ? (T - t0) : 8;
  const int lane = static_cast<int>(threadIdx.x);
  const int warp = static_cast<int>(threadIdx.y);
  const int tid = warp * 32 + lane;

  extern __shared__ char smem_raw[];
  const int blend_stride = D + 4;
  const size_t blend_bytes =
      static_cast<size_t>(8) * static_cast<size_t>(blend_stride) * sizeof(float);
  constexpr size_t kHBytes = 8 * 72 * sizeof(nv_bfloat16);
  constexpr size_t kSqBytes = 8 * 8 * sizeof(float);

  float* s_blend = reinterpret_cast<float*>(smem_raw);
  nv_bfloat16* s_h = reinterpret_cast<nv_bfloat16*>(smem_raw + blend_bytes);
  float* s_sq = reinterpret_cast<float*>(smem_raw + blend_bytes + kHBytes);
  float* s_inv =
      reinterpret_cast<float*>(smem_raw + blend_bytes + kHBytes + kSqBytes);

  // Cooperatively load h[g, t0..t0+8, 0..64] into padded s_h[8][72] (0 bank conflicts).
  const bool h_f32 = (h_is_f32 != 0);
  if (tid < 128) {
    const int r = tid >> 4;
    const int c4 = tid & 15;
    nv_bfloat16* sh_row = s_h + r * 72;
    if (r < n_rows) {
      const size_t h_row =
          static_cast<size_t>(g) * static_cast<size_t>(T) +
          static_cast<size_t>(t0 + r);
      if (h_f32) {
        const float4 hv = reinterpret_cast<const float4*>(h)[h_row * 16 + c4];
        ink2_store_bf16x4(sh_row, static_cast<size_t>(c4), hv);
      } else {
        reinterpret_cast<uint2*>(sh_row)[c4] =
            reinterpret_cast<const uint2*>(h)[h_row * 16 + c4];
      }
    } else {
      reinterpret_cast<uint2*>(sh_row)[c4] = make_uint2(0u, 0u);
    }
  }
  __syncthreads();

  // Preload this lane's 8 uint32 B-fragments from s_h[g_m][0..63] once for all c0 tiles.
  const int g_m = lane >> 2;
  const int t_m = lane & 3;
  const int tok0 = 2 * t_m;
  const int tok1 = tok0 + 1;
  const unsigned int* hb =
      reinterpret_cast<const unsigned int*>(s_h + g_m * 72) + t_m;
  const unsigned int b0_0 = hb[0];
  const unsigned int b1_0 = hb[4];
  const unsigned int b0_1 = hb[8];
  const unsigned int b1_1 = hb[12];
  const unsigned int b0_2 = hb[16];
  const unsigned int b1_2 = hb[20];
  const unsigned int b0_3 = hb[24];
  const unsigned int b1_3 = hb[28];

  const bool d_f32 = (dest_is_f32 != 0);
  const bool m_f32 = (msg_is_f32 != 0);
  const size_t d_row_base0 =
      (dest_bcast ? static_cast<size_t>(t0)
                  : static_cast<size_t>(g) * static_cast<size_t>(T) +
                        static_cast<size_t>(t0)) *
      static_cast<size_t>(D);
  const size_t m_row_base0 =
      (msg_bcast ? static_cast<size_t>(t0)
                 : static_cast<size_t>(g) * static_cast<size_t>(T) +
                       static_cast<size_t>(t0)) *
      static_cast<size_t>(D);
  const size_t d_off0 =
      d_row_base0 + static_cast<size_t>(tok0) * static_cast<size_t>(D);
  const size_t m_off0 =
      m_row_base0 + static_cast<size_t>(tok0) * static_cast<size_t>(D);
  const size_t d_off1 =
      d_row_base0 + static_cast<size_t>(tok1) * static_cast<size_t>(D);
  const size_t m_off1 =
      m_row_base0 + static_cast<size_t>(tok1) * static_cast<size_t>(D);

  const unsigned int* gf_g_u32 =
      reinterpret_cast<const unsigned int*>(gf_w) +
      static_cast<size_t>(g) * static_cast<size_t>(D) * 64 + t_m;
  const unsigned int* gf_f_u32 = gf_g_u32 + static_cast<size_t>(D) * 32;

  float sq_even = 0.0f;
  float sq_odd = 0.0f;

  // Pass 1: MMA up-proj (gain & forget) + tanh + exact no-FMA blend -> s_blend + sum_sq.
  for (int c0 = warp * 16; c0 < D; c0 += 128) {
    const int r0 = c0 + g_m;
    const int r1 = r0 + 8;

    const unsigned int* wg0 = gf_g_u32 + static_cast<size_t>(r0) * 32;
    const unsigned int* wg1 = gf_g_u32 + static_cast<size_t>(r1) * 32;
    float dg0 = 0.0f, dg1 = 0.0f, dg2 = 0.0f, dg3 = 0.0f;
    ink2_mma_m16n8k16_bf16(
        dg0, dg1, dg2, dg3, wg0[0], wg1[0], wg0[4], wg1[4], b0_0, b1_0);
    ink2_mma_m16n8k16_bf16(
        dg0, dg1, dg2, dg3, wg0[8], wg1[8], wg0[12], wg1[12], b0_1, b1_1);
    ink2_mma_m16n8k16_bf16(
        dg0, dg1, dg2, dg3, wg0[16], wg1[16], wg0[20], wg1[20], b0_2, b1_2);
    ink2_mma_m16n8k16_bf16(
        dg0, dg1, dg2, dg3, wg0[24], wg1[24], wg0[28], wg1[28], b0_3, b1_3);

    const unsigned int* wf0 = gf_f_u32 + static_cast<size_t>(r0) * 32;
    const unsigned int* wf1 = gf_f_u32 + static_cast<size_t>(r1) * 32;
    float df0 = 0.0f, df1 = 0.0f, df2 = 0.0f, df3 = 0.0f;
    ink2_mma_m16n8k16_bf16(
        df0, df1, df2, df3, wf0[0], wf1[0], wf0[4], wf1[4], b0_0, b1_0);
    ink2_mma_m16n8k16_bf16(
        df0, df1, df2, df3, wf0[8], wf1[8], wf0[12], wf1[12], b0_1, b1_1);
    ink2_mma_m16n8k16_bf16(
        df0, df1, df2, df3, wf0[16], wf1[16], wf0[20], wf1[20], b0_2, b1_2);
    ink2_mma_m16n8k16_bf16(
        df0, df1, df2, df3, wf0[24], wf1[24], wf0[28], wf1[28], b0_3, b1_3);

    if (tok0 < n_rows) {
      const float dv0 = ink2_load_f1(dest, d_off0 + static_cast<size_t>(r0), d_f32);
      const float mv0 = ink2_load_f1(msg, m_off0 + static_cast<size_t>(r0), m_f32);
      const float b0 = ink2_loop_blend_elem(dv0, mv0, dg0, df0);
      s_blend[tok0 * blend_stride + r0] = b0;
      sq_even = fmaf(b0, b0, sq_even);

      const float dv2 = ink2_load_f1(dest, d_off0 + static_cast<size_t>(r1), d_f32);
      const float mv2 = ink2_load_f1(msg, m_off0 + static_cast<size_t>(r1), m_f32);
      const float b2 = ink2_loop_blend_elem(dv2, mv2, dg2, df2);
      s_blend[tok0 * blend_stride + r1] = b2;
      sq_even = fmaf(b2, b2, sq_even);
    }
    if (tok1 < n_rows) {
      const float dv1 = ink2_load_f1(dest, d_off1 + static_cast<size_t>(r0), d_f32);
      const float mv1 = ink2_load_f1(msg, m_off1 + static_cast<size_t>(r0), m_f32);
      const float b1 = ink2_loop_blend_elem(dv1, mv1, dg1, df1);
      s_blend[tok1 * blend_stride + r0] = b1;
      sq_odd = fmaf(b1, b1, sq_odd);

      const float dv3 = ink2_load_f1(dest, d_off1 + static_cast<size_t>(r1), d_f32);
      const float mv3 = ink2_load_f1(msg, m_off1 + static_cast<size_t>(r1), m_f32);
      const float b3 = ink2_loop_blend_elem(dv3, mv3, dg3, df3);
      s_blend[tok1 * blend_stride + r1] = b3;
      sq_odd = fmaf(b3, b3, sq_odd);
    }
  }

  // Reduce sq_even / sq_odd across the 8 g_m lanes sharing the same t_m.
  sq_even += __shfl_xor_sync(0xffffffffu, sq_even, 16);
  sq_even += __shfl_xor_sync(0xffffffffu, sq_even, 8);
  sq_even += __shfl_xor_sync(0xffffffffu, sq_even, 4);
  sq_odd += __shfl_xor_sync(0xffffffffu, sq_odd, 16);
  sq_odd += __shfl_xor_sync(0xffffffffu, sq_odd, 8);
  sq_odd += __shfl_xor_sync(0xffffffffu, sq_odd, 4);
  if (g_m == 0) {
    s_sq[tok0 * 8 + warp] = sq_even;
    s_sq[tok1 * 8 + warp] = sq_odd;
  }
  __syncthreads();

  if (warp == 0 && lane < 8) {
    float tot = 0.0f;
#pragma unroll
    for (int w = 0; w < 8; ++w) {
      tot += s_sq[lane * 8 + w];
    }
    s_inv[lane] = rsqrtf(tot / static_cast<float>(D) + eps);
  }
  __syncthreads();

  // Pass 2: RMSNorm + coalesced BF16x4 store.
  const int D4 = D >> 2;
  const float4* w4 = reinterpret_cast<const float4*>(norm_w) +
      (norm_w_bcast ? 0 : static_cast<size_t>(g) * static_cast<size_t>(D4));
  const size_t out_base4 =
      (static_cast<size_t>(g) * static_cast<size_t>(T) + static_cast<size_t>(t0)) *
      static_cast<size_t>(D4);
  for (int tok = 0; tok < n_rows; ++tok) {
    const float inv_rms = s_inv[tok];
    const float4* b4 =
        reinterpret_cast<const float4*>(s_blend + tok * blend_stride);
    const size_t row_out4 =
        out_base4 + static_cast<size_t>(tok) * static_cast<size_t>(D4);
    for (int i4 = tid; i4 < D4; i4 += 256) {
      const float4 b = b4[i4];
      const float4 wv = w4[i4];
      const float4 res = make_float4(
          (b.x * inv_rms) * wv.x,
          (b.y * inv_rms) * wv.y,
          (b.z * inv_rms) * wv.z,
          (b.w * inv_rms) * wv.w);
      ink2_store_bf16x4(out, row_out4 + i4, res);
    }
  }
}
)cuda"
