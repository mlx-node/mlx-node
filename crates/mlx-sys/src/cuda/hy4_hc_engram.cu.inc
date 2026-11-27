R"cuda(
// ============================================================================
// Hy4-Flash Structural Kernels:
//   1. 4-Stream Hyper-Connections Pre / Head (NVFP4, SymQ8, F32, BF16 W_fn)
//   2. 4-Stream Hyper-Connections Post (Post-RMSNorm + Residual Add)
//   3. Engram Lookup Norm (per-column RMSNorm, kEngramEps)
//   4. Engram Gate + Residual Add (per-stream gated delta, kEngramEps)
//   5. MTP eh_norm_concat ([rms_norm(emb, enorm) | rms_norm(hidden, hnorm)])
// ============================================================================

__device__ __forceinline__ float hy4_sigmoid_f32(float x) {
  return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float hy4_sigmoid_fast(float x) {
  return __frcp_rn(1.0f + __expf(-x));
}

__device__ __forceinline__ float hy4_read_norm_w(
    const void* __restrict__ w, int idx, int is_bf16) {
  if (is_bf16) {
    return __bfloat162float(reinterpret_cast<const nv_bfloat16*>(w)[idx]);
  }
  return reinterpret_cast<const float*>(w)[idx];
}

__device__ __forceinline__ void hy4_unpack_bf16x8(const uint4& v, float out[8]) {
  const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 f = __bfloat1622float2(p[i]);
    out[2 * i] = f.x;
    out[2 * i + 1] = f.y;
  }
}

__device__ __forceinline__ uint4 hy4_pack_bf16x8(const float v[8]) {
  uint4 out;
  out.x = hy4_pack_bf16x2(v[0], v[1]);
  out.y = hy4_pack_bf16x2(v[2], v[3]);
  out.z = hy4_pack_bf16x2(v[4], v[5]);
  out.w = hy4_pack_bf16x2(v[6], v[7]);
  return out;
}

__device__ __forceinline__ void hy4_load_norm_w8(
    const void* __restrict__ w, int j8, int is_bf16, float out[8]) {
  if (is_bf16) {
    hy4_unpack_bf16x8(reinterpret_cast<const uint4*>(w)[j8], out);
  } else {
    const float4* wf = reinterpret_cast<const float4*>(w);
    const float4 a = wf[2 * j8];
    const float4 b = wf[2 * j8 + 1];
    out[0] = a.x; out[1] = a.y; out[2] = a.z; out[3] = a.w;
    out[4] = b.x; out[5] = b.y; out[6] = b.z; out[7] = b.w;
  }
}

__device__ __forceinline__ float2 hy4_load_norm_w2(
    const void* __restrict__ w, int j2, int is_bf16) {
  if (is_bf16) {
    const unsigned int u = reinterpret_cast<const unsigned int*>(w)[j2];
    return __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u));
  }
  return reinterpret_cast<const float2*>(w)[j2];
}

__device__ __forceinline__ float hy4_dot_bf16x8(const uint4& wu, const float xf[8]) {
  const __nv_bfloat162* wp = reinterpret_cast<const __nv_bfloat162*>(&wu);
  float d = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 wf = __bfloat1622float2(wp[i]);
    d = fmaf(wf.x, xf[2 * i], fmaf(wf.y, xf[2 * i + 1], d));
  }
  return d;
}

__device__ __forceinline__ float hy4_dot_e2m1_u2(uint2 w2, const float xf[16]) {
  float d = 0.0f;
  unsigned int u = w2.x;
  float2 f;
  f = hy4_e2m1x2_to_f2(u & 0xFFu);
  d = fmaf(f.x, xf[0], fmaf(f.y, xf[1], d));
  f = hy4_e2m1x2_to_f2((u >> 8) & 0xFFu);
  d = fmaf(f.x, xf[2], fmaf(f.y, xf[3], d));
  f = hy4_e2m1x2_to_f2((u >> 16) & 0xFFu);
  d = fmaf(f.x, xf[4], fmaf(f.y, xf[5], d));
  f = hy4_e2m1x2_to_f2((u >> 24) & 0xFFu);
  d = fmaf(f.x, xf[6], fmaf(f.y, xf[7], d));

  u = w2.y;
  f = hy4_e2m1x2_to_f2(u & 0xFFu);
  d = fmaf(f.x, xf[8], fmaf(f.y, xf[9], d));
  f = hy4_e2m1x2_to_f2((u >> 8) & 0xFFu);
  d = fmaf(f.x, xf[10], fmaf(f.y, xf[11], d));
  f = hy4_e2m1x2_to_f2((u >> 16) & 0xFFu);
  d = fmaf(f.x, xf[12], fmaf(f.y, xf[13], d));
  f = hy4_e2m1x2_to_f2((u >> 24) & 0xFFu);
  d = fmaf(f.x, xf[14], fmaf(f.y, xf[15], d));
  return d;
}

__device__ __forceinline__ float hy4_dot_symq8_u32(
    unsigned int w, float x0, float x1, float x2, float x3) {
  const unsigned int sw = w ^ 0x80808080u;
  float d = static_cast<float>(static_cast<signed char>(sw & 0xFFu)) * x0;
  d = fmaf(static_cast<float>(static_cast<signed char>((sw >> 8) & 0xFFu)), x1, d);
  d = fmaf(static_cast<float>(static_cast<signed char>((sw >> 16) & 0xFFu)), x2, d);
  d = fmaf(static_cast<float>(static_cast<signed char>((sw >> 24) & 0xFFu)), x3, d);
  return d;
}

)cuda"
R"cuda(
// ----------------------------------------------------------------------------
// Kernel 1: hy4_hc_pre_cuda (4-Stream Hyper-Connections Pre / Head)
// FN_MODE: 0 = NVFP4 (u32 [out,K/8], u8 [out,K/16])
//          1 = SymQ8 (u32 [out,K/4], f16 [out,K/32], s8 = byte ^ 0x80)
//          2 = F32   (f32 [out,K])
//          3 = BF16  (bf16 [out,K])
// Grid: dim3(M, 1, 1), Block: dim3(256, 1, 1)
// ----------------------------------------------------------------------------
template <int FN_MODE, bool HEAD_ONLY>
__global__ __launch_bounds__(256, 2) void hy4_hc_pre_cuda(
    const nv_bfloat16* __restrict__ x,
    const void* __restrict__ w_fn,
    const void* __restrict__ s_fn,
    const float* __restrict__ scale,
    const float* __restrict__ base,
    const void* __restrict__ norm_w,
    nv_bfloat16* __restrict__ out_h,
    float* __restrict__ out_post,
    int M,
    int HC,
    int D,
    float norm_eps,
    float hc_eps,
    float hc_mag,
    int has_norm,
    int norm_bf16) {
  const int m = static_cast<int>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;
  if (m >= M) {
    return;
  }

  const int K = HC * D;
  const int n_gates = HEAD_ONLY ? HC : (2 * HC);
  constexpr int MAX_GATES = HEAD_ONLY ? 8 : 16;
  const size_t x_row_off = static_cast<size_t>(m) * static_cast<size_t>(K);
  const nv_bfloat16* x_row = x + x_row_off;

  float sq_acc = 0.0f;
  float dot_acc[MAX_GATES];
#pragma unroll
  for (int r = 0; r < MAX_GATES; ++r) {
    dot_acc[r] = 0.0f;
  }

  if constexpr (FN_MODE == 0) {
    const int n_grp16 = K >> 4;
    const size_t w_stride_u2 = static_cast<size_t>(n_grp16);
    const size_t s_stride = static_cast<size_t>(n_grp16);
    const uint2* w_u2 = reinterpret_cast<const uint2*>(w_fn);
    const unsigned char* s_u8 = reinterpret_cast<const unsigned char*>(s_fn);
    const uint4* x_u4 = reinterpret_cast<const uint4*>(x_row);
    for (int g = tid; g < n_grp16; g += 256) {
      float xf[16];
      hy4_unpack_bf16x8(x_u4[2 * g], xf);
      hy4_unpack_bf16x8(x_u4[2 * g + 1], xf + 8);
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        sq_acc = fmaf(xf[i], xf[i], sq_acc);
      }
#pragma unroll
      for (int r = 0; r < MAX_GATES; ++r) {
        if (r < n_gates) {
          const uint2 w2 = hy4_ld_cs_u2(w_u2 + static_cast<size_t>(r) * w_stride_u2 + g);
          const float sc = hy4_fast_e4m3_to_f(s_u8[static_cast<size_t>(r) * s_stride + g]);
          dot_acc[r] = fmaf(sc, hy4_dot_e2m1_u2(w2, xf), dot_acc[r]);
        }
      }
    }
  } else if constexpr (FN_MODE == 1) {
    const int n_grp16 = K >> 4;
    const size_t w_stride_u4 = static_cast<size_t>(n_grp16);
    const size_t s_stride = static_cast<size_t>(K >> 5);
    const uint4* w_u4 = reinterpret_cast<const uint4*>(w_fn);
    const __half* s_f16 = reinterpret_cast<const __half*>(s_fn);
    const uint4* x_u4 = reinterpret_cast<const uint4*>(x_row);
    for (int g = tid; g < n_grp16; g += 256) {
      float xf[16];
      hy4_unpack_bf16x8(x_u4[2 * g], xf);
      hy4_unpack_bf16x8(x_u4[2 * g + 1], xf + 8);
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        sq_acc = fmaf(xf[i], xf[i], sq_acc);
      }
      const int sg = g >> 1;
#pragma unroll
      for (int r = 0; r < MAX_GATES; ++r) {
        if (r < n_gates) {
          const uint4 w4 = w_u4[static_cast<size_t>(r) * w_stride_u4 + g];
          const float sc = __half2float(s_f16[static_cast<size_t>(r) * s_stride + sg]);
          float d = hy4_dot_symq8_u32(w4.x, xf[0], xf[1], xf[2], xf[3]);
          d += hy4_dot_symq8_u32(w4.y, xf[4], xf[5], xf[6], xf[7]);
          d += hy4_dot_symq8_u32(w4.z, xf[8], xf[9], xf[10], xf[11]);
          d += hy4_dot_symq8_u32(w4.w, xf[12], xf[13], xf[14], xf[15]);
          dot_acc[r] = fmaf(sc, d, dot_acc[r]);
        }
      }
    }
  } else if constexpr (FN_MODE == 2) {
    const int n_c8 = K >> 3;
    const size_t w_stride_f4 = static_cast<size_t>(K >> 2);
    const float4* w_f4 = reinterpret_cast<const float4*>(w_fn);
    const uint4* x_u4 = reinterpret_cast<const uint4*>(x_row);
    for (int c8 = tid; c8 < n_c8; c8 += 256) {
      float xf[8];
      hy4_unpack_bf16x8(x_u4[c8], xf);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        sq_acc = fmaf(xf[i], xf[i], sq_acc);
      }
#pragma unroll
      for (int r = 0; r < MAX_GATES; ++r) {
        if (r < n_gates) {
          const float4* wp = w_f4 + static_cast<size_t>(r) * w_stride_f4 + static_cast<size_t>(c8) * 2;
          const float4 w0 = wp[0];
          const float4 w1 = wp[1];
          float d = fmaf(w0.x, xf[0], fmaf(w0.y, xf[1], fmaf(w0.z, xf[2], w0.w * xf[3])));
          d = fmaf(w1.x, xf[4], fmaf(w1.y, xf[5], fmaf(w1.z, xf[6], fmaf(w1.w, xf[7], d))));
          dot_acc[r] += d;
        }
      }
    }
  } else {
    const int n_c8 = K >> 3;
    const size_t w_stride_u4 = static_cast<size_t>(n_c8);
    const uint4* w_u4 = reinterpret_cast<const uint4*>(w_fn);
    const uint4* x_u4 = reinterpret_cast<const uint4*>(x_row);
    for (int c8 = tid; c8 < n_c8; c8 += 256) {
      float xf[8];
      hy4_unpack_bf16x8(x_u4[c8], xf);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        sq_acc = fmaf(xf[i], xf[i], sq_acc);
      }
#pragma unroll
      for (int r = 0; r < MAX_GATES; ++r) {
        if (r < n_gates) {
          const uint4 wv = w_u4[static_cast<size_t>(r) * w_stride_u4 + c8];
          float wf[8];
          hy4_unpack_bf16x8(wv, wf);
          float d = 0.0f;
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            d = fmaf(wf[i], xf[i], d);
          }
          dot_acc[r] += d;
        }
      }
    }
  }

  __shared__ float s_red[8][17];
  __shared__ float s_pre[8];
  __shared__ float s_h2[8];

  sq_acc = hy4_warp_sum(sq_acc);
  if (lane == 0) {
    s_red[warp][0] = sq_acc;
  }
#pragma unroll
  for (int r = 0; r < MAX_GATES; ++r) {
    if (r < n_gates) {
      const float ds = hy4_warp_sum(dot_acc[r]);
      if (lane == 0) {
        s_red[warp][1 + r] = ds;
      }
    }
  }
  __syncthreads();

  if (tid == 0) {
    float sq_tot = 0.0f;
#pragma unroll
    for (int w = 0; w < 8; ++w) {
      sq_tot += s_red[w][0];
    }
    const float rs = rsqrtf(sq_tot / static_cast<float>(K) + norm_eps);
    const float sc0 = scale[0];
    for (int c = 0; c < HC; ++c) {
      float dt = 0.0f;
#pragma unroll
      for (int w = 0; w < 8; ++w) {
        dt += s_red[w][1 + c];
      }
      const float z = (dt * rs) * sc0 + base[c];
      s_pre[c] = hy4_sigmoid_f32(z) + hc_eps;
    }
    if constexpr (!HEAD_ONLY) {
      const float sc1 = scale[1];
      float* p_dst = out_post + static_cast<size_t>(m) * static_cast<size_t>(HC);
      for (int c = 0; c < HC; ++c) {
        float dt = 0.0f;
#pragma unroll
        for (int w = 0; w < 8; ++w) {
          dt += s_red[w][1 + HC + c];
        }
        const float z = (dt * rs) * sc1 + base[HC + c];
        p_dst[c] = hc_mag * hy4_sigmoid_f32(z) + hc_eps;
      }
    }
  }
  __syncthreads();

  const int d8 = D >> 3;
  uint4* h_out_u4 = reinterpret_cast<uint4*>(out_h + static_cast<size_t>(m) * static_cast<size_t>(D));

  if (d8 <= 512) {
    float h_reg[2][8];
    float h2 = 0.0f;
#pragma unroll
    for (int k = 0; k < 2; ++k) {
      const int j8 = tid + k * 256;
      if (j8 < d8) {
#pragma unroll
        for (int i = 0; i < 8; ++i) h_reg[k][i] = 0.0f;
        for (int c = 0; c < HC; ++c) {
          const uint4 xv = reinterpret_cast<const uint4*>(
              x_row + static_cast<size_t>(c) * static_cast<size_t>(D))[j8];
          float xf[8];
          hy4_unpack_bf16x8(xv, xf);
          const float pc = s_pre[c];
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            h_reg[k][i] = fmaf(pc, xf[i], h_reg[k][i]);
          }
        }
        if (has_norm) {
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            h2 = fmaf(h_reg[k][i], h_reg[k][i], h2);
          }
        }
      }
    }
    float rs_h = 1.0f;
    if (has_norm) {
      h2 = hy4_warp_sum(h2);
      if (lane == 0) s_h2[warp] = h2;
      __syncthreads();
      float h2_tot = 0.0f;
#pragma unroll
      for (int w = 0; w < 8; ++w) h2_tot += s_h2[w];
      rs_h = rsqrtf(h2_tot / static_cast<float>(D) + norm_eps);
    }
#pragma unroll
    for (int k = 0; k < 2; ++k) {
      const int j8 = tid + k * 256;
      if (j8 < d8) {
        if (has_norm) {
          float nw[8];
          hy4_load_norm_w8(norm_w, j8, norm_bf16, nw);
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            h_reg[k][i] = h_reg[k][i] * rs_h * nw[i];
          }
        }
        h_out_u4[j8] = hy4_pack_bf16x8(h_reg[k]);
      }
    }
  } else {
    float rs_h = 1.0f;
    if (has_norm) {
      float h2 = 0.0f;
      for (int j8 = tid; j8 < d8; j8 += 256) {
        float acc[8] = {};
        for (int c = 0; c < HC; ++c) {
          const uint4 xv = reinterpret_cast<const uint4*>(
              x_row + static_cast<size_t>(c) * static_cast<size_t>(D))[j8];
          float xf[8];
          hy4_unpack_bf16x8(xv, xf);
          const float pc = s_pre[c];
#pragma unroll
          for (int i = 0; i < 8; ++i) acc[i] = fmaf(pc, xf[i], acc[i]);
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) h2 = fmaf(acc[i], acc[i], h2);
      }
      h2 = hy4_warp_sum(h2);
      if (lane == 0) s_h2[warp] = h2;
      __syncthreads();
      float h2_tot = 0.0f;
#pragma unroll
      for (int w = 0; w < 8; ++w) h2_tot += s_h2[w];
      rs_h = rsqrtf(h2_tot / static_cast<float>(D) + norm_eps);
    }
    for (int j8 = tid; j8 < d8; j8 += 256) {
      float acc[8] = {};
      for (int c = 0; c < HC; ++c) {
        const uint4 xv = reinterpret_cast<const uint4*>(
            x_row + static_cast<size_t>(c) * static_cast<size_t>(D))[j8];
        float xf[8];
        hy4_unpack_bf16x8(xv, xf);
        const float pc = s_pre[c];
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[i] = fmaf(pc, xf[i], acc[i]);
      }
      if (has_norm) {
        float nw[8];
        hy4_load_norm_w8(norm_w, j8, norm_bf16, nw);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          acc[i] = acc[i] * rs_h * nw[i];
        }
      }
      h_out_u4[j8] = hy4_pack_bf16x8(acc);
    }
  }
}
)cuda"
R"cuda(
// ----------------------------------------------------------------------------
// Kernel 1b: hy4_hc_pre_splitk_stage1 (Decode Split-K across SMs, M <= 16)
//   Warp-parallel over gate rows r (1 warp/row at HC=4, 128-bit LDG.128,
//   zero __syncthreads(), zero SMEM). Writes transposed partials
//   [M, 1 + n_gates, num_splits] for coalesced Stage-2 float4 loads.
// Grid: dim3(num_splits, M, 1), Block: dim3(HEAD_ONLY ? 128 : 256, 1, 1)
// ----------------------------------------------------------------------------
template <int FN_MODE, bool HEAD_ONLY>
__global__ __launch_bounds__(HEAD_ONLY ? 128 : 256, 4) void hy4_hc_pre_splitk_stage1(
    const nv_bfloat16* __restrict__ x,
    const void* __restrict__ w_fn,
    const void* __restrict__ s_fn,
    float* __restrict__ partials,
    int M,
    int HC,
    int D,
    int num_splits) {
  constexpr int NW = HEAD_ONLY ? 4 : 8;
  const int s = static_cast<int>(blockIdx.x);
  const int m = static_cast<int>(blockIdx.y);
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;
  if (m >= M || s >= num_splits) {
    return;
  }

  const int K = HC * D;
  const int n_gates = HEAD_ONLY ? HC : (2 * HC);
  const nv_bfloat16* x_row = x + static_cast<size_t>(m) * static_cast<size_t>(K);
  const size_t p_m_off = static_cast<size_t>(m) *
      static_cast<size_t>(1 + n_gates) * static_cast<size_t>(num_splits);
  const int r0 = warp;
  const int r1 = warp + NW;
  const bool has_r0 = (r0 < n_gates);
  const bool has_r1 = (r1 < n_gates);

  float sq_acc = 0.0f;
  float dot0 = 0.0f;
  float dot1 = 0.0f;

  if constexpr (FN_MODE == 0) {
    const int n_grp16 = K >> 4;
    const int per = (n_grp16 + num_splits - 1) / num_splits;
    const int g_lo = s * per;
    const int g_hi = min(g_lo + per, n_grp16);
    const size_t w_stride = static_cast<size_t>(n_grp16);
    const uint2* w_u2 = reinterpret_cast<const uint2*>(w_fn);
    const unsigned char* s_u8 = reinterpret_cast<const unsigned char*>(s_fn);
    const uint4* x_u4 = reinterpret_cast<const uint4*>(x_row);
    for (int g = g_lo + lane; g < g_hi; g += 32) {
      float xf[16];
      hy4_unpack_bf16x8(__ldg(x_u4 + 2 * g), xf);
      hy4_unpack_bf16x8(__ldg(x_u4 + 2 * g + 1), xf + 8);
      if (warp == 0) {
#pragma unroll
        for (int i = 0; i < 16; ++i) sq_acc = fmaf(xf[i], xf[i], sq_acc);
      }
      if (has_r0) {
        const uint2 w2 = hy4_ld_cs_u2(w_u2 + static_cast<size_t>(r0) * w_stride + g);
        const float sc = hy4_fast_e4m3_to_f(s_u8[static_cast<size_t>(r0) * w_stride + g]);
        dot0 = fmaf(sc, hy4_dot_e2m1_u2(w2, xf), dot0);
      }
      if (has_r1) {
        const uint2 w2 = hy4_ld_cs_u2(w_u2 + static_cast<size_t>(r1) * w_stride + g);
        const float sc = hy4_fast_e4m3_to_f(s_u8[static_cast<size_t>(r1) * w_stride + g]);
        dot1 = fmaf(sc, hy4_dot_e2m1_u2(w2, xf), dot1);
      }
    }
  } else if constexpr (FN_MODE == 1) {
    const int n_grp16 = K >> 4;
    const int per = (n_grp16 + num_splits - 1) / num_splits;
    const int g_lo = s * per;
    const int g_hi = min(g_lo + per, n_grp16);
    const size_t w_stride = static_cast<size_t>(n_grp16);
    const size_t s_stride = static_cast<size_t>(K >> 5);
    const uint4* w_u4 = reinterpret_cast<const uint4*>(w_fn);
    const __half* s_f16 = reinterpret_cast<const __half*>(s_fn);
    const uint4* x_u4 = reinterpret_cast<const uint4*>(x_row);
    for (int g = g_lo + lane; g < g_hi; g += 32) {
      float xf[16];
      hy4_unpack_bf16x8(__ldg(x_u4 + 2 * g), xf);
      hy4_unpack_bf16x8(__ldg(x_u4 + 2 * g + 1), xf + 8);
      if (warp == 0) {
#pragma unroll
        for (int i = 0; i < 16; ++i) sq_acc = fmaf(xf[i], xf[i], sq_acc);
      }
      const int sg = g >> 1;
      if (has_r0) {
        const uint4 w4 = w_u4[static_cast<size_t>(r0) * w_stride + g];
        const float sc = __half2float(s_f16[static_cast<size_t>(r0) * s_stride + sg]);
        float d = hy4_dot_symq8_u32(w4.x, xf[0], xf[1], xf[2], xf[3]);
        d += hy4_dot_symq8_u32(w4.y, xf[4], xf[5], xf[6], xf[7]);
        d += hy4_dot_symq8_u32(w4.z, xf[8], xf[9], xf[10], xf[11]);
        d += hy4_dot_symq8_u32(w4.w, xf[12], xf[13], xf[14], xf[15]);
        dot0 = fmaf(sc, d, dot0);
      }
      if (has_r1) {
        const uint4 w4 = w_u4[static_cast<size_t>(r1) * w_stride + g];
        const float sc = __half2float(s_f16[static_cast<size_t>(r1) * s_stride + sg]);
        float d = hy4_dot_symq8_u32(w4.x, xf[0], xf[1], xf[2], xf[3]);
        d += hy4_dot_symq8_u32(w4.y, xf[4], xf[5], xf[6], xf[7]);
        d += hy4_dot_symq8_u32(w4.z, xf[8], xf[9], xf[10], xf[11]);
        d += hy4_dot_symq8_u32(w4.w, xf[12], xf[13], xf[14], xf[15]);
        dot1 = fmaf(sc, d, dot1);
      }
    }
  } else if constexpr (FN_MODE == 2) {
    const int n_c8 = K >> 3;
    const int per = (n_c8 + num_splits - 1) / num_splits;
    const int c8_lo = s * per;
    const int c8_hi = min(c8_lo + per, n_c8);
    const size_t w_stride = static_cast<size_t>(K >> 2);
    const float4* w_f4 = reinterpret_cast<const float4*>(w_fn);
    const uint4* x_u4 = reinterpret_cast<const uint4*>(x_row);
    for (int c8 = c8_lo + lane; c8 < c8_hi; c8 += 32) {
      float xf[8];
      hy4_unpack_bf16x8(__ldg(x_u4 + c8), xf);
      if (warp == 0) {
#pragma unroll
        for (int i = 0; i < 8; ++i) sq_acc = fmaf(xf[i], xf[i], sq_acc);
      }
      if (has_r0) {
        const float4* wp = w_f4 + static_cast<size_t>(r0) * w_stride + static_cast<size_t>(c8) * 2;
        const float4 w0 = wp[0];
        const float4 w1 = wp[1];
        dot0 = fmaf(w0.x, xf[0], fmaf(w0.y, xf[1], fmaf(w0.z, xf[2], fmaf(w0.w, xf[3], dot0))));
        dot0 = fmaf(w1.x, xf[4], fmaf(w1.y, xf[5], fmaf(w1.z, xf[6], fmaf(w1.w, xf[7], dot0))));
      }
      if (has_r1) {
        const float4* wp = w_f4 + static_cast<size_t>(r1) * w_stride + static_cast<size_t>(c8) * 2;
        const float4 w0 = wp[0];
        const float4 w1 = wp[1];
        dot1 = fmaf(w0.x, xf[0], fmaf(w0.y, xf[1], fmaf(w0.z, xf[2], fmaf(w0.w, xf[3], dot1))));
        dot1 = fmaf(w1.x, xf[4], fmaf(w1.y, xf[5], fmaf(w1.z, xf[6], fmaf(w1.w, xf[7], dot1))));
      }
    }
  } else {
    const int n_c8 = K >> 3;
    const int per = (n_c8 + num_splits - 1) / num_splits;
    const int c8_lo = s * per;
    const int c8_hi = min(c8_lo + per, n_c8);
    const size_t w_stride = static_cast<size_t>(n_c8);
    const uint4* w_u4 = reinterpret_cast<const uint4*>(w_fn);
    const uint4* x_u4 = reinterpret_cast<const uint4*>(x_row);
    for (int c8 = c8_lo + lane; c8 < c8_hi; c8 += 32) {
      float xf[8];
      hy4_unpack_bf16x8(__ldg(x_u4 + c8), xf);
      if (warp == 0) {
#pragma unroll
        for (int i = 0; i < 8; ++i) sq_acc = fmaf(xf[i], xf[i], sq_acc);
      }
      if (has_r0) {
        dot0 += hy4_dot_bf16x8(w_u4[static_cast<size_t>(r0) * w_stride + c8], xf);
      }
      if (has_r1) {
        dot1 += hy4_dot_bf16x8(w_u4[static_cast<size_t>(r1) * w_stride + c8], xf);
      }
    }
  }

  if (warp == 0) {
    sq_acc = hy4_warp_sum(sq_acc);
    if (lane == 0) {
      partials[p_m_off + static_cast<size_t>(s)] = sq_acc;
    }
  }
  if (has_r0) {
    dot0 = hy4_warp_sum(dot0);
    if (lane == 0) {
      partials[p_m_off + static_cast<size_t>(1 + r0) * static_cast<size_t>(num_splits) +
               static_cast<size_t>(s)] = dot0;
    }
  }
  if (has_r1) {
    dot1 = hy4_warp_sum(dot1);
    if (lane == 0) {
      partials[p_m_off + static_cast<size_t>(1 + r1) * static_cast<size_t>(num_splits) +
               static_cast<size_t>(s)] = dot1;
    }
  }
}
)cuda"
R"cuda(
// ----------------------------------------------------------------------------
// Kernel 1c: hy4_hc_pre_tiled_stage1 (Prefill M-Tiled BM=32, HC <= 4, F32/BF16)
//   Loads w_fn [n_gates, BK=128] once into SMEM and reuses it across 32 tokens
//   (8 warps x 4 tokens/warp) with exact FP32 FMA accumulation.
//   Writes transposed partials [M, 1 + n_gates, num_splits].
// Grid: dim3((M + 31) / 32, num_splits, 1), Block: dim3(256, 1, 1)
// ----------------------------------------------------------------------------
template <int FN_MODE, bool HEAD_ONLY>
__global__ __launch_bounds__(256, 2) void hy4_hc_pre_tiled_stage1(
    const nv_bfloat16* __restrict__ x,
    const void* __restrict__ w_fn,
    float* __restrict__ partials,
    int M,
    int HC,
    int D,
    int num_splits) {
  const int m_base = static_cast<int>(blockIdx.x) * 32;
  const int s = static_cast<int>(blockIdx.y);
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;
  if (m_base >= M || s >= num_splits) {
    return;
  }

  const int K = HC * D;
  const int n_c4 = K >> 2;
  const int n_gates = HEAD_ONLY ? HC : (2 * HC);
  constexpr int MAX_GATES = HEAD_ONLY ? 4 : 8;

  const int n_blks = (n_c4 + 31) >> 5;
  const int blks_per = (n_blks + num_splits - 1) / num_splits;
  const int blk_lo = s * blks_per;
  const int blk_hi = min(blk_lo + blks_per, n_blks);

  const uint2* x_u2 = reinterpret_cast<const uint2*>(x);
  const size_t x_stride_u2 = static_cast<size_t>(n_c4);
  const size_t w_stride = static_cast<size_t>(n_c4);

  __shared__ float4 s_w[MAX_GATES][32];

  float sq_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float dot_acc[4][MAX_GATES];
#pragma unroll
  for (int tm = 0; tm < 4; ++tm) {
#pragma unroll
    for (int r = 0; r < MAX_GATES; ++r) {
      dot_acc[tm][r] = 0.0f;
    }
  }

  const int ld_r = warp;
  const int ld_c = lane;
  const int m_w0 = m_base + warp * 4;

  for (int blk = blk_lo; blk < blk_hi; ++blk) {
    const int c4_base = blk << 5;
    if (ld_r < n_gates) {
      const int c4 = c4_base + ld_c;
      float4 wv = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      if (c4 < n_c4) {
        if constexpr (FN_MODE == 2) {
          const float4* w_f4 = reinterpret_cast<const float4*>(w_fn);
          wv = w_f4[static_cast<size_t>(ld_r) * w_stride + c4];
        } else {
          const uint2* w_u2 = reinterpret_cast<const uint2*>(w_fn);
          const uint2 wu = w_u2[static_cast<size_t>(ld_r) * w_stride + c4];
          const __nv_bfloat162* wp = reinterpret_cast<const __nv_bfloat162*>(&wu);
          const float2 f0 = __bfloat1622float2(wp[0]);
          const float2 f1 = __bfloat1622float2(wp[1]);
          wv = make_float4(f0.x, f0.y, f1.x, f1.y);
        }
      }
      s_w[ld_r][ld_c] = wv;
    }
    __syncthreads();

    const int c4 = c4_base + lane;
    if (c4 < n_c4) {
      float xf[4][4];
#pragma unroll
      for (int tm = 0; tm < 4; ++tm) {
        const int m = m_w0 + tm;
        if (m < M) {
          const uint2 xu = __ldg(x_u2 + static_cast<size_t>(m) * x_stride_u2 + c4);
          const __nv_bfloat162* xp = reinterpret_cast<const __nv_bfloat162*>(&xu);
          const float2 f0 = __bfloat1622float2(xp[0]);
          const float2 f1 = __bfloat1622float2(xp[1]);
          xf[tm][0] = f0.x;
          xf[tm][1] = f0.y;
          xf[tm][2] = f1.x;
          xf[tm][3] = f1.y;
          sq_acc[tm] = fmaf(f0.x, f0.x, fmaf(f0.y, f0.y, fmaf(f1.x, f1.x, fmaf(f1.y, f1.y, sq_acc[tm]))));
        } else {
          xf[tm][0] = 0.0f;
          xf[tm][1] = 0.0f;
          xf[tm][2] = 0.0f;
          xf[tm][3] = 0.0f;
        }
      }
#pragma unroll
      for (int r = 0; r < MAX_GATES; ++r) {
        if (r < n_gates) {
          const float4 wv = s_w[r][lane];
#pragma unroll
          for (int tm = 0; tm < 4; ++tm) {
            dot_acc[tm][r] = fmaf(wv.x, xf[tm][0],
                fmaf(wv.y, xf[tm][1],
                    fmaf(wv.z, xf[tm][2],
                        fmaf(wv.w, xf[tm][3], dot_acc[tm][r]))));
          }
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int tm = 0; tm < 4; ++tm) {
    const int m = m_w0 + tm;
    if (m < M) {
      const float sq_tot = hy4_warp_sum(sq_acc[tm]);
      const size_t p_m_off = static_cast<size_t>(m) *
          static_cast<size_t>(1 + n_gates) * static_cast<size_t>(num_splits);
      if (lane == 0) {
        partials[p_m_off + static_cast<size_t>(s)] = sq_tot;
      }
#pragma unroll
      for (int r = 0; r < MAX_GATES; ++r) {
        if (r < n_gates) {
          const float dt = hy4_warp_sum(dot_acc[tm][r]);
          if (lane == 0) {
            partials[p_m_off + static_cast<size_t>(1 + r) * static_cast<size_t>(num_splits) +
                     static_cast<size_t>(s)] = dt;
          }
        }
      }
    }
  }
}
)cuda"
R"cuda(
// ----------------------------------------------------------------------------
// Kernel 1d: hy4_hc_pre_splitk_stage2 (Finish gates + stream mix + RMSNorm)
//   Reads transposed partials [M, 1 + n_gates, num_splits] with float4
//   coalesced warp-parallel reduction; preloads X & norm_w on cycle 1 for
//   single-pass D <= 3072 (d8 <= 384, block=384).
// Grid: dim3(M, 1, 1), Block: dim3(384, 1, 1)
// ----------------------------------------------------------------------------
template <bool HEAD_ONLY>
__global__ __launch_bounds__(384, 2) void hy4_hc_pre_splitk_stage2(
    const nv_bfloat16* __restrict__ x,
    const float* __restrict__ partials,
    const float* __restrict__ scale,
    const float* __restrict__ base,
    const void* __restrict__ norm_w,
    nv_bfloat16* __restrict__ out_h,
    float* __restrict__ out_post,
    int M,
    int HC,
    int D,
    int num_splits,
    float norm_eps,
    float hc_eps,
    float hc_mag,
    int has_norm,
    int norm_bf16) {
  const int m = static_cast<int>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;
  if (m >= M) {
    return;
  }

  const int K = HC * D;
  const int n_gates = HEAD_ONLY ? HC : (2 * HC);
  const int d8 = D >> 3;
  const nv_bfloat16* x_row = x + static_cast<size_t>(m) * static_cast<size_t>(K);
  const float* p_m = partials + static_cast<size_t>(m) *
      static_cast<size_t>(1 + n_gates) * static_cast<size_t>(num_splits);

  // Cycle-1 overlapped preload of X (HC==4) and norm_w for single-pass D <= 3072.
  uint4 xv0{}, xv1{}, xv2{}, xv3{};
  float nw_reg[8];
  const bool fast_hc4 = (d8 <= 384 && HC == 4);
  if (fast_hc4 && tid < d8) {
    xv0 = reinterpret_cast<const uint4*>(x_row)[tid];
    xv1 = reinterpret_cast<const uint4*>(x_row + static_cast<size_t>(D))[tid];
    xv2 = reinterpret_cast<const uint4*>(x_row + static_cast<size_t>(2) * static_cast<size_t>(D))[tid];
    xv3 = reinterpret_cast<const uint4*>(x_row + static_cast<size_t>(3) * static_cast<size_t>(D))[tid];
    if (has_norm) {
      hy4_load_norm_w8(norm_w, tid, norm_bf16, nw_reg);
    }
  }

  __shared__ float s_red[17];
  __shared__ float s_pre[8];
  __shared__ float s_h2[12];

  // Warp-parallel coalesced reduction across all 1 + n_gates rows of partials.
  for (int col = warp; col <= n_gates; col += 12) {
    const float* row = p_m + static_cast<size_t>(col) * static_cast<size_t>(num_splits);
    float acc = 0.0f;
    if ((num_splits & 3) == 0 && num_splits <= 64) {
      const int n4 = num_splits >> 2;
      if (lane < n4) {
        const float4 v = reinterpret_cast<const float4*>(row)[lane];
        acc = (v.x + v.y) + (v.z + v.w);
      }
      acc += __shfl_xor_sync(0xFFFFFFFFu, acc, 8);
      acc += __shfl_xor_sync(0xFFFFFFFFu, acc, 4);
      acc += __shfl_xor_sync(0xFFFFFFFFu, acc, 2);
      acc += __shfl_xor_sync(0xFFFFFFFFu, acc, 1);
    } else {
      for (int s = lane; s < num_splits; s += 32) {
        acc += row[s];
      }
      acc = hy4_warp_sum(acc);
    }
    if (lane == 0) {
      s_red[col] = acc;
    }
  }
  __syncthreads();

  if (tid < n_gates) {
    const float rs = rsqrtf(s_red[0] / static_cast<float>(K) + norm_eps);
    const float dt = s_red[1 + tid];
    if (tid < HC) {
      s_pre[tid] = hy4_sigmoid_fast((dt * rs) * scale[0] + base[tid]) + hc_eps;
    } else if constexpr (!HEAD_ONLY) {
      out_post[static_cast<size_t>(m) * static_cast<size_t>(HC) + (tid - HC)] =
          hc_mag * hy4_sigmoid_fast((dt * rs) * scale[1] + base[tid]) + hc_eps;
    }
  }
  __syncthreads();

  uint4* h_out_u4 = reinterpret_cast<uint4*>(out_h + static_cast<size_t>(m) * static_cast<size_t>(D));

  if (d8 <= 384) {
    float h_reg[8];
    float h2 = 0.0f;
    if (tid < d8) {
      if (fast_hc4) {
        const float p0 = s_pre[0], p1 = s_pre[1], p2 = s_pre[2], p3 = s_pre[3];
        float xf[8];
        hy4_unpack_bf16x8(xv0, xf);
#pragma unroll
        for (int i = 0; i < 8; ++i) h_reg[i] = p0 * xf[i];
        hy4_unpack_bf16x8(xv1, xf);
#pragma unroll
        for (int i = 0; i < 8; ++i) h_reg[i] = fmaf(p1, xf[i], h_reg[i]);
        hy4_unpack_bf16x8(xv2, xf);
#pragma unroll
        for (int i = 0; i < 8; ++i) h_reg[i] = fmaf(p2, xf[i], h_reg[i]);
        hy4_unpack_bf16x8(xv3, xf);
#pragma unroll
        for (int i = 0; i < 8; ++i) h_reg[i] = fmaf(p3, xf[i], h_reg[i]);
      } else {
#pragma unroll
        for (int i = 0; i < 8; ++i) h_reg[i] = 0.0f;
        for (int c = 0; c < HC; ++c) {
          const uint4 xv = reinterpret_cast<const uint4*>(
              x_row + static_cast<size_t>(c) * static_cast<size_t>(D))[tid];
          float xf[8];
          hy4_unpack_bf16x8(xv, xf);
          const float pc = s_pre[c];
#pragma unroll
          for (int i = 0; i < 8; ++i) h_reg[i] = fmaf(pc, xf[i], h_reg[i]);
        }
        if (has_norm) {
          hy4_load_norm_w8(norm_w, tid, norm_bf16, nw_reg);
        }
      }
      if (has_norm) {
#pragma unroll
        for (int i = 0; i < 8; ++i) h2 = fmaf(h_reg[i], h_reg[i], h2);
      }
    }
    float rs_h = 1.0f;
    if (has_norm) {
      h2 = hy4_warp_sum(h2);
      if (lane == 0) s_h2[warp] = h2;
      __syncthreads();
      float h2_tot = 0.0f;
#pragma unroll
      for (int w = 0; w < 12; ++w) h2_tot += s_h2[w];
      rs_h = rsqrtf(h2_tot / static_cast<float>(D) + norm_eps);
    }
    if (tid < d8) {
      if (has_norm) {
#pragma unroll
        for (int i = 0; i < 8; ++i) h_reg[i] = h_reg[i] * rs_h * nw_reg[i];
      }
      h_out_u4[tid] = hy4_pack_bf16x8(h_reg);
    }
  } else {
    float rs_h = 1.0f;
    if (has_norm) {
      float h2 = 0.0f;
      for (int j8 = tid; j8 < d8; j8 += 384) {
        float acc[8] = {};
        for (int c = 0; c < HC; ++c) {
          const uint4 xv = reinterpret_cast<const uint4*>(
              x_row + static_cast<size_t>(c) * static_cast<size_t>(D))[j8];
          float xf[8];
          hy4_unpack_bf16x8(xv, xf);
          const float pc = s_pre[c];
#pragma unroll
          for (int i = 0; i < 8; ++i) acc[i] = fmaf(pc, xf[i], acc[i]);
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) h2 = fmaf(acc[i], acc[i], h2);
      }
      h2 = hy4_warp_sum(h2);
      if (lane == 0) s_h2[warp] = h2;
      __syncthreads();
      float h2_tot = 0.0f;
#pragma unroll
      for (int w = 0; w < 12; ++w) h2_tot += s_h2[w];
      rs_h = rsqrtf(h2_tot / static_cast<float>(D) + norm_eps);
    }
    for (int j8 = tid; j8 < d8; j8 += 384) {
      float acc[8] = {};
      for (int c = 0; c < HC; ++c) {
        const uint4 xv = reinterpret_cast<const uint4*>(
            x_row + static_cast<size_t>(c) * static_cast<size_t>(D))[j8];
        float xf[8];
        hy4_unpack_bf16x8(xv, xf);
        const float pc = s_pre[c];
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[i] = fmaf(pc, xf[i], acc[i]);
      }
      if (has_norm) {
        float nw[8];
        hy4_load_norm_w8(norm_w, j8, norm_bf16, nw);
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[i] = acc[i] * rs_h * nw[i];
      }
      h_out_u4[j8] = hy4_pack_bf16x8(acc);
    }
  }
}
)cuda"
R"cuda(
// ----------------------------------------------------------------------------
// Kernel 1e: hy4_hc_post_pre_stage1 (Fused HC Post-Norm + Residual + Pre S1)
//   Combines x_next = bf16(x + prev_post * rms_norm(y, prev_norm_w)) with
//   next-layer split-K projection W_fn * x_next -> transposed partials
//   [M, 1 + 2*HC, num_splits]. Overlaps cold W_fn LDG.128 on cycle 1 with
//   all-active parallel x_next computation & out_x write.
// Grid: dim3(num_splits, M, 1), Block: dim3(256, 1, 1)
// ----------------------------------------------------------------------------
template <int FN_MODE>
__global__ __launch_bounds__(256, 4) void hy4_hc_post_pre_stage1(
    const nv_bfloat16* __restrict__ x,
    const nv_bfloat16* __restrict__ y,
    const float* __restrict__ prev_post,
    const void* __restrict__ prev_norm_w,
    const void* __restrict__ w_fn,
    const void* __restrict__ s_fn,
    nv_bfloat16* __restrict__ out_x,
    float* __restrict__ partials,
    int M,
    int HC,
    int D,
    int num_splits,
    float norm_eps,
    int has_prev_norm,
    int prev_norm_bf16) {
  constexpr int NW = 8;
  const int s = static_cast<int>(blockIdx.x);
  const int m = static_cast<int>(blockIdx.y);
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;
  if (m >= M || s >= num_splits) {
    return;
  }

  const int K = HC * D;
  const int d8 = D >> 3;
  const int d2 = D >> 1;
  const int n_gates = 2 * HC;
  const size_t x_row_off = static_cast<size_t>(m) * static_cast<size_t>(K);
  const size_t y_row_off = static_cast<size_t>(m) * static_cast<size_t>(D);
  const size_t p_m_off = static_cast<size_t>(m) *
      static_cast<size_t>(1 + n_gates) * static_cast<size_t>(num_splits);
  const uint4* y_u4 = reinterpret_cast<const uint4*>(y + y_row_off);
  const unsigned int* x_u1 = reinterpret_cast<const unsigned int*>(x + x_row_off);
  unsigned int* ox_u1 = reinterpret_cast<unsigned int*>(out_x + x_row_off);
  const unsigned int* y_u1 = reinterpret_cast<const unsigned int*>(y + y_row_off);
  const float* p_row = prev_post + static_cast<size_t>(m) * static_cast<size_t>(HC);

  __shared__ float s_red[NW];
  __shared__ unsigned int s_xn[256];

  const int r0 = warp;
  const int r1 = warp + NW;
  const bool has_r0 = (r0 < n_gates);
  const bool has_r1 = (r1 < n_gates);

  if constexpr (FN_MODE == 3) {
    const int n_c8 = K >> 3;
    const int per = (n_c8 + num_splits - 1) / num_splits;
    const int c8_lo = s * per;
    const int c8_hi = min(c8_lo + per, n_c8);
    const uint4* w_u4 = reinterpret_cast<const uint4*>(w_fn);
    const size_t w_stride = static_cast<size_t>(n_c8);

    // Fast overlapped path: per <= 64 (<= 2 chunks of 32 uint4) and HC <= 4 (1 warp/row).
    if (per <= 64 && n_gates <= NW) {
      const int c8_0 = c8_lo + lane;
      const int c8_1 = c8_lo + 32 + lane;
      const bool v0 = (c8_0 < c8_hi);
      const bool v1 = (c8_1 < c8_hi);
      uint4 w0{}, w1{};
      if (has_r0) {
        if (v0) w0 = w_u4[static_cast<size_t>(r0) * w_stride + c8_0];
        if (v1) w1 = w_u4[static_cast<size_t>(r0) * w_stride + c8_1];
      }

      float rs_y = 1.0f;
      if (has_prev_norm) {
        float y2 = 0.0f;
        for (int j8 = tid; j8 < d8; j8 += 256) {
          float yf[8];
          hy4_unpack_bf16x8(__ldg(y_u4 + j8), yf);
#pragma unroll
          for (int i = 0; i < 8; ++i) y2 = fmaf(yf[i], yf[i], y2);
        }
        y2 = hy4_warp_sum(y2);
        if (lane == 0) s_red[warp] = y2;
        __syncthreads();
        float y2_tot = 0.0f;
#pragma unroll
        for (int w = 0; w < NW; ++w) y2_tot += s_red[w];
        rs_y = rsqrtf(y2_tot / static_cast<float>(D) + norm_eps);
      }

      const int c2_lo = c8_lo << 2;
      const int cnt_c2 = max(0, c8_hi - c8_lo) << 2;
      float sq_acc = 0.0f;
      if (tid < cnt_c2) {
        const int c2 = c2_lo + tid;
        const int c = c2 / d2;
        const int j2 = c2 - c * d2;
        const unsigned int xu = x_u1[c2];
        const unsigned int yu = __ldg(y_u1 + j2);
        float2 xf = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xu));
        const float2 yf = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&yu));
        const float pc = p_row[c];
        if (has_prev_norm) {
          const float2 nw = hy4_load_norm_w2(prev_norm_w, j2, prev_norm_bf16);
          xf.x = fmaf(pc, yf.x * rs_y * nw.x, xf.x);
          xf.y = fmaf(pc, yf.y * rs_y * nw.y, xf.y);
        } else {
          xf.x = fmaf(pc, yf.x, xf.x);
          xf.y = fmaf(pc, yf.y, xf.y);
        }
        const unsigned int xnu = hy4_pack_bf16x2(xf.x, xf.y);
        ox_u1[c2] = xnu;
        const int ch = tid >> 7;
        const int in_ch = tid & 127;
        s_xn[ch * 128 + (in_ch & 3) * 32 + (in_ch >> 2)] = xnu;
        const float2 xnf = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xnu));
        sq_acc = fmaf(xnf.x, xnf.x, xnf.y * xnf.y);
      }

      sq_acc = hy4_warp_sum(sq_acc);
      if (lane == 0) s_red[warp] = sq_acc;
      __syncthreads();

      if (tid == 0) {
        float sq_tot = 0.0f;
#pragma unroll
        for (int w = 0; w < NW; ++w) sq_tot += s_red[w];
        partials[p_m_off + static_cast<size_t>(s)] = sq_tot;
      }

      if (has_r0) {
        float dt = 0.0f;
        if (v0) {
          const uint4 xnu{s_xn[lane], s_xn[32 + lane], s_xn[64 + lane], s_xn[96 + lane]};
          float xnf[8];
          hy4_unpack_bf16x8(xnu, xnf);
          dt += hy4_dot_bf16x8(w0, xnf);
        }
        if (v1) {
          const uint4 xnu{s_xn[128 + lane], s_xn[160 + lane], s_xn[192 + lane], s_xn[224 + lane]};
          float xnf[8];
          hy4_unpack_bf16x8(xnu, xnf);
          dt += hy4_dot_bf16x8(w1, xnf);
        }
        dt = hy4_warp_sum(dt);
        if (lane == 0) {
          partials[p_m_off + static_cast<size_t>(1 + r0) * static_cast<size_t>(num_splits) +
                   static_cast<size_t>(s)] = dt;
        }
      }
      return;
    }

    // General BF16 chunk loop (per > 64 or HC > 4).
    float rs_y = 1.0f;
    if (has_prev_norm) {
      float y2 = 0.0f;
      for (int j8 = tid; j8 < d8; j8 += 256) {
        float yf[8];
        hy4_unpack_bf16x8(__ldg(y_u4 + j8), yf);
#pragma unroll
        for (int i = 0; i < 8; ++i) y2 = fmaf(yf[i], yf[i], y2);
      }
      y2 = hy4_warp_sum(y2);
      if (lane == 0) s_red[warp] = y2;
      __syncthreads();
      float y2_tot = 0.0f;
#pragma unroll
      for (int w = 0; w < NW; ++w) y2_tot += s_red[w];
      rs_y = rsqrtf(y2_tot / static_cast<float>(D) + norm_eps);
    }

    float sq_acc = 0.0f, dot0 = 0.0f, dot1 = 0.0f;
    for (int c8_base = c8_lo; c8_base < c8_hi; c8_base += 64) {
      const int cnt_c8 = min(64, c8_hi - c8_base);
      const int cnt_c2 = cnt_c8 << 2;
      const int c2_lo = c8_base << 2;
      if (tid < cnt_c2) {
        const int c2 = c2_lo + tid;
        const int c = c2 / d2;
        const int j2 = c2 - c * d2;
        const unsigned int xu = x_u1[c2];
        const unsigned int yu = __ldg(y_u1 + j2);
        float2 xf = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xu));
        const float2 yf = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&yu));
        const float pc = p_row[c];
        if (has_prev_norm) {
          const float2 nw = hy4_load_norm_w2(prev_norm_w, j2, prev_norm_bf16);
          xf.x = fmaf(pc, yf.x * rs_y * nw.x, xf.x);
          xf.y = fmaf(pc, yf.y * rs_y * nw.y, xf.y);
        } else {
          xf.x = fmaf(pc, yf.x, xf.x);
          xf.y = fmaf(pc, yf.y, xf.y);
        }
        const unsigned int xnu = hy4_pack_bf16x2(xf.x, xf.y);
        ox_u1[c2] = xnu;
        const int ch = tid >> 7;
        const int in_ch = tid & 127;
        s_xn[ch * 128 + (in_ch & 3) * 32 + (in_ch >> 2)] = xnu;
        const float2 xnf = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xnu));
        sq_acc = fmaf(xnf.x, xnf.x, fmaf(xnf.y, xnf.y, sq_acc));
      }
      __syncthreads();
#pragma unroll
      for (int ch = 0; ch < 2; ++ch) {
        const int c8 = c8_base + ch * 32 + lane;
        if (c8 < c8_hi) {
          const uint4 xnu{s_xn[ch * 128 + lane], s_xn[ch * 128 + 32 + lane],
                          s_xn[ch * 128 + 64 + lane], s_xn[ch * 128 + 96 + lane]};
          float xnf[8];
          hy4_unpack_bf16x8(xnu, xnf);
          if (has_r0) dot0 += hy4_dot_bf16x8(w_u4[static_cast<size_t>(r0) * w_stride + c8], xnf);
          if (has_r1) dot1 += hy4_dot_bf16x8(w_u4[static_cast<size_t>(r1) * w_stride + c8], xnf);
        }
      }
      if (c8_base + 64 < c8_hi) __syncthreads();
    }

    sq_acc = hy4_warp_sum(sq_acc);
    if (lane == 0) s_red[warp] = sq_acc;
    __syncthreads();
    if (tid == 0) {
      float sq_tot = 0.0f;
#pragma unroll
      for (int w = 0; w < NW; ++w) sq_tot += s_red[w];
      partials[p_m_off + static_cast<size_t>(s)] = sq_tot;
    }
    if (has_r0) {
      dot0 = hy4_warp_sum(dot0);
      if (lane == 0) {
        partials[p_m_off + static_cast<size_t>(1 + r0) * static_cast<size_t>(num_splits) +
                 static_cast<size_t>(s)] = dot0;
      }
    }
    if (has_r1) {
      dot1 = hy4_warp_sum(dot1);
      if (lane == 0) {
        partials[p_m_off + static_cast<size_t>(1 + r1) * static_cast<size_t>(num_splits) +
                 static_cast<size_t>(s)] = dot1;
      }
    }
    return;
  }
)cuda"
R"cuda(
  // Non-BF16 modes (FN_MODE = 0 NVFP4, 1 SymQ8, 2 F32): compute rs_y once, then stream chunks.
  float rs_y = 1.0f;
  if (has_prev_norm) {
    float y2 = 0.0f;
    for (int j8 = tid; j8 < d8; j8 += 256) {
      float yf[8];
      hy4_unpack_bf16x8(__ldg(y_u4 + j8), yf);
#pragma unroll
      for (int i = 0; i < 8; ++i) y2 = fmaf(yf[i], yf[i], y2);
    }
    y2 = hy4_warp_sum(y2);
    if (lane == 0) s_red[warp] = y2;
    __syncthreads();
    float y2_tot = 0.0f;
#pragma unroll
    for (int w = 0; w < NW; ++w) y2_tot += s_red[w];
    rs_y = rsqrtf(y2_tot / static_cast<float>(D) + norm_eps);
  }

  float sq_acc = 0.0f, dot0 = 0.0f, dot1 = 0.0f;

  if constexpr (FN_MODE == 2) {
    const int n_c8 = K >> 3;
    const int per = (n_c8 + num_splits - 1) / num_splits;
    const int c8_lo = s * per;
    const int c8_hi = min(c8_lo + per, n_c8);
    const size_t w_stride = static_cast<size_t>(K >> 2);
    const float4* w_f4 = reinterpret_cast<const float4*>(w_fn);
    for (int c8_base = c8_lo; c8_base < c8_hi; c8_base += 64) {
      const int cnt_c8 = min(64, c8_hi - c8_base);
      const int cnt_c2 = cnt_c8 << 2;
      const int c2_lo = c8_base << 2;
      if (tid < cnt_c2) {
        const int c2 = c2_lo + tid;
        const int c = c2 / d2;
        const int j2 = c2 - c * d2;
        const unsigned int xu = x_u1[c2];
        const unsigned int yu = __ldg(y_u1 + j2);
        float2 xf = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xu));
        const float2 yf = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&yu));
        const float pc = p_row[c];
        if (has_prev_norm) {
          const float2 nw = hy4_load_norm_w2(prev_norm_w, j2, prev_norm_bf16);
          xf.x = fmaf(pc, yf.x * rs_y * nw.x, xf.x);
          xf.y = fmaf(pc, yf.y * rs_y * nw.y, xf.y);
        } else {
          xf.x = fmaf(pc, yf.x, xf.x);
          xf.y = fmaf(pc, yf.y, xf.y);
        }
        const unsigned int xnu = hy4_pack_bf16x2(xf.x, xf.y);
        ox_u1[c2] = xnu;
        const int ch = tid >> 7;
        const int in_ch = tid & 127;
        s_xn[ch * 128 + (in_ch & 3) * 32 + (in_ch >> 2)] = xnu;
        const float2 xnf = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xnu));
        sq_acc = fmaf(xnf.x, xnf.x, fmaf(xnf.y, xnf.y, sq_acc));
      }
      __syncthreads();
#pragma unroll
      for (int ch = 0; ch < 2; ++ch) {
        const int c8 = c8_base + ch * 32 + lane;
        if (c8 < c8_hi) {
          const uint4 xnu{s_xn[ch * 128 + lane], s_xn[ch * 128 + 32 + lane],
                          s_xn[ch * 128 + 64 + lane], s_xn[ch * 128 + 96 + lane]};
          float xnf[8];
          hy4_unpack_bf16x8(xnu, xnf);
          if (has_r0) {
            const float4* wp = w_f4 + static_cast<size_t>(r0) * w_stride + static_cast<size_t>(c8) * 2;
            const float4 w0 = wp[0], w1 = wp[1];
            dot0 = fmaf(w0.x, xnf[0], fmaf(w0.y, xnf[1], fmaf(w0.z, xnf[2], fmaf(w0.w, xnf[3], dot0))));
            dot0 = fmaf(w1.x, xnf[4], fmaf(w1.y, xnf[5], fmaf(w1.z, xnf[6], fmaf(w1.w, xnf[7], dot0))));
          }
          if (has_r1) {
            const float4* wp = w_f4 + static_cast<size_t>(r1) * w_stride + static_cast<size_t>(c8) * 2;
            const float4 w0 = wp[0], w1 = wp[1];
            dot1 = fmaf(w0.x, xnf[0], fmaf(w0.y, xnf[1], fmaf(w0.z, xnf[2], fmaf(w0.w, xnf[3], dot1))));
            dot1 = fmaf(w1.x, xnf[4], fmaf(w1.y, xnf[5], fmaf(w1.z, xnf[6], fmaf(w1.w, xnf[7], dot1))));
          }
        }
      }
      if (c8_base + 64 < c8_hi) __syncthreads();
    }
  } else {
    // FN_MODE == 0 (NVFP4) or 1 (SymQ8): 32 grp16 = 64 uint4 = 256 bf16x2 per chunk!
    const int n_grp16 = K >> 4;
    const int per = (n_grp16 + num_splits - 1) / num_splits;
    const int g_lo = s * per;
    const int g_hi = min(g_lo + per, n_grp16);
    const size_t w_stride = static_cast<size_t>(n_grp16);
    for (int g_base = g_lo; g_base < g_hi; g_base += 32) {
      const int cnt_g = min(32, g_hi - g_base);
      const int cnt_c2 = cnt_g << 3;
      const int c2_lo = g_base << 3;
      if (tid < cnt_c2) {
        const int c2 = c2_lo + tid;
        const int c = c2 / d2;
        const int j2 = c2 - c * d2;
        const unsigned int xu = x_u1[c2];
        const unsigned int yu = __ldg(y_u1 + j2);
        float2 xf = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xu));
        const float2 yf = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&yu));
        const float pc = p_row[c];
        if (has_prev_norm) {
          const float2 nw = hy4_load_norm_w2(prev_norm_w, j2, prev_norm_bf16);
          xf.x = fmaf(pc, yf.x * rs_y * nw.x, xf.x);
          xf.y = fmaf(pc, yf.y * rs_y * nw.y, xf.y);
        } else {
          xf.x = fmaf(pc, yf.x, xf.x);
          xf.y = fmaf(pc, yf.y, xf.y);
        }
        const unsigned int xnu = hy4_pack_bf16x2(xf.x, xf.y);
        ox_u1[c2] = xnu;
        // grp16 g_idx = tid >> 3 (0..31), word wd = tid & 7 (0..7) -> s_xn[wd * 32 + g_idx]
        s_xn[(tid & 7) * 32 + (tid >> 3)] = xnu;
        const float2 xnf = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xnu));
        sq_acc = fmaf(xnf.x, xnf.x, fmaf(xnf.y, xnf.y, sq_acc));
      }
      __syncthreads();
      const int g = g_base + lane;
      if (g < g_hi) {
        const uint4 xlo{s_xn[lane], s_xn[32 + lane], s_xn[64 + lane], s_xn[96 + lane]};
        const uint4 xhi{s_xn[128 + lane], s_xn[160 + lane], s_xn[192 + lane], s_xn[224 + lane]};
        float xf[16];
        hy4_unpack_bf16x8(xlo, xf);
        hy4_unpack_bf16x8(xhi, xf + 8);
        if constexpr (FN_MODE == 0) {
          const uint2* w_u2 = reinterpret_cast<const uint2*>(w_fn);
          const unsigned char* s_u8 = reinterpret_cast<const unsigned char*>(s_fn);
          if (has_r0) {
            const uint2 w2 = hy4_ld_cs_u2(w_u2 + static_cast<size_t>(r0) * w_stride + g);
            const float sc = hy4_fast_e4m3_to_f(s_u8[static_cast<size_t>(r0) * w_stride + g]);
            dot0 = fmaf(sc, hy4_dot_e2m1_u2(w2, xf), dot0);
          }
          if (has_r1) {
            const uint2 w2 = hy4_ld_cs_u2(w_u2 + static_cast<size_t>(r1) * w_stride + g);
            const float sc = hy4_fast_e4m3_to_f(s_u8[static_cast<size_t>(r1) * w_stride + g]);
            dot1 = fmaf(sc, hy4_dot_e2m1_u2(w2, xf), dot1);
          }
        } else {
          const uint4* w_u4 = reinterpret_cast<const uint4*>(w_fn);
          const __half* s_f16 = reinterpret_cast<const __half*>(s_fn);
          const size_t s_stride = static_cast<size_t>(K >> 5);
          const int sg = g >> 1;
          if (has_r0) {
            const uint4 w4 = w_u4[static_cast<size_t>(r0) * w_stride + g];
            const float sc = __half2float(s_f16[static_cast<size_t>(r0) * s_stride + sg]);
            float d = hy4_dot_symq8_u32(w4.x, xf[0], xf[1], xf[2], xf[3]);
            d += hy4_dot_symq8_u32(w4.y, xf[4], xf[5], xf[6], xf[7]);
            d += hy4_dot_symq8_u32(w4.z, xf[8], xf[9], xf[10], xf[11]);
            d += hy4_dot_symq8_u32(w4.w, xf[12], xf[13], xf[14], xf[15]);
            dot0 = fmaf(sc, d, dot0);
          }
          if (has_r1) {
            const uint4 w4 = w_u4[static_cast<size_t>(r1) * w_stride + g];
            const float sc = __half2float(s_f16[static_cast<size_t>(r1) * s_stride + sg]);
            float d = hy4_dot_symq8_u32(w4.x, xf[0], xf[1], xf[2], xf[3]);
            d += hy4_dot_symq8_u32(w4.y, xf[4], xf[5], xf[6], xf[7]);
            d += hy4_dot_symq8_u32(w4.z, xf[8], xf[9], xf[10], xf[11]);
            d += hy4_dot_symq8_u32(w4.w, xf[12], xf[13], xf[14], xf[15]);
            dot1 = fmaf(sc, d, dot1);
          }
        }
      }
      if (g_base + 32 < g_hi) __syncthreads();
    }
  }

  sq_acc = hy4_warp_sum(sq_acc);
  if (lane == 0) s_red[warp] = sq_acc;
  __syncthreads();
  if (tid == 0) {
    float sq_tot = 0.0f;
#pragma unroll
    for (int w = 0; w < NW; ++w) sq_tot += s_red[w];
    partials[p_m_off + static_cast<size_t>(s)] = sq_tot;
  }
  if (has_r0) {
    dot0 = hy4_warp_sum(dot0);
    if (lane == 0) {
      partials[p_m_off + static_cast<size_t>(1 + r0) * static_cast<size_t>(num_splits) +
               static_cast<size_t>(s)] = dot0;
    }
  }
  if (has_r1) {
    dot1 = hy4_warp_sum(dot1);
    if (lane == 0) {
      partials[p_m_off + static_cast<size_t>(1 + r1) * static_cast<size_t>(num_splits) +
               static_cast<size_t>(s)] = dot1;
    }
  }
}
)cuda"
R"cuda(
// ----------------------------------------------------------------------------
// Kernel 2: hy4_hc_post_cuda & hy4_hc_post_multi_cuda
//   y_n[t, j] = f32(y[t, j]) * rsqrt(mean(y[t]^2) + norm_eps) * post_norm_w[j]
//   out_x[t, c, j] = bf16(f32(x[t, c, j]) + post[t, c] * y_n[t, j])
//   Single-CTA (M >= 17 or D < 1024): Grid dim3(M, 1, 1), Block dim3(384, 1, 1)
//   Multi-CTA  (M <= 16 & D >= 1024): Grid dim3(num_ctas, M, 1), Block dim3(128, 1, 1)
// ----------------------------------------------------------------------------
__global__ __launch_bounds__(384, 2) void hy4_hc_post_cuda(
    const nv_bfloat16* __restrict__ x,
    const nv_bfloat16* __restrict__ y,
    const float* __restrict__ post,
    const void* __restrict__ post_norm_w,
    nv_bfloat16* __restrict__ out_x,
    int M,
    int HC,
    int D,
    float norm_eps,
    int has_norm,
    int norm_bf16) {
  const int m = static_cast<int>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;
  if (m >= M) {
    return;
  }

  const size_t y_off = static_cast<size_t>(m) * static_cast<size_t>(D);
  const size_t x_off = static_cast<size_t>(m) * static_cast<size_t>(HC) * static_cast<size_t>(D);

  __shared__ float s_post[8];
  __shared__ float s_y2[12];
  if (tid < HC) {
    s_post[tid] = post[static_cast<size_t>(m) * static_cast<size_t>(HC) + tid];
  }
  __syncthreads();

  const int d8 = D >> 3;
  const uint4* y_u4 = reinterpret_cast<const uint4*>(y + y_off);

  if (d8 <= 384) {
    float y_reg[8];
    float y2 = 0.0f;
    if (tid < d8) {
      hy4_unpack_bf16x8(y_u4[tid], y_reg);
      if (has_norm) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          y2 = fmaf(y_reg[i], y_reg[i], y2);
        }
      }
    }
    if (has_norm) {
      y2 = hy4_warp_sum(y2);
      if (lane == 0) s_y2[warp] = y2;
      __syncthreads();
      float y2_tot = 0.0f;
#pragma unroll
      for (int w = 0; w < 12; ++w) y2_tot += s_y2[w];
      const float rs = rsqrtf(y2_tot / static_cast<float>(D) + norm_eps);
      if (tid < d8) {
        float nw[8];
        hy4_load_norm_w8(post_norm_w, tid, norm_bf16, nw);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          y_reg[i] = y_reg[i] * rs * nw[i];
        }
      }
    }
    if (tid < d8) {
      for (int c = 0; c < HC; ++c) {
        const size_t c_off = x_off + static_cast<size_t>(c) * static_cast<size_t>(D);
        const float pc = s_post[c];
        float xf[8];
        hy4_unpack_bf16x8(reinterpret_cast<const uint4*>(x + c_off)[tid], xf);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          xf[i] = fmaf(pc, y_reg[i], xf[i]);
        }
        reinterpret_cast<uint4*>(out_x + c_off)[tid] = hy4_pack_bf16x8(xf);
      }
    }
  } else {
    float rs = 1.0f;
    if (has_norm) {
      float y2 = 0.0f;
      for (int j8 = tid; j8 < d8; j8 += 384) {
        float yf[8];
        hy4_unpack_bf16x8(y_u4[j8], yf);
#pragma unroll
        for (int i = 0; i < 8; ++i) y2 = fmaf(yf[i], yf[i], y2);
      }
      y2 = hy4_warp_sum(y2);
      if (lane == 0) s_y2[warp] = y2;
      __syncthreads();
      float y2_tot = 0.0f;
#pragma unroll
      for (int w = 0; w < 12; ++w) y2_tot += s_y2[w];
      rs = rsqrtf(y2_tot / static_cast<float>(D) + norm_eps);
    }
    for (int j8 = tid; j8 < d8; j8 += 384) {
      float yn[8];
      hy4_unpack_bf16x8(y_u4[j8], yn);
      if (has_norm) {
        float nw[8];
        hy4_load_norm_w8(post_norm_w, j8, norm_bf16, nw);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          yn[i] = yn[i] * rs * nw[i];
        }
      }
      for (int c = 0; c < HC; ++c) {
        const size_t c_off = x_off + static_cast<size_t>(c) * static_cast<size_t>(D);
        const float pc = s_post[c];
        float xf[8];
        hy4_unpack_bf16x8(reinterpret_cast<const uint4*>(x + c_off)[j8], xf);
#pragma unroll
        for (int i = 0; i < 8; ++i) xf[i] = fmaf(pc, yn[i], xf[i]);
        reinterpret_cast<uint4*>(out_x + c_off)[j8] = hy4_pack_bf16x8(xf);
      }
    }
  }
}

__global__ __launch_bounds__(128, 4) void hy4_hc_post_multi_cuda(
    const nv_bfloat16* __restrict__ x,
    const nv_bfloat16* __restrict__ y,
    const float* __restrict__ post,
    const void* __restrict__ post_norm_w,
    nv_bfloat16* __restrict__ out_x,
    int M,
    int HC,
    int D,
    int num_ctas,
    float norm_eps,
    int has_norm,
    int norm_bf16) {
  const int b = static_cast<int>(blockIdx.x);
  const int m = static_cast<int>(blockIdx.y);
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;
  if (m >= M || b >= num_ctas) {
    return;
  }

  const int d8 = D >> 3;
  const size_t y_off = static_cast<size_t>(m) * static_cast<size_t>(D);
  const size_t x_off = static_cast<size_t>(m) * static_cast<size_t>(HC) * static_cast<size_t>(D);
  const uint4* y_u4 = reinterpret_cast<const uint4*>(y + y_off);

  __shared__ float s_y2[4];
  float rs = 1.0f;
  if (has_norm) {
    float y2 = 0.0f;
    for (int j8 = tid; j8 < d8; j8 += 128) {
      float yf[8];
      hy4_unpack_bf16x8(__ldg(y_u4 + j8), yf);
#pragma unroll
      for (int i = 0; i < 8; ++i) y2 = fmaf(yf[i], yf[i], y2);
    }
    y2 = hy4_warp_sum(y2);
    if (lane == 0) s_y2[warp] = y2;
    __syncthreads();
    const float y2_tot = (s_y2[0] + s_y2[1]) + (s_y2[2] + s_y2[3]);
    rs = rsqrtf(y2_tot / static_cast<float>(D) + norm_eps);
  }

  const int per = (d8 + num_ctas - 1) / num_ctas;
  const int j8_lo = b * per;
  const int j8_hi = min(j8_lo + per, d8);
  const float* post_row = post + static_cast<size_t>(m) * static_cast<size_t>(HC);

  for (int j8 = j8_lo + lane; j8 < j8_hi; j8 += 32) {
    float yn[8];
    hy4_unpack_bf16x8(__ldg(y_u4 + j8), yn);
    if (has_norm) {
      float nw[8];
      hy4_load_norm_w8(post_norm_w, j8, norm_bf16, nw);
#pragma unroll
      for (int i = 0; i < 8; ++i) yn[i] = yn[i] * rs * nw[i];
    }
    for (int c = warp; c < HC; c += 4) {
      const size_t c_off = x_off + static_cast<size_t>(c) * static_cast<size_t>(D);
      float xf[8];
      hy4_unpack_bf16x8(reinterpret_cast<const uint4*>(x + c_off)[j8], xf);
      const float pc = post_row[c];
#pragma unroll
      for (int i = 0; i < 8; ++i) xf[i] = fmaf(pc, yn[i], xf[i]);
      reinterpret_cast<uint4*>(out_x + c_off)[j8] = hy4_pack_bf16x8(xf);
    }
  }
}

// ----------------------------------------------------------------------------
// Kernel 3: hy4_engram_lookup_norm_cuda
//   raw_rows: f32 [M, n_cols, e_hd], lookup_norm: f32 [n_cols, e_hd]
//   out:      bf16 [M, n_cols * e_hd]
// Grid: dim3(n_cols, M, 1), Block: dim3(128, 1, 1)
// ----------------------------------------------------------------------------
__global__ __launch_bounds__(128, 4) void hy4_engram_lookup_norm_cuda(
    const float* __restrict__ raw_rows,
    const float* __restrict__ lookup_norm,
    nv_bfloat16* __restrict__ out,
    int M,
    int n_cols,
    int e_hd,
    float eps) {
  const int col = static_cast<int>(blockIdx.x);
  const int m = static_cast<int>(blockIdx.y);
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;
  if (m >= M || col >= n_cols) {
    return;
  }

  const size_t base = (static_cast<size_t>(m) * static_cast<size_t>(n_cols) +
                       static_cast<size_t>(col)) * static_cast<size_t>(e_hd);
  const size_t nw_base = static_cast<size_t>(col) * static_cast<size_t>(e_hd);
  __shared__ float s_sq[4];

  if (e_hd <= 128) {
    const float v = (tid < e_hd) ? raw_rows[base + tid] : 0.0f;
    float sq = hy4_warp_sum(v * v);
    if (lane == 0) s_sq[warp] = sq;
    __syncthreads();
    const float sq_tot = (s_sq[0] + s_sq[1]) + (s_sq[2] + s_sq[3]);
    const float rs = rsqrtf(sq_tot / static_cast<float>(e_hd) + eps);
    if (tid < e_hd) {
      out[base + tid] = __float2bfloat16(v * rs * lookup_norm[nw_base + tid]);
    }
  } else {
    float sq = 0.0f;
    for (int j = tid; j < e_hd; j += 128) {
      const float v = raw_rows[base + j];
      sq = fmaf(v, v, sq);
    }
    sq = hy4_warp_sum(sq);
    if (lane == 0) s_sq[warp] = sq;
    __syncthreads();
    const float sq_tot = (s_sq[0] + s_sq[1]) + (s_sq[2] + s_sq[3]);
    const float rs = rsqrtf(sq_tot / static_cast<float>(e_hd) + eps);
    for (int j = tid; j < e_hd; j += 128) {
      out[base + j] = __float2bfloat16(
          raw_rows[base + j] * rs * lookup_norm[nw_base + j]);
    }
  }
}

)cuda"
R"cuda(
// ----------------------------------------------------------------------------
// Kernel 4: hy4_engram_gate_add_cuda
//   x: bf16 [M, HC, D], kv: bf16 [M, HC+1, D], kv_b: f32 [HC+1, D],
//   gate_w: f32 [HC, D], pre_gamma: f32 [D], hc_scale: f32 [HC, D],
//   keep_mask: uint8 [M] (nullable), out_x: bf16 [M, HC, D].
// Grid: dim3(M, 1, 1), Block: dim3(256, 1, 1)
// ----------------------------------------------------------------------------
__global__ __launch_bounds__(256, 2) void hy4_engram_gate_add_cuda(
    const nv_bfloat16* __restrict__ x,
    const nv_bfloat16* __restrict__ kv,
    const float* __restrict__ kv_b,
    const float* __restrict__ gate_w,
    const float* __restrict__ pre_gamma,
    const float* __restrict__ hc_scale,
    const unsigned char* __restrict__ keep_mask,
    nv_bfloat16* __restrict__ out_x,
    int M,
    int HC,
    int D,
    float eps) {
  const int m = static_cast<int>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;
  if (m >= M) {
    return;
  }

  const size_t hd = static_cast<size_t>(HC) * static_cast<size_t>(D);
  const size_t x_row_off = static_cast<size_t>(m) * hd;

  if (keep_mask != nullptr && keep_mask[m] == 0u) {
    if ((hd & 7u) == 0u) {
      const size_t hd8 = hd >> 3;
      const uint4* x_u4 = reinterpret_cast<const uint4*>(x + x_row_off);
      uint4* o_u4 = reinterpret_cast<uint4*>(out_x + x_row_off);
      for (size_t i = static_cast<size_t>(tid); i < hd8; i += 256u) {
        o_u4[i] = x_u4[i];
      }
    } else {
      for (size_t i = static_cast<size_t>(tid); i < hd; i += 256u) {
        out_x[x_row_off + i] = x[x_row_off + i];
      }
    }
    return;
  }

  const float inv_sqrt_d = rsqrtf(static_cast<float>(D));
  const size_t kv_row_off = static_cast<size_t>(m) * static_cast<size_t>(HC + 1) * static_cast<size_t>(D);
  const size_t val_off = static_cast<size_t>(HC) * static_cast<size_t>(D);

  __shared__ float s_red[16];
  __shared__ float s_sig[4096];

  if (D <= 4096 && (D & 7) == 0) {
    const int d8 = D >> 3;
    const uint4* v_u4 = reinterpret_cast<const uint4*>(kv + kv_row_off + val_off);
    const float4* vb_f4 = reinterpret_cast<const float4*>(kv_b + val_off);
    const float4* pg_f4 = reinterpret_cast<const float4*>(pre_gamma);

    float v_reg[2][8];
    float v2 = 0.0f;
#pragma unroll
    for (int k = 0; k < 2; ++k) {
      const int j8 = tid + k * 256;
      if (j8 < d8) {
        hy4_unpack_bf16x8(v_u4[j8], v_reg[k]);
        const float4 b0 = vb_f4[2 * j8];
        const float4 b1 = vb_f4[2 * j8 + 1];
        v_reg[k][0] += b0.x; v_reg[k][1] += b0.y; v_reg[k][2] += b0.z; v_reg[k][3] += b0.w;
        v_reg[k][4] += b1.x; v_reg[k][5] += b1.y; v_reg[k][6] += b1.z; v_reg[k][7] += b1.w;
#pragma unroll
        for (int i = 0; i < 8; ++i) v2 = fmaf(v_reg[k][i], v_reg[k][i], v2);
      }
    }
    v2 = hy4_warp_sum(v2);
    if (lane == 0) s_red[warp] = v2;
    __syncthreads();
    float v2_tot = 0.0f;
#pragma unroll
    for (int w = 0; w < 8; ++w) v2_tot += s_red[w];
    __syncthreads();
    const float rs_v = rsqrtf(v2_tot / static_cast<float>(D) + eps);
#pragma unroll
    for (int k = 0; k < 2; ++k) {
      const int j8 = tid + k * 256;
      if (j8 < d8) {
        const float4 p0 = pg_f4[2 * j8];
        const float4 p1 = pg_f4[2 * j8 + 1];
        const int col0 = j8 * 8;
        s_sig[col0 + 0] = v_reg[k][0] * rs_v * p0.x;
        s_sig[col0 + 1] = v_reg[k][1] * rs_v * p0.y;
        s_sig[col0 + 2] = v_reg[k][2] * rs_v * p0.z;
        s_sig[col0 + 3] = v_reg[k][3] * rs_v * p0.w;
        s_sig[col0 + 4] = v_reg[k][4] * rs_v * p1.x;
        s_sig[col0 + 5] = v_reg[k][5] * rs_v * p1.y;
        s_sig[col0 + 6] = v_reg[k][6] * rs_v * p1.z;
        s_sig[col0 + 7] = v_reg[k][7] * rs_v * p1.w;
      }
    }

    for (int c = 0; c < HC; ++c) {
      const size_t c_off = static_cast<size_t>(c) * static_cast<size_t>(D);
      const uint4* xc_u4 = reinterpret_cast<const uint4*>(x + x_row_off + c_off);
      const uint4* kc_u4 = reinterpret_cast<const uint4*>(kv + kv_row_off + c_off);
      const float4* kb_f4 = reinterpret_cast<const float4*>(kv_b + c_off);
      const float4* gw_f4 = reinterpret_cast<const float4*>(gate_w + c_off);
      const float4* hs_f4 = reinterpret_cast<const float4*>(hc_scale + c_off);
      uint4* oc_u4 = reinterpret_cast<uint4*>(out_x + x_row_off + c_off);

      float x_reg[2][8];
      float k_reg[2][8];
      float x2 = 0.0f;
      float k2 = 0.0f;
#pragma unroll
      for (int k = 0; k < 2; ++k) {
        const int j8 = tid + k * 256;
        if (j8 < d8) {
          hy4_unpack_bf16x8(xc_u4[j8], x_reg[k]);
          hy4_unpack_bf16x8(kc_u4[j8], k_reg[k]);
          const float4 b0 = kb_f4[2 * j8];
          const float4 b1 = kb_f4[2 * j8 + 1];
          k_reg[k][0] += b0.x; k_reg[k][1] += b0.y; k_reg[k][2] += b0.z; k_reg[k][3] += b0.w;
          k_reg[k][4] += b1.x; k_reg[k][5] += b1.y; k_reg[k][6] += b1.z; k_reg[k][7] += b1.w;
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            x2 = fmaf(x_reg[k][i], x_reg[k][i], x2);
            k2 = fmaf(k_reg[k][i], k_reg[k][i], k2);
          }
        }
      }
      x2 = hy4_warp_sum(x2);
      k2 = hy4_warp_sum(k2);
      if (lane == 0) {
        s_red[warp] = x2;
        s_red[8 + warp] = k2;
      }
      __syncthreads();
      float x2_tot = 0.0f;
      float k2_tot = 0.0f;
#pragma unroll
      for (int w = 0; w < 8; ++w) {
        x2_tot += s_red[w];
        k2_tot += s_red[8 + w];
      }
      __syncthreads();
      const float rs_x = rsqrtf(x2_tot / static_cast<float>(D) + eps);
      const float rs_k = rsqrtf(k2_tot / static_cast<float>(D) + eps);

      float dot_acc = 0.0f;
#pragma unroll
      for (int k = 0; k < 2; ++k) {
        const int j8 = tid + k * 256;
        if (j8 < d8) {
          const float4 g0 = gw_f4[2 * j8];
          const float4 g1 = gw_f4[2 * j8 + 1];
          const float gw8[8] = {g0.x, g0.y, g0.z, g0.w, g1.x, g1.y, g1.z, g1.w};
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            const float xn = x_reg[k][i] * rs_x;
            const float kn = k_reg[k][i] * rs_k;
            dot_acc = fmaf(xn * kn, gw8[i], dot_acc);
          }
        }
      }
      dot_acc = hy4_warp_sum(dot_acc);
      if (lane == 0) s_red[warp] = dot_acc;
      __syncthreads();
      float dot_tot = 0.0f;
#pragma unroll
      for (int w = 0; w < 8; ++w) dot_tot += s_red[w];

      const float dot_val = dot_tot * inv_sqrt_d;
      const float mag = sqrtf(fmaxf(fabsf(dot_val), 1e-6f));
      const float signed_a = (dot_val < 0.0f) ? -mag : mag;
      const float g_c = hy4_sigmoid_f32(signed_a);

#pragma unroll
      for (int k = 0; k < 2; ++k) {
        const int j8 = tid + k * 256;
        if (j8 < d8) {
          const float4 h0 = hs_f4[2 * j8];
          const float4 h1 = hs_f4[2 * j8 + 1];
          const float hs8[8] = {h0.x, h0.y, h0.z, h0.w, h1.x, h1.y, h1.z, h1.w};
          const int col0 = j8 * 8;
          float out8[8];
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            out8[i] = fmaf(s_sig[col0 + i], g_c + hs8[i], x_reg[k][i]);
          }
          oc_u4[j8] = hy4_pack_bf16x8(out8);
        }
      }
      if (c + 1 < HC) {
        __syncthreads();
      }
    }
  } else {
    float v2 = 0.0f;
    for (int j = tid; j < D; j += 256) {
      const float vv = __bfloat162float(kv[kv_row_off + val_off + j]) + kv_b[val_off + j];
      v2 = fmaf(vv, vv, v2);
    }
    v2 = hy4_warp_sum(v2);
    if (lane == 0) s_red[warp] = v2;
    __syncthreads();
    float v2_tot = 0.0f;
#pragma unroll
    for (int w = 0; w < 8; ++w) v2_tot += s_red[w];
    __syncthreads();
    const float rs_v = rsqrtf(v2_tot / static_cast<float>(D) + eps);

    for (int c = 0; c < HC; ++c) {
      const size_t c_off = static_cast<size_t>(c) * static_cast<size_t>(D);
      float x2 = 0.0f;
      float k2 = 0.0f;
      for (int j = tid; j < D; j += 256) {
        const float xv = __bfloat162float(x[x_row_off + c_off + j]);
        const float kv_v = __bfloat162float(kv[kv_row_off + c_off + j]) + kv_b[c_off + j];
        x2 = fmaf(xv, xv, x2);
        k2 = fmaf(kv_v, kv_v, k2);
      }
      x2 = hy4_warp_sum(x2);
      k2 = hy4_warp_sum(k2);
      if (lane == 0) {
        s_red[warp] = x2;
        s_red[8 + warp] = k2;
      }
      __syncthreads();
      float x2_tot = 0.0f;
      float k2_tot = 0.0f;
#pragma unroll
      for (int w = 0; w < 8; ++w) {
        x2_tot += s_red[w];
        k2_tot += s_red[8 + w];
      }
      __syncthreads();
      const float rs_x = rsqrtf(x2_tot / static_cast<float>(D) + eps);
      const float rs_k = rsqrtf(k2_tot / static_cast<float>(D) + eps);

      float dot_acc = 0.0f;
      for (int j = tid; j < D; j += 256) {
        const float xn = __bfloat162float(x[x_row_off + c_off + j]) * rs_x;
        const float kn = (__bfloat162float(kv[kv_row_off + c_off + j]) + kv_b[c_off + j]) * rs_k;
        dot_acc = fmaf(xn * kn, gate_w[c_off + j], dot_acc);
      }
      dot_acc = hy4_warp_sum(dot_acc);
      if (lane == 0) s_red[warp] = dot_acc;
      __syncthreads();
      float dot_tot = 0.0f;
#pragma unroll
      for (int w = 0; w < 8; ++w) dot_tot += s_red[w];

      const float dot_val = dot_tot * inv_sqrt_d;
      const float mag = sqrtf(fmaxf(fabsf(dot_val), 1e-6f));
      const float signed_a = (dot_val < 0.0f) ? -mag : mag;
      const float g_c = hy4_sigmoid_f32(signed_a);

      for (int j = tid; j < D; j += 256) {
        const float vv = __bfloat162float(kv[kv_row_off + val_off + j]) + kv_b[val_off + j];
        const float sig = vv * rs_v * pre_gamma[j];
        const float xv = __bfloat162float(x[x_row_off + c_off + j]);
        out_x[x_row_off + c_off + j] = __float2bfloat16(fmaf(sig, g_c + hc_scale[c_off + j], xv));
      }
      if (c + 1 < HC) {
        __syncthreads();
      }
    }
  }
}

)cuda"
R"cuda(
// ----------------------------------------------------------------------------
// Kernel 5: hy4_mtp_eh_norm_concat_cuda
//   out[m, 0..D-1]    = bf16(rms_norm(emb[m], enorm, norm_eps))
//   out[m, D..2*D-1]  = bf16(rms_norm(hidden[m], hnorm, norm_eps))
// Grid: dim3(2, M, 1), Block: dim3(256, 1, 1)
// ----------------------------------------------------------------------------
__global__ __launch_bounds__(256, 4) void hy4_mtp_eh_norm_concat_cuda(
    const nv_bfloat16* __restrict__ emb,
    const nv_bfloat16* __restrict__ hidden,
    const float* __restrict__ enorm,
    const float* __restrict__ hnorm,
    nv_bfloat16* __restrict__ out,
    int M,
    int D,
    float norm_eps) {
  const int half_idx = static_cast<int>(blockIdx.x);
  const int m = static_cast<int>(blockIdx.y);
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;
  if (m >= M || half_idx >= 2) {
    return;
  }

  const nv_bfloat16* src = ((half_idx == 0) ? emb : hidden) +
      static_cast<size_t>(m) * static_cast<size_t>(D);
  const float* nw = (half_idx == 0) ? enorm : hnorm;
  nv_bfloat16* dst = out + (static_cast<size_t>(m) * 2u + static_cast<size_t>(half_idx)) *
      static_cast<size_t>(D);
  __shared__ float s_sq[8];

  if (D <= 4096 && (D & 7) == 0) {
    const int d8 = D >> 3;
    const uint4* src_u4 = reinterpret_cast<const uint4*>(src);
    const float4* nw_f4 = reinterpret_cast<const float4*>(nw);
    uint4* dst_u4 = reinterpret_cast<uint4*>(dst);
    float reg[2][8];
    float sq = 0.0f;
#pragma unroll
    for (int k = 0; k < 2; ++k) {
      const int j8 = tid + k * 256;
      if (j8 < d8) {
        hy4_unpack_bf16x8(src_u4[j8], reg[k]);
#pragma unroll
        for (int i = 0; i < 8; ++i) sq = fmaf(reg[k][i], reg[k][i], sq);
      }
    }
    sq = hy4_warp_sum(sq);
    if (lane == 0) s_sq[warp] = sq;
    __syncthreads();
    float sq_tot = 0.0f;
#pragma unroll
    for (int w = 0; w < 8; ++w) sq_tot += s_sq[w];
    const float rs = rsqrtf(sq_tot / static_cast<float>(D) + norm_eps);
#pragma unroll
    for (int k = 0; k < 2; ++k) {
      const int j8 = tid + k * 256;
      if (j8 < d8) {
        const float4 w0 = nw_f4[2 * j8];
        const float4 w1 = nw_f4[2 * j8 + 1];
        reg[k][0] *= rs * w0.x; reg[k][1] *= rs * w0.y;
        reg[k][2] *= rs * w0.z; reg[k][3] *= rs * w0.w;
        reg[k][4] *= rs * w1.x; reg[k][5] *= rs * w1.y;
        reg[k][6] *= rs * w1.z; reg[k][7] *= rs * w1.w;
        dst_u4[j8] = hy4_pack_bf16x8(reg[k]);
      }
    }
  } else {
    float sq = 0.0f;
    for (int j = tid; j < D; j += 256) {
      const float v = __bfloat162float(src[j]);
      sq = fmaf(v, v, sq);
    }
    sq = hy4_warp_sum(sq);
    if (lane == 0) s_sq[warp] = sq;
    __syncthreads();
    float sq_tot = 0.0f;
#pragma unroll
    for (int w = 0; w < 8; ++w) sq_tot += s_sq[w];
    const float rs = rsqrtf(sq_tot / static_cast<float>(D) + norm_eps);
    for (int j = tid; j < D; j += 256) {
      dst[j] = __float2bfloat16(__bfloat162float(src[j]) * rs * nw[j]);
    }
  }
}
)cuda"
