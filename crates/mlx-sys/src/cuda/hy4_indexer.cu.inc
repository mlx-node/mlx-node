R"cuda(
// ============================================================================
// Hy4-Flash K-Pool Lightning Indexer Kernels (H_idx=16, d_idx=128, kpool=4)
//   1. hy4_kpool_compress_128_4  (fast d_idx=128, kpool=4 compressor + tail RoPE)
//   2. hy4_kpool_compress_generic (arbitrary d_idx, kpool compressor + tail RoPE)
//   3. hy4_indexer_dense_fill    (fast all-dense shortcut)
//   4. hy4_indexer_q_rope        (tail NORM-RoPE on q_idx_raw -> f32 or bf16)
//   5. hy4_indexer_score_decode_128 (shared-K FP32 scorer for M <= 16)
//   6. hy4_indexer_score_tc_128  (BF16 Tensor-Core scorer for M > 16, d_idx=128)
//   7. hy4_indexer_score_generic (FP32 fallback scorer for arbitrary d_idx)
//   8. hy4_indexer_radix_select  (exact 11+11+10-bit Radix-Select, lowest-g tie-break,
//                                 ascending group expansion + own-group tail)
// ============================================================================

__device__ __forceinline__ int hy4_idx_swiz128(int row, int c8) {
  return row * 128 + ((c8 ^ (row & 7)) << 3);
}

__device__ __forceinline__ uint4 hy4_idx_f16x8_to_bf16x8(const uint4& hv) {
  const __half2* hp = reinterpret_cast<const __half2*>(&hv);
  const float2 f0 = __half22float2(hp[0]);
  const float2 f1 = __half22float2(hp[1]);
  const float2 f2 = __half22float2(hp[2]);
  const float2 f3 = __half22float2(hp[3]);
  uint4 out;
  out.x = hy4_pack_bf16x2(f0.x, f0.y);
  out.y = hy4_pack_bf16x2(f1.x, f1.y);
  out.z = hy4_pack_bf16x2(f2.x, f2.y);
  out.w = hy4_pack_bf16x2(f3.x, f3.y);
  return out;
}

__device__ __forceinline__ float hy4_idx_read_tok(
    const void* __restrict__ ptr, size_t idx, bool is_bf16) {
  if (is_bf16) {
    return __bfloat162float(reinterpret_cast<const nv_bfloat16*>(ptr)[idx]);
  }
  return reinterpret_cast<const float*>(ptr)[idx];
}

// ----------------------------------------------------------------------------
// Kernel 1: hy4_kpool_compress_128_4 (d_idx=128, kpool=4)
// Grid: dim3(n_groups, 1, 1), Block: dim3(256, 1, 1)
// ----------------------------------------------------------------------------
template <bool TOK_BF16>
__global__ __launch_bounds__(256, 2) void hy4_kpool_compress_128_4(
    const void* __restrict__ tok_k,
    const float* __restrict__ w_val,
    const float* __restrict__ w_gate,
    const float* __restrict__ ape,
    const float* __restrict__ comp_norm,
    const float* __restrict__ inv_freq,
    __half* __restrict__ pooled_buf,
    int n_groups,
    long long start_group,
    int half_rope,
    int ring_cap,
    float norm_eps) {
  const int g = static_cast<int>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;
  if (g >= n_groups) {
    return;
  }

  __shared__ __align__(16) float4 x_sm[4][32];
  __shared__ float s_pooled[128];
  __shared__ float s_sq[4];

  if (tid < 128) {
    const int s = tid >> 5;
    const int c4 = tid & 31;
    const size_t row = (ring_cap > 0)
        ? ((static_cast<size_t>(start_group + static_cast<long long>(g)) * 4u +
            static_cast<size_t>(s)) % static_cast<size_t>(ring_cap))
        : (static_cast<size_t>(g) * 4u + static_cast<size_t>(s));
    const size_t base = row * 128u + static_cast<size_t>(c4) * 4u;
    if constexpr (TOK_BF16) {
      const nv_bfloat16* tp = reinterpret_cast<const nv_bfloat16*>(tok_k) + base;
      x_sm[s][c4] = make_float4(
          __bfloat162float(tp[0]), __bfloat162float(tp[1]),
          __bfloat162float(tp[2]), __bfloat162float(tp[3]));
    } else {
      x_sm[s][c4] = *reinterpret_cast<const float4*>(reinterpret_cast<const float*>(tok_k) + base);
    }
  }
  __syncthreads();

  const float4* wv4 = reinterpret_cast<const float4*>(w_val);
  const float4* wg4 = reinterpret_cast<const float4*>(w_gate);
  const float4 x0 = x_sm[0][lane];
  const float4 x1 = x_sm[1][lane];
  const float4 x2 = x_sm[2][lane];
  const float4 x3 = x_sm[3][lane];

  for (int iter = 0; iter < 16; ++iter) {
    const int c = warp + iter * 8;
    const float4 wv = wv4[static_cast<size_t>(c) * 32u + static_cast<size_t>(lane)];
    const float4 wg = wg4[static_cast<size_t>(c) * 32u + static_cast<size_t>(lane)];

    const float v0 = hy4_warp_sum(fmaf(x0.x, wv.x, fmaf(x0.y, wv.y, fmaf(x0.z, wv.z, x0.w * wv.w))));
    const float v1 = hy4_warp_sum(fmaf(x1.x, wv.x, fmaf(x1.y, wv.y, fmaf(x1.z, wv.z, x1.w * wv.w))));
    const float v2 = hy4_warp_sum(fmaf(x2.x, wv.x, fmaf(x2.y, wv.y, fmaf(x2.z, wv.z, x2.w * wv.w))));
    const float v3 = hy4_warp_sum(fmaf(x3.x, wv.x, fmaf(x3.y, wv.y, fmaf(x3.z, wv.z, x3.w * wv.w))));

    const float g0 = hy4_warp_sum(fmaf(x0.x, wg.x, fmaf(x0.y, wg.y, fmaf(x0.z, wg.z, x0.w * wg.w))));
    const float g1 = hy4_warp_sum(fmaf(x1.x, wg.x, fmaf(x1.y, wg.y, fmaf(x1.z, wg.z, x1.w * wg.w))));
    const float g2 = hy4_warp_sum(fmaf(x2.x, wg.x, fmaf(x2.y, wg.y, fmaf(x2.z, wg.z, x2.w * wg.w))));
    const float g3 = hy4_warp_sum(fmaf(x3.x, wg.x, fmaf(x3.y, wg.y, fmaf(x3.z, wg.z, x3.w * wg.w))));

    if (lane == 0) {
      const float a0 = g0 + ape[0 * 128 + c];
      const float a1 = g1 + ape[1 * 128 + c];
      const float a2 = g2 + ape[2 * 128 + c];
      const float a3 = g3 + ape[3 * 128 + c];
      const float gm = fmaxf(fmaxf(a0, a1), fmaxf(a2, a3));
      const float e0 = expf(a0 - gm);
      const float e1 = expf(a1 - gm);
      const float e2 = expf(a2 - gm);
      const float e3 = expf(a3 - gm);
      const float inv_es = 1.0f / ((e0 + e1) + (e2 + e3));
      s_pooled[c] = (v0 * e0 + v1 * e1 + v2 * e2 + v3 * e3) * inv_es;
    }
  }
  __syncthreads();

  float pv = (tid < 128) ? s_pooled[tid] : 0.0f;
  const float sq = hy4_warp_sum(pv * pv);
  if (lane == 0 && warp < 4) {
    s_sq[warp] = sq;
  }
  __syncthreads();

  const float sq_tot = (s_sq[0] + s_sq[1]) + (s_sq[2] + s_sq[3]);
  const float rs = rsqrtf(sq_tot / 128.0f + norm_eps);
  if (tid < 128) {
    s_pooled[tid] = pv * rs * comp_norm[tid];
  }
  __syncthreads();

  const int rope_dim = half_rope * 2;
  const int head_w = 128 - rope_dim;
  const size_t dst_off = static_cast<size_t>(start_group + static_cast<long long>(g)) * 128u;
  if (tid < head_w) {
    pooled_buf[dst_off + tid] = __float2half(s_pooled[tid]);
  } else if (tid < head_w + half_rope) {
    const int p = tid - head_w;
    const float pos_g = static_cast<float>((start_group + static_cast<long long>(g)) * 4LL);
    float s_v, c_v;
    sincosf(pos_g * inv_freq[p], &s_v, &c_v);
    const float r0 = s_pooled[head_w + 2 * p];
    const float r1 = s_pooled[head_w + 2 * p + 1];
    pooled_buf[dst_off + head_w + 2 * p] = __float2half(r0 * c_v - r1 * s_v);
    pooled_buf[dst_off + head_w + 2 * p + 1] = __float2half(r0 * s_v + r1 * c_v);
  }
}

)cuda"
R"cuda(
// ----------------------------------------------------------------------------
// Kernel 2: hy4_kpool_compress_generic (arbitrary d_idx, kpool <= 16)
//   Dynamic SMEM: d_idx * sizeof(float)
// Grid: dim3(n_groups, 1, 1), Block: dim3(256, 1, 1)
// ----------------------------------------------------------------------------
template <bool TOK_BF16>
__global__ __launch_bounds__(256, 2) void hy4_kpool_compress_generic(
    const void* __restrict__ tok_k,
    const float* __restrict__ w_val,
    const float* __restrict__ w_gate,
    const float* __restrict__ ape,
    const float* __restrict__ comp_norm,
    const float* __restrict__ inv_freq,
    __half* __restrict__ pooled_buf,
    int n_groups,
    int d_idx,
    int kpool,
    long long start_group,
    int half_rope,
    int ring_cap,
    float norm_eps) {
  extern __shared__ float s_pooled[];
  __shared__ float s_sq[8];

  const int g = static_cast<int>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;
  if (g >= n_groups) {
    return;
  }

  for (int c = warp; c < d_idx; c += 8) {
    const float* wv = w_val + static_cast<size_t>(c) * static_cast<size_t>(d_idx);
    const float* wg = w_gate + static_cast<size_t>(c) * static_cast<size_t>(d_idx);
    float v_arr[16];
    float a_arr[16];
    float gm = -1e30f;
    for (int s = 0; s < kpool; ++s) {
      const size_t row = (ring_cap > 0)
          ? ((static_cast<size_t>(start_group + static_cast<long long>(g)) *
              static_cast<size_t>(kpool) + static_cast<size_t>(s)) %
             static_cast<size_t>(ring_cap))
          : (static_cast<size_t>(g) * static_cast<size_t>(kpool) +
             static_cast<size_t>(s));
      const size_t t_off = row * static_cast<size_t>(d_idx);
      float dv = 0.0f;
      float dg = 0.0f;
      for (int k = lane; k < d_idx; k += 32) {
        const float xv = hy4_idx_read_tok(tok_k, t_off + k, TOK_BF16);
        dv = fmaf(xv, wv[k], dv);
        dg = fmaf(xv, wg[k], dg);
      }
      const float vs = hy4_warp_sum(dv);
      const float gs = hy4_warp_sum(dg);
      if (lane == 0) {
        v_arr[s] = vs;
        const float av = gs + ape[static_cast<size_t>(s) * static_cast<size_t>(d_idx) + c];
        a_arr[s] = av;
        gm = (s == 0) ? av : fmaxf(gm, av);
      }
    }
    if (lane == 0) {
      float es = 0.0f;
      float acc = 0.0f;
      for (int s = 0; s < kpool; ++s) {
        const float e = expf(a_arr[s] - gm);
        es += e;
        acc = fmaf(v_arr[s], e, acc);
      }
      s_pooled[c] = acc / es;
    }
  }
  __syncthreads();

  float sq = 0.0f;
  for (int j = tid; j < d_idx; j += 256) {
    const float v = s_pooled[j];
    sq = fmaf(v, v, sq);
  }
  sq = hy4_warp_sum(sq);
  if (lane == 0) s_sq[warp] = sq;
  __syncthreads();
  float sq_tot = 0.0f;
#pragma unroll
  for (int w = 0; w < 8; ++w) sq_tot += s_sq[w];
  const float rs = rsqrtf(sq_tot / static_cast<float>(d_idx) + norm_eps);
  for (int j = tid; j < d_idx; j += 256) {
    s_pooled[j] = s_pooled[j] * rs * comp_norm[j];
  }
  __syncthreads();

  const int rope_dim = half_rope * 2;
  const int head_w = d_idx - rope_dim;
  const size_t dst_off = static_cast<size_t>(start_group + static_cast<long long>(g)) *
                         static_cast<size_t>(d_idx);
  for (int j = tid; j < head_w; j += 256) {
    pooled_buf[dst_off + j] = __float2half(s_pooled[j]);
  }
  const float pos_g = static_cast<float>(
      (start_group + static_cast<long long>(g)) * static_cast<long long>(kpool));
  for (int p = tid; p < half_rope; p += 256) {
    float s_v, c_v;
    sincosf(pos_g * inv_freq[p], &s_v, &c_v);
    const float r0 = s_pooled[head_w + 2 * p];
    const float r1 = s_pooled[head_w + 2 * p + 1];
    pooled_buf[dst_off + head_w + 2 * p] = __float2half(r0 * c_v - r1 * s_v);
    pooled_buf[dst_off + head_w + 2 * p + 1] = __float2half(r0 * s_v + r1 * c_v);
  }
}

// ----------------------------------------------------------------------------
// Kernel 3: hy4_indexer_dense_fill
//   Fills out_idx[m, j] = (j <= slot_start + m) ? j : -1
// Grid: dim3(M, 1, 1), Block: dim3(256, 1, 1)
// ----------------------------------------------------------------------------
__global__ __launch_bounds__(256, 4) void hy4_indexer_dense_fill(
    int* __restrict__ out_idx,
    int M,
    int kmax,
    long long slot_start) {
  const int m = static_cast<int>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.x);
  if (m >= M) {
    return;
  }
  const long long p = slot_start + static_cast<long long>(m);
  int* dst = out_idx + static_cast<size_t>(m) * static_cast<size_t>(kmax);
  for (int j = tid; j < kmax; j += 256) {
    dst[j] = (static_cast<long long>(j) <= p) ? j : -1;
  }
}

// ----------------------------------------------------------------------------
// Kernel 4: hy4_indexer_q_rope
//   Applies tail NORM-RoPE to q_idx_raw[M, H_idx, d_idx] at rope_start + m.
// Grid: dim3(H_idx, M, 1), Block: dim3(128, 1, 1)
// ----------------------------------------------------------------------------
template <bool OUT_F32>
__global__ __launch_bounds__(128, 4) void hy4_indexer_q_rope(
    const nv_bfloat16* __restrict__ q_raw,
    const float* __restrict__ inv_freq,
    void* __restrict__ out_q,
    int M,
    int H_idx,
    int d_idx,
    int half_rope,
    long long rope_start) {
  const int h = static_cast<int>(blockIdx.x);
  const int m = static_cast<int>(blockIdx.y);
  const int tid = static_cast<int>(threadIdx.x);
  if (m >= M || h >= H_idx) {
    return;
  }
  const int rope_dim = half_rope * 2;
  const int head_w = d_idx - rope_dim;
  const size_t off = (static_cast<size_t>(m) * static_cast<size_t>(H_idx) +
                      static_cast<size_t>(h)) * static_cast<size_t>(d_idx);
  const nv_bfloat16* src = q_raw + off;
  const float pos_f = static_cast<float>(rope_start + static_cast<long long>(m));

  if constexpr (OUT_F32) {
    float* dst = reinterpret_cast<float*>(out_q) + off;
    for (int j = tid; j < head_w; j += 128) {
      dst[j] = __bfloat162float(src[j]);
    }
    for (int p = tid; p < half_rope; p += 128) {
      float s_v, c_v;
      sincosf(pos_f * inv_freq[p], &s_v, &c_v);
      const float x0 = __bfloat162float(src[head_w + 2 * p]);
      const float x1 = __bfloat162float(src[head_w + 2 * p + 1]);
      dst[head_w + 2 * p] = x0 * c_v - x1 * s_v;
      dst[head_w + 2 * p + 1] = x0 * s_v + x1 * c_v;
    }
  } else {
    nv_bfloat16* dst = reinterpret_cast<nv_bfloat16*>(out_q) + off;
    for (int j = tid; j < head_w; j += 128) {
      dst[j] = src[j];
    }
    for (int p = tid; p < half_rope; p += 128) {
      float s_v, c_v;
      sincosf(pos_f * inv_freq[p], &s_v, &c_v);
      const float x0 = __bfloat162float(src[head_w + 2 * p]);
      const float x1 = __bfloat162float(src[head_w + 2 * p + 1]);
      dst[head_w + 2 * p] = __float2bfloat16(x0 * c_v - x1 * s_v);
      dst[head_w + 2 * p + 1] = __float2bfloat16(x0 * s_v + x1 * c_v);
    }
  }
}
)cuda"
R"cuda(
// ----------------------------------------------------------------------------
// Kernel 5: hy4_indexer_score_decode_128 (Shared-K FP32 scorer, M <= 16, d_idx=128)
//   Reads pooled_buf[g] ONCE across all M queries and all H_idx heads!
//   16 half-warps = 16 groups per step, 2 steps = 32 groups per CTA.
// Grid: dim3((max_vis + 31) / 32, 1, 1), Block: dim3(256, 1, 1)
// ----------------------------------------------------------------------------
template <bool W_F32>
__global__ __launch_bounds__(256, 2) void hy4_indexer_score_decode_128(
    const float* __restrict__ q_f32,
    const void* __restrict__ w_proj,
    const __half* __restrict__ pooled_buf,
    float* __restrict__ scores,
    int M,
    int H_idx,
    int max_vis,
    int score_stride,
    float weight_scale) {
  const int tid = static_cast<int>(threadIdx.x);
  const int hw = tid >> 4;
  const int lane16 = tid & 15;
  const unsigned int hw_mask = (hw & 1) ? 0xFFFF0000u : 0x0000FFFFu;
  const int g_base = static_cast<int>(blockIdx.x) * 32;

#pragma unroll
  for (int step = 0; step < 2; ++step) {
    const int g = g_base + step * 16 + hw;
    if (g >= max_vis) {
      continue;
    }
    const uint4 kv = __ldg(reinterpret_cast<const uint4*>(
        pooled_buf + static_cast<size_t>(g) * 128u) + lane16);
    const __half2* kp = reinterpret_cast<const __half2*>(&kv);
    float kf[8];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float2 f = __half22float2(kp[j]);
      kf[2 * j] = f.x;
      kf[2 * j + 1] = f.y;
    }

    for (int m = 0; m < M; ++m) {
      float sum_sc = 0.0f;
      const size_t wm_off = static_cast<size_t>(m) * static_cast<size_t>(H_idx);
      for (int h = 0; h < H_idx; ++h) {
        const float4* qp = reinterpret_cast<const float4*>(
            q_f32 + (wm_off + static_cast<size_t>(h)) * 128u) + lane16 * 2;
        const float4 q0 = __ldg(qp);
        const float4 q1 = __ldg(qp + 1);
        float dot = fmaf(q0.x, kf[0], fmaf(q0.y, kf[1], fmaf(q0.z, kf[2], q0.w * kf[3])));
        dot = fmaf(q1.x, kf[4], fmaf(q1.y, kf[5], fmaf(q1.z, kf[6], fmaf(q1.w, kf[7], dot))));
        dot += __shfl_xor_sync(hw_mask, dot, 8, 16);
        dot += __shfl_xor_sync(hw_mask, dot, 4, 16);
        dot += __shfl_xor_sync(hw_mask, dot, 2, 16);
        dot += __shfl_xor_sync(hw_mask, dot, 1, 16);
        const float w_raw = W_F32
            ? reinterpret_cast<const float*>(w_proj)[wm_off + h]
            : __bfloat162float(reinterpret_cast<const nv_bfloat16*>(w_proj)[wm_off + h]);
        sum_sc = fmaf(w_raw * weight_scale, fmaxf(dot, 0.0f), sum_sc);
      }
      if (lane16 == 0) {
        scores[static_cast<size_t>(m) * static_cast<size_t>(score_stride) + g] = sum_sc;
      }
    }
  }
}

// ----------------------------------------------------------------------------
// Kernel 6: hy4_indexer_score_tc_128 (BF16 Tensor-Core Prefill Scorer, d_idx=128)
//   Tile: BM=64 queries, BN=64 groups, 8 warps (256 thr), 32 KB SMEM (3 CTAs/SM).
//   Skips causal upper-triangle tiles where b_base >= tile_max_vis.
// Grid: dim3((chunk_max_vis + 63) / 64, (q_cnt + 63) / 64, 1), Block: dim3(256, 1, 1)
// ----------------------------------------------------------------------------
template <bool W_F32>
__global__ __launch_bounds__(256, 2) void hy4_indexer_score_tc_128(
    const nv_bfloat16* __restrict__ q_bf16,
    const void* __restrict__ w_proj,
    const __half* __restrict__ pooled_buf,
    float* __restrict__ scores,
    int H_idx,
    int q0,
    int q_cnt,
    int chunk_max_vis,
    int score_stride,
    long long slot_start,
    int kpool,
    int select_tail,
    float weight_scale) {
  constexpr int BM = 64;
  constexpr int BN = 64;
  constexpr int ID = 128;

  const int b_base = static_cast<int>(blockIdx.x) * BN;
  const int ql_base = static_cast<int>(blockIdx.y) * BM;
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int warp_m = warp >> 1;
  const int warp_n = warp & 1;

  // Exact causal upper-triangle skip for this BM=64 query block
  const int last_ql = min(ql_base + BM - 1, q_cnt - 1);
  const long long p_last = slot_start + static_cast<long long>(q0 + last_ql);
  const long long tile_vis_ll = select_tail
      ? (p_last / static_cast<long long>(kpool))
      : ((p_last + 1LL) / static_cast<long long>(kpool));
  const int tile_max_vis = min(chunk_max_vis, static_cast<int>(tile_vis_ll));
  if (b_base >= tile_max_vis) {
    return;
  }

  __shared__ __align__(128) nv_bfloat16 sK[BN * ID];
  __shared__ __align__(128) nv_bfloat16 sQ[BM * ID];

  const int c8 = tid & 15;
  const int r0 = tid >> 4;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int r = r0 + i * 16;
    const int gb = b_base + r;
    uint4 bfv = make_uint4(0u, 0u, 0u, 0u);
    if (gb < tile_max_vis) {
      const uint4 hv = __ldg(reinterpret_cast<const uint4*>(
          pooled_buf + static_cast<size_t>(gb) * ID + c8 * 8));
      bfv = hy4_idx_f16x8_to_bf16x8(hv);
    }
    *reinterpret_cast<uint4*>(&sK[hy4_idx_swiz128(r, c8)]) = bfv;
  }

  float acc_sum[4][4] = {};
  const int q_ld_r = warp_m * 16 + (lane & 15);
  const int q_ld_c8_base = (lane >> 4) & 1;
  const int k_ld_r0 = ((lane >> 4) * 8) + (lane & 7);
  const int k_ld_c8_base = (lane >> 3) & 1;
  const int k_n_base = warp_n * 32;
  const int group_id = lane >> 2;
  const int tid_in_grp = lane & 3;
  const int ql0 = ql_base + warp_m * 16 + group_id;
  const int ql1 = ql0 + 8;

  for (int h = 0; h < H_idx; ++h) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int r = r0 + i * 16;
      const int ql = ql_base + r;
      const bool ok = (ql < q_cnt);
      const nv_bfloat16* gp = ok
          ? (q_bf16 + (static_cast<size_t>(q0 + ql) * static_cast<size_t>(H_idx) +
                       static_cast<size_t>(h)) * ID + c8 * 8)
          : q_bf16;
      ptx_cp_async16_cg_zfill(&sQ[hy4_idx_swiz128(r, c8)], gp, ok);
    }
    ptx_cp_commit();
    ptx_cp_wait<0>();
    __syncthreads();

    float acc_h[4][4] = {};
#pragma unroll
    for (int kt = 0; kt < 8; ++kt) {
      unsigned int a0, a1, a2, a3;
      ptx_ldmatrix_x4(a0, a1, a2, a3, &sQ[hy4_idx_swiz128(q_ld_r, q_ld_c8_base + kt * 2)]);
      const int kc8 = k_ld_c8_base + kt * 2;
      unsigned int b0, b1, b2, b3;
      ptx_ldmatrix_x4(b0, b1, b2, b3, &sK[hy4_idx_swiz128(k_n_base + k_ld_r0, kc8)]);
      ptx_mma_m16n8k16_bf16_c_v(
          acc_h[0][0], acc_h[0][1], acc_h[0][2], acc_h[0][3],
          a0, a1, a2, a3, b0, b1,
          acc_h[0][0], acc_h[0][1], acc_h[0][2], acc_h[0][3]);
      ptx_mma_m16n8k16_bf16_c_v(
          acc_h[1][0], acc_h[1][1], acc_h[1][2], acc_h[1][3],
          a0, a1, a2, a3, b2, b3,
          acc_h[1][0], acc_h[1][1], acc_h[1][2], acc_h[1][3]);
      ptx_ldmatrix_x4(b0, b1, b2, b3, &sK[hy4_idx_swiz128(k_n_base + 16 + k_ld_r0, kc8)]);
      ptx_mma_m16n8k16_bf16_c_v(
          acc_h[2][0], acc_h[2][1], acc_h[2][2], acc_h[2][3],
          a0, a1, a2, a3, b0, b1,
          acc_h[2][0], acc_h[2][1], acc_h[2][2], acc_h[2][3]);
      ptx_mma_m16n8k16_bf16_c_v(
          acc_h[3][0], acc_h[3][1], acc_h[3][2], acc_h[3][3],
          a0, a1, a2, a3, b2, b3,
          acc_h[3][0], acc_h[3][1], acc_h[3][2], acc_h[3][3]);
    }

    const float w0 = (ql0 < q_cnt)
        ? ((W_F32
                ? reinterpret_cast<const float*>(w_proj)[static_cast<size_t>(q0 + ql0) * H_idx + h]
                : __bfloat162float(reinterpret_cast<const nv_bfloat16*>(w_proj)[static_cast<size_t>(q0 + ql0) * H_idx + h])) *
           weight_scale)
        : 0.0f;
    const float w1 = (ql1 < q_cnt)
        ? ((W_F32
                ? reinterpret_cast<const float*>(w_proj)[static_cast<size_t>(q0 + ql1) * H_idx + h]
                : __bfloat162float(reinterpret_cast<const nv_bfloat16*>(w_proj)[static_cast<size_t>(q0 + ql1) * H_idx + h])) *
           weight_scale)
        : 0.0f;
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      acc_sum[nt][0] = fmaf(w0, fmaxf(acc_h[nt][0], 0.0f), acc_sum[nt][0]);
      acc_sum[nt][1] = fmaf(w0, fmaxf(acc_h[nt][1], 0.0f), acc_sum[nt][1]);
      acc_sum[nt][2] = fmaf(w1, fmaxf(acc_h[nt][2], 0.0f), acc_sum[nt][2]);
      acc_sum[nt][3] = fmaf(w1, fmaxf(acc_h[nt][3], 0.0f), acc_sum[nt][3]);
    }
    __syncthreads();
  }

  const int col_base = b_base + k_n_base + 2 * tid_in_grp;
  if (ql0 < q_cnt) {
    float* dst0 = scores + static_cast<size_t>(ql0) * static_cast<size_t>(score_stride);
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      const int c = col_base + nt * 8;
      if (c < score_stride) {
        *reinterpret_cast<float2*>(dst0 + c) = make_float2(acc_sum[nt][0], acc_sum[nt][1]);
      }
    }
  }
  if (ql1 < q_cnt) {
    float* dst1 = scores + static_cast<size_t>(ql1) * static_cast<size_t>(score_stride);
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      const int c = col_base + nt * 8;
      if (c < score_stride) {
        *reinterpret_cast<float2*>(dst1 + c) = make_float2(acc_sum[nt][2], acc_sum[nt][3]);
      }
    }
  }
}

// ----------------------------------------------------------------------------
// Kernel 7: hy4_indexer_score_generic (arbitrary d_idx fallback, exact FP32)
// Grid: dim3((chunk_max_vis + 7) / 8, q_cnt, 1), Block: dim3(32, 8, 1)
// ----------------------------------------------------------------------------
template <bool W_F32>
__global__ __launch_bounds__(256, 2) void hy4_indexer_score_generic(
    const float* __restrict__ q_f32,
    const void* __restrict__ w_proj,
    const __half* __restrict__ pooled_buf,
    float* __restrict__ scores,
    int H_idx,
    int d_idx,
    int q0,
    int q_cnt,
    int chunk_max_vis,
    int score_stride,
    float weight_scale) {
  const int g = static_cast<int>(blockIdx.x) * 8 + static_cast<int>(threadIdx.y);
  const int ql = static_cast<int>(blockIdx.y);
  const int lane = static_cast<int>(threadIdx.x);
  if (ql >= q_cnt || g >= chunk_max_vis) {
    return;
  }
  const int m = q0 + ql;
  const __half* kp = pooled_buf + static_cast<size_t>(g) * static_cast<size_t>(d_idx);
  const size_t wm_off = static_cast<size_t>(m) * static_cast<size_t>(H_idx);
  float total = 0.0f;
  for (int h = 0; h < H_idx; ++h) {
    const float* qp = q_f32 + (wm_off + static_cast<size_t>(h)) * static_cast<size_t>(d_idx);
    float dot = 0.0f;
    for (int d = lane; d < d_idx; d += 32) {
      dot = fmaf(qp[d], __half2float(kp[d]), dot);
    }
    dot = hy4_warp_sum(dot);
    const float w_raw = W_F32
        ? reinterpret_cast<const float*>(w_proj)[wm_off + h]
        : __bfloat162float(reinterpret_cast<const nv_bfloat16*>(w_proj)[wm_off + h]);
    total = fmaf(w_raw * weight_scale, fmaxf(dot, 0.0f), total);
  }
  if (lane == 0) {
    scores[static_cast<size_t>(ql) * static_cast<size_t>(score_stride) + g] = total;
  }
}
)cuda"
R"cuda(
// ----------------------------------------------------------------------------
// Kernel 8: hy4_indexer_radix_select
//   Exact IEEE-754 order-preserving 11+11+10-bit Radix-Select supporting
//   negative/positive scores, -0.0f canonicalization, NaN -> lowest, and exact
//   tie-breaking to LOWEST group index g.
//   Emits strictly ascending expanded cells directly into out_idx[q, 0..kmax-1],
//   followed by own-group tail [kp*vis .. p] and -1 padding.
// Grid: dim3(q_cnt, 1, 1), Block: dim3(256, 1, 1)
// ----------------------------------------------------------------------------
__device__ __forceinline__ unsigned int hy4_f2ui(float f) {
  if (isnan(f)) {
    return 0u;
  }
  if (f == 0.0f) {
    return 0x80000000u;
  }
  const unsigned int u = __float_as_uint(f);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

__device__ __forceinline__ unsigned int hy4_block_excl_scan_256(
    unsigned int val, unsigned int* s_warp, int tid, unsigned int& out_total) {
  const int lane = tid & 31;
  const int warp = tid >> 5;
  unsigned int x = val;
#pragma unroll
  for (int d = 1; d < 32; d <<= 1) {
    const unsigned int up = __shfl_up_sync(0xFFFFFFFFu, x, d);
    if (lane >= d) x += up;
  }
  if (lane == 31) s_warp[warp] = x;
  __syncthreads();
  if (warp == 0) {
    unsigned int wv = (lane < 8) ? s_warp[lane] : 0u;
#pragma unroll
    for (int d = 1; d < 8; d <<= 1) {
      const unsigned int up = __shfl_up_sync(0xFFFFFFFFu, wv, d);
      if (lane >= d) wv += up;
    }
    if (lane < 8) s_warp[lane] = wv;
  }
  __syncthreads();
  out_total = s_warp[7];
  const unsigned int res = ((warp > 0) ? s_warp[warp - 1] : 0u) + (x - val);
  __syncthreads();
  return res;
}

template <int NBKT>
__device__ __forceinline__ void hy4_hist_find_thr(
    const unsigned int* hist,
    unsigned int need,
    unsigned int* s_warp,
    unsigned int* s_res,
    int tid,
    unsigned int& out_bkt,
    unsigned int& out_gt) {
  constexpr int PER_T = NBKT / 256;
  const int hi_start = NBKT - 1 - tid * PER_T;
  unsigned int local_sum = 0u;
#pragma unroll
  for (int i = 0; i < PER_T; ++i) {
    local_sum += hist[hi_start - i];
  }
  unsigned int total_unused = 0u;
  const unsigned int excl = hy4_block_excl_scan_256(local_sum, s_warp, tid, total_unused);
  const unsigned int incl = excl + local_sum;
  if (excl < need && incl >= need) {
    unsigned int cur = excl;
#pragma unroll
    for (int i = 0; i < PER_T; ++i) {
      const int b = hi_start - i;
      const unsigned int c = hist[b];
      if (cur + c >= need) {
        s_res[0] = static_cast<unsigned int>(b);
        s_res[1] = cur;
        break;
      }
      cur += c;
    }
  }
  __syncthreads();
  out_bkt = s_res[0];
  out_gt = s_res[1];
  __syncthreads();
}

__global__ __launch_bounds__(256, 2) void hy4_indexer_radix_select(
    const float* __restrict__ scores,
    int* __restrict__ out_idx,
    int q0,
    int score_stride,
    long long slot_start,
    int kpool,
    int n_top,
    int kmax,
    int select_tail) {
  const int ql = static_cast<int>(blockIdx.x);
  const int q = q0 + ql;
  const int tid = static_cast<int>(threadIdx.x);

  const long long p = slot_start + static_cast<long long>(q);
  const long long vis_ll = select_tail
      ? (p / static_cast<long long>(kpool))
      : ((p + 1LL) / static_cast<long long>(kpool));
  const int vis = (vis_ll > 0LL) ? static_cast<int>(vis_ll) : 0;
  int* dst = out_idx + static_cast<size_t>(q) * static_cast<size_t>(kmax);

  // Dense shortcut: all visible groups + tail fit in kmax -> [0..p] padded with -1
  if (vis <= n_top) {
    for (int j = tid; j < kmax; j += 256) {
      dst[j] = (static_cast<long long>(j) <= p) ? j : -1;
    }
    return;
  }

  const long long tail_start = static_cast<long long>(vis) * static_cast<long long>(kpool);
  const int tail_len = (p + 1LL > tail_start) ? static_cast<int>(p + 1LL - tail_start) : 0;

  if (n_top == 0) {
    for (int j = tid; j < kmax; j += 256) {
      dst[j] = (j < tail_len) ? static_cast<int>(tail_start + j) : -1;
    }
    return;
  }

  __shared__ unsigned int hist[2048];
  __shared__ unsigned int s_warp[8];
  __shared__ unsigned int s_res[4];

  const float* srow = scores + static_cast<size_t>(ql) * static_cast<size_t>(score_stride);

  // Pass 0: top 11 bits [31..21]
#pragma unroll
  for (int i = 0; i < 8; ++i) hist[tid + i * 256] = 0u;
  __syncthreads();
  for (int g = tid; g < vis; g += 256) {
    const unsigned int key = hy4_f2ui(srow[g]);
    atomicAdd(&hist[key >> 21], 1u);
  }
  __syncthreads();
  unsigned int bkt0 = 0u, gt0 = 0u;
  hy4_hist_find_thr<2048>(hist, static_cast<unsigned int>(n_top), s_warp, s_res, tid, bkt0, gt0);
  const unsigned int need1 = static_cast<unsigned int>(n_top) - gt0;
  const unsigned int pref0 = bkt0 << 21;

  // Pass 1: middle 11 bits [20..10]
#pragma unroll
  for (int i = 0; i < 8; ++i) hist[tid + i * 256] = 0u;
  __syncthreads();
  for (int g = tid; g < vis; g += 256) {
    const unsigned int key = hy4_f2ui(srow[g]);
    if ((key & 0xFFE00000u) == pref0) {
      atomicAdd(&hist[(key >> 10) & 2047u], 1u);
    }
  }
  __syncthreads();
  unsigned int bkt1 = 0u, gt1 = 0u;
  hy4_hist_find_thr<2048>(hist, need1, s_warp, s_res, tid, bkt1, gt1);
  const unsigned int need2 = need1 - gt1;
  const unsigned int pref1 = pref0 | (bkt1 << 10);

  // Pass 2: bottom 10 bits [9..0]
#pragma unroll
  for (int i = 0; i < 4; ++i) hist[tid + i * 256] = 0u;
  __syncthreads();
  for (int g = tid; g < vis; g += 256) {
    const unsigned int key = hy4_f2ui(srow[g]);
    if ((key & 0xFFFFFC00u) == pref1) {
      atomicAdd(&hist[key & 1023u], 1u);
    }
  }
  __syncthreads();
  unsigned int bkt2 = 0u, gt2 = 0u;
  hy4_hist_find_thr<1024>(hist, need2, s_warp, s_res, tid, bkt2, gt2);
  const unsigned int need_ties = need2 - gt2;
  const unsigned int thr = pref1 | bkt2;

  // Collection pass: visits g = 0..vis-1 in strictly ascending order!
  // Keeps all key > thr plus first need_ties with key == thr -> exact tie-break
  // to lowest g AND already sorted ascending without any sort pass!
  unsigned int seen_eq = 0u;
  unsigned int out_cnt = 0u;
  for (int base = 0; base < vis; base += 256) {
    const int g = base + tid;
    unsigned int is_gt = 0u;
    unsigned int is_eq = 0u;
    if (g < vis) {
      const unsigned int key = hy4_f2ui(srow[g]);
      is_gt = (key > thr) ? 1u : 0u;
      is_eq = (key == thr) ? 1u : 0u;
    }

    unsigned int eq_tot = 0u;
    const unsigned int eq_ex = hy4_block_excl_scan_256(is_eq, s_warp, tid, eq_tot);
    const unsigned int eq_rank = seen_eq + eq_ex;
    seen_eq += eq_tot;

    const unsigned int keep = (is_gt != 0u || (is_eq != 0u && eq_rank < need_ties)) ? 1u : 0u;
    unsigned int keep_tot = 0u;
    const unsigned int keep_ex = hy4_block_excl_scan_256(keep, s_warp, tid, keep_tot);

    if (keep != 0u) {
      const unsigned int pos = out_cnt + keep_ex;
      if (pos < static_cast<unsigned int>(n_top)) {
        const int cell_base = g * kpool;
        const int dst_base = static_cast<int>(pos) * kpool;
        for (int s = 0; s < kpool; ++s) {
          dst[dst_base + s] = cell_base + s;
        }
      }
    }
    out_cnt += keep_tot;
    if (out_cnt >= static_cast<unsigned int>(n_top)) {
      break;
    }
  }

  // Write own-group tail [kp*vis .. p] + pad remaining slots to kmax with -1
  const int n_sel_cells = n_top * kpool;
  const int total_valid = n_sel_cells + tail_len;
  for (int j = n_sel_cells + tid; j < kmax; j += 256) {
    dst[j] = (j < total_valid) ? static_cast<int>(tail_start + (j - n_sel_cells)) : -1;
  }
}
)cuda"
