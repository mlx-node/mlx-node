R"cuda(
#include <cuda_bf16.h>

// ============================================================================
// Qwen3.8-Flash-Next (qwen4_exp) Sparse QSA Indexer + Zero-Gather Sparse SDPA
// Geometry: Hq=24, Hkv=2, GQA=12, D=256, Indexer IH=4, ID=128, BLOCK_SIZE=4,
//           TOPK=512 (budget=2048 tokens + <=3 causal tail / <=16 tree tokens).
// ============================================================================

__device__ __forceinline__ bool qsa_better(float va, int ia, float vb, int ib) {
  return (va > vb) || (va == vb && ia < ib);
}

template <bool DESC>
__device__ __forceinline__ void qsa_bitonic_sort_512(
    float* vals, int* idxs, int tid) {
#pragma unroll
  for (int k = 2; k <= 512; k <<= 1) {
#pragma unroll
    for (int j = k >> 1; j > 0; j >>= 1) {
      const int lo = tid & (j - 1);
      const int p = ((tid - lo) << 1) | lo;
      const int q = p + j;
      const bool dir_desc = ((p & k) == 0) ? DESC : !DESC;
      const float vp = vals[p];
      const float vq = vals[q];
      const int ip = idxs[p];
      const int iq = idxs[q];
      const bool p_better = qsa_better(vp, ip, vq, iq);
      if (dir_desc ? !p_better : p_better) {
        vals[p] = vq;
        vals[q] = vp;
        idxs[p] = iq;
        idxs[q] = ip;
      }
      __syncthreads();
    }
  }
}

__device__ __forceinline__ void qsa_bitonic_merge_top512(
    float* top_v, int* top_i, float* cand_v, int* cand_i, int cnt, int tid) {
  for (int i = tid; i < 512; i += 256) {
    if (i >= cnt) {
      cand_v[i] = -3.4028234663852886e38f;
      cand_i[i] = 0x7fffffff;
    }
  }
  __syncthreads();
  qsa_bitonic_sort_512<false>(cand_v, cand_i, tid);
  for (int i = tid; i < 512; i += 256) {
    if (qsa_better(cand_v[i], cand_i[i], top_v[i], top_i[i])) {
      top_v[i] = cand_v[i];
      top_i[i] = cand_i[i];
    }
  }
  __syncthreads();
#pragma unroll
  for (int j = 256; j > 0; j >>= 1) {
    const int lo = tid & (j - 1);
    const int p = ((tid - lo) << 1) | lo;
    const int q = p + j;
    const float vp = top_v[p];
    const float vq = top_v[q];
    const int ip = top_i[p];
    const int iq = top_i[q];
    if (!qsa_better(vp, ip, vq, iq)) {
      top_v[p] = vq;
      top_v[q] = vp;
      top_i[p] = iq;
      top_i[q] = ip;
    }
    __syncthreads();
  }
}

__device__ __forceinline__ void qsa_bitonic_sort_idx_asc_512(int* idxs, int tid) {
#pragma unroll
  for (int k = 2; k <= 512; k <<= 1) {
#pragma unroll
    for (int j = k >> 1; j > 0; j >>= 1) {
      const int lo = tid & (j - 1);
      const int p = ((tid - lo) << 1) | lo;
      const int q = p + j;
      const bool asc = ((p & k) == 0);
      const int ip = idxs[p];
      const int iq = idxs[q];
      if (asc ? (ip > iq) : (ip < iq)) {
        idxs[p] = iq;
        idxs[q] = ip;
      }
      __syncthreads();
    }
  }
}

// ----------------------------------------------------------------------------
// Kernel 1A: qwen4_qsa_index_score_cuda (multi-CTA decode/verify scorer, Q<=16)
// Reads pooled_k[num_blocks, 128] ONCE across all Q_len queries simultaneously!
//   iq:       [IH=4, Q_len, ID=128] bf16 (row-major [1, 4, Q_len, 128])
//   pooled_k: [num_blocks, ID=128] bf16
//   scores:   [Q_len, num_blocks] f32
// Grid: dim3((num_blocks + 31) / 32, 1, 1), Block: dim3(256, 1, 1)
//   (16 half-warps = 16 blocks/step, 2 steps = 32 blocks/CTA -> 128 CTAs @ K=16K)
// ----------------------------------------------------------------------------
__global__ __launch_bounds__(256, 2) void qwen4_qsa_index_score_cuda(
    const nv_bfloat16* __restrict__ iq,
    const nv_bfloat16* __restrict__ pooled_k,
    float* __restrict__ scores,
    int Q_len,
    int num_blocks) {
  constexpr int ID = 128;
  constexpr float INV_SQRT_ID = 0.08838834764831845f; // rsqrt(128)
  const int tid = threadIdx.x;
  const int hw = tid >> 4;     // 0..15
  const int lane16 = tid & 15; // 0..15 -> 8 dims each = 128 dims

  // Stage iq[4, Q_len, 128] into SMEM (<= 16 KB for Q_len <= 16)
  __shared__ __align__(16) nv_bfloat16 sQ[4 * 16 * ID];
  const int total_u4 = (4 * Q_len * ID) >> 3;
  for (int idx = tid; idx < total_u4; idx += 256) {
    reinterpret_cast<uint4*>(sQ)[idx] = reinterpret_cast<const uint4*>(iq)[idx];
  }
  __syncthreads();

  const int b_base = static_cast<int>(blockIdx.x) * 32;
  const unsigned int hw_mask = (hw & 1) ? 0xffff0000u : 0x0000ffffu;
#pragma unroll
  for (int step = 0; step < 2; ++step) {
    const int b = b_base + step * 16 + hw;
    if (b >= num_blocks) {
      continue;
    }
    const uint4 kv = reinterpret_cast<const uint4*>(pooled_k + (size_t)b * ID)[lane16];
    const __nv_bfloat162* kp = reinterpret_cast<const __nv_bfloat162*>(&kv);
    float kf[8];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float2 f = __bfloat1622float2(kp[j]);
      kf[2 * j] = f.x;
      kf[2 * j + 1] = f.y;
    }
    for (int q = 0; q < Q_len; ++q) {
      float sum_relu = 0.0f;
#pragma unroll
      for (int h = 0; h < 4; ++h) {
        const uint4 qv = reinterpret_cast<const uint4*>(
            sQ + ((size_t)h * Q_len + q) * ID)[lane16];
        const __nv_bfloat162* qp = reinterpret_cast<const __nv_bfloat162*>(&qv);
        float dot = 0.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const float2 f = __bfloat1622float2(qp[j]);
          dot = fmaf(f.x, kf[2 * j], fmaf(f.y, kf[2 * j + 1], dot));
        }
        dot += __shfl_xor_sync(hw_mask, dot, 8, 16);
        dot += __shfl_xor_sync(hw_mask, dot, 4, 16);
        dot += __shfl_xor_sync(hw_mask, dot, 2, 16);
        dot += __shfl_xor_sync(hw_mask, dot, 1, 16);
        sum_relu += fmaxf(dot, 0.0f);
      }
      if (lane16 == 0) {
        scores[(size_t)q * num_blocks + b] = sum_relu * INV_SQRT_ID;
      }
    }
  }
}

// ----------------------------------------------------------------------------
// Kernel 1B: qwen4_qsa_index_topk512_cuda (streaming threshold-pruned Top-512)
// Reads L2-resident scores[Q_len, num_blocks], selects top-min(512, complete)
// causal blocks per query, sorts block_ids ascending (-1 for unused slots).
// If tree_mode != 0: complete = cache_len / 4 for all queries.
// Else:              complete = (base_pos + q + 1) / 4 (clamped <= num_blocks).
// Grid: dim3(Q_len, 1, 1), Block: dim3(256, 1, 1)
// ----------------------------------------------------------------------------
__global__ __launch_bounds__(256, 2) void qwen4_qsa_index_topk512_cuda(
    const float* __restrict__ scores,
    int* __restrict__ block_ids,
    int* __restrict__ num_selected,
    int num_blocks,
    int base_pos,
    int tree_mode) {
  const int q = static_cast<int>(blockIdx.x);
  const int tid = threadIdx.x;
  int complete = tree_mode ? (base_pos >> 2) : ((base_pos + q + 1) >> 2);
  if (complete > num_blocks) complete = num_blocks;
  if (complete < 0) complete = 0;

  int* out_ids = block_ids + (size_t)q * 512;
  if (complete <= 512) {
    for (int i = tid; i < 512; i += 256) {
      out_ids[i] = (i < complete) ? i : -1;
    }
    if (tid == 0) num_selected[q] = complete;
    return;
  }

  __shared__ float top_v[512];
  __shared__ int top_i[512];
  __shared__ float cand_v[512];
  __shared__ int cand_i[512];
  __shared__ int cand_cnt;

  const float* srow = scores + (size_t)q * num_blocks;
  for (int i = tid; i < 512; i += 256) {
    top_v[i] = srow[i];
    top_i[i] = i;
  }
  if (tid == 0) cand_cnt = 0;
  __syncthreads();
  qsa_bitonic_sort_512<true>(top_v, top_i, tid);

  float thresh_v = top_v[511];
  int thresh_i = top_i[511];

  for (int b0 = 512; b0 < complete; b0 += 256) {
    const int b = b0 + tid;
    if (b < complete) {
      const float sv = srow[b];
      if (qsa_better(sv, b, thresh_v, thresh_i)) {
        const int pos = atomicAdd(&cand_cnt, 1);
        cand_v[pos] = sv;
        cand_i[pos] = b;
      }
    }
    __syncthreads();
    const int cur_cnt = cand_cnt;
    if (cur_cnt > 256) {
      qsa_bitonic_merge_top512(top_v, top_i, cand_v, cand_i, cur_cnt, tid);
      if (tid == 0) cand_cnt = 0;
      __syncthreads();
      thresh_v = top_v[511];
      thresh_i = top_i[511];
    }
  }
  const int rem_cnt = cand_cnt;
  if (rem_cnt > 0) {
    qsa_bitonic_merge_top512(top_v, top_i, cand_v, cand_i, rem_cnt, tid);
  }
  qsa_bitonic_sort_idx_asc_512(top_i, tid);
  for (int i = tid; i < 512; i += 256) {
    out_ids[i] = top_i[i];
  }
  if (tid == 0) num_selected[q] = 512;
}

// ----------------------------------------------------------------------------
// Kernel 1C: qwen4_qsa_index_select_fused_cuda (single-pass prefill Q>=16)
// Zero global score scratch buffer! Each CTA handles 1 query row q: holds
// iq[0..3, q, 0..127] in 32 regs/thread (16 half-warps = 16 blocks/iter),
// streams pooled_k with threshold-pruned Top-512 in SMEM.
// Grid: dim3(Q_len, 1, 1), Block: dim3(256, 1, 1)
// ----------------------------------------------------------------------------
__global__ __launch_bounds__(256, 2) void qwen4_qsa_index_select_fused_cuda(
    const nv_bfloat16* __restrict__ iq,
    const nv_bfloat16* __restrict__ pooled_k,
    int* __restrict__ block_ids,
    int* __restrict__ num_selected,
    int Q_len,
    int num_blocks,
    int base_pos,
    int tree_mode) {
  constexpr int ID = 128;
  constexpr float INV_SQRT_ID = 0.08838834764831845f;
  const int q = static_cast<int>(blockIdx.x);
  const int tid = threadIdx.x;
  const int hw = tid >> 4;
  const int lane16 = tid & 15;

  int complete = tree_mode ? (base_pos >> 2) : ((base_pos + q + 1) >> 2);
  if (complete > num_blocks) complete = num_blocks;
  if (complete < 0) complete = 0;

  int* out_ids = block_ids + (size_t)q * 512;
  if (complete <= 512) {
    for (int i = tid; i < 512; i += 256) {
      out_ids[i] = (i < complete) ? i : -1;
    }
    if (tid == 0) num_selected[q] = complete;
    return;
  }

  float qf[4][8];
#pragma unroll
  for (int h = 0; h < 4; ++h) {
    const uint4 qv = reinterpret_cast<const uint4*>(
        iq + ((size_t)h * Q_len + q) * ID)[lane16];
    const __nv_bfloat162* qp = reinterpret_cast<const __nv_bfloat162*>(&qv);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float2 f = __bfloat1622float2(qp[j]);
      qf[h][2 * j] = f.x;
      qf[h][2 * j + 1] = f.y;
    }
  }

  __shared__ float top_v[512];
  __shared__ int top_i[512];
  __shared__ float cand_v[512];
  __shared__ int cand_i[512];
  __shared__ int cand_cnt;

  const unsigned int hw_mask = (hw & 1) ? 0xffff0000u : 0x0000ffffu;
  auto score_block = [&](int b) -> float {
    const uint4 kv = reinterpret_cast<const uint4*>(pooled_k + (size_t)b * ID)[lane16];
    const __nv_bfloat162* kp = reinterpret_cast<const __nv_bfloat162*>(&kv);
    float kf[8];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float2 f = __bfloat1622float2(kp[j]);
      kf[2 * j] = f.x;
      kf[2 * j + 1] = f.y;
    }
    float sum_relu = 0.0f;
#pragma unroll
    for (int h = 0; h < 4; ++h) {
      float dot = 0.0f;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        dot = fmaf(qf[h][j], kf[j], dot);
      }
      dot += __shfl_xor_sync(hw_mask, dot, 8, 16);
      dot += __shfl_xor_sync(hw_mask, dot, 4, 16);
      dot += __shfl_xor_sync(hw_mask, dot, 2, 16);
      dot += __shfl_xor_sync(hw_mask, dot, 1, 16);
      sum_relu += fmaxf(dot, 0.0f);
    }
    return sum_relu * INV_SQRT_ID;
  };

  for (int b0 = 0; b0 < 512; b0 += 16) {
    const int b = b0 + hw;
    const float sc = score_block(b);
    if (lane16 == 0) {
      top_v[b] = sc;
      top_i[b] = b;
    }
  }
  if (tid == 0) cand_cnt = 0;
  __syncthreads();
  qsa_bitonic_sort_512<true>(top_v, top_i, tid);

  float thresh_v = top_v[511];
  int thresh_i = top_i[511];

  for (int b0 = 512; b0 < complete; b0 += 256) {
    for (int sub = 0; sub < 256; sub += 16) {
      const int b = b0 + sub + hw;
      if (b < complete) {
        const float sc = score_block(b);
        if (lane16 == 0 && qsa_better(sc, b, thresh_v, thresh_i)) {
          const int pos = atomicAdd(&cand_cnt, 1);
          cand_v[pos] = sc;
          cand_i[pos] = b;
        }
      }
    }
    __syncthreads();
    const int cur_cnt = cand_cnt;
    if (cur_cnt > 256) {
      qsa_bitonic_merge_top512(top_v, top_i, cand_v, cand_i, cur_cnt, tid);
      if (tid == 0) cand_cnt = 0;
      __syncthreads();
      thresh_v = top_v[511];
      thresh_i = top_i[511];
    }
  }
  const int rem_cnt = cand_cnt;
  if (rem_cnt > 0) {
    qsa_bitonic_merge_top512(top_v, top_i, cand_v, cand_i, rem_cnt, tid);
  }
  qsa_bitonic_sort_idx_asc_512(top_i, tid);
  for (int i = tid; i < 512; i += 256) {
    out_ids[i] = top_i[i];
  }
  if (tid == 0) num_selected[q] = 512;
}
)cuda"
R"cuda(
// ----------------------------------------------------------------------------
// Kernel 2: qwen4_qsa_sparse_sdpa_cuda (Zero-KV-Gather Tensor Core Sparse SDPA)
// 4-warp D-split CTA (128 threads, warp w owns D-slice [64w .. 64w+63] of D=256),
// processing ALL GQA=12 Q-heads of 1 (q, hkv, split) in a single m16n8k16 M-tile.
// - Q[12, 64] held 100% in registers (16 u32/thread), accO = 32 FP32 regs/thread.
// - sK[32,256] + sV[32,256] loaded directly from sparse block_ids[q, slot]*4+r
//   via cp.async.cg + 128B XOR swizzle (zero KV gather buffer!).
// - Unified extra tail tile covers causal incomplete tail (0..3 tokens) AND
//   optional tree-verify nodes (0..3 committed tail + 1..16 tree tokens <= 19).
// - SPLIT_K=false: direct BF16 write to out[B, Hq, Q_len, 256] (prefill / large Q).
// - SPLIT_K=true:  writes partial_O/m/l for qwen4_qsa_sparse_reduce_cuda (decode).
// - MASK_MODE=0: causal (query_end = base_pos + q + 1).
// - MASK_MODE=1: tree verify (committed cache_len = base_pos, tree_mask[Q_len]).
// Grid: dim3(num_splits, Q_len, B * Hkv), Block: dim3(32, 4, 1)
// ----------------------------------------------------------------------------
__device__ __forceinline__ float qsa_exp2f(float x) {
  float r;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}

__device__ __forceinline__ void qsa_mma_m16n8k16_bf16(
    float& d0, float& d1, float& d2, float& d3,
    unsigned int a0, unsigned int a1, unsigned int a2, unsigned int a3,
    unsigned int b0, unsigned int b1,
    float c0, float c1, float c2, float c3) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
        "f"(c0), "f"(c1), "f"(c2), "f"(c3));
}

__device__ __forceinline__ void qsa_ldmatrix4(
    unsigned int& r0, unsigned int& r1, unsigned int& r2, unsigned int& r3,
    const void* smem_ptr) {
  unsigned int addr = static_cast<unsigned int>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
      : "r"(addr));
}

__device__ __forceinline__ void qsa_ldmatrix4_trans(
    unsigned int& r0, unsigned int& r1, unsigned int& r2, unsigned int& r3,
    const void* smem_ptr) {
  unsigned int addr = static_cast<unsigned int>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
      : "r"(addr));
}

__device__ __forceinline__ void qsa_cp16_zfill(
    void* smem_ptr, const void* glob_ptr, bool pred) {
  unsigned int addr = static_cast<unsigned int>(__cvta_generic_to_shared(smem_ptr));
  int sz = pred ? 16 : 0;
  asm volatile(
      "cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
      :: "r"(addr), "l"(glob_ptr), "r"(sz) : "memory");
}

__device__ __forceinline__ int qsa_swiz256(int row, int c8) {
  return row * 256 + ((c8 ^ (row & 7)) << 3);
}

__device__ __forceinline__ unsigned int qsa_pack_bf16x2(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<const unsigned int*>(&v);
}

__device__ __forceinline__ void qsa_load_sparse_kv256(
    nv_bfloat16* dst, const nv_bfloat16* base, long long stride,
    const int* s_kpos, int tid) {
#pragma unroll
  for (int idx = tid; idx < 1024; idx += 128) {
    const int r = idx >> 5;   // 0..31
    const int c8 = idx & 31;  // 0..31 (256 / 8)
    const int kpos = s_kpos[r];
    const bool ok = (kpos >= 0);
    const nv_bfloat16* gp = ok ? (base + (long long)kpos * stride + c8 * 8) : base;
    qsa_cp16_zfill(&dst[qsa_swiz256(r, c8)], gp, ok);
  }
  asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template <bool SPLIT_K, int MASK_MODE>
__global__ __launch_bounds__(128, 2) void qwen4_qsa_sparse_sdpa_cuda(
    const nv_bfloat16* __restrict__ Q,       // [B, Hq, Q_len, 256]
    const nv_bfloat16* __restrict__ K,       // [B, Hkv, K_len, 256] (strided ok)
    const nv_bfloat16* __restrict__ V,       // [B, Hkv, K_len, 256] (strided ok)
    const int* __restrict__ block_ids,       // [B, Q_len, topk]
    const int* __restrict__ num_selected,    // [B, Q_len]
    const unsigned int* __restrict__ tree_mask, // [Q_len] when MASK_MODE==1
    nv_bfloat16* __restrict__ out_or_po,     // [B, Hq, Q_len, 256] or [B, Hkv, S, Q_len, GQA, 256]
    float* __restrict__ partial_m,           // [B, Hkv, S, Q_len, GQA] when SPLIT_K
    float* __restrict__ partial_l,           // [B, Hkv, S, Q_len, GQA] when SPLIT_K
    int Hkv, int gqa, int Q_len, int K_len, int topk, int base_pos,
    long long k_sb, long long k_sh, long long k_ss,
    long long v_sb, long long v_sh, long long v_ss,
    float scale_log2, int num_splits) {
  constexpr int D = 256;
  constexpr int BN = 32;
  const int split = static_cast<int>(blockIdx.x);
  const int q = static_cast<int>(blockIdx.y);
  const int bh = static_cast<int>(blockIdx.z);
  const int b = bh / Hkv;
  const int hkv = bh - b * Hkv;
  const int lane = static_cast<int>(threadIdx.x);
  const int warp = static_cast<int>(threadIdx.y); // 0..3 -> 64 dims each
  const int tid = warp * 32 + lane;
  const int group_id = lane >> 2;
  const int tid_in_grp = lane & 3;
  const int row0 = group_id;
  const int row1 = group_id + 8;

  const int num_sel = num_selected[(size_t)b * Q_len + q];
  const int num_blk_tiles = (num_sel + 7) >> 3;
  const int query_end = (MASK_MODE == 0)
      ? min(base_pos + q + 1, K_len)
      : min(base_pos, K_len);
  const int tail_start = (query_end >> 2) << 2;
  const int comm_tail_len = query_end - tail_start;
  const int has_tail = (MASK_MODE == 1) ? 1 : (comm_tail_len > 0 ? 1 : 0);
  const int total_tiles = num_blk_tiles + has_tail;
  const int tiles_per_split = (total_tiles + num_splits - 1) / num_splits;
  const int tile_start = split * tiles_per_split;
  const int tile_end = min(tile_start + tiles_per_split, total_tiles);

  const unsigned int tmask = (MASK_MODE == 1) ? tree_mask[q] : 0u;
  const int* blk_row = block_ids + ((size_t)b * Q_len + q) * topk;

  // Static SMEM: 16384 + 16384 + 8192 + 256 = 41216 B (<= 48 KB -> 2 CTAs/SM)
  __shared__ __align__(128) nv_bfloat16 sK[BN * D];
  __shared__ __align__(128) nv_bfloat16 sV[BN * D];
  __shared__ __align__(16) float sS[4][16][32];
  __shared__ int s_kpos[2][32];

  // Load warp's 64-dim slice of Q[0..gqa-1, q, 64*warp .. 64*warp+63] into regs
  const long long q_head_stride = (long long)Q_len * D;
  const nv_bfloat16* Q_base = Q + ((long long)b * Hkv + hkv) * gqa * q_head_stride + (long long)q * D;
  const nv_bfloat16* q_ptr0 = (row0 < gqa) ? (Q_base + (long long)row0 * q_head_stride + warp * 64) : nullptr;
  const nv_bfloat16* q_ptr1 = (row1 < gqa) ? (Q_base + (long long)row1 * q_head_stride + warp * 64) : nullptr;
  unsigned int q_reg[4][4];
#pragma unroll
  for (int kt = 0; kt < 4; ++kt) {
    const int k = kt * 16;
    q_reg[kt][0] = q_ptr0 ? __ldg(reinterpret_cast<const unsigned int*>(q_ptr0 + k + 2 * tid_in_grp)) : 0u;
    q_reg[kt][2] = q_ptr0 ? __ldg(reinterpret_cast<const unsigned int*>(q_ptr0 + k + 8 + 2 * tid_in_grp)) : 0u;
    q_reg[kt][1] = q_ptr1 ? __ldg(reinterpret_cast<const unsigned int*>(q_ptr1 + k + 2 * tid_in_grp)) : 0u;
    q_reg[kt][3] = q_ptr1 ? __ldg(reinterpret_cast<const unsigned int*>(q_ptr1 + k + 8 + 2 * tid_in_grp)) : 0u;
  }

  float accO[8][4];
#pragma unroll
  for (int t = 0; t < 8; ++t) {
    accO[t][0] = 0.0f;
    accO[t][1] = 0.0f;
    accO[t][2] = 0.0f;
    accO[t][3] = 0.0f;
  }
  float m_row0 = -1e30f;
  float m_row1 = -1e30f;
  float l_row0 = 0.0f;
  float l_row1 = 0.0f;

  const nv_bfloat16* K_base = K + (long long)b * k_sb + (long long)hkv * k_sh;
  const nv_bfloat16* V_base = V + (long long)b * v_sb + (long long)hkv * v_sh;

  auto fill_kpos = [&](int tile_idx, int buf) {
    if (tid < 32) {
      int kpos = -1;
      if (tile_idx < num_blk_tiles) {
        const int slot = tile_idx * 8 + (tid >> 2);
        if (slot < num_sel) {
          const int blk = blk_row[slot];
          if (blk >= 0) {
            const int kp = blk * 4 + (tid & 3);
            if (kp < query_end && kp < K_len) {
              kpos = kp;
            }
          }
        }
      } else {
        if (tid < comm_tail_len) {
          const int kp = tail_start + tid;
          if (kp < K_len) kpos = kp;
        } else if constexpr (MASK_MODE == 1) {
          const int j = tid - comm_tail_len;
          if (j < Q_len && ((tmask >> j) & 1u)) {
            const int kp = base_pos + j;
            if (kp < K_len) kpos = kp;
          }
        }
      }
      s_kpos[buf][tid] = kpos;
    }
  };

  const int k_ld_r0 = ((lane >> 4) * 8) + (lane & 7);
  const int k_ld_c8_base = warp * 8 + ((lane >> 3) & 1);
  const int v_ld_r0 = lane & 15;
  const int v_ld_c8_base = warp * 8 + ((lane >> 4) & 1);

  if (tile_start < tile_end) {
    fill_kpos(tile_start, tile_start & 1);
    __syncthreads();
    qsa_load_sparse_kv256(sK, K_base, k_ss, s_kpos[tile_start & 1], tid);
  }
)cuda"
R"cuda(
  for (int tile = tile_start; tile < tile_end; ++tile) {
    const int cur = tile & 1;
    const int nxt = cur ^ 1;
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    __syncthreads();

    // Launch sV fetch overlapped with QK^T, and prepare next tile's s_kpos
    qsa_load_sparse_kv256(sV, V_base, v_ss, s_kpos[cur], tid);
    if (tile + 1 < tile_end) {
      fill_kpos(tile + 1, nxt);
    }

    float s_acc[4][4] = {};
#pragma unroll
    for (int kt = 0; kt < 4; ++kt) {
      const int c8 = k_ld_c8_base + kt * 2;
      unsigned int b0, b1, b2, b3;
      qsa_ldmatrix4(b0, b1, b2, b3, &sK[qsa_swiz256(k_ld_r0, c8)]);
      qsa_mma_m16n8k16_bf16(
          s_acc[0][0], s_acc[0][1], s_acc[0][2], s_acc[0][3],
          q_reg[kt][0], q_reg[kt][1], q_reg[kt][2], q_reg[kt][3],
          b0, b1, s_acc[0][0], s_acc[0][1], s_acc[0][2], s_acc[0][3]);
      qsa_mma_m16n8k16_bf16(
          s_acc[1][0], s_acc[1][1], s_acc[1][2], s_acc[1][3],
          q_reg[kt][0], q_reg[kt][1], q_reg[kt][2], q_reg[kt][3],
          b2, b3, s_acc[1][0], s_acc[1][1], s_acc[1][2], s_acc[1][3]);
      qsa_ldmatrix4(b0, b1, b2, b3, &sK[qsa_swiz256(k_ld_r0 + 16, c8)]);
      qsa_mma_m16n8k16_bf16(
          s_acc[2][0], s_acc[2][1], s_acc[2][2], s_acc[2][3],
          q_reg[kt][0], q_reg[kt][1], q_reg[kt][2], q_reg[kt][3],
          b0, b1, s_acc[2][0], s_acc[2][1], s_acc[2][2], s_acc[2][3]);
      qsa_mma_m16n8k16_bf16(
          s_acc[3][0], s_acc[3][1], s_acc[3][2], s_acc[3][3],
          q_reg[kt][0], q_reg[kt][1], q_reg[kt][2], q_reg[kt][3],
          b2, b3, s_acc[3][0], s_acc[3][1], s_acc[3][2], s_acc[3][3]);
    }
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      const int c0 = nt * 8 + 2 * tid_in_grp;
      sS[warp][row0][c0] = s_acc[nt][0];
      sS[warp][row0][c0 + 1] = s_acc[nt][1];
      sS[warp][row1][c0] = s_acc[nt][2];
      sS[warp][row1][c0 + 1] = s_acc[nt][3];
    }

    // Wait for sV cp.async AND publish sS[0..3] + s_kpos[nxt]
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    __syncthreads();

    // Launch next tile's sK fetch overlapped with softmax + PV
    if (tile + 1 < tile_end) {
      qsa_load_sparse_kv256(sK, K_base, k_ss, s_kpos[nxt], tid);
    }

    // Reduce partial QK^T across 4 D-split warps + mask + online softmax
    float local_max0 = -1e30f;
    float local_max1 = -1e30f;
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      const int c0 = nt * 8 + 2 * tid_in_grp;
      const int c1 = c0 + 1;
      const bool v0 = (s_kpos[cur][c0] >= 0);
      const bool v1 = (s_kpos[cur][c1] >= 0);
      const float r00 = (sS[0][row0][c0] + sS[1][row0][c0]) + (sS[2][row0][c0] + sS[3][row0][c0]);
      const float r01 = (sS[0][row0][c1] + sS[1][row0][c1]) + (sS[2][row0][c1] + sS[3][row0][c1]);
      const float r10 = (sS[0][row1][c0] + sS[1][row1][c0]) + (sS[2][row1][c0] + sS[3][row1][c0]);
      const float r11 = (sS[0][row1][c1] + sS[1][row1][c1]) + (sS[2][row1][c1] + sS[3][row1][c1]);
      s_acc[nt][0] = v0 ? (r00 * scale_log2) : -1e30f;
      s_acc[nt][1] = v1 ? (r01 * scale_log2) : -1e30f;
      s_acc[nt][2] = v0 ? (r10 * scale_log2) : -1e30f;
      s_acc[nt][3] = v1 ? (r11 * scale_log2) : -1e30f;
      local_max0 = fmaxf(local_max0, fmaxf(s_acc[nt][0], s_acc[nt][1]));
      local_max1 = fmaxf(local_max1, fmaxf(s_acc[nt][2], s_acc[nt][3]));
    }
    local_max0 = fmaxf(local_max0, __shfl_xor_sync(0xffffffffu, local_max0, 1));
    local_max0 = fmaxf(local_max0, __shfl_xor_sync(0xffffffffu, local_max0, 2));
    local_max1 = fmaxf(local_max1, __shfl_xor_sync(0xffffffffu, local_max1, 1));
    local_max1 = fmaxf(local_max1, __shfl_xor_sync(0xffffffffu, local_max1, 2));

    const float new_m0 = fmaxf(m_row0, local_max0);
    const float new_m1 = fmaxf(m_row1, local_max1);
    const float alpha0 = (m_row0 > -1e20f) ? qsa_exp2f(m_row0 - new_m0) : 0.0f;
    const float alpha1 = (m_row1 > -1e20f) ? qsa_exp2f(m_row1 - new_m1) : 0.0f;

    float local_sum0 = 0.0f;
    float local_sum1 = 0.0f;
    unsigned int p_frag[2][4];
#pragma unroll
    for (int ks = 0; ks < 2; ++ks) {
      const int nt0 = ks * 2;
      const int nt1 = ks * 2 + 1;
      const float p00 = (s_acc[nt0][0] > -1e20f) ? qsa_exp2f(s_acc[nt0][0] - new_m0) : 0.0f;
      const float p01 = (s_acc[nt0][1] > -1e20f) ? qsa_exp2f(s_acc[nt0][1] - new_m0) : 0.0f;
      const float p10 = (s_acc[nt0][2] > -1e20f) ? qsa_exp2f(s_acc[nt0][2] - new_m1) : 0.0f;
      const float p11 = (s_acc[nt0][3] > -1e20f) ? qsa_exp2f(s_acc[nt0][3] - new_m1) : 0.0f;
      const float p02 = (s_acc[nt1][0] > -1e20f) ? qsa_exp2f(s_acc[nt1][0] - new_m0) : 0.0f;
      const float p03 = (s_acc[nt1][1] > -1e20f) ? qsa_exp2f(s_acc[nt1][1] - new_m0) : 0.0f;
      const float p12 = (s_acc[nt1][2] > -1e20f) ? qsa_exp2f(s_acc[nt1][2] - new_m1) : 0.0f;
      const float p13 = (s_acc[nt1][3] > -1e20f) ? qsa_exp2f(s_acc[nt1][3] - new_m1) : 0.0f;
      local_sum0 += (p00 + p01) + (p02 + p03);
      local_sum1 += (p10 + p11) + (p12 + p13);
      p_frag[ks][0] = qsa_pack_bf16x2(p00, p01);
      p_frag[ks][1] = qsa_pack_bf16x2(p10, p11);
      p_frag[ks][2] = qsa_pack_bf16x2(p02, p03);
      p_frag[ks][3] = qsa_pack_bf16x2(p12, p13);
    }
    local_sum0 += __shfl_xor_sync(0xffffffffu, local_sum0, 1);
    local_sum0 += __shfl_xor_sync(0xffffffffu, local_sum0, 2);
    local_sum1 += __shfl_xor_sync(0xffffffffu, local_sum1, 1);
    local_sum1 += __shfl_xor_sync(0xffffffffu, local_sum1, 2);

#pragma unroll
    for (int t = 0; t < 8; ++t) {
      accO[t][0] *= alpha0;
      accO[t][1] *= alpha0;
      accO[t][2] *= alpha1;
      accO[t][3] *= alpha1;
    }
    l_row0 = l_row0 * alpha0 + local_sum0;
    l_row1 = l_row1 * alpha1 + local_sum1;
    m_row0 = new_m0;
    m_row1 = new_m1;

    // PV MMA on warp's 64-dim slice of sV
#pragma unroll
    for (int ks = 0; ks < 2; ++ks) {
      const int vr = ks * 16 + v_ld_r0;
#pragma unroll
      for (int dt = 0; dt < 4; ++dt) {
        const int c8 = v_ld_c8_base + dt * 2;
        unsigned int b0, b1, b2, b3;
        qsa_ldmatrix4_trans(b0, b1, b2, b3, &sV[qsa_swiz256(vr, c8)]);
        const int nt0 = dt * 2;
        const int nt1 = dt * 2 + 1;
        qsa_mma_m16n8k16_bf16(
            accO[nt0][0], accO[nt0][1], accO[nt0][2], accO[nt0][3],
            p_frag[ks][0], p_frag[ks][1], p_frag[ks][2], p_frag[ks][3],
            b0, b1, accO[nt0][0], accO[nt0][1], accO[nt0][2], accO[nt0][3]);
        qsa_mma_m16n8k16_bf16(
            accO[nt1][0], accO[nt1][1], accO[nt1][2], accO[nt1][3],
            p_frag[ks][0], p_frag[ks][1], p_frag[ks][2], p_frag[ks][3],
            b2, b3, accO[nt1][0], accO[nt1][1], accO[nt1][2], accO[nt1][3]);
      }
    }
  }

  const float inv0 = (l_row0 > 0.0f) ? (1.0f / l_row0) : 0.0f;
  const float inv1 = (l_row1 > 0.0f) ? (1.0f / l_row1) : 0.0f;
  if constexpr (SPLIT_K) {
    const long long base_split =
        ((((long long)b * Hkv + hkv) * num_splits + split) * Q_len + q) * gqa;
    if (row0 < gqa) {
      if (warp == 0 && tid_in_grp == 0) {
        partial_m[base_split + row0] = m_row0;
        partial_l[base_split + row0] = l_row0;
      }
      nv_bfloat16* dst0 = out_or_po + (base_split + row0) * D + warp * 64;
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) {
        *reinterpret_cast<unsigned int*>(dst0 + nt * 8 + 2 * tid_in_grp) =
            qsa_pack_bf16x2(accO[nt][0] * inv0, accO[nt][1] * inv0);
      }
    }
    if (row1 < gqa) {
      if (warp == 0 && tid_in_grp == 0) {
        partial_m[base_split + row1] = m_row1;
        partial_l[base_split + row1] = l_row1;
      }
      nv_bfloat16* dst1 = out_or_po + (base_split + row1) * D + warp * 64;
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) {
        *reinterpret_cast<unsigned int*>(dst1 + nt * 8 + 2 * tid_in_grp) =
            qsa_pack_bf16x2(accO[nt][2] * inv1, accO[nt][3] * inv1);
      }
    }
  } else {
    nv_bfloat16* O_base = out_or_po + ((long long)b * Hkv + hkv) * gqa * q_head_stride + (long long)q * D;
    if (row0 < gqa) {
      nv_bfloat16* dst0 = O_base + (long long)row0 * q_head_stride + warp * 64;
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) {
        *reinterpret_cast<unsigned int*>(dst0 + nt * 8 + 2 * tid_in_grp) =
            qsa_pack_bf16x2(accO[nt][0] * inv0, accO[nt][1] * inv0);
      }
    }
    if (row1 < gqa) {
      nv_bfloat16* dst1 = O_base + (long long)row1 * q_head_stride + warp * 64;
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) {
        *reinterpret_cast<unsigned int*>(dst1 + nt * 8 + 2 * tid_in_grp) =
            qsa_pack_bf16x2(accO[nt][2] * inv1, accO[nt][3] * inv1);
      }
    }
  }
}

// ----------------------------------------------------------------------------
// Kernel 3: qwen4_qsa_sparse_reduce_cuda (Split-K online-softmax reduction)
// Reduces partial_O[B, Hkv, num_splits, Q_len, gqa, 256] -> out[B, Hq, Q_len, 256]
// Grid: dim3(Q_len * gqa, Hkv, B), Block: dim3(32, 1, 1) (8 dims/lane = 256)
// ----------------------------------------------------------------------------
__global__ __launch_bounds__(32, 8) void qwen4_qsa_sparse_reduce_cuda(
    const nv_bfloat16* __restrict__ partial_O,
    const float* __restrict__ partial_m,
    const float* __restrict__ partial_l,
    nv_bfloat16* __restrict__ out,
    int Hkv, int gqa, int Q_len, int num_splits) {
  constexpr int D = 256;
  const int mq = static_cast<int>(blockIdx.x); // q * gqa + gh
  const int q = mq / gqa;
  const int gh = mq - q * gqa;
  const int hkv = static_cast<int>(blockIdx.y);
  const int b = static_cast<int>(blockIdx.z);
  const int lane = static_cast<int>(threadIdx.x);
  const int M = Q_len * gqa;
  const long long bh_base = ((long long)b * Hkv + hkv) * num_splits;

  float max_m = -1e30f;
  for (int s = 0; s < num_splits; ++s) {
    max_m = fmaxf(max_m, partial_m[(bh_base + s) * M + mq]);
  }
  float acc[8] = {};
  float total_l = 0.0f;
  for (int s = 0; s < num_splits; ++s) {
    const long long idx = (bh_base + s) * M + mq;
    const float ls = partial_l[idx];
    if (ls > 0.0f) {
      const float w = ls * qsa_exp2f(partial_m[idx] - max_m);
      total_l += w;
      const uint4 ov = reinterpret_cast<const uint4*>(partial_O + idx * D)[lane];
      const __nv_bfloat162* op = reinterpret_cast<const __nv_bfloat162*>(&ov);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float2 f = __bfloat1622float2(op[j]);
        acc[2 * j] = fmaf(w, f.x, acc[2 * j]);
        acc[2 * j + 1] = fmaf(w, f.y, acc[2 * j + 1]);
      }
    }
  }
  const float inv = (total_l > 0.0f) ? (1.0f / total_l) : 0.0f;
  uint4 ov_out;
  __nv_bfloat162* op_out = reinterpret_cast<__nv_bfloat162*>(&ov_out);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    op_out[j] = __floats2bfloat162_rn(acc[2 * j] * inv, acc[2 * j + 1] * inv);
  }
  const long long out_off =
      (((long long)b * Hkv + hkv) * gqa + gh) * ((long long)Q_len * D) + (long long)q * D;
  reinterpret_cast<uint4*>(out + out_off)[lane] = ov_out;
}
)cuda"
R"cuda(
// ============================================================================
// Prefill Fast Path (Q > 16): BF16 Tensor-Core Indexer + Radix-Select Top-512
// + Parallel Pre-Merge + Q-Tiled (BQ=5, 60-row) & Disjoint-Fallback Sparse SDPA
// ============================================================================

__device__ __forceinline__ int qsa_swiz128(int row, int c8) {
  return row * 128 + ((c8 ^ (row & 7)) << 3);
}

__device__ __forceinline__ void qsa_cp16(void* smem_ptr, const void* glob_ptr) {
  unsigned int addr = static_cast<unsigned int>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "cp.async.cg.shared.global [%0], [%1], 16;\n"
      :: "r"(addr), "l"(glob_ptr) : "memory");
}

// ----------------------------------------------------------------------------
// Kernel 1D: qwen4_qsa_index_score_tc_cuda (BF16 Tensor-Core Prefill Scorer)
// Computes scores[q_local, b] = sum_{h=0..3} ReLU(dot(iq[h, q0+ql], pooled_k[b])) / sqrt(128)
// Tile: BM=64, BN=64, 8 warps (256 thr), double-buffered sQ (48 KB SMEM, 2 CTAs/SM, 0 spills).
// Grid: dim3((max_complete + 63) / 64, (q_cnt + 63) / 64, 1), Block: dim3(256, 1, 1)
// ----------------------------------------------------------------------------
__global__ __launch_bounds__(256, 2) void qwen4_qsa_index_score_tc_cuda(
    const nv_bfloat16* __restrict__ iq,
    const nv_bfloat16* __restrict__ pooled_k,
    float* __restrict__ scores,
    int Q_len,
    int q0,
    int q_cnt,
    int max_complete,
    int score_stride) {
  constexpr int BM = 64;
  constexpr int BN = 64;
  constexpr int ID = 128;
  constexpr float INV_SQRT_ID = 0.08838834764831845f;

  const int b_base = static_cast<int>(blockIdx.x) * BN;
  const int q_local_base = static_cast<int>(blockIdx.y) * BM;
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int warp_m = warp >> 1;
  const int warp_n = warp & 1;

  __shared__ __align__(128) nv_bfloat16 sK[BN * ID];    // 16 KB
  __shared__ __align__(128) nv_bfloat16 sQ[2][BM * ID]; // 32 KB (double-buffered)

  const int c8 = tid & 15;
  const int r0 = tid >> 4;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int r = r0 + i * 16;
    const int gb = b_base + r;
    const bool ok = (gb < max_complete);
    const nv_bfloat16* gp = ok ? (pooled_k + (size_t)gb * ID + c8 * 8) : pooled_k;
    qsa_cp16_zfill(&sK[qsa_swiz128(r, c8)], gp, ok);
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int r = r0 + i * 16;
    const int ql = q_local_base + r;
    const bool ok = (ql < q_cnt);
    const nv_bfloat16* gp = ok ? (iq + (size_t)(q0 + ql) * ID + c8 * 8) : iq;
    qsa_cp16_zfill(&sQ[0][qsa_swiz128(r, c8)], gp, ok);
  }
  asm volatile("cp.async.commit_group;\n" ::: "memory");

  float acc_sum[4][4];
#pragma unroll
  for (int nt = 0; nt < 4; ++nt) {
    acc_sum[nt][0] = 0.0f;
    acc_sum[nt][1] = 0.0f;
    acc_sum[nt][2] = 0.0f;
    acc_sum[nt][3] = 0.0f;
  }

  const int q_ld_r = warp_m * 16 + (lane & 15);
  const int q_ld_c8_base = (lane >> 4) & 1;
  const int k_ld_r0 = ((lane >> 4) * 8) + (lane & 7);
  const int k_ld_c8_base = (lane >> 3) & 1;
  const int k_n_base = warp_n * 32;

#pragma unroll
  for (int h = 0; h < 4; ++h) {
    const int cur = h & 1;
    const int nxt = cur ^ 1;
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    __syncthreads();

    if (h + 1 < 4) {
      const nv_bfloat16* iq_next = iq + (size_t)(h + 1) * Q_len * ID;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int r = r0 + i * 16;
        const int ql = q_local_base + r;
        const bool ok = (ql < q_cnt);
        const nv_bfloat16* gp = ok ? (iq_next + (size_t)(q0 + ql) * ID + c8 * 8) : iq_next;
        qsa_cp16_zfill(&sQ[nxt][qsa_swiz128(r, c8)], gp, ok);
      }
      asm volatile("cp.async.commit_group;\n" ::: "memory");
    }

    float acc_h[4][4];
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      acc_h[nt][0] = 0.0f;
      acc_h[nt][1] = 0.0f;
      acc_h[nt][2] = 0.0f;
      acc_h[nt][3] = 0.0f;
    }
#pragma unroll
    for (int kt = 0; kt < 8; ++kt) {
      unsigned int a0, a1, a2, a3;
      qsa_ldmatrix4(a0, a1, a2, a3, &sQ[cur][qsa_swiz128(q_ld_r, q_ld_c8_base + kt * 2)]);
      const int kc8 = k_ld_c8_base + kt * 2;
      unsigned int b0, b1, b2, b3;
      qsa_ldmatrix4(b0, b1, b2, b3, &sK[qsa_swiz128(k_n_base + k_ld_r0, kc8)]);
      qsa_mma_m16n8k16_bf16(
          acc_h[0][0], acc_h[0][1], acc_h[0][2], acc_h[0][3],
          a0, a1, a2, a3, b0, b1,
          acc_h[0][0], acc_h[0][1], acc_h[0][2], acc_h[0][3]);
      qsa_mma_m16n8k16_bf16(
          acc_h[1][0], acc_h[1][1], acc_h[1][2], acc_h[1][3],
          a0, a1, a2, a3, b2, b3,
          acc_h[1][0], acc_h[1][1], acc_h[1][2], acc_h[1][3]);
      qsa_ldmatrix4(b0, b1, b2, b3, &sK[qsa_swiz128(k_n_base + 16 + k_ld_r0, kc8)]);
      qsa_mma_m16n8k16_bf16(
          acc_h[2][0], acc_h[2][1], acc_h[2][2], acc_h[2][3],
          a0, a1, a2, a3, b0, b1,
          acc_h[2][0], acc_h[2][1], acc_h[2][2], acc_h[2][3]);
      qsa_mma_m16n8k16_bf16(
          acc_h[3][0], acc_h[3][1], acc_h[3][2], acc_h[3][3],
          a0, a1, a2, a3, b2, b3,
          acc_h[3][0], acc_h[3][1], acc_h[3][2], acc_h[3][3]);
    }
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      acc_sum[nt][0] += fmaxf(acc_h[nt][0], 0.0f);
      acc_sum[nt][1] += fmaxf(acc_h[nt][1], 0.0f);
      acc_sum[nt][2] += fmaxf(acc_h[nt][2], 0.0f);
      acc_sum[nt][3] += fmaxf(acc_h[nt][3], 0.0f);
    }
  }

  const int group_id = lane >> 2;
  const int tid_in_grp = lane & 3;
  const int ql0 = q_local_base + warp_m * 16 + group_id;
  const int ql1 = ql0 + 8;
  const int col_base = b_base + k_n_base + 2 * tid_in_grp;

  if (ql0 < q_cnt) {
    float* dst0 = scores + (size_t)ql0 * score_stride;
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      const int c = col_base + nt * 8;
      if (c < score_stride) {
        *reinterpret_cast<float2*>(dst0 + c) =
            make_float2(acc_sum[nt][0] * INV_SQRT_ID, acc_sum[nt][1] * INV_SQRT_ID);
      }
    }
  }
  if (ql1 < q_cnt) {
    float* dst1 = scores + (size_t)ql1 * score_stride;
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      const int c = col_base + nt * 8;
      if (c < score_stride) {
        *reinterpret_cast<float2*>(dst1 + c) =
            make_float2(acc_sum[nt][2] * INV_SQRT_ID, acc_sum[nt][3] * INV_SQRT_ID);
      }
    }
  }
}

__device__ __forceinline__ unsigned int qsa_score_to_u32(float v) {
  return (v > 0.0f) ? __float_as_uint(v) : 0u;
}

__device__ __forceinline__ unsigned int qsa_block_excl_scan_256(
    unsigned int val, unsigned int* s_warp, int tid, unsigned int& out_total) {
  const int lane = tid & 31;
  const int warp = tid >> 5;
  unsigned int x = val;
#pragma unroll
  for (int d = 1; d < 32; d <<= 1) {
    unsigned int up = __shfl_up_sync(0xffffffffu, x, d);
    if (lane >= d) x += up;
  }
  if (lane == 31) s_warp[warp] = x;
  __syncthreads();
  if (warp == 0) {
    unsigned int wv = (lane < 8) ? s_warp[lane] : 0u;
#pragma unroll
    for (int d = 1; d < 8; d <<= 1) {
      unsigned int up = __shfl_up_sync(0xffffffffu, wv, d);
      if (lane >= d) wv += up;
    }
    if (lane < 8) s_warp[lane] = wv;
  }
  __syncthreads();
  out_total = s_warp[7];
  const unsigned int res = ((warp > 0) ? s_warp[warp - 1] : 0u) + (x - val);
  // Prevent the next loop iteration from overwriting s_warp[] before slower warps read it.
  __syncthreads();
  return res;
}

template <int NBKT>
__device__ __forceinline__ void qsa_hist_find_thr(
    const unsigned int* hist, int need, unsigned int* s_warp,
    int* s_res, int tid, int& out_bkt, int& out_gt) {
  constexpr int PER_T = NBKT / 256;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  unsigned int local_sum = 0u;
  const int hi_start = NBKT - 1 - tid * PER_T;
#pragma unroll
  for (int i = 0; i < PER_T; ++i) {
    local_sum += hist[hi_start - i];
  }
  unsigned int x = local_sum;
#pragma unroll
  for (int d = 1; d < 32; d <<= 1) {
    unsigned int up = __shfl_up_sync(0xffffffffu, x, d);
    if (lane >= d) x += up;
  }
  if (lane == 31) s_warp[warp] = x;
  __syncthreads();
  if (warp == 0) {
    unsigned int wv = (lane < 8) ? s_warp[lane] : 0u;
#pragma unroll
    for (int d = 1; d < 8; d <<= 1) {
      unsigned int up = __shfl_up_sync(0xffffffffu, wv, d);
      if (lane >= d) wv += up;
    }
    if (lane < 8) s_warp[lane] = wv;
  }
  __syncthreads();
  const unsigned int excl = ((warp > 0) ? s_warp[warp - 1] : 0u) + (x - local_sum);
  const unsigned int incl = excl + local_sum;
  if (excl < static_cast<unsigned int>(need) && incl >= static_cast<unsigned int>(need)) {
    unsigned int cur = excl;
#pragma unroll
    for (int i = 0; i < PER_T; ++i) {
      const int b = hi_start - i;
      const unsigned int c = hist[b];
      if (cur + c >= static_cast<unsigned int>(need)) {
        s_res[0] = b;
        s_res[1] = static_cast<int>(cur);
        break;
      }
      cur += c;
    }
  }
  __syncthreads();
  out_bkt = s_res[0];
  out_gt = s_res[1];
}

__device__ __forceinline__ int qsa_lower_bound(const int* arr, int n, int val) {
  int lo = 0, hi = n;
  while (lo < hi) {
    const int mid = (lo + hi) >> 1;
    if (arr[mid] < val) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}
)cuda"
R"cuda(
// ----------------------------------------------------------------------------
// Kernel 1E: qwen4_qsa_index_topk512_radix_cuda (Zero-Sort Radix Top-512)
// Exact IEEE-754 order-preserving 11+10+10-bit Radix-Select with tie-breaking
// by smaller block index (emits strictly ascending block_ids, 0 B spills).
// Grid: dim3(q_cnt, 1, 1), Block: dim3(256, 1, 1)
// ----------------------------------------------------------------------------
__global__ __launch_bounds__(256, 4) void qwen4_qsa_index_topk512_radix_cuda(
    const float* __restrict__ scores,
    int* __restrict__ block_ids,
    int* __restrict__ num_selected,
    int q0,
    int num_blocks,
    int score_stride,
    int base_pos,
    int tree_mode) {
  const int ql = static_cast<int>(blockIdx.x);
  const int q = q0 + ql;
  const int tid = static_cast<int>(threadIdx.x);

  int complete = tree_mode ? (base_pos >> 2) : ((base_pos + q + 1) >> 2);
  if (complete > num_blocks) complete = num_blocks;
  if (complete < 0) complete = 0;

  int* out_ids = block_ids + (size_t)q * 512;
  if (complete <= 512) {
    for (int i = tid; i < 512; i += 256) out_ids[i] = (i < complete) ? i : -1;
    if (tid == 0) num_selected[q] = complete;
    return;
  }

  __shared__ unsigned int hist[2048];
  __shared__ unsigned int tie_u[1024];
  __shared__ int tie_b[1024];
  __shared__ int sA[512];
  __shared__ int sB[512];
  __shared__ unsigned int s_warp[8];
  __shared__ int s_res[4];

  const float* srow = scores + (size_t)ql * score_stride;

#pragma unroll
  for (int i = 0; i < 8; ++i) hist[tid + i * 256] = 0u;
  __syncthreads();

  for (int b0 = 0; b0 < complete; b0 += 1024) {
    const int b = b0 + tid * 4;
    if (b + 3 < complete) {
      const float4 fv = *reinterpret_cast<const float4*>(srow + b);
      atomicAdd(&hist[(qsa_score_to_u32(fv.x) >> 20) & 2047u], 1u);
      atomicAdd(&hist[(qsa_score_to_u32(fv.y) >> 20) & 2047u], 1u);
      atomicAdd(&hist[(qsa_score_to_u32(fv.z) >> 20) & 2047u], 1u);
      atomicAdd(&hist[(qsa_score_to_u32(fv.w) >> 20) & 2047u], 1u);
    } else if (b < complete) {
      const float4 fv = *reinterpret_cast<const float4*>(srow + b);
      atomicAdd(&hist[(qsa_score_to_u32(fv.x) >> 20) & 2047u], 1u);
      if (b + 1 < complete) atomicAdd(&hist[(qsa_score_to_u32(fv.y) >> 20) & 2047u], 1u);
      if (b + 2 < complete) atomicAdd(&hist[(qsa_score_to_u32(fv.z) >> 20) & 2047u], 1u);
    }
  }
  __syncthreads();

  int bkt1 = 0, gt1 = 0;
  qsa_hist_find_thr<2048>(hist, 512, s_warp, s_res, tid, bkt1, gt1);
  const int n_tie1 = static_cast<int>(hist[bkt1]);
  // Ensure all warps finish reading hist[bkt1] before degenerate fallback zeroes hist[].
  __syncthreads();
  const int need1 = 512 - gt1;
  const unsigned int b1_u = static_cast<unsigned int>(bkt1);

  if (n_tie1 <= 1024) {
    int cur_A = 0;
    int cur_T = 0;
    for (int b0 = 0; b0 < complete; b0 += 1024) {
      const int b = b0 + tid * 4;
      unsigned int u0 = 0u, u1 = 0u, u2 = 0u, u3 = 0u;
      const bool v0 = (b + 0 < complete), v1 = (b + 1 < complete);
      const bool v2 = (b + 2 < complete), v3 = (b + 3 < complete);
      if (v0) {
        const float4 fv = *reinterpret_cast<const float4*>(srow + b);
        u0 = qsa_score_to_u32(fv.x);
        u1 = qsa_score_to_u32(fv.y);
        u2 = qsa_score_to_u32(fv.z);
        u3 = qsa_score_to_u32(fv.w);
      }
      const unsigned int k0 = (u0 >> 20) & 2047u;
      const unsigned int k1 = (u1 >> 20) & 2047u;
      const unsigned int k2 = (u2 >> 20) & 2047u;
      const unsigned int k3 = (u3 >> 20) & 2047u;
      const bool a0 = v0 && (k0 > b1_u), a1 = v1 && (k1 > b1_u);
      const bool a2 = v2 && (k2 > b1_u), a3 = v3 && (k3 > b1_u);
      const bool t0 = v0 && (k0 == b1_u), t1 = v1 && (k1 == b1_u);
      const bool t2 = v2 && (k2 == b1_u), t3 = v3 && (k3 == b1_u);
      const unsigned int cA = (unsigned int)a0 + a1 + a2 + a3;
      const unsigned int cT = (unsigned int)t0 + t1 + t2 + t3;
      unsigned int tot_pack = 0u;
      const unsigned int off_pack = qsa_block_excl_scan_256((cT << 16) | cA, s_warp, tid, tot_pack);
      int posA = cur_A + static_cast<int>(off_pack & 0xffffu);
      int posT = cur_T + static_cast<int>(off_pack >> 16);
      if (a0) sA[posA++] = b + 0;
      if (a1) sA[posA++] = b + 1;
      if (a2) sA[posA++] = b + 2;
      if (a3) sA[posA++] = b + 3;
      if (t0) { tie_u[posT] = u0; tie_b[posT] = b + 0; posT++; }
      if (t1) { tie_u[posT] = u1; tie_b[posT] = b + 1; posT++; }
      if (t2) { tie_u[posT] = u2; tie_b[posT] = b + 2; posT++; }
      if (t3) { tie_u[posT] = u3; tie_b[posT] = b + 3; posT++; }
      cur_A += static_cast<int>(tot_pack & 0xffffu);
      cur_T += static_cast<int>(tot_pack >> 16);
    }
    __syncthreads();

    if (need1 == n_tie1) {
      for (int i = tid; i < n_tie1; i += 256) sB[i] = tie_b[i];
      __syncthreads();
    } else {
#pragma unroll
      for (int i = 0; i < 4; ++i) hist[tid + i * 256] = 0u;
      __syncthreads();
      for (int i = tid; i < n_tie1; i += 256) {
        atomicAdd(&hist[(tie_u[i] >> 10) & 1023u], 1u);
      }
      __syncthreads();
      int bkt2 = 0, gt2 = 0;
      qsa_hist_find_thr<1024>(hist, need1, s_warp, s_res, tid, bkt2, gt2);
      const int need2 = need1 - gt2;
      const unsigned int b2_u = static_cast<unsigned int>(bkt2);

#pragma unroll
      for (int i = 0; i < 4; ++i) hist[tid + i * 256] = 0u;
      __syncthreads();
      for (int i = tid; i < n_tie1; i += 256) {
        if (((tie_u[i] >> 10) & 1023u) == b2_u) {
          atomicAdd(&hist[tie_u[i] & 1023u], 1u);
        }
      }
      __syncthreads();
      int bkt3 = 0, gt3 = 0;
      qsa_hist_find_thr<1024>(hist, need2, s_warp, s_res, tid, bkt3, gt3);
      const int need3 = need2 - gt3;
      const unsigned int u_thr = (b1_u << 20) | (b2_u << 10) | static_cast<unsigned int>(bkt3);

      int cur_gt = 0;
      int cur_eq = 0;
      for (int i0 = 0; i0 < n_tie1; i0 += 256) {
        const int idx = i0 + tid;
        const bool ok = (idx < n_tie1);
        const unsigned int u = ok ? tie_u[idx] : 0u;
        const bool is_gt = ok && (u > u_thr);
        const bool is_eq = ok && (u == u_thr);
        unsigned int tot_p = 0u;
        const unsigned int off_p = qsa_block_excl_scan_256(
            ((unsigned int)is_eq << 16) | (unsigned int)is_gt, s_warp, tid, tot_p);
        const int gt_bef = cur_gt + static_cast<int>(off_p & 0xffffu);
        const int eq_bef = cur_eq + static_cast<int>(off_p >> 16);
        if (is_gt || (is_eq && eq_bef < need3)) {
          const int pos = gt_bef + (eq_bef < need3 ? eq_bef : need3);
          sB[pos] = tie_b[idx];
        }
        cur_gt += static_cast<int>(tot_p & 0xffffu);
        cur_eq += static_cast<int>(tot_p >> 16);
      }
      __syncthreads();
    }

    for (int i = tid; i < gt1; i += 256) {
      const int val = sA[i];
      out_ids[i + qsa_lower_bound(sB, need1, val)] = val;
    }
    for (int j = tid; j < need1; j += 256) {
      const int val = sB[j];
      out_ids[j + qsa_lower_bound(sA, gt1, val)] = val;
    }
    if (tid == 0) num_selected[q] = 512;
    return;
  }

  // Degenerate fallback (hist[bkt1] > 1024, e.g., massive ties / all-zero ReLU)
#pragma unroll
  for (int i = 0; i < 4; ++i) hist[tid + i * 256] = 0u;
  __syncthreads();
  for (int b0 = 0; b0 < complete; b0 += 1024) {
    const int b = b0 + tid * 4;
    if (b < complete) {
      const float4 fv = *reinterpret_cast<const float4*>(srow + b);
      const unsigned int u0 = qsa_score_to_u32(fv.x);
      if (((u0 >> 20) & 2047u) == b1_u) atomicAdd(&hist[(u0 >> 10) & 1023u], 1u);
      if (b + 1 < complete) {
        const unsigned int u1 = qsa_score_to_u32(fv.y);
        if (((u1 >> 20) & 2047u) == b1_u) atomicAdd(&hist[(u1 >> 10) & 1023u], 1u);
      }
      if (b + 2 < complete) {
        const unsigned int u2 = qsa_score_to_u32(fv.z);
        if (((u2 >> 20) & 2047u) == b1_u) atomicAdd(&hist[(u2 >> 10) & 1023u], 1u);
      }
      if (b + 3 < complete) {
        const unsigned int u3 = qsa_score_to_u32(fv.w);
        if (((u3 >> 20) & 2047u) == b1_u) atomicAdd(&hist[(u3 >> 10) & 1023u], 1u);
      }
    }
  }
  __syncthreads();
  int bkt2 = 0, gt2 = 0;
  qsa_hist_find_thr<1024>(hist, need1, s_warp, s_res, tid, bkt2, gt2);
  const int need2 = need1 - gt2;
  const unsigned int b12_u = (b1_u << 10) | static_cast<unsigned int>(bkt2);

#pragma unroll
  for (int i = 0; i < 4; ++i) hist[tid + i * 256] = 0u;
  __syncthreads();
  for (int b0 = 0; b0 < complete; b0 += 1024) {
    const int b = b0 + tid * 4;
    if (b < complete) {
      const float4 fv = *reinterpret_cast<const float4*>(srow + b);
      const unsigned int u0 = qsa_score_to_u32(fv.x);
      if ((u0 >> 10) == b12_u) atomicAdd(&hist[u0 & 1023u], 1u);
      if (b + 1 < complete) {
        const unsigned int u1 = qsa_score_to_u32(fv.y);
        if ((u1 >> 10) == b12_u) atomicAdd(&hist[u1 & 1023u], 1u);
      }
      if (b + 2 < complete) {
        const unsigned int u2 = qsa_score_to_u32(fv.z);
        if ((u2 >> 10) == b12_u) atomicAdd(&hist[u2 & 1023u], 1u);
      }
      if (b + 3 < complete) {
        const unsigned int u3 = qsa_score_to_u32(fv.w);
        if ((u3 >> 10) == b12_u) atomicAdd(&hist[u3 & 1023u], 1u);
      }
    }
  }
  __syncthreads();
  int bkt3 = 0, gt3 = 0;
  qsa_hist_find_thr<1024>(hist, need2, s_warp, s_res, tid, bkt3, gt3);
  const int need3 = need2 - gt3;
  const unsigned int u_thr = (b12_u << 10) | static_cast<unsigned int>(bkt3);

  int cur_gt = 0;
  int cur_eq = 0;
  for (int b0 = 0; b0 < complete; b0 += 1024) {
    const int b = b0 + tid * 4;
    unsigned int u0 = 0u, u1 = 0u, u2 = 0u, u3 = 0u;
    const bool v0 = (b + 0 < complete), v1 = (b + 1 < complete);
    const bool v2 = (b + 2 < complete), v3 = (b + 3 < complete);
    if (v0) {
      const float4 fv = *reinterpret_cast<const float4*>(srow + b);
      u0 = qsa_score_to_u32(fv.x);
      u1 = qsa_score_to_u32(fv.y);
      u2 = qsa_score_to_u32(fv.z);
      u3 = qsa_score_to_u32(fv.w);
    }
    const bool g0 = v0 && (u0 > u_thr), g1 = v1 && (u1 > u_thr);
    const bool g2 = v2 && (u2 > u_thr), g3 = v3 && (u3 > u_thr);
    const bool e0 = v0 && (u0 == u_thr), e1 = v1 && (u1 == u_thr);
    const bool e2 = v2 && (u2 == u_thr), e3 = v3 && (u3 == u_thr);
    const unsigned int cG = (unsigned int)g0 + g1 + g2 + g3;
    const unsigned int cE = (unsigned int)e0 + e1 + e2 + e3;
    unsigned int tot_p = 0u;
    const unsigned int off_p = qsa_block_excl_scan_256((cE << 16) | cG, s_warp, tid, tot_p);
    int gb = cur_gt + static_cast<int>(off_p & 0xffffu);
    int eb = cur_eq + static_cast<int>(off_p >> 16);
    if (g0) { out_ids[gb + (eb < need3 ? eb : need3)] = b + 0; gb++; }
    else if (e0) { if (eb < need3) out_ids[gb + eb] = b + 0; eb++; }
    if (g1) { out_ids[gb + (eb < need3 ? eb : need3)] = b + 1; gb++; }
    else if (e1) { if (eb < need3) out_ids[gb + eb] = b + 1; eb++; }
    if (g2) { out_ids[gb + (eb < need3 ? eb : need3)] = b + 2; gb++; }
    else if (e2) { if (eb < need3) out_ids[gb + eb] = b + 2; eb++; }
    if (g3) { out_ids[gb + (eb < need3 ? eb : need3)] = b + 3; gb++; }
    else if (e3) { if (eb < need3) out_ids[gb + eb] = b + 3; eb++; }
    cur_gt += static_cast<int>(tot_p & 0xffffu);
    cur_eq += static_cast<int>(tot_p >> 16);
  }
  if (tid == 0) num_selected[q] = 512;
}
)cuda"
R"cuda(
// ----------------------------------------------------------------------------
// Kernel 2A: qwen4_qsa_premerge_bq5_cuda (Parallel 5-Way Block Union + Adaptive Cutoff)
// Deduplicates 5 adjacent sorted 512-block lists in SMEM via parallel binary search.
// If union size <= max_tiles*8 (default max_tiles=152 -> 1216 blocks): writes
// tile_blks[qt, max_tiles, 8] + tile_cmask[qt, max_tiles, 5] + tile_counts[qt] >= 0.
// If union size > max_tiles*8 (low-overlap / disjoint): sets tile_counts[qt] = -1
// so qwen4_qsa_sdpa_prefill_indep4w_cuda handles those queries with zero union waste.
// Grid: dim3((Q_len + 4) / 5, 1, 1), Block: dim3(256, 1, 1)
// ----------------------------------------------------------------------------
__device__ __forceinline__ bool qsa_bsearch_exists(const int* arr, int n, int key) {
  if (n <= 0) return false;
  int lo = 0, hi = n;
  while (lo < hi) {
    const int mid = (lo + hi) >> 1;
    if (arr[mid] < key) lo = mid + 1;
    else hi = mid;
  }
  return (lo < n) && (arr[lo] == key);
}

__global__ __launch_bounds__(256, 4) void qwen4_qsa_premerge_bq5_cuda(
    const int* __restrict__ block_ids,
    const int* __restrict__ num_selected,
    int* __restrict__ tile_blks,
    unsigned int* __restrict__ tile_cmask,
    int* __restrict__ tile_counts,
    int Q_len,
    int topk,
    int max_tiles) {
  constexpr int BQ = 5;
  constexpr int MAX_UBLK = 1216; // 152 * 8
  const int qt = static_cast<int>(blockIdx.x);
  const int q_base = qt * BQ;
  const int valid_bq = min(BQ, Q_len - q_base);
  const int tid = static_cast<int>(threadIdx.x);
  const int cap_ublk = min(MAX_UBLK, max_tiles * 8);

  __shared__ int s_lists[BQ][512];        // 10240 B
  __shared__ int s_ns[BQ];                //    20 B
  __shared__ int s_ublk[MAX_UBLK];        //  4864 B
  __shared__ unsigned char s_umsk[MAX_UBLK]; // 1216 B
  __shared__ unsigned int s_warp[8];      //    32 B -> total ~16.4 KB

  if (tid < BQ) {
    s_ns[tid] = (tid < valid_bq) ? num_selected[q_base + tid] : 0;
  }
  __syncthreads();

#pragma unroll
  for (int r = 0; r < BQ; ++r) {
    const int ns_r = s_ns[r];
    if (ns_r > 0) {
      const int* src = block_ids + (size_t)(q_base + r) * topk;
      for (int i = tid; i < ns_r; i += 256) {
        s_lists[r][i] = src[i];
      }
    }
  }
  __syncthreads();

  int cur_u = 0;
#pragma unroll
  for (int r = 0; r < BQ; ++r) {
    const int ns_r = s_ns[r];
    if (ns_r <= 0) continue;
    for (int i0 = 0; i0 < ns_r; i0 += 256) {
      const int idx = i0 + tid;
      const bool valid = (idx < ns_r);
      const int blk = valid ? s_lists[r][idx] : -1;
      unsigned int mask = 0u;
      bool is_canon = false;
      if (valid && blk >= 0) {
        bool dup = false;
#pragma unroll
        for (int o = 0; o < BQ; ++o) {
          if (o < r) {
            if (qsa_bsearch_exists(s_lists[o], s_ns[o], blk)) { dup = true; break; }
          }
        }
        if (!dup) {
          is_canon = true;
          mask = 1u << r;
#pragma unroll
          for (int o = 0; o < BQ; ++o) {
            if (o > r) {
              if (qsa_bsearch_exists(s_lists[o], s_ns[o], blk)) mask |= (1u << o);
            }
          }
        }
      }
      unsigned int tot = 0u;
      const unsigned int off = qsa_block_excl_scan_256(is_canon ? 1u : 0u, s_warp, tid, tot);
      if (cur_u + static_cast<int>(tot) > cap_ublk) {
        if (tid == 0) tile_counts[qt] = -1;
        return;
      }
      if (is_canon) {
        const int pos = cur_u + static_cast<int>(off);
        s_ublk[pos] = blk;
        s_umsk[pos] = static_cast<unsigned char>(mask);
      }
      cur_u += static_cast<int>(tot);
    }
  }
  __syncthreads();

  const int num_tiles = (cur_u + 7) >> 3;
  if (tid == 0) tile_counts[qt] = num_tiles;

  int* out_blks = tile_blks + (size_t)qt * max_tiles * 8;
  unsigned int* out_cm = tile_cmask + (size_t)qt * max_tiles * BQ;

  const int total_slots = num_tiles * 8;
  for (int base = 0; base < total_slots; base += 256) {
    const int slot = base + tid;
    const bool in_slots = (slot < total_slots);
    const bool ok = (slot < cur_u);
    const int blk = ok ? s_ublk[slot] : -1;
    const unsigned int m = ok ? static_cast<unsigned int>(s_umsk[slot]) : 0u;
    if (in_slots) out_blks[slot] = blk;

    const int s = slot & 7;
    const int t = slot >> 3;
    const unsigned int nib = 0xfu << (s * 4);
#pragma unroll
    for (int r = 0; r < BQ; ++r) {
      unsigned int cm = ((m >> r) & 1u) ? nib : 0u;
      cm |= __shfl_xor_sync(0xffffffffu, cm, 1);
      cm |= __shfl_xor_sync(0xffffffffu, cm, 2);
      cm |= __shfl_xor_sync(0xffffffffu, cm, 4);
      if (s == 0 && in_slots) out_cm[t * BQ + r] = cm;
    }
  }
}

// ----------------------------------------------------------------------------
// Kernel 2B: qwen4_qsa_sdpa_prefill_qtiled5_cuda (BQ=5, 4w=128 thr, 60 Q-rows, GQA=12)
// 93.75% MMA row efficiency, 8 warps/SM (2 CTAs/SM), 48.2 KB static SMEM, 0 B spills.
// Skips Q-tiles marked tile_counts[qt] < 0 (handled by Kernel 2C).
// Grid: dim3((Q_len + 4) / 5, Hkv, 1), Block: dim3(128, 1, 1)
// ----------------------------------------------------------------------------
__global__ __launch_bounds__(128, 2) void qwen4_qsa_sdpa_prefill_qtiled5_cuda(
    const nv_bfloat16* __restrict__ Q,
    const nv_bfloat16* __restrict__ K,
    const nv_bfloat16* __restrict__ V,
    const int* __restrict__ tile_blks,
    const unsigned int* __restrict__ tile_cmask,
    const int* __restrict__ tile_counts,
    nv_bfloat16* __restrict__ O,
    int Hkv, int Q_len, int K_len, int base_pos, int max_tiles,
    long long k_sh, long long k_ss, long long v_sh, long long v_ss,
    float scale_log2) {
  constexpr int GQA = 12;
  constexpr int BQ = 5;
  constexpr int BR = 60;
  constexpr int BN = 32;
  constexpr int D = 256;

  const int q_tile = static_cast<int>(blockIdx.x);
  const int num_blk_tiles = tile_counts[q_tile];
  if (num_blk_tiles < 0) return;

  const int q_base = q_tile * BQ;
  const int hkv = static_cast<int>(blockIdx.y);
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int group_id = lane >> 2;
  const int tid_in_grp = lane & 3;
  const int valid_bq = min(BQ, Q_len - q_base);

  __shared__ __align__(128) nv_bfloat16 sQ_hi[BR * 128]; // 15 KB
  __shared__ __align__(128) nv_bfloat16 sK[BN * D];      // 16 KB
  __shared__ __align__(128) nv_bfloat16 sV[BN * D];      // 16 KB
  __shared__ int s_blk[2][8];                            // 64 B
  __shared__ unsigned int s_cm[2][BQ + 1];               // 48 B

  const long long q_head_stride = (long long)Q_len * D;
  const nv_bfloat16* Q_hkv = Q + (long long)hkv * GQA * q_head_stride;
  for (int idx = tid; idx < BR * 16; idx += 128) {
    const int r = idx >> 4;
    const int c8 = idx & 15;
    const int ql = r / GQA;
    const int gh = r - ql * GQA;
    const bool ok = (ql < valid_bq);
    const nv_bfloat16* row_ptr = ok ? (Q_hkv + (long long)gh * q_head_stride + (long long)(q_base + ql) * D) : Q_hkv;
    qsa_cp16_zfill(&sK[qsa_swiz128(r, c8)], row_ptr + c8 * 8, ok);
    qsa_cp16_zfill(&sQ_hi[qsa_swiz128(r, c8)], row_ptr + (c8 + 16) * 8, ok);
  }
  if (tid == 0) { s_cm[0][BQ] = 0u; s_cm[1][BQ] = 0u; }
  asm volatile("cp.async.commit_group;\n" ::: "memory");
  asm volatile("cp.async.wait_group 0;\n" ::: "memory");
  __syncthreads();

  unsigned int q_lo[8][4];
  const int raw_q_r = warp * 16 + (lane & 15);
  const int q_ld_r = (raw_q_r < BR) ? raw_q_r : 0;
  const int q_ld_c8_base = (lane >> 4) & 1;
#pragma unroll
  for (int kt = 0; kt < 8; ++kt) {
    qsa_ldmatrix4(q_lo[kt][0], q_lo[kt][1], q_lo[kt][2], q_lo[kt][3],
                  &sK[qsa_swiz128(q_ld_r, q_ld_c8_base + kt * 2)]);
  }
  __syncthreads();

  const int* my_blks = tile_blks + (size_t)q_tile * max_tiles * 8;
  const unsigned int* my_cms = tile_cmask + (size_t)q_tile * max_tiles * BQ;
  const nv_bfloat16* K_base = K + (long long)hkv * k_sh;
  const nv_bfloat16* V_base = V + (long long)hkv * v_sh;

  auto load_kv_unchecked = [&](nv_bfloat16* dst, const nv_bfloat16* base, long long stride, int buf) {
    const int r_in_blk = tid >> 5;
    const int c8 = tid & 31;
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      const int blk = s_blk[buf][s];
      const nv_bfloat16* gp = base + (long long)(blk * 4 + r_in_blk) * stride + c8 * 8;
      qsa_cp16(&dst[qsa_swiz256(s * 4 + r_in_blk, c8)], gp);
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
  };

  auto load_kv_checked = [&](nv_bfloat16* dst, const nv_bfloat16* base, long long stride, int buf) {
    const int r_in_blk = tid >> 5;
    const int c8 = tid & 31;
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      const int blk = s_blk[buf][s];
      const bool ok = (blk >= 0);
      const nv_bfloat16* gp = ok ? (base + (long long)(blk * 4 + r_in_blk) * stride + c8 * 8) : base;
      qsa_cp16_zfill(&dst[qsa_swiz256(s * 4 + r_in_blk, c8)], gp, ok);
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
  };

  if (num_blk_tiles > 0) {
    if (tid < 8) s_blk[0][tid] = my_blks[tid];
    if (tid < BQ) s_cm[0][tid] = my_cms[tid];
    __syncthreads();
    if (num_blk_tiles > 1) load_kv_unchecked(sK, K_base, k_ss, 0);
    else load_kv_checked(sK, K_base, k_ss, 0);
  }

  float accO[32][4];
#pragma unroll
  for (int t = 0; t < 32; ++t) {
    accO[t][0] = 0.0f; accO[t][1] = 0.0f; accO[t][2] = 0.0f; accO[t][3] = 0.0f;
  }
  float m_row0 = -1e30f, m_row1 = -1e30f, l_row0 = 0.0f, l_row1 = 0.0f;

  const int row0_idx = warp * 16 + group_id;
  const int row1_idx = row0_idx + 8;
  const int ql0 = (row0_idx < BR) ? (row0_idx / GQA) : BQ;
  const int ql1 = (row1_idx < BR) ? (row1_idx / GQA) : BQ;
  const int k_ld_r0 = ((lane >> 4) * 8) + (lane & 7);
  const int k_ld_c8_base = (lane >> 3) & 1;
  const int v_ld_r0 = lane & 15;
  const int v_ld_c8_base = (lane >> 4) & 1;
)cuda"
R"cuda(
  for (int tile = 0; tile < num_blk_tiles; ++tile) {
    const int cur = tile & 1;
    const int nxt = cur ^ 1;
    const bool is_last = (tile + 1 == num_blk_tiles);
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    __syncthreads();

    if (!is_last) load_kv_unchecked(sV, V_base, v_ss, cur);
    else load_kv_checked(sV, V_base, v_ss, cur);

    if (!is_last) {
      if (tid < 8) s_blk[nxt][tid] = my_blks[(tile + 1) * 8 + tid];
      if (tid < BQ) s_cm[nxt][tid] = my_cms[(tile + 1) * BQ + tid];
    }

    float s_acc[4][4] = {};
#pragma unroll
    for (int kt = 0; kt < 8; ++kt) {
      const int c8 = k_ld_c8_base + kt * 2;
      unsigned int b0, b1, b2, b3;
      qsa_ldmatrix4(b0, b1, b2, b3, &sK[qsa_swiz256(k_ld_r0, c8)]);
      qsa_mma_m16n8k16_bf16(s_acc[0][0], s_acc[0][1], s_acc[0][2], s_acc[0][3],
                            q_lo[kt][0], q_lo[kt][1], q_lo[kt][2], q_lo[kt][3],
                            b0, b1, s_acc[0][0], s_acc[0][1], s_acc[0][2], s_acc[0][3]);
      qsa_mma_m16n8k16_bf16(s_acc[1][0], s_acc[1][1], s_acc[1][2], s_acc[1][3],
                            q_lo[kt][0], q_lo[kt][1], q_lo[kt][2], q_lo[kt][3],
                            b2, b3, s_acc[1][0], s_acc[1][1], s_acc[1][2], s_acc[1][3]);
      qsa_ldmatrix4(b0, b1, b2, b3, &sK[qsa_swiz256(k_ld_r0 + 16, c8)]);
      qsa_mma_m16n8k16_bf16(s_acc[2][0], s_acc[2][1], s_acc[2][2], s_acc[2][3],
                            q_lo[kt][0], q_lo[kt][1], q_lo[kt][2], q_lo[kt][3],
                            b0, b1, s_acc[2][0], s_acc[2][1], s_acc[2][2], s_acc[2][3]);
      qsa_mma_m16n8k16_bf16(s_acc[3][0], s_acc[3][1], s_acc[3][2], s_acc[3][3],
                            q_lo[kt][0], q_lo[kt][1], q_lo[kt][2], q_lo[kt][3],
                            b2, b3, s_acc[3][0], s_acc[3][1], s_acc[3][2], s_acc[3][3]);
    }
#pragma unroll
    for (int kt = 0; kt < 8; ++kt) {
      unsigned int a0, a1, a2, a3;
      qsa_ldmatrix4(a0, a1, a2, a3, &sQ_hi[qsa_swiz128(q_ld_r, q_ld_c8_base + kt * 2)]);
      const int c8 = k_ld_c8_base + 16 + kt * 2;
      unsigned int b0, b1, b2, b3;
      qsa_ldmatrix4(b0, b1, b2, b3, &sK[qsa_swiz256(k_ld_r0, c8)]);
      qsa_mma_m16n8k16_bf16(s_acc[0][0], s_acc[0][1], s_acc[0][2], s_acc[0][3],
                            a0, a1, a2, a3, b0, b1, s_acc[0][0], s_acc[0][1], s_acc[0][2], s_acc[0][3]);
      qsa_mma_m16n8k16_bf16(s_acc[1][0], s_acc[1][1], s_acc[1][2], s_acc[1][3],
                            a0, a1, a2, a3, b2, b3, s_acc[1][0], s_acc[1][1], s_acc[1][2], s_acc[1][3]);
      qsa_ldmatrix4(b0, b1, b2, b3, &sK[qsa_swiz256(k_ld_r0 + 16, c8)]);
      qsa_mma_m16n8k16_bf16(s_acc[2][0], s_acc[2][1], s_acc[2][2], s_acc[2][3],
                            a0, a1, a2, a3, b0, b1, s_acc[2][0], s_acc[2][1], s_acc[2][2], s_acc[2][3]);
      qsa_mma_m16n8k16_bf16(s_acc[3][0], s_acc[3][1], s_acc[3][2], s_acc[3][3],
                            a0, a1, a2, a3, b2, b3, s_acc[3][0], s_acc[3][1], s_acc[3][2], s_acc[3][3]);
    }

    const unsigned int cm0 = s_cm[cur][ql0];
    const unsigned int cm1 = s_cm[cur][ql1];
    float local_max0 = -1e30f, local_max1 = -1e30f;
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      const int c0 = nt * 8 + 2 * tid_in_grp;
      s_acc[nt][0] = ((cm0 >> c0) & 1u) ? (s_acc[nt][0] * scale_log2) : -1e30f;
      s_acc[nt][1] = ((cm0 >> (c0 + 1)) & 1u) ? (s_acc[nt][1] * scale_log2) : -1e30f;
      s_acc[nt][2] = ((cm1 >> c0) & 1u) ? (s_acc[nt][2] * scale_log2) : -1e30f;
      s_acc[nt][3] = ((cm1 >> (c0 + 1)) & 1u) ? (s_acc[nt][3] * scale_log2) : -1e30f;
      local_max0 = fmaxf(local_max0, fmaxf(s_acc[nt][0], s_acc[nt][1]));
      local_max1 = fmaxf(local_max1, fmaxf(s_acc[nt][2], s_acc[nt][3]));
    }
    local_max0 = fmaxf(local_max0, __shfl_xor_sync(0xffffffffu, local_max0, 1));
    local_max0 = fmaxf(local_max0, __shfl_xor_sync(0xffffffffu, local_max0, 2));
    local_max1 = fmaxf(local_max1, __shfl_xor_sync(0xffffffffu, local_max1, 1));
    local_max1 = fmaxf(local_max1, __shfl_xor_sync(0xffffffffu, local_max1, 2));

    const float new_m0 = fmaxf(m_row0, local_max0);
    const float new_m1 = fmaxf(m_row1, local_max1);
    const float safe_m0 = (new_m0 > -1e20f) ? new_m0 : 0.0f;
    const float safe_m1 = (new_m1 > -1e20f) ? new_m1 : 0.0f;
    const float alpha0 = (m_row0 > -1e20f) ? qsa_exp2f(m_row0 - new_m0) : 0.0f;
    const float alpha1 = (m_row1 > -1e20f) ? qsa_exp2f(m_row1 - new_m1) : 0.0f;

    float local_sum0 = 0.0f, local_sum1 = 0.0f;
    unsigned int p_frag[2][4];
#pragma unroll
    for (int ks = 0; ks < 2; ++ks) {
      const int nt0 = ks * 2, nt1 = ks * 2 + 1;
      const float p00 = qsa_exp2f(s_acc[nt0][0] - safe_m0);
      const float p01 = qsa_exp2f(s_acc[nt0][1] - safe_m0);
      const float p10 = qsa_exp2f(s_acc[nt0][2] - safe_m1);
      const float p11 = qsa_exp2f(s_acc[nt0][3] - safe_m1);
      const float p02 = qsa_exp2f(s_acc[nt1][0] - safe_m0);
      const float p03 = qsa_exp2f(s_acc[nt1][1] - safe_m0);
      const float p12 = qsa_exp2f(s_acc[nt1][2] - safe_m1);
      const float p13 = qsa_exp2f(s_acc[nt1][3] - safe_m1);
      local_sum0 += (p00 + p01) + (p02 + p03);
      local_sum1 += (p10 + p11) + (p12 + p13);
      p_frag[ks][0] = qsa_pack_bf16x2(p00, p01);
      p_frag[ks][1] = qsa_pack_bf16x2(p10, p11);
      p_frag[ks][2] = qsa_pack_bf16x2(p02, p03);
      p_frag[ks][3] = qsa_pack_bf16x2(p12, p13);
    }
    local_sum0 += __shfl_xor_sync(0xffffffffu, local_sum0, 1);
    local_sum0 += __shfl_xor_sync(0xffffffffu, local_sum0, 2);
    local_sum1 += __shfl_xor_sync(0xffffffffu, local_sum1, 1);
    local_sum1 += __shfl_xor_sync(0xffffffffu, local_sum1, 2);

#pragma unroll
    for (int t = 0; t < 32; ++t) {
      accO[t][0] *= alpha0; accO[t][1] *= alpha0;
      accO[t][2] *= alpha1; accO[t][3] *= alpha1;
    }
    l_row0 = l_row0 * alpha0 + local_sum0;
    l_row1 = l_row1 * alpha1 + local_sum1;
    m_row0 = new_m0; m_row1 = new_m1;

    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    __syncthreads();
    if (!is_last) {
      if (tile + 2 < num_blk_tiles) load_kv_unchecked(sK, K_base, k_ss, nxt);
      else load_kv_checked(sK, K_base, k_ss, nxt);
    }

#pragma unroll
    for (int ks = 0; ks < 2; ++ks) {
      const int vr = ks * 16 + v_ld_r0;
#pragma unroll
      for (int dt = 0; dt < 16; ++dt) {
        const int c8 = v_ld_c8_base + dt * 2;
        unsigned int b0, b1, b2, b3;
        qsa_ldmatrix4_trans(b0, b1, b2, b3, &sV[qsa_swiz256(vr, c8)]);
        const int t0 = dt * 2, t1 = dt * 2 + 1;
        qsa_mma_m16n8k16_bf16(accO[t0][0], accO[t0][1], accO[t0][2], accO[t0][3],
                              p_frag[ks][0], p_frag[ks][1], p_frag[ks][2], p_frag[ks][3],
                              b0, b1, accO[t0][0], accO[t0][1], accO[t0][2], accO[t0][3]);
        qsa_mma_m16n8k16_bf16(accO[t1][0], accO[t1][1], accO[t1][2], accO[t1][3],
                              p_frag[ks][0], p_frag[ks][1], p_frag[ks][2], p_frag[ks][3],
                              b2, b3, accO[t1][0], accO[t1][1], accO[t1][2], accO[t1][3]);
      }
    }
  }

  // Final causal incomplete tail tile (<= 7 tokens, runs ONCE after loop)
  {
    const int min_tail = (min(base_pos + q_base + 1, K_len) >> 2) << 2;
    const int max_end = min(base_pos + q_base + valid_bq, K_len);
    const int tail_span = max_end - min_tail;
    if (tail_span > 0) {
      __syncthreads();
      for (int idx = tid; idx < 1024; idx += 128) {
        const int r = idx >> 5;
        const int c8 = idx & 31;
        const bool ok = (r < tail_span);
        const int kp = min_tail + r;
        qsa_cp16_zfill(&sK[qsa_swiz256(r, c8)], ok ? (K_base + (long long)kp * k_ss + c8 * 8) : K_base, ok);
        qsa_cp16_zfill(&sV[qsa_swiz256(r, c8)], ok ? (V_base + (long long)kp * v_ss + c8 * 8) : V_base, ok);
      }
      asm volatile("cp.async.commit_group;\n" ::: "memory");
      asm volatile("cp.async.wait_group 0;\n" ::: "memory");
      __syncthreads();

      const int qe0 = (ql0 < valid_bq) ? min(base_pos + q_base + ql0 + 1, K_len) : 0;
      const int ts0 = (qe0 >> 2) << 2;
      const int qe1 = (ql1 < valid_bq) ? min(base_pos + q_base + ql1 + 1, K_len) : 0;
      const int ts1 = (qe1 >> 2) << 2;

      float s00 = 0.0f, s01 = 0.0f, s02 = 0.0f, s03 = 0.0f;
      float s10 = 0.0f, s11 = 0.0f, s12 = 0.0f, s13 = 0.0f;
#pragma unroll
      for (int kt = 0; kt < 8; ++kt) {
        const int c8 = k_ld_c8_base + kt * 2;
        unsigned int b0, b1, b2, b3;
        qsa_ldmatrix4(b0, b1, b2, b3, &sK[qsa_swiz256(k_ld_r0, c8)]);
        qsa_mma_m16n8k16_bf16(s00, s01, s02, s03, q_lo[kt][0], q_lo[kt][1], q_lo[kt][2], q_lo[kt][3],
                              b0, b1, s00, s01, s02, s03);
        qsa_mma_m16n8k16_bf16(s10, s11, s12, s13, q_lo[kt][0], q_lo[kt][1], q_lo[kt][2], q_lo[kt][3],
                              b2, b3, s10, s11, s12, s13);
      }
#pragma unroll
      for (int kt = 0; kt < 8; ++kt) {
        unsigned int a0, a1, a2, a3;
        qsa_ldmatrix4(a0, a1, a2, a3, &sQ_hi[qsa_swiz128(q_ld_r, q_ld_c8_base + kt * 2)]);
        const int c8 = k_ld_c8_base + 16 + kt * 2;
        unsigned int b0, b1, b2, b3;
        qsa_ldmatrix4(b0, b1, b2, b3, &sK[qsa_swiz256(k_ld_r0, c8)]);
        qsa_mma_m16n8k16_bf16(s00, s01, s02, s03, a0, a1, a2, a3, b0, b1, s00, s01, s02, s03);
        qsa_mma_m16n8k16_bf16(s10, s11, s12, s13, a0, a1, a2, a3, b2, b3, s10, s11, s12, s13);
      }
      const int k0 = min_tail + 2 * tid_in_grp;
      const int k2 = min_tail + 8 + 2 * tid_in_grp;
      s00 = (k0 >= ts0 && k0 < qe0) ? (s00 * scale_log2) : -1e30f;
      s01 = (k0 + 1 >= ts0 && k0 + 1 < qe0) ? (s01 * scale_log2) : -1e30f;
      s02 = (k0 >= ts1 && k0 < qe1) ? (s02 * scale_log2) : -1e30f;
      s03 = (k0 + 1 >= ts1 && k0 + 1 < qe1) ? (s03 * scale_log2) : -1e30f;
      s10 = (k2 >= ts0 && k2 < qe0) ? (s10 * scale_log2) : -1e30f;
      s11 = (k2 + 1 >= ts0 && k2 + 1 < qe0) ? (s11 * scale_log2) : -1e30f;
      s12 = (k2 >= ts1 && k2 < qe1) ? (s12 * scale_log2) : -1e30f;
      s13 = (k2 + 1 >= ts1 && k2 + 1 < qe1) ? (s13 * scale_log2) : -1e30f;

      float local_max0 = fmaxf(fmaxf(s00, s01), fmaxf(s10, s11));
      float local_max1 = fmaxf(fmaxf(s02, s03), fmaxf(s12, s13));
      local_max0 = fmaxf(local_max0, __shfl_xor_sync(0xffffffffu, local_max0, 1));
      local_max0 = fmaxf(local_max0, __shfl_xor_sync(0xffffffffu, local_max0, 2));
      local_max1 = fmaxf(local_max1, __shfl_xor_sync(0xffffffffu, local_max1, 1));
      local_max1 = fmaxf(local_max1, __shfl_xor_sync(0xffffffffu, local_max1, 2));

      const float new_m0 = fmaxf(m_row0, local_max0);
      const float new_m1 = fmaxf(m_row1, local_max1);
      const float safe_m0 = (new_m0 > -1e20f) ? new_m0 : 0.0f;
      const float safe_m1 = (new_m1 > -1e20f) ? new_m1 : 0.0f;
      const float alpha0 = (m_row0 > -1e20f) ? qsa_exp2f(m_row0 - new_m0) : 0.0f;
      const float alpha1 = (m_row1 > -1e20f) ? qsa_exp2f(m_row1 - new_m1) : 0.0f;

      const float p00 = qsa_exp2f(s00 - safe_m0), p01 = qsa_exp2f(s01 - safe_m0);
      const float p10 = qsa_exp2f(s02 - safe_m1), p11 = qsa_exp2f(s03 - safe_m1);
      const float p02 = qsa_exp2f(s10 - safe_m0), p03 = qsa_exp2f(s11 - safe_m0);
      const float p12 = qsa_exp2f(s12 - safe_m1), p13 = qsa_exp2f(s13 - safe_m1);
      float local_sum0 = (p00 + p01) + (p02 + p03);
      float local_sum1 = (p10 + p11) + (p12 + p13);
      local_sum0 += __shfl_xor_sync(0xffffffffu, local_sum0, 1);
      local_sum0 += __shfl_xor_sync(0xffffffffu, local_sum0, 2);
      local_sum1 += __shfl_xor_sync(0xffffffffu, local_sum1, 1);
      local_sum1 += __shfl_xor_sync(0xffffffffu, local_sum1, 2);
      const unsigned int pf0 = qsa_pack_bf16x2(p00, p01);
      const unsigned int pf1 = qsa_pack_bf16x2(p10, p11);
      const unsigned int pf2 = qsa_pack_bf16x2(p02, p03);
      const unsigned int pf3 = qsa_pack_bf16x2(p12, p13);

#pragma unroll
      for (int t = 0; t < 32; ++t) {
        accO[t][0] *= alpha0; accO[t][1] *= alpha0;
        accO[t][2] *= alpha1; accO[t][3] *= alpha1;
      }
      l_row0 = l_row0 * alpha0 + local_sum0;
      l_row1 = l_row1 * alpha1 + local_sum1;

#pragma unroll
      for (int dt = 0; dt < 16; ++dt) {
        const int c8 = v_ld_c8_base + dt * 2;
        unsigned int b0, b1, b2, b3;
        qsa_ldmatrix4_trans(b0, b1, b2, b3, &sV[qsa_swiz256(v_ld_r0, c8)]);
        const int t0 = dt * 2, t1 = dt * 2 + 1;
        qsa_mma_m16n8k16_bf16(accO[t0][0], accO[t0][1], accO[t0][2], accO[t0][3],
                              pf0, pf1, pf2, pf3, b0, b1,
                              accO[t0][0], accO[t0][1], accO[t0][2], accO[t0][3]);
        qsa_mma_m16n8k16_bf16(accO[t1][0], accO[t1][1], accO[t1][2], accO[t1][3],
                              pf0, pf1, pf2, pf3, b2, b3,
                              accO[t1][0], accO[t1][1], accO[t1][2], accO[t1][3]);
      }
    }
  }

  const float inv0 = (l_row0 > 0.0f) ? (1.0f / l_row0) : 0.0f;
  const float inv1 = (l_row1 > 0.0f) ? (1.0f / l_row1) : 0.0f;
  nv_bfloat16* O_hkv = O + (long long)hkv * GQA * q_head_stride;
  if (ql0 < valid_bq) {
    const int gh0 = row0_idx - ql0 * GQA;
    nv_bfloat16* dst0 = O_hkv + (long long)gh0 * q_head_stride + (long long)(q_base + ql0) * D;
#pragma unroll
    for (int t = 0; t < 32; ++t) {
      *reinterpret_cast<unsigned int*>(&dst0[t * 8 + 2 * tid_in_grp]) =
          qsa_pack_bf16x2(accO[t][0] * inv0, accO[t][1] * inv0);
    }
  }
  if (ql1 < valid_bq) {
    const int gh1 = row1_idx - ql1 * GQA;
    nv_bfloat16* dst1 = O_hkv + (long long)gh1 * q_head_stride + (long long)(q_base + ql1) * D;
#pragma unroll
    for (int t = 0; t < 32; ++t) {
      *reinterpret_cast<unsigned int*>(&dst1[t * 8 + 2 * tid_in_grp]) =
          qsa_pack_bf16x2(accO[t][2] * inv1, accO[t][3] * inv1);
    }
  }
}
)cuda"
R"cuda(
// ----------------------------------------------------------------------------
// Kernel 2C: qwen4_qsa_sdpa_prefill_indep4w_cuda (Disjoint Fallback, 0 Barriers!)
// 4 independent warps (128 thr, 1 query/warp, BN=16, 45 KB SMEM, 2 CTAs/SM, 0 spills).
// Only executes queries whose BQ=5 tile_counts[q / 5] < 0 (low-overlap / disjoint);
// returns immediately in ~0.02 ms across 12L when all tiles use Kernel 2B.
// Grid: dim3((Q_len + 3) / 4, Hkv, 1), Block: dim3(128, 1, 1)
// ----------------------------------------------------------------------------
__global__ __launch_bounds__(128, 2) void qwen4_qsa_sdpa_prefill_indep4w_cuda(
    const nv_bfloat16* __restrict__ Q,
    const nv_bfloat16* __restrict__ K,
    const nv_bfloat16* __restrict__ V,
    const int* __restrict__ block_ids,
    const int* __restrict__ num_selected,
    const int* __restrict__ tile_counts,
    nv_bfloat16* __restrict__ O,
    int Hkv, int gqa, int Q_len, int K_len, int topk, int base_pos,
    long long k_sh, long long k_ss, long long v_sh, long long v_ss,
    float scale_log2) {
  constexpr int BQ = 4;
  constexpr int MAX_GQA = 12;
  constexpr int BN = 16;
  constexpr int D = 256;

  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid >> 5;
  const int q = static_cast<int>(blockIdx.x) * BQ + warp;
  if (q >= Q_len) return;
  if (tile_counts && tile_counts[q / 5] >= 0) return;

  const int hkv = static_cast<int>(blockIdx.y);
  const int lane = tid & 31;
  const int group_id = lane >> 2;
  const int tid_in_grp = lane & 3;

  __shared__ __align__(128) nv_bfloat16 sQ_hi[BQ * MAX_GQA * 128]; // 12 KB
  __shared__ __align__(128) nv_bfloat16 sKV[BQ][BN * D];           // 32 KB

  const long long q_head_stride = (long long)Q_len * D;
  const nv_bfloat16* Q_hkv = Q + (long long)hkv * gqa * q_head_stride + (long long)q * D;
  nv_bfloat16* my_sKV = sKV[warp];
  nv_bfloat16* my_sQhi = sQ_hi + warp * MAX_GQA * 128;

  for (int idx = lane; idx < MAX_GQA * 16; idx += 32) {
    const int r = idx >> 4;
    const int c8 = idx & 15;
    const bool ok = (r < gqa);
    const nv_bfloat16* row_ptr = ok ? (Q_hkv + (long long)r * q_head_stride) : Q_hkv;
    qsa_cp16_zfill(&my_sKV[qsa_swiz128(r, c8)], row_ptr + c8 * 8, ok);
    qsa_cp16_zfill(&my_sQhi[qsa_swiz128(r, c8)], row_ptr + (c8 + 16) * 8, ok);
  }
  asm volatile("cp.async.commit_group;\n" ::: "memory");
  asm volatile("cp.async.wait_group 0;\n" ::: "memory");
  __syncwarp();

  unsigned int q_lo[8][4];
  const int l15 = lane & 15;
  const int q_ld_r = (l15 < MAX_GQA) ? l15 : 0;
  const int q_ld_c8_base = (lane >> 4) & 1;
#pragma unroll
  for (int kt = 0; kt < 8; ++kt) {
    qsa_ldmatrix4(q_lo[kt][0], q_lo[kt][1], q_lo[kt][2], q_lo[kt][3],
                  &my_sKV[qsa_swiz128(q_ld_r, q_ld_c8_base + kt * 2)]);
  }
  __syncwarp();

  const int num_sel = num_selected[q];
  const int num_blk_tiles = (num_sel + 3) >> 2;
  const int query_end = min(base_pos + q + 1, K_len);
  const int tail_start = (query_end >> 2) << 2;
  const int comm_tail_len = query_end - tail_start;
  const int total_tiles = num_blk_tiles + (comm_tail_len > 0 ? 1 : 0);
  const int* blk_row = block_ids + (size_t)q * topk;

  const nv_bfloat16* K_base = K + (long long)hkv * k_sh;
  const nv_bfloat16* V_base = V + (long long)hkv * v_sh;

  float accO[32][4];
#pragma unroll
  for (int t = 0; t < 32; ++t) {
    accO[t][0] = 0.0f; accO[t][1] = 0.0f; accO[t][2] = 0.0f; accO[t][3] = 0.0f;
  }
  float m_row0 = -1e30f, m_row1 = -1e30f, l_row0 = 0.0f, l_row1 = 0.0f;

  const int k_ld_r0 = ((lane >> 4) * 8) + (lane & 7);
  const int k_ld_c8_base = (lane >> 3) & 1;
  const int v_ld_r0 = lane & 15;
  const int v_ld_c8_base = (lane >> 4) & 1;

  for (int tile = 0; tile < total_tiles; ++tile) {
    int my_kp = -1;
    if (lane < 16) {
      if (tile < num_blk_tiles) {
        const int slot = tile * 4 + (lane >> 2);
        if (slot < num_sel) {
          const int blk = __ldg(blk_row + slot);
          if (blk >= 0) {
            const int kp = blk * 4 + (lane & 3);
            if (kp < query_end) my_kp = kp;
          }
        }
      } else {
        if (lane < comm_tail_len) my_kp = tail_start + lane;
      }
    }
#pragma unroll
    for (int r = 0; r < 16; ++r) {
      const int kp = __shfl_sync(0xffffffffu, my_kp, r);
      const bool ok = (kp >= 0);
      const nv_bfloat16* gp = ok ? (K_base + (long long)kp * k_ss + lane * 8) : K_base;
      qsa_cp16_zfill(&my_sKV[qsa_swiz256(r, lane)], gp, ok);
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    __syncwarp();

    float s00 = 0.0f, s01 = 0.0f, s02 = 0.0f, s03 = 0.0f;
    float s10 = 0.0f, s11 = 0.0f, s12 = 0.0f, s13 = 0.0f;
#pragma unroll
    for (int kt = 0; kt < 8; ++kt) {
      const int c8 = k_ld_c8_base + kt * 2;
      unsigned int b0, b1, b2, b3;
      qsa_ldmatrix4(b0, b1, b2, b3, &my_sKV[qsa_swiz256(k_ld_r0, c8)]);
      qsa_mma_m16n8k16_bf16(s00, s01, s02, s03, q_lo[kt][0], q_lo[kt][1], q_lo[kt][2], q_lo[kt][3],
                            b0, b1, s00, s01, s02, s03);
      qsa_mma_m16n8k16_bf16(s10, s11, s12, s13, q_lo[kt][0], q_lo[kt][1], q_lo[kt][2], q_lo[kt][3],
                            b2, b3, s10, s11, s12, s13);
    }
#pragma unroll
    for (int kt = 0; kt < 8; ++kt) {
      unsigned int a0, a1, a2, a3;
      qsa_ldmatrix4(a0, a1, a2, a3, &my_sQhi[qsa_swiz128(q_ld_r, q_ld_c8_base + kt * 2)]);
      const int c8 = k_ld_c8_base + 16 + kt * 2;
      unsigned int b0, b1, b2, b3;
      qsa_ldmatrix4(b0, b1, b2, b3, &my_sKV[qsa_swiz256(k_ld_r0, c8)]);
      qsa_mma_m16n8k16_bf16(s00, s01, s02, s03, a0, a1, a2, a3, b0, b1, s00, s01, s02, s03);
      qsa_mma_m16n8k16_bf16(s10, s11, s12, s13, a0, a1, a2, a3, b2, b3, s10, s11, s12, s13);
    }
    __syncwarp();

#pragma unroll
    for (int r = 0; r < 16; ++r) {
      const int kp = __shfl_sync(0xffffffffu, my_kp, r);
      const bool ok = (kp >= 0);
      const nv_bfloat16* gp = ok ? (V_base + (long long)kp * v_ss + lane * 8) : V_base;
      qsa_cp16_zfill(&my_sKV[qsa_swiz256(r, lane)], gp, ok);
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");

    const unsigned int vmask = __ballot_sync(0xffffffffu, my_kp >= 0);
    const int c0 = 2 * tid_in_grp;
    const int c2 = 8 + 2 * tid_in_grp;
    const bool v0 = (vmask >> c0) & 1u, v1 = (vmask >> (c0 + 1)) & 1u;
    const bool v2 = (vmask >> c2) & 1u, v3 = (vmask >> (c2 + 1)) & 1u;
    s00 = v0 ? (s00 * scale_log2) : -1e30f;
    s01 = v1 ? (s01 * scale_log2) : -1e30f;
    s02 = v0 ? (s02 * scale_log2) : -1e30f;
    s03 = v1 ? (s03 * scale_log2) : -1e30f;
    s10 = v2 ? (s10 * scale_log2) : -1e30f;
    s11 = v3 ? (s11 * scale_log2) : -1e30f;
    s12 = v2 ? (s12 * scale_log2) : -1e30f;
    s13 = v3 ? (s13 * scale_log2) : -1e30f;

    float local_max0 = fmaxf(fmaxf(s00, s01), fmaxf(s10, s11));
    float local_max1 = fmaxf(fmaxf(s02, s03), fmaxf(s12, s13));
    local_max0 = fmaxf(local_max0, __shfl_xor_sync(0xffffffffu, local_max0, 1));
    local_max0 = fmaxf(local_max0, __shfl_xor_sync(0xffffffffu, local_max0, 2));
    local_max1 = fmaxf(local_max1, __shfl_xor_sync(0xffffffffu, local_max1, 1));
    local_max1 = fmaxf(local_max1, __shfl_xor_sync(0xffffffffu, local_max1, 2));

    const float new_m0 = fmaxf(m_row0, local_max0);
    const float new_m1 = fmaxf(m_row1, local_max1);
    const float safe_m0 = (new_m0 > -1e20f) ? new_m0 : 0.0f;
    const float safe_m1 = (new_m1 > -1e20f) ? new_m1 : 0.0f;
    const float alpha0 = (m_row0 > -1e20f) ? qsa_exp2f(m_row0 - new_m0) : 0.0f;
    const float alpha1 = (m_row1 > -1e20f) ? qsa_exp2f(m_row1 - new_m1) : 0.0f;

    const float p00 = qsa_exp2f(s00 - safe_m0), p01 = qsa_exp2f(s01 - safe_m0);
    const float p10 = qsa_exp2f(s02 - safe_m1), p11 = qsa_exp2f(s03 - safe_m1);
    const float p02 = qsa_exp2f(s10 - safe_m0), p03 = qsa_exp2f(s11 - safe_m0);
    const float p12 = qsa_exp2f(s12 - safe_m1), p13 = qsa_exp2f(s13 - safe_m1);
    float local_sum0 = (p00 + p01) + (p02 + p03);
    float local_sum1 = (p10 + p11) + (p12 + p13);
    local_sum0 += __shfl_xor_sync(0xffffffffu, local_sum0, 1);
    local_sum0 += __shfl_xor_sync(0xffffffffu, local_sum0, 2);
    local_sum1 += __shfl_xor_sync(0xffffffffu, local_sum1, 1);
    local_sum1 += __shfl_xor_sync(0xffffffffu, local_sum1, 2);
    const unsigned int pf0 = qsa_pack_bf16x2(p00, p01);
    const unsigned int pf1 = qsa_pack_bf16x2(p10, p11);
    const unsigned int pf2 = qsa_pack_bf16x2(p02, p03);
    const unsigned int pf3 = qsa_pack_bf16x2(p12, p13);

#pragma unroll
    for (int t = 0; t < 32; ++t) {
      accO[t][0] *= alpha0; accO[t][1] *= alpha0;
      accO[t][2] *= alpha1; accO[t][3] *= alpha1;
    }
    l_row0 = l_row0 * alpha0 + local_sum0;
    l_row1 = l_row1 * alpha1 + local_sum1;
    m_row0 = new_m0; m_row1 = new_m1;

    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    __syncwarp();

#pragma unroll
    for (int dt = 0; dt < 16; ++dt) {
      const int c8 = v_ld_c8_base + dt * 2;
      unsigned int b0, b1, b2, b3;
      qsa_ldmatrix4_trans(b0, b1, b2, b3, &my_sKV[qsa_swiz256(v_ld_r0, c8)]);
      const int t0 = dt * 2, t1 = dt * 2 + 1;
      qsa_mma_m16n8k16_bf16(accO[t0][0], accO[t0][1], accO[t0][2], accO[t0][3],
                            pf0, pf1, pf2, pf3, b0, b1,
                            accO[t0][0], accO[t0][1], accO[t0][2], accO[t0][3]);
      qsa_mma_m16n8k16_bf16(accO[t1][0], accO[t1][1], accO[t1][2], accO[t1][3],
                            pf0, pf1, pf2, pf3, b2, b3,
                            accO[t1][0], accO[t1][1], accO[t1][2], accO[t1][3]);
    }
    __syncwarp();
  }

  const float inv0 = (l_row0 > 0.0f) ? (1.0f / l_row0) : 0.0f;
  const float inv1 = (l_row1 > 0.0f) ? (1.0f / l_row1) : 0.0f;
  nv_bfloat16* O_hkv = O + (long long)hkv * gqa * q_head_stride + (long long)q * D;
  const int gh0 = group_id, gh1 = group_id + 8;
  if (gh0 < gqa) {
    nv_bfloat16* dst0 = O_hkv + (long long)gh0 * q_head_stride;
#pragma unroll
    for (int t = 0; t < 32; ++t) {
      *reinterpret_cast<unsigned int*>(&dst0[t * 8 + 2 * tid_in_grp]) =
          qsa_pack_bf16x2(accO[t][0] * inv0, accO[t][1] * inv0);
    }
  }
  if (gh1 < gqa) {
    nv_bfloat16* dst1 = O_hkv + (long long)gh1 * q_head_stride;
#pragma unroll
    for (int t = 0; t < 32; ++t) {
      *reinterpret_cast<unsigned int*>(&dst1[t * 8 + 2 * tid_in_grp]) =
          qsa_pack_bf16x2(accO[t][2] * inv1, accO[t][3] * inv1);
    }
  }
}
)cuda"
