R"cuda(
// ============================================================================
// Hy4-Flash Absorbed MLA Utility + Decode / Few-Token Split-K SDPA Kernels
//   1. hy4_ring_write_cuda          (circular / linear in-place buffer write)
//   2. hy4_mla_prep_kv_store_cuda   (optional q_a norm + kv_a norm + NORM-RoPE -> f16 kv_buf)
//   3. hy4_rope_norm_q_cuda         (concat q_abs + NORM-RoPE(q_raw[:, :, nope:]))
//   4. hy4_mla_few_split_512_64     (4-warp, 8-head shared-KV FP32 split-K for D_v=512, rope=64)
//   5. hy4_mla_few_split_generic    (general FP32 split-K for arbitrary kv_lora, rope)
//   6. hy4_mla_few_reduce           (FP32 split-K merge + exact overflow-safe sink softmax)
// ============================================================================

__device__ __forceinline__ long long hy4_few_max_ll(long long a, long long b) {
  return (a > b) ? a : b;
}

__device__ __forceinline__ long long hy4_few_min_ll(long long a, long long b) {
  return (a < b) ? a : b;
}

// ----------------------------------------------------------------------------
// Kernel 1: hy4_ring_write_cuda (dtype-agnostic byte copy by row)
// Grid: dim3(M, 1, 1), Block: dim3(256, 1, 1)
// ----------------------------------------------------------------------------
__global__ __launch_bounds__(256, 4) void hy4_ring_write_cuda(
    unsigned char* __restrict__ buf,
    const unsigned char* __restrict__ rows,
    int M,
    int row_bytes,
    long long start_pos,
    int ring_cap) {
  const int m = static_cast<int>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.x);
  if (m >= M) {
    return;
  }
  const long long logical = start_pos + static_cast<long long>(m);
  const long long dst_row = (ring_cap > 0)
      ? (logical % static_cast<long long>(ring_cap))
      : logical;
  unsigned char* dst = buf + dst_row * static_cast<long long>(row_bytes);
  const unsigned char* src = rows + static_cast<long long>(m) * static_cast<long long>(row_bytes);

  const bool aligned16 = ((row_bytes & 15) == 0) &&
      ((reinterpret_cast<unsigned long long>(dst) & 15ULL) == 0ULL) &&
      ((reinterpret_cast<unsigned long long>(src) & 15ULL) == 0ULL);
  if (aligned16) {
    const int n16 = row_bytes >> 4;
    uint4* d4 = reinterpret_cast<uint4*>(dst);
    const uint4* s4 = reinterpret_cast<const uint4*>(src);
    for (int i = tid; i < n16; i += 256) {
      d4[i] = s4[i];
    }
  } else if (((row_bytes & 1) == 0) &&
             ((reinterpret_cast<unsigned long long>(dst) & 1ULL) == 0ULL) &&
             ((reinterpret_cast<unsigned long long>(src) & 1ULL) == 0ULL)) {
    const int n2 = row_bytes >> 1;
    unsigned short* d2 = reinterpret_cast<unsigned short*>(dst);
    const unsigned short* s2 = reinterpret_cast<const unsigned short*>(src);
    for (int i = tid; i < n2; i += 256) {
      d2[i] = s2[i];
    }
  } else {
    for (int i = tid; i < row_bytes; i += 256) {
      dst[i] = src[i];
    }
  }
}

// ----------------------------------------------------------------------------
// Kernel 2: hy4_mla_prep_kv_store_cuda
//   Optional: out_qr[m, :] = bf16(rms_norm(q_a[m], q_a_norm, eps))
//   Always:   kv_buf[phys_row, :kv_lora] = f16(rms_norm(kv_a[m, :kv_lora], kv_a_norm, eps))
//             kv_buf[phys_row, kv_lora:] = f16(RoPE_NORM(kv_a[m, kv_lora:], rope_start+m))
// Grid: dim3(M, 1, 1), Block: dim3(256, 1, 1)
// ----------------------------------------------------------------------------
template <bool HAS_Q>
__global__ __launch_bounds__(256, 4) void hy4_mla_prep_kv_store_cuda(
    const nv_bfloat16* __restrict__ q_a,
    const nv_bfloat16* __restrict__ kv_a,
    const float* __restrict__ q_a_norm,
    const float* __restrict__ kv_a_norm,
    const float* __restrict__ inv_freq,
    nv_bfloat16* __restrict__ out_qr,
    __half* __restrict__ kv_buf,
    int M,
    int q_lora,
    int kv_lora,
    int half_rope,
    long long q_a_stride,
    long long kv_a_stride,
    long long slot_start,
    long long rope_start,
    int ring_cap,
    float norm_eps) {
  const int m = static_cast<int>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;
  if (m >= M) {
    return;
  }

  const int rope_dim = half_rope * 2;
  const int kv_row = kv_lora + rope_dim;
  const nv_bfloat16* kv_src = kv_a + static_cast<size_t>(m) * static_cast<size_t>(kv_a_stride);
  const long long slot = slot_start + static_cast<long long>(m);
  const long long phys = (ring_cap > 0) ? (slot % static_cast<long long>(ring_cap)) : slot;
  __half* kv_dst = kv_buf + phys * static_cast<long long>(kv_row);

  __shared__ float s_red[16];

  float sq_q = 0.0f;
  if constexpr (HAS_Q) {
    const nv_bfloat16* q_src = q_a + static_cast<size_t>(m) * static_cast<size_t>(q_a_stride);
    for (int j = tid; j < q_lora; j += 256) {
      const float v = __bfloat162float(q_src[j]);
      sq_q = fmaf(v, v, sq_q);
    }
    sq_q = hy4_warp_sum(sq_q);
  }

  float sq_kv = 0.0f;
  for (int j = tid; j < kv_lora; j += 256) {
    const float v = __bfloat162float(kv_src[j]);
    sq_kv = fmaf(v, v, sq_kv);
  }
  sq_kv = hy4_warp_sum(sq_kv);

  if (lane == 0) {
    s_red[warp] = sq_kv;
    if constexpr (HAS_Q) {
      s_red[8 + warp] = sq_q;
    }
  }
  __syncthreads();

  float tot_kv = 0.0f;
  float tot_q = 0.0f;
#pragma unroll
  for (int w = 0; w < 8; ++w) {
    tot_kv += s_red[w];
    if constexpr (HAS_Q) {
      tot_q += s_red[8 + w];
    }
  }

  const float rs_kv = rsqrtf(tot_kv / static_cast<float>(kv_lora) + norm_eps);
  for (int j = tid; j < kv_lora; j += 256) {
    const float v = __bfloat162float(kv_src[j]) * rs_kv * kv_a_norm[j];
    kv_dst[j] = __float2half(v);
  }

  if constexpr (HAS_Q) {
    const float rs_q = rsqrtf(tot_q / static_cast<float>(q_lora) + norm_eps);
    const nv_bfloat16* q_src = q_a + static_cast<size_t>(m) * static_cast<size_t>(q_a_stride);
    nv_bfloat16* q_dst = out_qr + static_cast<size_t>(m) * static_cast<size_t>(q_lora);
    for (int j = tid; j < q_lora; j += 256) {
      const float v = __bfloat162float(q_src[j]) * rs_q * q_a_norm[j];
      q_dst[j] = __float2bfloat16(v);
    }
  }

  const float pos_f = static_cast<float>(rope_start + static_cast<long long>(m));
  for (int p = tid; p < half_rope; p += 256) {
    float s_v, c_v;
    sincosf(pos_f * inv_freq[p], &s_v, &c_v);
    const float x0 = __bfloat162float(kv_src[kv_lora + 2 * p]);
    const float x1 = __bfloat162float(kv_src[kv_lora + 2 * p + 1]);
    kv_dst[kv_lora + 2 * p] = __float2half(x0 * c_v - x1 * s_v);
    kv_dst[kv_lora + 2 * p + 1] = __float2half(x0 * s_v + x1 * c_v);
  }
}

// ----------------------------------------------------------------------------
// Kernel 3: hy4_rope_norm_q_cuda
//   q_abs: bf16 [M, H, kv_lora]
//   q_raw: bf16 [M, H, nope_dim + rope_dim]
//   out_q: bf16 [M, H, kv_lora + rope_dim]
// Grid: dim3(H, M, 1), Block: dim3(128, 1, 1)
// ----------------------------------------------------------------------------
__global__ __launch_bounds__(128, 4) void hy4_rope_norm_q_cuda(
    const nv_bfloat16* __restrict__ q_abs,
    const nv_bfloat16* __restrict__ q_raw,
    const float* __restrict__ inv_freq,
    nv_bfloat16* __restrict__ out_q,
    int M,
    int H,
    int kv_lora,
    int nope_dim,
    int half_rope,
    long long rope_start) {
  const int h = static_cast<int>(blockIdx.x);
  const int m = static_cast<int>(blockIdx.y);
  const int tid = static_cast<int>(threadIdx.x);
  if (m >= M || h >= H) {
    return;
  }

  const int rope_dim = half_rope * 2;
  const int kv_row = kv_lora + rope_dim;
  const int q_raw_row = nope_dim + rope_dim;
  const size_t mh = static_cast<size_t>(m) * static_cast<size_t>(H) + static_cast<size_t>(h);
  const nv_bfloat16* abs_ptr = q_abs + mh * static_cast<size_t>(kv_lora);
  const nv_bfloat16* pe_ptr = q_raw + mh * static_cast<size_t>(q_raw_row) + static_cast<size_t>(nope_dim);
  nv_bfloat16* dst_ptr = out_q + mh * static_cast<size_t>(kv_row);

  if (((kv_lora & 7) == 0) && ((rope_dim & 7) == 0)) {
    const int n8 = kv_lora >> 3;
    const uint4* s4 = reinterpret_cast<const uint4*>(abs_ptr);
    uint4* d4 = reinterpret_cast<uint4*>(dst_ptr);
    for (int i = tid; i < n8; i += 128) {
      d4[i] = s4[i];
    }
  } else {
    for (int j = tid; j < kv_lora; j += 128) {
      dst_ptr[j] = abs_ptr[j];
    }
  }

  const float pos_f = static_cast<float>(rope_start + static_cast<long long>(m));
  for (int p = tid; p < half_rope; p += 128) {
    float s_v, c_v;
    sincosf(pos_f * inv_freq[p], &s_v, &c_v);
    const float x0 = __bfloat162float(pe_ptr[2 * p]);
    const float x1 = __bfloat162float(pe_ptr[2 * p + 1]);
    dst_ptr[kv_lora + 2 * p] = __float2bfloat16(x0 * c_v - x1 * s_v);
    dst_ptr[kv_lora + 2 * p + 1] = __float2bfloat16(x0 * s_v + x1 * c_v);
  }
}
)cuda"
R"cuda(
// ----------------------------------------------------------------------------
// Kernel 4: hy4_mla_few_split_512_64 (Decode / Few-token Split-K BF16 TC MMA)
//   Geometry: kv_lora = 512, rope_dim = 64 (kv_row = 576).
//   8 warps = 256 threads, 16 heads per CTA, BK = 16 KV rows/tile:
//     - Each warp holds 64 latent dims (+ 16 RoPE dims on warps 0..3) of Q in
//       registers and uses mma.sync.m16n8k16 for BOTH Q @ K^T and P @ V.
//     - Shares every KV row read across 16 heads with 128B XOR-swizzled SMEM
//       and skips s_phys + 1 barrier on dense/SWA (!HAS_IDX) calls.
//     - Writes unnormalized (partial_o, partial_m, partial_l) for hy4_mla_few_reduce.
// Grid: dim3(num_splits, (H + 15) / 16, M), Block: dim3(32, 8, 1)
// ----------------------------------------------------------------------------
__device__ __forceinline__ int hy4_few_swiz576(int row, int c8) {
  return row * 576 + ((c8 ^ (row & 7)) << 3);
}

__device__ __forceinline__ uint4 hy4_few_f16x8_to_bf16x8(const uint4& hv) {
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

template <bool HAS_IDX>
__global__ __launch_bounds__(256, 2) void hy4_mla_few_split_512_64(
    const nv_bfloat16* __restrict__ q,
    const __half* __restrict__ kv_buf,
    const int* __restrict__ idx,
    float* __restrict__ partial_o,
    float* __restrict__ partial_m,
    float* __restrict__ partial_l,
    int M,
    int H,
    int cap,
    int kmax,
    long long slot_start,
    int total_len,
    int ring_cap,
    int window,
    float scale,
    int num_splits) {
  constexpr int D_V = 512;
  constexpr int D_QK = 576;
  constexpr int BK = 16;

  const int split = static_cast<int>(blockIdx.x);
  const int h_base = static_cast<int>(blockIdx.y) * 16;
  const int m = static_cast<int>(blockIdx.z);
  const int lane = static_cast<int>(threadIdx.x);
  const int warp = static_cast<int>(threadIdx.y);
  const int tid = warp * 32 + lane;
  if (m >= M || h_base >= H) {
    return;
  }

  const int group_id = lane >> 2;
  const int tid_in_grp = lane & 3;
  const int row0 = group_id;
  const int row1 = group_id + 8;
  const int h0 = h_base + row0;
  const int h1 = h_base + row1;

  const nv_bfloat16* q_m_base = q + static_cast<size_t>(m) * static_cast<size_t>(H) * D_QK;
  const nv_bfloat16* q_ptr0 = (h0 < H) ? (q_m_base + static_cast<size_t>(h0) * D_QK) : nullptr;
  const nv_bfloat16* q_ptr1 = (h1 < H) ? (q_m_base + static_cast<size_t>(h1) * D_QK) : nullptr;

  unsigned int q_reg[5][4];
#pragma unroll
  for (int kt = 0; kt < 4; ++kt) {
    const int k = warp * 64 + kt * 16;
    q_reg[kt][0] = q_ptr0 ? __ldg(reinterpret_cast<const unsigned int*>(q_ptr0 + k + 2 * tid_in_grp)) : 0u;
    q_reg[kt][2] = q_ptr0 ? __ldg(reinterpret_cast<const unsigned int*>(q_ptr0 + k + 8 + 2 * tid_in_grp)) : 0u;
    q_reg[kt][1] = q_ptr1 ? __ldg(reinterpret_cast<const unsigned int*>(q_ptr1 + k + 2 * tid_in_grp)) : 0u;
    q_reg[kt][3] = q_ptr1 ? __ldg(reinterpret_cast<const unsigned int*>(q_ptr1 + k + 8 + 2 * tid_in_grp)) : 0u;
  }
  if (warp < 4) {
    const int k = 512 + warp * 16;
    q_reg[4][0] = q_ptr0 ? __ldg(reinterpret_cast<const unsigned int*>(q_ptr0 + k + 2 * tid_in_grp)) : 0u;
    q_reg[4][2] = q_ptr0 ? __ldg(reinterpret_cast<const unsigned int*>(q_ptr0 + k + 8 + 2 * tid_in_grp)) : 0u;
    q_reg[4][1] = q_ptr1 ? __ldg(reinterpret_cast<const unsigned int*>(q_ptr1 + k + 2 * tid_in_grp)) : 0u;
    q_reg[4][3] = q_ptr1 ? __ldg(reinterpret_cast<const unsigned int*>(q_ptr1 + k + 8 + 2 * tid_in_grp)) : 0u;
  } else {
    q_reg[4][0] = 0u;
    q_reg[4][1] = 0u;
    q_reg[4][2] = 0u;
    q_reg[4][3] = 0u;
  }

  __shared__ __align__(128) nv_bfloat16 sK[2][BK * D_QK];
  __shared__ __align__(16) float sS[8][16][BK];
  __shared__ int s_phys[3][BK];
  __shared__ int s_any_valid[3];

  const long long p_log = slot_start + static_cast<long long>(m);
  long long k_lo = 0;
  int n_items = 0;
  if constexpr (HAS_IDX) {
    n_items = kmax;
  } else {
    k_lo = (window > 0) ? hy4_few_max_ll(0LL, p_log + 1LL - static_cast<long long>(window)) : 0LL;
    const long long k_hi_cap = (ring_cap > 0)
        ? (static_cast<long long>(total_len) - 1LL)
        : hy4_few_min_ll(static_cast<long long>(total_len) - 1LL, static_cast<long long>(cap) - 1LL);
    const long long k_hi = hy4_few_min_ll(p_log, k_hi_cap);
    n_items = (k_hi >= k_lo) ? static_cast<int>(k_hi - k_lo + 1LL) : 0;
  }
  const int n_tiles = (n_items + BK - 1) / BK;
  const int tiles_per_split = (n_tiles + num_splits - 1) / num_splits;
  const int tile_start = split * tiles_per_split;
  const int tile_end = min(tile_start + tiles_per_split, n_tiles);
  const int* idx_row = HAS_IDX ? (idx + static_cast<size_t>(m) * static_cast<size_t>(kmax)) : nullptr;
  (void)idx_row;
  const float scale_log2e = scale * 1.4426950408889634f;

  float accO[8][4];
#pragma unroll
  for (int nt = 0; nt < 8; ++nt) {
    accO[nt][0] = 0.0f;
    accO[nt][1] = 0.0f;
    accO[nt][2] = 0.0f;
    accO[nt][3] = 0.0f;
  }
  float m_row0 = -1e30f;
  float m_row1 = -1e30f;
  float l_row0 = 0.0f;
  float l_row1 = 0.0f;

  const int k_ld_r0 = ((lane >> 4) * 8) + (lane & 7);
  const int k_ld_c8_base = (lane >> 3) & 1;
  const int v_ld_r0 = lane & 15;
  const int v_ld_c8_base = warp * 8 + ((lane >> 4) & 1);

  int pbuf = 0;
  int npbuf = 1;
  (void)pbuf;
  (void)npbuf;
  if constexpr (HAS_IDX) {
    if (tile_start < tile_end) {
      if (warp == 0) {
        const int item = tile_start * BK + lane;
        int phys = -1;
        if (lane < BK && item < n_items) {
          const int kp = idx_row[item];
          if (kp >= 0) {
            const int pr = (ring_cap > 0) ? (kp % ring_cap) : kp;
            if (pr >= 0 && pr < cap) phys = pr;
          }
        }
        if (lane < BK) s_phys[0][lane] = phys;
        const unsigned int vmask = __ballot_sync(0xFFFFFFFFu, (lane < BK) && (phys >= 0));
        if (lane == 0) s_any_valid[0] = (vmask != 0u) ? 1 : 0;
      }
      __syncthreads();
    }
  }

  for (int ti = tile_start; ti < tile_end; ++ti) {
    const int item_base = ti * BK;
    const int buf = (ti - tile_start) & 1;

    if constexpr (HAS_IDX) {
      if (s_any_valid[pbuf] == 0) {
        if (ti + 1 < tile_end && warp == 0) {
          const int item = (ti + 1) * BK + lane;
          int phys = -1;
          if (lane < BK && item < n_items) {
            const int kp = idx_row[item];
            if (kp >= 0) {
              const int pr = (ring_cap > 0) ? (kp % ring_cap) : kp;
              if (pr >= 0 && pr < cap) phys = pr;
            }
          }
          if (lane < BK) s_phys[npbuf][lane] = phys;
          const unsigned int vmask = __ballot_sync(0xFFFFFFFFu, (lane < BK) && (phys >= 0));
          if (lane == 0) s_any_valid[npbuf] = (vmask != 0u) ? 1 : 0;
        }
        __syncthreads();
        pbuf = npbuf;
        npbuf = (npbuf == 2) ? 0 : (npbuf + 1);
        continue;
      }
    }

    for (int i = tid; i < BK * 72; i += 256) {
      const int r = i / 72;
      const int c8 = i - r * 72;
      int phys = -1;
      if constexpr (HAS_IDX) {
        phys = s_phys[pbuf][r];
      } else {
        const int item = item_base + r;
        if (item < n_items) {
          const long long kp = k_lo + static_cast<long long>(item);
          const int pr = (ring_cap > 0)
              ? static_cast<int>(kp % static_cast<long long>(ring_cap))
              : static_cast<int>(kp);
          if (pr >= 0 && pr < cap) phys = pr;
        }
      }
      uint4 bfv = make_uint4(0u, 0u, 0u, 0u);
      if (phys >= 0) {
        const uint4 hv = __ldg(reinterpret_cast<const uint4*>(
            kv_buf + static_cast<size_t>(phys) * D_QK + c8 * 8));
        bfv = hy4_few_f16x8_to_bf16x8(hv);
      }
      *reinterpret_cast<uint4*>(&sK[buf][hy4_few_swiz576(r, c8)]) = bfv;
    }
    __syncthreads();
)cuda"
R"cuda(
    float s_acc[2][4] = {};
#pragma unroll
    for (int kt = 0; kt < 4; ++kt) {
      const int c8 = warp * 8 + kt * 2 + k_ld_c8_base;
      unsigned int b0, b1, b2, b3;
      ptx_ldmatrix_x4(b0, b1, b2, b3, &sK[buf][hy4_few_swiz576(k_ld_r0, c8)]);
      ptx_mma_m16n8k16_bf16_c_v(
          s_acc[0][0], s_acc[0][1], s_acc[0][2], s_acc[0][3],
          q_reg[kt][0], q_reg[kt][1], q_reg[kt][2], q_reg[kt][3],
          b0, b1, s_acc[0][0], s_acc[0][1], s_acc[0][2], s_acc[0][3]);
      ptx_mma_m16n8k16_bf16_c_v(
          s_acc[1][0], s_acc[1][1], s_acc[1][2], s_acc[1][3],
          q_reg[kt][0], q_reg[kt][1], q_reg[kt][2], q_reg[kt][3],
          b2, b3, s_acc[1][0], s_acc[1][1], s_acc[1][2], s_acc[1][3]);
    }
    if (warp < 4) {
      const int c8 = 64 + warp * 2 + k_ld_c8_base;
      unsigned int b0, b1, b2, b3;
      ptx_ldmatrix_x4(b0, b1, b2, b3, &sK[buf][hy4_few_swiz576(k_ld_r0, c8)]);
      ptx_mma_m16n8k16_bf16_c_v(
          s_acc[0][0], s_acc[0][1], s_acc[0][2], s_acc[0][3],
          q_reg[4][0], q_reg[4][1], q_reg[4][2], q_reg[4][3],
          b0, b1, s_acc[0][0], s_acc[0][1], s_acc[0][2], s_acc[0][3]);
      ptx_mma_m16n8k16_bf16_c_v(
          s_acc[1][0], s_acc[1][1], s_acc[1][2], s_acc[1][3],
          q_reg[4][0], q_reg[4][1], q_reg[4][2], q_reg[4][3],
          b2, b3, s_acc[1][0], s_acc[1][1], s_acc[1][2], s_acc[1][3]);
    }

#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
      const int c0 = nt * 8 + 2 * tid_in_grp;
      sS[warp][row0][c0] = s_acc[nt][0];
      sS[warp][row0][c0 + 1] = s_acc[nt][1];
      sS[warp][row1][c0] = s_acc[nt][2];
      sS[warp][row1][c0 + 1] = s_acc[nt][3];
    }
    if constexpr (HAS_IDX) {
      if (ti + 1 < tile_end && warp == 0) {
        const int item = (ti + 1) * BK + lane;
        int phys = -1;
        if (lane < BK && item < n_items) {
          const int kp = idx_row[item];
          if (kp >= 0) {
            const int pr = (ring_cap > 0) ? (kp % ring_cap) : kp;
            if (pr >= 0 && pr < cap) phys = pr;
          }
        }
        if (lane < BK) s_phys[npbuf][lane] = phys;
        const unsigned int vmask = __ballot_sync(0xFFFFFFFFu, (lane < BK) && (phys >= 0));
        if (lane == 0) s_any_valid[npbuf] = (vmask != 0u) ? 1 : 0;
      }
    }
    __syncthreads();

    float local_max0 = -3.4028234663852886e38f;
    float local_max1 = -3.4028234663852886e38f;
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
      const int c0 = nt * 8 + 2 * tid_in_grp;
      const int c1 = c0 + 1;
      const bool v0 = HAS_IDX ? (s_phys[pbuf][c0] >= 0) : (item_base + c0 < n_items);
      const bool v1 = HAS_IDX ? (s_phys[pbuf][c1] >= 0) : (item_base + c1 < n_items);
      float r00 = 0.0f, r01 = 0.0f, r10 = 0.0f, r11 = 0.0f;
#pragma unroll
      for (int w = 0; w < 8; ++w) {
        r00 += sS[w][row0][c0];
        r01 += sS[w][row0][c1];
        r10 += sS[w][row1][c0];
        r11 += sS[w][row1][c1];
      }
      s_acc[nt][0] = v0 ? (r00 * scale_log2e) : -1e30f;
      s_acc[nt][1] = v1 ? (r01 * scale_log2e) : -1e30f;
      s_acc[nt][2] = v0 ? (r10 * scale_log2e) : -1e30f;
      s_acc[nt][3] = v1 ? (r11 * scale_log2e) : -1e30f;
      if (v0) {
        local_max0 = fmaxf(local_max0, s_acc[nt][0]);
        local_max1 = fmaxf(local_max1, s_acc[nt][2]);
      }
      if (v1) {
        local_max0 = fmaxf(local_max0, s_acc[nt][1]);
        local_max1 = fmaxf(local_max1, s_acc[nt][3]);
      }
    }
    local_max0 = fmaxf(local_max0, __shfl_xor_sync(0xFFFFFFFFu, local_max0, 1));
    local_max0 = fmaxf(local_max0, __shfl_xor_sync(0xFFFFFFFFu, local_max0, 2));
    local_max1 = fmaxf(local_max1, __shfl_xor_sync(0xFFFFFFFFu, local_max1, 1));
    local_max1 = fmaxf(local_max1, __shfl_xor_sync(0xFFFFFFFFu, local_max1, 2));

    const float new_m0 = (l_row0 == 0.0f) ? local_max0 : fmaxf(m_row0, local_max0);
    const float new_m1 = (l_row1 == 0.0f) ? local_max1 : fmaxf(m_row1, local_max1);
    const float alpha0 = (l_row0 == 0.0f) ? 0.0f : ptx_ex2_approx_ftz(m_row0 - new_m0);
    const float alpha1 = (l_row1 == 0.0f) ? 0.0f : ptx_ex2_approx_ftz(m_row1 - new_m1);

    const bool v00 = HAS_IDX ? (s_phys[pbuf][2 * tid_in_grp] >= 0) : (item_base + 2 * tid_in_grp < n_items);
    const bool v01 = HAS_IDX ? (s_phys[pbuf][2 * tid_in_grp + 1] >= 0) : (item_base + 2 * tid_in_grp + 1 < n_items);
    const bool v02 = HAS_IDX ? (s_phys[pbuf][8 + 2 * tid_in_grp] >= 0) : (item_base + 8 + 2 * tid_in_grp < n_items);
    const bool v03 = HAS_IDX ? (s_phys[pbuf][8 + 2 * tid_in_grp + 1] >= 0) : (item_base + 8 + 2 * tid_in_grp + 1 < n_items);

    const float p00 = v00 ? ptx_ex2_approx_ftz(s_acc[0][0] - new_m0) : 0.0f;
    const float p01 = v01 ? ptx_ex2_approx_ftz(s_acc[0][1] - new_m0) : 0.0f;
    const float p10 = v00 ? ptx_ex2_approx_ftz(s_acc[0][2] - new_m1) : 0.0f;
    const float p11 = v01 ? ptx_ex2_approx_ftz(s_acc[0][3] - new_m1) : 0.0f;
    const float p02 = v02 ? ptx_ex2_approx_ftz(s_acc[1][0] - new_m0) : 0.0f;
    const float p03 = v03 ? ptx_ex2_approx_ftz(s_acc[1][1] - new_m0) : 0.0f;
    const float p12 = v02 ? ptx_ex2_approx_ftz(s_acc[1][2] - new_m1) : 0.0f;
    const float p13 = v03 ? ptx_ex2_approx_ftz(s_acc[1][3] - new_m1) : 0.0f;

    float local_sum0 = (p00 + p01) + (p02 + p03);
    float local_sum1 = (p10 + p11) + (p12 + p13);
    local_sum0 += __shfl_xor_sync(0xFFFFFFFFu, local_sum0, 1);
    local_sum0 += __shfl_xor_sync(0xFFFFFFFFu, local_sum0, 2);
    local_sum1 += __shfl_xor_sync(0xFFFFFFFFu, local_sum1, 1);
    local_sum1 += __shfl_xor_sync(0xFFFFFFFFu, local_sum1, 2);

    const unsigned int pf0 = hy4_pack_bf16x2(p00, p01);
    const unsigned int pf1 = hy4_pack_bf16x2(p10, p11);
    const unsigned int pf2 = hy4_pack_bf16x2(p02, p03);
    const unsigned int pf3 = hy4_pack_bf16x2(p12, p13);

#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
      accO[nt][0] *= alpha0;
      accO[nt][1] *= alpha0;
      accO[nt][2] *= alpha1;
      accO[nt][3] *= alpha1;
    }
    l_row0 = l_row0 * alpha0 + local_sum0;
    l_row1 = l_row1 * alpha1 + local_sum1;
    m_row0 = new_m0;
    m_row1 = new_m1;

#pragma unroll
    for (int dt = 0; dt < 4; ++dt) {
      unsigned int b0, b1, b2, b3;
      ptx_ldmatrix_x4_trans(b0, b1, b2, b3, &sK[buf][hy4_few_swiz576(v_ld_r0, v_ld_c8_base + dt * 2)]);
      const int nt0 = dt * 2;
      const int nt1 = dt * 2 + 1;
      ptx_mma_m16n8k16_bf16_c_v(
          accO[nt0][0], accO[nt0][1], accO[nt0][2], accO[nt0][3],
          pf0, pf1, pf2, pf3, b0, b1,
          accO[nt0][0], accO[nt0][1], accO[nt0][2], accO[nt0][3]);
      ptx_mma_m16n8k16_bf16_c_v(
          accO[nt1][0], accO[nt1][1], accO[nt1][2], accO[nt1][3],
          pf0, pf1, pf2, pf3, b2, b3,
          accO[nt1][0], accO[nt1][1], accO[nt1][2], accO[nt1][3]);
    }
    if constexpr (HAS_IDX) {
      pbuf = npbuf;
      npbuf = (npbuf == 2) ? 0 : (npbuf + 1);
    }
  }

  if (h0 < H) {
    const size_t base0 = ((static_cast<size_t>(m) * static_cast<size_t>(H) +
                           static_cast<size_t>(h0)) * static_cast<size_t>(num_splits) +
                          static_cast<size_t>(split));
    if (warp == 0 && tid_in_grp == 0) {
      partial_m[base0] = (l_row0 > 0.0f) ? (m_row0 * 0.6931471805599453f) : -1e30f;
      partial_l[base0] = l_row0;
    }
    float* dst0 = partial_o + base0 * D_V + warp * 64;
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
      *reinterpret_cast<float2*>(dst0 + nt * 8 + 2 * tid_in_grp) =
          make_float2(accO[nt][0], accO[nt][1]);
    }
  }
  if (h1 < H) {
    const size_t base1 = ((static_cast<size_t>(m) * static_cast<size_t>(H) +
                           static_cast<size_t>(h1)) * static_cast<size_t>(num_splits) +
                          static_cast<size_t>(split));
    if (warp == 0 && tid_in_grp == 0) {
      partial_m[base1] = (l_row1 > 0.0f) ? (m_row1 * 0.6931471805599453f) : -1e30f;
      partial_l[base1] = l_row1;
    }
    float* dst1 = partial_o + base1 * D_V + warp * 64;
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
      *reinterpret_cast<float2*>(dst1 + nt * 8 + 2 * tid_in_grp) =
          make_float2(accO[nt][2], accO[nt][3]);
    }
  }
}

)cuda"
R"cuda(
// ----------------------------------------------------------------------------
// Kernel 5: hy4_mla_few_split_generic (arbitrary kv_lora, rope)
//   Dynamic SMEM: kv_lora * sizeof(float) for per-warp accumulator sO.
// Grid: dim3(num_splits, H, M), Block: dim3(32, 1, 1)
// ----------------------------------------------------------------------------
template <bool HAS_IDX>
__global__ __launch_bounds__(32, 8) void hy4_mla_few_split_generic(
    const nv_bfloat16* __restrict__ q,
    const __half* __restrict__ kv_buf,
    const int* __restrict__ idx,
    float* __restrict__ partial_o,
    float* __restrict__ partial_m,
    float* __restrict__ partial_l,
    int M,
    int H,
    int kv_lora,
    int kv_row,
    int cap,
    int kmax,
    long long slot_start,
    int total_len,
    int ring_cap,
    int window,
    float scale,
    int num_splits) {
  extern __shared__ float sO[];
  const int split = static_cast<int>(blockIdx.x);
  const int h = static_cast<int>(blockIdx.y);
  const int m = static_cast<int>(blockIdx.z);
  const int lane = static_cast<int>(threadIdx.x);
  if (m >= M || h >= H) {
    return;
  }

  for (int d = lane; d < kv_lora; d += 32) {
    sO[d] = 0.0f;
  }

  const nv_bfloat16* q_row = q + (static_cast<size_t>(m) * static_cast<size_t>(H) +
                                  static_cast<size_t>(h)) * static_cast<size_t>(kv_row);
  const long long p_log = slot_start + static_cast<long long>(m);
  long long k_lo = 0;
  int n_items = 0;
  if constexpr (HAS_IDX) {
    n_items = kmax;
  } else {
    k_lo = (window > 0) ? hy4_few_max_ll(0LL, p_log + 1LL - static_cast<long long>(window)) : 0LL;
    const long long k_hi = hy4_few_min_ll(p_log, static_cast<long long>(total_len) - 1LL);
    n_items = (k_hi >= k_lo) ? static_cast<int>(k_hi - k_lo + 1LL) : 0;
  }
  const int per_split = (n_items + num_splits - 1) / num_splits;
  const int i_start = split * per_split;
  const int i_end = min(i_start + per_split, n_items);
  const int* idx_row = HAS_IDX ? (idx + static_cast<size_t>(m) * static_cast<size_t>(kmax)) : nullptr;

  float m_val = -1e30f;
  float l_val = 0.0f;

  for (int it = i_start; it < i_end; ++it) {
    int phys = -1;
    if constexpr (HAS_IDX) {
      const int kp = idx_row[it];
      if (kp >= 0) {
        const int pr = (ring_cap > 0) ? (kp % ring_cap) : kp;
        if (pr >= 0 && pr < cap) phys = pr;
      }
    } else {
      const long long kp = k_lo + static_cast<long long>(it);
      const int pr = (ring_cap > 0)
          ? static_cast<int>(kp % static_cast<long long>(ring_cap))
          : static_cast<int>(kp);
      if (pr >= 0 && pr < cap) phys = pr;
    }
    if (phys < 0) continue;

    const __half* k_ptr = kv_buf + static_cast<size_t>(phys) * static_cast<size_t>(kv_row);
    float dot = 0.0f;
    for (int d = lane; d < kv_row; d += 32) {
      dot = fmaf(__bfloat162float(q_row[d]), __half2float(k_ptr[d]), dot);
    }
    const float sc = hy4_warp_sum(dot) * scale;

    if (l_val == 0.0f) {
      m_val = sc;
      l_val = 1.0f;
      for (int d = lane; d < kv_lora; d += 32) {
        sO[d] = __half2float(k_ptr[d]);
      }
    } else if (sc > m_val) {
      const float fac = ptx_ex2_approx_ftz((m_val - sc) * 1.4426950408889634f);
      m_val = sc;
      l_val = fmaf(l_val, fac, 1.0f);
      for (int d = lane; d < kv_lora; d += 32) {
        sO[d] = fmaf(sO[d], fac, __half2float(k_ptr[d]));
      }
    } else {
      const float pr = ptx_ex2_approx_ftz((sc - m_val) * 1.4426950408889634f);
      l_val += pr;
      for (int d = lane; d < kv_lora; d += 32) {
        sO[d] = fmaf(pr, __half2float(k_ptr[d]), sO[d]);
      }
    }
  }

  const size_t base = ((static_cast<size_t>(m) * static_cast<size_t>(H) +
                        static_cast<size_t>(h)) * static_cast<size_t>(num_splits) +
                       static_cast<size_t>(split));
  if (lane == 0) {
    partial_m[base] = m_val;
    partial_l[base] = l_val;
  }
  float* dst = partial_o + base * static_cast<size_t>(kv_lora);
  for (int d = lane; d < kv_lora; d += 32) {
    dst[d] = sO[d];
  }
}

// ----------------------------------------------------------------------------
// Kernel 6: hy4_mla_few_reduce
//   Merges FP32 unnormalized partial_o[M, H, num_splits, kv_lora] with exact
//   overflow-safe sink softmax:
//     m_final = max(sinks[h], max_{s: l_s > 0} partial_m[s])
//     den     = exp(sinks[h] - m_final) + sum_s partial_l[s] * exp(partial_m[s] - m_final)
//     out     = bf16((sum_s exp(partial_m[s] - m_final) * partial_o[s]) / den)
// Grid: dim3(H, M, 1), Block: dim3(256, 1, 1)
// ----------------------------------------------------------------------------
__global__ __launch_bounds__(256, 4) void hy4_mla_few_reduce(
    const float* __restrict__ partial_o,
    const float* __restrict__ partial_m,
    const float* __restrict__ partial_l,
    const float* __restrict__ sinks,
    nv_bfloat16* __restrict__ out,
    int M,
    int H,
    int kv_lora,
    int num_splits) {
  const int h = static_cast<int>(blockIdx.x);
  const int m = static_cast<int>(blockIdx.y);
  const int tid = static_cast<int>(threadIdx.x);
  if (m >= M || h >= H) {
    return;
  }

  const size_t mh = static_cast<size_t>(m) * static_cast<size_t>(H) + static_cast<size_t>(h);
  const size_t sp_base = mh * static_cast<size_t>(num_splits);

  float g_max = -1e30f;
  bool any = false;
  for (int s = 0; s < num_splits; ++s) {
    if (partial_l[sp_base + s] > 0.0f) {
      const float ms = partial_m[sp_base + s];
      g_max = any ? fmaxf(g_max, ms) : ms;
      any = true;
    }
  }

  nv_bfloat16* dst = out + mh * static_cast<size_t>(kv_lora);
  if (!any) {
    for (int d = tid; d < kv_lora; d += 256) {
      dst[d] = __float2bfloat16(0.0f);
    }
    return;
  }

  const float sink_h = sinks[h];
  const float m_final = fmaxf(g_max, sink_h);
  __shared__ float s_w[32];
  if (num_splits <= 32 && tid < num_splits) {
    const float ls = partial_l[sp_base + tid];
    s_w[tid] = (ls > 0.0f)
        ? ptx_ex2_approx_ftz((partial_m[sp_base + tid] - m_final) * 1.4426950408889634f)
        : 0.0f;
  }
  __syncthreads();

  float total_l = ptx_ex2_approx_ftz((sink_h - m_final) * 1.4426950408889634f);
  for (int s = 0; s < num_splits; ++s) {
    const float ls = partial_l[sp_base + s];
    if (ls > 0.0f) {
      const float w = (num_splits <= 32)
          ? s_w[s]
          : ptx_ex2_approx_ftz((partial_m[sp_base + s] - m_final) * 1.4426950408889634f);
      total_l = fmaf(ls, w, total_l);
    }
  }
  const float inv_den = (total_l > 0.0f) ? (1.0f / total_l) : 0.0f;

  if ((kv_lora & 1) == 0) {
    const int n2 = kv_lora >> 1;
    unsigned int* dst2 = reinterpret_cast<unsigned int*>(dst);
    for (int p = tid; p < n2; p += 256) {
      float a0 = 0.0f;
      float a1 = 0.0f;
      for (int s = 0; s < num_splits; ++s) {
        const float w = (num_splits <= 32)
            ? s_w[s]
            : ((partial_l[sp_base + s] > 0.0f)
                   ? ptx_ex2_approx_ftz((partial_m[sp_base + s] - m_final) * 1.4426950408889634f)
                   : 0.0f);
        if (w > 0.0f) {
          const float2 ov = reinterpret_cast<const float2*>(
              partial_o + (sp_base + s) * static_cast<size_t>(kv_lora))[p];
          a0 = fmaf(w, ov.x, a0);
          a1 = fmaf(w, ov.y, a1);
        }
      }
      dst2[p] = hy4_pack_bf16x2(a0 * inv_den, a1 * inv_den);
    }
  } else {
    for (int d = tid; d < kv_lora; d += 256) {
      float a = 0.0f;
      for (int s = 0; s < num_splits; ++s) {
        const float w = (num_splits <= 32)
            ? s_w[s]
            : ((partial_l[sp_base + s] > 0.0f)
                   ? ptx_ex2_approx_ftz((partial_m[sp_base + s] - m_final) * 1.4426950408889634f)
                   : 0.0f);
        if (w > 0.0f) {
          a = fmaf(w, partial_o[(sp_base + s) * static_cast<size_t>(kv_lora) + d], a);
        }
      }
      dst[d] = __float2bfloat16(a * inv_den);
    }
  }
}
)cuda"
