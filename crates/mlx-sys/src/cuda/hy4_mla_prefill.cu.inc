R"cuda(
// ============================================================================
// Hy4-Flash Absorbed MLA Prefill Kernels (M >= 17):
//   1. hy4_mla_prefill_tc_512_64  (BF16 TC MMA m16n8k16 for D_qk=576, D_v=512)
//   2. hy4_mla_prefill_generic    (Exact FP32 fallback for arbitrary dims)
//
// Design of hy4_mla_prefill_tc_512_64:
//   - CTA owns BH=16 heads of 1 query token m (contiguous in [M, H, 576]!),
//     sharing the EXACT SAME sparse idx[m] / causal / SWA key stream across all
//     16 heads with zero triangle waste and zero sparse union overhead.
//   - 8 warps (256 threads) split D_v=512 into 64 dims/warp:
//       * accO[8][4] = 32 FP32 regs/thread (avoids 576-whole-tile spills!).
//       * q_reg[5][4] = 20 u32 regs/thread (warps 0..7 hold latent 64 dims each;
//         warps 0..3 also hold 16 RoPE dims each -> full 576-dim dot product).
//   - Single shared buffer sK[16][576] (BF16, 18 KB, 128B XOR swizzled) serves
//     BOTH K (0..575) and V (0..511); sS[8][16][16] (8 KB) reduces partial dots.
//   - Total static SMEM = 26.7 KB -> 3 CTAs/SM, 0 B register spills.
// ============================================================================

__device__ __forceinline__ long long hy4_pre_max_ll(long long a, long long b) {
  return (a > b) ? a : b;
}

__device__ __forceinline__ long long hy4_pre_min_ll(long long a, long long b) {
  return (a < b) ? a : b;
}

__device__ __forceinline__ int hy4_swiz576(int row, int c8) {
  return row * 576 + ((c8 ^ (row & 7)) << 3);
}

__device__ __forceinline__ uint4 hy4_f16x8_to_bf16x8(const uint4& hv) {
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
__global__ __launch_bounds__(256, 2) void hy4_mla_prefill_tc_512_64(
    const nv_bfloat16* __restrict__ q,
    const __half* __restrict__ kv_buf,
    const float* __restrict__ sinks,
    const int* __restrict__ idx,
    nv_bfloat16* __restrict__ out,
    int M,
    int H,
    int cap,
    int kmax,
    long long slot_start,
    int total_len,
    int ring_cap,
    int window,
    float scale) {
  constexpr int D_V = 512;
  constexpr int D_QK = 576;
  constexpr int BK = 16;

  const int h_base = static_cast<int>(blockIdx.x) * 16;
  const int m = static_cast<int>(blockIdx.y);
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

  // Load warp's slice of Q[m, h_base..h_base+15, :] into registers
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
    k_lo = (window > 0) ? hy4_pre_max_ll(0LL, p_log + 1LL - static_cast<long long>(window)) : 0LL;
    const long long k_hi_cap = (ring_cap > 0)
        ? (static_cast<long long>(total_len) - 1LL)
        : hy4_pre_min_ll(static_cast<long long>(total_len) - 1LL, static_cast<long long>(cap) - 1LL);
    const long long k_hi = hy4_pre_min_ll(p_log, k_hi_cap);
    n_items = (k_hi >= k_lo) ? static_cast<int>(k_hi - k_lo + 1LL) : 0;
  }
  const int n_tiles = (n_items + BK - 1) / BK;
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
    if (n_tiles > 0) {
      if (warp == 0) {
        const int item = lane;
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

  for (int ti = 0; ti < n_tiles; ++ti) {
    const int item_base = ti * BK;
    const int buf = ti & 1;

    if constexpr (HAS_IDX) {
      if (s_any_valid[pbuf] == 0) {
        if (ti + 1 < n_tiles && warp == 0) {
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

    // Load BK=16 f16 KV rows -> BF16 XOR-swizzled sK[buf] (16 * 72 = 1152 uint4s)
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
        bfv = hy4_f16x8_to_bf16x8(hv);
      }
      *reinterpret_cast<uint4*>(&sK[buf][hy4_swiz576(r, c8)]) = bfv;
    }
    __syncthreads();

)cuda"
R"cuda(
    // Q @ K^T partial dot products (each warp accumulates its own D-slice)
    float s_acc[2][4] = {};
#pragma unroll
    for (int kt = 0; kt < 4; ++kt) {
      const int c8 = warp * 8 + kt * 2 + k_ld_c8_base;
      unsigned int b0, b1, b2, b3;
      ptx_ldmatrix_x4(b0, b1, b2, b3, &sK[buf][hy4_swiz576(k_ld_r0, c8)]);
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
      ptx_ldmatrix_x4(b0, b1, b2, b3, &sK[buf][hy4_swiz576(k_ld_r0, c8)]);
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
      if (ti + 1 < n_tiles && warp == 0) {
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

    // Reduce partial QK^T across all 8 warps + mask + online softmax
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

    // P @ V on warp's 64-dim slice of sK[buf] (V = K[:, :512])
#pragma unroll
    for (int dt = 0; dt < 4; ++dt) {
      unsigned int b0, b1, b2, b3;
      ptx_ldmatrix_x4_trans(b0, b1, b2, b3, &sK[buf][hy4_swiz576(v_ld_r0, v_ld_c8_base + dt * 2)]);
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

  // Exact overflow-safe sink softmax epilogue -> bf16 [M, H, 512]
  nv_bfloat16* o_m_base = out + static_cast<size_t>(m) * static_cast<size_t>(H) * D_V;
  if (h0 < H) {
    float scale0 = 0.0f;
    if (l_row0 > 0.0f) {
      const float sk0 = sinks[h0] * 1.4426950408889634f;
      const float mf0 = fmaxf(m_row0, sk0);
      const float ks0 = ptx_ex2_approx_ftz(m_row0 - mf0);
      const float den0 = fmaf(l_row0, ks0, ptx_ex2_approx_ftz(sk0 - mf0));
      scale0 = (den0 > 0.0f) ? (ks0 / den0) : 0.0f;
    }
    nv_bfloat16* dst0 = o_m_base + static_cast<size_t>(h0) * D_V + warp * 64;
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
      *reinterpret_cast<unsigned int*>(dst0 + nt * 8 + 2 * tid_in_grp) =
          hy4_pack_bf16x2(accO[nt][0] * scale0, accO[nt][1] * scale0);
    }
  }
  if (h1 < H) {
    float scale1 = 0.0f;
    if (l_row1 > 0.0f) {
      const float sk1 = sinks[h1] * 1.4426950408889634f;
      const float mf1 = fmaxf(m_row1, sk1);
      const float ks1 = ptx_ex2_approx_ftz(m_row1 - mf1);
      const float den1 = fmaf(l_row1, ks1, ptx_ex2_approx_ftz(sk1 - mf1));
      scale1 = (den1 > 0.0f) ? (ks1 / den1) : 0.0f;
    }
    nv_bfloat16* dst1 = o_m_base + static_cast<size_t>(h1) * D_V + warp * 64;
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
      *reinterpret_cast<unsigned int*>(dst1 + nt * 8 + 2 * tid_in_grp) =
          hy4_pack_bf16x2(accO[nt][2] * scale1, accO[nt][3] * scale1);
    }
  }
}

)cuda"
R"cuda(
// ----------------------------------------------------------------------------
// Kernel 1b: hy4_mla_prefill_swa_q16_512_64 (Q-Tiled Sliding-Window / Causal)
//   CTA owns BQ=16 consecutive query tokens of 1 head h, sharing the KV stream
//   across all 16 tokens in registers/SMEM AND across all H heads in L2!
//   Double-buffered sK[2][16*576] (36 KB) + sS[8][16][16] (8 KB) = 44 KB SMEM
//   (2 CTAs/SM, 2 barriers/tile, fast ex2.approx.ftz, 0 B register spills).
// Grid: dim3(H, (M + 15) / 16, 1), Block: dim3(32, 8, 1)
// ----------------------------------------------------------------------------
__global__ __launch_bounds__(256, 2) void hy4_mla_prefill_swa_q16_512_64(
    const nv_bfloat16* __restrict__ q,
    const __half* __restrict__ kv_buf,
    const float* __restrict__ sinks,
    nv_bfloat16* __restrict__ out,
    int M,
    int H,
    int cap,
    long long slot_start,
    int total_len,
    int ring_cap,
    int window,
    float scale) {
  constexpr int D_V = 512;
  constexpr int D_QK = 576;
  constexpr int BK = 16;

  const int h = static_cast<int>(blockIdx.x);
  const int m0 = static_cast<int>(blockIdx.y) * 16;
  const int lane = static_cast<int>(threadIdx.x);
  const int warp = static_cast<int>(threadIdx.y);
  const int tid = warp * 32 + lane;
  if (h >= H || m0 >= M) {
    return;
  }

  const int group_id = lane >> 2;
  const int tid_in_grp = lane & 3;
  const int row0 = group_id;
  const int row1 = group_id + 8;
  const int m_r0 = m0 + row0;
  const int m_r1 = m0 + row1;

  const size_t q_row_stride = static_cast<size_t>(H) * D_QK;
  const nv_bfloat16* q_ptr0 = (m_r0 < M)
      ? (q + static_cast<size_t>(m_r0) * q_row_stride + static_cast<size_t>(h) * D_QK)
      : nullptr;
  const nv_bfloat16* q_ptr1 = (m_r1 < M)
      ? (q + static_cast<size_t>(m_r1) * q_row_stride + static_cast<size_t>(h) * D_QK)
      : nullptr;

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

  const long long p0 = slot_start + static_cast<long long>(m0);
  const int m_last = min(M - 1, m0 + 15);
  const long long p_last = slot_start + static_cast<long long>(m_last);
  const long long k_lo = (window > 0)
      ? hy4_pre_max_ll(0LL, p0 + 1LL - static_cast<long long>(window))
      : 0LL;
  const long long k_hi_cap = (ring_cap > 0)
      ? (static_cast<long long>(total_len) - 1LL)
      : hy4_pre_min_ll(static_cast<long long>(total_len) - 1LL, static_cast<long long>(cap) - 1LL);
  const long long k_hi = hy4_pre_min_ll(p_last, k_hi_cap);
  const int n_items = (k_hi >= k_lo) ? static_cast<int>(k_hi - k_lo + 1LL) : 0;
  const int n_tiles = (n_items + BK - 1) / BK;

  const long long p_row0 = slot_start + static_cast<long long>(m_r0);
  const long long p_row1 = slot_start + static_cast<long long>(m_r1);
  const int rel_lo0 = (window > 0)
      ? static_cast<int>(hy4_pre_min_ll(
            static_cast<long long>(n_items),
            hy4_pre_max_ll(0LL, p_row0 + 1LL - static_cast<long long>(window) - k_lo)))
      : 0;
  const int rel_lo1 = (window > 0)
      ? static_cast<int>(hy4_pre_min_ll(
            static_cast<long long>(n_items),
            hy4_pre_max_ll(0LL, p_row1 + 1LL - static_cast<long long>(window) - k_lo)))
      : 0;
  const int rel_hi0 = (m_r0 < M)
      ? static_cast<int>(hy4_pre_max_ll(-1LL, hy4_pre_min_ll(static_cast<long long>(n_items - 1), p_row0 - k_lo)))
      : -1;
  const int rel_hi1 = (m_r1 < M)
      ? static_cast<int>(hy4_pre_max_ll(-1LL, hy4_pre_min_ll(static_cast<long long>(n_items - 1), p_row1 - k_lo)))
      : -1;
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

  for (int ti = 0; ti < n_tiles; ++ti) {
    const int item_base = ti * BK;
    const int buf = ti & 1;

    for (int i = tid; i < BK * 72; i += 256) {
      const int r = i / 72;
      const int c8 = i - r * 72;
      const int item = item_base + r;
      uint4 bfv = make_uint4(0u, 0u, 0u, 0u);
      if (item < n_items) {
        const long long kp = k_lo + static_cast<long long>(item);
        const int phys = (ring_cap > 0)
            ? static_cast<int>(kp % static_cast<long long>(ring_cap))
            : static_cast<int>(kp);
        if (phys >= 0 && phys < cap) {
          const uint4 hv = __ldg(reinterpret_cast<const uint4*>(
              kv_buf + static_cast<size_t>(phys) * D_QK + c8 * 8));
          bfv = hy4_f16x8_to_bf16x8(hv);
        }
      }
      *reinterpret_cast<uint4*>(&sK[buf][hy4_swiz576(r, c8)]) = bfv;
    }
    __syncthreads();
)cuda"
R"cuda(
    float s_acc[2][4] = {};
#pragma unroll
    for (int kt = 0; kt < 4; ++kt) {
      const int c8 = warp * 8 + kt * 2 + k_ld_c8_base;
      unsigned int b0, b1, b2, b3;
      ptx_ldmatrix_x4(b0, b1, b2, b3, &sK[buf][hy4_swiz576(k_ld_r0, c8)]);
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
      ptx_ldmatrix_x4(b0, b1, b2, b3, &sK[buf][hy4_swiz576(k_ld_r0, c8)]);
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
    __syncthreads();

    float local_max0 = -3.4028234663852886e38f;
    float local_max1 = -3.4028234663852886e38f;
    bool v_r0[4];
    bool v_r1[4];
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
      const int c0 = nt * 8 + 2 * tid_in_grp;
      const int c1 = c0 + 1;
      const int it0 = item_base + c0;
      const int it1 = item_base + c1;
      const bool v00 = (it0 >= rel_lo0) && (it0 <= rel_hi0);
      const bool v01 = (it1 >= rel_lo0) && (it1 <= rel_hi0);
      const bool v10 = (it0 >= rel_lo1) && (it0 <= rel_hi1);
      const bool v11 = (it1 >= rel_lo1) && (it1 <= rel_hi1);
      v_r0[nt * 2] = v00;
      v_r0[nt * 2 + 1] = v01;
      v_r1[nt * 2] = v10;
      v_r1[nt * 2 + 1] = v11;
      float r00 = 0.0f, r01 = 0.0f, r10 = 0.0f, r11 = 0.0f;
#pragma unroll
      for (int w = 0; w < 8; ++w) {
        r00 += sS[w][row0][c0];
        r01 += sS[w][row0][c1];
        r10 += sS[w][row1][c0];
        r11 += sS[w][row1][c1];
      }
      s_acc[nt][0] = v00 ? (r00 * scale_log2e) : -1e30f;
      s_acc[nt][1] = v01 ? (r01 * scale_log2e) : -1e30f;
      s_acc[nt][2] = v10 ? (r10 * scale_log2e) : -1e30f;
      s_acc[nt][3] = v11 ? (r11 * scale_log2e) : -1e30f;
      if (v00) local_max0 = fmaxf(local_max0, s_acc[nt][0]);
      if (v01) local_max0 = fmaxf(local_max0, s_acc[nt][1]);
      if (v10) local_max1 = fmaxf(local_max1, s_acc[nt][2]);
      if (v11) local_max1 = fmaxf(local_max1, s_acc[nt][3]);
    }
    local_max0 = fmaxf(local_max0, __shfl_xor_sync(0xFFFFFFFFu, local_max0, 1));
    local_max0 = fmaxf(local_max0, __shfl_xor_sync(0xFFFFFFFFu, local_max0, 2));
    local_max1 = fmaxf(local_max1, __shfl_xor_sync(0xFFFFFFFFu, local_max1, 1));
    local_max1 = fmaxf(local_max1, __shfl_xor_sync(0xFFFFFFFFu, local_max1, 2));

    const bool has_v0 = (local_max0 > -3.0e38f);
    const bool has_v1 = (local_max1 > -3.0e38f);
    const float new_m0 = has_v0
        ? ((l_row0 == 0.0f) ? local_max0 : fmaxf(m_row0, local_max0))
        : m_row0;
    const float new_m1 = has_v1
        ? ((l_row1 == 0.0f) ? local_max1 : fmaxf(m_row1, local_max1))
        : m_row1;
    const float alpha0 = (l_row0 == 0.0f) ? 0.0f
        : (has_v0 ? ptx_ex2_approx_ftz(m_row0 - new_m0) : 1.0f);
    const float alpha1 = (l_row1 == 0.0f) ? 0.0f
        : (has_v1 ? ptx_ex2_approx_ftz(m_row1 - new_m1) : 1.0f);

    const float p00 = v_r0[0] ? ptx_ex2_approx_ftz(s_acc[0][0] - new_m0) : 0.0f;
    const float p01 = v_r0[1] ? ptx_ex2_approx_ftz(s_acc[0][1] - new_m0) : 0.0f;
    const float p10 = v_r1[0] ? ptx_ex2_approx_ftz(s_acc[0][2] - new_m1) : 0.0f;
    const float p11 = v_r1[1] ? ptx_ex2_approx_ftz(s_acc[0][3] - new_m1) : 0.0f;
    const float p02 = v_r0[2] ? ptx_ex2_approx_ftz(s_acc[1][0] - new_m0) : 0.0f;
    const float p03 = v_r0[3] ? ptx_ex2_approx_ftz(s_acc[1][1] - new_m0) : 0.0f;
    const float p12 = v_r1[2] ? ptx_ex2_approx_ftz(s_acc[1][2] - new_m1) : 0.0f;
    const float p13 = v_r1[3] ? ptx_ex2_approx_ftz(s_acc[1][3] - new_m1) : 0.0f;

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
      ptx_ldmatrix_x4_trans(b0, b1, b2, b3, &sK[buf][hy4_swiz576(v_ld_r0, v_ld_c8_base + dt * 2)]);
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
  }

  // Exact overflow-safe sink softmax epilogue -> bf16 [M, H, 512] (shared sinks[h])
  const float sk_log2 = sinks[h] * 1.4426950408889634f;
  const size_t o_row_stride = static_cast<size_t>(H) * D_V;
  if (m_r0 < M) {
    float scale0 = 0.0f;
    if (l_row0 > 0.0f) {
      const float mf0 = fmaxf(m_row0, sk_log2);
      const float ks0 = ptx_ex2_approx_ftz(m_row0 - mf0);
      const float den0 = fmaf(l_row0, ks0, ptx_ex2_approx_ftz(sk_log2 - mf0));
      scale0 = (den0 > 0.0f) ? (ks0 / den0) : 0.0f;
    }
    nv_bfloat16* dst0 = out + static_cast<size_t>(m_r0) * o_row_stride +
        static_cast<size_t>(h) * D_V + warp * 64;
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
      *reinterpret_cast<unsigned int*>(dst0 + nt * 8 + 2 * tid_in_grp) =
          hy4_pack_bf16x2(accO[nt][0] * scale0, accO[nt][1] * scale0);
    }
  }
  if (m_r1 < M) {
    float scale1 = 0.0f;
    if (l_row1 > 0.0f) {
      const float mf1 = fmaxf(m_row1, sk_log2);
      const float ks1 = ptx_ex2_approx_ftz(m_row1 - mf1);
      const float den1 = fmaf(l_row1, ks1, ptx_ex2_approx_ftz(sk_log2 - mf1));
      scale1 = (den1 > 0.0f) ? (ks1 / den1) : 0.0f;
    }
    nv_bfloat16* dst1 = out + static_cast<size_t>(m_r1) * o_row_stride +
        static_cast<size_t>(h) * D_V + warp * 64;
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
      *reinterpret_cast<unsigned int*>(dst1 + nt * 8 + 2 * tid_in_grp) =
          hy4_pack_bf16x2(accO[nt][2] * scale1, accO[nt][3] * scale1);
    }
  }
}

)cuda"
R"cuda(
// ----------------------------------------------------------------------------
// Kernel 2: hy4_mla_prefill_generic (Exact FP32 fallback for arbitrary dims)
//   Dynamic SMEM: kv_lora * sizeof(float)
// Grid: dim3(H, M, 1), Block: dim3(32, 1, 1)
// ----------------------------------------------------------------------------
template <bool HAS_IDX>
__global__ __launch_bounds__(32, 8) void hy4_mla_prefill_generic(
    const nv_bfloat16* __restrict__ q,
    const __half* __restrict__ kv_buf,
    const float* __restrict__ sinks,
    const int* __restrict__ idx,
    nv_bfloat16* __restrict__ out,
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
    float scale) {
  extern __shared__ float sO[];
  const int h = static_cast<int>(blockIdx.x);
  const int m = static_cast<int>(blockIdx.y);
  const int lane = static_cast<int>(threadIdx.x);
  if (m >= M || h >= H) {
    return;
  }

  for (int d = lane; d < kv_lora; d += 32) {
    sO[d] = 0.0f;
  }

  const size_t mh = static_cast<size_t>(m) * static_cast<size_t>(H) + static_cast<size_t>(h);
  const nv_bfloat16* q_row = q + mh * static_cast<size_t>(kv_row);
  const long long p_log = slot_start + static_cast<long long>(m);
  long long k_lo = 0;
  int n_items = 0;
  if constexpr (HAS_IDX) {
    n_items = kmax;
  } else {
    k_lo = (window > 0) ? hy4_pre_max_ll(0LL, p_log + 1LL - static_cast<long long>(window)) : 0LL;
    const long long k_hi = hy4_pre_min_ll(p_log, static_cast<long long>(total_len) - 1LL);
    n_items = (k_hi >= k_lo) ? static_cast<int>(k_hi - k_lo + 1LL) : 0;
  }
  const int* idx_row = HAS_IDX ? (idx + static_cast<size_t>(m) * static_cast<size_t>(kmax)) : nullptr;

  float m_val = -1e30f;
  float l_val = 0.0f;

  for (int it = 0; it < n_items; ++it) {
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

  float scale_o = 0.0f;
  if (l_val > 0.0f) {
    const float sk = sinks[h];
    const float mf = fmaxf(m_val, sk);
    const float ks = ptx_ex2_approx_ftz((m_val - mf) * 1.4426950408889634f);
    const float den = fmaf(l_val, ks, ptx_ex2_approx_ftz((sk - mf) * 1.4426950408889634f));
    scale_o = (den > 0.0f) ? (ks / den) : 0.0f;
  }
  nv_bfloat16* dst = out + mh * static_cast<size_t>(kv_lora);
  for (int d = lane; d < kv_lora; d += 32) {
    dst[d] = __float2bfloat16(sO[d] * scale_o);
  }
}
)cuda"
