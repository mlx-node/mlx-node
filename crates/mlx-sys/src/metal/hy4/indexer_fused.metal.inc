R"METAL(
inline void hy4_idx_m3_mma(
    thread float2& acc,
    const half2 a_val,
    thread const simdgroup_half8x8& b_mat) {
  simdgroup_half8x8 a_mat;
  simdgroup_float8x8 c_mat;
  a_mat.thread_elements()[0] = a_val.x;
  a_mat.thread_elements()[1] = a_val.y;
  c_mat.thread_elements()[0] = acc.x;
  c_mat.thread_elements()[1] = acc.y;
  simdgroup_multiply_accumulate(c_mat, a_mat, b_mat, c_mat);
  acc.x = c_mat.thread_elements()[0];
  acc.y = c_mat.thread_elements()[1];
}

template <
    bool PAGED,
    int INDEX_ROWS,
    bool SHARE_S,
    typename QPtr,
    typename WPtr,
    typename KPtr,
    typename PagePtr,
    typename ParamsPtr>
inline void hy4_indexer_score_decode_impl(
    QPtr q,
    WPtr w,
    KPtr k_pool,
    PagePtr page_ids,
    ParamsPtr params,
    device float* out,
    threadgroup half* k_tile,
    int H,
    uint s_grid,
    uint g_tile,
    uint lane) {
  constexpr uint PAD = 136u;

  const uint S = params[0];
  const uint N_g = params[1];
  const uint start_g = params[2];
  const uint g0 = g_tile * 32u;
  if ((!SHARE_S && s_grid >= S) || g0 >= N_g) {
    return;
  }

  // Each half-simdgroup (16 lanes) loads one 256-byte key row via float4 (half8).
  const uint lane_r = lane >> 4;
  const uint c8 = lane & 15u;
  #pragma unroll
  for (uint rb = 0u; rb < 32u; rb += 2u) {
    const uint r = rb + lane_r;
    const uint g = g0 + r;
    threadgroup float4* dst =
        reinterpret_cast<threadgroup float4*>(k_tile + r * PAD + c8 * 8u);
    if (g < N_g) {
      const uint g_abs = start_g + g;
      size_t k_row;
      if (PAGED) {
        const uint page_idx = g_abs / uint(INDEX_ROWS);
        const uint local_row = g_abs - page_idx * uint(INDEX_ROWS);
        const int phys_page = page_ids[page_idx];
        k_row = size_t(phys_page) * size_t(INDEX_ROWS) + size_t(local_row);
      } else {
        k_row = size_t(g_abs);
      }
      *dst = *reinterpret_cast<const device float4*>(
          k_pool + k_row * 128u + size_t(c8) * 8u);
    } else {
      *dst = float4(0.0f);
    }
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);

  const uint qid = lane >> 2;
  const uint fm = (qid & 4u) + ((lane >> 1) & 3u);
  const uint fn = ((qid & 2u) << 1) + ((lane & 1u) << 1);
  const bool ok0 = int(fm) < H;
  const bool ok1 = (int(fm) + 8) < H;
  const bool has_hi_heads = (H > 8);

  const uint s_lo = SHARE_S ? 0u : s_grid;
  const uint s_hi = SHARE_S ? S : (s_grid + 1u);

  for (uint s = s_lo; s < s_hi; ++s) {
    const device float* q_s = q + size_t(s) * size_t(H) * 128u;
    float2 dot0[4];
    float2 dot1[4];
    #pragma unroll
    for (uint kg = 0u; kg < 4u; ++kg) {
      dot0[kg] = float2(0.0f);
      dot1[kg] = float2(0.0f);
    }

    // Split F32 Q into FP16 hi + residual to preserve <= 1e-3 accuracy across D=128.
    #pragma unroll
    for (uint d = 0u; d < 128u; d += 8u) {
      const float2 q0_f = ok0
          ? float2(
                q_s[size_t(fm) * 128u + size_t(d + fn)],
                q_s[size_t(fm) * 128u + size_t(d + fn + 1u)])
          : float2(0.0f);
      const half2 q0_hi = half2(q0_f);
      const half2 q0_lo = half2(q0_f - float2(q0_hi));

      half2 q1_hi = half2(0.0h);
      half2 q1_lo = half2(0.0h);
      if (has_hi_heads) {
        const float2 q1_f = ok1
            ? float2(
                  q_s[size_t(fm + 8u) * 128u + size_t(d + fn)],
                  q_s[size_t(fm + 8u) * 128u + size_t(d + fn + 1u)])
            : float2(0.0f);
        q1_hi = half2(q1_f);
        q1_lo = half2(q1_f - float2(q1_hi));
      }

      #pragma unroll
      for (uint kg = 0u; kg < 4u; ++kg) {
        simdgroup_half8x8 mk;
        simdgroup_load(mk, k_tile + (kg * 8u) * PAD + d, PAD, ulong2(0), true);
        hy4_idx_m3_mma(dot0[kg], q0_hi, mk);
        hy4_idx_m3_mma(dot0[kg], q0_lo, mk);
        if (has_hi_heads) {
          hy4_idx_m3_mma(dot1[kg], q1_hi, mk);
          hy4_idx_m3_mma(dot1[kg], q1_lo, mk);
        }
      }
    }

    const float w0 = ok0 ? w[size_t(s) * size_t(H) + size_t(fm)] : 0.0f;
    const float w1 = ok1 ? w[size_t(s) * size_t(H) + size_t(fm + 8u)] : 0.0f;
    device float* out_row = out + size_t(s) * size_t(N_g);

    #pragma unroll
    for (uint kg = 0u; kg < 4u; ++kg) {
      float2 sk(
          metal::fma(
              w0,
              metal::max(dot0[kg].x, 0.0f),
              w1 * metal::max(dot1[kg].x, 0.0f)),
          metal::fma(
              w0,
              metal::max(dot0[kg].y, 0.0f),
              w1 * metal::max(dot1[kg].y, 0.0f)));
      // Lanes sharing the same fn differ only in bits 1, 2, 4 (fm = [b4 b2 b1]).
      sk += simd_shuffle_xor(sk, ushort(2));
      sk += simd_shuffle_xor(sk, ushort(4));
      sk += simd_shuffle_xor(sk, ushort(16));
      if (fm == 0u) {
        const uint gc = g0 + kg * 8u + fn;
        if (gc < N_g) {
          out_row[gc] = sk.x;
        }
        if (gc + 1u < N_g) {
          out_row[gc + 1u] = sk.y;
        }
      }
    }
  }
}

template <
    bool PAGED,
    int INDEX_ROWS,
    bool HAS_VIS,
    typename QPtr,
    typename WPtr,
    typename KPtr,
    typename PagePtr,
    typename VisPtr,
    typename ParamsPtr>
inline void hy4_indexer_score_prefill_impl(
    QPtr q,
    WPtr w,
    KPtr k_pool,
    PagePtr page_ids,
    VisPtr visible_groups,
    ParamsPtr params,
    device float* out,
    threadgroup half* k_tile,
    int H,
    uint g_tile,
    uint s_tile,
    uint tid,
    uint lane,
    uint sg) {
  constexpr uint PAD = 136u;

  const uint S = params[0];
  const uint N_g = params[1];
  const uint start_g = params[2];
  const uint s0 = s_tile * 64u;
  const uint g0 = g_tile * 32u;
  if (s0 >= S || g0 >= N_g) {
    return;
  }

  // Every SG computes the identical tile-max visibility via one simd_max (no barrier).
  uint vis_max = N_g;
  if (HAS_VIS) {
    const uint r0 = s0 + lane;
    const uint r1 = s0 + 32u + lane;
    const int m0 = (r0 < S) ? visible_groups[r0] : 0;
    const int m1 = (r1 < S) ? visible_groups[r1] : 0;
    const int mv = simd_max(metal::max(m0, m1));
    vis_max = (mv > 0) ? metal::min(uint(mv), N_g) : 0u;
    if (g0 >= vis_max) {
      return;
    }
  }

  // 128 threads cooperatively load 32 x 128 halves (512 float4s = 4 per thread).
  for (uint i = tid; i < 32u * 16u; i += 128u) {
    const uint r = i >> 4;
    const uint c8 = i & 15u;
    const uint g = g0 + r;
    threadgroup float4* dst =
        reinterpret_cast<threadgroup float4*>(k_tile + r * PAD + c8 * 8u);
    if (g < vis_max) {
      const uint g_abs = start_g + g;
      size_t k_row;
      if (PAGED) {
        const uint page_idx = g_abs / uint(INDEX_ROWS);
        const uint local_row = g_abs - page_idx * uint(INDEX_ROWS);
        const int phys_page = page_ids[page_idx];
        k_row = size_t(phys_page) * size_t(INDEX_ROWS) + size_t(local_row);
      } else {
        k_row = size_t(g_abs);
      }
      *dst = *reinterpret_cast<const device float4*>(
          k_pool + k_row * 128u + size_t(c8) * 8u);
    } else {
      *dst = float4(0.0f);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint s_sg = s0 + sg * 16u;
  if (s_sg >= S) {
    return;
  }

  const uint qid = lane >> 2;
  const uint fm = (qid & 4u) + ((lane >> 1) & 3u);
  const uint fn = ((qid & 2u) << 1) + ((lane & 1u) << 1);
  const uint sr0 = s_sg + fm;
  const uint sr1 = sr0 + 8u;
  const bool ok0 = (sr0 < S);
  const bool ok1 = (sr1 < S);

  float2 acc0[4];
  float2 acc1[4];
  #pragma unroll
  for (uint kg = 0u; kg < 4u; ++kg) {
    acc0[kg] = float2(0.0f);
    acc1[kg] = float2(0.0f);
  }

  for (uint h = 0u; h < uint(H); ++h) {
    const float w0 = ok0 ? w[size_t(sr0) * size_t(H) + size_t(h)] : 0.0f;
    const float w1 = ok1 ? w[size_t(sr1) * size_t(H) + size_t(h)] : 0.0f;
    const device float* q_ptr0 =
        q + (size_t(ok0 ? sr0 : 0u) * size_t(H) + size_t(h)) * 128u;
    const device float* q_ptr1 =
        q + (size_t(ok1 ? sr1 : 0u) * size_t(H) + size_t(h)) * 128u;

    float2 dot0[4];
    float2 dot1[4];
    #pragma unroll
    for (uint kg = 0u; kg < 4u; ++kg) {
      dot0[kg] = float2(0.0f);
      dot1[kg] = float2(0.0f);
    }

    #pragma unroll
    for (uint d = 0u; d < 128u; d += 8u) {
      const half2 q0 = ok0
          ? half2(half(q_ptr0[d + fn]), half(q_ptr0[d + fn + 1u]))
          : half2(0.0h);
      const half2 q1 = ok1
          ? half2(half(q_ptr1[d + fn]), half(q_ptr1[d + fn + 1u]))
          : half2(0.0h);
      #pragma unroll
      for (uint kg = 0u; kg < 4u; ++kg) {
        if (g0 + kg * 8u < vis_max) {
          simdgroup_half8x8 mk;
          simdgroup_load(mk, k_tile + (kg * 8u) * PAD + d, PAD, ulong2(0), true);
          hy4_idx_m3_mma(dot0[kg], q0, mk);
          hy4_idx_m3_mma(dot1[kg], q1, mk);
        }
      }
    }

    #pragma unroll
    for (uint kg = 0u; kg < 4u; ++kg) {
      if (g0 + kg * 8u < vis_max) {
        acc0[kg].x = metal::fma(w0, metal::max(dot0[kg].x, 0.0f), acc0[kg].x);
        acc0[kg].y = metal::fma(w0, metal::max(dot0[kg].y, 0.0f), acc0[kg].y);
        acc1[kg].x = metal::fma(w1, metal::max(dot1[kg].x, 0.0f), acc1[kg].x);
        acc1[kg].y = metal::fma(w1, metal::max(dot1[kg].y, 0.0f), acc1[kg].y);
      }
    }
  }

  if (ok0) {
    device float* out0 = out + size_t(sr0) * size_t(N_g);
    #pragma unroll
    for (uint kg = 0u; kg < 4u; ++kg) {
      const uint gc = g0 + kg * 8u + fn;
      if (gc < vis_max) {
        out0[gc] = acc0[kg].x;
      }
      if (gc + 1u < vis_max) {
        out0[gc + 1u] = acc0[kg].y;
      }
    }
  }
  if (ok1) {
    device float* out1 = out + size_t(sr1) * size_t(N_g);
    #pragma unroll
    for (uint kg = 0u; kg < 4u; ++kg) {
      const uint gc = g0 + kg * 8u + fn;
      if (gc < vis_max) {
        out1[gc] = acc1[kg].x;
      }
      if (gc + 1u < vis_max) {
        out1[gc + 1u] = acc1[kg].y;
      }
    }
  }
}

#ifdef HY4_INDEXER_NAX
template <
    bool PAGED,
    int INDEX_ROWS,
    bool HAS_VIS,
    typename QPtr,
    typename WPtr,
    typename KPtr,
    typename PagePtr,
    typename VisPtr,
    typename ParamsPtr>
inline void hy4_indexer_score_prefill_nax_impl(
    QPtr q,
    WPtr w,
    KPtr k_pool,
    PagePtr page_ids,
    VisPtr visible_groups,
    ParamsPtr params,
    device float* out,
    threadgroup half* k_tile,
    int H,
    uint g_tile,
    uint s_tile,
    uint tid,
    uint lane,
    uint sg) {
  constexpr uint BN_G = 64u;
  constexpr uint N_SUB_MAX = BN_G / 32u;
  constexpr uint HALF_STRIDE = BN_G * 64u;

  const uint S = params[0];
  const uint N_g = params[1];
  const uint start_g = params[2];
  const uint s0 = s_tile * 64u;
  const uint g0 = g_tile * BN_G;
  if (s0 >= S || g0 >= N_g) {
    return;
  }

  // Every SG computes the identical tile-max visibility via one simd_max (no barrier).
  uint vis_max = N_g;
  if constexpr (HAS_VIS) {
    const uint r0 = s0 + lane;
    const uint r1 = s0 + 32u + lane;
    const int m0 = (r0 < S) ? visible_groups[r0] : 0;
    const int m1 = (r1 < S) ? visible_groups[r1] : 0;
    const int mv = simd_max(metal::max(m0, m1));
    vis_max = (mv > 0) ? metal::min(uint(mv), N_g) : 0u;
    if (g0 >= vis_max) {
      return;
    }
  }

  const uint n_sub = metal::min(N_SUB_MAX, (vis_max - g0 + 31u) >> 5);
  const uint load_rows = n_sub * 32u;
  const bool full_g = (g0 + BN_G) <= vis_max;

  if (full_g) {
    for (uint i = tid; i < BN_G * 16u; i += 128u) {
      const uint r = i >> 4;
      const uint c16 = i & 15u;
      const uint half_idx = c16 >> 3;
      const uint c8 = c16 & 7u;
      const uint sw = uint(hy4_swz_c8(int(c8), int(r)));
      const uint g_abs = start_g + g0 + r;
      size_t k_row;
      if constexpr (PAGED) {
        const uint page_idx = g_abs / uint(INDEX_ROWS);
        const uint local_row = g_abs - page_idx * uint(INDEX_ROWS);
        const int phys_page = page_ids[page_idx];
        k_row = size_t(phys_page) * size_t(INDEX_ROWS) + size_t(local_row);
      } else {
        k_row = size_t(g_abs);
      }
      threadgroup float4* dst = reinterpret_cast<threadgroup float4*>(
          k_tile + half_idx * HALF_STRIDE + r * 64u + sw * 8u);
      *dst = *reinterpret_cast<const device float4*>(
          k_pool + k_row * 128u + size_t(c16) * 8u);
    }
  } else {
    for (uint i = tid; i < load_rows * 16u; i += 128u) {
      const uint r = i >> 4;
      const uint c16 = i & 15u;
      const uint half_idx = c16 >> 3;
      const uint c8 = c16 & 7u;
      const uint sw = uint(hy4_swz_c8(int(c8), int(r)));
      const uint g = g0 + r;
      threadgroup float4* dst = reinterpret_cast<threadgroup float4*>(
          k_tile + half_idx * HALF_STRIDE + r * 64u + sw * 8u);
      if (g < vis_max) {
        const uint g_abs = start_g + g;
        size_t k_row;
        if constexpr (PAGED) {
          const uint page_idx = g_abs / uint(INDEX_ROWS);
          const uint local_row = g_abs - page_idx * uint(INDEX_ROWS);
          const int phys_page = page_ids[page_idx];
          k_row = size_t(phys_page) * size_t(INDEX_ROWS) + size_t(local_row);
        } else {
          k_row = size_t(g_abs);
        }
        *dst = *reinterpret_cast<const device float4*>(
            k_pool + k_row * 128u + size_t(c16) * 8u);
      } else {
        *dst = float4(0.0f);
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint s_sg = s0 + sg * 16u;
  const short valid_s = short(max(0, min(16, int(S) - int(s_sg))));
  if (valid_s <= 0) {
    return;
  }

  const short2 sc = mlx::steel::BaseNAXFrag::get_coord();
  const short fm = sc.y;
  const short fn = sc.x;
  const uint sr0 = s_sg + uint(fm);
  const uint sr1 = sr0 + 8u;
  const bool ok0 = sr0 < S;
  const bool ok1 = sr1 < S;
  const int q_ld = H * 128;

  mlx::steel::NAXTile<float, 1, 2> acc[N_SUB_MAX];
  STEEL_PRAGMA_UNROLL
  for (uint gb = 0u; gb < N_SUB_MAX; ++gb) {
    acc[gb].clear();
  }

  if (n_sub == N_SUB_MAX) {
    for (uint h = 0u; h < uint(H); ++h) {
      const float w0 = ok0 ? w[size_t(sr0) * size_t(H) + size_t(h)] : 0.0f;
      const float w1 = ok1 ? w[size_t(sr1) * size_t(H) + size_t(h)] : 0.0f;
      const device float* q_ptr =
          q + (size_t(s_sg) * size_t(H) + size_t(h)) * 128u;

      mlx::steel::NAXTile<half, 1, 4> q_h0;
      mlx::steel::NAXTile<half, 1, 4> q_h1;
      if (valid_s == 16) {
        q_h0.load(q_ptr, q_ld);
        q_h1.load(q_ptr + 64, q_ld);
      } else {
        q_h0.load_rows(q_ptr, q_ld, valid_s);
        q_h1.load_rows(q_ptr + 64, q_ld, valid_s);
      }

      STEEL_PRAGMA_UNROLL
      for (uint gb = 0u; gb < N_SUB_MAX; ++gb) {
        mlx::steel::NAXTile<float, 1, 2> dot_h;
        dot_h.clear();
        mlx::steel::NAXTile<half, 2, 4> kt;
        hy4_load_tile_swz64(kt, k_tile + size_t(gb * 32u) * 64u, fm, fn);
        mlx::steel::tile_matmad_nax(
            dot_h, q_h0, metal::false_type{}, kt, metal::true_type{});
        hy4_load_tile_swz64(
            kt, k_tile + HALF_STRIDE + size_t(gb * 32u) * 64u, fm, fn);
        mlx::steel::tile_matmad_nax(
            dot_h, q_h1, metal::false_type{}, kt, metal::true_type{});

        STEEL_PRAGMA_UNROLL
        for (short f = 0; f < 2; ++f) {
          const thread auto& df = dot_h.frag_at(0, f);
          thread auto& af = acc[gb].frag_at(0, f);
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < 4; ++j) {
            af[j] = metal::fma(w0, metal::max(df[j], 0.0f), af[j]);
            af[4 + j] = metal::fma(w1, metal::max(df[4 + j], 0.0f), af[4 + j]);
          }
        }
      }
    }
  } else {
    for (uint h = 0u; h < uint(H); ++h) {
      const float w0 = ok0 ? w[size_t(sr0) * size_t(H) + size_t(h)] : 0.0f;
      const float w1 = ok1 ? w[size_t(sr1) * size_t(H) + size_t(h)] : 0.0f;
      const device float* q_ptr =
          q + (size_t(s_sg) * size_t(H) + size_t(h)) * 128u;

      mlx::steel::NAXTile<half, 1, 4> q_h0;
      mlx::steel::NAXTile<half, 1, 4> q_h1;
      if (valid_s == 16) {
        q_h0.load(q_ptr, q_ld);
        q_h1.load(q_ptr + 64, q_ld);
      } else {
        q_h0.load_rows(q_ptr, q_ld, valid_s);
        q_h1.load_rows(q_ptr + 64, q_ld, valid_s);
      }

      STEEL_PRAGMA_UNROLL
      for (uint gb = 0u; gb < N_SUB_MAX; ++gb) {
        if (gb < n_sub) {
          mlx::steel::NAXTile<float, 1, 2> dot_h;
          dot_h.clear();
          mlx::steel::NAXTile<half, 2, 4> kt;
          hy4_load_tile_swz64(kt, k_tile + size_t(gb * 32u) * 64u, fm, fn);
          mlx::steel::tile_matmad_nax(
              dot_h, q_h0, metal::false_type{}, kt, metal::true_type{});
          hy4_load_tile_swz64(
              kt, k_tile + HALF_STRIDE + size_t(gb * 32u) * 64u, fm, fn);
          mlx::steel::tile_matmad_nax(
              dot_h, q_h1, metal::false_type{}, kt, metal::true_type{});

          STEEL_PRAGMA_UNROLL
          for (short f = 0; f < 2; ++f) {
            const thread auto& df = dot_h.frag_at(0, f);
            thread auto& af = acc[gb].frag_at(0, f);
            STEEL_PRAGMA_UNROLL
            for (short j = 0; j < 4; ++j) {
              af[j] = metal::fma(w0, metal::max(df[j], 0.0f), af[j]);
              af[4 + j] = metal::fma(w1, metal::max(df[4 + j], 0.0f), af[4 + j]);
            }
          }
        }
      }
    }
  }

  STEEL_PRAGMA_UNROLL
  for (uint gb = 0u; gb < N_SUB_MAX; ++gb) {
    if (gb < n_sub) {
      const uint g_sub = g0 + gb * 32u;
      const short valid_g = short(max(0, min(32, int(vis_max) - int(g_sub))));
      device float* out_ptr =
          out + size_t(s_sg) * size_t(N_g) + size_t(g_sub);
      if (valid_s == 16 && valid_g == 32) {
        acc[gb].store(out_ptr, N_g);
      } else if (valid_g > 0) {
        acc[gb].store_slice(out_ptr, N_g, short2(0, 0), short2(valid_g, valid_s));
      }
    }
  }
}

template <
    bool PAGED,
    int INDEX_ROWS,
    bool SHARE_S,
    typename QPtr,
    typename WPtr,
    typename KPtr,
    typename PagePtr,
    typename ParamsPtr>
inline void hy4_indexer_score_decode_nax_impl(
    QPtr q,
    WPtr w,
    KPtr k_pool,
    PagePtr page_ids,
    ParamsPtr params,
    device float* out,
    threadgroup half* k_tile,
    int H,
    uint s_grid,
    uint g_tile,
    uint lane) {
  constexpr uint PAD = 136u;

  const uint S = params[0];
  const uint N_g = params[1];
  const uint start_g = params[2];
  const uint g0 = g_tile * 32u;
  if ((!SHARE_S && s_grid >= S) || g0 >= N_g) {
    return;
  }

  // Each half-warp (16 lanes) loads one 256-byte key row via half8 (float4).
  const uint lane_r = lane >> 4;
  const uint c8 = lane & 15u;
  STEEL_PRAGMA_UNROLL
  for (uint rb = 0u; rb < 32u; rb += 2u) {
    const uint r = rb + lane_r;
    const uint g = g0 + r;
    threadgroup float4* dst =
        reinterpret_cast<threadgroup float4*>(k_tile + r * PAD + c8 * 8u);
    if (g < N_g) {
      const uint g_abs = start_g + g;
      size_t k_row;
      if constexpr (PAGED) {
        const uint page_idx = g_abs / uint(INDEX_ROWS);
        const uint local_row = g_abs - page_idx * uint(INDEX_ROWS);
        const int phys_page = page_ids[page_idx];
        k_row = size_t(phys_page) * size_t(INDEX_ROWS) + size_t(local_row);
      } else {
        k_row = size_t(g_abs);
      }
      *dst = *reinterpret_cast<const device float4*>(
          k_pool + k_row * 128u + size_t(c8) * 8u);
    } else {
      *dst = float4(0.0f);
    }
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);

  using hfrag_t = typename mlx::steel::BaseNAXFrag::template dtype_frag_t<half>;
  using ffrag_t = typename mlx::steel::BaseNAXFrag::template dtype_frag_t<float>;

  const short2 sc = mlx::steel::BaseNAXFrag::get_coord();
  const short fm = sc.y;
  const short fn = sc.x;
  const bool ok0 = int(fm) < H;
  const bool ok1 = (int(fm) + 8) < H;

  const uint s_lo = SHARE_S ? 0u : s_grid;
  const uint s_hi = SHARE_S ? S : (s_grid + 1u);

  for (uint s = s_lo; s < s_hi; ++s) {
    const device float* q_s = q + size_t(s) * size_t(H) * 128u;
    ffrag_t dot0 = ffrag_t(0.0f);
    ffrag_t dot1 = ffrag_t(0.0f);

    // Split F32 Q into FP16 hi + residual to preserve <= 1e-3 accuracy across D=128.
    STEEL_PRAGMA_UNROLL
    for (int d = 0; d < 128; d += 16) {
      const float4 qa = ok0
          ? *reinterpret_cast<const device float4*>(
                q_s + size_t(fm) * 128u + size_t(d + int(fn)))
          : float4(0.0f);
      const float4 qb = ok1
          ? *reinterpret_cast<const device float4*>(
                q_s + size_t(fm + 8) * 128u + size_t(d + int(fn)))
          : float4(0.0f);
      const half4 ha = half4(qa);
      const half4 hb = half4(qb);
      const half4 la = half4(qa - float4(ha));
      const half4 lb = half4(qb - float4(hb));
      const hfrag_t q_hi(ha.x, ha.y, ha.z, ha.w, hb.x, hb.y, hb.z, hb.w);
      const hfrag_t q_lo(la.x, la.y, la.z, la.w, lb.x, lb.y, lb.z, lb.w);

      const half4 k0a = *reinterpret_cast<const threadgroup half4*>(
          k_tile + size_t(fm) * PAD + size_t(d + int(fn)));
      const half4 k0b = *reinterpret_cast<const threadgroup half4*>(
          k_tile + size_t(fm + 8) * PAD + size_t(d + int(fn)));
      const half4 k1a = *reinterpret_cast<const threadgroup half4*>(
          k_tile + size_t(16 + fm) * PAD + size_t(d + int(fn)));
      const half4 k1b = *reinterpret_cast<const threadgroup half4*>(
          k_tile + size_t(24 + fm) * PAD + size_t(d + int(fn)));
      const hfrag_t kf0(k0a.x, k0a.y, k0a.z, k0a.w, k0b.x, k0b.y, k0b.z, k0b.w);
      const hfrag_t kf1(k1a.x, k1a.y, k1a.z, k1a.w, k1b.x, k1b.y, k1b.z, k1b.w);

      mlx::steel::BaseNAXFrag::mma(
          dot0, dot1, q_hi, metal::false_type{}, kf0, kf1, metal::true_type{});
      mlx::steel::BaseNAXFrag::mma(
          dot0, dot1, q_lo, metal::false_type{}, kf0, kf1, metal::true_type{});
    }

    const float w0 = ok0 ? w[size_t(s) * size_t(H) + size_t(fm)] : 0.0f;
    const float w1 = ok1 ? w[size_t(s) * size_t(H) + size_t(fm) + 8u] : 0.0f;

    float4 s0(
        metal::fma(w0, metal::max(dot0[0], 0.0f), w1 * metal::max(dot0[4], 0.0f)),
        metal::fma(w0, metal::max(dot0[1], 0.0f), w1 * metal::max(dot0[5], 0.0f)),
        metal::fma(w0, metal::max(dot0[2], 0.0f), w1 * metal::max(dot0[6], 0.0f)),
        metal::fma(w0, metal::max(dot0[3], 0.0f), w1 * metal::max(dot0[7], 0.0f)));
    float4 s1(
        metal::fma(w0, metal::max(dot1[0], 0.0f), w1 * metal::max(dot1[4], 0.0f)),
        metal::fma(w0, metal::max(dot1[1], 0.0f), w1 * metal::max(dot1[5], 0.0f)),
        metal::fma(w0, metal::max(dot1[2], 0.0f), w1 * metal::max(dot1[6], 0.0f)),
        metal::fma(w0, metal::max(dot1[3], 0.0f), w1 * metal::max(dot1[7], 0.0f)));

    // Lanes sharing the same fn differ only in bits 1, 2, 4 (fm = [b4 b2 b1]).
    s0 += simd_shuffle_xor(s0, ushort(2));
    s0 += simd_shuffle_xor(s0, ushort(4));
    s0 += simd_shuffle_xor(s0, ushort(16));

    s1 += simd_shuffle_xor(s1, ushort(2));
    s1 += simd_shuffle_xor(s1, ushort(4));
    s1 += simd_shuffle_xor(s1, ushort(16));

    if (fm == 0) {
      device float* out_row = out + size_t(s) * size_t(N_g);
      const uint base0 = g0 + uint(fn);
      const uint base1 = base0 + 16u;
      if (base0 + 4u <= N_g) {
        out_row[base0 + 0u] = s0.x;
        out_row[base0 + 1u] = s0.y;
        out_row[base0 + 2u] = s0.z;
        out_row[base0 + 3u] = s0.w;
      } else {
        for (uint j = 0u; j < 4u; ++j) {
          if (base0 + j < N_g) {
            out_row[base0 + j] = s0[j];
          }
        }
      }
      if (base1 + 4u <= N_g) {
        out_row[base1 + 0u] = s1.x;
        out_row[base1 + 1u] = s1.y;
        out_row[base1 + 2u] = s1.z;
        out_row[base1 + 3u] = s1.w;
      } else {
        for (uint j = 0u; j < 4u; ++j) {
          if (base1 + j < N_g) {
            out_row[base1 + j] = s1[j];
          }
        }
      }
    }
  }
}
#endif

// Order-preserving IEEE-754 float -> uint32 map. Canonicalizes -0.0f to +0.0f
// so tie-breaking matches IEEE-754 equality (`-0.0f == +0.0f`); maps NaN to 0
// (strictly below -inf).
inline uint hy4_f2ui(float f) {
  if (metal::isnan(f)) {
    return 0u;
  }
  if (f == 0.0f) {
    return 0x80000000u;
  }
  const uint u = as_type<uint>(f);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

template <bool PAGED, typename PagePtr>
inline int hy4_map_cell(uint cell, PagePtr page_ids, uint block_size) {
  if constexpr (PAGED) {
    if (block_size == 32u) {
      return page_ids[cell >> 5] * 32 + int(cell & 31u);
    }
    return page_ids[cell / block_size] * int(block_size) +
        int(cell % block_size);
  } else {
    return int(cell);
  }
}

template <
    bool PAGED,
    typename ScoresPtr,
    typename VisPtr,
    typename TailStartPtr,
    typename TailLenPtr,
    typename PagePtr,
    typename ParamsPtr>
inline void hy4_indexer_topk_cells_impl(
    ScoresPtr scores,
    VisPtr visible_groups,
    TailStartPtr tail_starts,
    TailLenPtr tail_lens,
    PagePtr page_ids,
    ParamsPtr params,
    device int* out,
    threadgroup atomic_uint* hist,
    threadgroup uint* sg_eq,
    threadgroup uint* sg_keep,
    threadgroup uint* sel_groups,
    threadgroup uint* tg_state,
    uint S,
    uint N_g,
    uint r,
    uint tid,
    uint lane,
    uint sg) {
  if (r >= S) {
    return;
  }
  const uint n_top = params[0];
  const uint kpool = params[1];
  const uint kmax = params[2];
  const uint block_size = params[3];

  const int raw_vis = visible_groups[r];
  const uint vis = (raw_vis > 0) ? metal::min(uint(raw_vis), N_g) : 0u;
  const int raw_ts = tail_starts[r];
  const int raw_tl = tail_lens[r];
  const uint tail_start = (raw_ts > 0) ? uint(raw_ts) : 0u;
  const uint tail_len = (raw_tl > 0) ? uint(raw_tl) : 0u;
  const size_t out_base = size_t(r) * size_t(kmax);

  if (vis <= n_top) {
    const uint n_vis_cells = vis * kpool;
    const uint total_valid = n_vis_cells + tail_len;
    for (uint i = tid; i < kmax; i += 256u) {
      int slot = -1;
      if (i < n_vis_cells) {
        slot = hy4_map_cell<PAGED>(i, page_ids, block_size);
      } else if (i < total_valid) {
        slot = hy4_map_cell<PAGED>(
            tail_start + (i - n_vis_cells), page_ids, block_size);
      }
      out[out_base + size_t(i)] = slot;
    }
    return;
  }

  const size_t sc_base = size_t(r) * size_t(N_g);

  if (tid == 0u) {
    tg_state[0] = 0u;
    tg_state[1] = 0u;
    tg_state[2] = n_top;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (int p = 3; p >= 0; --p) {
    const uint shift = uint(p) * 8u;
    const uint prefix = tg_state[0];
    const uint mask = tg_state[1];
    atomic_store_explicit(&hist[tid], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint g = tid; g < vis; g += 256u) {
      const uint key = hy4_f2ui(scores[sc_base + size_t(g)]);
      if ((key & mask) == prefix) {
        const uint bucket = (key >> shift) & 0xFFu;
        atomic_fetch_add_explicit(&hist[bucket], 1u, memory_order_relaxed);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
      const uint rem = tg_state[2];
      uint cum = 0u;
      uint chosen = 0u;
      for (int b = 255; b >= 0; --b) {
        const uint c = atomic_load_explicit(&hist[b], memory_order_relaxed);
        if (cum + c >= rem) {
          chosen = uint(b);
          break;
        }
        cum += c;
      }
      tg_state[0] = prefix | (chosen << shift);
      tg_state[1] = mask | (0xFFu << shift);
      tg_state[2] = rem - cum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  const uint thr = tg_state[0];
  const uint need_ties = tg_state[2];

  uint seen_eq = 0u;
  uint out_cnt = 0u;
  for (uint base = 0u; base < vis; base += 256u) {
    const uint g = base + tid;
    uint is_gt = 0u;
    uint is_eq = 0u;
    if (g < vis) {
      const uint key = hy4_f2ui(scores[sc_base + size_t(g)]);
      is_gt = (key > thr) ? 1u : 0u;
      is_eq = (key == thr) ? 1u : 0u;
    }

    const uint eq_lane_ex = simd_prefix_exclusive_sum(is_eq);
    const uint eq_sg_tot = simd_sum(is_eq);
    if (lane == 0u) {
      sg_eq[sg] = eq_sg_tot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint eq_sg_ex = 0u;
    for (uint i = 0u; i < sg; ++i) {
      eq_sg_ex += sg_eq[i];
    }
    uint eq_chunk_tot = eq_sg_ex;
    for (uint i = sg; i < 8u; ++i) {
      eq_chunk_tot += sg_eq[i];
    }
    const uint eq_rank = seen_eq + eq_sg_ex + eq_lane_ex;
    seen_eq += eq_chunk_tot;

    const uint keep =
        (is_gt != 0u || (is_eq != 0u && eq_rank < need_ties)) ? 1u : 0u;

    const uint keep_lane_ex = simd_prefix_exclusive_sum(keep);
    const uint keep_sg_tot = simd_sum(keep);
    if (lane == 0u) {
      sg_keep[sg] = keep_sg_tot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint keep_sg_ex = 0u;
    for (uint i = 0u; i < sg; ++i) {
      keep_sg_ex += sg_keep[i];
    }
    uint keep_chunk_tot = keep_sg_ex;
    for (uint i = sg; i < 8u; ++i) {
      keep_chunk_tot += sg_keep[i];
    }
    if (keep != 0u) {
      const uint pos = out_cnt + keep_sg_ex + keep_lane_ex;
      if (pos < n_top) {
        sel_groups[pos] = g;
      }
    }
    out_cnt += keep_chunk_tot;
    if (out_cnt >= n_top) {
      break;
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint n_sel_cells = n_top * kpool;
  const uint total_valid = n_sel_cells + tail_len;
  for (uint i = tid; i < kmax; i += 256u) {
    int slot = -1;
    if (i < n_sel_cells) {
      const uint g_id =
          (kpool == 4u) ? sel_groups[i >> 2] : sel_groups[i / kpool];
      const uint sub = (kpool == 4u) ? (i & 3u) : (i % kpool);
      slot = hy4_map_cell<PAGED>(g_id * kpool + sub, page_ids, block_size);
    } else if (i < total_valid) {
      slot = hy4_map_cell<PAGED>(
          tail_start + (i - n_sel_cells), page_ids, block_size);
    }
    out[out_base + size_t(i)] = slot;
  }
}

template <
    typename TailPtr,
    typename TokPtr,
    typename WValPtr,
    typename WGatePtr,
    typename ApePtr,
    typename NormPtr,
    typename InvFreqPtr,
    typename ParamsPtr,
    typename EpsPtr,
    typename OutT>
inline void hy4_kpool_compress_impl(
    TailPtr tail_buf,
    TokPtr tok,
    WValPtr w_val,
    WGatePtr w_gate,
    ApePtr ape,
    NormPtr comp_norm,
    InvFreqPtr inv_freq,
    ParamsPtr params,
    EpsPtr eps_ptr,
    device OutT* out,
    threadgroup float4* x_tg,
    threadgroup float* tg_pooled,
    threadgroup float* tg_sq,
    int rope_dim,
    uint g,
    uint tid,
    uint lane,
    uint sg) {
  const uint n_groups = params[0];
  const uint first_group = params[1];
  const uint open_len = params[2];
  const uint tail_start_slot = params[3];
  const uint tail_cap = params[4];
  if (g >= n_groups) {
    return;
  }

  if (tid < 128u) {
    const uint s = tid >> 5;
    const uint c4 = tid & 31u;
    const uint t = g * 4u + s;
    float4 xv;
    if (t < open_len) {
      const uint slot = (tail_start_slot + t) % tail_cap;
      const size_t off = size_t(slot) * 128u + size_t(c4) * 4u;
      xv = float4(
          tail_buf[off], tail_buf[off + 1u], tail_buf[off + 2u], tail_buf[off + 3u]);
    } else {
      const uint tr = t - open_len;
      const size_t off = size_t(tr) * 128u + size_t(c4) * 4u;
      xv = float4(tok[off], tok[off + 1u], tok[off + 2u], tok[off + 3u]);
    }
    x_tg[s * 32u + c4] = xv;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const device float4* wv4 = reinterpret_cast<const device float4*>(w_val);
  const device float4* wg4 = reinterpret_cast<const device float4*>(w_gate);
  const float4 x0 = x_tg[0u * 32u + lane];
  const float4 x1 = x_tg[1u * 32u + lane];
  const float4 x2 = x_tg[2u * 32u + lane];
  const float4 x3 = x_tg[3u * 32u + lane];

  for (uint iter = 0u; iter < 16u; ++iter) {
    const uint c = sg + iter * 8u;
    const float4 wv = wv4[size_t(c) * 32u + size_t(lane)];
    const float4 wg = wg4[size_t(c) * 32u + size_t(lane)];
    const float v0 = simd_sum(metal::dot(x0, wv));
    const float v1 = simd_sum(metal::dot(x1, wv));
    const float v2 = simd_sum(metal::dot(x2, wv));
    const float v3 = simd_sum(metal::dot(x3, wv));
    const float g0 = simd_sum(metal::dot(x0, wg));
    const float g1 = simd_sum(metal::dot(x1, wg));
    const float g2 = simd_sum(metal::dot(x2, wg));
    const float g3 = simd_sum(metal::dot(x3, wg));
    if (lane == 0u) {
      const float a0 = g0 + ape[0u * 128u + c];
      const float a1 = g1 + ape[1u * 128u + c];
      const float a2 = g2 + ape[2u * 128u + c];
      const float a3 = g3 + ape[3u * 128u + c];
      const float gm = metal::max(metal::max(a0, a1), metal::max(a2, a3));
      const float e0 = metal::exp(a0 - gm);
      const float e1 = metal::exp(a1 - gm);
      const float e2 = metal::exp(a2 - gm);
      const float e3 = metal::exp(a3 - gm);
      const float es = e0 + e1 + e2 + e3;
      tg_pooled[c] =
          v0 * (e0 / es) + v1 * (e1 / es) + v2 * (e2 / es) + v3 * (e3 / es);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float pv = 0.0f;
  if (tid < 128u) {
    pv = tg_pooled[tid];
  }
  const float sq = simd_sum(pv * pv);
  if (lane == 0u && sg < 4u) {
    tg_sq[sg] = sq;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const float sq_tot = tg_sq[0] + tg_sq[1] + tg_sq[2] + tg_sq[3];
  const float rs = metal::precise::rsqrt(sq_tot / 128.0f + eps_ptr[0]);
  if (tid < 128u) {
    tg_pooled[tid] = pv * rs * comp_norm[tid];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint head_w = 128u - uint(rope_dim);
  const uint half_rd = uint(rope_dim) >> 1;
  const size_t out_off = size_t(g) * 128u;
  if (tid < head_w) {
    out[out_off + size_t(tid)] = OutT(tg_pooled[tid]);
  } else if (tid < head_w + half_rd) {
    const uint p = tid - head_w;
    const float pos_g = float((first_group + g) * 4u);
    const float theta = pos_g * inv_freq[p];
    const float c_v = metal::precise::cos(theta);
    const float s_v = metal::precise::sin(theta);
    const float r0 = tg_pooled[head_w + 2u * p];
    const float r1 = tg_pooled[head_w + 2u * p + 1u];
    out[out_off + size_t(head_w + 2u * p)] = OutT(r0 * c_v - r1 * s_v);
    out[out_off + size_t(head_w + 2u * p + 1u)] = OutT(r0 * s_v + r1 * c_v);
  }
}

template <uint TG_X, typename OutT, typename XPtr, typename PosPtr, typename InvFreqPtr>
inline void hy4_partial_rope_tail_impl(
    XPtr x,
    PosPtr pos,
    InvFreqPtr inv_freq,
    device OutT* out,
    int M,
    int H,
    int D,
    int half_rd,
    size_t x_s0,
    size_t x_s1,
    uint h,
    uint m,
    uint tid) {
  if (int(m) >= M || int(h) >= H) {
    return;
  }
  const uint rd = uint(half_rd) * 2u;
  const uint head_w = uint(D) - rd;
  const size_t x_off = size_t(m) * x_s0 + size_t(h) * x_s1;
  const size_t o_off = (size_t(m) * size_t(H) + size_t(h)) * size_t(D);

  for (uint j = tid; j < head_w; j += TG_X) {
    out[o_off + size_t(j)] = OutT(float(x[x_off + size_t(j)]));
  }
  const float pos_m = float(pos[m]);
  for (uint p = tid; p < uint(half_rd); p += TG_X) {
    const float theta = pos_m * inv_freq[p];
    const float c = metal::precise::cos(theta);
    const float s = metal::precise::sin(theta);
    const float x0 = float(x[x_off + size_t(head_w + 2u * p)]);
    const float x1 = float(x[x_off + size_t(head_w + 2u * p + 1u)]);
    out[o_off + size_t(head_w + 2u * p)] = OutT(x0 * c - x1 * s);
    out[o_off + size_t(head_w + 2u * p + 1u)] = OutT(x0 * s + x1 * c);
  }
}
)METAL"
