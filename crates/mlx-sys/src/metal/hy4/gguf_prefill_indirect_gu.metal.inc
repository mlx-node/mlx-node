R"METAL(
constexpr int BM = 32;
constexpr int BN = 64;
constexpr int BK = 64;
constexpr int PAD_A = 72;
constexpr int PAD_W = 40;

const int R = token_rows_shape[0];
const int N_TOK = x_shape[0];
const int K = x_shape[1];
const int S = dims[0];
const int M = dims[1];

const uint tile = threadgroup_position_in_grid.y;
const uint begin = tiles[3u * tile];
const uint end = tiles[3u * tile + 1u];
const uint slot = tiles[3u * tile + 2u];
if (begin >= end || end > uint(R)) return;

const uint col = threadgroup_position_in_grid.x * uint(BN);
if (col >= uint(M)) return;

const uint sg_idx = simdgroup_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;
const uint tid = sg_idx * 32u + lane;
const short tm = short((sg_idx / 2u) * 16u);
const short tn = short((sg_idx % 2u) * 32u);
const short valid_m = short(max(0, min(16, int(end - begin) - int(tm))));
const short valid_n = short(max(0, min(32, int(M) - int(col + uint(tn)))));
const short2 coord = BaseNAXFrag::get_coord();
const short fm = coord.y;
const short fn = coord.x;

// As_hi (4,608 B) + As_lo (4,608 B) + Ws (5,120 B) + S/M/sum_x (768 B) = 15,104 B <= 16 KiB (2 TGs/core).
alignas(16) threadgroup half As_hi[BM * PAD_A];
alignas(16) threadgroup half As_lo[BM * PAD_A];
alignas(16) threadgroup half Ws[BN * PAD_W];
alignas(16) threadgroup float S_tg[BN];
alignas(16) threadgroup float M_tg[BN];
alignas(16) threadgroup float sum_x[BM * 2];

NAXTile<float, 1, 2> cg;
NAXTile<float, 1, 2> cu;
cg.clear();
cu.clear();

if (slot < uint(S)) {
  const uint ar = tid >> 2;
  const uint ac = (tid & 3u) * 16u;
  bool live = (begin + ar) < end;
  const uint token = live ? token_rows[begin + ar] : 0u;
  live = live && (token < uint(N_TOK));
  float4 av[4];
  if (live) {
    const device float4* src = reinterpret_cast<const device float4*>(
        x + size_t(token) * size_t(K) + size_t(ac));
    av[0] = src[0];
    av[1] = src[1];
    av[2] = src[2];
    av[3] = src[3];
  } else {
    threadgroup half4* d_hi =
        reinterpret_cast<threadgroup half4*>(As_hi + ar * PAD_A + ac);
    threadgroup half4* d_lo =
        reinterpret_cast<threadgroup half4*>(As_lo + ar * PAD_A + ac);
    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) {
      d_hi[i] = half4(0.0h);
      d_lo[i] = half4(0.0h);
    }
    if ((tid & 1u) == 0u) {
      sum_x[ar * 2u + ((tid >> 1) & 1u)] = 0.0f;
    }
  }

  // 4 threads per weight row cooperatively load 8 B each = 32 B Q4_K chunk.
  const uint row0 = tid >> 2;
  const uint lane4 = tid & 3u;
  const ushort quad_base = ushort(lane & ~3u);
  const bool w_live[2] = {
      (col + row0) < uint(M),
      (col + row0 + 32u) < uint(M),
  };
  const size_t row_bytes = size_t(K / 256) * 144u;
  const device uint8_t* g_row[2];
  const device uint8_t* u_row[2];
  STEEL_PRAGMA_UNROLL
  for (int p = 0; p < 2; ++p) {
    const uint wr = row0 + uint(p) * 32u;
    const size_t roff = (size_t(slot) * size_t(M) + size_t(col + wr)) * row_bytes;
    g_row[p] = gate_buf + size_t(offsets[0]) + roff;
    u_row[p] = up_buf + size_t(offsets[1]) + roff;
    if (!w_live[p]) {
      threadgroup half4* dst =
          reinterpret_cast<threadgroup half4*>(Ws + wr * PAD_W + lane4 * 8u);
      dst[0] = half4(0.0h);
      dst[1] = half4(0.0h);
      if (lane4 == 0u) {
        S_tg[wr] = 0.0f;
        M_tg[wr] = 0.0f;
      }
    }
  }

  float g_d[2], g_dmin[2];
  uint32_t g_sc[2][3];
  uint32_t g_qs[2][2];
  float u_d[2], u_dmin[2];
  uint32_t u_sc[2][3];
  uint32_t u_qs[2][2];

  STEEL_PRAGMA_UNROLL
  for (int p = 0; p < 2; ++p) {
    const uint32_t hdr = w_live[p]
        ? reinterpret_cast<const device uint32_t*>(g_row[p])[lane4]
        : 0u;
    const half2 dm = as_type<half2>(simd_shuffle(hdr, quad_base));
    g_d[p] = float(dm.x);
    g_dmin[p] = float(dm.y);
    g_sc[p][0] = simd_shuffle(hdr, quad_base | 1u);
    g_sc[p][1] = simd_shuffle(hdr, quad_base | 2u);
    g_sc[p][2] = simd_shuffle(hdr, quad_base | 3u);
    if (w_live[p]) {
      const device uint32_t* qsp = reinterpret_cast<const device uint32_t*>(
          g_row[p] + 16u + size_t(lane4) * 8u);
      g_qs[p][0] = qsp[0];
      g_qs[p][1] = qsp[1];
    }
  }

  const bool active = (valid_m > 0 && valid_n > 0);

  for (int base = 0; base < K; base += BK) {
    const int c = (base >> 6) & 3;
    const int g0 = 2 * c;
    const int g1 = g0 + 1;

    if (base > 0) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float s_local = 0.0f;
    if (live) {
      threadgroup half4* d_hi =
          reinterpret_cast<threadgroup half4*>(As_hi + ar * PAD_A + ac);
      threadgroup half4* d_lo =
          reinterpret_cast<threadgroup half4*>(As_lo + ar * PAD_A + ac);
      float4 sv = float4(0.0f);
      STEEL_PRAGMA_UNROLL
      for (int i = 0; i < 4; ++i) {
        const float4 v = av[i];
        const half4 hi = half4(v);
        d_hi[i] = hi;
        d_lo[i] = half4(v - float4(hi));
        sv += v;
      }
      s_local = (sv.x + sv.y) + (sv.z + sv.w);
    }
    const float s_pair = s_local + simd_shuffle_xor(s_local, 1u);
    if (live && (tid & 1u) == 0u) {
      sum_x[ar * 2u + ((tid >> 1) & 1u)] = s_pair;
    }

    // Group g0 (low nibbles): exact 4-bit integers in half + post-MMA f32 scale/bias.
    STEEL_PRAGMA_UNROLL
    for (int p = 0; p < 2; ++p) {
      if (w_live[p]) {
        const uint wr = row0 + uint(p) * 32u;
        if (lane4 == 0u) {
          float dg, mg;
          hy4_gguf_q4k_group_scale(g_sc[p], g_d[p], g_dmin[p], g0, dg, mg);
          S_tg[wr] = dg;
          M_tg[wr] = mg;
        }
        threadgroup half4* dst =
            reinterpret_cast<threadgroup half4*>(Ws + wr * PAD_W + lane4 * 8u);
        STEEL_PRAGMA_UNROLL
        for (int i = 0; i < 2; ++i) {
          const uint32_t w = g_qs[p][i];
          dst[i] = half4(
              half(w & 0x0Fu),
              half((w >> 8) & 0x0Fu),
              half((w >> 16) & 0x0Fu),
              half((w >> 24) & 0x0Fu));
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Prefetch Up weights for this K-strip while Gate g0 MMA runs.
    const size_t blk_off = size_t(base >> 8) * 144u;
    const size_t q_off = 16u + size_t(c) * 32u + size_t(lane4) * 8u;
    STEEL_PRAGMA_UNROLL
    for (int p = 0; p < 2; ++p) {
      const device uint8_t* ublk = u_row[p] + blk_off;
      if ((base & 255) == 0) {
        const uint32_t hdr = w_live[p]
            ? reinterpret_cast<const device uint32_t*>(ublk)[lane4]
            : 0u;
        const half2 dm = as_type<half2>(simd_shuffle(hdr, quad_base));
        u_d[p] = float(dm.x);
        u_dmin[p] = float(dm.y);
        u_sc[p][0] = simd_shuffle(hdr, quad_base | 1u);
        u_sc[p][1] = simd_shuffle(hdr, quad_base | 2u);
        u_sc[p][2] = simd_shuffle(hdr, quad_base | 3u);
      }
      if (w_live[p]) {
        const device uint32_t* qsp =
            reinterpret_cast<const device uint32_t*>(ublk + q_off);
        u_qs[p][0] = qsp[0];
        u_qs[p][1] = qsp[1];
      }
    }

    NAXTile<half, 1, 2> A_hi;
    NAXTile<half, 1, 2> A_lo;
    float sx0 = 0.0f;
    float sx1 = 0.0f;
    if (active) {
      A_hi.template load<half, PAD_A, 1>(As_hi + tm * PAD_A);
      A_lo.template load<half, PAD_A, 1>(As_lo + tm * PAD_A);
      sx0 = sum_x[uint(tm + fm) * 2u];
      sx1 = sum_x[uint(tm + fm + 8) * 2u];
      NAXTile<half, 2, 2> B;
      B.template load<half, PAD_W, 1>(Ws + tn * PAD_W);
      NAXTile<float, 1, 2> P;
      P.clear();
      tile_matmad_nax(P, A_hi, metal::false_type{}, B, metal::true_type{});
      tile_matmad_nax(P, A_lo, metal::false_type{}, B, metal::true_type{});
      const float4 d0v = *reinterpret_cast<const threadgroup float4*>(S_tg + tn + fn);
      const float4 d1v = *reinterpret_cast<const threadgroup float4*>(S_tg + tn + 16 + fn);
      const float4 m0v = *reinterpret_cast<const threadgroup float4*>(M_tg + tn + fn);
      const float4 m1v = *reinterpret_cast<const threadgroup float4*>(M_tg + tn + 16 + fn);
      thread auto& g0f = cg.frag_at(0, 0);
      thread auto& g1f = cg.frag_at(0, 1);
      const thread auto& p0 = P.frag_at(0, 0);
      const thread auto& p1 = P.frag_at(0, 1);
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < 4; ++j) {
        g0f[j] = metal::fma(p0[j], d0v[j], metal::fma(sx0, m0v[j], g0f[j]));
        g0f[4 + j] = metal::fma(p0[4 + j], d0v[j], metal::fma(sx1, m0v[j], g0f[4 + j]));
        g1f[j] = metal::fma(p1[j], d1v[j], metal::fma(sx0, m1v[j], g1f[j]));
        g1f[4 + j] = metal::fma(p1[4 + j], d1v[j], metal::fma(sx1, m1v[j], g1f[4 + j]));
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    STEEL_PRAGMA_UNROLL
    for (int p = 0; p < 2; ++p) {
      if (w_live[p]) {
        const uint wr = row0 + uint(p) * 32u;
        if (lane4 == 0u) {
          float du, mu;
          hy4_gguf_q4k_group_scale(u_sc[p], u_d[p], u_dmin[p], g0, du, mu);
          S_tg[wr] = du;
          M_tg[wr] = mu;
        }
        threadgroup half4* dst =
            reinterpret_cast<threadgroup half4*>(Ws + wr * PAD_W + lane4 * 8u);
        STEEL_PRAGMA_UNROLL
        for (int i = 0; i < 2; ++i) {
          const uint32_t w = u_qs[p][i];
          dst[i] = half4(
              half(w & 0x0Fu),
              half((w >> 8) & 0x0Fu),
              half((w >> 16) & 0x0Fu),
              half((w >> 24) & 0x0Fu));
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (active) {
      NAXTile<half, 2, 2> B;
      B.template load<half, PAD_W, 1>(Ws + tn * PAD_W);
      NAXTile<float, 1, 2> P;
      P.clear();
      tile_matmad_nax(P, A_hi, metal::false_type{}, B, metal::true_type{});
      tile_matmad_nax(P, A_lo, metal::false_type{}, B, metal::true_type{});
      const float4 d0v = *reinterpret_cast<const threadgroup float4*>(S_tg + tn + fn);
      const float4 d1v = *reinterpret_cast<const threadgroup float4*>(S_tg + tn + 16 + fn);
      const float4 m0v = *reinterpret_cast<const threadgroup float4*>(M_tg + tn + fn);
      const float4 m1v = *reinterpret_cast<const threadgroup float4*>(M_tg + tn + 16 + fn);
      thread auto& u0f = cu.frag_at(0, 0);
      thread auto& u1f = cu.frag_at(0, 1);
      const thread auto& p0 = P.frag_at(0, 0);
      const thread auto& p1 = P.frag_at(0, 1);
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < 4; ++j) {
        u0f[j] = metal::fma(p0[j], d0v[j], metal::fma(sx0, m0v[j], u0f[j]));
        u0f[4 + j] = metal::fma(p0[4 + j], d0v[j], metal::fma(sx1, m0v[j], u0f[4 + j]));
        u1f[j] = metal::fma(p1[j], d1v[j], metal::fma(sx0, m1v[j], u1f[j]));
        u1f[4 + j] = metal::fma(p1[4 + j], d1v[j], metal::fma(sx1, m1v[j], u1f[4 + j]));
      }
    }

    // Group g1 (high nibbles): elements base+32..base+63.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    STEEL_PRAGMA_UNROLL
    for (int p = 0; p < 2; ++p) {
      if (w_live[p]) {
        const uint wr = row0 + uint(p) * 32u;
        if (lane4 == 0u) {
          float dg, mg;
          hy4_gguf_q4k_group_scale(g_sc[p], g_d[p], g_dmin[p], g1, dg, mg);
          S_tg[wr] = dg;
          M_tg[wr] = mg;
        }
        threadgroup half4* dst =
            reinterpret_cast<threadgroup half4*>(Ws + wr * PAD_W + lane4 * 8u);
        STEEL_PRAGMA_UNROLL
        for (int i = 0; i < 2; ++i) {
          const uint32_t wh = g_qs[p][i] >> 4;
          dst[i] = half4(
              half(wh & 0x0Fu),
              half((wh >> 8) & 0x0Fu),
              half((wh >> 16) & 0x0Fu),
              half((wh >> 24) & 0x0Fu));
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (active) {
      A_hi.template load<half, PAD_A, 1>(As_hi + tm * PAD_A + 32);
      A_lo.template load<half, PAD_A, 1>(As_lo + tm * PAD_A + 32);
      sx0 = sum_x[uint(tm + fm) * 2u + 1u];
      sx1 = sum_x[uint(tm + fm + 8) * 2u + 1u];
      NAXTile<half, 2, 2> B;
      B.template load<half, PAD_W, 1>(Ws + tn * PAD_W);
      NAXTile<float, 1, 2> P;
      P.clear();
      tile_matmad_nax(P, A_hi, metal::false_type{}, B, metal::true_type{});
      tile_matmad_nax(P, A_lo, metal::false_type{}, B, metal::true_type{});
      const float4 d0v = *reinterpret_cast<const threadgroup float4*>(S_tg + tn + fn);
      const float4 d1v = *reinterpret_cast<const threadgroup float4*>(S_tg + tn + 16 + fn);
      const float4 m0v = *reinterpret_cast<const threadgroup float4*>(M_tg + tn + fn);
      const float4 m1v = *reinterpret_cast<const threadgroup float4*>(M_tg + tn + 16 + fn);
      thread auto& g0f = cg.frag_at(0, 0);
      thread auto& g1f = cg.frag_at(0, 1);
      const thread auto& p0 = P.frag_at(0, 0);
      const thread auto& p1 = P.frag_at(0, 1);
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < 4; ++j) {
        g0f[j] = metal::fma(p0[j], d0v[j], metal::fma(sx0, m0v[j], g0f[j]));
        g0f[4 + j] = metal::fma(p0[4 + j], d0v[j], metal::fma(sx1, m0v[j], g0f[4 + j]));
        g1f[j] = metal::fma(p1[j], d1v[j], metal::fma(sx0, m1v[j], g1f[j]));
        g1f[4 + j] = metal::fma(p1[4 + j], d1v[j], metal::fma(sx1, m1v[j], g1f[4 + j]));
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    STEEL_PRAGMA_UNROLL
    for (int p = 0; p < 2; ++p) {
      if (w_live[p]) {
        const uint wr = row0 + uint(p) * 32u;
        if (lane4 == 0u) {
          float du, mu;
          hy4_gguf_q4k_group_scale(u_sc[p], u_d[p], u_dmin[p], g1, du, mu);
          S_tg[wr] = du;
          M_tg[wr] = mu;
        }
        threadgroup half4* dst =
            reinterpret_cast<threadgroup half4*>(Ws + wr * PAD_W + lane4 * 8u);
        STEEL_PRAGMA_UNROLL
        for (int i = 0; i < 2; ++i) {
          const uint32_t wh = u_qs[p][i] >> 4;
          dst[i] = half4(
              half(wh & 0x0Fu),
              half((wh >> 8) & 0x0Fu),
              half((wh >> 16) & 0x0Fu),
              half((wh >> 24) & 0x0Fu));
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Prefetch next activation strip and next Gate weights while Up g1 MMA runs.
    const int next = base + BK;
    if (next < K) {
      if (live) {
        const device float4* src = reinterpret_cast<const device float4*>(
            x + size_t(token) * size_t(K) + size_t(next) + size_t(ac));
        av[0] = src[0];
        av[1] = src[1];
        av[2] = src[2];
        av[3] = src[3];
      }
      const size_t nblk_off = size_t(next >> 8) * 144u;
      const size_t nq_off =
          16u + size_t((next >> 6) & 3) * 32u + size_t(lane4) * 8u;
      STEEL_PRAGMA_UNROLL
      for (int p = 0; p < 2; ++p) {
        const device uint8_t* gblk = g_row[p] + nblk_off;
        if ((next & 255) == 0) {
          const uint32_t hdr = w_live[p]
              ? reinterpret_cast<const device uint32_t*>(gblk)[lane4]
              : 0u;
          const half2 dm = as_type<half2>(simd_shuffle(hdr, quad_base));
          g_d[p] = float(dm.x);
          g_dmin[p] = float(dm.y);
          g_sc[p][0] = simd_shuffle(hdr, quad_base | 1u);
          g_sc[p][1] = simd_shuffle(hdr, quad_base | 2u);
          g_sc[p][2] = simd_shuffle(hdr, quad_base | 3u);
        }
        if (w_live[p]) {
          const device uint32_t* qsp =
              reinterpret_cast<const device uint32_t*>(gblk + nq_off);
          g_qs[p][0] = qsp[0];
          g_qs[p][1] = qsp[1];
        }
      }
    }

    if (active) {
      NAXTile<half, 2, 2> B;
      B.template load<half, PAD_W, 1>(Ws + tn * PAD_W);
      NAXTile<float, 1, 2> P;
      P.clear();
      tile_matmad_nax(P, A_hi, metal::false_type{}, B, metal::true_type{});
      tile_matmad_nax(P, A_lo, metal::false_type{}, B, metal::true_type{});
      const float4 d0v = *reinterpret_cast<const threadgroup float4*>(S_tg + tn + fn);
      const float4 d1v = *reinterpret_cast<const threadgroup float4*>(S_tg + tn + 16 + fn);
      const float4 m0v = *reinterpret_cast<const threadgroup float4*>(M_tg + tn + fn);
      const float4 m1v = *reinterpret_cast<const threadgroup float4*>(M_tg + tn + 16 + fn);
      thread auto& u0f = cu.frag_at(0, 0);
      thread auto& u1f = cu.frag_at(0, 1);
      const thread auto& p0 = P.frag_at(0, 0);
      const thread auto& p1 = P.frag_at(0, 1);
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < 4; ++j) {
        u0f[j] = metal::fma(p0[j], d0v[j], metal::fma(sx0, m0v[j], u0f[j]));
        u0f[4 + j] = metal::fma(p0[4 + j], d0v[j], metal::fma(sx1, m0v[j], u0f[4 + j]));
        u1f[j] = metal::fma(p1[j], d1v[j], metal::fma(sx0, m1v[j], u1f[j]));
        u1f[4 + j] = metal::fma(p1[4 + j], d1v[j], metal::fma(sx1, m1v[j], u1f[4 + j]));
      }
    }
  }
}

if (valid_m > 0 && valid_n > 0) {
  const float lim = limit[0];
  constexpr int FE = NAXTile<float, 1, 2>::kElemsPerFrag;
  NAXTile<float, 1, 2> D;
  STEEL_PRAGMA_UNROLL
  for (short f = 0; f < 2; ++f) {
    const thread auto& gf = cg.frag_at(0, f);
    const thread auto& uf = cu.frag_at(0, f);
    thread auto& df = D.frag_at(0, f);
    STEEL_PRAGMA_UNROLL
    for (short e = 0; e < FE; ++e) {
      float gv = gf[e];
      float uv = uf[e];
      // Asymmetric clamp: never lower-clamp gate.
      if (lim > 0.0f) {
        gv = metal::min(gv, lim);
        uv = metal::clamp(uv, -lim, lim);
      }
      df[e] = (gv / (1.0f + metal::exp(-gv))) * uv;
    }
  }
  auto out_ptr =
      out + size_t(begin + uint(tm)) * size_t(M) + size_t(col + uint(tn));
  if (valid_m == 16 && valid_n == 32) {
    D.store(out_ptr, M);
  } else {
    D.store_slice(out_ptr, M, short2(0, 0), short2(valid_n, valid_m));
  }
}
)METAL"