R"METAL(
constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 64;
constexpr int PAD_SWZ = 64;
constexpr int PAD_H = 40;
constexpr int TG_HALVES = BM * PAD_SWZ + BN * PAD_SWZ * 2;

const int R = token_rows_shape[0];
const int N_TOK = x_shape[0];
const int K = x_shape[1];
const size_t x_s0 = size_t(x_strides[0]);
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

// 24,576 B <= 32 KiB: As (8 KiB) + Bs_gate (8 KiB) + Bs_up (8 KiB).
alignas(16) threadgroup half tg_mem[TG_HALVES];

NAXTile<float, 1, 2> cg;
NAXTile<float, 1, 2> cu;
cg.clear();
cu.clear();

if (slot < uint(S)) {
  const uint ar = tid >> 2;
  const uint ac4 = tid & 3u;
  bool live = (begin + ar) < end;
  const uint token = live ? token_rows[begin + ar] : 0u;
  live = live && (token < uint(N_TOK));

  // SGs 0..3 own gate weights, SGs 4..7 own up weights (simdgroup-uniform).
  const bool is_up = (sg_idx >= 4u);
  const uint ltid = tid & 127u;
  const uint row0 = ltid >> 2;
  const uint lane4 = ltid & 3u;
  const ushort quad_base = ushort(lane & ~3u);
  const bool w_live[2] = {
      (col + row0) < uint(M),
      (col + row0 + 32u) < uint(M),
  };
  const size_t row_bytes = size_t(K / 256) * 144u;
  const device uint8_t* w_base =
      (is_up ? up_buf : gate_buf) + size_t(offsets[is_up ? 1 : 0]);
  const device uint8_t* w_row[2];
  STEEL_PRAGMA_UNROLL
  for (int p = 0; p < 2; ++p) {
    const uint wr = row0 + uint(p) * 32u;
    const size_t roff =
        (size_t(slot) * size_t(M) + size_t(col + wr)) * row_bytes;
    w_row[p] = w_base + roff;
  }

  float w_d[2], w_dmin[2];
  uint32_t w_sc[2][3];
  uint32_t w_qs[2][2];

  STEEL_PRAGMA_UNROLL
  for (int p = 0; p < 2; ++p) {
    const uint32_t hdr = w_live[p]
        ? reinterpret_cast<const device uint32_t*>(w_row[p])[lane4]
        : 0u;
    const half2 dm = as_type<half2>(simd_shuffle(hdr, quad_base));
    w_d[p] = float(dm.x);
    w_dmin[p] = float(dm.y);
    w_sc[p][0] = simd_shuffle(hdr, quad_base | 1u);
    w_sc[p][1] = simd_shuffle(hdr, quad_base | 2u);
    w_sc[p][2] = simd_shuffle(hdr, quad_base | 3u);
    if (w_live[p]) {
      const device uint32_t* qsp = reinterpret_cast<const device uint32_t*>(
          w_row[p] + 16u + size_t(lane4) * 8u);
      w_qs[p][0] = qsp[0];
      w_qs[p][1] = qsp[1];
    }
  }

  const bool active = (valid_m > 0 && valid_n > 0);
  const short2 coord = BaseNAXFrag::get_coord();
  const short fm = coord.y;
  const short fn = coord.x;

  if constexpr (!HILO) {
    threadgroup half* As = tg_mem;
    threadgroup half* Bs_gate = As + BM * PAD_SWZ;
    threadgroup half* Bs_up = Bs_gate + BN * PAD_SWZ;
    threadgroup half* Bs_dst = is_up ? Bs_up : Bs_gate;

    const uint ac = ac4 * 16u;
    const uint sw_a0 = uint(hy4_swz_c8(int(ac4 * 2u), int(ar)));
    const uint sw_a1 = uint(hy4_swz_c8(int(ac4 * 2u + 1u), int(ar)));
    const uint sw_lo = uint(hy4_swz_c8(int(lane4), int(row0)));
    const uint sw_hi = uint(hy4_swz_c8(int(4u + lane4), int(row0)));

    half4 av[4];
    if (live) {
      const device float4* src = reinterpret_cast<const device float4*>(
          x + size_t(token) * x_s0 + size_t(ac));
      av[0] = half4(src[0]);
      av[1] = half4(src[1]);
      av[2] = half4(src[2]);
      av[3] = half4(src[3]);
    } else {
      threadgroup half4* d0 =
          reinterpret_cast<threadgroup half4*>(As + ar * PAD_SWZ + sw_a0 * 8u);
      threadgroup half4* d1 =
          reinterpret_cast<threadgroup half4*>(As + ar * PAD_SWZ + sw_a1 * 8u);
      d0[0] = half4(0.0h);
      d0[1] = half4(0.0h);
      d1[0] = half4(0.0h);
      d1[1] = half4(0.0h);
    }

    STEEL_PRAGMA_UNROLL
    for (int p = 0; p < 2; ++p) {
      if (!w_live[p]) {
        const uint wr = row0 + uint(p) * 32u;
        threadgroup half4* d_lo = reinterpret_cast<threadgroup half4*>(
            Bs_dst + wr * PAD_SWZ + sw_lo * 8u);
        threadgroup half4* d_hi = reinterpret_cast<threadgroup half4*>(
            Bs_dst + wr * PAD_SWZ + sw_hi * 8u);
        d_lo[0] = half4(0.0h);
        d_lo[1] = half4(0.0h);
        d_hi[0] = half4(0.0h);
        d_hi[1] = half4(0.0h);
      }
    }

    for (int base = 0; base < K; base += BK) {
      const int c = (base >> 6) & 3;
      const int g0 = 2 * c;
      const int g1 = g0 + 1;

      if (base > 0) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }

      if (live) {
        threadgroup half4* d0 = reinterpret_cast<threadgroup half4*>(
            As + ar * PAD_SWZ + sw_a0 * 8u);
        threadgroup half4* d1 = reinterpret_cast<threadgroup half4*>(
            As + ar * PAD_SWZ + sw_a1 * 8u);
        d0[0] = av[0];
        d0[1] = av[1];
        d1[0] = av[2];
        d1[1] = av[3];
      }

      STEEL_PRAGMA_UNROLL
      for (int p = 0; p < 2; ++p) {
        float d0 = 0.0f, m0 = 0.0f, d1 = 0.0f, m1 = 0.0f;
        if (w_live[p] && lane4 == 0u) {
          hy4_gguf_q4k_group_scale(w_sc[p], w_d[p], w_dmin[p], g0, d0, m0);
          hy4_gguf_q4k_group_scale(w_sc[p], w_d[p], w_dmin[p], g1, d1, m1);
        }
        const float d0_f = simd_shuffle(d0, quad_base);
        const float m0_f = simd_shuffle(m0, quad_base);
        const float d1_f = simd_shuffle(d1, quad_base);
        const float m1_f = simd_shuffle(m1, quad_base);
        if (w_live[p]) {
          const uint wr = row0 + uint(p) * 32u;
          threadgroup half4* d_lo = reinterpret_cast<threadgroup half4*>(
              Bs_dst + wr * PAD_SWZ + sw_lo * 8u);
          threadgroup half4* d_hi = reinterpret_cast<threadgroup half4*>(
              Bs_dst + wr * PAD_SWZ + sw_hi * 8u);
          STEEL_PRAGMA_UNROLL
          for (int i = 0; i < 2; ++i) {
            const uchar4 b = as_type<uchar4>(w_qs[p][i]);
            d_lo[i] = half4(metal::fma(float4(b & uchar4(0x0Fu)), d0_f, m0_f));
            d_hi[i] = half4(metal::fma(float4(b >> 4), d1_f, m1_f));
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      const int next = base + BK;
      if (next < K) {
        if (live) {
          const device float4* src = reinterpret_cast<const device float4*>(
              x + size_t(token) * x_s0 + size_t(next) + size_t(ac));
          av[0] = half4(src[0]);
          av[1] = half4(src[1]);
          av[2] = half4(src[2]);
          av[3] = half4(src[3]);
        }
        const size_t nblk_off = size_t(next >> 8) * 144u;
        const size_t nq_off =
            16u + size_t((next >> 6) & 3) * 32u + size_t(lane4) * 8u;
        STEEL_PRAGMA_UNROLL
        for (int p = 0; p < 2; ++p) {
          const device uint8_t* nblk = w_row[p] + nblk_off;
          if ((next & 255) == 0) {
            const uint32_t hdr = w_live[p]
                ? reinterpret_cast<const device uint32_t*>(nblk)[lane4]
                : 0u;
            const half2 dm = as_type<half2>(simd_shuffle(hdr, quad_base));
            w_d[p] = float(dm.x);
            w_dmin[p] = float(dm.y);
            w_sc[p][0] = simd_shuffle(hdr, quad_base | 1u);
            w_sc[p][1] = simd_shuffle(hdr, quad_base | 2u);
            w_sc[p][2] = simd_shuffle(hdr, quad_base | 3u);
          }
          if (w_live[p]) {
            const device uint32_t* qsp =
                reinterpret_cast<const device uint32_t*>(nblk + nq_off);
            w_qs[p][0] = qsp[0];
            w_qs[p][1] = qsp[1];
          }
        }
      }

      if (active) {
        NAXTile<half, 1, 4> A;
        NAXTile<half, 2, 4> B;
        hy4_load_tile_swz64(A, As + size_t(tm) * PAD_SWZ, fm, fn);
        hy4_load_tile_swz64(B, Bs_gate + size_t(tn) * PAD_SWZ, fm, fn);
        tile_matmad_nax(cg, A, metal::false_type{}, B, metal::true_type{});
        hy4_load_tile_swz64(B, Bs_up + size_t(tn) * PAD_SWZ, fm, fn);
        tile_matmad_nax(cu, A, metal::false_type{}, B, metal::true_type{});
      }
    }
  } else {
    // 21,760 B <= 24 KiB: As_hi/lo + Wg + Wu (each 64x40 f16) + scales + sum_x.
    threadgroup half* As_hi = tg_mem;
    threadgroup half* As_lo = As_hi + BM * PAD_H;
    threadgroup half* Wg = As_lo + BM * PAD_H;
    threadgroup half* Wu = Wg + BN * PAD_H;
    threadgroup float* Sg = reinterpret_cast<threadgroup float*>(Wu + BN * PAD_H);
    threadgroup float* Mg = Sg + BN;
    threadgroup float* Su = Mg + BN;
    threadgroup float* Mu = Su + BN;
    threadgroup float* sum_x = Mu + BN;

    threadgroup half* Ws_dst = is_up ? Wu : Wg;
    threadgroup float* S_dst = is_up ? Su : Sg;
    threadgroup float* M_dst = is_up ? Mu : Mg;

    const uint ac_sub = ac4 * 8u;
    if (!live) {
      threadgroup half4* d_hi =
          reinterpret_cast<threadgroup half4*>(As_hi + ar * PAD_H + ac_sub);
      threadgroup half4* d_lo =
          reinterpret_cast<threadgroup half4*>(As_lo + ar * PAD_H + ac_sub);
      d_hi[0] = half4(0.0h);
      d_hi[1] = half4(0.0h);
      d_lo[0] = half4(0.0h);
      d_lo[1] = half4(0.0h);
      if (ac4 == 0u) {
        sum_x[ar] = 0.0f;
      }
    }

    STEEL_PRAGMA_UNROLL
    for (int p = 0; p < 2; ++p) {
      if (!w_live[p]) {
        const uint wr = row0 + uint(p) * 32u;
        threadgroup half4* dst =
            reinterpret_cast<threadgroup half4*>(Ws_dst + wr * PAD_H + lane4 * 8u);
        dst[0] = half4(0.0h);
        dst[1] = half4(0.0h);
        if (lane4 == 0u) {
          S_dst[wr] = 0.0f;
          M_dst[wr] = 0.0f;
        }
      }
    }

    for (int base = 0; base < K; base += BK) {
      const int c = (base >> 6) & 3;
      STEEL_PRAGMA_UNROLL
      for (int sub = 0; sub < 2; ++sub) {
        const int g_idx = 2 * c + sub;
        const int k_off = base + sub * 32;

        if (base > 0 || sub > 0) {
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        float s_local = 0.0f;
        if (live) {
          const device float4* src = reinterpret_cast<const device float4*>(
              x + size_t(token) * x_s0 + size_t(k_off) + size_t(ac_sub));
          threadgroup half4* d_hi =
              reinterpret_cast<threadgroup half4*>(As_hi + ar * PAD_H + ac_sub);
          threadgroup half4* d_lo =
              reinterpret_cast<threadgroup half4*>(As_lo + ar * PAD_H + ac_sub);
          float4 sv = float4(0.0f);
          STEEL_PRAGMA_UNROLL
          for (int i = 0; i < 2; ++i) {
            const float4 v = src[i];
            const half4 hi = half4(v);
            d_hi[i] = hi;
            d_lo[i] = half4(v - float4(hi));
            sv += v;
          }
          s_local = (sv.x + sv.y) + (sv.z + sv.w);
        }
        float s_quad = s_local + simd_shuffle_xor(s_local, 1u);
        s_quad += simd_shuffle_xor(s_quad, 2u);
        if (live && ac4 == 0u) {
          sum_x[ar] = s_quad;
        }

        STEEL_PRAGMA_UNROLL
        for (int p = 0; p < 2; ++p) {
          if (w_live[p]) {
            const uint wr = row0 + uint(p) * 32u;
            if (lane4 == 0u) {
              float dw, mw;
              hy4_gguf_q4k_group_scale(w_sc[p], w_d[p], w_dmin[p], g_idx, dw, mw);
              S_dst[wr] = dw;
              M_dst[wr] = mw;
            }
            threadgroup half4* dst = reinterpret_cast<threadgroup half4*>(
                Ws_dst + wr * PAD_H + lane4 * 8u);
            STEEL_PRAGMA_UNROLL
            for (int i = 0; i < 2; ++i) {
              const uchar4 b = as_type<uchar4>(w_qs[p][i]);
              dst[i] = half4(sub ? (b >> 4) : (b & uchar4(0x0Fu)));
            }
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sub == 1 && base + BK < K) {
          const int next = base + BK;
          const size_t nblk_off = size_t(next >> 8) * 144u;
          const size_t nq_off =
              16u + size_t((next >> 6) & 3) * 32u + size_t(lane4) * 8u;
          STEEL_PRAGMA_UNROLL
          for (int p = 0; p < 2; ++p) {
            const device uint8_t* nblk = w_row[p] + nblk_off;
            if ((next & 255) == 0) {
              const uint32_t hdr = w_live[p]
                  ? reinterpret_cast<const device uint32_t*>(nblk)[lane4]
                  : 0u;
              const half2 dm = as_type<half2>(simd_shuffle(hdr, quad_base));
              w_d[p] = float(dm.x);
              w_dmin[p] = float(dm.y);
              w_sc[p][0] = simd_shuffle(hdr, quad_base | 1u);
              w_sc[p][1] = simd_shuffle(hdr, quad_base | 2u);
              w_sc[p][2] = simd_shuffle(hdr, quad_base | 3u);
            }
            if (w_live[p]) {
              const device uint32_t* qsp =
                  reinterpret_cast<const device uint32_t*>(nblk + nq_off);
              w_qs[p][0] = qsp[0];
              w_qs[p][1] = qsp[1];
            }
          }
        }

        if (active) {
          NAXTile<half, 1, 2> A_hi;
          NAXTile<half, 1, 2> A_lo;
          A_hi.template load<half, PAD_H, 1>(As_hi + size_t(tm) * PAD_H);
          A_lo.template load<half, PAD_H, 1>(As_lo + size_t(tm) * PAD_H);
          const float sx0 = sum_x[uint(tm + fm)];
          const float sx1 = sum_x[uint(tm + fm + 8)];

          {
            NAXTile<half, 2, 2> B;
            B.template load<half, PAD_H, 1>(Wg + size_t(tn) * PAD_H);
            NAXTile<float, 1, 2> P;
            P.clear();
            tile_matmad_nax(P, A_hi, metal::false_type{}, B, metal::true_type{});
            tile_matmad_nax(P, A_lo, metal::false_type{}, B, metal::true_type{});
            const float4 d0v = *reinterpret_cast<const threadgroup float4*>(Sg + tn + fn);
            const float4 d1v = *reinterpret_cast<const threadgroup float4*>(Sg + tn + 16 + fn);
            const float4 m0v = *reinterpret_cast<const threadgroup float4*>(Mg + tn + fn);
            const float4 m1v = *reinterpret_cast<const threadgroup float4*>(Mg + tn + 16 + fn);
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

          {
            NAXTile<half, 2, 2> B;
            B.template load<half, PAD_H, 1>(Wu + size_t(tn) * PAD_H);
            NAXTile<float, 1, 2> P;
            P.clear();
            tile_matmad_nax(P, A_hi, metal::false_type{}, B, metal::true_type{});
            tile_matmad_nax(P, A_lo, metal::false_type{}, B, metal::true_type{});
            const float4 d0v = *reinterpret_cast<const threadgroup float4*>(Su + tn + fn);
            const float4 d1v = *reinterpret_cast<const threadgroup float4*>(Su + tn + 16 + fn);
            const float4 m0v = *reinterpret_cast<const threadgroup float4*>(Mu + tn + fn);
            const float4 m1v = *reinterpret_cast<const threadgroup float4*>(Mu + tn + 16 + fn);
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