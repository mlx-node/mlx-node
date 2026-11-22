R"METAL(
constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = HILO ? 32 : 64;
constexpr int PAD = HILO ? 40 : 72;
constexpr int TG_HALVES = HILO ? (4 * BM * PAD + 10 * BN) : (3 * BM * PAD);

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
const ushort2 coord = hy4_m3_sg_coord(lane);
const ushort fn = coord.x;
const ushort fm = coord.y;
const bool active = (valid_m > 0 && valid_n > 0);

// Default (BK=64): As + Bs_gate + Bs_up = 27,648 B (27.0 KiB, 256T, 2 barriers per 64 K).
// HILO (BK=32): As_hi/lo + Wg + Wu + Sg/Mg/Su/Mu/sum_x = 21,760 B (21.25 KiB).
alignas(16) threadgroup half tg_mem[TG_HALVES];

if constexpr (!HILO) {
  threadgroup half* As = tg_mem;
  threadgroup half* Bs_gate = As + BM * PAD;
  threadgroup half* Bs_up = Bs_gate + BN * PAD;

  simdgroup_float8x8 cg[2][4];
  simdgroup_float8x8 cu[2][4];
  STEEL_PRAGMA_UNROLL
  for (short mi = 0; mi < 2; ++mi) {
    STEEL_PRAGMA_UNROLL
    for (short ni = 0; ni < 4; ++ni) {
      cg[mi][ni] = make_filled_simdgroup_matrix<float, 8>(0.0f);
      cu[mi][ni] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }
  }

  if (slot < uint(S)) {
    // 8 threads per row x 8 elements/thread = 64 K elements; 256 threads cover 32 rows/pass.
    const uint ar0 = tid >> 3;
    const uint ac = (tid & 7u) * 8u;

    bool a_live[2];
    const device float4* x_ptr[2];
    half4 av[2][2];
    STEEL_PRAGMA_UNROLL
    for (int p = 0; p < 2; ++p) {
      const uint ar = ar0 + uint(p) * 32u;
      bool lv = (begin + ar) < end;
      const uint tok = lv ? token_rows[begin + ar] : 0u;
      lv = lv && (tok < uint(N_TOK));
      a_live[p] = lv;
      if (lv) {
        x_ptr[p] = reinterpret_cast<const device float4*>(
            x + size_t(tok) * x_s0 + size_t(ac));
        av[p][0] = half4(x_ptr[p][0]);
        av[p][1] = half4(x_ptr[p][1]);
      } else {
        x_ptr[p] = nullptr;
        threadgroup half4* d =
            reinterpret_cast<threadgroup half4*>(As + ar * PAD + ac);
        d[0] = half4(0.0h);
        d[1] = half4(0.0h);
      }
    }

    // SGs 0..3 own gate weights, SGs 4..7 own up weights (simdgroup-uniform).
    const bool is_up = (sg_idx >= 4u);
    const uint ltid = tid & 127u;
    const uint row0 = ltid >> 2;
    const uint lane4 = ltid & 3u;
    const ushort quad_base = ushort(lane & ~3u);
    threadgroup half* Bs_dst = is_up ? Bs_up : Bs_gate;

    const size_t row_bytes = size_t(K / 256) * 144u;
    const device uint8_t* w_base =
        (is_up ? up_buf : gate_buf) + size_t(offsets[is_up ? 1 : 0]);
    const bool w_live[2] = {
        (col + row0) < uint(M),
        (col + row0 + 32u) < uint(M),
    };
    const device uint8_t* w_row[2];
    float w_d[2], w_dmin[2];
    uint32_t w_sc[2][3];
    uint32_t w_qs[2][2];

    STEEL_PRAGMA_UNROLL
    for (int p = 0; p < 2; ++p) {
      const uint wr = row0 + uint(p) * 32u;
      w_row[p] = w_base +
          (size_t(slot) * size_t(M) + size_t(col + wr)) * row_bytes;
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
      } else {
        threadgroup half4* d0 =
            reinterpret_cast<threadgroup half4*>(Bs_dst + wr * PAD + lane4 * 8u);
        threadgroup half4* d1 = reinterpret_cast<threadgroup half4*>(
            Bs_dst + wr * PAD + 32u + lane4 * 8u);
        d0[0] = half4(0.0h);
        d0[1] = half4(0.0h);
        d1[0] = half4(0.0h);
        d1[1] = half4(0.0h);
      }
    }

    for (int base = 0; base < K; base += 64) {
      const int c = (base >> 6) & 3;
      if (base > 0) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }

      STEEL_PRAGMA_UNROLL
      for (int p = 0; p < 2; ++p) {
        if (a_live[p]) {
          const uint ar = ar0 + uint(p) * 32u;
          threadgroup half4* d =
              reinterpret_cast<threadgroup half4*>(As + ar * PAD + ac);
          d[0] = av[p][0];
          d[1] = av[p][1];
        }
      }

      STEEL_PRAGMA_UNROLL
      for (int p = 0; p < 2; ++p) {
        float dg0 = 0.0f, mg0 = 0.0f, dg1 = 0.0f, mg1 = 0.0f;
        if (w_live[p] && lane4 == 0u) {
          hy4_gguf_q4k_group_scale(w_sc[p], w_d[p], w_dmin[p], 2 * c, dg0, mg0);
          hy4_gguf_q4k_group_scale(
              w_sc[p], w_d[p], w_dmin[p], 2 * c + 1, dg1, mg1);
        }
        const float d_f0 = simd_shuffle(dg0, quad_base);
        const float m_f0 = simd_shuffle(mg0, quad_base);
        const float d_f1 = simd_shuffle(dg1, quad_base);
        const float m_f1 = simd_shuffle(mg1, quad_base);
        if (w_live[p]) {
          const uint wr = row0 + uint(p) * 32u;
          threadgroup half4* dst0 = reinterpret_cast<threadgroup half4*>(
              Bs_dst + wr * PAD + lane4 * 8u);
          threadgroup half4* dst1 = reinterpret_cast<threadgroup half4*>(
              Bs_dst + wr * PAD + 32u + lane4 * 8u);
          STEEL_PRAGMA_UNROLL
          for (int i = 0; i < 2; ++i) {
            const uchar4 b = as_type<uchar4>(w_qs[p][i]);
            dst0[i] = half4(metal::fma(float4(b & uchar4(0x0Fu)), d_f0, m_f0));
            dst1[i] = half4(metal::fma(float4(b >> 4), d_f1, m_f1));
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      const int next_k = base + 64;
      if (next_k < K) {
        STEEL_PRAGMA_UNROLL
        for (int p = 0; p < 2; ++p) {
          if (a_live[p]) {
            x_ptr[p] += 16;
            av[p][0] = half4(x_ptr[p][0]);
            av[p][1] = half4(x_ptr[p][1]);
          }
        }
        const size_t nblk_off = size_t(next_k >> 8) * 144u;
        const size_t nq_off =
            16u + size_t((next_k >> 6) & 3) * 32u + size_t(lane4) * 8u;
        STEEL_PRAGMA_UNROLL
        for (int p = 0; p < 2; ++p) {
          const device uint8_t* nblk = w_row[p] + nblk_off;
          if ((next_k & 255) == 0) {
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
        STEEL_PRAGMA_UNROLL
        for (short kk = 0; kk < BK; kk += 8) {
          simdgroup_half8x8 ma0, ma1;
          simdgroup_load(ma0, As + (tm + 0) * PAD + kk, PAD);
          simdgroup_load(ma1, As + (tm + 8) * PAD + kk, PAD);
          STEEL_PRAGMA_UNROLL
          for (short ni = 0; ni < 4; ++ni) {
            simdgroup_half8x8 mbg;
            simdgroup_load(
                mbg, Bs_gate + (tn + ni * 8) * PAD + kk, PAD, ulong2(0), true);
            simdgroup_multiply_accumulate(cg[0][ni], ma0, mbg, cg[0][ni]);
            simdgroup_multiply_accumulate(cg[1][ni], ma1, mbg, cg[1][ni]);
          }
          STEEL_PRAGMA_UNROLL
          for (short ni = 0; ni < 4; ++ni) {
            simdgroup_half8x8 mbu;
            simdgroup_load(
                mbu, Bs_up + (tn + ni * 8) * PAD + kk, PAD, ulong2(0), true);
            simdgroup_multiply_accumulate(cu[0][ni], ma0, mbu, cu[0][ni]);
            simdgroup_multiply_accumulate(cu[1][ni], ma1, mbu, cu[1][ni]);
          }
        }
      }
    }
  }

  if (active) {
    const float lim = limit[0];
    STEEL_PRAGMA_UNROLL
    for (short mi = 0; mi < 2; ++mi) {
      const short r_off = mi * 8 + short(fm);
      if (r_off < valid_m) {
        device half* row_base =
            out + size_t(begin + uint(tm) + uint(r_off)) * size_t(M) +
            size_t(col + uint(tn));
        STEEL_PRAGMA_UNROLL
        for (short ni = 0; ni < 4; ++ni) {
          hy4_m3_swiglu_store(
              hy4_m3_te(cg[mi][ni]), hy4_m3_te(cu[mi][ni]), lim,
              row_base, ni * 8 + short(fn), valid_n);
        }
      }
    }
  }
} else {
  float2 cg_acc[2][4];
  float2 cu_acc[2][4];
  STEEL_PRAGMA_UNROLL
  for (short mi = 0; mi < 2; ++mi) {
    STEEL_PRAGMA_UNROLL
    for (short ni = 0; ni < 4; ++ni) {
      cg_acc[mi][ni] = float2(0.0f);
      cu_acc[mi][ni] = float2(0.0f);
    }
  }

  if (slot < uint(S)) {
    const uint ar = tid >> 2;
    const uint ac4 = tid & 3u;
    const uint ac = ac4 * 8u;
    bool live = (begin + ar) < end;
    const uint token = live ? token_rows[begin + ar] : 0u;
    live = live && (token < uint(N_TOK));

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
      w_row[p] = w_base +
          (size_t(slot) * size_t(M) + size_t(col + wr)) * row_bytes;
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

    threadgroup half* As_hi = tg_mem;
    threadgroup half* As_lo = As_hi + BM * PAD;
    threadgroup half* Wg = As_lo + BM * PAD;
    threadgroup half* Wu = Wg + BN * PAD;
    threadgroup float* Sg = reinterpret_cast<threadgroup float*>(Wu + BN * PAD);
    threadgroup float* Mg = Sg + BN;
    threadgroup float* Su = Mg + BN;
    threadgroup float* Mu = Su + BN;
    threadgroup float* sum_x = Mu + BN;

    threadgroup half* Ws_dst = is_up ? Wu : Wg;
    threadgroup float* S_dst = is_up ? Su : Sg;
    threadgroup float* M_dst = is_up ? Mu : Mg;

    if (!live) {
      threadgroup half4* d_hi =
          reinterpret_cast<threadgroup half4*>(As_hi + ar * PAD + ac);
      threadgroup half4* d_lo =
          reinterpret_cast<threadgroup half4*>(As_lo + ar * PAD + ac);
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
            reinterpret_cast<threadgroup half4*>(Ws_dst + wr * PAD + lane4 * 8u);
        dst[0] = half4(0.0h);
        dst[1] = half4(0.0h);
        if (lane4 == 0u) {
          S_dst[wr] = 0.0f;
          M_dst[wr] = 0.0f;
        }
      }
    }

    for (int base = 0; base < K; base += 64) {
      const int c = (base >> 6) & 3;
      STEEL_PRAGMA_UNROLL
      for (int sub = 0; sub < 2; ++sub) {
        const int g_idx = 2 * c + sub;
        const int k_off = base + sub * BK;

        if (base > 0 || sub > 0) {
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        float s_local = 0.0f;
        if (live) {
          const device float4* src = reinterpret_cast<const device float4*>(
              x + size_t(token) * x_s0 + size_t(k_off) + size_t(ac));
          threadgroup half4* d_hi =
              reinterpret_cast<threadgroup half4*>(As_hi + ar * PAD + ac);
          threadgroup half4* d_lo =
              reinterpret_cast<threadgroup half4*>(As_lo + ar * PAD + ac);
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
                Ws_dst + wr * PAD + lane4 * 8u);
            STEEL_PRAGMA_UNROLL
            for (int i = 0; i < 2; ++i) {
              const uchar4 b = as_type<uchar4>(w_qs[p][i]);
              dst[i] = half4(sub ? (b >> 4) : (b & uchar4(0x0Fu)));
            }
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sub == 1 && base + 64 < K) {
          const int next = base + 64;
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
          const float sx0 = sum_x[uint(tm) + uint(fm)];
          const float sx1 = sum_x[uint(tm) + 8u + uint(fm)];

          simdgroup_float8x8 pg[2][4];
          simdgroup_float8x8 pu[2][4];
          STEEL_PRAGMA_UNROLL
          for (short mi = 0; mi < 2; ++mi) {
            STEEL_PRAGMA_UNROLL
            for (short ni = 0; ni < 4; ++ni) {
              pg[mi][ni] = make_filled_simdgroup_matrix<float, 8>(0.0f);
              pu[mi][ni] = make_filled_simdgroup_matrix<float, 8>(0.0f);
            }
          }
          STEEL_PRAGMA_UNROLL
          for (short kk = 0; kk < BK; kk += 8) {
            simdgroup_half8x8 ah0, ah1, al0, al1;
            simdgroup_load(ah0, As_hi + (tm + 0) * PAD + kk, PAD);
            simdgroup_load(ah1, As_hi + (tm + 8) * PAD + kk, PAD);
            simdgroup_load(al0, As_lo + (tm + 0) * PAD + kk, PAD);
            simdgroup_load(al1, As_lo + (tm + 8) * PAD + kk, PAD);
            STEEL_PRAGMA_UNROLL
            for (short ni = 0; ni < 4; ++ni) {
              simdgroup_half8x8 bg;
              simdgroup_load(
                  bg, Wg + (tn + ni * 8) * PAD + kk, PAD, ulong2(0), true);
              simdgroup_multiply_accumulate(pg[0][ni], ah0, bg, pg[0][ni]);
              simdgroup_multiply_accumulate(pg[1][ni], ah1, bg, pg[1][ni]);
              simdgroup_multiply_accumulate(pg[0][ni], al0, bg, pg[0][ni]);
              simdgroup_multiply_accumulate(pg[1][ni], al1, bg, pg[1][ni]);
            }
            STEEL_PRAGMA_UNROLL
            for (short ni = 0; ni < 4; ++ni) {
              simdgroup_half8x8 bu;
              simdgroup_load(
                  bu, Wu + (tn + ni * 8) * PAD + kk, PAD, ulong2(0), true);
              simdgroup_multiply_accumulate(pu[0][ni], ah0, bu, pu[0][ni]);
              simdgroup_multiply_accumulate(pu[1][ni], ah1, bu, pu[1][ni]);
              simdgroup_multiply_accumulate(pu[0][ni], al0, bu, pu[0][ni]);
              simdgroup_multiply_accumulate(pu[1][ni], al1, bu, pu[1][ni]);
            }
          }

          STEEL_PRAGMA_UNROLL
          for (short ni = 0; ni < 4; ++ni) {
            const uint cn = uint(tn) + uint(ni) * 8u + uint(fn);
            const float2 dg = float2(Sg[cn], Sg[cn + 1u]);
            const float2 mg = float2(Mg[cn], Mg[cn + 1u]);
            const float2 pg0 = hy4_m3_te(pg[0][ni]);
            const float2 pg1 = hy4_m3_te(pg[1][ni]);
            cg_acc[0][ni] = metal::fma(pg0, dg, metal::fma(float2(sx0), mg, cg_acc[0][ni]));
            cg_acc[1][ni] = metal::fma(pg1, dg, metal::fma(float2(sx1), mg, cg_acc[1][ni]));

            const float2 du = float2(Su[cn], Su[cn + 1u]);
            const float2 mu = float2(Mu[cn], Mu[cn + 1u]);
            const float2 pu0 = hy4_m3_te(pu[0][ni]);
            const float2 pu1 = hy4_m3_te(pu[1][ni]);
            cu_acc[0][ni] = metal::fma(pu0, du, metal::fma(float2(sx0), mu, cu_acc[0][ni]));
            cu_acc[1][ni] = metal::fma(pu1, du, metal::fma(float2(sx1), mu, cu_acc[1][ni]));
          }
        }
      }
    }
  }

  if (active) {
    const float lim = limit[0];
    STEEL_PRAGMA_UNROLL
    for (short mi = 0; mi < 2; ++mi) {
      const short r_off = mi * 8 + short(fm);
      if (r_off < valid_m) {
        device half* row_base =
            out + size_t(begin + uint(tm) + uint(r_off)) * size_t(M) +
            size_t(col + uint(tn));
        STEEL_PRAGMA_UNROLL
        for (short ni = 0; ni < 4; ++ni) {
          hy4_m3_swiglu_store(
              cg_acc[mi][ni], cu_acc[mi][ni], lim,
              row_base, ni * 8 + short(fn), valid_n);
        }
      }
    }
  }
}
)METAL"