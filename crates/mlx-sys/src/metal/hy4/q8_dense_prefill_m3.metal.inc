R"METAL(
constexpr int BM = M_TILE;
constexpr int BN = N_TILE;
constexpr int SG_M = (!FAST_1MMA && SYM_F16 && BM == 64) ? 4 : 2;
constexpr int SG_N = 2;
constexpr int NUM_SG = SG_M * SG_N;
constexpr int NUM_THREADS = NUM_SG * 32;
constexpr int TM = BM / SG_M;
constexpr int TM8 = TM / 8;
constexpr int TN = BN / SG_N;
constexpr int TN8 = TN / 8;
constexpr int BK = 32;
constexpr int PAD = 40;
constexpr int A_HALVES = BM * PAD;
constexpr int W_HALVES = BN * PAD;
constexpr int TG_HALVES = FAST_1MMA
    ? (A_HALVES + W_HALVES)
    : (SYM_F16 ? (2 * A_HALVES + W_HALVES) : (2 * A_HALVES + 2 * W_HALVES));
constexpr int S_TG_ELEMS = (!FAST_1MMA && SYM_F16) ? BN : 1;

const int M = x_shape[0];
const int G = GROUPED ? x_shape[1] : 1;
const int K = GROUPED ? x_shape[2] : x_shape[1];
const int N = w_shape[0] / G;

const uint tile = threadgroup_position_in_grid.y;
const uint begin = tile * uint(BM);
const uint end = min(begin + uint(BM), uint(M));
if (begin >= end) return;
const uint col = threadgroup_position_in_grid.x * uint(BN);
const uint g = GROUPED ? threadgroup_position_in_grid.z : 0u;
if (g >= uint(G)) return;

const uint sg = simdgroup_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;
const uint tid = sg * 32u + lane;
const short tm = short((sg / uint(SG_N)) * uint(TM));
const short tn = short((sg % uint(SG_N)) * uint(TN));
const short valid_m = short(max(0, min(TM, int(end - begin) - int(tm))));
const short valid_n = ALIGNED_N
    ? short(TN)
    : short(max(0, min(TN, N - int(col) - int(tn))));
const ushort2 coord = hy4_m3_sg_coord(lane);
const ushort fn = coord.x;
const ushort fm = coord.y;

// Fast 1-MMA: BM=64, BN=64, BK=32 -> 10.0 KiB TG memory (3 TGs/core, 128T).
// Multi-MMA paths stay <= 15.25 KiB TG memory.
alignas(16) threadgroup half tg_mem[TG_HALVES];
alignas(16) threadgroup float S_tg[S_TG_ELEMS];

threadgroup half* As_hi = tg_mem;
threadgroup half* As_lo = FAST_1MMA ? tg_mem : (As_hi + A_HALVES);
threadgroup half* Ws = FAST_1MMA ? (As_hi + A_HALVES) : (As_lo + A_HALVES);
threadgroup half* Ws_lo = (FAST_1MMA || SYM_F16) ? Ws : (Ws + W_HALVES);

const bool active = (valid_m > 0 && valid_n > 0);
const size_t x_s0 = size_t(x_strides[0]);
const size_t x_g_off = GROUPED ? (size_t(g) * size_t(x_strides[1])) : 0u;

if constexpr (FAST_1MMA) {
  constexpr int A_ROWS_PER_PASS = NUM_THREADS / 4;
  constexpr int A_PASSES = BM / A_ROWS_PER_PASS;
  const uint ar0 = tid >> 2;
  const uint ac4 = tid & 3u;
  const uint ac = ac4 * 8u;

  const device float4* x_src[A_PASSES];
  half4 xh0[A_PASSES];
  half4 xh1[A_PASSES];
  STEEL_PRAGMA_UNROLL
  for (int p = 0; p < A_PASSES; ++p) {
    const uint r = ar0 + uint(p) * uint(A_ROWS_PER_PASS);
    const uint mr = min(begin + r, uint(M - 1));
    x_src[p] = reinterpret_cast<const device float4*>(
        x + size_t(mr) * x_s0 + x_g_off + size_t(ac));
    if constexpr (X_F16) {
      const float4 raw = x_src[p][0];
      xh0[p] = as_type<half4>(raw.xy);
      xh1[p] = as_type<half4>(raw.zw);
    } else {
      xh0[p] = half4(x_src[p][0]);
      xh1[p] = half4(x_src[p][1]);
    }
  }

  // NUM_THREADS == BN * 2 (128 == 64 * 2): two threads share one 32-element Q8 block.
  const uint qr = tid >> 1;
  const uint qhalf = tid & 1u;
  const bool row_ok = ALIGNED_N || (int(col + qr) < N);
  const size_t wrow = size_t(g) * size_t(N) + size_t(col + qr);
  const device uint4* w_ptr =
      reinterpret_cast<const device uint4*>(w) +
      wrow * size_t(K >> 4) + size_t(qhalf);
  auto s_ptr = scales + wrow * size_t(K >> 5);

  uint4 qw0 = uint4(0);
  float qs_f = 0.0f;
  if (row_ok) {
    qw0 = w_ptr[0];
    if (qhalf == 0u) {
      qs_f = float(*s_ptr);
    }
  }
  const ushort pair_base = ushort(lane & ~1u);
  qs_f = simd_shuffle(qs_f, pair_base);

  simdgroup_float8x8 D[TM8][TN8];
  STEEL_PRAGMA_UNROLL
  for (short mi = 0; mi < TM8; ++mi) {
    STEEL_PRAGMA_UNROLL
    for (short ni = 0; ni < TN8; ++ni) {
      D[mi][ni] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }
  }

  for (int base = 0; base < K; base += BK) {
    if (base > 0) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    STEEL_PRAGMA_UNROLL
    for (int p = 0; p < A_PASSES; ++p) {
      const uint r = ar0 + uint(p) * uint(A_ROWS_PER_PASS);
      threadgroup half4* dst =
          reinterpret_cast<threadgroup half4*>(As_hi + r * PAD + ac);
      dst[0] = xh0[p];
      dst[1] = xh1[p];
    }

    threadgroup half4* w_dst =
        reinterpret_cast<threadgroup half4*>(Ws + qr * PAD + qhalf * 16u);
    if (row_ok) {
      const half qs_h = half(qs_f);
      STEEL_PRAGMA_UNROLL
      for (int j = 0; j < 4; ++j) {
        w_dst[j] = half4(as_type<char4>(qw0[j] ^ 0x80808080u)) * qs_h;
      }
    } else {
      STEEL_PRAGMA_UNROLL
      for (int j = 0; j < 4; ++j) {
        w_dst[j] = half4(0.0h);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    w_ptr += 2;
    s_ptr += 1;
    if (base + BK < K) {
      STEEL_PRAGMA_UNROLL
      for (int p = 0; p < A_PASSES; ++p) {
        if constexpr (X_F16) {
          x_src[p] += 4;
          const float4 raw = x_src[p][0];
          xh0[p] = as_type<half4>(raw.xy);
          xh1[p] = as_type<half4>(raw.zw);
        } else {
          x_src[p] += 8;
          xh0[p] = half4(x_src[p][0]);
          xh1[p] = half4(x_src[p][1]);
        }
      }
      if (row_ok) {
        qw0 = w_ptr[0];
        if (qhalf == 0u) {
          qs_f = float(*s_ptr);
        }
      }
      qs_f = simd_shuffle(qs_f, pair_base);
    }

    if (active) {
      STEEL_PRAGMA_UNROLL
      for (short kk = 0; kk < BK; kk += 8) {
        simdgroup_half8x8 ma[TM8];
        STEEL_PRAGMA_UNROLL
        for (short mi = 0; mi < TM8; ++mi) {
          simdgroup_load(ma[mi], As_hi + (tm + mi * 8) * PAD + kk, PAD);
        }
        STEEL_PRAGMA_UNROLL
        for (short ni = 0; ni < TN8; ++ni) {
          simdgroup_half8x8 mb;
          simdgroup_load(
              mb, Ws + (tn + ni * 8) * PAD + kk, PAD, ulong2(0), true);
          STEEL_PRAGMA_UNROLL
          for (short mi = 0; mi < TM8; ++mi) {
            simdgroup_multiply_accumulate(D[mi][ni], ma[mi], mb, D[mi][ni]);
          }
        }
      }
    }
  }

  if (active) {
    STEEL_PRAGMA_UNROLL
    for (short mi = 0; mi < TM8; ++mi) {
      const short r_off = mi * 8 + short(fm);
      if (r_off < valid_m) {
        const size_t row_idx = size_t(begin + uint(tm) + uint(r_off));
        device OUT_T* out_row =
            out + (row_idx * size_t(G) + size_t(g)) * size_t(N) +
            size_t(col + uint(tn));
        const auto aux_row = (EPILOGUE == 1)
            ? (aux + row_idx * size_t(aux_strides[0]) + size_t(col + uint(tn)))
            : (aux + row_idx * size_t(aux_strides[0]) +
               size_t(g) * size_t(N) + size_t(col + uint(tn)));
        STEEL_PRAGMA_UNROLL
        for (short ni = 0; ni < TN8; ++ni) {
          hy4_m3_q8_store_pair<EPILOGUE>(
              hy4_m3_te(D[mi][ni]), out_row, aux_row,
              ni * 8 + short(fn), valid_n);
        }
      }
    }
  }
} else {
  constexpr int A_ROWS_PER_PASS = NUM_THREADS / 8;
  constexpr int A_PASSES = BM / A_ROWS_PER_PASS;
  const uint ar0 = tid >> 3;
  const uint ac4 = tid & 7u;
  const device vec<X_T, 4>* x_src[A_PASSES];
  float4 xv[A_PASSES];
  STEEL_PRAGMA_UNROLL
  for (int p = 0; p < A_PASSES; ++p) {
    const uint r = ar0 + uint(p) * uint(A_ROWS_PER_PASS);
    const uint mr = min(begin + r, uint(M - 1));
    x_src[p] = reinterpret_cast<const device vec<X_T, 4>*>(
        x + size_t(mr) * x_s0 + x_g_off + size_t(ac4 * 4u));
    xv[p] = float4(x_src[p][0]);
  }

  if constexpr (SYM_F16) {
    constexpr int W_TPR = NUM_THREADS / BN;
    const uint qr = tid / uint(W_TPR);
    const uint qw_lane = tid % uint(W_TPR);
    const bool row_ok = ALIGNED_N || (int(col + qr) < N);
    const size_t wrow = size_t(g) * size_t(N) + size_t(col + qr);
    auto s_ptr = scales + wrow * size_t(K >> 5);

    const device uint4* w_ptr4 =
        reinterpret_cast<const device uint4*>(w) +
        wrow * size_t(K >> 4) + size_t(qw_lane);
    const device uint2* w_ptr2 =
        reinterpret_cast<const device uint2*>(w) +
        wrow * size_t(K >> 3) + size_t(qw_lane);

    uint4 qw4 = uint4(0);
    uint2 qw2 = uint2(0);
    float qs = 0.0f;
    if (row_ok) {
      if constexpr (W_TPR == 2) {
        qw4 = w_ptr4[0];
      } else {
        qw2 = w_ptr2[0];
      }
      if (qw_lane == 0u) {
        qs = float(*s_ptr);
      }
    }

    float2 D_acc[TM8][TN8];
    STEEL_PRAGMA_UNROLL
    for (short mi = 0; mi < TM8; ++mi) {
      STEEL_PRAGMA_UNROLL
      for (short ni = 0; ni < TN8; ++ni) {
        D_acc[mi][ni] = float2(0.0f);
      }
    }

    for (int base = 0; base < K; base += BK) {
      if (base > 0) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }

      STEEL_PRAGMA_UNROLL
      for (int p = 0; p < A_PASSES; ++p) {
        const uint r = ar0 + uint(p) * uint(A_ROWS_PER_PASS);
        const float4 v = xv[p];
        const half4 hi = half4(v);
        *reinterpret_cast<threadgroup half4*>(As_hi + r * PAD + ac4 * 4u) = hi;
        *reinterpret_cast<threadgroup half4*>(As_lo + r * PAD + ac4 * 4u) =
            half4(v - float4(hi));
      }

      if (qw_lane == 0u) {
        S_tg[qr] = row_ok ? qs : 0.0f;
      }
      if constexpr (W_TPR == 2) {
        threadgroup half4* dst =
            reinterpret_cast<threadgroup half4*>(Ws + qr * PAD + qw_lane * 16u);
        if (row_ok) {
          STEEL_PRAGMA_UNROLL
          for (int j = 0; j < 4; ++j) {
            dst[j] = half4(as_type<char4>(qw4[j] ^ 0x80808080u));
          }
        } else {
          STEEL_PRAGMA_UNROLL
          for (int j = 0; j < 4; ++j) {
            dst[j] = half4(0.0h);
          }
        }
      } else {
        threadgroup half4* dst =
            reinterpret_cast<threadgroup half4*>(Ws + qr * PAD + qw_lane * 8u);
        if (row_ok) {
          dst[0] = half4(as_type<char4>(qw2.x ^ 0x80808080u));
          dst[1] = half4(as_type<char4>(qw2.y ^ 0x80808080u));
        } else {
          dst[0] = half4(0.0h);
          dst[1] = half4(0.0h);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      w_ptr4 += 2;
      w_ptr2 += 4;
      s_ptr += 1;
      if (base + BK < K) {
        STEEL_PRAGMA_UNROLL
        for (int p = 0; p < A_PASSES; ++p) {
          x_src[p] += 8;
          xv[p] = float4(x_src[p][0]);
        }
        if (row_ok) {
          if constexpr (W_TPR == 2) {
            qw4 = w_ptr4[0];
          } else {
            qw2 = w_ptr2[0];
          }
          if (qw_lane == 0u) {
            qs = float(*s_ptr);
          }
        }
      }

      if (active) {
        simdgroup_float8x8 P[TM8][TN8];
        STEEL_PRAGMA_UNROLL
        for (short mi = 0; mi < TM8; ++mi) {
          STEEL_PRAGMA_UNROLL
          for (short ni = 0; ni < TN8; ++ni) {
            P[mi][ni] = make_filled_simdgroup_matrix<float, 8>(0.0f);
          }
        }
        STEEL_PRAGMA_UNROLL
        for (short kk = 0; kk < BK; kk += 8) {
          simdgroup_half8x8 ah[TM8], al[TM8];
          STEEL_PRAGMA_UNROLL
          for (short mi = 0; mi < TM8; ++mi) {
            simdgroup_load(ah[mi], As_hi + (tm + mi * 8) * PAD + kk, PAD);
            simdgroup_load(al[mi], As_lo + (tm + mi * 8) * PAD + kk, PAD);
          }
          STEEL_PRAGMA_UNROLL
          for (short ni = 0; ni < TN8; ++ni) {
            simdgroup_half8x8 mb;
            simdgroup_load(
                mb, Ws + (tn + ni * 8) * PAD + kk, PAD, ulong2(0), true);
            STEEL_PRAGMA_UNROLL
            for (short mi = 0; mi < TM8; ++mi) {
              simdgroup_multiply_accumulate(P[mi][ni], ah[mi], mb, P[mi][ni]);
              simdgroup_multiply_accumulate(P[mi][ni], al[mi], mb, P[mi][ni]);
            }
          }
        }
        STEEL_PRAGMA_UNROLL
        for (short ni = 0; ni < TN8; ++ni) {
          const uint cn = uint(tn) + uint(ni) * 8u + uint(fn);
          const float2 sc = float2(S_tg[cn], S_tg[cn + 1u]);
          STEEL_PRAGMA_UNROLL
          for (short mi = 0; mi < TM8; ++mi) {
            D_acc[mi][ni] = metal::fma(hy4_m3_te(P[mi][ni]), sc, D_acc[mi][ni]);
          }
        }
      }
    }

    if (active) {
      STEEL_PRAGMA_UNROLL
      for (short mi = 0; mi < TM8; ++mi) {
        const short r_off = mi * 8 + short(fm);
        if (r_off < valid_m) {
          const size_t row_idx = size_t(begin + uint(tm) + uint(r_off));
          device OUT_T* out_row =
              out + (row_idx * size_t(G) + size_t(g)) * size_t(N) +
              size_t(col + uint(tn));
          const auto aux_row = (EPILOGUE == 1)
              ? (aux + row_idx * size_t(aux_strides[0]) + size_t(col + uint(tn)))
              : (aux + row_idx * size_t(aux_strides[0]) +
                 size_t(g) * size_t(N) + size_t(col + uint(tn)));
          STEEL_PRAGMA_UNROLL
          for (short ni = 0; ni < TN8; ++ni) {
            hy4_m3_q8_store_pair<EPILOGUE>(
                D_acc[mi][ni], out_row, aux_row,
                ni * 8 + short(fn), valid_n);
          }
        }
      }
    }
  } else {
    const uint qr = tid >> 1;
    const uint qhalf = tid & 1u;
    const bool row_ok = ALIGNED_N || (int(col + qr) < N);
    const size_t wrow = size_t(g) * size_t(N) + size_t(col + qr);
    const device uint4* w_ptr =
        reinterpret_cast<const device uint4*>(w) +
        wrow * size_t(K >> 4) + size_t(qhalf);
    auto s_ptr = scales + wrow * size_t(K >> 5);
    auto b_ptr = biases + wrow * size_t(K >> 5);

    uint4 qw0 = uint4(0);
    float qs = 0.0f;
    float qbias = 0.0f;
    if (row_ok) {
      qw0 = w_ptr[0];
      if (qhalf == 0u) {
        qs = float(*s_ptr);
        qbias = float(*b_ptr);
      }
    }
    const ushort pair_base = ushort(lane & ~1u);
    qs = simd_shuffle(qs, pair_base);
    qbias = simd_shuffle(qbias, pair_base);

    simdgroup_float8x8 D[TM8][TN8];
    STEEL_PRAGMA_UNROLL
    for (short mi = 0; mi < TM8; ++mi) {
      STEEL_PRAGMA_UNROLL
      for (short ni = 0; ni < TN8; ++ni) {
        D[mi][ni] = make_filled_simdgroup_matrix<float, 8>(0.0f);
      }
    }

    for (int base = 0; base < K; base += BK) {
      if (base > 0) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }

      STEEL_PRAGMA_UNROLL
      for (int p = 0; p < A_PASSES; ++p) {
        const uint r = ar0 + uint(p) * uint(A_ROWS_PER_PASS);
        const float4 v = xv[p];
        const half4 hi = half4(v);
        *reinterpret_cast<threadgroup half4*>(As_hi + r * PAD + ac4 * 4u) = hi;
        *reinterpret_cast<threadgroup half4*>(As_lo + r * PAD + ac4 * 4u) =
            half4(v - float4(hi));
      }

      threadgroup half4* dst =
          reinterpret_cast<threadgroup half4*>(Ws + qr * PAD + qhalf * 16u);
      threadgroup half4* dst_lo =
          reinterpret_cast<threadgroup half4*>(Ws_lo + qr * PAD + qhalf * 16u);
      if (row_ok) {
        const float4 vsc = float4(qs);
        const float4 vbi = float4(qbias);
        STEEL_PRAGMA_UNROLL
        for (int j = 0; j < 4; ++j) {
          const float4 wv = metal::fma(float4(as_type<uchar4>(qw0[j])), vsc, vbi);
          const half4 wh = half4(wv);
          dst[j] = wh;
          dst_lo[j] = half4(wv - float4(wh));
        }
      } else {
        STEEL_PRAGMA_UNROLL
        for (int j = 0; j < 4; ++j) {
          dst[j] = half4(0.0h);
          dst_lo[j] = half4(0.0h);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      w_ptr += 2;
      s_ptr += 1;
      b_ptr += 1;
      if (base + BK < K) {
        STEEL_PRAGMA_UNROLL
        for (int p = 0; p < A_PASSES; ++p) {
          x_src[p] += 8;
          xv[p] = float4(x_src[p][0]);
        }
        if (row_ok) {
          qw0 = w_ptr[0];
          if (qhalf == 0u) {
            qs = float(*s_ptr);
            qbias = float(*b_ptr);
          }
        }
        qs = simd_shuffle(qs, pair_base);
        qbias = simd_shuffle(qbias, pair_base);
      }

      if (active) {
        STEEL_PRAGMA_UNROLL
        for (short kk = 0; kk < BK; kk += 8) {
          simdgroup_half8x8 ah[TM8], al[TM8];
          STEEL_PRAGMA_UNROLL
          for (short mi = 0; mi < TM8; ++mi) {
            simdgroup_load(ah[mi], As_hi + (tm + mi * 8) * PAD + kk, PAD);
            simdgroup_load(al[mi], As_lo + (tm + mi * 8) * PAD + kk, PAD);
          }
          STEEL_PRAGMA_UNROLL
          for (short ni = 0; ni < TN8; ++ni) {
            simdgroup_half8x8 bh, bl;
            simdgroup_load(
                bh, Ws + (tn + ni * 8) * PAD + kk, PAD, ulong2(0), true);
            simdgroup_load(
                bl, Ws_lo + (tn + ni * 8) * PAD + kk, PAD, ulong2(0), true);
            STEEL_PRAGMA_UNROLL
            for (short mi = 0; mi < TM8; ++mi) {
              simdgroup_multiply_accumulate(D[mi][ni], ah[mi], bh, D[mi][ni]);
              simdgroup_multiply_accumulate(D[mi][ni], al[mi], bh, D[mi][ni]);
              simdgroup_multiply_accumulate(D[mi][ni], ah[mi], bl, D[mi][ni]);
            }
          }
        }
      }
    }

    if (active) {
      STEEL_PRAGMA_UNROLL
      for (short mi = 0; mi < TM8; ++mi) {
        const short r_off = mi * 8 + short(fm);
        if (r_off < valid_m) {
          const size_t row_idx = size_t(begin + uint(tm) + uint(r_off));
          device OUT_T* out_row =
              out + (row_idx * size_t(G) + size_t(g)) * size_t(N) +
              size_t(col + uint(tn));
          const auto aux_row = (EPILOGUE == 1)
              ? (aux + row_idx * size_t(aux_strides[0]) + size_t(col + uint(tn)))
              : (aux + row_idx * size_t(aux_strides[0]) +
                 size_t(g) * size_t(N) + size_t(col + uint(tn)));
          STEEL_PRAGMA_UNROLL
          for (short ni = 0; ni < TN8; ++ni) {
            hy4_m3_q8_store_pair<EPILOGUE>(
                hy4_m3_te(D[mi][ni]), out_row, aux_row,
                ni * 8 + short(fn), valid_n);
          }
        }
      }
    }
  }
}
)METAL"