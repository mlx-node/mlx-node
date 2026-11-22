R"METAL(
constexpr int BM = M_TILE;
constexpr int BN = N_TILE;
constexpr int SG_M = 2;
constexpr int SG_N = (FAST_1MMA && BN == 128) ? 4 : 2;
constexpr int NUM_SG = SG_M * SG_N;
constexpr int NUM_THREADS = NUM_SG * 32;
constexpr int TM = BM / SG_M;
constexpr int TM_F = TM / 16;
constexpr int TN = BN / SG_N;
constexpr int TN_F = TN / 16;
constexpr int BK = FAST_1MMA ? 64 : 32;
constexpr int PAD = FAST_1MMA ? 64 : 40;
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

// FAST_1MMA 64x128x64: As (8 KiB) + Ws (16 KiB) = 24 KiB <= 32 KiB; 4-SG paths <= 16 KiB.
alignas(16) threadgroup C tg_mem[TG_HALVES];
alignas(16) threadgroup float S_tg[S_TG_ELEMS];

threadgroup C* As_hi = tg_mem;
threadgroup C* As_lo = FAST_1MMA ? tg_mem : (As_hi + A_HALVES);
threadgroup C* Ws = FAST_1MMA ? (As_hi + A_HALVES) : (As_lo + A_HALVES);
threadgroup C* Ws_lo = (FAST_1MMA || SYM_F16) ? Ws : (Ws + W_HALVES);

NAXTile<float, TM_F, TN_F> D;
D.clear();

const bool active = (valid_m > 0 && valid_n > 0);
const size_t x_s0 = size_t(x_strides[0]);
const size_t x_g_off = GROUPED ? (size_t(g) * size_t(x_strides[1])) : 0u;

if constexpr (FAST_1MMA) {
  constexpr int A_ROWS_PER_PASS = NUM_THREADS / 8;
  constexpr int A_PASSES = BM / A_ROWS_PER_PASS;
  const uint ar0 = tid >> 3;
  const uint ac8 = tid & 7u;

  const device float4* x_src[A_PASSES];
  half4 xh0[A_PASSES];
  half4 xh1[A_PASSES];
  STEEL_PRAGMA_UNROLL
  for (int p = 0; p < A_PASSES; ++p) {
    const uint r = ar0 + uint(p) * uint(A_ROWS_PER_PASS);
    const uint mr = min(begin + r, uint(M - 1));
    x_src[p] = reinterpret_cast<const device float4*>(
        x + size_t(mr) * x_s0 + x_g_off + size_t(ac8 * 8u));
    if constexpr (X_F16) {
      const float4 raw = x_src[p][0];
      xh0[p] = as_type<half4>(raw.xy);
      xh1[p] = as_type<half4>(raw.zw);
    } else {
      xh0[p] = half4(x_src[p][0]);
      xh1[p] = half4(x_src[p][1]);
    }
  }

  // NUM_THREADS == BN * 2: each thread owns one 32-element Q8_0 block.
  const uint qr = tid >> 1;
  const uint qblk = tid & 1u;
  const bool row_ok = ALIGNED_N || (int(col + qr) < N);
  const size_t wrow = size_t(g) * size_t(N) + size_t(col + qr);
  const device uint4* w_ptr =
      reinterpret_cast<const device uint4*>(w) +
      wrow * size_t(K >> 4) + size_t(qblk * 2u);
  auto s_ptr = scales + wrow * size_t(K >> 5) + size_t(qblk);

  uint4 qw0 = uint4(0);
  uint4 qw1 = uint4(0);
  half qs_h = half(0.0h);
  if (row_ok) {
    qw0 = w_ptr[0];
    qw1 = w_ptr[1];
    qs_h = half(*s_ptr);
  }

  const short2 coord = BaseNAXFrag::get_coord();
  const short fm = coord.y;
  const short fn = coord.x;

  for (int base = 0; base < K; base += 64) {
    if (base > 0) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    STEEL_PRAGMA_UNROLL
    for (int p = 0; p < A_PASSES; ++p) {
      const uint r = ar0 + uint(p) * uint(A_ROWS_PER_PASS);
      const uint sw = uint(hy4_swz_c8(int(ac8), int(r)));
      threadgroup half4* dst =
          reinterpret_cast<threadgroup half4*>(As_hi + r * 64u + sw * 8u);
      dst[0] = xh0[p];
      dst[1] = xh1[p];
    }

    const uint c8_base = qblk * 4u;
    if (row_ok) {
      STEEL_PRAGMA_UNROLL
      for (int g8 = 0; g8 < 2; ++g8) {
        const uint sw = uint(hy4_swz_c8(int(c8_base) + g8, int(qr)));
        threadgroup half4* dst =
            reinterpret_cast<threadgroup half4*>(Ws + qr * 64u + sw * 8u);
        dst[0] = half4(as_type<char4>(qw0[2 * g8] ^ 0x80808080u)) * qs_h;
        dst[1] = half4(as_type<char4>(qw0[2 * g8 + 1] ^ 0x80808080u)) * qs_h;
      }
      STEEL_PRAGMA_UNROLL
      for (int g8 = 0; g8 < 2; ++g8) {
        const uint sw = uint(hy4_swz_c8(int(c8_base) + 2 + g8, int(qr)));
        threadgroup half4* dst =
            reinterpret_cast<threadgroup half4*>(Ws + qr * 64u + sw * 8u);
        dst[0] = half4(as_type<char4>(qw1[2 * g8] ^ 0x80808080u)) * qs_h;
        dst[1] = half4(as_type<char4>(qw1[2 * g8 + 1] ^ 0x80808080u)) * qs_h;
      }
    } else {
      STEEL_PRAGMA_UNROLL
      for (int g8 = 0; g8 < 4; ++g8) {
        const uint sw = uint(hy4_swz_c8(int(c8_base) + g8, int(qr)));
        threadgroup half4* dst =
            reinterpret_cast<threadgroup half4*>(Ws + qr * 64u + sw * 8u);
        dst[0] = half4(0.0h);
        dst[1] = half4(0.0h);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    w_ptr += 4;
    s_ptr += 2;
    if (base + 64 < K) {
      STEEL_PRAGMA_UNROLL
      for (int p = 0; p < A_PASSES; ++p) {
        if constexpr (X_F16) {
          x_src[p] += 8;
          const float4 raw = x_src[p][0];
          xh0[p] = as_type<half4>(raw.xy);
          xh1[p] = as_type<half4>(raw.zw);
        } else {
          x_src[p] += 16;
          xh0[p] = half4(x_src[p][0]);
          xh1[p] = half4(x_src[p][1]);
        }
      }
      if (row_ok) {
        qw0 = w_ptr[0];
        qw1 = w_ptr[1];
        qs_h = half(*s_ptr);
      }
    }

    if (active) {
      NAXTile<half, TM_F, 4> A_hi;
      NAXTile<half, 2, 4> B_hi;
      hy4_load_tile_swz64(A_hi, As_hi + size_t(tm) * 64u, fm, fn);
      hy4_load_tile_swz64(B_hi, Ws + size_t(tn) * 64u, fm, fn);
      tile_matmad_nax(
          D,
          A_hi,
          metal::bool_constant<false>{},
          B_hi,
          metal::bool_constant<true>{});
    }
  }
} else {
  constexpr int A_PASSES = BM / 16;
  const uint ar0 = tid >> 3;
  const uint ac4 = tid & 7u;
  const device vec<X_T, 4>* x_src[A_PASSES];
  float4 xv[A_PASSES];
  STEEL_PRAGMA_UNROLL
  for (int p = 0; p < A_PASSES; ++p) {
    const uint r = ar0 + uint(p) * 16u;
    const uint mr = min(begin + r, uint(M - 1));
    x_src[p] = reinterpret_cast<const device vec<X_T, 4>*>(
        x + size_t(mr) * x_s0 + x_g_off + size_t(ac4 * 4u));
    xv[p] = float4(x_src[p][0]);
  }

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
      if constexpr (!SYM_F16) {
        qbias = float(*b_ptr);
      }
    }
  }
  const ushort pair_base = ushort(lane & ~1u);
  qs = simd_shuffle(qs, pair_base);
  if constexpr (!SYM_F16) {
    qbias = simd_shuffle(qbias, pair_base);
  }

  const short fn = BaseNAXFrag::get_coord().x;

  for (int base = 0; base < K; base += 32) {
    if (base > 0) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    STEEL_PRAGMA_UNROLL
    for (int p = 0; p < A_PASSES; ++p) {
      const uint r = ar0 + uint(p) * 16u;
      const float4 v = xv[p];
      const vec<C, 4> hi = vec<C, 4>(v);
      *reinterpret_cast<threadgroup vec<C, 4>*>(As_hi + r * PAD + ac4 * 4u) = hi;
      *reinterpret_cast<threadgroup vec<C, 4>*>(As_lo + r * PAD + ac4 * 4u) =
          vec<C, 4>(v - float4(hi));
    }

    if constexpr (SYM_F16) {
      if (qhalf == 0u) {
        S_tg[qr] = row_ok ? qs : 0.0f;
      }
      threadgroup vec<C, 4>* dst =
          reinterpret_cast<threadgroup vec<C, 4>*>(Ws + qr * PAD + qhalf * 16u);
      if (row_ok) {
        STEEL_PRAGMA_UNROLL
        for (int j = 0; j < 4; ++j) {
          dst[j] = vec<C, 4>(as_type<char4>(qw0[j] ^ 0x80808080u));
        }
      } else {
        STEEL_PRAGMA_UNROLL
        for (int j = 0; j < 4; ++j) {
          dst[j] = vec<C, 4>(C(0));
        }
      }
    } else {
      threadgroup vec<C, 4>* dst =
          reinterpret_cast<threadgroup vec<C, 4>*>(Ws + qr * PAD + qhalf * 16u);
      threadgroup vec<C, 4>* dst_lo =
          reinterpret_cast<threadgroup vec<C, 4>*>(Ws_lo + qr * PAD + qhalf * 16u);
      if (row_ok) {
        const float4 vsc = float4(qs);
        const float4 vbi = float4(qbias);
        STEEL_PRAGMA_UNROLL
        for (int j = 0; j < 4; ++j) {
          const float4 wv = metal::fma(float4(as_type<uchar4>(qw0[j])), vsc, vbi);
          const vec<C, 4> wh = vec<C, 4>(wv);
          dst[j] = wh;
          dst_lo[j] = vec<C, 4>(wv - float4(wh));
        }
      } else {
        STEEL_PRAGMA_UNROLL
        for (int j = 0; j < 4; ++j) {
          dst[j] = vec<C, 4>(C(0));
          dst_lo[j] = vec<C, 4>(C(0));
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    w_ptr += 2;
    s_ptr += 1;
    if constexpr (!SYM_F16) {
      b_ptr += 1;
    }
    if (base + 32 < K) {
      STEEL_PRAGMA_UNROLL
      for (int p = 0; p < A_PASSES; ++p) {
        x_src[p] += 8;
        xv[p] = float4(x_src[p][0]);
      }
      if (row_ok) {
        qw0 = w_ptr[0];
        if (qhalf == 0u) {
          qs = float(*s_ptr);
          if constexpr (!SYM_F16) {
            qbias = float(*b_ptr);
          }
        }
      }
      qs = simd_shuffle(qs, pair_base);
      if constexpr (!SYM_F16) {
        qbias = simd_shuffle(qbias, pair_base);
      }
    }

    if (active) {
      NAXTile<C, TM_F, 2> A_hi;
      NAXTile<C, TM_F, 2> A_lo;
      NAXTile<C, 2, 2> B_hi;
      A_hi.template load<C, PAD, 1>(As_hi + size_t(tm) * size_t(PAD));
      A_lo.template load<C, PAD, 1>(As_lo + size_t(tm) * size_t(PAD));
      B_hi.template load<C, PAD, 1>(Ws + size_t(tn) * size_t(PAD));
      if constexpr (SYM_F16) {
        NAXTile<float, TM_F, 2> P;
        P.clear();
        tile_matmad_nax(
            P,
            A_hi,
            metal::bool_constant<false>{},
            B_hi,
            metal::bool_constant<true>{});
        tile_matmad_nax(
            P,
            A_lo,
            metal::bool_constant<false>{},
            B_hi,
            metal::bool_constant<true>{});
        const float4 sc0 = *reinterpret_cast<const threadgroup float4*>(S_tg + tn + fn);
        const float4 sc1 = *reinterpret_cast<const threadgroup float4*>(S_tg + tn + 16 + fn);
        STEEL_PRAGMA_UNROLL
        for (short mm = 0; mm < TM_F; ++mm) {
          thread auto& d0 = D.frag_at(mm, 0);
          thread auto& d1 = D.frag_at(mm, 1);
          const thread auto& p0 = P.frag_at(mm, 0);
          const thread auto& p1 = P.frag_at(mm, 1);
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < 4; ++j) {
            d0[j] = metal::fma(p0[j], sc0[j], d0[j]);
            d0[4 + j] = metal::fma(p0[4 + j], sc0[j], d0[4 + j]);
            d1[j] = metal::fma(p1[j], sc1[j], d1[j]);
            d1[4 + j] = metal::fma(p1[4 + j], sc1[j], d1[4 + j]);
          }
        }
      } else {
        NAXTile<C, 2, 2> B_lo;
        B_lo.template load<C, PAD, 1>(Ws_lo + size_t(tn) * size_t(PAD));
        tile_matmad_nax(
            D,
            A_hi,
            metal::bool_constant<false>{},
            B_hi,
            metal::bool_constant<true>{});
        tile_matmad_nax(
            D,
            A_lo,
            metal::bool_constant<false>{},
            B_hi,
            metal::bool_constant<true>{});
        tile_matmad_nax(
            D,
            A_hi,
            metal::bool_constant<false>{},
            B_lo,
            metal::bool_constant<true>{});
      }
    }
  }
}

if (active) {
  if constexpr (EPILOGUE == 1) {
    const int aux_ld = int(aux_strides[0]);
    const auto aux_ptr =
        aux + size_t(begin + uint(tm)) * size_t(aux_strides[0]) +
        size_t(col + uint(tn));
    NAXTile<float, TM_F, TN_F> A_aux;
    if (valid_m == TM && valid_n == TN) {
      A_aux.load(aux_ptr, aux_ld);
    } else {
      A_aux.load_safe(aux_ptr, aux_ld, short2(valid_n, valid_m));
    }
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < NAXTile<float, TM_F, TN_F>::kElemsPerTile; ++i) {
      const float gv = D.elems()[i];
      D.elems()[i] = (1.0f / (1.0f + metal::exp(-gv))) * A_aux.elems()[i];
    }
  } else if constexpr (EPILOGUE == 2) {
    const int aux_ld = int(aux_strides[0]);
    const auto aux_ptr =
        aux + size_t(begin + uint(tm)) * size_t(aux_strides[0]) +
        size_t(g) * size_t(N) + size_t(col + uint(tn));
    NAXTile<float, TM_F, TN_F> A_aux;
    if (valid_m == TM && valid_n == TN) {
      A_aux.load(aux_ptr, aux_ld);
    } else {
      A_aux.load_safe(aux_ptr, aux_ld, short2(valid_n, valid_m));
    }
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < NAXTile<float, TM_F, TN_F>::kElemsPerTile; ++i) {
      const float gv = A_aux.elems()[i];
      D.elems()[i] *= 1.0f / (1.0f + metal::exp(-gv));
    }
  }

  const int out_ld = GROUPED ? (G * N) : N;
  device OUT_T* out_base =
      out + (size_t(begin + uint(tm)) * size_t(G) + size_t(g)) * size_t(N) +
      size_t(col + uint(tn));
  if (valid_m == TM && valid_n == TN) {
    D.store(out_base, out_ld);
  } else {
    D.store_slice(out_base, out_ld, short2(0, 0), short2(valid_n, valid_m));
  }
}
)METAL"