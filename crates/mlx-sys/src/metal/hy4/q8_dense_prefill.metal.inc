R"METAL(
constexpr int BM = M_TILE;
constexpr int TM = BM / 2;
constexpr int TM_F = TM / 16;
constexpr int BN = N_TILE;
constexpr int TN = BN / 2;
constexpr int TN_F = TN / 16;
constexpr int BK = 32;
constexpr int PAD = 40;
constexpr int A_PASSES = BM / 16;
constexpr int A_HALVES = BM * PAD;
constexpr int W_HALVES = BN * PAD;
constexpr int TG_HALVES = FAST_1MMA
    ? (A_HALVES + W_HALVES)
    : (SYM_F16 ? (2 * A_HALVES + W_HALVES) : (2 * A_HALVES + 2 * W_HALVES));
constexpr int S_TG_ELEMS = SYM_F16 ? BN : 1;

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
const short tm = short((sg / 2u) * uint(TM));
const short tn = short((sg % 2u) * uint(TN));
const short valid_m = ALIGNED_M
    ? short(TM)
    : short(max(0, min(TM, int(end - begin) - int(tm))));
const short valid_n = ALIGNED_N
    ? short(TN)
    : short(max(0, min(TN, N - int(col) - int(tn))));

// Every specialization fits <= 15,872 B <= 16 KiB (2 TGs/core).
alignas(16) threadgroup C tg_mem[TG_HALVES];
alignas(16) threadgroup float S_tg[S_TG_ELEMS];

threadgroup C* As_hi = tg_mem;
threadgroup C* As_lo = FAST_1MMA ? tg_mem : (As_hi + A_HALVES);
threadgroup C* Ws = FAST_1MMA ? (As_hi + A_HALVES) : (As_lo + A_HALVES);
threadgroup C* Ws_lo = (FAST_1MMA || SYM_F16) ? Ws : (Ws + W_HALVES);

NAXTile<float, TM_F, TN_F> D;
D.clear();

const uint ar0 = tid >> 3;
const uint ac4 = tid & 7u;
const size_t x_s0 = size_t(x_strides[0]);
const size_t x_g_off = GROUPED ? (size_t(g) * size_t(x_strides[1])) : 0u;
const device float4* x_src[A_PASSES];
bool a_live[A_PASSES];
float4 xv[A_PASSES];
STEEL_PRAGMA_UNROLL
for (int p = 0; p < A_PASSES; ++p) {
  const uint r = ar0 + uint(p) * 16u;
  const uint mr = begin + r;
  a_live[p] = ALIGNED_M || (mr < uint(M));
  x_src[p] = reinterpret_cast<const device float4*>(
      x + size_t(min(mr, uint(M - 1))) * x_s0 + x_g_off + size_t(ac4 * 4u));
  if (a_live[p]) {
    xv[p] = x_src[p][0];
  } else {
    *reinterpret_cast<threadgroup vec<C, 4>*>(As_hi + r * PAD + ac4 * 4u) = vec<C, 4>(C(0));
    if constexpr (!FAST_1MMA) {
      *reinterpret_cast<threadgroup vec<C, 4>*>(As_lo + r * PAD + ac4 * 4u) = vec<C, 4>(C(0));
    }
  }
}

const uint qr = (BN == 128) ? tid : (tid >> 1);
const uint qhalf = (BN == 128) ? 0u : (tid & 1u);
const bool row_ok = ALIGNED_N || (int(col + qr) < N);
const size_t wrow = size_t(g) * size_t(N) + size_t(col + qr);
const device uint4* w_ptr =
    reinterpret_cast<const device uint4*>(w) +
    wrow * size_t(K >> 4) + size_t(qhalf);
auto s_ptr = scales + wrow * size_t(K >> 5);
auto b_ptr = biases + wrow * size_t(K >> 5);

uint4 qw0 = uint4(0);
uint4 qw1 = uint4(0);
float qs = 0.0f;
float qbias = 0.0f;
if (row_ok) {
  qw0 = w_ptr[0];
  if constexpr (BN == 128) {
    qw1 = w_ptr[1];
  }
  if (qhalf == 0u) {
    qs = float(*s_ptr);
    if constexpr (!SYM_F16) {
      qbias = float(*b_ptr);
    }
  }
}
if constexpr (BN == 64) {
  const ushort pair_base = ushort(lane & ~1u);
  qs = simd_shuffle(qs, pair_base);
  if constexpr (!SYM_F16) {
    qbias = simd_shuffle(qbias, pair_base);
  }
}

const bool active = (ALIGNED_M && ALIGNED_N) || (valid_m > 0 && valid_n > 0);
const short fn = BaseNAXFrag::get_coord().x;

for (int base = 0; base < K; base += BK) {
  if (base > 0) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  STEEL_PRAGMA_UNROLL
  for (int p = 0; p < A_PASSES; ++p) {
    if (a_live[p]) {
      const uint r = ar0 + uint(p) * 16u;
      const float4 v = xv[p];
      const vec<C, 4> hi = vec<C, 4>(v);
      *reinterpret_cast<threadgroup vec<C, 4>*>(As_hi + r * PAD + ac4 * 4u) = hi;
      if constexpr (!FAST_1MMA) {
        *reinterpret_cast<threadgroup vec<C, 4>*>(As_lo + r * PAD + ac4 * 4u) =
            vec<C, 4>(v - float4(hi));
      }
    }
  }

  if constexpr (SYM_F16) {
    if (qhalf == 0u) {
      S_tg[qr] = row_ok ? qs : 0.0f;
    }
    if constexpr (BN == 128) {
      threadgroup vec<C, 4>* dst =
          reinterpret_cast<threadgroup vec<C, 4>*>(Ws + qr * PAD);
      if (row_ok) {
        STEEL_PRAGMA_UNROLL
        for (int j = 0; j < 4; ++j) {
          dst[j] = vec<C, 4>(as_type<char4>(qw0[j] ^ 0x80808080u));
        }
        STEEL_PRAGMA_UNROLL
        for (int j = 0; j < 4; ++j) {
          dst[4 + j] = vec<C, 4>(as_type<char4>(qw1[j] ^ 0x80808080u));
        }
      } else {
        STEEL_PRAGMA_UNROLL
        for (int j = 0; j < 8; ++j) {
          dst[j] = vec<C, 4>(C(0));
        }
      }
    } else {
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
  if (base + BK < K) {
    STEEL_PRAGMA_UNROLL
    for (int p = 0; p < A_PASSES; ++p) {
      x_src[p] += 8;
      if (a_live[p]) {
        xv[p] = x_src[p][0];
      }
    }
    if (row_ok) {
      qw0 = w_ptr[0];
      if constexpr (BN == 128) {
        qw1 = w_ptr[1];
      }
      if (qhalf == 0u) {
        qs = float(*s_ptr);
        if constexpr (!SYM_F16) {
          qbias = float(*b_ptr);
        }
      }
    }
    if constexpr (BN == 64) {
      const ushort pair_base = ushort(lane & ~1u);
      qs = simd_shuffle(qs, pair_base);
      if constexpr (!SYM_F16) {
        qbias = simd_shuffle(qbias, pair_base);
      }
    }
  }

  if (active) {
    NAXTile<C, TM_F, 2> A_hi;
    A_hi.template load<C, PAD, 1>(As_hi + size_t(tm) * size_t(PAD));
    if constexpr (FAST_1MMA) {
      STEEL_PRAGMA_UNROLL
      for (short ns = 0; ns < TN_F / 2; ++ns) {
        const short n_off = tn + ns * 32;
        NAXTile<C, 2, 2> B_sub;
        B_sub.template load<C, PAD, 1>(Ws + size_t(n_off) * size_t(PAD));
        NAXTile<float, TM_F, 2> P;
        P.clear();
        tile_matmad_nax(
            P,
            A_hi,
            metal::bool_constant<false>{},
            B_sub,
            metal::bool_constant<true>{});
        const float4 sc0 = *reinterpret_cast<const threadgroup float4*>(S_tg + n_off + fn);
        const float4 sc1 = *reinterpret_cast<const threadgroup float4*>(S_tg + n_off + 16 + fn);
        STEEL_PRAGMA_UNROLL
        for (short mm = 0; mm < TM_F; ++mm) {
          thread auto& d0 = D.frag_at(mm, ns * 2);
          thread auto& d1 = D.frag_at(mm, ns * 2 + 1);
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
      }
    } else if constexpr (SYM_F16) {
      NAXTile<C, TM_F, 2> A_lo;
      NAXTile<C, 2, 2> B_hi;
      A_lo.template load<C, PAD, 1>(As_lo + size_t(tm) * size_t(PAD));
      B_hi.template load<C, PAD, 1>(Ws + size_t(tn) * size_t(PAD));
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
      NAXTile<C, TM_F, 2> A_lo;
      NAXTile<C, 2, 2> B_hi;
      NAXTile<C, 2, 2> B_lo;
      A_lo.template load<C, PAD, 1>(As_lo + size_t(tm) * size_t(PAD));
      B_hi.template load<C, PAD, 1>(Ws + size_t(tn) * size_t(PAD));
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

if (active) {
  if constexpr (EPILOGUE == 1) {
    const int aux_ld = int(aux_strides[0]);
    const device float* aux_ptr =
        aux + size_t(begin + uint(tm)) * size_t(aux_strides[0]) +
        size_t(col + uint(tn));
    NAXTile<float, TM_F, TN_F> A_aux;
    if constexpr (ALIGNED_M && ALIGNED_N) {
      A_aux.load(aux_ptr, aux_ld);
    } else if (valid_m == TM && valid_n == TN) {
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
    const device float* aux_ptr =
        aux + size_t(begin + uint(tm)) * size_t(aux_strides[0]) +
        size_t(g) * size_t(N) + size_t(col + uint(tn));
    NAXTile<float, TM_F, TN_F> A_aux;
    if constexpr (ALIGNED_M && ALIGNED_N) {
      A_aux.load(aux_ptr, aux_ld);
    } else if (valid_m == TM && valid_n == TN) {
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
  device float* out_base =
      out + (size_t(begin + uint(tm)) * size_t(G) + size_t(g)) * size_t(N) +
      size_t(col + uint(tn));
  if constexpr (ALIGNED_M && ALIGNED_N) {
    D.store(out_base, out_ld);
  } else if (valid_m == TM && valid_n == TN) {
    D.store(out_base, out_ld);
  } else {
    D.store_slice(out_base, out_ld, short2(0, 0), short2(valid_n, valid_m));
  }
}
)METAL"