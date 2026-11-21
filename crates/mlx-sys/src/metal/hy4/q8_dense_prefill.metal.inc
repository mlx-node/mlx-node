R"METAL(
constexpr int BM = M_TILE;
constexpr int TM = BM / 2;
constexpr int BN = 64;
constexpr int BK = 64;
constexpr int PAD = 64 + 16 / sizeof(C);
constexpr int WS_LO_ELEMS = SYM_F16 ? 1 : (BN * PAD);
constexpr int S_TG_ELEMS = SYM_F16 ? (2 * BN) : 1;

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
const short tn = short((sg % 2u) * 32u);
const short valid_m = ALIGNED_M
    ? short(TM)
    : short(max(0, min(TM, int(end - begin) - int(tm))));
const short valid_n = ALIGNED_N
    ? short(32)
    : short(max(0, min(32, N - int(col) - int(tn))));

alignas(16) threadgroup C Ws[BN * PAD];
alignas(16) threadgroup C Ws_lo[WS_LO_ELEMS];
alignas(16) threadgroup float S_tg[S_TG_ELEMS];
NAXTile<float, TM / 16, 2> D;
D.clear();

const uint qr = tid / 2u;
const uint qblk = tid % 2u;
const uint qb = qblk * 32u;
const bool row_ok = ALIGNED_N || (int(col + qr) < N);
const size_t wrow = size_t(g) * size_t(N) + size_t(col + qr);
const device uint4* w_ptr =
    reinterpret_cast<const device uint4*>(w) +
    wrow * size_t(K >> 4) + size_t(qb >> 4);
auto s_ptr = scales + wrow * size_t(K >> 5) + size_t(qb >> 5);
auto b_ptr = biases + wrow * size_t(K >> 5) + size_t(qb >> 5);
threadgroup vec<C, 4>* ws_dst =
    reinterpret_cast<threadgroup vec<C, 4>*>(Ws + qr * PAD + qb);
threadgroup vec<C, 4>* ws_lo_dst = reinterpret_cast<threadgroup vec<C, 4>*>(
    SYM_F16 ? Ws_lo : (Ws_lo + qr * PAD + qb));

const int x_ld = GROUPED ? (G * K) : K;
const device float* x_ptr =
    x + (size_t(begin + uint(tm)) * size_t(G) + size_t(g)) * size_t(K);
const bool active = (ALIGNED_M && ALIGNED_N) || (valid_m > 0 && valid_n > 0);
const short fn = BaseNAXFrag::get_coord().x;

uint4 qw0 = uint4(0);
uint4 qw1 = uint4(0);
float qs = 0.0f;
float qbias = 0.0f;
if (row_ok) {
  qw0 = w_ptr[0];
  qw1 = w_ptr[1];
  qs = float(*s_ptr);
  if constexpr (!SYM_F16) {
    qbias = float(*b_ptr);
  }
}

for (int base = 0; base < K; base += BK) {
  if (base > 0) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if constexpr (SYM_F16) {
    // Store exact signed int8 codes in half (`u ^ 0x80 == int8(u - 128)`) and
    // apply the per-block f32 scale after each 32-K FP16 NAX MMA so weights
    // incur zero FP16 rounding error with only 2 MMAs per K-block.
    S_tg[qblk * uint(BN) + qr] = row_ok ? qs : 0.0f;
    if (row_ok) {
      STEEL_PRAGMA_UNROLL
      for (int j = 0; j < 4; ++j) {
        ws_dst[j] = vec<C, 4>(as_type<char4>(qw0[j] ^ 0x80808080u));
      }
      STEEL_PRAGMA_UNROLL
      for (int j = 0; j < 4; ++j) {
        ws_dst[4 + j] = vec<C, 4>(as_type<char4>(qw1[j] ^ 0x80808080u));
      }
    } else {
      STEEL_PRAGMA_UNROLL
      for (int j = 0; j < 8; ++j) {
        ws_dst[j] = vec<C, 4>(C(0));
      }
    }
  } else {
    if (row_ok) {
      const float4 vsc = float4(qs);
      const float4 vbi = float4(qbias);
      STEEL_PRAGMA_UNROLL
      for (int j = 0; j < 4; ++j) {
        const float4 wv = metal::fma(float4(as_type<uchar4>(qw0[j])), vsc, vbi);
        const vec<C, 4> wh = vec<C, 4>(wv);
        ws_dst[j] = wh;
        ws_lo_dst[j] = vec<C, 4>(wv - float4(wh));
      }
      STEEL_PRAGMA_UNROLL
      for (int j = 0; j < 4; ++j) {
        const float4 wv = metal::fma(float4(as_type<uchar4>(qw1[j])), vsc, vbi);
        const vec<C, 4> wh = vec<C, 4>(wv);
        ws_dst[4 + j] = wh;
        ws_lo_dst[4 + j] = vec<C, 4>(wv - float4(wh));
      }
    } else {
      STEEL_PRAGMA_UNROLL
      for (int j = 0; j < 8; ++j) {
        ws_dst[j] = vec<C, 4>(C(0));
        ws_lo_dst[j] = vec<C, 4>(C(0));
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  w_ptr += 4;
  s_ptr += 2;
  if constexpr (!SYM_F16) {
    b_ptr += 2;
  }
  if (base + BK < K && row_ok) {
    qw0 = w_ptr[0];
    qw1 = w_ptr[1];
    qs = float(*s_ptr);
    if constexpr (!SYM_F16) {
      qbias = float(*b_ptr);
    }
  }

  if (active) {
    STEEL_PRAGMA_NO_UNROLL
    for (int kk_idx = 0; kk_idx < 2; ++kk_idx) {
      const int kk = kk_idx * 32;
      NAXTile<C, 2, 2> B_hi;
      B_hi.template load<C, PAD, 1>(Ws + size_t(tn) * size_t(PAD) + size_t(kk));
      if constexpr (SYM_F16) {
        const threadgroup float* s_blk = S_tg + size_t(kk_idx * BN) + size_t(tn);
        const float4 sc0 = *reinterpret_cast<const threadgroup float4*>(s_blk + fn);
        const float4 sc1 =
            *reinterpret_cast<const threadgroup float4*>(s_blk + 16 + fn);
        STEEL_PRAGMA_UNROLL
        for (short mm = 0; mm < TM / 16; ++mm) {
          NAXTile<float, 1, 2> A_f32;
          const device float* xm_ptr =
              x_ptr + size_t(mm * 16) * size_t(x_ld) + size_t(kk);
          if constexpr (ALIGNED_M) {
            A_f32.load(xm_ptr, x_ld);
          } else {
            const short mm_valid =
                short(max(0, min(16, int(valid_m) - int(mm) * 16)));
            if (mm_valid <= 0) {
              continue;
            }
            if (mm_valid == 16) {
              A_f32.load(xm_ptr, x_ld);
            } else {
              A_f32.load_safe(xm_ptr, x_ld, short2(32, mm_valid));
            }
          }
          NAXTile<C, 1, 2> A_hi;
          NAXTile<C, 1, 2> A_lo;
          STEEL_PRAGMA_UNROLL
          for (short i = 0; i < NAXTile<float, 1, 2>::kElemsPerTile; ++i) {
            const float v = A_f32.elems()[i];
            const C hi = C(v);
            A_hi.elems()[i] = hi;
            A_lo.elems()[i] = C(v - float(hi));
          }
          NAXTile<float, 1, 2> P;
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
          thread auto& d0 = D.frag_at(mm, 0);
          thread auto& d1 = D.frag_at(mm, 1);
          const thread auto& p0 = P.frag_at(0, 0);
          const thread auto& p1 = P.frag_at(0, 1);
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < 4; ++j) {
            d0[j] = metal::fma(p0[j], sc0[j], d0[j]);
            d0[4 + j] = metal::fma(p0[4 + j], sc0[j], d0[4 + j]);
            d1[j] = metal::fma(p1[j], sc1[j], d1[j]);
            d1[4 + j] = metal::fma(p1[4 + j], sc1[j], d1[4 + j]);
          }
        }
      } else {
        NAXTile<float, TM / 16, 2> A_f32;
        if constexpr (ALIGNED_M) {
          A_f32.load(x_ptr + kk, x_ld);
        } else if (valid_m == TM) {
          A_f32.load(x_ptr + kk, x_ld);
        } else {
          A_f32.load_safe(x_ptr + kk, x_ld, short2(32, valid_m));
        }
        NAXTile<C, TM / 16, 2> A_hi;
        NAXTile<C, TM / 16, 2> A_lo;
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < NAXTile<float, TM / 16, 2>::kElemsPerTile; ++i) {
          const float v = A_f32.elems()[i];
          const C hi = C(v);
          A_hi.elems()[i] = hi;
          A_lo.elems()[i] = C(v - float(hi));
        }
        NAXTile<C, 2, 2> B_lo;
        B_lo.template load<C, PAD, 1>(
            Ws_lo + size_t(tn) * size_t(PAD) + size_t(kk));
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

  x_ptr += BK;
}

if (active) {
  const int out_ld = GROUPED ? (G * N) : N;
  device float* out_base =
      out + (size_t(begin + uint(tm)) * size_t(G) + size_t(g)) * size_t(N) +
      size_t(col + uint(tn));
  if constexpr (ALIGNED_M && ALIGNED_N) {
    D.store(out_base, out_ld);
  } else if (valid_m == TM && valid_n == 32) {
    D.store(out_base, out_ld);
  } else {
    D.store_slice(out_base, out_ld, short2(0, 0), short2(valid_n, valid_m));
  }
}
)METAL"
