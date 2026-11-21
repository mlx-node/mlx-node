R"METAL(
constexpr int BM = M_TILE;
constexpr int TM = BM / 2;
constexpr int BN = 64;
constexpr int BK = 64;
constexpr int PAD = 64 + 16 / sizeof(C);

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
NAXTile<float, TM / 16, 2> D;
D.clear();

const uint qr = tid / 2u;
const uint qb = (tid % 2u) * 32u;
const bool row_ok = ALIGNED_N || (int(col + qr) < N);
const size_t wrow = size_t(g) * size_t(N) + size_t(col + qr);
const device uchar4* w_ptr =
    reinterpret_cast<const device uchar4*>(w) +
    wrow * size_t(K >> 2) + size_t(qb >> 2);
auto s_ptr = scales + wrow * size_t(K >> 5) + size_t(qb >> 5);
auto b_ptr = biases + wrow * size_t(K >> 5) + size_t(qb >> 5);
threadgroup vec<C, 4>* ws_dst =
    reinterpret_cast<threadgroup vec<C, 4>*>(Ws + qr * PAD + qb);

const int x_ld = GROUPED ? (G * K) : K;
const device float* x_ptr =
    x + (size_t(begin + uint(tm)) * size_t(G) + size_t(g)) * size_t(K);
const bool active = (ALIGNED_M && ALIGNED_N) || (valid_m > 0 && valid_n > 0);

for (int base = 0; base < K; base += BK) {
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (row_ok) {
    const float qs = float(*s_ptr);
    float qbias;
    if constexpr (SYM_F16) {
      qbias = -128.0f * qs;
    } else {
      qbias = float(*b_ptr);
    }
    const float4 vsc = float4(qs);
    const float4 vbi = float4(qbias);
    STEEL_PRAGMA_UNROLL
    for (int j = 0; j < 8; ++j) {
      ws_dst[j] = vec<C, 4>(metal::fma(float4(w_ptr[j]), vsc, vbi));
    }
  } else {
    STEEL_PRAGMA_UNROLL
    for (int j = 0; j < 8; ++j) {
      ws_dst[j] = vec<C, 4>(C(0));
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (active) {
    STEEL_PRAGMA_NO_UNROLL
    for (int kk = 0; kk < BK; kk += 32) {
      NAXTile<C, TM / 16, 2> A;
      NAXTile<C, 2, 2> B;
      if constexpr (ALIGNED_M) {
        A.load(x_ptr + kk, x_ld);
      } else if (valid_m == TM) {
        A.load(x_ptr + kk, x_ld);
      } else {
        A.load_safe(x_ptr + kk, x_ld, short2(32, valid_m));
      }
      B.template load<C, PAD, 1>(Ws + size_t(tn) * size_t(PAD) + size_t(kk));
      if constexpr (HI_LO_SPLIT && metal::is_same_v<C, float>) {
        NAXTile<C, TM / 16, 2> A_lo;
        NAXTile<C, 2, 2> B_lo;
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < NAXTile<C, TM / 16, 2>::kElemsPerTile; ++i) {
          const float v = A.elems()[i];
          const float hi = as_type<float>(as_type<uint>(v) & 0xffffe000u);
          A.elems()[i] = hi;
          A_lo.elems()[i] = v - hi;
        }
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < NAXTile<C, 2, 2>::kElemsPerTile; ++i) {
          const float v = B.elems()[i];
          const float hi = as_type<float>(as_type<uint>(v) & 0xffffe000u);
          B.elems()[i] = hi;
          B_lo.elems()[i] = v - hi;
        }
        tile_matmad_nax(
            D,
            A,
            metal::bool_constant<false>{},
            B,
            metal::bool_constant<true>{});
        tile_matmad_nax(
            D,
            A,
            metal::bool_constant<false>{},
            B_lo,
            metal::bool_constant<true>{});
        tile_matmad_nax(
            D,
            A_lo,
            metal::bool_constant<false>{},
            B,
            metal::bool_constant<true>{});
      } else {
        tile_matmad_nax(
            D,
            A,
            metal::bool_constant<false>{},
            B,
            metal::bool_constant<true>{});
      }
    }
  }

  w_ptr += 16;
  s_ptr += 2;
  if constexpr (!SYM_F16) {
    b_ptr += 2;
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
