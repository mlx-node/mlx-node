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
constexpr int TG_HALVES = FAST_1MMA ? (A_HALVES + W_HALVES) : (2 * A_HALVES + W_HALVES);
constexpr int S_TG_ELEMS = FAST_1MMA ? 1 : BN;

const int M = routed_latent_shape[0];
const int K = routed_latent_shape[1];
const int N = w_up_shape[0];
const float lim = limit[0];

const uint tile = threadgroup_position_in_grid.y;
const uint begin = tile * uint(BM);
const uint end = min(begin + uint(BM), uint(M));
if (begin >= end) return;
const uint col = threadgroup_position_in_grid.x * uint(BN);

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
alignas(16) threadgroup half tg_mem[TG_HALVES];
alignas(16) threadgroup float S_tg[S_TG_ELEMS];

threadgroup half* As_hi = tg_mem;
threadgroup half* As_lo = FAST_1MMA ? tg_mem : (As_hi + A_HALVES);
threadgroup half* Ws = FAST_1MMA ? (As_hi + A_HALVES) : (As_lo + A_HALVES);

NAXTile<float, TM_F, TN_F> D;
D.clear();

const bool active = (valid_m > 0 && valid_n > 0);
const size_t rl_s0 = size_t(routed_latent_strides[0]);
const size_t sgu_s0 = size_t(shared_gu_strides[0]);
const size_t k_f4 = size_t(K >> 2);

if constexpr (FAST_1MMA) {
  constexpr int A_ROWS_PER_PASS = NUM_THREADS / 8;
  constexpr int A_PASSES = BM / A_ROWS_PER_PASS;
  const uint ar0 = tid >> 3;
  const uint ac8 = tid & 7u;

  const device float4* rl_src[A_PASSES];
  const device float4* sgu_src[A_PASSES];
  half4 xh0[A_PASSES];
  half4 xh1[A_PASSES];
  STEEL_PRAGMA_UNROLL
  for (int p = 0; p < A_PASSES; ++p) {
    const uint r = ar0 + uint(p) * uint(A_ROWS_PER_PASS);
    const size_t mr_c = size_t(min(begin + r, uint(M - 1)));
    rl_src[p] = reinterpret_cast<const device float4*>(
        routed_latent + mr_c * rl_s0 + size_t(ac8 * 8u));
    sgu_src[p] = reinterpret_cast<const device float4*>(
        shared_gu + mr_c * sgu_s0 + size_t(ac8 * 8u));
    xh0[p] = half4(rl_src[p][0]);
    xh1[p] = half4(rl_src[p][1]);
  }

  const uint qr = tid >> 1;
  const uint qblk = tid & 1u;
  const bool row_ok = ALIGNED_N || (int(col + qr) < N);
  const size_t wrow = size_t(col + qr);
  const device uint4* w_ptr =
      reinterpret_cast<const device uint4*>(w_up) +
      wrow * size_t(K >> 4) + size_t(qblk * 2u);
  auto s_ptr = s_up + wrow * size_t(K >> 5) + size_t(qblk);

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

  // Pass 1: routed_latent @ w_up^T
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
        rl_src[p] += 16;
        xh0[p] = half4(rl_src[p][0]);
        xh1[p] = half4(rl_src[p][1]);
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

  // Pass 2: swiglu(shared_gu[:, :K], shared_gu[:, K:]) @ w_sd^T added into D.
  STEEL_PRAGMA_UNROLL
  for (int p = 0; p < A_PASSES; ++p) {
    xh0[p] = half4(hy4_dual_swiglu4<CLAMP>(sgu_src[p][0], sgu_src[p][k_f4], lim));
    xh1[p] = half4(hy4_dual_swiglu4<CLAMP>(sgu_src[p][1], sgu_src[p][k_f4 + 1], lim));
  }
  w_ptr = reinterpret_cast<const device uint4*>(w_sd) +
      wrow * size_t(K >> 4) + size_t(qblk * 2u);
  s_ptr = s_sd + wrow * size_t(K >> 5) + size_t(qblk);
  if (row_ok) {
    qw0 = w_ptr[0];
    qw1 = w_ptr[1];
    qs_h = half(*s_ptr);
  }

  for (int base = 0; base < K; base += 64) {
    // WAR barrier: protects Pass 1 final iteration (base==0) and prior Pass 2 iteration.
    threadgroup_barrier(mem_flags::mem_threadgroup);

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
        sgu_src[p] += 16;
        xh0[p] = half4(hy4_dual_swiglu4<CLAMP>(sgu_src[p][0], sgu_src[p][k_f4], lim));
        xh1[p] = half4(hy4_dual_swiglu4<CLAMP>(sgu_src[p][1], sgu_src[p][k_f4 + 1], lim));
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
  const device float4* rl_src[A_PASSES];
  const device float4* sgu_src[A_PASSES];
  float4 xv[A_PASSES];
  STEEL_PRAGMA_UNROLL
  for (int p = 0; p < A_PASSES; ++p) {
    const uint r = ar0 + uint(p) * 16u;
    const size_t mr_c = size_t(min(begin + r, uint(M - 1)));
    rl_src[p] = reinterpret_cast<const device float4*>(
        routed_latent + mr_c * rl_s0 + size_t(ac4 * 4u));
    sgu_src[p] = reinterpret_cast<const device float4*>(
        shared_gu + mr_c * sgu_s0 + size_t(ac4 * 4u));
    xv[p] = rl_src[p][0];
  }

  const uint qr = tid >> 1;
  const uint qhalf = tid & 1u;
  const bool row_ok = ALIGNED_N || (int(col + qr) < N);
  const size_t wrow = size_t(col + qr);
  const device uint4* w_ptr =
      reinterpret_cast<const device uint4*>(w_up) +
      wrow * size_t(K >> 4) + size_t(qhalf);
  auto s_ptr = s_up + wrow * size_t(K >> 5);

  uint4 qw0 = uint4(0);
  float qs = 0.0f;
  if (row_ok) {
    qw0 = w_ptr[0];
    if (qhalf == 0u) {
      qs = float(*s_ptr);
    }
  }
  qs = simd_shuffle(qs, ushort(lane & ~1u));
  const short fn = BaseNAXFrag::get_coord().x;

  for (int base = 0; base < K; base += 32) {
    if (base > 0) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    STEEL_PRAGMA_UNROLL
    for (int p = 0; p < A_PASSES; ++p) {
      const uint r = ar0 + uint(p) * 16u;
      const float4 v = xv[p];
      const half4 hi = half4(v);
      *reinterpret_cast<threadgroup half4*>(As_hi + r * PAD + ac4 * 4u) = hi;
      *reinterpret_cast<threadgroup half4*>(As_lo + r * PAD + ac4 * 4u) =
          half4(v - float4(hi));
    }

    if (qhalf == 0u) {
      S_tg[qr] = row_ok ? qs : 0.0f;
    }
    threadgroup half4* dst =
        reinterpret_cast<threadgroup half4*>(Ws + qr * PAD + qhalf * 16u);
    if (row_ok) {
      STEEL_PRAGMA_UNROLL
      for (int j = 0; j < 4; ++j) {
        dst[j] = half4(as_type<char4>(qw0[j] ^ 0x80808080u));
      }
    } else {
      STEEL_PRAGMA_UNROLL
      for (int j = 0; j < 4; ++j) {
        dst[j] = half4(0.0h);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    w_ptr += 2;
    s_ptr += 1;
    if (base + 32 < K) {
      STEEL_PRAGMA_UNROLL
      for (int p = 0; p < A_PASSES; ++p) {
        rl_src[p] += 8;
        xv[p] = rl_src[p][0];
      }
      if (row_ok) {
        qw0 = w_ptr[0];
        if (qhalf == 0u) {
          qs = float(*s_ptr);
        }
      }
      qs = simd_shuffle(qs, ushort(lane & ~1u));
    }

    if (active) {
      NAXTile<half, TM_F, 2> A_hi;
      NAXTile<half, TM_F, 2> A_lo;
      NAXTile<half, 2, 2> B_hi;
      A_hi.template load<half, PAD, 1>(As_hi + size_t(tm) * size_t(PAD));
      A_lo.template load<half, PAD, 1>(As_lo + size_t(tm) * size_t(PAD));
      B_hi.template load<half, PAD, 1>(Ws + size_t(tn) * size_t(PAD));
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
    }
  }

  STEEL_PRAGMA_UNROLL
  for (int p = 0; p < A_PASSES; ++p) {
    const float4 gv = sgu_src[p][0];
    const float4 uv = sgu_src[p][k_f4];
    xv[p] = hy4_dual_swiglu4<CLAMP>(gv, uv, lim);
  }
  w_ptr = reinterpret_cast<const device uint4*>(w_sd) +
      wrow * size_t(K >> 4) + size_t(qhalf);
  s_ptr = s_sd + wrow * size_t(K >> 5);
  if (row_ok) {
    qw0 = w_ptr[0];
    if (qhalf == 0u) {
      qs = float(*s_ptr);
    }
  }
  qs = simd_shuffle(qs, ushort(lane & ~1u));

  for (int base = 0; base < K; base += 32) {
    // WAR barrier: protects Pass 1 final iteration (base==0) and prior Pass 2 iteration.
    threadgroup_barrier(mem_flags::mem_threadgroup);

    STEEL_PRAGMA_UNROLL
    for (int p = 0; p < A_PASSES; ++p) {
      const uint r = ar0 + uint(p) * 16u;
      const float4 v = xv[p];
      const half4 hi = half4(v);
      *reinterpret_cast<threadgroup half4*>(As_hi + r * PAD + ac4 * 4u) = hi;
      *reinterpret_cast<threadgroup half4*>(As_lo + r * PAD + ac4 * 4u) =
          half4(v - float4(hi));
    }

    if (qhalf == 0u) {
      S_tg[qr] = row_ok ? qs : 0.0f;
    }
    threadgroup half4* dst =
        reinterpret_cast<threadgroup half4*>(Ws + qr * PAD + qhalf * 16u);
    if (row_ok) {
      STEEL_PRAGMA_UNROLL
      for (int j = 0; j < 4; ++j) {
        dst[j] = half4(as_type<char4>(qw0[j] ^ 0x80808080u));
      }
    } else {
      STEEL_PRAGMA_UNROLL
      for (int j = 0; j < 4; ++j) {
        dst[j] = half4(0.0h);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    w_ptr += 2;
    s_ptr += 1;
    if (base + 32 < K) {
      STEEL_PRAGMA_UNROLL
      for (int p = 0; p < A_PASSES; ++p) {
        sgu_src[p] += 8;
        const float4 gv = sgu_src[p][0];
        const float4 uv = sgu_src[p][k_f4];
        xv[p] = hy4_dual_swiglu4<CLAMP>(gv, uv, lim);
      }
      if (row_ok) {
        qw0 = w_ptr[0];
        if (qhalf == 0u) {
          qs = float(*s_ptr);
        }
      }
      qs = simd_shuffle(qs, ushort(lane & ~1u));
    }

    if (active) {
      NAXTile<half, TM_F, 2> A_hi;
      NAXTile<half, TM_F, 2> A_lo;
      NAXTile<half, 2, 2> B_hi;
      A_hi.template load<half, PAD, 1>(As_hi + size_t(tm) * size_t(PAD));
      A_lo.template load<half, PAD, 1>(As_lo + size_t(tm) * size_t(PAD));
      B_hi.template load<half, PAD, 1>(Ws + size_t(tn) * size_t(PAD));
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
    }
  }
}

if (active) {
  device float* out_base =
      out + size_t(begin + uint(tm)) * size_t(N) + size_t(col + uint(tn));
  if (valid_m == TM && valid_n == TN) {
    D.store(out_base, N);
  } else {
    D.store_slice(out_base, N, short2(0, 0), short2(valid_n, valid_m));
  }
}
)METAL"