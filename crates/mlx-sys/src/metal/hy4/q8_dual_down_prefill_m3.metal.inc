R"METAL(
constexpr int BM = M_TILE;
constexpr int BN = N_TILE;
constexpr int SG_M = ((FAST_1MMA && BN == 128) || (!FAST_1MMA && BM == 64)) ? 4 : 2;
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
const ushort2 coord = hy4_m3_sg_coord(lane);
const ushort fn = coord.x;
const ushort fm = coord.y;

// All tile configs stay <= 15.25 KiB TG memory (2 TGs/core).
alignas(16) threadgroup half tg_mem[TG_HALVES];
alignas(16) threadgroup float S_tg[S_TG_ELEMS];

threadgroup half* As_hi = tg_mem;
threadgroup half* As_lo = FAST_1MMA ? tg_mem : (As_hi + A_HALVES);
threadgroup half* Ws = FAST_1MMA ? (As_hi + A_HALVES) : (As_lo + A_HALVES);

const bool active = (valid_m > 0 && valid_n > 0);
const size_t rl_s0 = size_t(routed_latent_strides[0]);
const size_t sgu_s0 = size_t(shared_gu_strides[0]);
const size_t k_f4 = size_t(K >> 2);

if constexpr (FAST_1MMA) {
  constexpr int A_ROWS_PER_PASS = NUM_THREADS / 4;
  constexpr int A_PASSES = BM / A_ROWS_PER_PASS;
  const uint ar0 = tid >> 2;
  const uint ac4 = tid & 3u;
  const uint ac = ac4 * 8u;

  const device float4* rl_src[A_PASSES];
  half4 xh0[A_PASSES];
  half4 xh1[A_PASSES];
  STEEL_PRAGMA_UNROLL
  for (int p = 0; p < A_PASSES; ++p) {
    const uint r = ar0 + uint(p) * uint(A_ROWS_PER_PASS);
    const size_t mr_c = size_t(min(begin + r, uint(M - 1)));
    rl_src[p] = reinterpret_cast<const device float4*>(
        routed_latent + mr_c * rl_s0 + size_t(ac));
    xh0[p] = half4(rl_src[p][0]);
    xh1[p] = half4(rl_src[p][1]);
  }

  // NUM_THREADS == BN * 2: two threads share one 32-element Q8 block.
  const uint qr = tid >> 1;
  const uint qhalf = tid & 1u;
  const bool row_ok = ALIGNED_N || (int(col + qr) < N);
  const size_t wrow = size_t(col + qr);
  const device uint4* w_ptr =
      reinterpret_cast<const device uint4*>(w_up) +
      wrow * size_t(K >> 4) + size_t(qhalf);
  auto s_ptr = s_up + wrow * size_t(K >> 5);

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

  // Pass 1: routed_latent @ w_up^T
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
        rl_src[p] += 8;
        xh0[p] = half4(rl_src[p][0]);
        xh1[p] = half4(rl_src[p][1]);
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

  // Pass 2: swiglu(shared_gu[:, :K], shared_gu[:, K:]) @ w_sd^T added into D.
  const device float4* sgu_src[A_PASSES];
  STEEL_PRAGMA_UNROLL
  for (int p = 0; p < A_PASSES; ++p) {
    const uint r = ar0 + uint(p) * uint(A_ROWS_PER_PASS);
    const size_t mr_c = size_t(min(begin + r, uint(M - 1)));
    sgu_src[p] = reinterpret_cast<const device float4*>(
        shared_gu + mr_c * sgu_s0 + size_t(ac));
    xh0[p] = half4(hy4_dual_swiglu4<CLAMP>(sgu_src[p][0], sgu_src[p][k_f4], lim));
    xh1[p] = half4(hy4_dual_swiglu4<CLAMP>(sgu_src[p][1], sgu_src[p][k_f4 + 1], lim));
  }
  w_ptr = reinterpret_cast<const device uint4*>(w_sd) +
      wrow * size_t(K >> 4) + size_t(qhalf);
  s_ptr = s_sd + wrow * size_t(K >> 5);
  if (row_ok) {
    qw0 = w_ptr[0];
    if (qhalf == 0u) {
      qs_f = float(*s_ptr);
    }
  }
  qs_f = simd_shuffle(qs_f, pair_base);

  for (int base = 0; base < K; base += BK) {
    // WAR barrier protects Pass 1 final iteration (base==0) and prior Pass 2 iteration.
    threadgroup_barrier(mem_flags::mem_threadgroup);

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
        sgu_src[p] += 8;
        xh0[p] = half4(hy4_dual_swiglu4<CLAMP>(sgu_src[p][0], sgu_src[p][k_f4], lim));
        xh1[p] = half4(hy4_dual_swiglu4<CLAMP>(sgu_src[p][1], sgu_src[p][k_f4 + 1], lim));
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
        device float* out_row =
            out + size_t(begin + uint(tm) + uint(r_off)) * size_t(N) +
            size_t(col + uint(tn));
        STEEL_PRAGMA_UNROLL
        for (short ni = 0; ni < TN8; ++ni) {
          const short c_off = ni * 8 + short(fn);
          const float2 v = hy4_m3_te(D[mi][ni]);
          if (c_off < valid_n) {
            out_row[c_off] = v.x;
          }
          if (c_off + 1 < valid_n) {
            out_row[c_off + 1] = v.y;
          }
        }
      }
    }
  }
} else {
  constexpr int A_ROWS_PER_PASS = NUM_THREADS / 8;
  constexpr int A_PASSES = BM / A_ROWS_PER_PASS;
  const uint ar0 = tid >> 3;
  const uint ac4 = tid & 7u;
  const device float4* rl_src[A_PASSES];
  float4 xv[A_PASSES];
  STEEL_PRAGMA_UNROLL
  for (int p = 0; p < A_PASSES; ++p) {
    const uint r = ar0 + uint(p) * uint(A_ROWS_PER_PASS);
    const size_t mr_c = size_t(min(begin + r, uint(M - 1)));
    rl_src[p] = reinterpret_cast<const device float4*>(
        routed_latent + mr_c * rl_s0 + size_t(ac4 * 4u));
    xv[p] = rl_src[p][0];
  }

  constexpr int W_TPR = NUM_THREADS / BN;
  const uint qr = tid / uint(W_TPR);
  const uint qw_lane = tid % uint(W_TPR);
  const bool row_ok = ALIGNED_N || (int(col + qr) < N);
  const size_t wrow = size_t(col + qr);
  const device uint4* w_ptr4 =
      reinterpret_cast<const device uint4*>(w_up) +
      wrow * size_t(K >> 4) + size_t(qw_lane);
  const device uint2* w_ptr2 =
      reinterpret_cast<const device uint2*>(w_up) +
      wrow * size_t(K >> 3) + size_t(qw_lane);
  auto s_ptr = s_up + wrow * size_t(K >> 5);

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

  // Pass 1: routed_latent @ w_up^T
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
        rl_src[p] += 8;
        xv[p] = rl_src[p][0];
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

  // Pass 2: swiglu(shared_gu) @ w_sd^T added into D_acc.
  const device float4* sgu_src[A_PASSES];
  STEEL_PRAGMA_UNROLL
  for (int p = 0; p < A_PASSES; ++p) {
    const uint r = ar0 + uint(p) * uint(A_ROWS_PER_PASS);
    const size_t mr_c = size_t(min(begin + r, uint(M - 1)));
    sgu_src[p] = reinterpret_cast<const device float4*>(
        shared_gu + mr_c * sgu_s0 + size_t(ac4 * 4u));
    xv[p] = hy4_dual_swiglu4<CLAMP>(sgu_src[p][0], sgu_src[p][k_f4], lim);
  }
  w_ptr4 = reinterpret_cast<const device uint4*>(w_sd) +
      wrow * size_t(K >> 4) + size_t(qw_lane);
  w_ptr2 = reinterpret_cast<const device uint2*>(w_sd) +
      wrow * size_t(K >> 3) + size_t(qw_lane);
  s_ptr = s_sd + wrow * size_t(K >> 5);
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

  for (int base = 0; base < K; base += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);

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
        sgu_src[p] += 8;
        xv[p] = hy4_dual_swiglu4<CLAMP>(sgu_src[p][0], sgu_src[p][k_f4], lim);
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
        device float* out_row =
            out + size_t(begin + uint(tm) + uint(r_off)) * size_t(N) +
            size_t(col + uint(tn));
        STEEL_PRAGMA_UNROLL
        for (short ni = 0; ni < TN8; ++ni) {
          const short c_off = ni * 8 + short(fn);
          const float2 v = D_acc[mi][ni];
          if (c_off < valid_n) {
            out_row[c_off] = v.x;
          }
          if (c_off + 1 < valid_n) {
            out_row[c_off + 1] = v.y;
          }
        }
      }
    }
  }
}
)METAL"