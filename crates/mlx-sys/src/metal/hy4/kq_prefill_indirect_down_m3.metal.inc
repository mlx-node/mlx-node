R"METAL(
constexpr int BM = 32;
constexpr int BN = 64;
constexpr int BK = 32;
constexpr int PAD = 40;

const int R = hidden_shape[0];
const int M = hidden_shape[1];
const int S = wd_shape[0];
const int H = wd_shape[1];

const uint tile = threadgroup_position_in_grid.y;
const uint begin = tiles[3u * tile];
const uint end = tiles[3u * tile + 1u];
const uint slot = tiles[3u * tile + 2u];
if (begin >= end || end > uint(R)) return;

const uint col = threadgroup_position_in_grid.x * uint(BN);
if (col >= uint(H)) return;

const uint sg_idx = simdgroup_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;
const uint tid = sg_idx * 32u + lane;
const short tm = short((sg_idx / 2u) * 16u);
const short tn = short((sg_idx % 2u) * 32u);
const short valid_m = short(max(0, min(16, int(end - begin) - int(tm))));
const short valid_n = short(max(0, min(32, int(H) - int(col + uint(tn)))));
const ushort2 coord = hy4_m3_sg_coord(lane);
const ushort fn = coord.x;
const ushort fm = coord.y;

// As (2,560 B) + Bs (5,120 B) = 7,680 B (7.5 KiB).
alignas(16) threadgroup half As[BM * PAD];
alignas(16) threadgroup half Bs[BN * PAD];

simdgroup_float8x8 cd[2][4];
STEEL_PRAGMA_UNROLL
for (short mi = 0; mi < 2; ++mi) {
  STEEL_PRAGMA_UNROLL
  for (short ni = 0; ni < 4; ++ni) {
    cd[mi][ni] = make_filled_simdgroup_matrix<float, 8>(0.0f);
  }
}

if (slot < uint(S)) {
  const uint ar = tid >> 2;
  const uint ac = (tid & 3u) * 8u;
  const bool a_live = (begin + ar) < end;

  half4 av[2];
  if (a_live) {
    if constexpr (IN_F16) {
      const device float4* src = reinterpret_cast<const device float4*>(
          hidden + size_t(begin + ar) * size_t(M) + size_t(ac));
      const float4 r0 = src[0];
      av[0] = as_type<half4>(r0.xy);
      av[1] = as_type<half4>(r0.zw);
    } else {
      const device float4* src = reinterpret_cast<const device float4*>(
          hidden + size_t(begin + ar) * size_t(M) + size_t(ac));
      av[0] = half4(src[0]);
      av[1] = half4(src[1]);
    }
  } else {
    threadgroup half4* dst =
        reinterpret_cast<threadgroup half4*>(As + ar * PAD + ac);
    dst[0] = half4(0.0h);
    dst[1] = half4(0.0h);
  }

  // 128 threads over 64 output rows = 2 threads per row (16 weights/thread).
  const uint qr = tid >> 1;
  const uint qhalf = tid & 1u;
  const bool row_ok = (col + qr) < uint(H);
  const size_t wr = size_t(slot) * size_t(H) + size_t(col + qr);
  threadgroup half4* w_dst =
      reinterpret_cast<threadgroup half4*>(Bs + qr * PAD + qhalf * 16u);
  if (!row_ok) {
    STEEL_PRAGMA_UNROLL
    for (int j = 0; j < 4; ++j) {
      w_dst[j] = half4(0.0h);
    }
  }

  const bool active = (valid_m > 0 && valid_n > 0);

  if constexpr (BITS == 4) {
    const device uint2* wp =
        reinterpret_cast<const device uint2*>(wd) +
        wr * size_t(M >> 4) + size_t(qhalf);
    const size_t g_base = wr * size_t(M >> 5);
    const ushort pair_base = ushort(lane & ~1u);

    uint2 qw = uint2(0u);
    float sc = 0.0f, bi = 0.0f;
    if (row_ok) {
      qw = wp[0];
      if (qhalf == 0u) {
        hy4_kq_decode_q4_group(sd, bd, g_base, sc, bi);
      }
    }
    sc = simd_shuffle(sc, pair_base);
    bi = simd_shuffle(bi, pair_base);

    for (int base = 0; base < M; base += BK) {
      if (base > 0) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
      if (a_live) {
        threadgroup half4* dst =
            reinterpret_cast<threadgroup half4*>(As + ar * PAD + ac);
        dst[0] = av[0];
        dst[1] = av[1];
      }
      if (row_ok) {
        hy4_kq_unpack_q4_half16(qw, sc, bi, w_dst);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      wp += 2;
      const int next = base + BK;
      if (next < M) {
        if (a_live) {
          if constexpr (IN_F16) {
            const device float4* src = reinterpret_cast<const device float4*>(
                hidden + size_t(begin + ar) * size_t(M) + size_t(next) + size_t(ac));
            const float4 r0 = src[0];
            av[0] = as_type<half4>(r0.xy);
            av[1] = as_type<half4>(r0.zw);
          } else {
            const device float4* src = reinterpret_cast<const device float4*>(
                hidden + size_t(begin + ar) * size_t(M) + size_t(next) + size_t(ac));
            av[0] = half4(src[0]);
            av[1] = half4(src[1]);
          }
        }
        if (row_ok) {
          qw = wp[0];
          if (qhalf == 0u) {
            hy4_kq_decode_q4_group(sd, bd, g_base + size_t(next >> 5), sc, bi);
          }
        }
        sc = simd_shuffle(sc, pair_base);
        bi = simd_shuffle(bi, pair_base);
      }

      if (active) {
        STEEL_PRAGMA_UNROLL
        for (short kk = 0; kk < BK; kk += 8) {
          simdgroup_half8x8 ma0, ma1;
          simdgroup_load(ma0, As + (tm + 0) * PAD + kk, PAD);
          simdgroup_load(ma1, As + (tm + 8) * PAD + kk, PAD);
          STEEL_PRAGMA_UNROLL
          for (short ni = 0; ni < 4; ++ni) {
            simdgroup_half8x8 mb;
            simdgroup_load(
                mb, Bs + (tn + ni * 8) * PAD + kk, PAD, ulong2(0), true);
            simdgroup_multiply_accumulate(cd[0][ni], ma0, mb, cd[0][ni]);
            simdgroup_multiply_accumulate(cd[1][ni], ma1, mb, cd[1][ni]);
          }
        }
      }
    }
  } else {
    // Q6_K: group_size = 16, 6 u32 words per 32 weights = 3 u32 words per thread.
    const device uint32_t* wp =
        wd + wr * (size_t(M >> 5) * 6u) + size_t(qhalf) * 3u;
    const size_t g_base = wr * size_t(M >> 4) + size_t(qhalf);

    uint32_t w0 = 0u, w1 = 0u, w2 = 0u;
    float sc = 0.0f;
    if (row_ok) {
      w0 = wp[0];
      w1 = wp[1];
      w2 = wp[2];
      sc = hy4_kq_decode_q6_scale(sd, bd, g_base);
    }

    for (int base = 0; base < M; base += BK) {
      if (base > 0) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
      if (a_live) {
        threadgroup half4* dst =
            reinterpret_cast<threadgroup half4*>(As + ar * PAD + ac);
        dst[0] = av[0];
        dst[1] = av[1];
      }
      if (row_ok) {
        hy4_kq_unpack_q6_group16(w0, w1, w2, sc, w_dst);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      wp += 6;
      const int next = base + BK;
      if (next < M) {
        if (a_live) {
          if constexpr (IN_F16) {
            const device float4* src = reinterpret_cast<const device float4*>(
                hidden + size_t(begin + ar) * size_t(M) + size_t(next) + size_t(ac));
            const float4 r0 = src[0];
            av[0] = as_type<half4>(r0.xy);
            av[1] = as_type<half4>(r0.zw);
          } else {
            const device float4* src = reinterpret_cast<const device float4*>(
                hidden + size_t(begin + ar) * size_t(M) + size_t(next) + size_t(ac));
            av[0] = half4(src[0]);
            av[1] = half4(src[1]);
          }
        }
        if (row_ok) {
          w0 = wp[0];
          w1 = wp[1];
          w2 = wp[2];
          sc = hy4_kq_decode_q6_scale(sd, bd, g_base + size_t(next >> 4));
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
            simdgroup_half8x8 mb;
            simdgroup_load(
                mb, Bs + (tn + ni * 8) * PAD + kk, PAD, ulong2(0), true);
            simdgroup_multiply_accumulate(cd[0][ni], ma0, mb, cd[0][ni]);
            simdgroup_multiply_accumulate(cd[1][ni], ma1, mb, cd[1][ni]);
          }
        }
      }
    }
  }
}

if (valid_m > 0 && valid_n > 0) {
  STEEL_PRAGMA_UNROLL
  for (short mi = 0; mi < 2; ++mi) {
    const short r_off = mi * 8 + short(fm);
    if (r_off < valid_m) {
      device float* row_base =
          out + size_t(begin + uint(tm) + uint(r_off)) * size_t(H) +
          size_t(col + uint(tn));
      STEEL_PRAGMA_UNROLL
      for (short ni = 0; ni < 4; ++ni) {
        const short c_off = ni * 8 + short(fn);
        const float2 v = hy4_m3_te(cd[mi][ni]);
        if (c_off < valid_n) {
          row_base[c_off] = v.x;
        }
        if (c_off + 1 < valid_n) {
          row_base[c_off + 1] = v.y;
        }
      }
    }
  }
}
)METAL"
