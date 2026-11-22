R"METAL(
constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 32;
constexpr int PAD = 40;

const int R = hidden_shape[0];
const int M = hidden_shape[1];
const int S = dims[0];
const int H = dims[1];

const uint tile = threadgroup_position_in_grid.y;
const uint begin = tiles[3u * tile];
const uint end = tiles[3u * tile + 1u];
const uint slot = tiles[3u * tile + 2u];
if (begin >= end || end > uint(R)) return;

const uint col = threadgroup_position_in_grid.x * uint(BN);
if (col >= uint(H)) return;

// WM=4, WN=2 (256 threads = 8 SGs): halves accumulator & weight state regs,
// skips idle 16-row SGs on partial tiles (<=48 rows), and stays at 10.0 KiB TG mem.
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

// As (5,120 B) + Bs (5,120 B) = 10,240 B (10.0 KiB, 2 TGs/core = 512T/core).
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
  // 256 threads = 64 rows x 4 threads/row (8 elements/thread).
  const uint ar = tid >> 2;
  const uint ac = (tid & 3u) * 8u;
  const bool a_live = (begin + ar) < end;

  const device float4* a_src = nullptr;
  half4 av0, av1;
  if (a_live) {
    a_src = reinterpret_cast<const device float4*>(
        hidden + size_t(begin + ar) * size_t(M) + size_t(ac));
    if constexpr (IN_F16) {
      const float4 r0 = a_src[0];
      av0 = as_type<half4>(r0.xy);
      av1 = as_type<half4>(r0.zw);
    } else {
      av0 = half4(a_src[0]);
      av1 = half4(a_src[1]);
    }
  } else {
    threadgroup half4* d =
        reinterpret_cast<threadgroup half4*>(As + ar * PAD + ac);
    d[0] = half4(0.0h);
    d[1] = half4(0.0h);
  }

  // 256 threads = 64 weight rows x 4 threads/row (8 consecutive bytes each = 32 B).
  const uint wr = tid >> 2;
  const uint lane4 = tid & 3u;
  const ushort quad_base = ushort(lane & ~3u);
  const bool w_live = (col + wr) < uint(H);
  constexpr size_t BLK_BYTES = (BITS == 6) ? 210u : 144u;
  const size_t row_bytes = size_t(M / 256) * BLK_BYTES;
  const device uint8_t* w_row = down_buf + size_t(down_off[0]) +
      (size_t(slot) * size_t(H) + size_t(col + wr)) * row_bytes;
  if (!w_live) {
    threadgroup half4* d =
        reinterpret_cast<threadgroup half4*>(Bs + wr * PAD + lane4 * 8u);
    d[0] = half4(0.0h);
    d[1] = half4(0.0h);
  }

  const bool active = (valid_m > 0 && valid_n > 0);

  if constexpr (BITS == 4) {
    const uint32_t hdr = w_live
        ? reinterpret_cast<const device uint32_t*>(w_row)[lane4]
        : 0u;
    const half2 dm = as_type<half2>(simd_shuffle(hdr, quad_base));
    float blk_d = float(dm.x);
    float blk_dmin = float(dm.y);
    uint32_t sc_w[3] = {
        simd_shuffle(hdr, quad_base | 1u),
        simd_shuffle(hdr, quad_base | 2u),
        simd_shuffle(hdr, quad_base | 3u),
    };
    uint32_t qs_w[2] = {0u, 0u};
    if (w_live) {
      const device uint32_t* qsp = reinterpret_cast<const device uint32_t*>(
          w_row + 16u + size_t(lane4) * 8u);
      qs_w[0] = qsp[0];
      qs_w[1] = qsp[1];
    }

    for (int base = 0; base < M; base += 64) {
      const int c = (base >> 6) & 3;
      STEEL_PRAGMA_UNROLL
      for (int sub = 0; sub < 2; ++sub) {
        const int g_idx = 2 * c + sub;
        const int k_off = base + sub * BK;

        if (base > 0 || sub > 0) {
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (a_live) {
          threadgroup half4* d =
              reinterpret_cast<threadgroup half4*>(As + ar * PAD + ac);
          d[0] = av0;
          d[1] = av1;
        }
        float dg = 0.0f, mg = 0.0f;
        if (w_live && lane4 == 0u) {
          hy4_gguf_q4k_group_scale(sc_w, blk_d, blk_dmin, g_idx, dg, mg);
        }
        const float d_f = simd_shuffle(dg, quad_base);
        const float m_f = simd_shuffle(mg, quad_base);
        if (w_live) {
          threadgroup half4* dst =
              reinterpret_cast<threadgroup half4*>(Bs + wr * PAD + lane4 * 8u);
          STEEL_PRAGMA_UNROLL
          for (int i = 0; i < 2; ++i) {
            const uchar4 b = as_type<uchar4>(qs_w[i]);
            const uchar4 nib = sub ? (b >> 4) : (b & uchar4(0x0Fu));
            dst[i] = half4(metal::fma(float4(nib), d_f, m_f));
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const int next_k = k_off + BK;
        if (next_k < M) {
          if (a_live) {
            if constexpr (IN_F16) {
              a_src += 4;
              const float4 r0 = a_src[0];
              av0 = as_type<half4>(r0.xy);
              av1 = as_type<half4>(r0.zw);
            } else {
              a_src += 8;
              av0 = half4(a_src[0]);
              av1 = half4(a_src[1]);
            }
          }
          if (sub == 1) {
            const size_t nblk_off = size_t(next_k >> 8) * 144u;
            const size_t nq_off =
                16u + size_t((next_k >> 6) & 3) * 32u + size_t(lane4) * 8u;
            const device uint8_t* nblk = w_row + nblk_off;
            if ((next_k & 255) == 0) {
              const uint32_t nhdr = w_live
                  ? reinterpret_cast<const device uint32_t*>(nblk)[lane4]
                  : 0u;
              const half2 ndm = as_type<half2>(simd_shuffle(nhdr, quad_base));
              blk_d = float(ndm.x);
              blk_dmin = float(ndm.y);
              sc_w[0] = simd_shuffle(nhdr, quad_base | 1u);
              sc_w[1] = simd_shuffle(nhdr, quad_base | 2u);
              sc_w[2] = simd_shuffle(nhdr, quad_base | 3u);
            }
            if (w_live) {
              const device uint32_t* qsp =
                  reinterpret_cast<const device uint32_t*>(nblk + nq_off);
              qs_w[0] = qsp[0];
              qs_w[1] = qsp[1];
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
  } else {
    // block_q6_K is 210 bytes (2-byte aligned ONLY); use packed_ushort4/half only.
    uchar4 ql0_v[2];
    uchar4 ql1_v[2];
    uchar4 qh_v[2];
    float4 sc = float4(0.0f);
    const size_t l_off = size_t(lane4) * 8u;

    {
      float d_val = 0.0f;
      uint2 sc_u2 = uint2(0u);
      if (w_live) {
        const ushort4 v0 = ushort4(*reinterpret_cast<const device packed_ushort4*>(
            w_row + l_off));
        const ushort4 v1 = ushort4(*reinterpret_cast<const device packed_ushort4*>(
            w_row + 32u + l_off));
        const ushort4 vh = ushort4(*reinterpret_cast<const device packed_ushort4*>(
            w_row + 128u + l_off));
        ql0_v[0] = as_type<uchar4>(v0.xy);
        ql0_v[1] = as_type<uchar4>(v0.zw);
        ql1_v[0] = as_type<uchar4>(v1.xy);
        ql1_v[1] = as_type<uchar4>(v1.zw);
        qh_v[0] = as_type<uchar4>(vh.xy);
        qh_v[1] = as_type<uchar4>(vh.zw);
        if (lane4 == 0u) {
          d_val = float(*reinterpret_cast<const device half*>(w_row + 208u));
          const ushort4 sc_raw = ushort4(
              *reinterpret_cast<const device packed_ushort4*>(w_row + 192u));
          sc_u2 = uint2(as_type<uint>(sc_raw.xy), as_type<uint>(sc_raw.zw));
        }
      }
      d_val = simd_shuffle(d_val, quad_base);
      sc_u2 = simd_shuffle(sc_u2, quad_base);
      if (w_live) {
        const char4 c0 = as_type<char4>(sc_u2.x);
        const char4 c1 = as_type<char4>(sc_u2.y);
        const float4 sv = ((lane4 & 2u) == 0u)
            ? float4(float(c0.x), float(c0.z), float(c1.x), float(c1.z))
            : float4(float(c0.y), float(c0.w), float(c1.y), float(c1.w));
        sc = sv * d_val;
      }
    }

    for (int base128 = 0; base128 < M; base128 += 128) {
      STEEL_PRAGMA_UNROLL
      for (int q_step = 0; q_step < 4; ++q_step) {
        const int k_off = base128 + q_step * BK;
        if (base128 > 0 || q_step > 0) {
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (a_live) {
          threadgroup half4* d =
              reinterpret_cast<threadgroup half4*>(As + ar * PAD + ac);
          d[0] = av0;
          d[1] = av1;
        }
        const uchar ql_sh = (q_step >= 2) ? 4u : 0u;
        const uchar qh_sh = uchar(q_step * 2);
        if (w_live) {
          const float sc_cur = sc[q_step];
          threadgroup half4* dst =
              reinterpret_cast<threadgroup half4*>(Bs + wr * PAD + lane4 * 8u);
          STEEL_PRAGMA_UNROLL
          for (int i = 0; i < 2; ++i) {
            const uchar4 ql = (q_step & 1) ? ql1_v[i] : ql0_v[i];
            const uchar4 q =
                ((ql >> ql_sh) & uchar4(0x0Fu)) |
                (((qh_v[i] >> qh_sh) & uchar4(0x03u)) << 4);
            dst[i] = half4(float4(char4(q) - char4(32)) * sc_cur);
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const int next_k = k_off + BK;
        if (next_k < M) {
          if (a_live) {
            if constexpr (IN_F16) {
              a_src += 4;
              const float4 r0 = a_src[0];
              av0 = as_type<half4>(r0.xy);
              av1 = as_type<half4>(r0.zw);
            } else {
              a_src += 8;
              av0 = half4(a_src[0]);
              av1 = half4(a_src[1]);
            }
          }
          if (q_step == 3) {
            const size_t nblk_off = size_t(next_k >> 8) * 210u;
            const size_t nip = size_t((next_k >> 7) & 1);
            float d_val = 0.0f;
            uint2 sc_u2 = uint2(0u);
            if (w_live) {
              const device uint8_t* nblk = w_row + nblk_off;
              const ushort4 v0 = ushort4(*reinterpret_cast<const device packed_ushort4*>(
                  nblk + nip * 64u + l_off));
              const ushort4 v1 = ushort4(*reinterpret_cast<const device packed_ushort4*>(
                  nblk + nip * 64u + 32u + l_off));
              const ushort4 vh = ushort4(*reinterpret_cast<const device packed_ushort4*>(
                  nblk + 128u + nip * 32u + l_off));
              ql0_v[0] = as_type<uchar4>(v0.xy);
              ql0_v[1] = as_type<uchar4>(v0.zw);
              ql1_v[0] = as_type<uchar4>(v1.xy);
              ql1_v[1] = as_type<uchar4>(v1.zw);
              qh_v[0] = as_type<uchar4>(vh.xy);
              qh_v[1] = as_type<uchar4>(vh.zw);
              if (lane4 == 0u) {
                d_val = float(*reinterpret_cast<const device half*>(nblk + 208u));
                const ushort4 sc_raw = ushort4(
                    *reinterpret_cast<const device packed_ushort4*>(
                        nblk + 192u + nip * 8u));
                sc_u2 = uint2(as_type<uint>(sc_raw.xy), as_type<uint>(sc_raw.zw));
              }
            }
            d_val = simd_shuffle(d_val, quad_base);
            sc_u2 = simd_shuffle(sc_u2, quad_base);
            if (w_live) {
              const char4 c0 = as_type<char4>(sc_u2.x);
              const char4 c1 = as_type<char4>(sc_u2.y);
              const float4 sv = ((lane4 & 2u) == 0u)
                  ? float4(float(c0.x), float(c0.z), float(c1.x), float(c1.z))
                  : float4(float(c0.y), float(c0.w), float(c1.y), float(c1.w));
              sc = sv * d_val;
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
}

if (valid_m > 0 && valid_n > 0) {
  STEEL_PRAGMA_UNROLL
  for (short mi = 0; mi < 2; ++mi) {
    const short r_off = mi * 8 + short(fm);
    if (r_off < valid_m) {
      device OUT_T* row_base =
          out + size_t(begin + uint(tm) + uint(r_off)) * size_t(H) +
          size_t(col + uint(tn));
      STEEL_PRAGMA_UNROLL
      for (short ni = 0; ni < 4; ++ni) {
        const short c_off = ni * 8 + short(fn);
        const float2 v = hy4_m3_te(cd[mi][ni]);
        if (c_off < valid_n) {
          row_base[c_off] = OUT_T(v.x);
        }
        if (c_off + 1 < valid_n) {
          row_base[c_off + 1] = OUT_T(v.y);
        }
      }
    }
  }
}
)METAL"